import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import akshare as ak
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===================== 基础数据获取函数（小幅优化） =====================
def fetch_fund_info(fund_code: str) -> dict:
    try:
        df = ak.fund_individual_basic_info_xq(symbol=fund_code)
    except Exception as e:
        print(f"获取基金{fund_code}信息失败: {str(e)}")
        return {}
    info = {}
    for _, row in df.iterrows():
        k = str(row.get("item", "")).strip()
        v = str(row.get("value", "")).strip()
        if k:
            info[k] = v
    info["基金代码"] = fund_code
    return info

def fetch_fund_weekly_nav(fund_code: str, years: int = 3) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
    except Exception:
        try:
            df = ak.fund_em_open_fund_info(fund=fund_code, indicator="单位净值走势")
        except Exception:
            return pd.DataFrame()
    
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    # 统一日期列名和格式
    date_col = "净值日期" if "净值日期" in df.columns else "日期" if "日期" in df.columns else None
    if not date_col:
        return pd.DataFrame()
    
    df["date"] = pd.to_datetime(df[date_col])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    
    # 统一净值列名
    price_col = "单位净值" if "单位净值" in df.columns else "收盘" if "收盘" in df.columns else None
    if not price_col:
        return pd.DataFrame()
    
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values("date").dropna(subset=[price_col, "date"]).set_index("date")
    
    # 周频重采样（周五），补充多个均线
    weekly = df[[price_col]].resample("W-FRI").last().dropna()
    weekly = weekly.rename(columns={price_col: "close"})
    
    # 增加多周期均线（10/20/30周）
    weekly["ma10"] = weekly["close"].rolling(10).mean()
    weekly["ma20"] = weekly["close"].rolling(20).mean()
    weekly["ma30"] = weekly["close"].rolling(30).mean()
    
    # 收益率（简单/对数）
    weekly["ret"] = weekly["close"].pct_change()
    weekly["log_ret"] = np.log(weekly["close"] / weekly["close"].shift(1))
    
    # 波动率（20周）
    weekly["vol"] = weekly["ret"].rolling(20).std()
    
    return weekly.dropna()

def fetch_index_weekly_close(index_symbol: str = "sh000300", years: int = 5) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
    
    try:
        df = ak.stock_zh_index_daily(symbol=index_symbol)
    except Exception:
        try:
            code = index_symbol.replace("sh", "").replace("sz", "")
            df = ak.index_zh_a_hist(symbol=code, period="daily")
        except Exception as e:
            print(f"获取指数{index_symbol}数据失败: {str(e)}")
            return pd.DataFrame()
    
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    # 统一日期和价格列
    date_col = "日期" if "日期" in df.columns else "date" if "date" in df.columns else None
    price_col = "收盘" if "收盘" in df.columns else "close" if "close" in df.columns else None
    if not date_col or not price_col:
        return pd.DataFrame()
    
    df["date"] = pd.to_datetime(df[date_col])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    
    df = df.sort_values("date").dropna(subset=[price_col, "date"]).set_index("date")
    
    # 周频重采样
    weekly = df[[price_col]].resample("W-FRI").last().dropna()
    weekly = weekly.rename(columns={price_col: "close"})
    weekly["ret"] = weekly["close"].pct_change()
    weekly["log_ret"] = np.log(weekly["close"] / weekly["close"].shift(1))
    
    return weekly.dropna()

# ===================== 核心分析函数（重点优化） =====================
def compute_ma_slope(series: pd.Series, window: int = 10) -> tuple[float, float]:
    """
    计算均线斜率 + R²（拟合优度），更准确判断趋势
    return: (斜率, R²)
    """
    if len(series.dropna()) < window:
        return 0.0, 0.0
    
    series_clean = series.dropna().iloc[-window:]
    x = np.arange(len(series_clean))
    y = series_clean.values
    
    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # 标准化斜率（消除量纲影响）
    slope_normalized = slope / series_clean.mean() * 100
    
    return slope_normalized, r_value **2

def judge_stage_enhanced(weekly_df: pd.DataFrame) -> dict:
    """
    增强版阶段判断：
    1: 筑底阶段 | 2: 上升阶段 | 3: 顶部震荡 | 4: 下跌阶段
    返回详细的阶段信息而非单一数字
    """
    if len(weekly_df) < 30:
        return {"stage": 0, "confidence": 0.0, "reason": "数据不足"}
    
    latest = weekly_df.iloc[-1]
    prev_week = weekly_df.iloc[-2]
    
    # 核心指标计算
    ma10, ma20, ma30 = latest["ma10"], latest["ma20"], latest["ma30"]
    close = latest["close"]
    vol = latest["vol"]
    avg_vol = weekly_df["vol"].rolling(20).mean().iloc[-1]
    
    # 均线斜率 + 拟合优度（判断趋势强度）
    slope30, r2_30 = compute_ma_slope(weekly_df["ma30"], 10)
    slope20, r2_20 = compute_ma_slope(weekly_df["ma20"], 10)
    
    # 价格相对均线的位置
    diff30 = (close / ma30 - 1) * 100  # 百分比
    diff20 = (close / ma20 - 1) * 100
    diff10 = (close / ma10 - 1) * 100
    
    # 均线排列（多头/空头）
    ma_arrangement = 1 if ma10 > ma20 > ma30 else -1 if ma10 < ma20 < ma30 else 0
    
    # 波动率判断（趋势确认）
    vol_ratio = vol / avg_vol if avg_vol > 0 else 1
    
    # 阶段判断逻辑（增强版）
    stage = 0
    confidence = 0.0
    reason = ""
    
    # 4: 下跌阶段（高置信度）
    if (diff30 < -3 and slope30 < -0.1 and r2_30 > 0.7 and 
        ma_arrangement == -1 and close < prev_week["close"]):
        stage = 4
        confidence = min(0.9, abs(slope30)/1 + (abs(diff30)/5) + (1 - vol_ratio/2))
        reason = f"价格低于30周均线{diff30:.1f}%，均线斜率{slope30:.2f}（R²={r2_30:.2f}），空头排列，确认下跌趋势"
    
    # 2: 上升阶段（高置信度）
    elif (diff30 > 1 and slope30 > 0.1 and r2_30 > 0.7 and 
          ma_arrangement == 1 and close > prev_week["close"] and vol_ratio > 0.8):
        stage = 2
        confidence = min(0.9, slope30/1 + diff30/10 + r2_30)
        reason = f"价格高于30周均线{diff30:.1f}%，均线斜率{slope30:.2f}（R²={r2_30:.2f}），多头排列，量能充足，确认上升趋势"
    
    # 3: 顶部震荡
    elif (diff30 > 0 and slope30 < 0.1 and r2_30 < 0.5 and 
          abs(diff10) < 2 and vol_ratio > 1.2):
        stage = 3
        confidence = min(0.8, (abs(diff10)/2) + (vol_ratio/2) + (1 - r2_30))
        reason = f"价格在均线上方但均线走平，短期波动加大（波动率{vol_ratio:.1f}倍），顶部震荡特征"
    
    # 1: 筑底阶段
    elif (diff30 < 0 and slope30 > -0.1 and r2_30 < 0.5 and 
          vol_ratio < 0.8 and close > ma10):
        stage = 1
        confidence = min(0.8, (1 - abs(diff30)/5) + (1 - vol_ratio) + r2_30)
        reason = f"价格接近均线（偏离{diff30:.1f}%），成交量萎缩（波动率{vol_ratio:.1f}倍），筑底特征明显"
    
    # 未明确阶段
    else:
        stage = 1
        confidence = 0.3
        reason = f"无明确趋势特征，均线偏离{diff30:.1f}%，斜率{slope30:.2f}，暂归为筑底观察"
    
    return {
        "stage": stage,
        "confidence": round(confidence, 2),
        "reason": reason,
        "key_metrics": {
            "ma30_diff_pct": round(diff30, 2),
            "ma30_slope": round(slope30, 2),
            "ma30_r2": round(r2_30, 2),
            "vol_ratio": round(vol_ratio, 2),
            "ma_arrangement": ma_arrangement
        }
    }

def relative_strength_enhanced(fund_weekly: pd.DataFrame, index_weekly: pd.DataFrame, lookback_periods: list = [12, 26, 52]) -> dict:
    """
    增强版相对强度：
    1. 多周期计算（3/6/12个月）
    2. 考虑波动率调整的超额收益
    3. 胜率（基金跑赢指数的周数占比）
    """
    # 对齐日期
    aligned = fund_weekly.join(index_weekly[["ret", "log_ret"]], how="inner", rsuffix="_index")
    if len(aligned) < max(lookback_periods):
        return {"rs_scores": {}, "win_rate": 0.0, "risk_adjusted_rs": 0.0}
    
    rs_scores = {}
    win_rates = {}
    
    for lookback in lookback_periods:
        if len(aligned) < lookback:
            rs_scores[f"{lookback}周"] = 0.0
            win_rates[f"{lookback}周"] = 0.0
            continue
        
        window = aligned.iloc[-lookback:]
        
        # 累计收益
        fund_cum_ret = (1 + window["ret"]).prod() - 1
        index_cum_ret = (1 + window["ret_index"]).prod() - 1
        rs_scores[f"{lookback}周"] = round(fund_cum_ret - index_cum_ret, 4)
        
        # 胜率
        fund_win = (window["ret"] > window["ret_index"]).sum()
        win_rates[f"{lookback}周"] = round(fund_win / len(window), 2)
    
    # 风险调整后的超额收益（夏普比率思路）
    full_window = aligned.iloc[-52:] if len(aligned) >= 52 else aligned
    excess_ret = full_window["log_ret"] - full_window["log_ret_index"]
    risk_adjusted_rs = excess_ret.mean() / excess_ret.std() if excess_ret.std() > 0 else 0.0
    
    return {
        "rs_scores": rs_scores,
        "win_rates": win_rates,
        "risk_adjusted_rs": round(risk_adjusted_rs, 3),
        "latest_rs": rs_scores.get("12周", 0.0)
    }

def risk_assessment(weekly_df: pd.DataFrame) -> dict:
    """
    风险评估：
    1. 最大回撤
    2. 下行波动率
    3. 夏普比率（无风险利率按年化2%计算）
    """
    if len(weekly_df) < 20:
        return {"max_drawdown": 0.0, "downside_vol": 0.0, "sharpe": 0.0}
    
    # 最大回撤
    roll_max = weekly_df["close"].rolling(window=len(weekly_df), min_periods=1).max()
    drawdown = (weekly_df["close"] / roll_max - 1) * 100
    max_drawdown = round(drawdown.min(), 2)
    
    # 下行波动率（仅考虑负收益）
    downside_returns = weekly_df["ret"][weekly_df["ret"] < 0]
    downside_vol = downside_returns.std() * np.sqrt(52)  # 年化
    downside_vol = round(downside_vol * 100, 2)
    
    # 夏普比率（周度收益年化 - 无风险利率）/ 年化波动率
    annual_ret = weekly_df["ret"].mean() * 52
    annual_vol = weekly_df["ret"].std() * np.sqrt(52)
    sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else 0.0
    
    return {
        "max_drawdown": max_drawdown,
        "downside_vol_pct": downside_vol,
        "sharpe_ratio": round(sharpe, 3)
    }

def generate_advice_enhanced(stage_info: dict, rs_info: dict, risk_info: dict) -> dict:
    """
    增强版投资建议：
    结合阶段、相对强度、风险指标给出分级建议
    """
    stage = stage_info["stage"]
    stage_confidence = stage_info["confidence"]
    rs_latest = rs_info["latest_rs"]
    risk_adjusted_rs = rs_info["risk_adjusted_rs"]
    max_dd = risk_info["max_drawdown"]
    sharpe = risk_info["sharpe_ratio"]
    
    # 基础建议
    base_advice = {
        0: ("等待", "数据不足，无法判断趋势", 0),
        1: ("观望", "筑底阶段，等待放量突破确认", 50),
        2: ("买入", "上升阶段，趋势明确", 80),
        3: ("减仓", "顶部震荡，趋势走弱", 40),
        4: ("卖出", "下跌阶段，风险较高", 20)
    }
    
    action, note, base_score = base_advice.get(stage, ("观望", "无明确判断", 30))
    
    # 调整因子：置信度、相对强度、风险
    confidence_factor = stage_confidence  # 0-0.9
    rs_factor = min(1.0, max(0.0, rs_latest + 0.1))  # 超额收益调整
    risk_factor = min(1.0, max(0.5, 1 + (sharpe / 2)))  # 夏普比率调整
    
    # 最终评分（0-100）
    final_score = round(base_score * confidence_factor * rs_factor * risk_factor, 1)
    
    # 细化建议
    if stage == 2:
        # 上升阶段细分
        if rs_latest > 0.05 and sharpe > 1.0:
            note = f"{note}，相对强度优秀（{rs_latest:.1%}），风险调整收益高（夏普{sharpe}），建议重仓配置"
            action = "重仓买入"
        elif rs_latest > 0 and sharpe > 0:
            note = f"{note}，相对强度良好（{rs_latest:.1%}），建议分批建仓/定投"
        else:
            note = f"{note}，但相对强度偏弱（{rs_latest:.1%}）/风险较高（最大回撤{max_dd}%），建议轻仓参与"
            action = "轻仓买入"
    
    elif stage == 1:
        # 筑底阶段细分
        if rs_latest > 0 and max_dd > -10:
            note = f"{note}，但相对强度为正（{rs_latest:.1%}）且回撤可控，可小仓位左侧布局"
            action = "轻仓布局"
    
    elif stage == 3:
        # 顶部震荡细分
        if rs_latest > 0 and stage_confidence < 0.5:
            note = f"{note}，但相对强度仍为正（{rs_latest:.1%}），建议部分止盈而非全部卖出"
            action = "部分止盈"
    
    elif stage == 4:
        # 下跌阶段细分
        if max_dd < -20 and sharpe < 0:
            note = f"{note}，最大回撤{max_dd}%，夏普比率{sharpe}，建议立即止损"
            action = "止损卖出"
    
    # 仓位建议
    position_suggestion = {
        "重仓买入": 70-90,
        "买入": 40-60,
        "轻仓买入": 10-30,
        "轻仓布局": 5-15,
        "部分止盈": 20-40,
        "减仓": 0-20,
        "卖出": 0,
        "止损卖出": 0,
        "观望": 0
    }
    
    return {
        "建议操作": action,
        "建议仓位(%)": position_suggestion.get(action, 0),
        "建议说明": note,
        "评分": final_score,
        "建议置信度": round(stage_confidence * 100, 1)
    }

# ===================== 主分析函数 =====================
def analyze_fund_enhanced(fund_code: str, benchmark_code: str = "sh000300") -> dict:
    """
    增强版基金分析主函数
    """
    # 1. 获取基础数据
    fund_info = fetch_fund_info(fund_code)
    fund_weekly = fetch_fund_weekly_nav(fund_code, years=3)
    index_weekly = fetch_index_weekly_close(benchmark_code, years=3)
    
    # 数据校验
    if len(fund_weekly) == 0:
        return {"错误": "无法获取基金净值数据", "基金代码": fund_code}
    if len(index_weekly) == 0:
        return {"错误": "无法获取基准指数数据", "基金代码": fund_code}
    
    # 2. 核心分析
    stage_result = judge_stage_enhanced(fund_weekly)
    rs_result = relative_strength_enhanced(fund_weekly, index_weekly)
    risk_result = risk_assessment(fund_weekly)
    advice_result = generate_advice_enhanced(stage_result, rs_result, risk_result)
    
    # 3. 整合结果
    latest_date = fund_weekly.index[-1].strftime("%Y-%m-%d")
    latest_close = round(float(fund_weekly.iloc[-1]["close"]), 4)
    latest_ma30 = round(float(fund_weekly.iloc[-1]["ma30"]), 4)
    
    result = {
        "基金基本信息": fund_info,
        "分析日期": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "最新数据": {
            "净值日期": latest_date,
            "单位净值": latest_close,
            "30周均线": latest_ma30,
            "最大回撤(%)": risk_result["max_drawdown"],
            "下行波动率(%)": risk_result["downside_vol_pct"],
            "夏普比率": risk_result["sharpe_ratio"]
        },
        "趋势分析": stage_result,
        "相对强度分析": rs_result,
        "风险评估": risk_result,
        "投资建议": advice_result
    }
    
    return result