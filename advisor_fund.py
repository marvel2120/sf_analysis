import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import akshare as ak
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RS_POSITIVE_THRESHOLD = 0.0
MAX_DRAWDOWN_LIMIT = -30.0
STAGE_CONFIDENCE_MIN = 0.5

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

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    计算RSI相对强弱指标
    return: RSI值 (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # 中性值
    
    # 计算价格变化
    delta = prices.diff()
    
    # 分离上涨和下跌
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # 计算平均上涨和下跌
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # 计算RS和RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

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
    
    # 计算价格动量（避免追高）
    price_change_4w = (close / weekly_df["close"].iloc[-4] - 1) * 100 if len(weekly_df) >= 4 else 0
    price_change_8w = (close / weekly_df["close"].iloc[-8] - 1) * 100 if len(weekly_df) >= 8 else 0
    
    # 计算RSI（相对强弱指标，避免超买）
    rsi = calculate_rsi(weekly_df["close"], 14)
    
    # 阶段判断逻辑（优化版 - 增加顶部识别）
    stage = 0
    confidence = 0.0
    reason = ""
    
    # 4: 下跌阶段（高置信度）
    if (diff30 < -3 and slope30 < -0.1 and r2_30 > 0.7 and 
        ma_arrangement == -1 and close < prev_week["close"]):
        stage = 4
        confidence = min(0.9, abs(slope30)/1 + (abs(diff30)/5) + (1 - vol_ratio/2))
        reason = f"价格低于30周均线{diff30:.1f}%，均线斜率{slope30:.2f}（R²={r2_30:.2f}），空头排列，确认下跌趋势"
    
    # 3: 顶部震荡（增强识别）
    elif (diff30 > 2 and slope30 < 0.05 and r2_30 < 0.6 and 
          (price_change_4w > 8 or price_change_8w > 15) and 
          (rsi > 70 or vol_ratio > 1.5)):
        stage = 3
        confidence = min(0.85, (abs(diff30)/3) + (vol_ratio/3) + (rsi/100))
        reason = f"价格远离均线（偏离{diff30:.1f}%），短期涨幅过大（4周{price_change_4w:.1f}%），RSI超买（{rsi:.0f}），顶部风险高"
    
    # 2: 上升阶段（增加买入条件，避免追高）
    elif (diff30 > 1 and diff30 < 8 and slope30 > 0.1 and r2_30 > 0.7 and 
          ma_arrangement == 1 and close > prev_week["close"] and 
          vol_ratio > 0.8 and price_change_4w < 10 and rsi < 75):
        stage = 2
        confidence = min(0.9, slope30/1 + diff30/8 + r2_30)
        reason = f"价格高于30周均线{diff30:.1f}%（未远离），均线斜率{slope30:.2f}（R²={r2_30:.2f}），多头排列，量能充足，上升趋势健康"
    
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
        reason = f"无明确趋势特征，均线偏离{diff30:.1f}%，斜率{slope30:.2f}，短期涨幅{price_change_4w:.1f}%，暂归为筑底观察"
    
    return {
        "stage": stage,
        "confidence": round(confidence, 2),
        "reason": reason,
        "key_metrics": {
            "ma30_diff_pct": round(diff30, 2),
            "ma30_slope": round(slope30, 2),
            "ma30_r2": round(r2_30, 2),
            "vol_ratio": round(vol_ratio, 2),
            "ma_arrangement": ma_arrangement,
            "price_change_4w": round(price_change_4w, 2),
            "price_change_8w": round(price_change_8w, 2),
            "rsi": round(rsi, 1)
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
        return {"rs_scores": {}, "win_rates": {}, "risk_adjusted_rs": 0.0, "latest_rs": 0.0}
    
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
        return {
            "max_drawdown": 0.0,
            "downside_vol_pct": 0.0,
            "sharpe_ratio": 0.0,
            "annual_return_pct": 0.0,
        }
    
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
    
    # risk_assessment函数内补充
    annual_return = weekly_df["log_ret"].mean() * 52 * 100  # 年化收益率(%)
    # 返回值补充
    return {
        "max_drawdown": max_drawdown,
        "downside_vol_pct": downside_vol,
        "sharpe_ratio": round(sharpe, 3),
        "annual_return_pct": round(annual_return, 2)  # 新增年化收益率
    }

def generate_advice_enhanced(stage_info: dict, rs_info: dict, risk_info: dict) -> dict:
    """
    优化版投资建议：
    1. 增加买入时机判断，避免追高
    2. 强化风险控制逻辑
    3. 细化仓位管理策略
    """
    stage = stage_info.get("stage", 0)
    stage_confidence = stage_info.get("confidence", 0.0)
    key_metrics = stage_info.get("key_metrics", {})
    
    rs_latest = rs_info.get("latest_rs", 0.0)
    risk_adjusted_rs = rs_info.get("risk_adjusted_rs", 0.0)
    
    max_dd = risk_info.get("max_drawdown", 0.0)
    sharpe = risk_info.get("sharpe_ratio", 0.0)
    annual_return = risk_info.get("annual_return_pct", 0.0)
    
    # 提取关键指标用于决策
    rsi = key_metrics.get("rsi", 50.0)
    price_change_4w = key_metrics.get("price_change_4w", 0.0)
    ma30_diff = key_metrics.get("ma30_diff_pct", 0.0)
    
    # 基础建议模板（根据阶段、风险、相对强度综合判断）
    base_recommendations = {
        0: ("等待", "数据不足，无法判断趋势", 0, 0),
        1: ("观望", "筑底阶段，等待更明确信号", 45, 15),  # 提高基础分数，降低仓位
        2: ("买入", "上升阶段，趋势明确", 75, 50),      # 降低基础分数，避免过度乐观
        3: ("减仓", "顶部震荡，控制风险", 35, 20),      # 顶部阶段，保守处理
        4: ("卖出", "下跌阶段，规避风险", 15, 0)        # 严格控制风险
    }
    
    action, base_note, base_score, base_position = base_recommendations.get(stage, ("观望", "无明确判断", 30, 10))
    
    # ===== 风险控制因子 =====
    risk_factors = []
    risk_score = 0
    
    # 1. 回撤风险因子
    if max_dd < -25:  # 回撤过大
        risk_factors.append(f"最大回撤过大({max_dd:.1f}%)")
        risk_score -= 15
    elif max_dd < -15:  # 回撤较大
        risk_factors.append(f"回撤较大({max_dd:.1f}%)")
        risk_score -= 8
    
    # 2. 夏普比率因子
    if sharpe < -0.5:  # 风险极高
        risk_factors.append(f"风险极高(夏普{sharpe:.2f})")
        risk_score -= 20
    elif sharpe < 0:   # 风险较高
        risk_factors.append(f"风险较高(夏普{sharpe:.2f})")
        risk_score -= 10
    elif sharpe < 0.3: # 风险一般
        risk_factors.append(f"风险一般(夏普{sharpe:.2f})")
        risk_score -= 5
    elif sharpe > 1.0: # 风险控制好
        risk_score += 5
    
    # 3. 买入时机因子（避免追高）
    if stage == 2 and price_change_4w > 12:  # 短期涨幅过大
        risk_factors.append(f"短期涨幅过大({price_change_4w:.1f}%)")
        risk_score -= 12
        action = "谨慎买入"  # 修改建议
        base_note = "上升趋势但短期涨幅过大，建议等待回调"
    
    if rsi > 75:  # RSI超买
        risk_factors.append(f"RSI超买({rsi:.0f})")
        risk_score -= 8
    elif rsi < 25:  # RSI超卖
        risk_score += 3
    
    # 4. 均线偏离因子
    if abs(ma30_diff) > 10:  # 远离均线
        risk_factors.append(f"远离均线({ma30_diff:+.1f}%)")
        risk_score -= 6
    
    # ===== 收益增强因子 =====
    enhancement_factors = []
    enhancement_score = 0
    
    # 1. 相对强度因子
    if rs_latest > 0.08:  # 相对强度优秀
        enhancement_factors.append(f"相对强度优秀({rs_latest:.1%})")
        enhancement_score += 10
    elif rs_latest > 0.03:  # 相对强度良好
        enhancement_factors.append(f"相对强度良好({rs_latest:.1%})")
        enhancement_score += 5
    elif rs_latest < -0.05: # 相对强度较弱
        enhancement_factors.append(f"相对强度较弱({rs_latest:.1%})")
        enhancement_score -= 8
    
    # 2. 风险调整后收益因子
    if risk_adjusted_rs > 0.5:  # 风险调整收益优秀
        enhancement_factors.append(f"风险调整收益优秀({risk_adjusted_rs:.2f})")
        enhancement_score += 8
    elif risk_adjusted_rs > 0.2:  # 风险调整收益良好
        enhancement_factors.append(f"风险调整收益良好({risk_adjusted_rs:.2f})")
        enhancement_score += 4
    
    # 3. 年化收益因子
    if annual_return > 15:  # 年化收益优秀
        enhancement_factors.append(f"年化收益优秀({annual_return:.1f}%)")
        enhancement_score += 6
    elif annual_return > 8:   # 年化收益良好
        enhancement_factors.append(f"年化收益良好({annual_return:.1f}%)")
        enhancement_score += 3
    
    # ===== 综合评分计算 =====
    final_score = base_score + enhancement_score + risk_score
    final_score = max(0, min(100, final_score))  # 限制在0-100范围内
    
    # 根据综合评分调整建议
    if final_score >= 80:
        action = "重仓买入"
        final_position = 70
    elif final_score >= 65:
        action = "买入"
        final_position = 50
    elif final_score >= 50:
        action = "轻仓买入"
        final_position = 30
    elif final_score >= 35:
        action = "观望"
        final_position = 15
    elif final_score >= 20:
        action = "减仓"
        final_position = 5
    else:
        action = "卖出"
        final_position = 0
    
    # ===== 生成详细说明 =====
    detailed_note = base_note
    
    if enhancement_factors:
        detailed_note += f"；收益因素：[{'; '.join(enhancement_factors)}]"
    
    if risk_factors:
        detailed_note += f"；风险因素：[{'; '.join(risk_factors)}]"
    
    # 添加具体的投资建议
    if stage == 2 and final_score < 65:
        detailed_note += "；建议等待回调至均线附近再考虑买入"
    elif stage == 1 and final_score > 50:
        detailed_note += "；可考虑分批建仓，设置止损位"
    elif stage == 3:
        detailed_note += "；建议分批减仓，锁定收益"
    elif stage == 4:
        detailed_note += "；建议立即减仓或止损，保护本金"
    
    return {
        "建议操作": action,
        "建议仓位(%)": final_position,
        "建议说明": detailed_note,
        "评分": round(final_score, 1),
        "建议置信度": round(stage_confidence * 100, 1),
        "风险评分": risk_score,
        "增强评分": enhancement_score
    }

# ===================== 回测函数 =====================
def backtest_fund_strategy(fund_code: str, benchmark_code: str = "sh000300", years: int = 5) -> dict:
    fund_weekly = fetch_fund_weekly_nav(fund_code, years=years)
    index_weekly = fetch_index_weekly_close(benchmark_code, years=years)
    if len(fund_weekly) < 60 or len(index_weekly) < 60:
        return {"错误": "历史数据不足以回测", "基金代码": fund_code}
    aligned = fund_weekly.join(index_weekly[["ret"]], how="inner", rsuffix="_index")
    dates = aligned.index
    if len(dates) < 60:
        return {"错误": "历史数据不足以回测", "基金代码": fund_code}
    start_idx = 30
    positions = []
    strat_ret = []
    for i in range(start_idx, len(dates) - 1):
        end_date = dates[i]
        slice_fund = fund_weekly.loc[:end_date]
        slice_index = index_weekly.loc[:end_date]
        stage_result = judge_stage_enhanced(slice_fund)
        rs_result = relative_strength_enhanced(slice_fund, slice_index)
        risk_result = risk_assessment(slice_fund)
        advice_result = generate_advice_enhanced(stage_result, rs_result, risk_result)
        pos = advice_result.get("建议仓位(%)", 0) / 100.0
        next_date = dates[i + 1]
        r = fund_weekly.loc[next_date, "ret"]
        positions.append(pos)
        strat_ret.append(pos * r)
    strat_ret_series = pd.Series(strat_ret, index=dates[start_idx + 1:])
    if len(strat_ret_series) == 0:
        return {"错误": "回测窗口为空", "基金代码": fund_code}
    equity = (1 + strat_ret_series).cumprod()
    weeks = len(strat_ret_series)
    annual_ret = (equity.iloc[-1] ** (52 / weeks) - 1) * 100
    roll_max = equity.cummax()
    drawdown = (equity / roll_max - 1) * 100
    max_drawdown = drawdown.min()
    win_rate = (strat_ret_series > 0).sum() / weeks * 100
    pos_series = pd.Series(positions, index=dates[start_idx:len(dates) - 1])
    trades = (pos_series.diff().abs() > 0.05).sum()
    summary = {
        "年化收益率(%)": round(float(annual_ret), 2),
        "最大回撤(%)": round(float(max_drawdown), 2),
        "胜率(%)": round(float(win_rate), 2),
        "交易次数": int(trades),
    }
    equity_df = pd.DataFrame({"策略净值": equity})
    return {"基金代码": fund_code, "回测概要": summary, "净值曲线": equity_df}

def generate_trading_strategy(advice_result: dict, stage_info: dict, latest_data: dict) -> dict:
    """
    生成具体的交易策略：
    1. 分批买入计划
    2. 止损位设置
    3. 加仓/减仓条件
    4. 目标收益设定
    """
    action = advice_result.get("建议操作", "观望")
    suggested_position = advice_result.get("建议仓位(%)", 0)
    
    # 获取关键数据
    latest_nav = latest_data.get("单位净值", 0)
    ma30 = latest_data.get("30周均线", 0)
    stage = stage_info.get("stage", 0)
    key_metrics = stage_info.get("key_metrics", {})
    
    strategy = {
        "操作计划": "",
        "分批买入": [],
        "止损位": 0,
        "目标位": 0,
        "加仓条件": "",
        "减仓条件": "",
        "风险提示": ""
    }
    
    if action in ["买入", "轻仓买入", "重仓买入"] and latest_nav > 0:
        # 分批买入策略
        if suggested_position >= 50:  # 重仓买入
            strategy["分批买入"] = [
                {"批次": 1, "比例": 30, "条件": f"当前价位{latest_nav:.3f}附近"},
                {"批次": 2, "比例": 30, "条件": f"回调至{latest_nav*0.97:.3f}或突破前高"},
                {"批次": 3, "比例": 40, "条件": f"确认趋势后加仓至{latest_nav*1.05:.3f}"}
            ]
        elif suggested_position >= 30:  # 轻仓买入
            strategy["分批买入"] = [
                {"批次": 1, "比例": 60, "条件": f"当前价位{latest_nav:.3f}附近"},
                {"批次": 2, "比例": 40, "条件": f"回调至{latest_nav*0.95:.3f}加仓"}
            ]
        else:  # 试探性买入
            strategy["分批买入"] = [
                {"批次": 1, "比例": 100, "条件": f"当前价位{latest_nav:.3f}附近，小仓位试探"}
            ]
        
        # 止损位设置（基于技术位和回撤容忍度）
        if stage == 2:  # 上升趋势
            # 止损设在30周均线下方5%或前期低点
            stop_loss_1 = ma30 * 0.95
            stop_loss_2 = latest_nav * 0.92  # 8%止损
            strategy["止损位"] = max(stop_loss_1, stop_loss_2)
        else:  # 其他阶段更保守
            strategy["止损位"] = latest_nav * 0.90  # 10%止损
        
        # 目标位设定
        if stage == 2:
            strategy["目标位"] = latest_nav * 1.20  # 20%目标收益
        else:
            strategy["目标位"] = latest_nav * 1.15  # 15%目标收益
        
        # 加仓条件
        strategy["加仓条件"] = f"1) 价格站稳30周均线上方；2) 相对强度持续改善；3) 成交量放大确认突破"
        
        # 减仓条件
        strategy["减仓条件"] = f"1) 价格跌破止损位{strategy['止损位']:.3f}；2) 相对强度明显恶化；3) 趋势阶段转为下跌"
        
        strategy["操作计划"] = f"建议{suggested_position}%仓位，分{len(strategy['分批买入'])}批买入，严格止损"
        strategy["风险提示"] = "市场有风险，投资需谨慎。建议单只基金仓位不超过总资产的20%"
        
    elif action in ["减仓", "卖出"]:
        strategy["操作计划"] = f"建议减仓至{suggested_position}%或清仓"
        strategy["减仓条件"] = "1) 趋势明显转弱；2) 相对强度持续下降；3) 达到止损位"
        strategy["风险提示"] = "下跌趋势中，保护本金为第一要务"
        
    elif action == "观望":
        strategy["操作计划"] = "暂时观望，等待更明确信号"
        strategy["加仓条件"] = "1) 趋势明确转向上升；2) 相对强度转正且持续改善"
        strategy["风险提示"] = "方向不明时，耐心等待比盲目操作更安全"
    
    return strategy

# ===================== 主分析函数 =====================
def analyze_fund_enhanced(fund_code: str, benchmark_code: str = "sh000300") -> dict:
    """
    优化版基金分析主函数 - 整合新的投资策略和风险控制
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
    
    # 2. 核心分析（使用优化后的函数）
    stage_result = judge_stage_enhanced(fund_weekly)
    rs_result = relative_strength_enhanced(fund_weekly, index_weekly)
    risk_result = risk_assessment(fund_weekly)
    advice_result = generate_advice_enhanced(stage_result, rs_result, risk_result)
    
    # 3. 生成具体交易策略
    latest_data = {
        "净值日期": fund_weekly.index[-1].strftime("%Y-%m-%d"),
        "单位净值": round(float(fund_weekly.iloc[-1]["close"]), 4),
        "30周均线": round(float(fund_weekly.iloc[-1]["ma30"]), 4),
        "最大回撤(%)": risk_result.get("max_drawdown", 0.0),
        "下行波动率(%)": risk_result.get("downside_vol_pct", 0.0),
        "夏普比率": risk_result.get("sharpe_ratio", 0.0),
        "年化收益率(%)": risk_result.get("annual_return_pct", 0.0),
    }
    
    trading_strategy = generate_trading_strategy(advice_result, stage_result, latest_data)
    
    # 4. 整合完整结果
    industry_tag = fund_info.get("投资类型") or fund_info.get("基金类型") or fund_info.get("投资风格") or ""
    
    result = {
        "基金基本信息": fund_info,
        "行业标签": industry_tag,
        "分析日期": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "最新数据": latest_data,
        "趋势分析": stage_result,
        "相对强度分析": rs_result,
        "风险评估": risk_result,
        "投资建议": advice_result,
        "交易策略": trading_strategy,
        "历史周度数据": fund_weekly[["close", "ma10", "ma20", "ma30"]]
    }
    
    return result