"""
基金分析模块（中短线优化版 v2.0）

核心优化：
1. 模糊阶段判定取代硬阈值 — 不再"强行归入筑底"，输出各阶段概率
2. 市场状态识别 — 根据沪深300判断牛/熊/震荡，动态调整仓位上限
3. ATR动态仓位管理 — 波动率高时降仓位、放宽止损
4. DeepSeek API 交叉验证 — 大模型对信号给出第二意见
5. ML 分类器融合 — RandomForest 与规则系统加权融合
6. 中短线优化 — 主均线改为20周，近年收益更高权重
7. 删除冗余 — 去掉 RS_POSITIVE_THRESHOLD/MAX_DRAWDOWN_LIMIT/STAGE_CONFIDENCE_MIN
8. 删除评分重复计算 — 夏普和回撤不再双重惩罚
"""
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import akshare as ak
import warnings
warnings.filterwarnings('ignore')

from market_utils import (
    compute_ma_slope, compute_rsi_series, calculate_rsi, compute_atr,
    compute_bollinger_bands, compute_macd,
    fetch_index_weekly_close,
    relative_strength_enhanced, risk_assessment,
    detect_market_regime, calculate_position_size,
    fuzzy_stage_judgment
)
from ml_classifier import MLStageClassifier
from deepseek_integration import DeepSeekClient
import config as cfg


# ===================== 全局初始化（延迟初始化） =====================
_deepseek_client = None
_ml_classifier = None


def get_deepseek_client() -> DeepSeekClient:
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = DeepSeekClient(
            api_key=cfg.DEEPSEEK_CONFIG.get('api_key', ''),
            model=cfg.DEEPSEEK_CONFIG.get('model', 'deepseek-chat'),
            timeout=cfg.DEEPSEEK_CONFIG.get('timeout', 15)
        )
    return _deepseek_client


def get_ml_classifier() -> MLStageClassifier:
    global _ml_classifier
    if _ml_classifier is None:
        _ml_classifier = MLStageClassifier(min_samples=cfg.ML_CONFIG.get('min_samples', 80))
    return _ml_classifier


# ===================== 数据获取 =====================

def fetch_fund_info(fund_code: str) -> dict:
    """获取基金基本信息"""
    try:
        df = ak.fund_individual_basic_info_xq(symbol=fund_code)
    except Exception as e:
        print(f"获取基金{fund_code}信息失败: {str(e)}")
        return {"基金代码": fund_code}
    info = {}
    for _, row in df.iterrows():
        k = str(row.get("item", "")).strip()
        v = str(row.get("value", "")).strip()
        if k:
            info[k] = v
    info["基金代码"] = fund_code
    return info


def fetch_fund_weekly_nav(fund_code: str, years: int = 3) -> pd.DataFrame:
    """
    获取基金周线数据

    中短线优化：
    - 主均线改为20周（比30周更敏感）
    - 保留30周作为长期参考
    - 增加ATR计算
    - 增加MACD
    """
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

    date_col = "净值日期" if "净值日期" in df.columns else "日期" if "日期" in df.columns else None
    if not date_col:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    price_col = "单位净值" if "单位净值" in df.columns else "收盘" if "收盘" in df.columns else None
    if not price_col:
        return pd.DataFrame()

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values("date").dropna(subset=[price_col, "date"]).set_index("date")

    weekly = df[[price_col]].resample("W-FRI").last().dropna()
    weekly = weekly.rename(columns={price_col: "close"})

    # 多周期均线（中短线：主用20周，参考10周和30周）
    weekly["ma10"] = weekly["close"].rolling(10).mean()
    weekly["ma20"] = weekly["close"].rolling(20).mean()
    weekly["ma30"] = weekly["close"].rolling(30).mean()

    # 收益率
    weekly["ret"] = weekly["close"].pct_change()
    weekly["log_ret"] = np.log(weekly["close"] / weekly["close"].shift(1))

    # 波动率
    weekly["vol"] = weekly["ret"].rolling(20).std()

    # 基金没有 high/low/volume，但我们可以用周内日频数据模拟
    # 对于场外基金，净值只有每日一个价格，所以ATR用 close 变化近似
    weekly["atr"] = weekly["close"].diff().abs().rolling(14).mean()

    # 布林带（基于close本身）
    bb = compute_bollinger_bands(weekly["close"], 20, 2)
    weekly["bb_upper"] = bb["upper"]
    weekly["bb_lower"] = bb["lower"]

    # MACD
    macd = compute_macd(weekly["close"])
    weekly["macd"] = macd["macd"]
    weekly["macd_signal"] = macd["signal"]
    weekly["macd_histogram"] = macd["histogram"]

    return weekly.dropna()


# ===================== 评分系统（v2 优化版）=========================

def generate_advice_v2(
    fuzzy_result: dict,
    rs_info: dict,
    risk_info: dict,
    market_regime: dict,
    position_info: dict,
    deepseek_result: dict = None,
    weekly_df: pd.DataFrame = None
) -> dict:
    """
    v2 评分系统

    与 v1 核心区别：
    1. 使用模糊概率作为基础分（不再是硬编码的阶段固定分）
    2. 市场状态影响仓位上限
    3. ATR动态仓位置入
    4. 删除夏普/回撤双重计算
    5. 中短线加持：近期动量权重更高
    6. DeepSeek意见可微调评分
    """
    stage = fuzzy_result.get("stage", 0)
    stage_probs = fuzzy_result.get("stage_probs", {})
    confidence = fuzzy_result.get("confidence", 0.0)
    key_metrics = fuzzy_result.get("key_metrics", {})

    rs_latest = rs_info.get("latest_rs", 0.0)
    risk_adjusted_rs = rs_info.get("risk_adjusted_rs", 0.0)
    momentum = rs_info.get("momentum_score", 0.0)

    max_dd = risk_info.get("max_drawdown", 0.0)
    recent_dd = risk_info.get("recent_drawdown", 0.0)
    sharpe = risk_info.get("sharpe_ratio", 0.0)
    annual_return = risk_info.get("annual_return_pct", 0.0)

    regime = market_regime.get("regime", "unknown")
    regime_config = cfg.MARKET_REGIME_CONFIG.get(regime, cfg.MARKET_REGIME_CONFIG['unknown'])
    max_position = regime_config['max_position']

    rsi = key_metrics.get("rsi", 50.0)
    ret_4w = key_metrics.get("ret_4w", 0.0)
    ma_arrangement = key_metrics.get("ma_arrangement", 0)
    diff30 = key_metrics.get("diff30_pct", 0.0)

    # ===== 基础分 = 模糊概率加权 =====
    # 上升阶段权重高，筑底和顶部中等，下跌负权重
    stage_weight = {
        "accumulation": 40,
        "rising": 70,
        "top": 35,
        "falling": 15
    }
    base_score = sum(stage_probs.get(s, 0) * stage_weight[s] for s in stage_weight)

    # 置信度折扣：低置信度降低基础分
    base_score *= (0.5 + 0.5 * confidence)

    # ===== 增强因子（收益相关）=====
    enhancement_score = 0.0
    enhancement_details = {}

    # 1. 相对强度（权重提升，中短线关注）
    if rs_latest > 0.08:
        enhancement_score += 12
        enhancement_details["相对强度"] = "+12"
    elif rs_latest > 0.03:
        enhancement_score += 8
        enhancement_details["相对强度"] = "+8"
    elif rs_latest < -0.05:
        enhancement_score -= 10
        enhancement_details["相对强度"] = "-10"

    # 2. 动量得分（新增，中短线核心）
    if momentum > 0.3:
        enhancement_score += 10
        enhancement_details["动量"] = "+10"
    elif momentum > 0.1:
        enhancement_score += 5
        enhancement_details["动量"] = "+5"
    elif momentum < -0.2:
        enhancement_score -= 8
        enhancement_details["动量"] = "-8"

    # 3. 风险调整超额收益
    if risk_adjusted_rs > 0.5:
        enhancement_score += 6
        enhancement_details["风险调整收益"] = "+6"
    elif risk_adjusted_rs > 0.2:
        enhancement_score += 3
        enhancement_details["风险调整收益"] = "+3"

    # 4. 年化收益
    if annual_return > 15:
        enhancement_score += 5
        enhancement_details["年化收益"] = "+5"
    elif annual_return > 8:
        enhancement_score += 3
        enhancement_details["年化收益"] = "+3"

    # 5. 均线排列加分
    if ma_arrangement == 1:
        enhancement_score += 5
        enhancement_details["均线多头"] = "+5"
    elif ma_arrangement == -1:
        enhancement_score -= 5
        enhancement_details["均线空头"] = "-5"

    enhancement_score = max(-20, min(30, enhancement_score))

    # ===== 风险因子 =====
    risk_score = 0.0
    risk_details = {}

    # 1. 最大回撤（一次计算，不再重复调整）
    if max_dd < -30:
        risk_score -= 20
        risk_details["最大回撤"] = "-20"
    elif max_dd < -20:
        risk_score -= 15
        risk_details["最大回撤"] = "-15"
    elif max_dd < -10:
        risk_score -= 8
        risk_details["最大回撤"] = "-8"
    elif max_dd > -5:
        risk_score += 8
        risk_details["最大回撤"] = "+8"

    # 2. 夏普比率（一次计算）
    if sharpe < -0.5:
        risk_score -= 15
        risk_details["夏普比率"] = "-15"
    elif sharpe < 0:
        risk_score -= 10
        risk_details["夏普比率"] = "-10"
    elif sharpe > 1.5:
        risk_score += 12
        risk_details["夏普比率"] = "+12"
    elif sharpe > 0.8:
        risk_score += 6
        risk_details["夏普比率"] = "+6"

    # 3. RSI风险
    if rsi > 78:
        risk_score -= 10
        risk_details["RSI超买"] = "-10"
    elif rsi > 72:
        risk_score -= 5
        risk_details["RSI偏高"] = "-5"
    elif rsi < 25:
        risk_score += 5
        risk_details["RSI超卖机会"] = "+5"

    # 4. 短期涨幅过大风险
    if stage == 2 and ret_4w > 12:
        risk_score -= 10
        risk_details["短期涨幅过大"] = "-10"
    elif ret_4w > 8:
        risk_score -= 3
        risk_details["短期涨幅偏大"] = "-3"

    # 5. 近期回撤（新增，中短线敏感）
    if recent_dd < -5:
        risk_score -= 5
        risk_details["近期回撤大"] = "-5"

    risk_score = max(-40, min(20, risk_score))

    # ===== 综合评分 =====
    final_score = base_score + enhancement_score + risk_score
    final_score = max(0, min(100, final_score))

    # ===== DeepSeek 微调 =====
    if deepseek_result and deepseek_result.get("deepseek_opinion"):
        opinion = deepseek_result["deepseek_opinion"]
        if opinion.get("agree") == False:
            final_score -= 5  # 不赞同则扣5分
        elif opinion.get("agree") == True:
            final_score += 3  # 赞同加3分

    final_score = max(0, min(100, final_score))

    # ===== 仓位决策 =====
    # 信号强度（0-1）
    signal_strength = final_score / 100.0

    # 使用ATR动态仓位（如果已有position_info）
    if position_info:
        suggested_position = position_info['position_pct']
        stop_loss_pct = position_info['stop_loss_pct']
        target_pct = position_info['target_pct']
    else:
        # fallback: 简单仓位
        if final_score >= 75:
            suggested_position = min(60, max_position)
        elif final_score >= 60:
            suggested_position = min(40, max_position)
        elif final_score >= 45:
            suggested_position = min(25, max_position)
        elif final_score >= 30:
            suggested_position = min(10, max_position)
        else:
            suggested_position = 0
        stop_loss_pct = 5.0
        target_pct = 10.0

    # 市场状态调整仓位上限
    suggested_position = min(suggested_position, max_position)

    # ===== 建议操作 =====
    if final_score >= 75:
        if regime in ("strong_bull", "bull"):
            action = "重仓买入"
        else:
            action = "买入"
    elif final_score >= 60:
        action = "买入"
    elif final_score >= 45:
        action = "轻仓买入"
    elif final_score >= 30:
        action = "观望"
    elif final_score >= 20:
        action = "减仓"
    else:
        action = "卖出"

    # 市场状态修正：熊市不买入
    if regime in ("strong_bear", "bear") and action in ("买入", "重仓买入", "轻仓买入"):
        action = "观望"
        suggested_position = min(suggested_position, 15)

    # ===== 生成说明 =====
    stage_desc = {1: "筑底", 2: "上升", 3: "顶部", 4: "下跌"}
    note = fuzzy_result.get("reason", "")

    advice_note = (
        f"[{stage_desc.get(stage, '未知')}] "
        f"市场状态:{regime_config['description']} "
        f"评分:{final_score:.0f}"
    )

    if suggested_position > 0:
        advice_note += f" 建议仓位{suggested_position:.0f}%"
        if position_info:
            advice_note += f"(ATR止损{stop_loss_pct:.1f}% 目标{target_pct:.1f}%)"
    else:
        advice_note += " 建议空仓观望"

    # 构建评分详情
    score_details = {
        "基础分数": round(base_score, 1),
        "增强分数": enhancement_score,
        "风险分数": risk_score,
        "最终分数": final_score,
        "风险因子详情": risk_details,
        "增强因子详情": enhancement_details,
        "市场状态": regime,
        "市场描述": regime_config['description'],
        "仓位上限": max_position,
    }

    return {
        "建议操作": action,
        "建议仓位(%)": suggested_position,
        "建议说明": advice_note,
        "评分": round(final_score, 1),
        "建议置信度": round(confidence * 100, 1),
        "风险评分": risk_score,
        "增强评分": enhancement_score,
        "评分详情": score_details,
        "止损建议(%)": stop_loss_pct,
        "目标收益(%)": target_pct,
    }


# ===================== 交易策略生成 =====================

def generate_trading_strategy(
    advice_result: dict,
    fuzzy_result: dict,
    latest_data: dict,
    market_regime: dict
) -> dict:
    """
    生成具体交易策略

    中短线优化：
    - 分批更快（2批而非3批）
    - 止损更紧
    - 目标收益更合理
    """
    action = advice_result.get("建议操作", "观望")
    suggested_position = advice_result.get("建议仓位(%)", 0)
    latest_nav = latest_data.get("单位净值", 0)
    ma20 = latest_data.get("20周均线", 0)
    stop_loss_pct = advice_result.get("止损建议(%)", 5.0)
    target_pct = advice_result.get("目标收益(%)", 10.0)

    regime = market_regime.get("regime", "unknown")

    strategy = {
        "操作计划": "",
        "分批买入": [],
        "止损位": 0,
        "目标位": 0,
        "加仓条件": "",
        "减仓条件": "",
        "风险提示": "",
    }

    if action in ("买入", "重仓买入", "轻仓买入") and latest_nav > 0:
        # 中短线分批更快
        if suggested_position >= 50:
            strategy["分批买入"] = [
                {"批次": 1, "比例": 50, "条件": f"当前净值{latest_nav:.3f}附近"},
                {"批次": 2, "比例": 50, "条件": f"回调至{latest_nav * (1 - stop_loss_pct/200):.3f}或站稳20周均线"}
            ]
        else:
            strategy["分批买入"] = [
                {"批次": 1, "比例": 100, "条件": f"当前净值{latest_nav:.3f}附近一次性建仓"}
            ]

        # 动态止损
        strategy["止损位"] = round(latest_nav * (1 - stop_loss_pct / 100), 4)
        strategy["目标位"] = round(latest_nav * (1 + target_pct / 100), 4)

        strategy["加仓条件"] = (
            "1) 净值站稳20周均线上方 "
            "2) 相对强度持续改善 "
            "3) 市场状态不恶化"
        )
        strategy["减仓条件"] = (
            f"1) 净值跌破止损位{strategy['止损位']} "
            "2) 相对强度转负 "
            "3) 市场进入熊市状态"
        )

        regime_config = cfg.MARKET_REGIME_CONFIG.get(regime, {})
        strategy["操作计划"] = (
            f"建议{suggested_position:.0f}%仓位, "
            f"分{len(strategy['分批买入'])}批买入, "
            f"止损{stop_loss_pct:.1f}%, "
            f"目标{target_pct:.1f}%, "
            f"市场状态: {regime_config.get('description', '')}"
        )
        strategy["风险提示"] = "中短线交易请注意及时止盈止损，避免贪多"

    elif action in ("减仓", "卖出"):
        strategy["操作计划"] = f"建议减仓至{suggested_position:.0f}%或清仓离场"
        strategy["减仓条件"] = "1) 趋势明显转弱 2) 相对强度持续下降 3) 达到止损位"
        strategy["风险提示"] = "下跌趋势中，保护本金为第一要务，场外基金赎回需T+1到账"

    else:  # 观望
        strategy["操作计划"] = "暂时观望，等待更明确信号"
        strategy["加仓条件"] = f"1) 净值重回20周均线上方 2) 相对强度转正 3) 市场状态改善"
        strategy["风险提示"] = "方向不明时，耐心等待比盲目操作更安全"

    return strategy


# ===================== 回测函数 =====================

def backtest_fund_strategy(fund_code: str, benchmark_code: str = "sh000300", years: int = 5) -> dict:
    """
    策略历史回测（v2 使用新的评分系统）
    """
    fund_weekly = fetch_fund_weekly_nav(fund_code, years=years)
    index_weekly = fetch_index_weekly_close(benchmark_code, years=years)

    if len(fund_weekly) < 60 or len(index_weekly) < 60:
        return {"错误": f"历史数据不足（{len(fund_weekly)}周），需要至少60周"}

    aligned = fund_weekly.join(index_weekly[["ret"]], how="inner", rsuffix="_index")
    dates = aligned.index

    if len(dates) < 60:
        return {"错误": "对齐后数据不足，需要至少60周"}

    # 从第30周开始回测（需要足够的历史计算指标）
    start_idx = 30
    positions = []
    strat_ret = []
    regime_history = []
    signal_details = []

    for i in range(start_idx, len(dates) - 1):
        end_date = dates[i]
        slice_fund = fund_weekly.loc[:end_date]
        slice_index = index_weekly.loc[:end_date]

        # 使用v2分析链
        fuzzy = fuzzy_stage_judgment(slice_fund)
        rs = relative_strength_enhanced(slice_fund, slice_index)
        risk = risk_assessment(slice_fund)
        regime = detect_market_regime(slice_index)

        # ATR仓位
        current_price = slice_fund.iloc[-1]["close"]
        atr_val = slice_fund.iloc[-1].get("atr", current_price * 0.02) or current_price * 0.02
        signal_strength = fuzzy["confidence"]
        pos_info = calculate_position_size(
            signal_strength, atr_val, current_price,
            market_regime=regime["regime"],
            max_position=cfg.MARKET_REGIME_CONFIG.get(regime["regime"], {}).get("max_position", 50)
        )

        advice = generate_advice_v2(fuzzy, rs, risk, regime, pos_info)

        pos = advice.get("建议仓位(%)", 0) / 100.0

        next_date = dates[i + 1]
        r = fund_weekly.loc[next_date, "ret"]
        positions.append(pos)
        strat_ret.append(pos * r)
        regime_history.append(regime["regime"])

        # 收集ML训练用的特征数据
        km = fuzzy.get("key_metrics", {})
        total_weeks = len(dates)
        fwd_ret_8w = None
        if i + 9 < total_weeks:
            fwd_close = fund_weekly.loc[dates[i + 9], "close"]
            fwd_ret_8w = (fwd_close / current_price - 1) * 100

        signal_details.append({
            "diff30_pct": km.get("diff30_pct", 0),
            "diff10_pct": km.get("diff10_pct", 0),
            "ma30_slope": km.get("ma30_slope", 0),
            "ma30_r2": km.get("ma30_r2", 0),
            "rsi": km.get("rsi", 50),
            "ret_4w": km.get("ret_4w", 0),
            "ret_8w": km.get("ret_8w", 0),
            "ma_arrangement": km.get("ma_arrangement", 0),
            "vol_ratio": km.get("vol_ratio", 1.0),
            "8周": fwd_ret_8w,
        })

    strat_ret_series = pd.Series(strat_ret, index=dates[start_idx + 1:])

    if len(strat_ret_series) == 0:
        return {"错误": "回测窗口为空"}

    # 策略净值
    equity = (1 + strat_ret_series).cumprod()
    weeks = len(strat_ret_series)

    annual_ret = (equity.iloc[-1] ** (52 / weeks) - 1) * 100 if equity.iloc[-1] > 0 else -100
    roll_max = equity.cummax()
    drawdown = (equity / roll_max - 1) * 100
    max_drawdown = drawdown.min()
    win_rate = (strat_ret_series > 0).sum() / weeks * 100

    pos_series = pd.Series(positions, index=dates[start_idx:len(dates) - 1])
    trades = (pos_series.diff().abs() > 0.05).sum()

    # 买入持有对比
    buy_hold_ret = (fund_weekly.loc[dates[start_idx + 1]:dates[-1], "close"].iloc[-1] /
                    fund_weekly.loc[dates[start_idx], "close"] - 1) * 100

    summary = {
        "年化收益率(%)": round(float(annual_ret), 2),
        "最大回撤(%)": round(float(max_drawdown), 2),
        "胜率(%)": round(float(win_rate), 2),
        "交易次数": int(trades),
        "买入持有收益(%)": round(float(buy_hold_ret), 2),
        "超额收益(%)": round(float(annual_ret - buy_hold_ret / max(weeks/52, 0.5)), 2) if weeks > 26 else 0,
    }

    # 分市场状态的表现
    regime_df = pd.DataFrame({"regime": regime_history}, index=dates[start_idx:len(dates) - 1])
    if len(regime_df) > 0:
        pos_df = pd.DataFrame({"position": positions}, index=dates[start_idx:len(dates) - 1])
        ret_df = pd.DataFrame({"ret": strat_ret}, index=dates[start_idx + 1:len(dates)])
        combined = regime_df.join(pos_df, how="inner").join(ret_df, how="inner")
        for r in combined["regime"].unique():
            subset = combined[combined["regime"] == r]
            if len(subset) >= 4:
                regime_ret = (1 + subset["ret"]).prod() - 1
                summary[f"市场_{r}_收益"] = round(float(regime_ret * 100), 2)
                summary[f"市场_{r}_周数"] = len(subset)

    equity_df = pd.DataFrame({"策略净值": equity})
    signal_df = pd.DataFrame(signal_details)
    return {"基金代码": fund_code, "回测概要": summary, "净值曲线": equity_df, "信号明细": signal_df}


# ===================== 主分析函数 =====================

def analyze_fund_enhanced(fund_code: str, benchmark_code: str = "sh000300") -> dict:
    """
    增强版基金主分析函数（v2.0）

    整合模糊判定 + 市场状态 + ATR仓位 + DeepSeek + ML
    """
    # 1. 获取数据
    fund_info = fetch_fund_info(fund_code)
    fund_weekly = fetch_fund_weekly_nav(fund_code, years=3)
    index_weekly = fetch_index_weekly_close(benchmark_code, years=3)

    if len(fund_weekly) == 0:
        return {"错误": "无法获取基金净值数据", "基金代码": fund_code}
    if len(index_weekly) == 0:
        return {"错误": "无法获取基准指数数据", "基金代码": fund_code}

    # 2. 核心分析
    # 2a. 模糊阶段判定
    fuzzy_result = fuzzy_stage_judgment(fund_weekly)

    # 2b. 相对强度
    rs_result = relative_strength_enhanced(fund_weekly, index_weekly)

    # 2c. 风险评估
    risk_result = risk_assessment(fund_weekly)

    # 2d. 市场状态
    market_regime = detect_market_regime(index_weekly)

    # 2e. ATR动态仓位
    current_price = float(fund_weekly.iloc[-1]["close"])
    atr_val = float(fund_weekly.iloc[-1].get("atr", current_price * 0.02) or current_price * 0.02)
    signal_strength = fuzzy_result["confidence"]
    max_pos = cfg.MARKET_REGIME_CONFIG.get(market_regime["regime"], {}).get("max_position", 50)
    position_info = calculate_position_size(
        signal_strength, atr_val, current_price,
        market_regime=market_regime["regime"],
        max_position=max_pos
    )

    # 2f. ML分类器（如果可用）
    fund_cfg = cfg.ANALYSIS_CONFIG['fund']
    ml_result = None
    if fund_cfg.get('enable_ml', True) and cfg.ML_CONFIG.get('enabled', True):
        try:
            classifier = get_ml_classifier()
            if classifier.available:
                ml_pred = classifier.predict_stage(fuzzy_result.get("key_metrics", {}))
                fused = classifier.fuse_with_rules(
                    fuzzy_result, ml_pred,
                    ml_weight=cfg.ML_CONFIG.get('ml_weight', 0.3)
                )
                ml_result = fused
                # 如果ML置信度高，用融合结果替换模糊判定
                if fused.get("confidence", 0) > fuzzy_result.get("confidence", 0) * 1.2:
                    fuzzy_result = fused
                    print(f"  [ML] 融合后阶段: {fused['stage']} 置信度: {fused['confidence']:.2%}")
            else:
                # ML未训练，尝试从回测数据训练
                if len(fund_weekly) >= 80:
                    try:
                        bt = backtest_fund_strategy(fund_code, benchmark_code, years=3)
                        if "信号明细" in bt:
                            train_result = classifier.train_from_backtest(bt["信号明细"])
                            print(f"  [ML] 训练结果: {train_result.get('status', 'unknown')}")
                    except Exception as e:
                        print(f"  [ML] 训练跳过: {e}")
        except Exception as e:
            print(f"  [ML] 不可用: {e}")

    # 2g. DeepSeek验证
    deepseek_result = None
    ds_client = get_deepseek_client()
    if fund_cfg.get('enable_deepseek', True) and ds_client.available:
        try:
            deepseek_result = ds_client.validate_signal(
                fuzzy_result["stage"],
                fuzzy_result["confidence"],
                fuzzy_result.get("key_metrics", {}),
                market_regime=market_regime["regime"]
            )
            if deepseek_result.get("deepseek_opinion"):
                opinion = deepseek_result["deepseek_opinion"]
                print(f"  [DeepSeek] {'同意' if opinion.get('agree') else '不同意'}系统判断")
        except Exception as e:
            print(f"  [DeepSeek] 调用失败: {e}")

    # 2h. 投资建议
    advice_result = generate_advice_v2(
        fuzzy_result, rs_result, risk_result,
        market_regime, position_info,
        deepseek_result=deepseek_result
    )

    # 3. 交易策略
    recent_5w_data = []
    if len(fund_weekly) >= 5:
        for i in range(1, 6):
            if i <= len(fund_weekly):
                week_data = fund_weekly.iloc[-i]
                nav = week_data["close"]
                ma20 = week_data["ma20"]
                distance = ((nav - ma20) / ma20 * 100) if ma20 > 0 else 0
                recent_5w_data.append({
                    "日期": fund_weekly.index[-i].strftime("%Y-%m-%d"),
                    "单位净值": round(float(nav), 4),
                    "20周均线": round(float(ma20), 4),
                    "距离(%)": round(distance, 2)
                })
        recent_5w_data.reverse()

    latest_data = {
        "净值日期": fund_weekly.index[-1].strftime("%Y-%m-%d"),
        "单位净值": current_price,
        "30周均线": round(float(fund_weekly.iloc[-1]["ma30"]), 4),
        "20周均线": round(float(fund_weekly.iloc[-1]["ma20"]), 4),
        "10周均线": round(float(fund_weekly.iloc[-1]["ma10"]), 4),
        "最近5周数据": recent_5w_data,
        "最大回撤(%)": risk_result.get("max_drawdown", 0.0),
        "下行波动率(%)": risk_result.get("downside_vol", 0.0),
        "夏普比率": risk_result.get("sharpe_ratio", 0.0),
        "年化收益率(%)": risk_result.get("annual_return_pct", 0.0),
        "ATR(%)": position_info.get("atr_pct", 0),
        "市场状态": market_regime.get("regime", "unknown"),
        "市场描述": market_regime.get("description", ""),
    }

    trading_strategy = generate_trading_strategy(advice_result, fuzzy_result, latest_data, market_regime)

    # 4. 集成结果
    industry_tag = fund_info.get("投资类型") or fund_info.get("基金类型") or fund_info.get("投资风格") or ""

    result = {
        "基金基本信息": fund_info,
        "行业标签": industry_tag,
        "分析日期": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "最新数据": latest_data,
        "趋势分析": fuzzy_result,
        "相对强度分析": rs_result,
        "风险评估": risk_result,
        "市场状态分析": market_regime,
        "仓位分析": position_info,
        "投资建议": advice_result,
        "交易策略": trading_strategy,
        "ML分析": ml_result if ml_result else None,
        "DeepSeek分析": deepseek_result["deepseek_opinion"] if deepseek_result and deepseek_result.get("deepseek_opinion") else None,
        "历史周度数据": fund_weekly[["close", "ma10", "ma20", "ma30", "macd", "macd_signal", "macd_histogram"]],
    }

    return result
