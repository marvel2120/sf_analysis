"""
共享工具模块 - 提取 advisor_fund.py 和 advisor_stock.py 中的通用函数
并新增市场状态识别、ATR动态仓位等优化功能
"""
import functools
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===================== 基础统计函数 =====================

def compute_ma_slope(series: pd.Series, window: int = 10) -> tuple:
    """
    计算均线斜率 + R² 拟合优度
    return: (标准化斜率, R²)
    """
    s = series.dropna()
    if len(s) < window:
        return 0.0, 0.0
    x = np.arange(window)
    y = s.iloc[-window:].to_numpy()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    slope_normalized = slope / s.iloc[-window:].mean() * 100
    return float(slope_normalized), float(r_value ** 2)


def compute_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI序列（Wilder平滑法，比SMA更敏感）"""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    # Wilder平滑：使用EMA with alpha=1/period
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gains / avg_losses.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """计算当前RSI值"""
    rsi_series = compute_rsi_series(prices, period)
    val = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50.0
    return float(val) if not pd.isna(val) else 50.0


def compute_atr(weekly_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算平均真实波幅 ATR"""
    if not all(c in weekly_df.columns for c in ["high", "low"]):
        return pd.Series(index=weekly_df.index, dtype=float)
    high = weekly_df["high"]
    low = weekly_df["low"]
    close = weekly_df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> dict:
    """计算布林带"""
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return {"upper": middle + num_std * std, "lower": middle - num_std * std, "middle": middle}


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """计算MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


# ===================== 数据获取函数 =====================

@functools.lru_cache(maxsize=16)
def _fetch_index_raw(index_symbol: str, years: int) -> tuple:
    """缓存层：获取原始指数日线数据，返回可哈希的元组"""
    import akshare as ak
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")

    df = None
    # method 1: stock_zh_index_daily (新版无 start_date/end_date 参数)
    try:
        df = ak.stock_zh_index_daily(symbol=index_symbol)
    except Exception:
        pass

    if df is None or len(df) == 0:
        # method 2: index_zh_a_hist 支持日期参数
        try:
            code = index_symbol.replace("sh", "").replace("sz", "")
            df = ak.index_zh_a_hist(symbol=code, period="daily",
                                     start_date=start_date, end_date=end_date)
        except Exception:
            return ()

    if df is None or len(df) == 0:
        return ()

    # 按日期过滤（防止 method 1 返回过多数据）
    date_col = "日期" if "日期" in df.columns else "date" if "date" in df.columns else None
    if date_col and years > 0:
        df[date_col] = pd.to_datetime(df[date_col])
        cutoff = pd.Timestamp(end_date) - pd.DateOffset(years=years)
        df = df[df[date_col] >= cutoff]

    if len(df) == 0:
        return ()

    # 序列化 DataFrame 为元组以便缓存
    return (df.to_dict("records"), list(df.columns))


def fetch_index_weekly_close(index_symbol: str = "sh000300", years: int = 5) -> pd.DataFrame:
    """获取基准指数周线数据（带缓存）"""
    raw = _fetch_index_raw(index_symbol, years)
    if not raw:
        return pd.DataFrame()

    records, columns = raw
    df = pd.DataFrame.from_records(records)[columns]

    date_col = "日期" if "日期" in df.columns else "date" if "date" in df.columns else None
    price_col = "收盘" if "收盘" in df.columns else "close" if "close" in df.columns else None
    if not date_col or not price_col:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values("date").dropna(subset=[price_col, "date"]).set_index("date")

    weekly = df[[price_col]].resample("W-FRI").last().dropna()
    weekly = weekly.rename(columns={price_col: "close"})
    weekly["ret"] = weekly["close"].pct_change()
    weekly["log_ret"] = np.log(weekly["close"] / weekly["close"].shift(1))
    return weekly.dropna()


# ===================== 相对强度分析 =====================

def relative_strength_enhanced(asset_weekly: pd.DataFrame, index_weekly: pd.DataFrame,
                                lookback_periods: list = None) -> dict:
    """
    增强版相对强度：多周期超额收益 + 胜率 + 风险调整后超额收益
    """
    if lookback_periods is None:
        lookback_periods = [12, 26, 52]

    aligned = asset_weekly.join(index_weekly[["ret", "log_ret"]], how="inner", rsuffix="_index")
    if len(aligned) < min(lookback_periods):
        return {
            "rs_scores": {f"{p}周": 0.0 for p in lookback_periods},
            "win_rates": {f"{p}周": 0.0 for p in lookback_periods},
            "risk_adjusted_rs": 0.0, "latest_rs": 0.0,
            "momentum_score": 0.0
        }

    rs_scores = {}
    win_rates = {}
    for lookback in lookback_periods:
        if len(aligned) < lookback:
            rs_scores[f"{lookback}周"] = 0.0
            win_rates[f"{lookback}周"] = 0.0
            continue
        window = aligned.iloc[-lookback:]
        fund_cum = (1 + window["ret"]).prod() - 1
        index_cum = (1 + window["ret_index"]).prod() - 1
        rs_scores[f"{lookback}周"] = round(fund_cum - index_cum, 4)
        win_rates[f"{lookback}周"] = round((window["ret"] > window["ret_index"]).sum() / len(window), 3)

    # 风险调整后超额收益（信息比率）
    full = aligned.iloc[-52:] if len(aligned) >= 52 else aligned
    excess = full["log_ret"] - full["log_ret_index"]
    risk_adjusted = excess.mean() / excess.std() if excess.std() > 0 else 0.0

    # 动量得分：近4周超额收益权重更高（中短线优化）
    recent = aligned.iloc[-4:]
    if len(recent) >= 4:
        recent_excess = (recent["ret"] - recent["ret_index"]).mean()
        momentum = recent_excess / (recent["ret"].std() + 1e-8)
    else:
        momentum = 0.0

    return {
        "rs_scores": rs_scores,
        "win_rates": win_rates,
        "risk_adjusted_rs": round(risk_adjusted, 3),
        "latest_rs": rs_scores.get("12周", 0.0),
        "momentum_score": round(float(momentum), 3)
    }


# ===================== 风险评估 =====================

def risk_assessment(weekly_df: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    """
    风险评估：最大回撤、下行波动率、夏普比率、年化收益率
    """
    if len(weekly_df) < 20:
        return {"max_drawdown": 0.0, "downside_vol": 0.0, "sharpe_ratio": 0.0, "annual_return_pct": 0.0}

    # 最大回撤
    roll_max = weekly_df["close"].cummax()
    drawdown = (weekly_df["close"] / roll_max - 1) * 100
    max_drawdown = round(drawdown.min(), 2)

    # 年化收益和波动
    annual_ret = float(weekly_df["ret"].mean() * 52)
    annual_vol = float(weekly_df["ret"].std() * np.sqrt(52))
    sharpe = (annual_ret - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    # 下行波动率（仅负收益）
    downside = weekly_df["ret"][weekly_df["ret"] < 0]
    downside_vol = downside.std() * np.sqrt(52) * 100 if len(downside) > 0 else 0.0

    # 年化对数收益率
    annual_return = weekly_df["log_ret"].mean() * 52 * 100

    # 近期回撤（近4周最大回撤，用于短线风险判断）
    recent = weekly_df.iloc[-4:]
    recent_roll_max = recent["close"].cummax()
    recent_dd = (recent["close"] / recent_roll_max - 1) * 100
    recent_drawdown = round(recent_dd.min(), 2) if len(recent_dd) > 0 else 0.0

    return {
        "max_drawdown": max_drawdown,
        "recent_drawdown": recent_drawdown,
        "downside_vol": round(downside_vol, 2),
        "sharpe_ratio": round(sharpe, 3),
        "annual_return_pct": round(annual_return, 2),
        "annual_volatility": round(annual_vol * 100, 2)
    }


# ===================== 市场状态检测（新增）=========================

def detect_market_regime(index_weekly: pd.DataFrame) -> dict:
    """
    检测市场整体状态，用于过滤交易信号
    牛/熊/震荡市的判断会影响仓位上限和信号置信度阈值
    """
    result = {"regime": "unknown", "score": 0, "description": "数据不足"}

    if len(index_weekly) < 30:
        return result

    latest = index_weekly.iloc[-1]
    close = latest["close"]

    # 计算 20周/30周/50周 均线
    ma20 = index_weekly["close"].rolling(20).mean().iloc[-1]
    ma30 = index_weekly["close"].rolling(30).mean().iloc[-1]
    ma50 = index_weekly["close"].rolling(50).mean().iloc[-1] if len(index_weekly) >= 50 else ma30

    # 均线斜率和排列
    slope20, r2_20 = compute_ma_slope(index_weekly["close"], 10)
    slope50, r2_50 = 0.0, 0.0
    if len(index_weekly) >= 50:
        slope50, r2_50 = compute_ma_slope(index_weekly["close"].rolling(50).mean(), 10)

    # 价格相对均线位置
    diff_ma20 = (close / ma20 - 1) * 100
    diff_ma50 = (close / ma50 - 1) * 100

    # 均线排列
    bull_arrange = close > ma20 > ma30 > ma50
    bear_arrange = close < ma20 < ma30 < ma50

    # 计算市场宽度（近52周价格百分位）
    price_percentile = (close / index_weekly["close"].rolling(52).max()) * 100

    # 综合得分：+100（强牛） 到 -100（强熊）
    score = 0

    # 趋势得分（权重40%）
    if slope20 > 0.1 and r2_20 > 0.6:
        score += 20
    elif slope20 > 0.05:
        score += 10
    elif slope20 < -0.1 and r2_20 > 0.6:
        score -= 20
    elif slope20 < -0.05:
        score -= 10

    # 价格位置得分（权重30%）
    if bull_arrange:
        score += 20
    elif bear_arrange:
        score -= 20
    elif close > ma20 > ma30:
        score += 10
    elif close < ma20 < ma30:
        score -= 10

    # 均线偏离得分（权重30%）
    if diff_ma20 > 1:
        score += 10
    elif diff_ma20 < -3:
        score -= 10

    # 动量得分
    returns_4w = (close / index_weekly["close"].iloc[-4] - 1) * 100 if len(index_weekly) >= 4 else 0
    if returns_4w > 3:
        score += 10
    elif returns_4w < -3:
        score -= 10

    # 波动率（高波动 = 不稳定市场）
    vol_20w = index_weekly["ret"].rolling(20).std().iloc[-1] * np.sqrt(52)
    high_volatility = vol_20w > 0.25 if not pd.isna(vol_20w) else False

    # 分类
    if score >= 40:
        regime = "strong_bull"
        desc = "强势牛市，适合积极配置"
    elif score >= 15:
        regime = "bull"
        desc = "温和牛市，适合正常配置"
    elif score >= -10:
        regime = "sideways"
        desc = "震荡市场，适合波段操作"
    elif score >= -35:
        regime = "bear"
        desc = "熊市，适合谨慎防守"
    else:
        regime = "strong_bear"
        desc = "深度熊市，适合空仓观望"

    if high_volatility and regime in ("bull", "sideways"):
        desc += "，注意高波动风险"
        regime = f"volatile_{regime}"

    return {
        "regime": regime,
        "score": score,
        "description": desc,
        "slope_20w": round(slope20, 4),
        "ma_arrangement": "bull" if bull_arrange else ("bear" if bear_arrange else "mixed"),
        "price_percentile_52w": round(float(price_percentile.iloc[-1] if hasattr(price_percentile, 'iloc') else price_percentile), 1),
        "volatility_20w": round(float(vol_20w * 100), 2) if not pd.isna(vol_20w) else 0,
        "returns_4w": round(returns_4w, 2)
    }


# ===================== ATR动态仓位计算（新增）=========================

def calculate_position_size(signal_strength: float, atr_value: float, price: float,
                            market_regime: str = "sideways", max_position: float = 70.0) -> dict:
    """
    基于ATR和信号强度的动态仓位管理

    参数:
        signal_strength: 信号强度 0-1
        atr_value: ATR值
        price: 当前价格
        market_regime: 市场状态
        max_position: 最大允许仓位%

    返回:
        仓位比例、止损距离、目标距离
    """
    if price <= 0 or atr_value <= 0:
        return {"position_pct": 0, "stop_loss_pct": 5.0, "target_pct": 10.0}

    # ATR百分比
    atr_pct = (atr_value / price) * 100

    # 市场状态调整
    regime_multipliers = {
        "strong_bull": 1.2,
        "bull": 1.0,
        "sideways": 0.8,
        "volatile_sideways": 0.6,
        "bear": 0.5,
        "strong_bear": 0.2,
        "volatile_bull": 0.7,
        "unknown": 0.5
    }
    regime_mult = regime_multipliers.get(market_regime, 0.5)

    # 基础仓位 = 信号强度 * 市场乘数 * 最大仓位
    base_position = signal_strength * regime_mult * max_position
    # ATR调整：高波动标的降低仓位
    vol_adjustment = 1.0 / (1 + atr_pct / 5)
    final_position = base_position * vol_adjustment
    final_position = max(0, min(max_position, final_position))

    # 动态止损：ATR的2-3倍
    stop_loss_atr = 2.0 if market_regime in ("bull", "strong_bull") else 2.5
    stop_loss_pct = atr_pct * stop_loss_atr

    # 动态目标：ATR的3-5倍
    target_atr = 4.0 if market_regime in ("bull", "strong_bull") else 3.0
    target_pct = atr_pct * target_atr

    return {
        "position_pct": round(final_position, 1),
        "atr_pct": round(atr_pct, 2),
        "stop_loss_pct": round(max(stop_loss_pct, 3.0), 2),
        "target_pct": round(target_pct, 2),
        "regime_multiplier": round(regime_mult, 2),
        "vol_adjustment": round(vol_adjustment, 2)
    }


# ===================== 模糊阶段判定（新增，替代硬阈值）=========================

def fuzzy_stage_judgment(weekly_df: pd.DataFrame) -> dict:
    """
    模糊逻辑阶段判定，替代原来的硬阈值判断
    对每个阶段输出概率（0-1），取最高概率阶段

    优势：
    1. 不会把"不确定"强行归为阶段1
    2. 输出连续概率而非布尔值
    3. 对边缘情况更鲁棒
    """
    result = {
        "stage": 0, "confidence": 0.0, "reason": "数据不足",
        "stage_probs": {"accumulation": 0.0, "rising": 0.0, "top": 0.0, "falling": 0.0},
        "key_metrics": {}
    }

    if len(weekly_df) < 20:
        return result

    latest = weekly_df.iloc[-1]
    prev = weekly_df.iloc[-2]
    close = latest["close"]
    ma10 = latest.get("ma10", close)
    ma20 = latest.get("ma20", close)
    ma30 = latest.get("ma30", close)

    # 核心指标
    diff10 = (close / ma10 - 1) * 100 if ma10 > 0 else 0
    diff20 = (close / ma20 - 1) * 100 if ma20 > 0 else 0
    diff30 = (close / ma30 - 1) * 100 if ma30 > 0 else 0

    slope30, r2_30 = compute_ma_slope(weekly_df["ma30"], 10) if "ma30" in weekly_df.columns else (0, 0)
    slope20, r2_20 = compute_ma_slope(weekly_df["ma20"], 10) if "ma20" in weekly_df.columns else (0, 0)

    # 均线排列
    bull_count = sum([ma10 > ma20, ma20 > ma30, ma10 > ma30])
    bear_count = sum([ma10 < ma20, ma20 < ma30, ma10 < ma30])
    ma_arrangement = 1 if bull_count >= 2 else (-1 if bear_count >= 2 else 0)

    # 近期涨跌幅
    ret_4w = (close / weekly_df["close"].iloc[-4] - 1) * 100 if len(weekly_df) >= 4 else 0
    ret_2w = (close / weekly_df["close"].iloc[-2] - 1) * 100 if len(weekly_df) >= 2 else 0
    ret_8w = (close / weekly_df["close"].iloc[-8] - 1) * 100 if len(weekly_df) >= 8 else 0

    # RSI
    rsi = calculate_rsi(weekly_df["close"], 14)

    # 波动率
    current_vol = latest.get("vol", 0) if "vol" in latest else weekly_df["ret"].iloc[-1]
    avg_vol = weekly_df["vol"].rolling(20).mean().iloc[-1] if "vol" in weekly_df.columns else weekly_df["ret"].rolling(20).std().iloc[-1]
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    # === 计算各阶段概率 ===

    # 1. 上升阶段概率
    prob_rising = 0.0
    # 均线多头加分
    if ma_arrangement == 1:
        prob_rising += 0.25
    # 价格在均线上方加分
    prob_rising += max(0, min(0.25, diff30 / 15))
    # 均线斜率向上加分
    prob_rising += max(0, min(0.25, slope30 / 0.5))
    # 近期涨幅适度加分（但涨幅过大反而减分）
    if 0 < ret_4w < 12:
        prob_rising += 0.15
    elif ret_4w >= 12:
        prob_rising -= 0.1  # 涨太快了
    # RSI中性偏强
    if 40 < rsi < 70:
        prob_rising += 0.1
    elif rsi >= 70:
        prob_rising -= 0.1  # 超买
    prob_rising = max(0, min(1, prob_rising))

    # 2. 顶部阶段概率
    prob_top = 0.0
    # 涨幅过大
    if ret_4w > 10 or ret_8w > 18:
        prob_top += 0.25
    # RSI超买
    if rsi > 70:
        prob_top += 0.2
    elif rsi > 65:
        prob_top += 0.1
    # 价格远离均线
    if diff30 > 8:
        prob_top += 0.15
    elif diff30 > 5:
        prob_top += 0.1
    # 成交量异常放大
    if vol_ratio > 1.5:
        prob_top += 0.15
    # 均线斜率趋平
    if abs(slope30) < 0.05 and r2_30 < 0.5 and diff30 > 3:
        prob_top += 0.15
    prob_top = max(0, min(1, prob_top))

    # 3. 下跌阶段概率
    prob_falling = 0.0
    # 均线空头
    if ma_arrangement == -1:
        prob_falling += 0.25
    # 价格在均线下方
    prob_falling += max(0, min(0.25, -diff30 / 12))
    # 均线斜率向下
    prob_falling += max(0, min(0.25, -slope30 / 0.4))
    # 近期下跌
    if ret_4w < -3:
        prob_falling += 0.15
    elif ret_4w < -1:
        prob_falling += 0.08
    # RSI弱势
    if rsi < 35:
        prob_falling += 0.1
    prob_falling = max(0, min(1, prob_falling))

    # 4. 筑底阶段概率
    prob_accumulation = 0.0
    # 价格接近均线（小幅偏离）
    if -3 < diff30 < 1:
        prob_accumulation += 0.2
    # 缩量（波动率低）
    if vol_ratio < 0.8:
        prob_accumulation += 0.2
    # 均线走平
    if abs(slope30) < 0.08:
        prob_accumulation += 0.15
    # 价格止跌
    if -2 < ret_4w < 3:
        prob_accumulation += 0.15
    # RSI中性
    if 40 <= rsi <= 55:
        prob_accumulation += 0.1
    prob_accumulation = max(0, min(1, prob_accumulation))

    # 归一化（确保总和≈1）
    total = prob_accumulation + prob_rising + prob_top + prob_falling
    if total > 0:
        prob_accumulation /= total
        prob_rising /= total
        prob_top /= total
        prob_falling /= total

    probs = {
        "accumulation": round(prob_accumulation, 3),
        "rising": round(prob_rising, 3),
        "top": round(prob_top, 3),
        "falling": round(prob_falling, 3)
    }

    # 取最高概率阶段
    stage_names = ["accumulation", "rising", "top", "falling"]
    stage_map = {"accumulation": 1, "rising": 2, "top": 3, "falling": 4}
    max_stage = max(stage_names, key=lambda s: probs[s])
    stage = stage_map[max_stage]
    confidence = probs[max_stage]

    # 生成理由
    reasons = []
    if prob_rising > 0.35:
        reasons.append(f"上升概率{prob_rising:.0%}")
    if prob_top > 0.25:
        reasons.append(f"顶部风险{prob_top:.0%}")
    if prob_falling > 0.25:
        reasons.append(f"下跌风险{prob_falling:.0%}")
    if prob_accumulation > 0.3:
        reasons.append(f"筑底迹象{prob_accumulation:.0%}")
    if not reasons:
        reasons.append("信号不明确")

    reason_str = f"阶段{stage}：{'；'.join(reasons)}"
    if prob_rising > 0.5 and prob_rising > max(prob_top, prob_falling) * 1.5:
        reason_str += "，上升趋势明确"
    elif prob_top > 0.4 and prob_falling > 0.2:
        reason_str += "，注意顶部反转风险"
    elif prob_falling > 0.5:
        reason_str += "，下跌趋势确认"
    elif prob_accumulation > 0.4:
        reason_str += "，筑底特征明显"

    return {
        "stage": stage,
        "confidence": round(min(confidence, 1.0), 3),
        "reason": reason_str,
        "stage_probs": probs,
        "key_metrics": {
            "diff30_pct": round(diff30, 2),
            "diff10_pct": round(diff10, 2),
            "ma30_slope": round(slope30, 4),
            "ma30_r2": round(r2_30, 3),
            "ma_arrangement": ma_arrangement,
            "rsi": round(rsi, 1),
            "ret_4w": round(ret_4w, 2),
            "ret_8w": round(ret_8w, 2),
            "vol_ratio": round(vol_ratio, 2),
        }
    }
