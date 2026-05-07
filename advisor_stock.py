from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import akshare as ak
import warnings

# 从共享模块导入通用工具函数（消除重复代码）
from market_utils import (
    compute_ma_slope, compute_rsi_series, calculate_rsi,
    compute_atr, compute_bollinger_bands, compute_macd,
    fetch_index_weekly_close, relative_strength_enhanced,
    risk_assessment as risk_assessment_market,  # 保留原名用于兼容
    detect_market_regime
)
from deepseek_integration import DeepSeekClient
from ml_classifier import MLStageClassifier
import config as cfg
warnings.filterwarnings('ignore')

# ===================== 全局初始化 =====================
_deepseek_client = None
_ml_classifier = None


def get_ml_classifier() -> MLStageClassifier:
    global _ml_classifier
    if _ml_classifier is None:
        _ml_classifier = MLStageClassifier(
            min_samples=cfg.ML_CONFIG.get('min_samples', 80),
            model_type=cfg.ML_CONFIG.get('model_type', 'xgb')
        )
    return _ml_classifier


def get_deepseek_client() -> DeepSeekClient:
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = DeepSeekClient(
            api_key=cfg.DEEPSEEK_CONFIG.get('api_key', ''),
            model=cfg.DEEPSEEK_CONFIG.get('model', 'deepseek-chat'),
            timeout=cfg.DEEPSEEK_CONFIG.get('timeout', 15)
        )
    return _deepseek_client


# ===================== 基础数据获取 =====================

def fetch_stock_info(stock_code: str) -> dict:
    if ak is None:
        return {"股票代码": stock_code}
    info = {}
    df = None
    try:
        df = ak.stock_individual_info_em(symbol=stock_code)
    except Exception:
        df = None
    if df is not None and len(df) > 0:
        for _, row in df.iterrows():
            k = str(row.get("item", "")).strip()
            v = str(row.get("value", "")).strip()
            if k:
                info[k] = v
    if not info:
        try:
            spot = ak.stock_zh_a_spot_em()
            spot["代码"] = spot["代码"].astype(str)
            row = spot[spot["代码"] == stock_code]
            if len(row) > 0:
                r = row.iloc[0]
                name = str(r.get("名称", ""))
                info = {
                    "股票代码": stock_code,
                    "股票名称": name,
                    "股票简称": name,
                    "最新价": str(r.get("最新价", "")),
                    "涨跌幅": str(r.get("涨跌幅", "")),
                }
        except Exception:
            info = {"股票代码": stock_code}
    if not info.get("股票名称") and not info.get("股票简称"):
        try:
            hist = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date="20250101", end_date="20250110", adjust="qfq")
            if hist is not None and len(hist) > 0:
                name = str(hist.iloc[0].get("股票名称", ""))
                if name:
                    info["股票名称"] = name
                    info["股票简称"] = name
        except Exception:
            pass
    if "股票简称" not in info and "股票名称" in info:
        info["股票简称"] = info["股票名称"]
    if "股票名称" not in info and "股票简称" in info:
        info["股票名称"] = info["股票简称"]
    info["股票代码"] = stock_code
    return info


def fetch_stock_weekly(stock_code: str, years: int = 5) -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame()
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
    df = None
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    except Exception:
        df = None
    if df is None or len(df) == 0:
        try:
            pref = "sh" if stock_code.startswith("6") else "sz"
            try:
                df = ak.stock_zh_a_daily(symbol=f"{pref}{stock_code}", start_date=start_date, end_date=end_date, adjust="qfq")
            except Exception:
                df = ak.stock_zh_a_daily(symbol=f"{pref}{stock_code}", start_date=start_date, end_date=end_date)
        except Exception:
            df = None
    if df is None or len(df) == 0:
        return pd.DataFrame()

    col_map = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude", "换手率": "turnover"
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").set_index("date")

    need_cols = ["close"]
    if "open" in df.columns:
        need_cols.append("open")
    if "high" in df.columns:
        need_cols.append("high")
    if "low" in df.columns:
        need_cols.append("low")
    if "volume" in df.columns:
        need_cols.append("volume")

    weekly_dict = {}
    for col in need_cols:
        if col == "volume":
            weekly_dict[col] = df[col].resample("W-FRI").sum()
        elif col in ("high", "max"):
            weekly_dict[col] = df[col].resample("W-FRI").max()
        elif col in ("low", "min"):
            weekly_dict[col] = df[col].resample("W-FRI").min()
        elif col == "open":
            weekly_dict[col] = df[col].resample("W-FRI").first()
        else:
            weekly_dict[col] = df[col].resample("W-FRI").last()
    weekly = pd.DataFrame(weekly_dict).dropna(subset=["close"])

    weekly["ma10"] = weekly["close"].rolling(10).mean()
    weekly["ma20"] = weekly["close"].rolling(20).mean()
    weekly["ma30"] = weekly["close"].rolling(30).mean()
    weekly["ret"] = weekly["close"].pct_change()
    weekly["log_ret"] = np.log(weekly["close"] / weekly["close"].shift(1))

    if "volume" in weekly.columns:
        weekly["vol_ma5"] = weekly["volume"].rolling(5).mean()
        weekly["vol_ma20"] = weekly["volume"].rolling(20).mean()

    if all(c in weekly.columns for c in ["high", "low"]):
        weekly["atr"] = compute_atr(weekly, 14)
    weekly["support"] = weekly["close"].rolling(20).min()
    weekly["resistance"] = weekly["close"].rolling(20).max()

    bb = compute_bollinger_bands(weekly["close"], 20, 2)
    weekly["bb_upper"] = bb["upper"]
    weekly["bb_lower"] = bb["lower"]

    weekly["volatility"] = weekly["ret"].rolling(10).std() * np.sqrt(52)

    weekly["rsi"] = compute_rsi_series(weekly["close"], 14)

    macd_data = compute_macd(weekly["close"])
    weekly["macd"] = macd_data["macd"]
    weekly["macd_signal"] = macd_data["signal"]
    weekly["macd_histogram"] = macd_data["histogram"]

    weekly = weekly.dropna(subset=["ma30", "ma10", "ma20"])
    return weekly


# ===================== 核心分析 =====================

def judge_stage_enhanced(weekly_df: pd.DataFrame) -> dict:
    if len(weekly_df) < 30:
        return {"stage": 0, "confidence": 0.0, "reason": "数据不足"}

    latest = weekly_df.iloc[-1]
    prev = weekly_df.iloc[-2]
    ma10, ma20, ma30 = latest["ma10"], latest["ma20"], latest["ma30"]
    close = latest["close"]
    rsi = latest.get("rsi", 50.0)
    vol_ratio = 1.0
    if "volume" in weekly_df.columns and "vol_ma20" in weekly_df.columns:
        vol_ratio = latest["volume"] / latest["vol_ma20"] if latest["vol_ma20"] > 0 else 1.0

    diff10 = (close / ma10 - 1) * 100
    diff20 = (close / ma20 - 1) * 100
    diff30 = (close / ma30 - 1) * 100

    slope30, r2_30 = compute_ma_slope(weekly_df["ma30"], 10)
    slope20, r2_20 = compute_ma_slope(weekly_df["ma20"], 10)

    ma_arrangement = 1 if ma10 > ma20 > ma30 else (-1 if ma10 < ma20 < ma30 else 0)

    price_change_4w = (close / weekly_df["close"].iloc[-4] - 1) * 100 if len(weekly_df) >= 4 else 0
    price_change_8w = (close / weekly_df["close"].iloc[-8] - 1) * 100 if len(weekly_df) >= 8 else 0

    stage = 0
    confidence = 0.0
    reason = ""

    if (diff30 < -3 and slope30 < -0.1 and r2_30 > 0.6
            and ma_arrangement == -1 and close < prev["close"]):
        stage = 4
        confidence = min(0.9, abs(slope30) + abs(diff30) / 10)
        reason = f"低于30周均线{diff30:.1f}%，斜率{slope30:.3f}(R²={r2_30:.2f})，空头排列，下跌趋势确认"

    elif ((diff30 > 2 and slope30 < 0.05 and r2_30 < 0.6)
          or (price_change_4w > 10 and rsi > 70)
          or (diff30 > 8 and vol_ratio > 1.5)):
        stage = 3
        confidence = min(0.85, abs(diff30) / 10 + (rsi / 100) + (vol_ratio / 3))
        reasons = []
        if diff30 > 2 and slope30 < 0.05:
            reasons.append(f"均线趋平(斜率{slope30:.3f})")
        if price_change_4w > 10:
            reasons.append(f"4周涨幅{price_change_4w:.1f}%")
        if rsi > 70:
            reasons.append(f"RSI超买({rsi:.0f})")
        if vol_ratio > 1.5:
            reasons.append(f"成交量异常({vol_ratio:.1f}倍)")
        reason = f"顶部风险信号: {'; '.join(reasons)}"

    elif (diff30 > 1 and diff30 < 8 and slope30 > 0.1 and r2_30 > 0.6
          and ma_arrangement == 1 and close > prev["close"]
          and price_change_4w < 12 and rsi < 75):
        stage = 2
        confidence = min(0.9, slope30 + diff30 / 10 + r2_30)
        reason = (f"多头排列，高于30周均线{diff30:.1f}%，"
                  f"斜率{slope30:.3f}(R²={r2_30:.2f})，RSI={rsi:.0f}，上升趋势健康")

    elif (diff30 < 0 and slope30 > -0.1 and r2_30 < 0.5
          and vol_ratio < 0.9):
        stage = 1
        confidence = min(0.8, (1 - abs(diff30) / 5) + (1 - vol_ratio) + r2_30)
        reason = f"接近均线(偏离{diff30:.1f}%)，量能萎缩({vol_ratio:.1f}倍)，筑底特征"

    else:
        stage = 1
        confidence = 0.4
        reason = "无明显趋势信号，归为筑底观察期"

    return {
        "stage": stage,
        "confidence": round(min(confidence, 1.0), 3),
        "reason": reason,
        "key_metrics": {
            "ma10": float(ma10), "ma20": float(ma20), "ma30": float(ma30),
            "diff30_pct": round(diff30, 2),
            "diff10_pct": round(diff10, 2),
            "ma30_slope": round(slope30, 4),
            "ma30_r2": round(r2_30, 3),
            "ma_arrangement": ma_arrangement,
            "rsi": round(rsi, 1),
            "price_change_4w": round(price_change_4w, 2),
            "ret_4w": round(price_change_4w, 2),
            "ret_8w": round(price_change_8w, 2),
            "vol_ratio": round(vol_ratio, 2),
        }
    }


def relative_strength_enhanced(stock_weekly: pd.DataFrame, index_weekly: pd.DataFrame,
                                lookback_periods: list = None) -> dict:
    if lookback_periods is None:
        lookback_periods = [12, 26, 52]
    aligned = stock_weekly.join(index_weekly[["ret", "log_ret"]], how="inner", rsuffix="_index")
    if len(aligned) < min(lookback_periods):
        return {
            "rs_scores": {f"{p}周": 0.0 for p in lookback_periods},
            "win_rates": {f"{p}周": 0.0 for p in lookback_periods},
            "risk_adjusted_rs": 0.0, "latest_rs": 0.0
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

    full = aligned.iloc[-52:] if len(aligned) >= 52 else aligned
    excess = full["log_ret"] - full["log_ret_index"]
    risk_adjusted = excess.mean() / excess.std() if excess.std() > 0 else 0.0

    return {
        "rs_scores": rs_scores,
        "win_rates": win_rates,
        "risk_adjusted_rs": round(risk_adjusted, 3),
        "latest_rs": rs_scores.get("12周", 0.0),
    }


def risk_assessment_stock(weekly_df: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    if len(weekly_df) < 20:
        return {"max_drawdown": 0.0, "annual_volatility": 0.0, "downside_vol": 0.0, "atr": 0.0, "sharpe_ratio": 0.0}

    roll_max = weekly_df["close"].cummax()
    drawdown = (weekly_df["close"] / roll_max - 1) * 100
    max_drawdown = round(drawdown.min(), 2)

    annual_ret = float(weekly_df["ret"].mean() * 52)
    annual_vol = float(weekly_df["ret"].std() * np.sqrt(52))
    sharpe = (annual_ret - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    downside = weekly_df["ret"][weekly_df["ret"] < 0]
    downside_vol = round(downside.std() * np.sqrt(52) * 100, 2) if len(downside) > 0 else 0.0

    atr_val = round(float(weekly_df["atr"].iloc[-1]), 4) if "atr" in weekly_df.columns else 0.0

    return {
        "max_drawdown": max_drawdown,
        "annual_volatility": round(annual_vol * 100, 2),
        "downside_vol": downside_vol,
        "atr": atr_val,
        "sharpe_ratio": round(sharpe, 3),
    }


def volume_analysis(weekly_df: pd.DataFrame) -> dict:
    result = {"volume_ok": True, "volume_trend": "normal", "divergence": 0}
    if "volume" not in weekly_df.columns:
        return result

    latest_vol = weekly_df.iloc[-1]["volume"]
    vol_ma20 = weekly_df.iloc[-1]["vol_ma20"]
    vol_ma5 = weekly_df.iloc[-1]["vol_ma5"]

    vol_ratio = latest_vol / vol_ma20 if vol_ma20 > 0 else 1.0

    vol_percentile = (weekly_df["volume"].rank(pct=True).iloc[-1]) * 100

    result["vol_ratio"] = round(vol_ratio, 2)
    result["vol_percentile"] = round(vol_percentile, 1)
    result["volume_ok"] = vol_ratio > 1.3

    if vol_ma5 > vol_ma20 * 1.2:
        result["volume_trend"] = "放量"
    elif vol_ma5 < vol_ma20 * 0.8:
        result["volume_trend"] = "缩量"
    else:
        result["volume_trend"] = "正常"

    close = weekly_df["close"]
    vol_series = weekly_df["volume"]
    recent_close = close.iloc[-5:]
    recent_vol = vol_series.iloc[-5:]
    if len(recent_close) >= 5:
        close_up = all(recent_close.iloc[i] > recent_close.iloc[i - 1] for i in range(1, 5))
        vol_down = all(recent_vol.iloc[i] < recent_vol.iloc[i - 1] for i in range(1, 5))
        if close_up and vol_down:
            result["divergence"] = -1
        close_down = all(recent_close.iloc[i] < recent_close.iloc[i - 1] for i in range(1, 5))
        vol_up = all(recent_vol.iloc[i] > recent_vol.iloc[i - 1] for i in range(1, 5))
        if close_down and vol_up:
            result["divergence"] = 1

    return result


def detect_breakout(weekly_df: pd.DataFrame, lookback: int = 12, threshold: float = 0.01) -> bool:
    if len(weekly_df) < lookback + 1:
        return False
    recent = weekly_df.iloc[-(lookback + 1):-1]
    upper = recent["close"].max()
    latest = weekly_df.iloc[-1]
    return bool(latest["close"] > upper * (1 + threshold))


def detect_breakdown(weekly_df: pd.DataFrame, threshold: float = 0.01) -> bool:
    if len(weekly_df) == 0 or "support" not in weekly_df.columns:
        return False
    latest = weekly_df.iloc[-1]
    support = latest.get("support")
    close = latest.get("close")
    if pd.isna(support) or pd.isna(close) or support == 0:
        return False
    return bool(close < support * (1 - threshold))


# ===================== 投资建议生成 =====================

def generate_advice_enhanced(
    stage_info: dict,
    rs_info: dict,
    risk_info: dict,
    vol_info: dict,
    breakout: bool,
    breakdown: bool,
    weekly_df: pd.DataFrame
) -> dict:
    stage = stage_info["stage"]
    confidence = stage_info["confidence"]
    km = stage_info.get("key_metrics", {})

    rs_scores = rs_info.get("rs_scores", {})
    rs_latest = rs_info.get("latest_rs", 0.0)
    win_rate = rs_info.get("win_rates", {}).get("12周", 0.0)
    risk_adjusted_rs = rs_info.get("risk_adjusted_rs", 0.0)

    max_dd = risk_info.get("max_drawdown", 0.0)
    volatility = risk_info.get("annual_volatility", 0.0)
    atr_val = risk_info.get("atr", 0.0)

    rsi = km.get("rsi", 50.0)
    diff30 = km.get("diff30_pct", 0.0)
    price_change_4w = km.get("price_change_4w", 0.0)
    vol_ratio = km.get("vol_ratio", 1.0)
    vol_ok = vol_info.get("volume_ok", True)
    vol_divergence = vol_info.get("divergence", 0)

    base_scores = {0: 30, 1: 45, 2: 65, 3: 35, 4: 15}
    base_score = base_scores.get(stage, 30)

    enhancement_score = 0
    enhancement_factors = []

    if rs_latest > 0.08:
        enhancement_score += 10
        enhancement_factors.append(f"相对强度优秀(+10)")
    elif rs_latest > 0.03:
        enhancement_score += 6
        enhancement_factors.append(f"相对强度良好(+6)")
    elif rs_latest > 0.005:
        enhancement_score += 3
        enhancement_factors.append(f"相对强度偏正(+3)")
    elif rs_latest < -0.05:
        enhancement_score -= 8
        enhancement_factors.append(f"相对强度较弱(-8)")

    if win_rate > 0.6:
        enhancement_score += 5
        enhancement_factors.append(f"胜率较高(+5)")
    elif win_rate > 0.5:
        enhancement_score += 2
        enhancement_factors.append(f"胜率中性(+2)")
    elif win_rate < 0.4:
        enhancement_score -= 3
        enhancement_factors.append(f"胜率偏低(-3)")

    if risk_adjusted_rs > 0.5:
        enhancement_score += 8
        enhancement_factors.append(f"风险调整收益优秀(+8)")
    elif risk_adjusted_rs > 0.2:
        enhancement_score += 4
        enhancement_factors.append(f"风险调整收益良好(+4)")

    if breakout and vol_ok:
        enhancement_score += 10
        enhancement_factors.append(f"放量突破(+10)")
    elif breakout and not vol_ok:
        enhancement_score += 3
        enhancement_factors.append(f"缩量突破(+3)")

    risk_score = 0
    risk_factors = []

    if max_dd < -30:
        risk_score -= 25
        risk_factors.append(f"最大回撤极大(-25)")
    elif max_dd < -20:
        risk_score -= 18
        risk_factors.append(f"最大回撤过大(-18)")
    elif max_dd < -10:
        risk_score -= 8
        risk_factors.append(f"回撤较大(-8)")
    elif max_dd > -5:
        risk_score += 10
        risk_factors.append(f"回撤控制优秀(+10)")

    if rsi > 75:
        risk_score -= 10
        risk_factors.append(f"RSI超买(-10)")
    elif rsi > 70:
        risk_score -= 5
        risk_factors.append(f"RSI偏高(-5)")
    elif rsi < 25:
        risk_score += 5
        risk_factors.append(f"RSI超卖(+5)")

    if abs(diff30) > 10:
        risk_score -= 6
        risk_factors.append(f"远离均线(-6)")
    elif abs(diff30) > 6:
        risk_score -= 3

    if vol_divergence == -1:
        risk_score -= 8
        risk_factors.append(f"量价顶背离(-8)")
    elif vol_divergence == 1:
        risk_score += 5
        risk_factors.append(f"量价底背离(+5)")

    if breakdown:
        risk_score -= 15
        risk_factors.append(f"跌破支撑(-15)")

    if volatility > 40:
        risk_score -= 5
        risk_factors.append(f"高波动(-5)")

    final_score = base_score + enhancement_score + risk_score
    final_score = max(0, min(100, final_score))

    if final_score >= 80:
        action = "买入"
        suggested_position = 60
    elif final_score >= 65:
        action = "买入"
        suggested_position = 45
    elif final_score >= 50:
        action = "谨慎买入"
        suggested_position = 30
    elif final_score >= 35:
        action = "观望"
        suggested_position = 15
    elif final_score >= 20:
        action = "减仓"
        suggested_position = 5
    else:
        action = "清仓"
        suggested_position = 0

    note_parts = []
    stage_names = {0: "未知", 1: "筑底期", 2: "上升期", 3: "顶部期", 4: "下跌期"}
    stage_name = stage_names.get(stage, "未知")
    note_parts.append(f"第{stage}阶段（{stage_name}）")

    if stage == 2:
        if final_score >= 65:
            if enhancement_factors:
                note_parts.append("趋势健康，" + "；".join(enhancement_factors[:2]))
            note_parts.append("建议分批建仓，回踩30周均线补仓")
        else:
            note_parts.append("上升趋势但信号偏弱，建议等待更明确信号")
    elif stage == 4:
        note_parts.append("下跌趋势，注意风险控制")
        if breakdown:
            note_parts.append("已跌破支撑，建议立即止损")
        else:
            note_parts.append("建议严格设置止损")
    elif stage == 3:
        note_parts.append("顶部区域，" + ("建议分批减仓" if final_score < 40 else "谨慎持有，设置止盈"))
    elif stage == 1:
        note_parts.append("筑底阶段，等待趋势明确")

    if risk_factors:
        note_parts.append("风险：" + "；".join(risk_factors[:2]))

    return {
        "建议": action,
        "说明": "。".join(note_parts),
        "评分": final_score,
        "建议仓位(%)": suggested_position,
        "评分详情": {
            "基础分数": base_score,
            "增强分数": enhancement_score,
            "风险分数": risk_score,
            "最终分数": final_score,
            "增强因子": enhancement_factors,
            "风险因子": risk_factors,
        },
        "关键参数": {
            "阶段": stage,
            "阶段置信度": confidence,
            "相对强度": round(rs_latest, 4),
            "多周期相对强度": rs_scores,
            "胜率": win_rate,
            "风险调整超额": risk_adjusted_rs,
            "是否突破": breakout,
            "量能状态": vol_ok,
            "量价比率": vol_ratio,
            "是否跌破支撑": breakdown,
            "RSI": rsi,
            "均线偏离%": diff30,
            "4周涨幅%": price_change_4w,
            "最大回撤%": max_dd,
            "年化波动率%": volatility,
        }
    }


def generate_trading_strategy(
    advice: dict,
    weekly_df: pd.DataFrame,
    support: float,
    resistance: float
) -> dict:
    action = advice["建议"]
    suggested_position = advice.get("建议仓位(%)", 0)
    latest_close = float(weekly_df.iloc[-1]["close"])
    atr_val = float(weekly_df["atr"].iloc[-1]) if "atr" in weekly_df.columns else latest_close * 0.05

    strategy = {
        "建议仓位(%)": suggested_position,
        "当前价": round(latest_close, 2),
        "分批买入": [],
        "止损位": round(latest_close - atr_val * 2, 2),
        "目标位": round(latest_close + atr_val * 3, 2),
    }

    if action in ("买入", "谨慎买入") and suggested_position > 0:
        batches = []
        if suggested_position >= 40:
            batches.append({"批次": 1, "比例": 50, "条件": f"当前价{latest_close:.2f}附近"})
            batches.append({"批次": 2, "比例": 30, "条件": f"回踩30周均线{weekly_df.iloc[-1]['ma30']:.2f}附近"})
            batches.append({"批次": 3, "比例": 20, "条件": f"放量突破阻力位{resistance:.2f}加仓"})
        elif suggested_position >= 20:
            batches.append({"批次": 1, "比例": 60, "条件": f"当前价{latest_close:.2f}附近"})
            batches.append({"批次": 2, "比例": 40, "条件": f"回踩30周均线{weekly_df.iloc[-1]['ma30']:.2f}加仓"})
        else:
            batches.append({"批次": 1, "比例": 100, "条件": f"当前价{latest_close:.2f}附近试探性建仓"})
        strategy["分批买入"] = batches

    ma30 = weekly_df.iloc[-1]["ma30"]
    strategy["止损位"] = round(min(support * 0.98, latest_close - atr_val * 2), 2)
    strategy["目标位"] = round(max(resistance * 1.05, latest_close + atr_val * 3), 2)

    if action == "买入":
        strategy["加仓条件"] = "RS持续为正 + 回踩MA30不破 + 放量突破前高"
        strategy["减仓条件"] = f"跌破止损位{strategy['止损位']} 或 RSI>80 或 相对强度转负"
    elif action in ("减仓", "清仓"):
        strategy["减仓条件"] = "建议按计划减仓，反弹至均线附近是较好的减仓时机"
    else:
        strategy["加仓条件"] = "等待趋势明确（突破阻力位+RS转正+量能放大）"
        strategy["减仓条件"] = f"跌破支撑位{support:.2f}应考虑止损"

    return strategy


# ===================== 主分析函数 =====================

def analyze_stock(stock_code: str) -> dict:
    try:
        info = fetch_stock_info(stock_code)
        weekly = fetch_stock_weekly(stock_code)

        if weekly is None or len(weekly) < 40:
            return {
                "股票代码": stock_code,
                "股票名称": info.get("股票简称") or info.get("股票名称", ""),
                "分析日期": datetime.today().strftime("%Y-%m-%d"),
                "最新收盘": np.nan, "30周均值": np.nan,
                "阶段": np.nan, "相对强度": np.nan,
                "是否突破": False, "量能是否放大": False,
                "支撑位": np.nan, "阻力位": np.nan, "止损建议": np.nan,
                "投资建议": "", "投资说明": "历史数据不足，暂无法进行有效分析",
                "投资评分": np.nan, "错误信息": "历史数据不足"
            }

        index_weekly = fetch_index_weekly_close("sh000300")

        stage_info = judge_stage_enhanced(weekly)
        rs_info = relative_strength_enhanced(weekly, index_weekly) if len(index_weekly) > 0 else None
        risk_info = risk_assessment_stock(weekly)
        vol_info = volume_analysis(weekly)

        breakout_up = detect_breakout(weekly, 12, 0.01)
        breakdown = detect_breakdown(weekly, 0.01)

        if rs_info is None:
            rs_info = {
                "rs_scores": {}, "win_rates": {},
                "risk_adjusted_rs": 0.0, "latest_rs": 0.0
            }

        advice = generate_advice_enhanced(
            stage_info, rs_info, risk_info, vol_info,
            breakout_up, breakdown, weekly
        )

        # Market regime
        market_regime = detect_market_regime(index_weekly) if len(index_weekly) > 0 else {"regime": "unknown", "score": 0, "description": "数据不足"}

        # ML 分类器预测（如果已训练）
        ml_fused = None
        stock_cfg = cfg.ANALYSIS_CONFIG.get('fund', {})
        if stock_cfg.get('enable_ml', True) and cfg.ML_CONFIG.get('enabled', True):
            try:
                classifier = get_ml_classifier()
                if classifier.available:
                    ml_pred = classifier.predict_stage(stage_info.get("key_metrics", {}))
                    fused = classifier.fuse_with_rules(
                        stage_info, ml_pred,
                        ml_weight=cfg.ML_CONFIG.get('ml_weight', 0.3)
                    )
                    if fused.get("confidence", 0) > stage_info.get("confidence", 0) * 1.2:
                        print(f"  [ML] 股票{stock_code}: 融合后阶段 {fused['stage']} "
                              f"置信度 {fused['confidence']:.2%}")
                    ml_fused = fused
            except Exception as e:
                print(f"  [ML] 预测失败: {e}")

        # DeepSeek AI 交叉验证
        deepseek_result = None
        stock_cfg = cfg.ANALYSIS_CONFIG.get('fund', {})
        ds_client = get_deepseek_client()
        if stock_cfg.get('enable_deepseek', True) and ds_client.available:
            try:
                km = stage_info.get("key_metrics", {})
                deepseek_result = ds_client.validate_signal(
                    stage_info["stage"],
                    stage_info["confidence"],
                    km,
                    market_regime=market_regime["regime"]
                )
                if deepseek_result and deepseek_result.get("deepseek_opinion"):
                    opinion = deepseek_result["deepseek_opinion"]
                    print(f"  [DeepSeek] 股票{stock_code}: {'同意' if opinion.get('agree') else '不同意'}系统判断")
            except Exception as e:
                print(f"  [DeepSeek] 调用失败: {e}")

        latest_close = float(weekly.iloc[-1]["close"])
        latest_ma30 = float(weekly.iloc[-1]["ma30"])
        support = float(weekly.iloc[-1]["support"])
        resistance = float(weekly.iloc[-1]["resistance"])
        rsi_val = float(weekly.iloc[-1].get("rsi", 50))

        integer_price = np.floor(support)
        stop_loss = (integer_price - 0.05) if support > integer_price else (support - 0.05)

        recent_5_weeks = []
        for date, row in weekly.iloc[-5:].iterrows():
            close_p = row["close"]
            ma30_p = row["ma30"]
            gap_pct = ((close_p - ma30_p) / ma30_p * 100) if pd.notna(ma30_p) and ma30_p > 0 else 0.0
            recent_5_weeks.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": close_p,
                "ma30": ma30_p,
                "gap_pct": gap_pct
            })

        strategy = generate_trading_strategy(advice, weekly, support, resistance)

        result = {
            "股票代码": stock_code,
            "股票名称": info.get("股票简称") or info.get("股票名称", ""),
            "分析日期": datetime.today().strftime("%Y-%m-%d"),
            "最新收盘": latest_close,
            "30周均值": latest_ma30,
            "阶段": stage_info["stage"],
            "阶段置信度": stage_info["confidence"],
            "阶段说明": stage_info["reason"],
            "相对强度": round(rs_info["latest_rs"], 4),
            "多周期相对强度": rs_info["rs_scores"],
            "胜率": rs_info["win_rates"],
            "风险调整超额收益": rs_info["risk_adjusted_rs"],
            "是否突破": breakout_up,
            "量能是否放大": vol_info.get("volume_ok", True),
            "量价比率": vol_info.get("vol_ratio", 1.0),
            "量能趋势": vol_info.get("volume_trend", "normal"),
            "量价背离": vol_info.get("divergence", 0),
            "RSI": rsi_val,
            "支撑位": support,
            "阻力位": resistance,
            "止损建议": stop_loss,
            "最大回撤%": risk_info["max_drawdown"],
            "年化波动率%": risk_info["annual_volatility"],
            "ATR": risk_info["atr"],
            "夏普比率": risk_info["sharpe_ratio"],
            "投资建议": advice["建议"],
            "投资说明": advice["说明"],
            "投资评分": advice["评分"],
            "建议仓位(%)": advice.get("建议仓位(%)", 0),
            "评分详情": advice.get("评分详情", {}),
            "关键参数": advice.get("关键参数", {}),
            "交易策略": strategy,
            "近五周均线差距": recent_5_weeks,
            "市场状态分析": market_regime,
            "DeepSeek分析": deepseek_result["deepseek_opinion"] if deepseek_result and deepseek_result.get("deepseek_opinion") else None,
            "错误信息": ""
        }

        return result

    except Exception as e:
        return {
            "股票代码": stock_code,
            "股票名称": info.get("股票简称") or info.get("股票名称", "") if "info" in locals() else "",
            "分析日期": datetime.today().strftime("%Y-%m-%d"),
            "最新收盘": np.nan, "30周均值": np.nan,
            "阶段": np.nan, "相对强度": np.nan,
            "是否突破": False, "量能是否放大": False,
            "支撑位": np.nan, "阻力位": np.nan, "止损建议": np.nan,
            "投资建议": "", "投资说明": "", "投资评分": np.nan,
            "错误信息": f"分析出错: {str(e)}"
        }


# ===================== 回测模块 =====================

def backtest_stock_strategy(stock_code: str, years: int = 3, min_history: int = 60) -> dict:
    weekly = fetch_stock_weekly(stock_code, years)
    if len(weekly) < min_history + 10:
        return {"错误": f"数据不足，至少需要{min_history + 10}周数据"}

    index_weekly = fetch_index_weekly_close("sh000300")
    if len(index_weekly) == 0:
        return {"错误": "无法获取基准指数数据"}

    signals = []
    total_weeks = len(weekly)

    for i in range(min_history, total_weeks):
        historical = weekly.iloc[:i + 1]
        hist_index = index_weekly[index_weekly.index.isin(historical.index)]

        stage_info = judge_stage_enhanced(historical)
        rs_info = relative_strength_enhanced(historical, hist_index) if len(hist_index) > 0 else None
        risk_info = risk_assessment_stock(historical)
        vol_info = volume_analysis(historical)

        if rs_info is None:
            rs_info = {"rs_scores": {}, "win_rates": {}, "risk_adjusted_rs": 0.0, "latest_rs": 0.0}

        advice = generate_advice_enhanced(
            stage_info, rs_info, risk_info, vol_info,
            detect_breakout(historical),
            detect_breakdown(historical),
            historical
        )

        current_close = float(historical.iloc[-1]["close"])

        forward_ret = {}
        for fwd in [4, 8, 12]:
            if i + fwd < total_weeks:
                fwd_close = float(weekly.iloc[i + fwd]["close"])
                forward_ret[f"{fwd}周"] = (fwd_close / current_close - 1) * 100
            else:
                forward_ret[f"{fwd}周"] = None

        km = stage_info.get("key_metrics", {})
        signals.append({
            "date": historical.index[-1],
            "close": current_close,
            "stage": stage_info["stage"],
            "confidence": stage_info["confidence"],
            "action": advice["建议"],
            "score": advice["评分"],
            "position": advice.get("建议仓位(%)", 0),
            "rs": rs_info["latest_rs"],
            # ML 特征列
            "diff30_pct": km.get("diff30_pct", 0),
            "diff10_pct": km.get("diff10_pct", 0),
            "ma30_slope": km.get("ma30_slope", 0),
            "ma30_r2": km.get("ma30_r2", 0),
            "rsi": km.get("rsi", 50),
            "ret_4w": km.get("ret_4w", 0),
            "ret_8w": km.get("ret_8w", 0),
            "ma_arrangement": km.get("ma_arrangement", 0),
            "vol_ratio": km.get("vol_ratio", 1.0),
            **forward_ret
        })

    df = pd.DataFrame(signals)
    if len(df) == 0:
        return {"错误": "回测未产生有效信号"}

    buy_signals = df[df["action"].isin(["买入", "谨慎买入"])]
    sell_signals = df[df["action"].isin(["减仓", "清仓", "止损"])]

    stats_result = {"总交易周数": len(df)}

    for fwd in ["4周", "8周", "12周"]:
        col = f"{fwd}收益%"
        valid = buy_signals[buy_signals[fwd].notna()]
        if len(valid) == 0:
            stats_result[f"买入信号{fwd}胜率"] = np.nan
            stats_result[f"买入信号{fwd}平均收益%"] = np.nan
            continue
        wins = (valid[fwd] > 0).sum()
        stats_result[f"买入信号{fwd}胜率"] = round(wins / len(valid) * 100, 1)
        stats_result[f"买入信号{fwd}平均收益%"] = round(valid[fwd].mean(), 2)

    for fwd in ["4周", "8周", "12周"]:
        col = f"{fwd}收益%"
        valid = sell_signals[sell_signals[fwd].notna()]
        if len(valid) == 0:
            continue
        correct = (valid[fwd] < 0).sum()
        stats_result[f"卖出信号{fwd}准确率"] = round(correct / len(valid) * 100, 1)

    score_bins = [0, 20, 40, 60, 80, 100]
    score_labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    df["评分分组"] = pd.cut(df["score"], bins=score_bins, labels=score_labels)
    for label in score_labels:
        group = df[df["评分分组"] == label]
        if len(group) > 0:
            valid_8w = group[group["8周"].notna()]
            if len(valid_8w) > 0:
                stats_result[f"评分{label}信号数"] = len(group)
                stats_result[f"评分{label}8周胜率"] = round((valid_8w["8周"] > 0).sum() / len(valid_8w) * 100, 1)
                stats_result[f"评分{label}8周平均收益%"] = round(valid_8w["8周"].mean(), 2)

    latest_stage = stage_info["stage"] if signals else "未知"

    # ML 分类器训练（如果可用）
    try:
        classifier = get_ml_classifier()
        if not classifier.available and len(df) >= classifier.min_samples:
            train_result = classifier.train_from_backtest(df)
            if train_result.get("status") == "trained":
                print(f"  [ML] 股票{stock_code} 训练完成: "
                      f"准确率={train_result.get('test_accuracy', 0):.1%}, "
                      f"样本={train_result.get('samples', 0)}")
    except Exception:
        pass

    return {
        "回测概要": stats_result,
        "信号明细": df,
        "最新阶段": latest_stage,
    }
