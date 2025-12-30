from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import akshare as ak
import warnings
warnings.filterwarnings('ignore')
def fetch_stock_info(stock_code: str) -> dict:
    if ak is None:
        return {}
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
            row = spot[spot["代码"] == stock_code]
            if len(row) > 0:
                r = row.iloc[0]
                info = {
                    "股票代码": stock_code,
                    "股票名称": str(r.get("名称", "")),
                    "最新价": str(r.get("最新价", "")),
                    "涨跌幅": str(r.get("涨跌幅", "")),
                }
        except Exception:
            info = {"股票代码": stock_code}
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
    if "日期" in df.columns:
        df["date"] = pd.to_datetime(df["日期"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        return pd.DataFrame()
    price_col = "收盘" if "收盘" in df.columns else ("close" if "close" in df.columns else None)
    if price_col is None:
        return pd.DataFrame()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    if vol_col := ("成交量" if "成交量" in df.columns else ("volume" if "volume" in df.columns else None)):
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
    df = df.sort_values("date")
    df = df.set_index("date")
    weekly = df[[price_col] + ([vol_col] if vol_col else [])].resample("W-FRI").last().dropna()
    weekly = weekly.rename(columns={price_col: "close", vol_col: "volume"} if vol_col else {price_col: "close"})
    weekly["ma30"] = weekly["close"].rolling(30).mean()
    weekly["ret"] = weekly["close"].pct_change()
    weekly["support"] = weekly["close"].rolling(20).min()
    weekly["resistance"] = weekly["close"].rolling(20).max()
    weekly = weekly.dropna()
    return weekly


def fetch_index_weekly_close(index_symbol: str = "sh000300", years: int = 3) -> pd.DataFrame:
    if ak is None:
        return pd.DataFrame()
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
    df = None
    try:
        df = ak.stock_zh_index_daily(symbol=index_symbol, start_date=start_date, end_date=end_date)
    except Exception:
        df = None
    if df is None or len(df) == 0:
        try:
            code = index_symbol.replace("sh", "").replace("sz", "")
            df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date)
        except Exception:
            df = None
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if "日期" in df.columns:
        df["date"] = pd.to_datetime(df["日期"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        return pd.DataFrame()
    price_col = "收盘" if "收盘" in df.columns else ("close" if "close" in df.columns else None)
    if price_col is None:
        return pd.DataFrame()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values("date")
    df = df.set_index("date")
    weekly = df[[price_col]].resample("W-FRI").last().dropna()
    weekly = weekly.rename(columns={price_col: "close"})
    weekly["ret"] = weekly["close"].pct_change()
    weekly = weekly.dropna()
    return weekly


def compute_ma_slope(series: pd.Series, window: int = 10) -> float:
    s = series.dropna()
    if len(s) < window:
        return 0.0
    x = np.arange(window)
    y = s.iloc[-window:].to_numpy()
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


def judge_stage(weekly_df: pd.DataFrame) -> int:
    latest = weekly_df.iloc[-1]
    ma = latest["ma30"]
    close = latest["close"]
    slope = compute_ma_slope(weekly_df["ma30"], 10)
    diff = close / ma - 1 if ma and ma != 0 else 0
    if close < ma and slope < 0 and diff < -0.03:
        return 4
    if close > ma and slope > 0:
        return 2
    if close > ma and slope <= 0:
        return 3
    return 1


def relative_strength(stock_weekly: pd.DataFrame, index_weekly: pd.DataFrame, lookback: int = 12) -> float:
    aligned = stock_weekly.join(index_weekly[["ret"]], how="inner", rsuffix="_index")
    if len(aligned) < lookback:
        return 0.0
    w = aligned.iloc[-lookback:]
    s_ret = w["ret"].mean()
    i_ret = w["ret_index"].mean() if "ret_index" in w.columns else 0.0
    return float(s_ret - i_ret)


def detect_breakout(weekly_df: pd.DataFrame, lookback: int = 12, threshold: float = 0.01) -> bool:
    if len(weekly_df) < lookback + 1:
        return False
    recent = weekly_df.iloc[-(lookback + 1):-1]
    upper = recent["close"].max()
    latest = weekly_df.iloc[-1]
    return latest["close"] > upper * (1 + threshold)


def generate_advice(
    stage: int, 
    rs_score: float, 
    breakout: bool, 
    volume_ok: bool,
    rs_threshold: float = 0.005  # 相对强度临界值（微弱正/负的区分）
) -> dict:
    """
    生成投资建议，基于阶段、相对强度、突破、量能多维度判断
    :param stage: 股票阶段（1:筑底, 2:上升, 3:顶部, 4:下跌）
    :param rs_score: 相对强度（股票-指数周度收益率均值）
    :param breakout: 是否突破阻力位
    :param volume_ok: 量能是否放大（突破时成交量>近12周均值1.5倍）
    :param rs_threshold: 相对强度临界值，区分「微弱正/负」
    :return: 包含建议、说明、评分的字典
    """
    # === 核心逻辑：按「趋势阶段→强弱信号→量能/突破」分层 ===
    # 1. 第二阶段（上升趋势）：核心分「强势/中性/弱势」
    if stage == 2:
        # 1.1 强势：上升+跑赢指数+突破+放量 → 买入
        if rs_score > rs_threshold and breakout and volume_ok:
            action = "买入"
            note = "第二阶段（上升趋势）：跑赢指数+放量突破阻力位，建议分批建仓（优先左侧1/3仓位，回踩均线补仓）"
            score = 85
        # 1.2 中性1：上升+跑赢指数+突破但缩量 → 观望（等待回踩确认）
        elif rs_score > rs_threshold and breakout and not volume_ok:
            action = "观望"
            note = "第二阶段（上升趋势）：跑赢指数+突破阻力位，但量能未放大，等待回踩30周均线确认后再建仓"
            score = 70
        # 1.3 中性2：上升+跑赢指数但未突破 → 观望（等待突破/回踩）
        elif rs_score > rs_threshold and not breakout:
            action = "观望"
            note = "第二阶段（上升趋势）：跑赢指数，但未突破阻力位，可等待突破或回踩30周均线后布局"
            score = 65
        # 1.4 弱势1：上升但跑输指数（微弱负）+ 突破 → 观望（警惕补跌）
        elif abs(rs_score) <= rs_threshold and breakout:
            action = "观望"
            note = "第二阶段（上升趋势）：相对强度接近0+突破阻力位，暂观望，待rs转正且量能放大后再介入"
            score = 60
        # 1.5 弱势2：上升但明显跑输指数 → 观望（优先换更强标的）
        else:
            action = "观望"
            note = f"第二阶段（上升趋势）：但相对强度为{rs_score:.4f}（跑输指数），建议观望，优先选择跑赢指数的标的"
            score = 55

    # 2. 第一阶段（筑底期）：仅观望，等待趋势确认
    elif stage == 1:
        action = "观望"
        note = "第一阶段（筑底期）：股价在均线下方+均线斜率向下，等待周线突破30周均线且均线走升后再关注"
        score = 60

    # 3. 第三阶段（顶部期）：分「强顶/弱顶」
    elif stage == 3:
        # 3.1 顶部但仍跑赢指数 → 减仓（分批止盈）
        if rs_score > rs_threshold:
            action = "减仓"
            note = "第三阶段（顶部趋势）：股价在均线上但均线斜率向下+仍跑赢指数，建议分批减仓（先减1/2仓位）"
            score = 45
        # 3.2 顶部且跑输指数 → 重仓减仓
        else:
            action = "减仓"
            note = f"第三阶段（顶部趋势）：股价在均线上但均线斜率向下+跑输指数（rs={rs_score:.4f}），建议减仓至1/3以下"
            score = 40

    # 4. 第四阶段（下跌趋势）：分「极端弱势/普通弱势」
    elif stage == 4:
        # 4.1 下跌但未破关键支撑 → 止损（小仓位）
        if not breakout:  # 未跌破支撑位，仍有反弹可能
            action = "止损"
            note = "第四阶段（下跌趋势）：股价在均线下方+均线斜率向下，建议严格止损（止损位=支撑位-0.05）"
            score = 25
        # 4.2 下跌且跌破支撑 → 清仓
        else:
            action = "清仓"
            note = "第四阶段（下跌趋势）：股价跌破支撑位+均线向下，建议立即清仓，避免深度套牢"
            score = 15

    # 5. 异常阶段（如阶段值错误）：兜底观望
    else:
        action = "观望"
        note = f"阶段值异常（stage={stage}），暂无法判断趋势，建议观望"
        score = 50

    return {
        "建议": action,
        "说明": note,
        "评分": score,
        "关键参数": {  # 补充参数便于复盘
            "阶段": stage,
            "相对强度": round(rs_score, 4),
            "是否突破": breakout,
            "量能是否放大": volume_ok
        }
    }


def analyze_stock(stock_code: str) -> dict:
    """分析单个股票"""
    try:
        info = fetch_stock_info(stock_code)
        weekly = fetch_stock_weekly(stock_code)
        index_weekly = fetch_index_weekly_close("sh000300")
        stage = judge_stage(weekly)
        rs = relative_strength(weekly, index_weekly, 12) if len(index_weekly) > 0 else 0.0
        bo = bool(detect_breakout(weekly, 12, 0.01))
        vol_ok = True
        
        if "volume" in weekly.columns and bo:
            recent = weekly.iloc[-13:-1]
            vol_mean = recent["volume"].mean()
            vol_ok = bool(weekly.iloc[-1]["volume"] > vol_mean * 1.5)
        
        advice = generate_advice(stage, rs, bo, vol_ok)
        latest_close = float(weekly.iloc[-1]["close"])
        latest_ma = float(weekly.iloc[-1]["ma30"])
        support = float(weekly.iloc[-1]["support"])
        resistance = float(weekly.iloc[-1]["resistance"])
        integer_price = np.floor(support)
        stop_loss = (integer_price - 0.05) if support > integer_price else (support - 0.05)
        
        # 整理结果为扁平结构，方便写入CSV
        result = {
            "股票代码": stock_code,
            "股票名称": info.get("股票简称", ""),
            "分析日期": datetime.today().strftime("%Y-%m-%d"),
            "最新收盘": latest_close,
            "30周均值": latest_ma,
            "阶段": stage,
            "相对强度": rs,
            "是否突破": bo,
            "量能是否放大": vol_ok,
            "支撑位": support,
            "阻力位": resistance,
            "止损建议": stop_loss,
            "投资建议": advice["建议"],
            "投资说明": advice["说明"],
            "投资评分": advice["评分"],
            "错误信息": ""
        }
        
        return result
    
    except Exception as e:
        return {
            "股票代码": stock_code,
            "股票名称": "",
            "分析日期": datetime.today().strftime("%Y-%m-%d"),
            "最新收盘": np.nan,
            "30周均值": np.nan,
            "阶段": np.nan,
            "相对强度": np.nan,
            "是否突破": False,
            "量能是否放大": False,
            "支撑位": np.nan,
            "阻力位": np.nan,
            "止损建议": np.nan,
            "投资建议": "",
            "投资说明": "",
            "投资评分": np.nan,
            "错误信息": f"分析出错: {str(e)}"
        }

