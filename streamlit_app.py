import streamlit as st
import sys
import os
import pandas as pd
import concurrent.futures

# 添加当前目录到路径，确保可以导入 advisor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advisor_stock import analyze_stock, fetch_stock_weekly, backtest_stock_strategy
from advisor_fund import analyze_fund_enhanced, backtest_fund_strategy

# 设置页面配置
st.set_page_config(
    page_title="智能投资分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .advice-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .buy-advice {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
    }
    .sell-advice {
        background-color: #ffeaea;
        border-left: 4px solid #dc3545;
    }
    .hold-advice {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<div class="main-header">📈 智能投资分析系统</div>', unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 分析配置")
    
    # 分析类型选择
    analysis_type = st.radio(
        "选择分析类型:",
        ["股票分析", "基金分析"],
        horizontal=True
    )
    
    # 代码输入
    batch_mode = False
    invest_amount = 0.0
    if analysis_type == "股票分析":
        code = st.text_input("请输入股票代码:", placeholder="单只如: 000001，或多只用逗号分隔")
        st.caption("支持A股股票代码，如: 000001, 600000, 300001；多只股票用逗号分隔可进行批量分析")
        batch_mode = st.checkbox("批量分析（逗号分隔多只股票代码）")
        invest_amount = st.number_input("总投资金额(元)", min_value=0.0, value=0.0, step=1000.0)
    else:
        code = st.text_input("请输入基金代码:", placeholder="单只如: 000001，或多只用逗号分隔")
        st.caption("支持开放式基金代码，如: 000001, 110011；多只基金用逗号分隔可进行批量分析")
        batch_mode = st.checkbox("批量分析（逗号分隔多个基金代码）")
        invest_amount = st.number_input("总投资金额(元)", min_value=0.0, value=0.0, step=1000.0)
    
    # DeepSeek API Key 配置
    with st.expander("🤖 DeepSeek AI 配置（可选）"):
        import config as _cfg

        # 是否启用 DeepSeek
        enable_ds = st.checkbox(
            "启用 DeepSeek AI 分析",
            value=st.session_state.get("enable_deepseek", _cfg.DEEPSEEK_CONFIG.get('enabled', True)),
            help="关闭后完全使用规则系统进行分析，不依赖 API Key"
        )
        st.session_state.enable_deepseek = enable_ds
        _cfg.DEEPSEEK_CONFIG['enabled'] = enable_ds
        _cfg.ANALYSIS_CONFIG['fund']['enable_deepseek'] = enable_ds

        # API Key 输入：如果前端传了 key 就用前端的，否则用环境变量 DEEPSEEK_API_KEY
        prev_key = st.session_state.get("deepseek_key", "")
        ds_key = st.text_input(
            "DeepSeek API Key（留空则使用环境变量 DEEPSEEK_API_KEY）",
            value=prev_key,
            type="password",
            placeholder="sk-xxx，留空则使用环境变量"
        )

        if ds_key and ds_key != prev_key:
            # 前端传入新 key → 使用前端 key
            st.session_state.deepseek_key = ds_key
            _cfg.DEEPSEEK_CONFIG['api_key'] = ds_key
            import advisor_fund
            advisor_fund._deepseek_client = None  # 重置客户端缓存
        elif not ds_key and prev_key:
            # 用户清除了 key → 恢复为环境变量
            del st.session_state.deepseek_key
            _cfg.DEEPSEEK_CONFIG['api_key'] = os.environ.get("DEEPSEEK_API_KEY", "")
            import advisor_fund
            advisor_fund._deepseek_client = None  # 重置客户端缓存

        if enable_ds:
            if _cfg.DEEPSEEK_CONFIG.get('api_key'):
                source = "（使用前端密钥）" if ds_key else "（使用环境变量）"
                st.caption(f"✅ DeepSeek 已配置{source}")
            else:
                st.caption("⚠️ 未检测到 API Key，请输入或设置 DEEPSEEK_API_KEY 环境变量")
        else:
            st.caption("💡 DeepSeek 已禁用，使用规则系统进行分析")

    # 分析按钮
    analyze_button = st.button("🔍 开始分析", type="primary", use_container_width=True)
    
    # 使用说明
    with st.expander("📖 使用说明"):
        st.write("""
        **股票分析功能：**
        - 技术分析（趋势、相对强弱）
        - 支撑阻力位识别
        - 买卖建议生成
        
        **基金分析功能：**
        - 业绩表现分析
        - 风险评估
        - 与基准指数对比
        - 投资建议
        
        **注意事项：**
        - 确保输入正确的代码格式
        - 分析需要联网获取数据
        - 结果仅供参考，投资有风险
        """)
    
    # 分析历史（使用session_state存储）
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if st.session_state.analysis_history:
        with st.expander("📊 最近分析"):
            for i, (analysis_type, code, timestamp) in enumerate(reversed(st.session_state.analysis_history[-5:])):
                st.write(f"{i+1}. {analysis_type} - {code} ({timestamp})")

def get_advice_class(advice):
    """根据建议类型返回对应的CSS类"""
    if '买入' in advice or '强烈' in advice:
        return 'buy-advice'
    elif '卖出' in advice or '回避' in advice:
        return 'sell-advice'
    else:
        return 'hold-advice'

def display_welcome():
    """显示欢迎页面"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 欢迎使用智能投资分析系统
        
        本系统提供专业的股票和基金分析服务，帮助您做出更明智的投资决策。
        
        **主要功能：**
        
        📊 **股票分析**
        - 技术趋势分析
        - 相对强弱评估  
        - 支撑阻力识别
        - 买卖时机判断
        
        💰 **基金分析** 
        - 业绩表现评估
        - 风险收益分析
        - 与基准对比
        - 投资建议生成
        
        **使用方法：**
        1. 在左侧选择分析类型（股票/基金）
        2. 输入对应的代码
        3. 点击开始分析按钮
        4. 查看详细的分析报告
        
        **免责声明：**
        本系统提供的分析结果仅供参考，不构成投资建议。投资有风险，决策需谨慎。
        """)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="color: #1f77b4;">📈 实时数据</h4>
            <p>基于最新的市场数据进行分析</p>
            <br>
            <h4 style="color: #28a745;">🔒 安全可靠</h4>
            <p>本地分析，保护您的隐私</p>
            <br>
            <h4 style="color: #ffc107;">⚡ 快速高效</h4>
            <p>秒级响应，即时获取结果</p>
        </div>
        """, unsafe_allow_html=True)

def display_stock_analysis(result):
    """显示增强版股票分析结果"""
    st.header(f"📊 {result.get('股票代码', 'Unknown')} {result.get('股票名称', '')} 股票分析报告")
    
    score = result.get('投资评分', 60)
    stage = result.get('阶段', 1)
    stage_names = {1: "筑底期", 2: "上升期", 3: "顶部期", 4: "下跌期"}
    stage_name = stage_names.get(stage, f"第{stage}阶段")
    confidence = result.get('阶段置信度', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("当前趋势", stage_name, delta=f"置信度{confidence*100:.0f}%")
    with col2:
        rs_value = result.get('相对强度', 0)
        st.metric("相对强度(12周)", f"{rs_value:+.4f}")
    with col3:
        rsi_val = result.get('RSI', 50)
        st.metric("RSI(14周)", f"{rsi_val:.1f}",
                 delta="超买" if rsi_val > 70 else "超卖" if rsi_val < 30 else None,
                 delta_color="inverse" if rsi_val > 70 else "normal")
    with col4:
        vol_trend = result.get('量能趋势', 'normal')
        vol_trend_map = {"放量": "📈 放量", "缩量": "📉 缩量", "normal": "正常"}
        st.metric("量能趋势", vol_trend_map.get(vol_trend, "正常"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        latest_price = result.get('最新收盘', 0)
        st.metric("最新收盘", f"{latest_price:.2f}")
    with col2:
        ma30 = result.get('30周均值', 0)
        st.metric("30周均值", f"{ma30:.3f}")
    with col3:
        support = result.get('支撑位', 0)
        st.metric("支撑位", f"{support:.2f}")
    with col4:
        resistance = result.get('阻力位', 0)
        st.metric("阻力位", f"{resistance:.2f}")

    advice = result.get('投资建议', '观望')
    advice_class = get_advice_class(advice)
    pos_pct = result.get('建议仓位(%)', 0)
    
    st.subheader("💡 投资建议")
    advice_html = f"""
    <div class="advice-box {advice_class}">
        <h3>{advice} (评分: {score}/100 | 建议仓位: {pos_pct}%)</h3>
        <p><strong>投资说明:</strong> {result.get('投资说明', '暂无详细说明')}</p>
        {f'<p><strong>止损建议:</strong> {result.get("止损建议", 0):.2f}</p>' if result.get('止损建议') else ''}
        {f'<p><strong>阶段说明:</strong> {result.get("阶段说明", "")}</p>' if result.get('阶段说明') else ''}
    </div>
    """
    st.markdown(advice_html, unsafe_allow_html=True)

    # 交易策略
    strategy = result.get('交易策略', {})
    if strategy:
        with st.expander("🎯 交易策略", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**建议仓位:** {strategy.get('建议仓位(%)', 0)}%")
                st.write(f"**当前价:** {strategy.get('当前价', 0):.2f}")
                st.write(f"**止损位:** {strategy.get('止损位', 0):.2f}")
                st.write(f"**目标位:** {strategy.get('目标位', 0):.2f}")
            with col2:
                if strategy.get('分批买入'):
                    st.write("**分批买入计划:**")
                    for b in strategy['分批买入']:
                        st.write(f"- 第{b['批次']}批: {b['比例']}% {b['条件']}")
                if strategy.get('加仓条件'):
                    st.write(f"**加仓条件:** {strategy['加仓条件']}")
                if strategy.get('减仓条件'):
                    st.write(f"**减仓条件:** {strategy['减仓条件']}")

    # 近五周均线差距
    recent_5_weeks = result.get('近五周均线差距', [])
    if recent_5_weeks:
        st.subheader("📅 最近五周与30周均线差距趋势")
        df_gap = pd.DataFrame(recent_5_weeks)
        if not df_gap.empty:
            df_gap['差距变化'] = df_gap['gap_pct'].apply(lambda x: f"{x:+.2f}%")
            df_gap['收盘价'] = df_gap['close'].apply(lambda x: f"{x:.2f}")
            df_gap['30周均线'] = df_gap['ma30'].apply(lambda x: f"{x:.2f}")
            df_gap = df_gap.rename(columns={'date': '日期'})
            st.table(df_gap[['日期', '收盘价', '30周均线', '差距变化']].set_index('日期'))

    # 周线走势
    stock_code = result.get('股票代码', '').strip()
    if stock_code:
        try:
            weekly_data = fetch_stock_weekly(stock_code, years=3)
        except Exception:
            weekly_data = None
        if weekly_data is not None and len(weekly_data) > 0:
            try:
                chart_df = weekly_data[["close", "ma30", "support", "resistance"]].copy()
                chart_df = chart_df.rename(columns={
                    "close": "收盘价", "ma30": "30周均线",
                    "support": "支撑位", "resistance": "阻力位",
                })
                st.subheader("📈 周线走势（含30周均线、支撑/阻力）")
                st.line_chart(chart_df)
            except Exception:
                pass

    # 市场状态（新增）
    market_regime = result.get('市场状态分析', {})
    if market_regime:
        regime = market_regime.get('regime', 'unknown')
        regime_icons = {
            'strong_bull': '🚀', 'bull': '📈', 'sideways': '➡️',
            'volatile_sideways': '🌊', 'volatile_bull': '⚡',
            'bear': '📉', 'strong_bear': '🔻', 'unknown': '❓'
        }
        score = market_regime.get('score', 0)
        desc = market_regime.get('description', '')
        st.subheader("🌍 市场状态")
        st.info(f"{regime_icons.get(regime, '❓')} **{regime}** (评分: {score}) — {desc}")

    # DeepSeek AI 验证（新增）
    ds_opinion = result.get('DeepSeek分析')
    if ds_opinion:
        with st.expander("🤖 DeepSeek AI 验证", expanded=False):
            if isinstance(ds_opinion, dict):
                agree = ds_opinion.get('agree')
                if agree is not None:
                    st.write(f"**是否同意系统判断:** {'✅ 同意' if agree else '⚠️ 不同意'}")
                alt_stage = ds_opinion.get('alternative_stage')
                alt_conf = ds_opinion.get('alternative_confidence')
                if alt_stage:
                    st.write(f"**替代判断:** {alt_stage} (置信度: {alt_conf}%)")
                signals = ds_opinion.get('key_signals', [])
                if signals:
                    st.write("**看多信号:**")
                    for s in signals:
                        st.write(f"- {s}")
                risks = ds_opinion.get('key_risks', [])
                if risks:
                    st.write("**看空信号:**")
                    for r in risks:
                        st.write(f"- {r}")
                advice_text = ds_opinion.get('advice')
                if advice_text:
                    st.info(f"💡 {advice_text}")

    # 详细分析
    with st.expander("📈 技术分析详情"):
        st.write(f"**分析日期:** {result.get('分析日期', '未知')}")
        st.write(f"**当前阶段:** 第{result.get('阶段', 1)}阶段 ({stage_name})")
        if result.get('错误信息'):
            st.error(f"⚠️ 分析警告: {result.get('错误信息')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**技术指标:**")
            st.write(f"- 相对强度(12周): {result.get('相对强度', 0):+.4f}")
            st.write(f"- RSI(14周): {result.get('RSI', 50):.1f}")
            st.write(f"- 突破状态: {'已突破' if result.get('是否突破') else '未突破'}")
            st.write(f"- 量能: {'放大' if result.get('量能是否放大') else '正常'}")
            st.write(f"- 量能趋势: {result.get('量能趋势', 'normal')}")
            st.write(f"- 量价比率: {result.get('量价比率', 1.0):.2f}")
            divergence = result.get('量价背离', 0)
            if divergence != 0:
                st.write(f"- 量价背离: {'顶背离⚠️' if divergence == -1 else '底背离💡'}")
        with col2:
            st.write("**风险指标:**")
            st.write(f"- 最大回撤: {result.get('最大回撤%', 0):.1f}%")
            st.write(f"- 年化波动率: {result.get('年化波动率%', 0):.1f}%")
            st.write(f"- ATR: {result.get('ATR', 0):.4f}")
            st.write(f"- 支撑位: {support:.2f}")
            st.write(f"- 阻力位: {resistance:.2f}")
            st.write(f"- 止损建议: {result.get('止损建议', 0):.2f}")

    # 多周期相对强度
    rs_scores = result.get('多周期相对强度', {})
    win_rates = result.get('胜率', {})
    if rs_scores:
        with st.expander("📊 多周期相对强度分析"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**相对强度（超额收益）:**")
                for period in ['12周', '26周', '52周']:
                    v = rs_scores.get(period, 0)
                    st.write(f"- {period}: {v:+.4f}")
                st.write(f"- 风险调整超额: {result.get('风险调整超额收益', 0):+.3f}")
            with col2:
                st.write("**跑赢指数胜率:**")
                for period in ['12周', '26周', '52周']:
                    v = win_rates.get(period, 0)
                    st.write(f"- {period}: {v*100:.0f}%")

    # 评分详情
    score_details = result.get('评分详情', {})
    if score_details:
        with st.expander("🧮 评分计算详情"):
            st.write("**评分构成：**")
            st.write(f"- 基础分数（阶段分）: {score_details.get('基础分数', 0)}")
            st.write(f"- 增强分数（收益因子）: {score_details.get('增强分数', 0)}")
            st.write(f"- 风险分数（风险因子）: {score_details.get('风险分数', 0)}")
            st.write(f"- 最终分数: {score_details.get('最终分数', 0)}")
            
            st.write("**增强因子:**")
            for f in score_details.get('增强因子', []):
                st.write(f"- {f}")
            st.write("**风险因子:**")
            for f in score_details.get('风险因子', []):
                st.write(f"- {f}")

def display_fund_analysis(result):
    """显示基金分析结果 - 基于新的数据结构"""
    # 获取基金基本信息
    fund_info = result.get('基金基本信息', {})
    fund_code = fund_info.get('基金代码', 'Unknown') if fund_info else 'Unknown'
    fund_name = fund_info.get('基金名称', '') if fund_info else ''
    
    st.header(f"💰 {fund_code} {fund_name} 基金分析报告")
    
    # 获取各类数据
    latest_data = result.get('最新数据', {})
    trend_analysis = result.get('趋势分析', {})
    rs_analysis = result.get('相对强度分析', {})
    risk_analysis = result.get('风险评估', {})
    advice_result = result.get('投资建议', {})
    
    # 市场状态显示（新增）
    market_regime = result.get('市场状态分析', {})
    regime = market_regime.get('regime', 'unknown')
    regime_icons = {
        'strong_bull': '🚀', 'bull': '📈', 'sideways': '➡️',
        'volatile_sideways': '🌊', 'volatile_bull': '⚡',
        'bear': '📉', 'strong_bear': '🔻', 'unknown': '❓'
    }
    regime_desc = market_regime.get('description', '')

    # 核心指标卡片 - 顶部展示（新增市场状态）
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("市场状态", f"{regime_icons.get(regime, '❓')} {regime}",
                 help=regime_desc)

    with col2:
        stage = trend_analysis.get('stage', '未知')
        stage_confidence = trend_analysis.get('confidence', 0) * 100
        stage_reason = trend_analysis.get('reason', '')
        st.metric("当前趋势", f"阶段{stage}", help=stage_reason)

    with col3:
        latest_nav = latest_data.get('单位净值', 0)
        ma20 = latest_data.get('20周均线', 0)
        nav_vs_ma20 = ((latest_nav - ma20) / ma20 * 100) if ma20 > 0 else 0
        st.metric("最新净值", f"{latest_nav:.4f}",
                 delta=f"{nav_vs_ma20:+.1f}% vs 20周均线" if nav_vs_ma20 != 0 else None)

    with col4:
        max_drawdown = latest_data.get('最大回撤(%)', risk_analysis.get('max_drawdown', 0))
        st.metric("最大回撤", f"{max_drawdown:.1f}%",
                 delta="风险较高" if max_drawdown < -20 else "风险适中" if max_drawdown < -10 else "风险较低",
                 delta_color="inverse")

    with col5:
        sharpe_ratio = latest_data.get('夏普比率', risk_analysis.get('sharpe_ratio', 0))
        st.metric("夏普比率", f"{sharpe_ratio:.2f}",
                 delta="优秀" if sharpe_ratio > 1.0 else "良好" if sharpe_ratio > 0.5 else "一般",
                 delta_color="normal" if sharpe_ratio > 0.5 else "inverse")
    
    # 投资建议 - 突出显示
    advice = advice_result.get('建议操作', '观望')
    advice_score = advice_result.get('评分', 60)
    advice_desc = advice_result.get('建议说明', '')
    advice_confidence = advice_result.get('建议置信度', 30)
    advice_class = get_advice_class(advice)
    
    st.subheader("💡 投资建议")
    advice_html = f"""
    <div class="advice-box {advice_class}">
        <h3>{advice} (评分: {advice_score:.0f}/100, 置信度: {advice_confidence:.0f}%)</h3>
        <p><strong>建议说明:</strong> {advice_desc}</p>
        <p><strong>建议仓位:</strong> {advice_result.get('建议仓位(%)', 0):.0f}% /
           <strong>止损:</strong> {advice_result.get('止损建议(%)', 5):.1f}% /
           <strong>目标:</strong> {advice_result.get('目标收益(%)', 10):.1f}%</p>
    </div>
    """
    st.markdown(advice_html, unsafe_allow_html=True)

    # 模糊阶段概率展示（新增）
    stage_probs = trend_analysis.get('stage_probs', {})
    if stage_probs and any(v > 0 for v in stage_probs.values()):
        st.subheader("📊 阶段概率分布")
        prob_cols = st.columns(4)
        prob_stages = [
            ("筑底期", "accumulation", "#6c757d"),
            ("上升期", "rising", "#28a745"),
            ("顶部期", "top", "#ffc107"),
            ("下跌期", "falling", "#dc3545")
        ]
        for col, (label, key, color) in zip(prob_cols, prob_stages):
            prob = stage_probs.get(key, 0) * 100
            col.metric(label, f"{prob:.0f}%")
            col.markdown(f'<div style="background:{color};height:5px;border-radius:3px;width:{min(prob,100)}%"></div>',
                        unsafe_allow_html=True)

    # ATR仓位分析（新增）
    position_info = result.get('仓位分析', {})
    if position_info:
        with st.expander("📐 动态仓位分析（ATR）", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("建议仓位", f"{position_info.get('position_pct', 0):.1f}%")
            col2.metric("ATR波动率", f"{position_info.get('atr_pct', 0):.2f}%")
            col3.metric("止损距离", f"{position_info.get('stop_loss_pct', 0):.1f}%")
            col4.metric("目标收益", f"{position_info.get('target_pct', 0):.1f}%")
            st.caption(f"市场乘数: {position_info.get('regime_multiplier', 0):.1f}x | "
                      f"波动调整: {position_info.get('vol_adjustment', 0):.1f}x | "
                      f"基于ATR的动态仓位在高波动时自动降低")
    
    # 评分计算详情
    score_details = advice_result.get('评分详情', {})
    if score_details:
        with st.expander("📊 评分计算详情"):
            st.write("**评分规则：**")
            st.write("- 基础分数：根据趋势阶段确定（0-75分）")
            st.write("- 风险分数：基于最大回撤、夏普比率等风险指标（-100至+15分）")
            st.write("- 增强分数：基于相对强度、年化收益等收益指标（-10至+24分）")
            st.write("- 额外调整：夏普比率和最大回撤的加权调整")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**评分构成：**")
                st.write(f"- 基础分数：{score_details.get('基础分数', 0)}")
                st.write(f"- 风险分数：{score_details.get('风险分数', 0)}")
                st.write(f"- 增强分数：{score_details.get('增强分数', 0)}")
                st.write(f"- 夏普比率调整：{score_details.get('夏普比率调整', 0)}")
                st.write(f"- 最大回撤调整：{score_details.get('最大回撤调整', 0)}")
                st.write(f"- 最终分数：{score_details.get('最终分数', 0)}")
            
            with col2:
                st.write("**风险因子得分：**")
                risk_details = score_details.get('风险因子详情', {})
                for factor, score in risk_details.items():
                    if score != 0:
                        st.write(f"- {factor}：{score}")
                
                st.write("**增强因子得分：**")
                enhancement_details = score_details.get('增强因子详情', {})
                for factor, score in enhancement_details.items():
                    if score != 0:
                        st.write(f"- {factor}：{score}")
    
    # 净值周线走势可视化（含20/30周均线）
    weekly_data = result.get('历史周度数据')
    if weekly_data is not None:
        try:
            df_weekly = weekly_data.copy()
            chart_df = df_weekly[["close", "ma20", "ma30"]].copy()
            chart_df = chart_df.rename(columns={
                "close": "单位净值",
                "ma20": "20周均线",
                "ma30": "30周均线",
            })
            st.subheader("📈 净值周线走势（含20/30周均线）")
            st.line_chart(chart_df)
        except Exception:
            pass
    
    # 趋势分析详情
    with st.expander("📈 趋势分析"):
        if trend_analysis:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**关键指标:**")
                key_metrics = trend_analysis.get('key_metrics', {})
                st.write(f"- 10周均线偏离: {key_metrics.get('diff10_pct', 0):+.1f}%")
                st.write(f"- 30周均线偏离: {key_metrics.get('diff30_pct', 0):+.1f}%")
                st.write(f"- 30周均线斜率: {key_metrics.get('ma30_slope', 0):+.3f}")
                st.write(f"- 均线拟合度R²: {key_metrics.get('ma30_r2', 0):.3f}")
                st.write(f"- RSI(14): {key_metrics.get('rsi', 50):.1f}")

            with col2:
                st.write("**趋势判断:**")
                st.write(f"- 当前阶段: 第{trend_analysis.get('stage', 1)}阶段")
                st.write(f"- 置信度: {trend_analysis.get('confidence', 0)*100:.0f}%")
                st.write(f"- 判断理由: {trend_analysis.get('reason', '暂无')}")

                # 均线排列状态
                ma_arrangement = key_metrics.get('ma_arrangement', 0)
                arrangement_text = {1: "多头排列", 0: "缠绕整理", -1: "空头排列"}.get(ma_arrangement, "未知")
                st.write(f"- 均线排列: {arrangement_text}")
                st.write(f"- 4周涨幅: {key_metrics.get('ret_4w', 0):+.2f}%")
                st.write(f"- 8周涨幅: {key_metrics.get('ret_8w', 0):+.2f}%")
        else:
            st.write("暂无趋势分析数据")
    
    # 相对强度分析
    with st.expander("📊 相对强度分析"):
        if rs_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**相对强度得分:**")
                rs_scores = rs_analysis.get('rs_scores', {})
                st.write(f"- 12周相对强度: {rs_scores.get('12周', 0):+.3f}")
                st.write(f"- 26周相对强度: {rs_scores.get('26周', 0):+.3f}")
                st.write(f"- 52周相对强度: {rs_scores.get('52周', 0):+.3f}")
                st.write(f"- 最新相对强度: {rs_analysis.get('latest_rs', 0):+.3f}")
            
            with col2:
                st.write("**胜率统计:**")
                win_rates = rs_analysis.get('win_rates', {})
                st.write(f"- 12周胜率: {win_rates.get('12周', 0)*100:.0f}%")
                st.write(f"- 26周胜率: {win_rates.get('26周', 0)*100:.0f}%")
                st.write(f"- 52周胜率: {win_rates.get('52周', 0)*100:.0f}%")
                st.write(f"- 风险调整收益: {rs_analysis.get('risk_adjusted_rs', 0):+.3f}")
        else:
            st.write("暂无相对强度分析数据")
    
    # 风险评估
    with st.expander("⚠️ 风险评估"):
        if risk_analysis or latest_data:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**风险指标:**")
                max_drawdown = latest_data.get('最大回撤(%)', risk_analysis.get('max_drawdown', 0))
                downside_vol = latest_data.get('下行波动率(%)', 0)
                st.write(f"- 最大回撤: {max_drawdown:.1f}%")
                st.write(f"- 下行波动率: {downside_vol:.1f}%")
                st.write(f"- 夏普比率: {latest_data.get('夏普比率', 0):.3f}")
                st.write(f"- 年化收益率: {latest_data.get('年化收益率(%)', 0):.2f}%")
            with col2:
                st.write("**风险等级评估:**")
                # 基于回撤的风险等级
                if max_drawdown < -30:
                    risk_level = "高风险"
                elif max_drawdown < -20:
                    risk_level = "中高风险"
                elif max_drawdown < -10:
                    risk_level = "中等风险"
                else:
                    risk_level = "低风险"
                
                st.write(f"- 风险等级: {risk_level}")
                st.write(f"- 回撤承受能力: {'较弱' if max_drawdown < -25 else '一般' if max_drawdown < -15 else '较强'}")
                
                # 夏普比率评估
                sharpe = latest_data.get('夏普比率', 0)
                sharpe_assessment = "优秀" if sharpe > 1.0 else "良好" if sharpe > 0.5 else "一般" if sharpe > 0 else "较差"
                st.write(f"- 风险调整后收益: {sharpe_assessment}")
        else:
            st.write("暂无风险评估数据")
    
    # ML分析展示（新增）
    ml_info = result.get('ML分析')
    if ml_info and ml_info.get("ml_info"):
        with st.expander("🧠 机器学习分析"):
            st.write(f"**ML预测阶段:** 第{ml_info.get('stage', '?')}阶段 "
                    f"(置信度: {ml_info.get('confidence', 0)*100:.0f}%)")
            st.write(f"**阶段概率:** {ml_info.get('stage_probs', {})}")
            feat_imp = ml_info.get('ml_info', {}).get('feature_importance', {})
            if feat_imp:
                st.write("**特征重要性:**")
                for feat, imp in sorted(feat_imp.items(), key=lambda x: -x[1]):
                    st.write(f"- {feat}: {imp:.1%}")

    # DeepSeek分析展示（新增）
    ds_opinion = result.get('DeepSeek分析')
    if ds_opinion:
        with st.expander("🤖 DeepSeek AI 验证"):
            if isinstance(ds_opinion, dict):
                agree = ds_opinion.get('agree')
                if agree is not None:
                    st.write(f"**是否同意系统判断:** {'✅ 同意' if agree else '⚠️ 不同意'}")
                alt_stage = ds_opinion.get('alternative_stage')
                alt_conf = ds_opinion.get('alternative_confidence')
                if alt_stage:
                    st.write(f"**替代判断:** {alt_stage} (置信度: {alt_conf}%)")
                signals = ds_opinion.get('key_signals', [])
                if signals:
                    st.write("**看多信号:**")
                    for s in signals:
                        st.write(f"- {s}")
                risks = ds_opinion.get('key_risks', [])
                if risks:
                    st.write("**看空信号:**")
                    for r in risks:
                        st.write(f"- {r}")
                advice_text = ds_opinion.get('advice')
                if advice_text:
                    st.info(f"💡 {advice_text}")

    # 交易策略
    trading_strategy = result.get('交易策略', {})
    if trading_strategy:
        with st.expander("🎯 具体交易策略"):
            st.write(f"**{trading_strategy.get('操作计划', '')}**")
            
            if trading_strategy.get('分批买入'):
                st.write("**分批买入计划:**")
                for batch in trading_strategy['分批买入']:
                    st.write(f"- 第{batch['批次']}批: {batch['比例']}%仓位，条件: {batch['条件']}")
            
            col1, col2 = st.columns(2)
            with col1:
                if trading_strategy.get('止损位') and trading_strategy['止损位'] > 0:
                    st.write(f"**止损位:** {trading_strategy['止损位']:.3f}")
                if trading_strategy.get('目标位') and trading_strategy['目标位'] > 0:
                    st.write(f"**目标位:** {trading_strategy['目标位']:.3f}")
            
            with col2:
                if trading_strategy.get('加仓条件'):
                    st.write(f"**加仓条件:** {trading_strategy['加仓条件']}")
                if trading_strategy.get('减仓条件'):
                    st.write(f"**减仓条件:** {trading_strategy['减仓条件']}")
            
            if trading_strategy.get('风险提示'):
                st.info(f"💡 {trading_strategy['风险提示']}")
    
    # 基金基本信息
    with st.expander("ℹ️ 基金基本信息"):
        if fund_info:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**基本信息:**")
                st.write(f"- 基金代码: {fund_info.get('基金代码', '未知')}")
                st.write(f"- 基金名称: {fund_info.get('基金名称', '未知')}")
                st.write(f"- 基金类型: {fund_info.get('基金类型', '未知')}")
                st.write(f"- 行业标签: {result.get('行业标签', fund_info.get('投资类型', '未知'))}")
                st.write(f"- 成立时间: {fund_info.get('成立时间', '未知')}")
                st.write(f"- 最新规模: {fund_info.get('最新规模', '未知')}")
            
            with col2:
                st.write("**管理团队:**")
                st.write(f"- 基金经理: {fund_info.get('基金经理', '未知')}")
                st.write(f"- 基金公司: {fund_info.get('基金公司', '未知')}")
                st.write(f"- 托管银行: {fund_info.get('托管银行', '未知')}")
                st.write(f"- 投资目标: {fund_info.get('投资目标', '暂无')}")
                
                # 投资策略
                if fund_info.get('投资策略'):
                    with st.expander("投资策略详情"):
                        st.write(fund_info.get('投资策略'))
        else:
            st.write("暂无基金基本信息")
    
    # 最近5周净值与30周均线距离分析
    recent_5w_data = latest_data.get('最近5周数据', [])
    if recent_5w_data:
        st.subheader("📊 最近5周净值与30周均线距离")
        df_5w = pd.DataFrame(recent_5w_data)
        st.dataframe(df_5w, use_container_width=True)
        
        # 可视化最近5周的距离变化
        if len(df_5w) > 0:
            chart_df = df_5w[['日期', '距离(%)']]
            chart_df = chart_df.set_index('日期')
            st.line_chart(chart_df, use_container_width=True)
    
    # 分析元数据
    with st.expander("🔍 分析元数据"):
        st.write(f"**分析时间:** {result.get('分析日期', '未知')}")
        st.write(f"**最新数据日期:** {latest_data.get('净值日期', '未知')}")
        st.write(f"**业绩比较基准:** {fund_info.get('业绩比较基准', '暂无')}")
        st.write(f"**数据完整性:** {'完整' if not result.get('错误') else '有缺失'}")

# 主内容区域
if analyze_button and code:
    if not code.strip():
        st.error("请输入有效的代码！")
    else:
        with st.spinner(f"正在分析 {code}，请稍候..."):
            try:
                if analysis_type == "股票分析":
                    if batch_mode:
                        codes = [c.strip() for c in code.split(",") if c.strip()]
                        results = []
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        status_text.text(f"正在并行分析 {len(codes)} 只股票...")

                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(codes), 8)) as executor:
                            future_map = {executor.submit(analyze_stock, c): c for c in codes}
                            for idx, future in enumerate(concurrent.futures.as_completed(future_map)):
                                c = future_map[future]
                                status_text.text(f"处理中 ({idx+1}/{len(codes)}): {c}")
                                try:
                                    r = future.result()
                                except Exception:
                                    r = {"错误信息": f"分析异常", "股票代码": c, "股票名称": ""}

                                error_msg = r.get("错误信息", "")
                                if error_msg == "历史数据不足":
                                    r["分析状态"] = "数据不足"
                                elif error_msg:
                                    r["分析状态"] = "分析失败"
                                else:
                                    r["分析状态"] = "成功"

                                recent_5w = r.get("近五周均线差距", [])
                                gap_trend = ""
                                if len(recent_5w) >= 3:
                                    gaps = [w["gap_pct"] for w in recent_5w]
                                    if gaps[-1] > gaps[0]:
                                        gap_trend = "逐步靠近 ↑" if gaps[-1] < 0 else "逐步远离 ↑"
                                    elif gaps[-1] < gaps[0]:
                                        gap_trend = "逐步远离 ↓" if gaps[-1] < 0 else "逐步靠近 ↓"
                                    else:
                                        gap_trend = "基本持平"

                                strategy = r.get("交易策略", {})
                                row = {
                                    "股票代码": c,
                                    "股票名称": r.get("股票名称", ""),
                                    "投资建议": r.get("投资建议", ""),
                                    "评分": r.get("投资评分", 0),
                                    "建议仓位(%)": r.get("建议仓位(%)", 0),
                                    "当前阶段": r.get("阶段", ""),
                                    "相对强度": r.get("相对强度", 0),
                                    "RSI": r.get("RSI", 50),
                                    "夏普比率": r.get("夏普比率", 0),
                                    "最大回撤(%)": r.get("最大回撤%", 0),
                                    "30周均值": r.get("30周均值", 0),
                                    "最新收盘": r.get("最新收盘", 0),
                                    "是否在30周线上": "是" if r.get("最新收盘", 0) > r.get("30周均值", 0) else "否",
                                    "近5周差距变化趋势": gap_trend,
                                    "止损位": r.get("止损建议", ""),
                                    "目标位": strategy.get("目标位", ""),
                                    "分析状态": r.get("分析状态", "成功"),
                                }
                                results.append(row)
                                progress_bar.progress((idx + 1) / len(codes))

                        status_text.empty()
                        progress_bar.empty()
                        
                        st.success(f"批量分析完成！共分析 {len(results)} 只股票")
                        df = pd.DataFrame(results)
                        if "评分" in df.columns:
                            df = df.sort_values("评分", ascending=False)
                        
                        if invest_amount and invest_amount > 0:
                            weights = df["评分"].clip(lower=0)
                            total_weight = weights.sum()
                            if total_weight > 0:
                                df["目标资金(元)"] = (weights / total_weight * invest_amount).round(2)
                        
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # 批量汇总统计
                        with st.expander("📊 批量汇总统计"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                buy_count = len(df[df["投资建议"].isin(["买入", "谨慎买入"])])
                                st.metric("建议买入/谨慎买入", f"{buy_count}只")
                            with col2:
                                avg_score = df["评分"].mean()
                                st.metric("平均评分", f"{avg_score:.1f}")
                            with col3:
                                above_ma30 = (df["是否在30周线上"] == "是").sum()
                                st.metric("在30周线上方", f"{above_ma30}只")
                    else:
                        result = analyze_stock(code.strip())
                        if "错误" in result:
                            st.error(f"分析失败: {result['错误']}")
                        else:
                            st.success("分析完成！")
                            display_stock_analysis(result)
                            
                            with st.spinner("正在执行策略历史回测..."):
                                backtest = backtest_stock_strategy(code.strip())
                                if "错误" in backtest:
                                    st.info(f"回测说明: {backtest['错误']}")
                                else:
                                    st.subheader("📈 策略历史回测")
                                    summary = backtest.get("回测概要", {})
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("回测周期", f"{summary.get('总交易周数', 0)}周")
                                    with col2:
                                        buy_win = summary.get('买入信号8周胜率', 0)
                                        st.metric("买入信号8周胜率", f"{buy_win:.1f}%" if not pd.isna(buy_win) else "N/A")
                                    with col3:
                                        buy_avg = summary.get('买入信号8周平均收益%', 0)
                                        st.metric("买入信号8周均值", f"{buy_avg:+.2f}%" if not pd.isna(buy_avg) else "N/A")
                                    with col4:
                                        sell_acc = summary.get('卖出信号8周准确率', 0)
                                        st.metric("卖出信号8周准确率", f"{sell_acc:.1f}%" if sell_acc else "N/A")
                                    
                                    with st.expander("📊 按评分分组的回测表现"):
                                        st.write("不同评分区间的信号表现：")
                                        for label in ["0-20", "20-40", "40-60", "60-80", "80-100"]:
                                            cnt = summary.get(f'评分{label}信号数', 0)
                                            if cnt:
                                                wr = summary.get(f'评分{label}8周胜率', 0)
                                                avg_ret = summary.get(f'评分{label}8周平均收益%', 0)
                                                st.write(f"- 评分{label}: {cnt}次信号, 8周胜率{wr:.1f}%, 平均收益{avg_ret:+.2f}%")
                else:
                    if batch_mode:
                        codes = [c.strip() for c in code.split(",") if c.strip()]
                        results = []
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        status_text.text(f"正在分析 {len(codes)} 只基金...")

                        for idx, c in enumerate(codes):
                            status_text.text(f"处理中 ({idx+1}/{len(codes)}): {c}")
                            try:
                                r = analyze_fund_enhanced(c)
                            except Exception:
                                results.append({"基金代码": c, "错误": f"分析异常"})
                                progress_bar.progress((idx + 1) / len(codes))
                                continue

                            if "错误" in r:
                                results.append({"基金代码": c, "错误": r.get("错误", "")})
                                progress_bar.progress((idx + 1) / len(codes))
                                continue

                            fund_info = r.get("基金基本信息", {})
                            trend = r.get("趋势分析", {})
                            rs_analysis = r.get("相对强度分析", {})
                            risk_analysis = r.get("风险评估", {})
                            latest = r.get("最新数据", {})
                            advice = r.get("投资建议", {})
                            s_advice = r.get("交易策略", {})
                            rs_scores = rs_analysis.get("rs_scores", {})
                            latest_nav = latest.get("单位净值", 0)
                            latest_ma30 = latest.get("30周均线", 0)
                            above_ma30 = latest_ma30 > 0 and latest_nav > latest_ma30
                            nav_vs_ma30 = ((latest_nav - latest_ma30) / latest_ma30 * 100) if latest_ma30 > 0 else 0

                            recent_5w_data = latest.get("最近5周数据", [])
                            recent_5w_distances = []
                            for week in recent_5w_data:
                                recent_5w_distances.append(f"{week['日期']}: {week['距离(%)']}%")

                            row = {
                                "基金代码": fund_info.get("基金代码", c),
                                "基金名称": fund_info.get("基金名称", ""),
                                "交易策略": s_advice,
                                "行业": r.get("行业标签", fund_info.get("投资类型", "")),
                                "当前阶段": trend.get("stage", 0),
                                "阶段置信度(%)": trend.get("confidence", 0) * 100,
                                "12周相对强度": rs_scores.get("12周", 0),
                                "夏普比率": risk_analysis.get("sharpe_ratio", 0),
                                "最大回撤(%)": risk_analysis.get("max_drawdown", 0),
                                "30周均线": latest_ma30,
                                "最新净值": latest_nav,
                                "与30周均线差距(%)": round(nav_vs_ma30, 2),
                                "是否在30周线上": "是" if above_ma30 else "否",
                                "最近5周差距变化": "\n".join(recent_5w_distances) if recent_5w_distances else "无数据",
                                "建议操作": advice.get("建议操作", ""),
                                "建议仓位(%)": advice.get("建议仓位(%)", 0),
                                "评分": advice.get("评分", 0),
                            }
                            results.append(row)
                            progress_bar.progress((idx + 1) / len(codes))

                        status_text.empty()
                        progress_bar.empty()
                        if results:
                            st.success("批量分析完成")
                            df = pd.DataFrame(results)
                            if "评分" in df.columns:
                                df = df.sort_values("评分", ascending=False)
                            if invest_amount and invest_amount > 0 and "建议仓位(%)" in df.columns and "评分" in df.columns:
                                weights = df["建议仓位(%)"].clip(lower=0) * df["评分"].clip(lower=0)
                                total_weight = weights.sum()
                                if total_weight > 0:
                                    df["目标资金(元)"] = (weights / total_weight * invest_amount).round(2)
                            st.dataframe(df, use_container_width=True)
                    else:
                        result = analyze_fund_enhanced(code.strip())
                        print(result)
                        if "错误" in result:
                            st.error(f"分析失败: {result['错误']}")
                        else:
                            st.success("分析完成！")
                            display_fund_analysis(result)
                            backtest = backtest_fund_strategy(code.strip())
                            if "错误" in backtest:
                                st.info(f"回测未执行: {backtest['错误']}")
                            else:
                                st.subheader("📈 策略回测（周频）")
                                summary = backtest.get("回测概要", {})
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("年化收益率", f"{summary.get('年化收益率(%)', 0):.2f}%")
                                with col2:
                                    st.metric("最大回撤", f"{summary.get('最大回撤(%)', 0):.2f}%")
                                with col3:
                                    st.metric("胜率", f"{summary.get('胜率(%)', 0):.2f}%")
                                with col4:
                                    st.metric("交易次数", summary.get("交易次数", 0))
                                equity_df = backtest.get("净值曲线")
                                if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
                                    st.line_chart(equity_df)
            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")
                st.info("请检查代码是否正确，或稍后重试")

elif analyze_button and not code:
    st.warning("请先输入要分析的代码！")

else:
    # 显示欢迎页面
    display_welcome()