import streamlit as st
import sys
import os

# 添加当前目录到路径，确保可以导入 advisor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advisor_stock import analyze_stock
from advisor_fund import analyze_fund_enhanced

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
    if analysis_type == "股票分析":
        code = st.text_input("请输入股票代码:", placeholder="例如: 000001")
        st.caption("支持A股股票代码，如: 000001, 600000, 300001")
    else:
        code = st.text_input("请输入基金代码:", placeholder="例如: 000001")
        st.caption("支持开放式基金代码，如: 000001, 110011")
    
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
    """显示股票分析结果"""
    st.header(f"📊 {result.get('股票代码', 'Unknown')} {result.get('股票名称', '')} 股票分析报告")
    
    # 基本信息卡片 - 适配实际的返回值格式
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 根据投资评分判断趋势阶段
        score = result.get('投资评分', 60)
        if score >= 80:
            stage = "强势上升"
        elif score >= 70:
            stage = "温和上升"
        elif score >= 60:
            stage = "震荡整理"
        elif score >= 40:
            stage = "弱势整理"
        else:
            stage = "下降趋势"
        st.metric("当前趋势", stage)
    
    with col2:
        rs_value = result.get('相对强度', 0)
        st.metric("相对强度", f"{rs_value:.3f}")
    
    with col3:
        breakout = result.get('是否突破', False)
        st.metric("突破信号", "是" if breakout else "否")
    
    with col4:
        volume_ok = result.get('量能是否放大', True)
        st.metric("量能配合", "良好" if volume_ok else "不足")
    
    # 价格信息卡片
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
    
    # 投资建议
    advice = result.get('投资建议', '观望')
    advice_class = get_advice_class(advice)
    score = result.get('投资评分', 60)
    
    st.subheader("💡 投资建议")
    advice_html = f"""
    <div class="advice-box {advice_class}">
        <h3>{advice} (评分: {score}/100)</h3>
        <p><strong>投资说明:</strong> {result.get('投资说明', '暂无详细说明')}</p>
        {f'<p><strong>止损建议:</strong> {result.get("止损建议", 0):.2f}</p>' if result.get('止损建议') else ''}
    </div>
    """
    st.markdown(advice_html, unsafe_allow_html=True)
    
    # 详细分析
    with st.expander("📈 技术分析详情"):
        st.write(f"**分析日期:** {result.get('分析日期', '未知')}")
        st.write(f"**当前阶段:** 第{result.get('阶段', 1)}阶段")
        
        # 错误信息显示
        if result.get('错误信息'):
            st.error(f"⚠️ 分析警告: {result.get('错误信息')}")
        
        # 技术指标说明
        col1, col2 = st.columns(2)
        with col1:
            st.write("**关键指标:**")
            st.write(f"- 相对强度: {rs_value:.4f}")
            st.write(f"- 突破状态: {'已突破' if breakout else '未突破'}")
            st.write(f"- 量能状态: {'放大' if volume_ok else '正常'}")
        
        with col2:
            st.write("**关键价位:**")
            st.write(f"- 支撑位: {support:.2f}")
            st.write(f"- 阻力位: {resistance:.2f}")
            if result.get('止损建议'):
                st.write(f"- 止损位: {result.get('止损建议'):.2f}")

def display_fund_analysis(result):
    """显示基金分析结果"""
    # 获取基金基本信息
    fund_info = result.get('基金基本信息', {})
    fund_code = fund_info.get('基金代码', 'Unknown') if fund_info else 'Unknown'
    fund_name = fund_info.get('基金名称', '') if fund_info else ''
    
    st.header(f"💰 {fund_code} {fund_name} 基金分析报告")
    
    # 获取最新数据
    latest_data = result.get('最新数据', {})
    trend_analysis = result.get('趋势分析', {})
    rs_analysis = result.get('相对强度分析', {})
    risk_analysis = result.get('风险评估', {})
    advice_result = result.get('投资建议', {})
    
    # 基本信息卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stage = trend_analysis.get('stage', '未知')
        stage_confidence = trend_analysis.get('confidence', 0) * 100
        st.metric("当前趋势", f"{stage} ({stage_confidence:.0f}%)")
    
    with col2:
        rs_scores = rs_analysis.get('rs_scores', {})
        rs_12w = rs_scores.get('12周', 0.0)
        st.metric("3个月相对强度", f"{rs_12w:.2f}")
    
    with col3:
        latest_nav = latest_data.get('单位净值', 0)
        st.metric("最新净值", f"{latest_nav:.4f}")
    
    with col4:
        sharpe_ratio = risk_analysis.get('sharpe_ratio', 0)
        st.metric("夏普比率", f"{sharpe_ratio:.2f}")
    
    # 投资建议
    advice = advice_result.get('建议操作', '观望')
    advice_score = advice_result.get('评分', 60)
    advice_desc = advice_result.get('建议说明', '')
    advice_class = get_advice_class(advice)
    
    st.subheader("💡 投资建议")
    advice_html = f"""
    <div class="advice-box {advice_class}">
        <h3>{advice} (评分: {advice_score}/100)</h3>
        <p><strong>建议说明:</strong> {advice_desc}</p>
        <p><strong>建议仓位:</strong> {advice_result.get('建议仓位(%)', 0)}%</p>
    </div>
    """
    st.markdown(advice_html, unsafe_allow_html=True)
    
    # 详细指标
    with st.expander("📊 详细指标"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**相对强度表现:**")
            rs_scores = rs_analysis.get('rs_scores', {})
            st.write(f"- 3个月相对强度: {rs_scores.get('12周', 0):.3f}")
            st.write(f"- 6个月相对强度: {rs_scores.get('26周', 0):.3f}")
            st.write(f"- 12个月相对强度: {rs_scores.get('52周', 0):.3f}")
            st.write(f"- 风险调整超额收益: {rs_analysis.get('risk_adjusted_rs', 0):.3f}")
        
        with col2:
            st.write("**风险评估:**")
            st.write(f"- 最大回撤: {risk_analysis.get('max_drawdown', 0):.2f}%")
            st.write(f"- 下行波动率: {latest_data.get('下行波动率(%)', 0):.2f}%")
            st.write(f"- 夏普比率: {sharpe_ratio:.3f}")
            st.write(f"- 30周均线: {latest_data.get('30周均线', 0):.4f}")
    
    # 基金基本信息
    with st.expander("ℹ️ 基金基本信息"):
        if fund_info:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**基金名称:** {fund_info.get('基金简称', '未知')}")
                st.write(f"**基金类型:** {fund_info.get('基金类型', '未知')}")
                st.write(f"**成立日期:** {fund_info.get('成立日期', '未知')}")
                st.write(f"**基金规模:** {fund_info.get('基金规模', '未知')}")
            with col2:
                st.write(f"**基金经理:** {fund_info.get('基金经理', '未知')}")
                st.write(f"**管理公司:** {fund_info.get('基金公司', '未知')}")
                st.write(f"**托管银行:** {fund_info.get('托管银行', '未知')}")
                st.write(f"**管理费率:** {fund_info.get('管理费率', '未知')}")
        else:
            st.write("暂无基金基本信息")
    
    # 分析信息
    with st.expander("🔍 分析详情"):
        st.write(f"**分析时间:** {result.get('分析日期', '未知')}")
        st.write(f"**最新数据日期:** {latest_data.get('净值日期', '未知')}")
        st.write(f"**数据完整性:** {'完整' if not result.get('错误') else '有缺失'}")
        st.write(f"- 近1年收益: {result.get('近1年收益', 0):.1f}%")
        st.write(f"- 近2年收益: {result.get('近2年收益', 0):.1f}%")
        st.write(f"- 近3年收益: {result.get('近3年收益', 0):.1f}%")
        st.write(f"- 超额收益: {result.get('超额收益', 0):.1f}%")
        
        with col2:
            st.write("**风险评估:**")
            st.write(f"- 波动率: {result.get('波动率', 0):.1f}%")
            st.write(f"- 最大回撤: {result.get('最大回撤', 0):.1f}%")
            st.write(f"- 夏普比率: {result.get('夏普比率', 0):.2f}")
            st.write(f"- 风险评分: {result.get('风险评分', 0):.1f}")
    
    # 基金基本信息
    with st.expander("ℹ️ 基金基本信息"):
        info = result.get('基金信息', {})
        if info:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**基金名称:** {info.get('基金简称', '未知')}")
                st.write(f"**基金类型:** {info.get('基金类型', '未知')}")
                st.write(f"**成立日期:** {info.get('成立日期', '未知')}")
            with col2:
                st.write(f"**基金经理:** {info.get('基金经理', '未知')}")
                st.write(f"**管理公司:** {info.get('基金公司', '未知')}")
                st.write(f"**基金规模:** {info.get('基金规模', '未知')}")
        else:
            st.write("暂无基金基本信息")

# 主内容区域
if analyze_button and code:
    if not code.strip():
        st.error("请输入有效的代码！")
    else:
        with st.spinner(f"正在分析 {code}，请稍候..."):
            try:
                if analysis_type == "股票分析":
                    result = analyze_stock(code.strip())
                else:
                    result = analyze_fund_enhanced(code.strip())
                print(result)
                if "错误" in result:
                    st.error(f"分析失败: {result['错误']}")
                else:
                    # 显示分析结果
                    st.success("分析完成！")
                    
                    if analysis_type == "股票分析":
                        display_stock_analysis(result)
                    else:
                        display_fund_analysis(result)
                        
            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")
                st.info("请检查代码是否正确，或稍后重试")

elif analyze_button and not code:
    st.warning("请先输入要分析的代码！")

else:
    # 显示欢迎页面
    display_welcome()