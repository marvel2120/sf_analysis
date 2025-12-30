"""
智能投资分析系统 - 配置文件
包含应用配置、常量定义和默认参数
"""

import os

# 应用配置
APP_NAME = "智能投资分析系统"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "基于Streamlit的股票基金分析Web应用"

# Streamlit配置
STREAMLIT_CONFIG = {
    'page_title': APP_NAME,
    'page_icon': '📈',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# 默认服务器配置
SERVER_CONFIG = {
    'host': 'localhost',
    'port': 8501,
    'headless': True
}

# 分析参数配置
ANALYSIS_CONFIG = {
    # 股票分析参数
    'stock': {
        'relative_strength_period': 12,  # 相对强弱计算周期（周）
        'breakout_period': 12,          # 突破检测周期（周）
        'volume_threshold': 1.5,          # 量能阈值倍数
        'min_data_points': 50,           # 最小数据点要求
    },
    
    # 基金分析参数
    'fund': {
        'benchmark_index': 'sh000300',   # 基准指数
        'analysis_years': 3,             # 分析年限
        'min_weeks_for_stage': 8,        # 趋势判断最小周数
        'rs_lookback_weeks': 26,         # 相对强弱回看周数
        'risk_free_rate': 0.03,          # 无风险利率（年化）
    }
}

# 投资建议映射
ADVICE_MAPPING = {
    '强烈买入': {'class': 'buy-advice', 'color': '#28a745', 'icon': '🚀'},
    '买入': {'class': 'buy-advice', 'color': '#28a745', 'icon': '👍'},
    '观望': {'class': 'hold-advice', 'color': '#ffc107', 'icon': '⏸️'},
    '卖出': {'class': 'sell-advice', 'color': '#dc3545', 'icon': '👎'},
    '强烈卖出': {'class': 'sell-advice', 'color': '#dc3545', 'icon': '⚠️'},
}

# 趋势阶段定义
STAGE_DEFINITIONS = {
    '上升趋势': {'color': '#28a745', 'description': '价格持续上涨，建议关注'},
    '下降趋势': {'color': '#dc3545', 'description': '价格持续下跌，建议谨慎'},
    '震荡整理': {'color': '#6c757d', 'description': '价格横盘震荡，等待方向选择'},
}

# 数据获取配置
DATA_CONFIG = {
    'retry_times': 3,                    # 重试次数
    'timeout': 30,                       # 超时时间（秒）
    'cache_hours': 6,                    # 缓存时间（小时）
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'investment_analysis.log'
}

# 错误消息
ERROR_MESSAGES = {
    'network_error': '网络连接失败，请检查网络后重试',
    'data_not_found': '未找到相关数据，请检查代码是否正确',
    'analysis_failed': '分析失败，请稍后重试',
    'invalid_code': '代码格式不正确，请输入有效的股票或基金代码',
    'insufficient_data': '数据量不足，无法完成分析',
}

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 创建必要的目录
for directory in [CACHE_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# 开发模式配置
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

if DEBUG_MODE:
    LOGGING_CONFIG['level'] = 'DEBUG'
    DATA_CONFIG['cache_hours'] = 0  # 开发模式下禁用缓存