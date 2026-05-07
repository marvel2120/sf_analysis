"""
智能投资分析系统 - 配置文件
包含应用配置、常量定义和默认参数
"""

import os

# 应用配置
APP_NAME = "智能投资分析系统"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "基于Streamlit的基金股票分析Web应用（中短线优化版）"

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
    'port': 8881,
    'headless': True
}

# DeepSeek 配置（用户可在界面或环境变量中设置）
DEEPSEEK_CONFIG = {
    'api_key': os.environ.get("DEEPSEEK_API_KEY", ""),
    'model': 'deepseek-chat',
    'timeout': 15,
    'enabled': True  # 设为False可完全禁用
}

# ML 模型配置
ML_CONFIG = {
    'enabled': True,
    'min_samples': 80,
    'ml_weight': 0.3,        # ML在融合中的权重
    'retrain_interval_days': 7  # 自动重训练间隔
}

# 分析参数配置
ANALYSIS_CONFIG = {
    # 基金分析参数（中短线优化）
    'fund': {
        'benchmark_index': 'sh000300',
        'analysis_years': 3,
        'primary_ma': 20,           # 主均线周期（中短线从30改为20周）
        'secondary_ma': 10,         # 辅均线周期
        'tertiary_ma': 30,          # 长期均线参考
        'rsi_period': 14,
        'risk_free_rate': 0.02,
        'max_position_pct': 70,     # 最大仓位
        'enable_deepseek': True,
        'enable_ml': True,
    },

    # 股票分析参数
    'stock': {
        'relative_strength_period': 12,
        'breakout_period': 12,
        'volume_threshold': 1.5,
        'min_data_points': 50,
    },

    # 回测参数
    'backtest': {
        'min_history_weeks': 60,
        'forward_weeks': [4, 8, 12],  # 评估未来N周表现
        'position_threshold': 0.05,    # 调仓敏感度
    }
}

# 市场状态对应仓位限制（中短线优化）
MARKET_REGIME_CONFIG = {
    'strong_bull': {'max_position': 80, 'description': '强势牛市，积极配置'},
    'bull': {'max_position': 70, 'description': '温和牛市，正常配置'},
    'sideways': {'max_position': 50, 'description': '震荡市场，波段操作'},
    'volatile_sideways': {'max_position': 35, 'description': '高波动震荡，谨慎波段'},
    'volatile_bull': {'max_position': 45, 'description': '高波动牛市，控制仓位'},
    'bear': {'max_position': 25, 'description': '熊市，防守为主'},
    'strong_bear': {'max_position': 10, 'description': '深度熊市，空仓观望'},
    'unknown': {'max_position': 30, 'description': '信号不明确，轻仓试探'},
}

# 趋势阶段定义
STAGE_DEFINITIONS = {
    '上升趋势': {'color': '#28a745', 'description': '价格持续上涨，建议关注'},
    '下降趋势': {'color': '#dc3545', 'description': '价格持续下跌，建议谨慎'},
    '震荡整理': {'color': '#6c757d', 'description': '价格横盘震荡，等待方向选择'},
}

# 数据获取配置
DATA_CONFIG = {
    'retry_times': 3,
    'timeout': 30,
    'cache_hours': 6,
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'investment_analysis.log'
}

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 创建必要的目录
for directory in [CACHE_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
