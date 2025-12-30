#!/usr/bin/env python3
"""
智能投资分析系统 - 启动脚本
提供命令行参数和错误处理
"""

import subprocess
import sys
import os
import argparse

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        sys.exit(1)
    print(f"✅ Python版本检查通过: {sys.version}")

def check_dependencies():
    """检查必需的依赖包"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'akshare',
        'plotly',
        'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install streamlit plotly")
        sys.exit(1)
    
    print("✅ 依赖包检查通过")

def start_streamlit(port=8501, host='localhost', debug=False):
    """启动Streamlit应用"""
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'streamlit_app.py',
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'true'
    ]
    
    if debug:
        cmd.extend(['--logger.level', 'debug'])
    
    print(f"🚀 启动Streamlit应用...")
    print(f"📍 访问地址: http://{host}:{port}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
        sys.exit(0)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='智能投资分析系统 - Web应用启动器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run_app.py                    # 默认启动
    python run_app.py --port 8080        # 自定义端口
    python run_app.py --host 0.0.0.0     # 允许外部访问
    python run_app.py --debug            # 调试模式
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8501,
        help='服务端口 (默认: 8501)'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='服务地址 (默认: localhost)'
    )
    
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='启用调试模式'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='跳过环境检查'
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("🎯 智能投资分析系统")
    print("=" * 50)
    
    # 环境检查
    if not args.skip_checks:
        check_python_version()
        check_dependencies()
    
    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 启动应用
    start_streamlit(args.port, args.host, args.debug)

if __name__ == '__main__':
    main()