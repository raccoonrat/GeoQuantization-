#!/usr/bin/env python3
"""
带代理处理的实验启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_proxy_environment():
    """设置代理环境"""
    print("配置代理环境...")
    
    # 检查SOCKS5代理
    socks_proxy = None
    for var in ['SOCKS_PROXY', 'socks_proxy', 'ALL_PROXY', 'all_proxy']:
        value = os.environ.get(var)
        if value and 'socks5' in value.lower():
            socks_proxy = value
            break
    
    if socks_proxy:
        print(f"检测到SOCKS5代理: {socks_proxy}")
        
        # 转换socks5h为socks5
        if 'socks5h://' in socks_proxy:
            new_proxy = socks_proxy.replace('socks5h://', 'socks5://')
            print(f"转换代理协议: {socks_proxy} -> {new_proxy}")
            
            # 设置HTTP代理环境变量
            os.environ['HTTP_PROXY'] = new_proxy
            os.environ['HTTPS_PROXY'] = new_proxy
            os.environ['http_proxy'] = new_proxy
            os.environ['https_proxy'] = new_proxy
    
    # 设置HuggingFace配置
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # 设置镜像
    if not os.environ.get('HF_ENDPOINT'):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("已设置HuggingFace镜像")

def main():
    """主函数"""
    print("🚀 GeoQuantization Experiment Runner (with Proxy Support)")
    print("=" * 60)
    
    # 设置代理环境
    setup_proxy_environment()
    
    # 创建输出目录
    output_dir = "prm_outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    print("🧪 Starting experiment...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        # 运行实验
        result = subprocess.run([
            sys.executable, "prm_experiment.py"
        ], env=env, check=True, capture_output=True, text=True)
        
        print("✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
        # 显示输出文件
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.glob("*"))
            if files:
                print("\n📄 Generated files:")
                for f in files:
                    print(f"   - {f.name}")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
