#!/usr/bin/env python3
"""
使用HuggingFace镜像的实验启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_mirror_environment():
    """设置镜像环境"""
    print("配置HuggingFace镜像环境...")
    
    # 设置HuggingFace镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    # 设置缓存目录
    cache_dir = os.path.join(os.getcwd(), 'hf_cache')
    os.environ['HF_HOME'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"✅ 已设置HuggingFace镜像: {os.environ['HF_ENDPOINT']}")
    print(f"✅ 缓存目录: {cache_dir}")

def test_connection():
    """测试连接"""
    print("测试镜像连接...")
    
    try:
        import requests
        response = requests.get('https://hf-mirror.com/api/models', timeout=10)
        if response.status_code == 200:
            print("✅ 镜像连接成功")
            return True
        else:
            print(f"⚠️  镜像连接状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 镜像连接失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 GeoQuantization Experiment Runner (with Mirror)")
    print("=" * 60)
    
    # 设置镜像环境
    setup_mirror_environment()
    
    # 测试连接
    if not test_connection():
        print("⚠️  镜像连接测试失败，但继续运行...")
    
    # 创建输出目录
    output_dir = "prm_outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    print("🧪 Starting experiment...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        env['PYTHONUNBUFFERED'] = '1'
        
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
