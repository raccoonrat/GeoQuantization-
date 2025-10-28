#!/usr/bin/env python3
"""
WSL2环境检查脚本
"""

import os
import sys
import psutil
import platform
import subprocess

def check_system_info():
    """检查系统信息"""
    print("系统信息:")
    print(f"  平台: {platform.platform()}")
    print(f"  Python版本: {sys.version}")
    print(f"  架构: {platform.architecture()}")
    
    # 检查WSL2
    try:
        with open('/proc/version', 'r') as f:
            version = f.read()
            if 'microsoft' in version.lower():
                print("  ✅ 检测到WSL2环境")
            else:
                print("  ⚠️  可能不在WSL2环境中")
    except:
        print("  ❌ 无法检测WSL2环境")

def check_memory():
    """检查内存使用"""
    print("\n内存信息:")
    memory = psutil.virtual_memory()
    print(f"  总内存: {memory.total // (1024**3)}GB")
    print(f"  可用内存: {memory.available // (1024**3)}GB")
    print(f"  使用率: {memory.percent}%")
    
    if memory.percent > 80:
        print("  ⚠️  内存使用率过高，可能导致WSL2终止")
    else:
        print("  ✅ 内存使用率正常")

def check_disk_space():
    """检查磁盘空间"""
    print("\n磁盘空间:")
    disk = psutil.disk_usage('/')
    print(f"  总空间: {disk.total // (1024**3)}GB")
    print(f"  可用空间: {disk.free // (1024**3)}GB")
    print(f"  使用率: {(disk.used / disk.total) * 100:.1f}%")
    
    if disk.free < 2 * (1024**3):  # 小于2GB
        print("  ⚠️  磁盘空间不足")
    else:
        print("  ✅ 磁盘空间充足")

def check_python_packages():
    """检查Python包"""
    print("\nPython包检查:")
    
    required_packages = [
        'torch', 'transformers', 'numpy', 'scikit-learn',
        'matplotlib', 'pandas', 'tqdm', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少的包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def check_environment_variables():
    """检查环境变量"""
    print("\n环境变量:")
    
    important_vars = [
        'HTTP_PROXY', 'HTTPS_PROXY', 'SOCKS_PROXY', 'ALL_PROXY',
        'HF_ENDPOINT', 'PYTHONPATH', 'PYTHONIOENCODING'
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}: {value}")
        else:
            print(f"  {var}: 未设置")

def check_network():
    """检查网络连接"""
    print("\n网络连接测试:")
    
    test_urls = [
        'https://huggingface.co',
        'https://hf-mirror.com',
        'https://www.google.com'
    ]
    
    try:
        import requests
        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                print(f"  ✅ {url}: {response.status_code}")
            except Exception as e:
                print(f"  ❌ {url}: {e}")
    except ImportError:
        print("  ❌ requests包未安装")

def main():
    """主函数"""
    print("🔍 WSL2环境检查")
    print("=" * 50)
    
    # 检查系统信息
    check_system_info()
    
    # 检查内存
    check_memory()
    
    # 检查磁盘空间
    check_disk_space()
    
    # 检查Python包
    packages_ok = check_python_packages()
    
    # 检查环境变量
    check_environment_variables()
    
    # 检查网络
    check_network()
    
    print("\n" + "=" * 50)
    
    if packages_ok:
        print("✅ 环境检查通过，可以运行实验")
        print("\n推荐运行:")
        print("  python run_wsl2.py")
    else:
        print("❌ 环境检查失败，请先安装缺少的包")
        print("\n运行:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
