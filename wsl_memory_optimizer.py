#!/usr/bin/env python3
"""
WSL内存优化脚本
"""

import os
import sys
import psutil
import subprocess
import time

def check_wsl_memory():
    """检查WSL内存使用"""
    print("检查WSL内存使用...")
    
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total // (1024**3)}GB")
    print(f"可用内存: {memory.available // (1024**3)}GB")
    print(f"使用率: {memory.percent}%")
    
    if memory.percent > 80:
        print("⚠️  内存使用率过高，可能导致WSL崩溃")
        return False
    else:
        print("✅ 内存使用率正常")
        return True

def optimize_memory():
    """优化内存使用"""
    print("优化内存使用...")
    
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # 禁用GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print("✅ 已设置内存优化环境变量")

def cleanup_memory():
    """清理内存"""
    print("清理内存...")
    
    try:
        import gc
        gc.collect()
        print("✅ 已清理Python内存")
    except:
        pass
    
    # 尝试清理系统缓存
    try:
        subprocess.run(['sync'], check=False)
        subprocess.run(['echo', '3'], stdout=open('/proc/sys/vm/drop_caches', 'w'), check=False)
        print("✅ 已清理系统缓存")
    except:
        print("⚠️  无法清理系统缓存")

def check_disk_space():
    """检查磁盘空间"""
    print("检查磁盘空间...")
    
    disk = psutil.disk_usage('/')
    print(f"总空间: {disk.total // (1024**3)}GB")
    print(f"可用空间: {disk.free // (1024**3)}GB")
    print(f"使用率: {(disk.used / disk.total) * 100:.1f}%")
    
    if disk.free < 2 * (1024**3):  # 小于2GB
        print("⚠️  磁盘空间不足")
        return False
    else:
        print("✅ 磁盘空间充足")
        return True

def suggest_wsl_config():
    """建议WSL配置"""
    print("\n建议的WSL配置:")
    print("在Windows中创建或编辑 %USERPROFILE%\\.wslconfig:")
    print("")
    print("[wsl2]")
    print("memory=8GB")
    print("processors=4")
    print("swap=2GB")
    print("")
    print("然后重启WSL:")
    print("wsl --shutdown")
    print("wsl")

def main():
    """主函数"""
    print("🔧 WSL内存优化工具")
    print("=" * 50)
    
    # 检查内存
    memory_ok = check_wsl_memory()
    
    # 检查磁盘空间
    disk_ok = check_disk_space()
    
    # 优化内存
    optimize_memory()
    
    # 清理内存
    cleanup_memory()
    
    print("\n" + "=" * 50)
    
    if memory_ok and disk_ok:
        print("✅ 系统状态良好，可以运行实验")
        print("推荐运行: python run_safe.py")
    else:
        print("❌ 系统状态不佳，建议优化")
        suggest_wsl_config()
    
    return 0 if (memory_ok and disk_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
