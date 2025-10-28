#!/usr/bin/env python3
"""
安全实验启动脚本 - 避免WSL崩溃
"""

import os
import sys
import subprocess
import psutil
import signal
import time
from pathlib import Path

def setup_safe_environment():
    """设置安全环境"""
    print("🛡️ 设置安全环境...")
    
    # 强制CPU模式
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用GPU
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # 设置HuggingFace镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    # 设置缓存目录
    cache_dir = os.path.join(os.getcwd(), 'hf_cache')
    os.environ['HF_HOME'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"✅ 已禁用GPU，强制CPU模式")
    print(f"✅ 已设置HuggingFace镜像")
    print(f"✅ 缓存目录: {cache_dir}")

def check_memory_before_start():
    """启动前检查内存"""
    print("检查内存使用情况...")
    
    memory = psutil.virtual_memory()
    print(f"内存使用: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
    
    if memory.percent > 80:
        print("⚠️  内存使用率过高，建议释放内存后再运行")
        choice = input("是否继续运行? (y/n): ").lower().strip()
        if choice != 'y':
            print("已取消运行")
            return False
    
    return True

def monitor_memory_during_run(process):
    """运行期间监控内存"""
    print("开始监控内存使用...")
    
    start_time = time.time()
    max_runtime = 300  # 5分钟超时
    
    while process.poll() is None:
        # 检查运行时间
        if time.time() - start_time > max_runtime:
            print("⏰ 运行时间超时，终止进程")
            process.terminate()
            return False
        
        # 检查内存使用
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            print(f"⚠️  内存使用率过高: {memory.percent}%")
            print("终止进程以避免WSL崩溃")
            process.terminate()
            return False
        
        time.sleep(10)  # 每10秒检查一次
    
    return True

def run_safe_experiment():
    """运行安全实验"""
    print("🧪 运行安全实验...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        env['PYTHONUNBUFFERED'] = '1'
        
        # 运行安全实验
        process = subprocess.Popen([
            sys.executable, "safe_experiment.py"
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 监控内存
        if not monitor_memory_during_run(process):
            return False
        
        # 等待进程完成
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("✅ 安全实验完成!")
            print("STDOUT:", stdout)
            return True
        else:
            print(f"❌ 安全实验失败: {process.returncode}")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行安全实验失败: {e}")
        return False

def run_with_safe_config():
    """使用安全配置运行主实验"""
    print("🧪 使用安全配置运行主实验...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        env['PYTHONUNBUFFERED'] = '1'
        
        # 运行主实验（使用安全配置）
        process = subprocess.Popen([
            sys.executable, "prm_experiment.py"
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 监控内存
        if not monitor_memory_during_run(process):
            return False
        
        # 等待进程完成
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print("✅ 主实验完成!")
            print("STDOUT:", stdout)
            return True
        else:
            print(f"❌ 主实验失败: {process.returncode}")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return False
            
    except Exception as e:
        print(f"❌ 运行主实验失败: {e}")
        return False

def main():
    """主函数"""
    print("🛡️ 安全实验启动器")
    print("=" * 50)
    
    # 设置安全环境
    setup_safe_environment()
    
    # 检查内存
    if not check_memory_before_start():
        return 1
    
    # 创建输出目录
    output_dir = "prm_outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 选择运行方式
    print("\n选择运行方式:")
    print("1. 安全实验（推荐）")
    print("2. 主实验（使用安全配置）")
    print("3. 仅检查环境")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        print("运行安全实验...")
        success = run_safe_experiment()
    elif choice == "2":
        print("运行主实验（使用安全配置）...")
        success = run_with_safe_config()
    elif choice == "3":
        print("仅检查环境，不运行实验")
        success = True
    else:
        print("无效选择，运行安全实验...")
        success = run_safe_experiment()
    
    if success:
        print("\n✅ 实验完成!")
        print(f"📁 结果保存在: {output_dir}")
        
        # 显示输出文件
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.glob("*"))
            if files:
                print("\n📄 生成的文件:")
                for f in files:
                    print(f"   - {f.name}")
    else:
        print("\n❌ 实验失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n致命错误: {e}")
        sys.exit(1)
