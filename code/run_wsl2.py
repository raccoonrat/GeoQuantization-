#!/usr/bin/env python3
"""
WSL2专用实验启动脚本
解决WSL2环境下的闪退和内存问题
"""

import os
import sys
import gc
import psutil
import signal
import subprocess
from pathlib import Path

def check_wsl2_environment():
    """检查WSL2环境"""
    print("检查WSL2环境...")
    
    # 检查是否在WSL2中
    try:
        with open('/proc/version', 'r') as f:
            version = f.read()
            if 'microsoft' in version.lower():
                print("✅ 检测到WSL2环境")
            else:
                print("⚠️  可能不在WSL2环境中")
    except:
        print("❌ 无法检测WSL2环境")
    
    # 检查内存使用
    memory = psutil.virtual_memory()
    print(f"内存使用: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
    
    if memory.percent > 80:
        print("⚠️  内存使用率过高，可能导致WSL2终止")
        return False
    
    return True

def setup_memory_optimization():
    """设置内存优化"""
    print("配置内存优化...")
    
    # 设置环境变量优化内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # 设置较小的batch size
    os.environ['BATCH_SIZE'] = '4'
    os.environ['MAX_LENGTH'] = '128'
    
    print("已设置内存优化环境变量")

def setup_encoding():
    """设置编码"""
    print("配置编码设置...")
    
    # 设置UTF-8编码
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    
    # 设置标准输出编码
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

def setup_proxy_for_wsl2():
    """为WSL2设置代理"""
    print("配置WSL2代理设置...")
    
    # 检查代理设置
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
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    # 设置镜像
    if not os.environ.get('HF_ENDPOINT'):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("已设置HuggingFace镜像")

def create_minimal_config():
    """创建最小化配置"""
    print("创建最小化配置...")
    
    config_content = """# WSL2最小化配置
model:
  name: "facebook/opt-125m"
  device: "cpu"  # 强制使用CPU避免GPU内存问题
  max_length: 128
  batch_size: 2

dataset:
  calib_name: "wikitext"
  calib_subset: "wikitext-2-raw-v1"
  calib_samples: 50  # 减少样本数
  eval_samples: 50

experiment:
  seed: 42
  topk_eigenvectors: 20  # 减少特征向量数
  noise_sigmas: [0.0, 1e-4, 1e-3]  # 减少噪声测试点
  repeats: 2  # 减少重复次数
  output_dir: "prm_outputs"

geometry:
  umap:
    n_neighbors: 10
    min_dist: 0.1
  clustering:
    eps: 0.5
    min_samples: 3
  thresholds:
    func_cos_min: 0.7
    func_sparsity_max: 0.5
    sens_cos_max: 0.3
    sens_sparsity_min: 0.9
"""
    
    with open('wsl2_config.yaml', 'w') as f:
        f.write(config_content)
    
    print("已创建WSL2最小化配置文件")

def run_experiment_safely():
    """安全运行实验"""
    print("安全运行实验...")
    
    try:
        # 设置信号处理
        def signal_handler(signum, frame):
            print(f"\n收到信号 {signum}，正在清理...")
            gc.collect()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 运行实验
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        env['PYTHONUNBUFFERED'] = '1'
        
        # 使用最小化配置
        result = subprocess.run([
            sys.executable, "prm_experiment.py"
        ], env=env, check=True, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        
        print("✅ 实验完成!")
        print("STDOUT:", result.stdout)
        
        return 0
        
    except subprocess.TimeoutExpired:
        print("❌ 实验超时（30分钟）")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验失败: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return 1
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return 1
    finally:
        # 清理内存
        gc.collect()

def main():
    """主函数"""
    print("🐧 WSL2专用实验启动器")
    print("=" * 50)
    
    # 检查WSL2环境
    if not check_wsl2_environment():
        print("⚠️  环境检查失败，但继续运行...")
    
    # 设置编码
    setup_encoding()
    
    # 设置内存优化
    setup_memory_optimization()
    
    # 设置代理
    setup_proxy_for_wsl2()
    
    # 创建最小化配置
    create_minimal_config()
    
    # 创建输出目录
    output_dir = "prm_outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    print("\n🧪 开始实验...")
    result = run_experiment_safely()
    
    if result == 0:
        print("\n✅ 实验成功完成!")
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
    
    return result

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n致命错误: {e}")
        sys.exit(1)
