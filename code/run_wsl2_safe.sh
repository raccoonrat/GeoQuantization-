#!/bin/bash

# WSL2安全运行脚本
echo "🐧 WSL2安全实验启动器"
echo "=================================="

# 设置编码
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# 设置内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 设置代理（如果需要）
if [ ! -z "$SOCKS_PROXY" ]; then
    echo "检测到SOCKS代理: $SOCKS_PROXY"
    # 转换socks5h为socks5
    if [[ $SOCKS_PROXY == *"socks5h://"* ]]; then
        NEW_PROXY=${SOCKS_PROXY/socks5h:\/\//socks5:\/\/}
        echo "转换代理协议: $SOCKS_PROXY -> $NEW_PROXY"
        export HTTP_PROXY=$NEW_PROXY
        export HTTPS_PROXY=$NEW_PROXY
        export http_proxy=$NEW_PROXY
        export https_proxy=$NEW_PROXY
    fi
fi

# 设置HuggingFace配置
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_ENDPOINT=https://hf-mirror.com

# 创建输出目录
mkdir -p prm_outputs

# 检查环境
echo "检查WSL2环境..."
python3 check_wsl2.py

if [ $? -eq 0 ]; then
    echo "✅ 环境检查通过"
    
    # 选择运行方式
    echo ""
    echo "选择运行方式:"
    echo "1. 最小化测试 (推荐)"
    echo "2. 完整实验"
    echo "3. 仅检查环境"
    
    read -p "请输入选择 (1-3): " choice
    
    case $choice in
        1)
            echo "运行最小化测试..."
            python3 minimal_experiment.py
            ;;
        2)
            echo "运行完整实验..."
            python3 run_wsl2.py
            ;;
        3)
            echo "仅检查环境，不运行实验"
            ;;
        *)
            echo "无效选择，运行最小化测试..."
            python3 minimal_experiment.py
            ;;
    esac
else
    echo "❌ 环境检查失败"
    echo "请先安装依赖: pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "实验完成!"
