#!/bin/bash

echo "🚀 GeoQuantization Experiment Runner (Linux/Mac)"
echo "================================================="

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# 设置环境变量
export PYTHONPATH=$(pwd)

# 运行实验
echo "🧪 Starting experiment..."
python3 run_experiment.py --setup --model facebook/opt-125m --samples 200

if [ $? -eq 0 ]; then
    echo "✅ Experiment completed successfully!"
    echo "📁 Check results in prm_outputs/ directory"
else
    echo "❌ Experiment failed"
    exit 1
fi
