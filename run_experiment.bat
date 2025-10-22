@echo off
echo 🚀 GeoQuantization Experiment Runner (Windows)
echo ================================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM 设置环境变量
set PYTHONPATH=%CD%

REM 运行实验
echo 🧪 Starting experiment...
python run_experiment.py --setup --model facebook/opt-125m --samples 200

if errorlevel 1 (
    echo ❌ Experiment failed
    pause
    exit /b 1
)

echo ✅ Experiment completed successfully!
echo 📁 Check results in prm_outputs/ directory
pause
