@echo off
REM WSL2安全运行脚本 (Windows批处理)
echo 🐧 WSL2安全实验启动器
echo ==================================

REM 设置编码
set PYTHONIOENCODING=utf-8
set LANG=en_US.UTF-8
set LC_ALL=en_US.UTF-8

REM 设置内存优化
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1

REM 设置代理（如果需要）
if not "%SOCKS_PROXY%"=="" (
    echo 检测到SOCKS代理: %SOCKS_PROXY%
    REM 转换socks5h为socks5
    set NEW_PROXY=%SOCKS_PROXY:socks5h://=socks5://%
    if not "%NEW_PROXY%"=="%SOCKS_PROXY%" (
        echo 转换代理协议: %SOCKS_PROXY% -^> %NEW_PROXY%
        set HTTP_PROXY=%NEW_PROXY%
        set HTTPS_PROXY=%NEW_PROXY%
        set http_proxy=%NEW_PROXY%
        set https_proxy=%NEW_PROXY%
    )
)

REM 设置HuggingFace配置
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_DISABLE_PROGRESS_BARS=1
set HF_ENDPOINT=https://hf-mirror.com

REM 创建输出目录
if not exist prm_outputs mkdir prm_outputs

REM 检查环境
echo 检查WSL2环境...
python check_wsl2.py

if %errorlevel% equ 0 (
    echo ✅ 环境检查通过
    
    echo.
    echo 选择运行方式:
    echo 1. 最小化测试 (推荐)
    echo 2. 完整实验
    echo 3. 仅检查环境
    
    set /p choice=请输入选择 (1-3): 
    
    if "%choice%"=="1" (
        echo 运行最小化测试...
        python minimal_experiment.py
    ) else if "%choice%"=="2" (
        echo 运行完整实验...
        python run_wsl2.py
    ) else if "%choice%"=="3" (
        echo 仅检查环境，不运行实验
    ) else (
        echo 无效选择，运行最小化测试...
        python minimal_experiment.py
    )
) else (
    echo ❌ 环境检查失败
    echo 请先安装依赖: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo 实验完成!
pause
