#!/usr/bin/env python3
"""
GeoQuantization Experiment Setup Script
实验环境设置脚本
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def install_requirements():
    """安装依赖包"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    dirs = [
        "prm_outputs",
        "prm_outputs/plots",
        "prm_outputs/results",
        "prm_outputs/logs",
        "data",
        "models"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def create_gitignore():
    """创建.gitignore文件"""
    gitignore_content = """# GeoQuantization Experiment .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Experiment outputs
prm_outputs/
data/
models/
*.log

# OS
.DS_Store
Thumbs.db

# Large files
*.bin
*.safetensors
*.h5
*.pkl
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("📝 Created .gitignore file")

def test_imports():
    """测试关键包导入"""
    print("🧪 Testing imports...")
    try:
        import torch
        import transformers
        import datasets
        import numpy as np
        import matplotlib.pyplot as plt
        import umap
        import sklearn
        print("✅ All key packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """主设置函数"""
    print("🚀 Setting up GeoQuantization Experiment Environment")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create .gitignore
    print("\n📝 Creating .gitignore...")
    create_gitignore()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        sys.exit(1)
    
    # Test imports
    print("\n🧪 Testing installation...")
    if not test_imports():
        print("❌ Setup failed at import testing")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python prm_experiment.py")
    print("2. Check results in prm_outputs/ directory")
    print("3. View plots and CSV files for analysis")

if __name__ == "__main__":
    main()
