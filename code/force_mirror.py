#!/usr/bin/env python3
"""
强制使用HuggingFace镜像的配置脚本
"""

import os
import sys
import json
from pathlib import Path

def setup_force_mirror():
    """强制设置镜像配置"""
    print("强制配置HuggingFace镜像...")
    
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    # 设置缓存目录
    cache_dir = os.path.join(os.getcwd(), 'hf_cache')
    os.environ['HF_HOME'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"✅ 环境变量设置完成")
    print(f"   HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    print(f"   缓存目录: {cache_dir}")

def create_huggingface_config():
    """创建HuggingFace配置文件"""
    print("创建HuggingFace配置文件...")
    
    # 创建配置目录
    config_dir = Path.home() / '.cache' / 'huggingface'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建hub配置
    hub_config_dir = config_dir / 'hub'
    hub_config_dir.mkdir(exist_ok=True)
    
    config_file = hub_config_dir / 'config.json'
    
    config = {
        "endpoint": "https://hf-mirror.com",
        "local_files_only": False,
        "use_auth_token": False,
        "cache_dir": os.environ.get('HF_HOME', './hf_cache')
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 配置文件已创建: {config_file}")

def create_local_config():
    """创建本地配置文件"""
    print("创建本地配置文件...")
    
    # 创建本地配置目录
    local_config_dir = Path('./hf_cache/config')
    local_config_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建配置文件
    config_file = local_config_dir / 'hub.json'
    
    config = {
        "endpoint": "https://hf-mirror.com",
        "local_files_only": False,
        "use_auth_token": False
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 本地配置文件已创建: {config_file}")

def test_direct_download():
    """测试直接下载"""
    print("测试直接下载...")
    
    try:
        import requests
        
        # 测试镜像连接
        test_urls = [
            "https://hf-mirror.com/api/models",
            "https://hf-mirror.com/facebook/opt-125m/resolve/main/tokenizer.json"
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    print(f"✅ {url}: 连接成功")
                else:
                    print(f"⚠️  {url}: 状态码 {response.status_code}")
            except Exception as e:
                print(f"❌ {url}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 直接下载测试失败: {e}")
        return False

def test_transformers_with_mirror():
    """测试transformers使用镜像"""
    print("测试transformers使用镜像...")
    
    try:
        # 重新导入transformers以确保使用新的环境变量
        if 'transformers' in sys.modules:
            del sys.modules['transformers']
        
        from transformers import AutoTokenizer
        
        # 测试下载
        model_name = "facebook/opt-125m"
        print(f"正在下载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=False,
            cache_dir=os.environ.get('HF_HOME', './hf_cache')
        )
        
        print("✅ 模型下载成功!")
        
        # 测试分词
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ 分词测试成功: {tokens.input_ids.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ transformers测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 强制HuggingFace镜像配置工具")
    print("=" * 50)
    
    # 设置镜像
    setup_force_mirror()
    
    # 创建配置文件
    create_huggingface_config()
    create_local_config()
    
    # 测试直接下载
    direct_ok = test_direct_download()
    
    # 测试transformers
    transformers_ok = test_transformers_with_mirror()
    
    print("\n" + "=" * 50)
    
    if direct_ok and transformers_ok:
        print("✅ 强制镜像配置成功!")
        print("现在可以运行实验:")
        print("  python prm_experiment.py")
    else:
        print("❌ 强制镜像配置失败")
        print("请检查网络连接或尝试其他解决方案")
    
    return 0 if (direct_ok and transformers_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
