#!/usr/bin/env python3
"""
HuggingFace镜像配置脚本
"""

import os
import sys
import requests
import json
from pathlib import Path

def test_mirror_connection():
    """测试镜像连接"""
    print("测试HuggingFace镜像连接...")
    
    mirrors = [
        "https://hf-mirror.com",
        "https://huggingface.co",
        "https://hf.co"
    ]
    
    for mirror in mirrors:
        try:
            response = requests.get(f"{mirror}/api/models", timeout=10)
            if response.status_code == 200:
                print(f"✅ {mirror}: 连接成功")
                return mirror
            else:
                print(f"⚠️  {mirror}: 状态码 {response.status_code}")
        except Exception as e:
            print(f"❌ {mirror}: {e}")
    
    return None

def setup_huggingface_mirror():
    """设置HuggingFace镜像"""
    print("配置HuggingFace镜像...")
    
    # 测试镜像连接
    best_mirror = test_mirror_connection()
    
    if not best_mirror:
        print("❌ 所有镜像都无法连接，使用默认配置")
        best_mirror = "https://hf-mirror.com"
    
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = best_mirror
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    # 设置缓存目录
    cache_dir = os.path.join(os.getcwd(), 'hf_cache')
    os.environ['HF_HOME'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"✅ 已设置HuggingFace镜像: {best_mirror}")
    print(f"✅ 缓存目录: {cache_dir}")
    
    return best_mirror

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
        "endpoint": os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'),
        "local_files_only": False,
        "use_auth_token": False,
        "cache_dir": os.environ.get('HF_HOME', './hf_cache')
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ 配置文件已创建: {config_file}")

def test_model_download():
    """测试模型下载"""
    print("测试模型下载...")
    
    try:
        from transformers import AutoTokenizer
        
        # 测试下载小模型
        model_name = "facebook/opt-125m"
        print(f"正在下载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ 模型下载成功!")
        
        # 测试分词
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ 分词测试成功: {tokens.input_ids.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        return False

def test_dataset_download():
    """测试数据集下载"""
    print("测试数据集下载...")
    
    try:
        from datasets import load_dataset
        
        # 测试下载小数据集
        dataset_name = "wikitext"
        subset = "wikitext-2-raw-v1"
        print(f"正在下载数据集: {dataset_name}/{subset}")
        
        dataset = load_dataset(dataset_name, subset, split='train[:10]')
        print(f"✅ 数据集下载成功! 样本数: {len(dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集下载失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 HuggingFace镜像配置工具")
    print("=" * 50)
    
    # 设置镜像
    mirror = setup_huggingface_mirror()
    
    # 创建配置文件
    create_huggingface_config()
    
    # 测试模型下载
    model_ok = test_model_download()
    
    # 测试数据集下载
    dataset_ok = test_dataset_download()
    
    print("\n" + "=" * 50)
    
    if model_ok and dataset_ok:
        print("✅ 镜像配置成功!")
        print(f"镜像地址: {mirror}")
        print("现在可以运行实验:")
        print("  python prm_experiment.py")
    else:
        print("❌ 镜像配置失败")
        print("请检查网络连接或尝试其他镜像")
    
    return 0 if (model_ok and dataset_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
