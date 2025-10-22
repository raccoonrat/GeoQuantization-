#!/usr/bin/env python3
"""
测试HuggingFace镜像配置
"""

import os
import sys
import requests

def test_mirror_connection():
    """测试镜像连接"""
    print("测试HuggingFace镜像连接...")
    
    mirrors = [
        "https://hf-mirror.com",
        "https://huggingface.co"
    ]
    
    for mirror in mirrors:
        try:
            print(f"测试 {mirror}...")
            response = requests.get(f"{mirror}/api/models", timeout=10)
            if response.status_code == 200:
                print(f"✅ {mirror}: 连接成功")
                return mirror
            else:
                print(f"⚠️  {mirror}: 状态码 {response.status_code}")
        except Exception as e:
            print(f"❌ {mirror}: {e}")
    
    return None

def test_model_download():
    """测试模型下载"""
    print("\n测试模型下载...")
    
    try:
        # 强制设置镜像环境变量
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '0'
        
        # 设置缓存目录
        cache_dir = os.path.join(os.getcwd(), 'hf_cache')
        os.environ['HF_HOME'] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"镜像地址: {os.environ['HF_ENDPOINT']}")
        print(f"缓存目录: {cache_dir}")
        
        from transformers import AutoTokenizer
        
        # 测试下载小模型
        model_name = "facebook/opt-125m"
        print(f"正在下载模型: {model_name}")
        
        # 使用镜像地址直接下载
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=False,
            cache_dir=cache_dir
        )
        print("✅ 模型下载成功!")
        
        # 测试分词
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ 分词测试成功: {tokens.input_ids.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print("尝试使用备用方法...")
        
        # 尝试使用requests直接下载
        try:
            import requests
            mirror_url = f"https://hf-mirror.com/facebook/opt-125m/resolve/main/tokenizer.json"
            response = requests.get(mirror_url, timeout=30)
            if response.status_code == 200:
                print("✅ 镜像连接测试成功!")
                return True
            else:
                print(f"❌ 镜像连接失败: {response.status_code}")
                return False
        except Exception as e2:
            print(f"❌ 备用方法也失败: {e2}")
            return False

def test_dataset_download():
    """测试数据集下载"""
    print("\n测试数据集下载...")
    
    try:
        from datasets import load_dataset
        
        # 设置镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        
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
    print("🔧 HuggingFace镜像测试工具")
    print("=" * 50)
    
    # 测试镜像连接
    mirror = test_mirror_connection()
    
    if not mirror:
        print("❌ 所有镜像都无法连接")
        return 1
    
    # 测试模型下载
    model_ok = test_model_download()
    
    # 测试数据集下载
    dataset_ok = test_dataset_download()
    
    print("\n" + "=" * 50)
    
    if model_ok and dataset_ok:
        print("✅ 镜像配置测试成功!")
        print(f"推荐镜像: {mirror}")
        print("现在可以运行实验:")
        print("  python run_with_mirror.py")
    else:
        print("❌ 镜像配置测试失败")
        print("请检查网络连接")
    
    return 0 if (model_ok and dataset_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
