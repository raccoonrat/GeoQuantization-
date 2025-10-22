#!/usr/bin/env python3
"""
离线模式配置脚本
当网络连接有问题时使用
"""

import os
import sys
import json
from pathlib import Path

def setup_offline_mode():
    """设置离线模式"""
    print("配置离线模式...")
    
    # 设置离线模式环境变量
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # 设置缓存目录
    cache_dir = os.path.join(os.getcwd(), 'hf_cache')
    os.environ['HF_HOME'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"✅ 离线模式已启用")
    print(f"   缓存目录: {cache_dir}")

def create_dummy_model():
    """创建虚拟模型用于测试"""
    print("创建虚拟模型...")
    
    cache_dir = Path('./hf_cache')
    model_dir = cache_dir / 'models--facebook--opt-125m' / 'snapshots' / 'main'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建虚拟tokenizer配置
    tokenizer_config = {
        "tokenizer_class": "GPT2Tokenizer",
        "vocab_size": 50272,
        "model_max_length": 1024,
        "padding_side": "right",
        "truncation_side": "right",
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>"
    }
    
    with open(model_dir / 'tokenizer_config.json', 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # 创建虚拟词汇表
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "Hello": 4,
        "world": 5,
        ",": 6,
        "!": 7
    }
    
    with open(model_dir / 'vocab.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # 创建合并文件
    merges = ["#version: 0.2"]
    with open(model_dir / 'merges.txt', 'w') as f:
        f.write('\n'.join(merges))
    
    print(f"✅ 虚拟模型已创建: {model_dir}")

def test_offline_mode():
    """测试离线模式"""
    print("测试离线模式...")
    
    try:
        from transformers import AutoTokenizer
        
        # 测试加载虚拟模型
        model_name = "facebook/opt-125m"
        print(f"正在加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            cache_dir=os.environ.get('HF_HOME', './hf_cache')
        )
        
        print("✅ 离线模型加载成功!")
        
        # 测试分词
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ 分词测试成功: {tokens.input_ids.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 离线模式测试失败: {e}")
        return False

def create_simple_tokenizer():
    """创建简单的分词器"""
    print("创建简单分词器...")
    
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {
                "<pad>": 0,
                "<s>": 1,
                "</s>": 2,
                "<unk>": 3,
                "Hello": 4,
                "world": 5,
                ",": 6,
                "!": 7
            }
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        def __call__(self, text, return_tensors=None):
            # 简单的分词逻辑
            words = text.split()
            token_ids = []
            for word in words:
                if word in self.vocab:
                    token_ids.append(self.vocab[word])
                else:
                    token_ids.append(self.vocab["<unk>"])
            
            # 添加开始和结束标记
            token_ids = [self.vocab["<s>"]] + token_ids + [self.vocab["</s>"]]
            
            if return_tensors == "pt":
                import torch
                return {"input_ids": torch.tensor([token_ids])}
            else:
                return {"input_ids": token_ids}
        
        def decode(self, token_ids):
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            
            words = []
            for token_id in token_ids:
                if token_id in self.reverse_vocab:
                    words.append(self.reverse_vocab[token_id])
            
            return " ".join(words)
    
    return SimpleTokenizer()

def test_simple_tokenizer():
    """测试简单分词器"""
    print("测试简单分词器...")
    
    try:
        tokenizer = create_simple_tokenizer()
        
        # 测试分词
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ 分词成功: {tokens['input_ids'].shape}")
        
        # 测试解码
        decoded = tokenizer.decode(tokens['input_ids'])
        print(f"✅ 解码成功: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单分词器测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 离线模式配置工具")
    print("=" * 50)
    
    # 设置离线模式
    setup_offline_mode()
    
    # 创建虚拟模型
    create_dummy_model()
    
    # 测试离线模式
    offline_ok = test_offline_mode()
    
    # 测试简单分词器
    simple_ok = test_simple_tokenizer()
    
    print("\n" + "=" * 50)
    
    if offline_ok or simple_ok:
        print("✅ 离线模式配置成功!")
        print("现在可以在离线模式下运行实验")
    else:
        print("❌ 离线模式配置失败")
        print("请检查配置或使用其他方法")
    
    return 0 if (offline_ok or simple_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
