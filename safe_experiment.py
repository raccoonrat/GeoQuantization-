#!/usr/bin/env python3
"""
安全实验脚本 - 避免GPU内存不足和WSL崩溃
"""

import os
import sys
import gc
import psutil
import logging
from pathlib import Path

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# 强制CPU模式 - 避免GPU内存问题
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'

# 设置缓存目录
cache_dir = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HOME'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('safe_experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_memory():
    """检查内存使用情况"""
    memory = psutil.virtual_memory()
    logger.info(f"内存使用: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
    
    if memory.percent > 85:
        logger.warning("内存使用率过高，可能导致WSL崩溃")
        return False
    
    return True

def safe_model_test():
    """安全模型测试"""
    logger.info("开始安全模型测试...")
    
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        logger.info(f"设备: {'CPU' if not torch.cuda.is_available() else 'GPU'}")
        
        # 强制使用CPU
        device = torch.device('cpu')
        logger.info(f"使用设备: {device}")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 使用最小模型
        model_name = "facebook/opt-125m"
        logger.info(f"加载模型: {model_name}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型 - 使用float32避免内存问题
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # 使用float32
            device_map="cpu",  # 强制CPU
            low_cpu_mem_usage=True  # 低内存使用
        )
        
        logger.info("模型加载成功!")
        
        # 简单测试 - 使用最小输入
        test_text = "Hello"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=10,  # 最小长度
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"测试结果: {result}")
        
        # 清理内存
        del model, tokenizer, outputs
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return False

def safe_geometry_test():
    """安全几何分析测试"""
    logger.info("开始安全几何分析测试...")
    
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
        
        # 创建小规模模拟数据
        np.random.seed(42)
        data = np.random.randn(50, 5)  # 减少数据量
        
        # PCA降维
        pca = PCA(n_components=3)  # 减少组件数
        reduced_data = pca.fit_transform(data)
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=0.5, min_samples=2)  # 减少最小样本数
        labels = clustering.fit_predict(reduced_data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"聚类结果: {n_clusters} 个簇")
        
        # 清理内存
        del data, reduced_data, labels
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"几何分析测试失败: {e}")
        return False

def safe_noise_test():
    """安全噪声测试"""
    logger.info("开始安全噪声测试...")
    
    try:
        import torch
        import numpy as np
        
        # 创建小规模测试数据
        test_data = torch.randn(10, 5)  # 小规模数据
        
        # 测试不同噪声水平
        noise_levels = [0.0, 0.01, 0.1]
        
        for noise_level in noise_levels:
            noise = torch.randn_like(test_data) * noise_level
            noisy_data = test_data + noise
            
            # 计算变化
            change = torch.mean(torch.abs(noisy_data - test_data)).item()
            logger.info(f"噪声水平 {noise_level}: 平均变化 {change:.4f}")
        
        # 清理内存
        del test_data, noise, noisy_data
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"噪声测试失败: {e}")
        return False

def save_safe_results():
    """保存安全实验结果"""
    logger.info("保存安全实验结果...")
    
    try:
        import pandas as pd
        import json
        
        results = {
            "timestamp": str(pd.Timestamp.now()),
            "model_test": "passed",
            "geometry_test": "passed", 
            "noise_test": "passed",
            "memory_usage": "optimized",
            "device": "CPU",
            "model_size": "small"
        }
        
        # 创建输出目录
        output_dir = Path("prm_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 保存结果
        with open(output_dir / "safe_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # 创建CSV报告
        df = pd.DataFrame([results])
        df.to_csv(output_dir / "safe_results.csv", index=False)
        
        logger.info(f"结果已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def main():
    """主函数"""
    logger.info("🛡️ 安全实验开始")
    logger.info("=" * 50)
    
    # 检查内存
    if not check_memory():
        logger.warning("内存使用率过高，但继续运行...")
    
    try:
        # 安全模型测试
        if not safe_model_test():
            logger.error("模型测试失败")
            return 1
        
        # 安全几何分析测试
        if not safe_geometry_test():
            logger.error("几何分析测试失败")
            return 1
        
        # 安全噪声测试
        if not safe_noise_test():
            logger.error("噪声测试失败")
            return 1
        
        # 保存结果
        save_safe_results()
        
        logger.info("✅ 安全实验完成!")
        logger.info("所有测试通过，WSL应该不会崩溃")
        
        return 0
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        return 1
    finally:
        # 清理内存
        gc.collect()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"致命错误: {e}")
        sys.exit(1)
