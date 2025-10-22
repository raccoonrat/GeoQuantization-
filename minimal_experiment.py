#!/usr/bin/env python3
"""
WSL2最小化实验脚本
避免内存问题和WSL2终止
"""

import os
import sys
import gc
import logging
from pathlib import Path

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# 设置内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('minimal_experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def minimal_model_test():
    """最小化模型测试"""
    logger.info("开始最小化模型测试...")
    
    try:
        # 导入必要的包
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"Transformers版本: {transformers.__version__}")
        
        # 使用CPU避免GPU内存问题
        device = "cpu"
        logger.info(f"使用设备: {device}")
        
        # 加载小模型
        model_name = "facebook/opt-125m"
        logger.info(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # 使用float32避免内存问题
            device_map="cpu"
        )
        
        logger.info("模型加载成功!")
        
        # 简单测试
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=20,
                num_return_sequences=1,
                do_sample=False
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"测试结果: {result}")
        
        # 清理内存
        del model, tokenizer
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return False

def minimal_geometry_test():
    """最小化几何分析测试"""
    logger.info("开始最小化几何分析测试...")
    
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.cluster import DBSCAN
        
        # 创建模拟数据
        np.random.seed(42)
        data = np.random.randn(100, 10)
        
        # PCA降维
        pca = PCA(n_components=5)
        reduced_data = pca.fit_transform(data)
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=0.5, min_samples=3)
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

def save_results():
    """保存结果"""
    logger.info("保存实验结果...")
    
    results = {
        "timestamp": str(pd.Timestamp.now()),
        "model_test": "passed",
        "geometry_test": "passed",
        "memory_usage": "optimized"
    }
    
    # 创建输出目录
    output_dir = Path("prm_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 保存结果
    import json
    with open(output_dir / "minimal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"结果已保存到: {output_dir / 'minimal_results.json'}")

def main():
    """主函数"""
    logger.info("🐧 WSL2最小化实验开始")
    logger.info("=" * 50)
    
    try:
        # 测试模型
        if not minimal_model_test():
            logger.error("模型测试失败")
            return 1
        
        # 测试几何分析
        if not minimal_geometry_test():
            logger.error("几何分析测试失败")
            return 1
        
        # 保存结果
        save_results()
        
        logger.info("✅ 最小化实验完成!")
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
