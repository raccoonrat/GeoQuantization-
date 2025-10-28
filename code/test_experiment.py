#!/usr/bin/env python3
"""
简化的实验测试脚本
"""

import os
import sys
import logging

# 设置简单的日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试关键包导入"""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        logger.error(f"PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"Transformers import failed: {e}")
        return False
    
    try:
        import yaml
        logger.info("YAML import successful")
    except ImportError as e:
        logger.error(f"YAML import failed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
    except ImportError as e:
        logger.error(f"NumPy import failed: {e}")
        return False
    
    return True

def test_config():
    """测试配置加载"""
    logger.info("Testing configuration...")
    
    try:
        from prm_experiment import load_config
        config = load_config()
        logger.info(f"Config loaded successfully")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Samples: {config.calib_samples}")
        return True
    except Exception as e:
        logger.error(f"Config loading failed: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    logger.info("Testing model loading...")
    
    try:
        from prm_experiment import ModelWrapper
        # 使用很小的模型进行测试
        model = ModelWrapper("facebook/opt-125m", "cpu")
        logger.info("Model loaded successfully")
        logger.info(f"Parameter groups: {len(model.param_groups)}")
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("Starting GeoQuantization experiment tests...")
    
    tests = [
        ("Import test", test_imports),
        ("Config test", test_config),
        ("Model test", test_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                logger.info(f"PASSED: {test_name}")
                passed += 1
            else:
                logger.error(f"FAILED: {test_name}")
        except Exception as e:
            logger.error(f"ERROR in {test_name}: {e}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("All tests passed! Ready to run full experiment.")
        return 0
    else:
        logger.error("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
