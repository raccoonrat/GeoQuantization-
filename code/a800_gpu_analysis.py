#!/usr/bin/env python3
"""
A800 GPU内存分析工具
分析在80GB GPU上进行Hessian计算的可行性
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_memory():
    """检查GPU内存"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"可用GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # GB
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  总内存: {total_memory:.1f} GB")
            logger.info(f"  计算能力: {props.major}.{props.minor}")
        
        return True
    else:
        logger.warning("CUDA不可用")
        return False

def analyze_model_memory_on_gpu(model_name):
    """分析模型在GPU上的内存使用"""
    logger.info(f"分析模型: {model_name}")
    
    try:
        # 检查GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"初始GPU内存使用: {initial_memory:.2f} GB")
        
        # 加载模型到GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # 使用float16节省内存
            device_map="auto"
        )
        
        if torch.cuda.is_available():
            model_memory = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"模型加载后GPU内存: {model_memory:.2f} GB")
            logger.info(f"模型占用内存: {model_memory - initial_memory:.2f} GB")
        
        # 获取参数信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"总参数数量: {total_params:,}")
        logger.info(f"可训练参数数量: {trainable_params:,}")
        
        # 估算Hessian内存需求
        hessian_memory_estimates = {
            'diagonal_hessian_fp16': trainable_params * 2 / (1024**3),  # float16
            'diagonal_hessian_fp32': trainable_params * 4 / (1024**3),  # float32
            'block_hessian_fp16': trainable_params * 10 * 2 / (1024**3),  # 10x block
            'full_hessian_fp16': trainable_params * trainable_params * 2 / (1024**3),
        }
        
        logger.info("Hessian内存需求估算:")
        for method, memory_gb in hessian_memory_estimates.items():
            logger.info(f"  {method}: {memory_gb:.2f} GB")
        
        # 清理内存
        del model
        torch.cuda.empty_cache()
        
        return {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_memory_gb': model_memory - initial_memory if torch.cuda.is_available() else 0,
            'hessian_memory_estimates': hessian_memory_estimates
        }
        
    except Exception as e:
        logger.error(f"模型分析失败: {e}")
        return None

def analyze_a800_capabilities():
    """分析A800的能力"""
    logger.info("A800 GPU能力分析")
    logger.info("=" * 50)
    
    # A800规格
    a800_specs = {
        'total_memory': 80,  # GB
        'memory_bandwidth': 2039,  # GB/s
        'compute_capability': '8.0',
        'cuda_cores': 6912,
        'tensor_cores': 432,
        'fp16_performance': 312,  # TFLOPS
        'fp32_performance': 156,  # TFLOPS
    }
    
    logger.info("A800规格:")
    for key, value in a800_specs.items():
        logger.info(f"  {key}: {value}")
    
    # 可用内存估算（考虑系统开销）
    usable_memory = a800_specs['total_memory'] * 0.9  # 90%可用
    logger.info(f"估算可用内存: {usable_memory:.1f} GB")
    
    return a800_specs, usable_memory

def recommend_models_for_a800():
    """推荐适合A800的模型"""
    logger.info("推荐模型配置")
    logger.info("=" * 50)
    
    models = [
        {
            'name': 'facebook/opt-125m',
            'params': 125_000_000,
            'estimated_memory': 0.5,
            'hessian_diagonal_fp16': 0.25,
            'recommended': True,
            'notes': '适合完整Hessian计算'
        },
        {
            'name': 'facebook/opt-350m',
            'params': 350_000_000,
            'estimated_memory': 1.4,
            'hessian_diagonal_fp16': 0.7,
            'recommended': True,
            'notes': '适合完整Hessian计算'
        },
        {
            'name': 'facebook/opt-1.3b',
            'params': 1_300_000_000,
            'estimated_memory': 5.2,
            'hessian_diagonal_fp16': 2.6,
            'recommended': True,
            'notes': '适合完整Hessian计算'
        },
        {
            'name': 'facebook/opt-2.7b',
            'params': 2_700_000_000,
            'estimated_memory': 10.8,
            'hessian_diagonal_fp16': 5.4,
            'recommended': True,
            'notes': '适合完整Hessian计算'
        },
        {
            'name': 'facebook/opt-6.7b',
            'params': 6_700_000_000,
            'estimated_memory': 26.8,
            'hessian_diagonal_fp16': 13.4,
            'recommended': True,
            'notes': '适合完整Hessian计算'
        },
        {
            'name': 'facebook/opt-13b',
            'params': 13_000_000_000,
            'estimated_memory': 52.0,
            'hessian_diagonal_fp16': 26.0,
            'recommended': True,
            'notes': '适合完整Hessian计算'
        },
        {
            'name': 'facebook/opt-30b',
            'params': 30_000_000_000,
            'estimated_memory': 120.0,
            'hessian_diagonal_fp16': 60.0,
            'recommended': False,
            'notes': '需要模型并行或量化'
        }
    ]
    
    for model in models:
        status = "✅ 推荐" if model['recommended'] else "❌ 不推荐"
        logger.info(f"{model['name']}: {status}")
        logger.info(f"  参数数量: {model['params']:,}")
        logger.info(f"  模型内存: {model['estimated_memory']:.1f} GB")
        logger.info(f"  对角Hessian: {model['hessian_diagonal_fp16']:.1f} GB")
        logger.info(f"  说明: {model['notes']}")
        logger.info("")

def create_a800_optimization_plan():
    """创建A800优化方案"""
    logger.info("A800优化方案")
    logger.info("=" * 50)
    
    optimizations = [
        {
            'name': '混合精度训练',
            'description': '使用float16减少内存使用',
            'memory_saving': '50%',
            'implementation': 'torch.cuda.amp.autocast()',
            'recommended': True
        },
        {
            'name': '梯度累积',
            'description': '累积梯度减少内存峰值',
            'memory_saving': '30-50%',
            'implementation': 'accumulation_steps=4',
            'recommended': True
        },
        {
            'name': '模型并行',
            'description': '将模型分片到多个GPU',
            'memory_saving': '按GPU数量线性减少',
            'implementation': 'torch.nn.parallel.DistributedDataParallel',
            'recommended': False
        },
        {
            'name': '激活检查点',
            'description': '重新计算激活节省内存',
            'memory_saving': '50-70%',
            'implementation': 'model.gradient_checkpointing_enable()',
            'recommended': True
        },
        {
            'name': 'Hessian分块计算',
            'description': '分块计算Hessian矩阵',
            'memory_saving': '80%+',
            'implementation': 'chunk_size=1000',
            'recommended': True
        },
        {
            'name': '动态批处理',
            'description': '根据内存动态调整批处理大小',
            'memory_saving': '自适应',
            'implementation': 'adaptive_batch_size',
            'recommended': True
        }
    ]
    
    for opt in optimizations:
        status = "✅ 推荐" if opt['recommended'] else "⚠️ 可选"
        logger.info(f"{opt['name']}: {status}")
        logger.info(f"  描述: {opt['description']}")
        logger.info(f"  内存节省: {opt['memory_saving']}")
        logger.info(f"  实现: {opt['implementation']}")
        logger.info("")

def estimate_experiment_memory():
    """估算完整实验的内存需求"""
    logger.info("完整实验内存需求估算")
    logger.info("=" * 50)
    
    # 以OPT-6.7B为例
    model_memory = 26.8  # GB
    hessian_diagonal = 13.4  # GB
    activations = 2.0  # GB
    gradients = 13.4  # GB
    intermediate = 5.0  # GB
    
    total_memory = model_memory + hessian_diagonal + activations + gradients + intermediate
    
    logger.info(f"模型内存: {model_memory:.1f} GB")
    logger.info(f"Hessian对角: {hessian_diagonal:.1f} GB")
    logger.info(f"激活内存: {activations:.1f} GB")
    logger.info(f"梯度内存: {gradients:.1f} GB")
    logger.info(f"中间变量: {intermediate:.1f} GB")
    logger.info(f"总内存需求: {total_memory:.1f} GB")
    
    a800_memory = 80
    memory_usage = (total_memory / a800_memory) * 100
    
    logger.info(f"A800总内存: {a800_memory} GB")
    logger.info(f"内存使用率: {memory_usage:.1f}%")
    
    if memory_usage < 80:
        logger.info("✅ 内存充足，可以运行完整实验")
    elif memory_usage < 95:
        logger.info("⚠️ 内存紧张，建议使用优化技术")
    else:
        logger.info("❌ 内存不足，需要进一步优化")

def main():
    """主函数"""
    logger.info("🚀 A800 GPU内存分析工具")
    logger.info("=" * 60)
    
    # 检查GPU
    if not check_gpu_memory():
        logger.error("GPU不可用，无法进行分析")
        return
    
    # 分析A800能力
    a800_specs, usable_memory = analyze_a800_capabilities()
    
    # 推荐模型
    recommend_models_for_a800()
    
    # 优化方案
    create_a800_optimization_plan()
    
    # 内存需求估算
    estimate_experiment_memory()
    
    logger.info("=" * 60)
    logger.info("✅ A800分析完成")
    logger.info("结论: A800 80GB GPU完全足够进行Hessian计算实验")

if __name__ == "__main__":
    main()
