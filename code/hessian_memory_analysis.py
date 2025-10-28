#!/usr/bin/env python3
"""
Hessian内存消耗分析工具
预估不同模型和配置下的内存需求
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_parameter_count(model_name):
    """获取模型参数数量"""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return total_params, trainable_params
    except Exception as e:
        logger.error(f"获取模型参数失败: {e}")
        return None, None

def estimate_hessian_memory_usage(model_name, batch_size=1, sequence_length=128):
    """预估Hessian计算的内存使用"""
    logger.info(f"分析模型: {model_name}")
    
    try:
        # 获取模型参数数量
        total_params, trainable_params = get_model_parameter_count(model_name)
        if total_params is None:
            return None
        
        logger.info(f"总参数数量: {total_params:,}")
        logger.info(f"可训练参数数量: {trainable_params:,}")
        
        # 内存计算（以字节为单位）
        param_memory = total_params * 4  # float32 = 4 bytes
        
        # 梯度内存（与参数相同大小）
        gradient_memory = trainable_params * 4
        
        # Hessian矩阵内存（对于二阶导数）
        # 注意：完整Hessian矩阵是 N×N，但通常只计算对角或块对角
        hessian_diagonal_memory = trainable_params * 4  # 对角元素
        hessian_full_memory = trainable_params * trainable_params * 4  # 完整矩阵
        
        # 激活内存（前向传播）
        # 估算：每层激活大小 ≈ batch_size * sequence_length * hidden_size
        # 这里使用经验公式
        estimated_hidden_size = min(4096, total_params // 100000)  # 经验估算
        activation_memory = batch_size * sequence_length * estimated_hidden_size * 4
        
        # 中间计算内存
        intermediate_memory = param_memory * 2  # 中间变量
        
        # 总内存估算
        memory_breakdown = {
            'parameters': param_memory,
            'gradients': gradient_memory,
            'hessian_diagonal': hessian_diagonal_memory,
            'hessian_full': hessian_full_memory,
            'activations': activation_memory,
            'intermediate': intermediate_memory
        }
        
        # 不同策略的内存需求
        strategies = {
            'diagonal_hessian': param_memory + gradient_memory + hessian_diagonal_memory + activation_memory + intermediate_memory,
            'block_diagonal_hessian': param_memory + gradient_memory + hessian_diagonal_memory * 10 + activation_memory + intermediate_memory,
            'full_hessian': param_memory + gradient_memory + hessian_full_memory + activation_memory + intermediate_memory,
            'approximate_hessian': param_memory + gradient_memory + hessian_diagonal_memory + activation_memory + intermediate_memory * 0.5
        }
        
        return {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_breakdown': memory_breakdown,
            'strategies': strategies
        }
        
    except Exception as e:
        logger.error(f"内存分析失败: {e}")
        return None

def format_memory_size(bytes_size):
    """格式化内存大小"""
    if bytes_size < 1024**3:
        return f"{bytes_size / (1024**2):.1f} MB"
    else:
        return f"{bytes_size / (1024**3):.1f} GB"

def analyze_multiple_models():
    """分析多个模型的内存需求"""
    models = [
        "facebook/opt-125m",
        "facebook/opt-350m", 
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b"
    ]
    
    results = []
    
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"分析模型: {model_name}")
        logger.info(f"{'='*60}")
        
        result = estimate_hessian_memory_usage(model_name)
        if result:
            results.append(result)
            
            # 显示结果
            logger.info(f"参数数量: {result['total_params']:,}")
            logger.info(f"可训练参数: {result['trainable_params']:,}")
            
            logger.info("\n内存分解:")
            for key, value in result['memory_breakdown'].items():
                logger.info(f"  {key}: {format_memory_size(value)}")
            
            logger.info("\n不同策略的内存需求:")
            for strategy, memory in result['strategies'].items():
                logger.info(f"  {strategy}: {format_memory_size(memory)}")
    
    return results

def recommend_strategy(results):
    """推荐最佳策略"""
    logger.info(f"\n{'='*60}")
    logger.info("策略推荐")
    logger.info(f"{'='*60}")
    
    # 获取系统内存
    system_memory = psutil.virtual_memory().total
    available_memory = psutil.virtual_memory().available
    
    logger.info(f"系统总内存: {format_memory_size(system_memory)}")
    logger.info(f"可用内存: {format_memory_size(available_memory)}")
    
    for result in results:
        model_name = result['model_name']
        logger.info(f"\n模型: {model_name}")
        
        # 检查每种策略的可行性
        for strategy, memory_req in result['strategies'].items():
            if memory_req < available_memory * 0.8:  # 使用80%的可用内存
                status = "✅ 可行"
            elif memory_req < system_memory * 0.8:
                status = "⚠️  需要关闭其他程序"
            else:
                status = "❌ 内存不足"
            
            logger.info(f"  {strategy}: {format_memory_size(memory_req)} - {status}")

def create_memory_optimization_plan():
    """创建内存优化计划"""
    logger.info(f"\n{'='*60}")
    logger.info("内存优化建议")
    logger.info(f"{'='*60}")
    
    optimizations = [
        {
            'name': '梯度检查点',
            'description': '使用gradient checkpointing减少激活内存',
            'memory_saving': '50-70%',
            'implementation': 'model.gradient_checkpointing_enable()'
        },
        {
            'name': '混合精度',
            'description': '使用float16减少内存使用',
            'memory_saving': '50%',
            'implementation': 'torch.cuda.amp.autocast()'
        },
        {
            'name': '参数分片',
            'description': '将模型参数分片到多个设备',
            'memory_saving': '按设备数量线性减少',
            'implementation': 'torch.nn.parallel.DistributedDataParallel'
        },
        {
            'name': 'Hessian近似',
            'description': '使用L-BFGS或有限差分近似Hessian',
            'memory_saving': '90%+',
            'implementation': 'scipy.optimize.L-BFGS-B'
        },
        {
            'name': '批处理减少',
            'description': '减少批处理大小',
            'memory_saving': '线性减少',
            'implementation': 'batch_size=1'
        },
        {
            'name': '序列长度限制',
            'description': '限制输入序列长度',
            'memory_saving': '线性减少',
            'implementation': 'max_length=64'
        }
    ]
    
    for opt in optimizations:
        logger.info(f"\n{opt['name']}:")
        logger.info(f"  描述: {opt['description']}")
        logger.info(f"  内存节省: {opt['memory_saving']}")
        logger.info(f"  实现: {opt['implementation']}")

def main():
    """主函数"""
    logger.info("🔍 Hessian内存消耗分析工具")
    logger.info("=" * 60)
    
    # 分析多个模型
    results = analyze_multiple_models()
    
    if results:
        # 推荐策略
        recommend_strategy(results)
        
        # 优化建议
        create_memory_optimization_plan()
        
        # 保存结果
        import json
        with open('hessian_memory_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n✅ 分析完成，结果已保存到 hessian_memory_analysis.json")
    else:
        logger.error("❌ 分析失败")

if __name__ == "__main__":
    main()
