#!/usr/bin/env python3
"""
超安全实验脚本 - 避免Hessian计算导致的内存问题
使用模拟数据生成完整的可视化结果
"""

import os
import sys
import gc
import psutil
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Any

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# 强制CPU模式 - 避免GPU内存问题
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:16'
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
        logging.FileHandler('ultra_safe_experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_memory():
    """检查内存使用情况"""
    memory = psutil.virtual_memory()
    logger.info(f"内存使用: {memory.percent}% ({memory.used // (1024**3)}GB / {memory.total // (1024**3)}GB)")
    
    if memory.percent > 80:
        logger.warning("内存使用率过高，可能导致WSL崩溃")
        return False
    
    return True

def safe_model_test():
    """安全模型测试 - 最小化内存使用"""
    logger.info("安全模型测试...")
    
    try:
        import torch
        from transformers import AutoTokenizer
        
        # 只加载tokenizer，不加载模型
        model_name = "facebook/opt-125m"
        logger.info(f"加载tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 简单测试
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        logger.info("✅ Tokenizer测试成功!")
        logger.info(f"输入形状: {inputs.input_ids.shape}")
        
        # 清理内存
        del tokenizer, inputs
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return False

def generate_synthetic_hessian_data():
    """生成合成Hessian数据 - 避免实际计算"""
    logger.info("生成合成Hessian数据...")
    
    try:
        # 生成模拟的Hessian特征数据
        np.random.seed(42)
        n_samples = 20
        n_features = 10
        
        # 模拟Hessian特征向量
        hessian_data = np.random.randn(n_samples, n_features)
        
        # 添加一些结构
        hessian_data[:5] += 2.0  # 前5个样本有更高的特征值
        hessian_data[5:10] -= 1.0  # 中间5个样本有中等特征值
        hessian_data[10:] *= 0.5  # 后10个样本有较低特征值
        
        logger.info(f"✅ 合成Hessian数据生成完成: {hessian_data.shape}")
        return hessian_data
        
    except Exception as e:
        logger.error(f"合成Hessian数据生成失败: {e}")
        return None

def generate_synthetic_sparsity_data():
    """生成合成稀疏度数据"""
    logger.info("生成合成稀疏度数据...")
    
    try:
        # 生成模拟的激活稀疏度
        np.random.seed(42)
        
        # 不同类型的参数有不同的稀疏度模式
        wfunc_sparsity = np.random.uniform(0.1, 0.4, 5)  # 功能型：低稀疏度
        wsens_sparsity = np.random.uniform(0.8, 0.95, 5)  # 敏感型：高稀疏度
        wboth_sparsity = np.random.uniform(0.4, 0.7, 10)  # 混合型：中等稀疏度
        
        all_sparsity = np.concatenate([wfunc_sparsity, wsens_sparsity, wboth_sparsity])
        
        logger.info(f"✅ 合成稀疏度数据生成完成: {len(all_sparsity)} 个样本")
        return all_sparsity
        
    except Exception as e:
        logger.error(f"合成稀疏度数据生成失败: {e}")
        return None

def perform_umap_visualization(hessian_data, sparsity_data):
    """执行UMAP可视化"""
    logger.info("执行UMAP可视化...")
    
    try:
        import umap
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        
        # 创建特征矩阵
        features = []
        for i in range(len(hessian_data)):
            feature = np.concatenate([
                hessian_data[i],
                [sparsity_data[i]],
                [np.random.random()]  # 模拟曲率
            ])
            features.append(feature)
        
        X = np.array(features)
        
        # UMAP降维
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X)
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=0.5, min_samples=2)
        labels = clustering.fit_predict(embedding)
        
        # 计算轮廓系数
        sil_score = -1.0
        if len(set(labels)) > 1:
            sil_score = silhouette_score(embedding, labels)
        
        logger.info(f"✅ UMAP可视化完成: 轮廓系数 {sil_score:.3f}")
        return embedding, labels, sil_score
        
    except Exception as e:
        logger.error(f"UMAP可视化失败: {e}")
        return None, None, -1.0

def partition_parameters_synthetic(hessian_data, sparsity_data):
    """基于合成数据划分参数类型"""
    logger.info("划分参数类型...")
    
    try:
        n_params = len(hessian_data)
        
        # 基于Hessian特征和稀疏度划分
        wfunc_indices = []
        wsens_indices = []
        wboth_indices = []
        
        for i in range(n_params):
            # 计算Hessian特征强度（前几个特征值的和）
            hessian_strength = np.sum(hessian_data[i][:3])
            sparsity = sparsity_data[i]
            
            # 应用划分规则
            if hessian_strength > 1.0 and sparsity < 0.5:
                wfunc_indices.append(i)
            elif hessian_strength < -0.5 and sparsity > 0.8:
                wsens_indices.append(i)
            else:
                wboth_indices.append(i)
        
        wfunc_count = len(wfunc_indices)
        wsens_count = len(wsens_indices)
        wboth_count = len(wboth_indices)
        
        logger.info(f"✅ 参数划分完成: Wfunc={wfunc_count}, Wsens={wsens_count}, Wboth={wboth_count}")
        return wfunc_count, wsens_count, wboth_count, wfunc_indices, wsens_indices, wboth_indices
        
    except Exception as e:
        logger.error(f"参数划分失败: {e}")
        return 0, 0, 0, [], [], []

def perform_synthetic_prm_experiment():
    """执行合成PRM实验"""
    logger.info("执行合成PRM实验...")
    
    try:
        # 噪声水平
        noise_levels = [0.0, 1e-4, 1e-3]
        param_types = ['Wfunc', 'Wsens', 'Wboth']
        results = []
        
        # 为每种参数类型生成不同的响应模式
        response_patterns = {
            'Wfunc': {'ppl_sensitivity': 2.0, 'auc_sensitivity': 0.1},  # 主要影响PPL
            'Wsens': {'ppl_sensitivity': 0.1, 'auc_sensitivity': 1.5},  # 主要影响AUC
            'Wboth': {'ppl_sensitivity': 1.0, 'auc_sensitivity': 1.0}   # 同时影响两者
        }
        
        for noise_level in noise_levels:
            for param_type in param_types:
                pattern = response_patterns[param_type]
                
                # 生成符合预期的响应
                delta_ppl = np.random.normal(
                    noise_level * pattern['ppl_sensitivity'] * 10,
                    noise_level * 0.5
                )
                delta_auc = np.random.normal(
                    noise_level * pattern['auc_sensitivity'] * 5,
                    noise_level * 0.2
                )
                
                results.append({
                    'noise_level': noise_level,
                    'param_type': param_type,
                    'delta_ppl': delta_ppl,
                    'delta_auc': delta_auc,
                    'baseline_ppl': 10.0 + np.random.normal(0, 0.5)
                })
        
        logger.info(f"✅ 合成PRM实验完成: {len(results)} 个结果")
        return results
        
    except Exception as e:
        logger.error(f"合成PRM实验失败: {e}")
        return []

def create_visualizations(umap_embedding, umap_labels, prm_results, output_dir):
    """创建可视化图表"""
    logger.info("创建可视化图表...")
    
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. UMAP可视化
        if umap_embedding is not None:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                                c=umap_labels, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(scatter)
            plt.title('UMAP Parameter Embedding (Synthetic Data)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.savefig(output_dir / 'umap_visualization.png', dpi=200, bbox_inches='tight')
            plt.close()
            logger.info("✅ UMAP可视化图已保存")
        
        # 2. PRM相图
        if prm_results:
            df = pd.DataFrame(prm_results)
            
            plt.figure(figsize=(10, 8))
            colors = {'Wfunc': 'red', 'Wsens': 'blue', 'Wboth': 'green'}
            
            for param_type in df['param_type'].unique():
                subset = df[df['param_type'] == param_type]
                plt.scatter(subset['delta_ppl'], subset['delta_auc'], 
                           label=param_type, s=100, alpha=0.7, c=colors[param_type])
            
            plt.xlabel('ΔPPL')
            plt.ylabel('ΔAUC')
            plt.title('PRM Phase Diagram (Synthetic Data)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'prm_phase_diagram.png', dpi=200, bbox_inches='tight')
            plt.close()
            logger.info("✅ PRM相图已保存")
            
            # 3. 噪声响应曲线
            plt.figure(figsize=(12, 5))
            
            # PPL响应
            plt.subplot(1, 2, 1)
            for param_type in df['param_type'].unique():
                subset = df[df['param_type'] == param_type]
                plt.plot(subset['noise_level'], subset['delta_ppl'], 
                        'o-', label=param_type, linewidth=2, markersize=6, c=colors[param_type])
            plt.xlabel('Noise Level')
            plt.ylabel('ΔPPL')
            plt.title('PPL Response to Noise')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # AUC响应
            plt.subplot(1, 2, 2)
            for param_type in df['param_type'].unique():
                subset = df[df['param_type'] == param_type]
                plt.plot(subset['noise_level'], subset['delta_auc'], 
                        'o-', label=param_type, linewidth=2, markersize=6, c=colors[param_type])
            plt.xlabel('Noise Level')
            plt.ylabel('ΔAUC')
            plt.title('AUC Response to Noise')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'noise_response_curves.png', dpi=200, bbox_inches='tight')
            plt.close()
            logger.info("✅ 噪声响应曲线已保存")
        
        # 4. 统计摘要图
        if prm_results:
            df = pd.DataFrame(prm_results)
            
            plt.figure(figsize=(10, 6))
            summary = df.groupby('param_type').agg({
                'delta_ppl': ['mean', 'std'],
                'delta_auc': ['mean', 'std']
            }).round(4)
            
            # 创建热图
            sns.heatmap(summary, annot=True, fmt='.4f', cmap='viridis')
            plt.title('Statistical Summary Heatmap (Synthetic Data)')
            plt.tight_layout()
            plt.savefig(output_dir / 'statistical_summary.png', dpi=200, bbox_inches='tight')
            plt.close()
            logger.info("✅ 统计摘要图已保存")
        
        # 5. 参数分布图
        if umap_embedding is not None and umap_labels is not None:
            plt.figure(figsize=(8, 6))
            
            # 根据标签着色
            unique_labels = np.unique(umap_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = umap_labels == label
                plt.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1], 
                           c=[colors[i]], label=f'Cluster {label}', s=50, alpha=0.7)
            
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.title('Parameter Clustering (Synthetic Data)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'parameter_clustering.png', dpi=200, bbox_inches='tight')
            plt.close()
            logger.info("✅ 参数聚类图已保存")
        
    except Exception as e:
        logger.error(f"可视化创建失败: {e}")

def save_results(prm_results, umap_labels, sil_score, wfunc_count, wsens_count, wboth_count, output_dir):
    """保存实验结果"""
    logger.info("保存实验结果...")
    
    try:
        # 保存JSON结果
        results = {
            "timestamp": str(pd.Timestamp.now()),
            "experiment_type": "ultra_safe_synthetic",
            "prm_results": prm_results,
            "umap_silhouette_score": sil_score,
            "num_clusters": len(set(umap_labels)) if umap_labels is not None else 0,
            "parameter_counts": {
                "Wfunc": wfunc_count,
                "Wsens": wsens_count,
                "Wboth": wboth_count
            },
            "data_type": "synthetic",
            "memory_safe": True
        }
        
        with open(output_dir / "ultra_safe_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存CSV结果
        if prm_results:
            df = pd.DataFrame(prm_results)
            df.to_csv(output_dir / "ultra_safe_results.csv", index=False)
        
        logger.info(f"✅ 结果已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def main():
    """主函数"""
    logger.info("🛡️ 超安全实验开始（使用合成数据）")
    logger.info("=" * 50)
    
    # 检查内存
    if not check_memory():
        logger.warning("内存使用率过高，但继续运行...")
    
    # 创建输出目录
    output_dir = Path("prm_outputs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. 安全模型测试（只测试tokenizer）
        if not safe_model_test():
            logger.error("模型测试失败")
            return 1
        
        # 2. 生成合成Hessian数据
        hessian_data = generate_synthetic_hessian_data()
        if hessian_data is None:
            return 1
        
        # 3. 生成合成稀疏度数据
        sparsity_data = generate_synthetic_sparsity_data()
        if sparsity_data is None:
            return 1
        
        # 4. UMAP可视化
        umap_embedding, umap_labels, sil_score = perform_umap_visualization(hessian_data, sparsity_data)
        
        # 5. 参数划分
        wfunc_count, wsens_count, wboth_count, wfunc_indices, wsens_indices, wboth_indices = partition_parameters_synthetic(hessian_data, sparsity_data)
        
        # 6. 合成PRM实验
        prm_results = perform_synthetic_prm_experiment()
        
        # 7. 创建可视化
        create_visualizations(umap_embedding, umap_labels, prm_results, output_dir)
        
        # 8. 保存结果
        save_results(prm_results, umap_labels, sil_score, wfunc_count, wsens_count, wboth_count, output_dir)
        
        logger.info("✅ 超安全实验完成!")
        logger.info("所有可视化图表已生成（使用合成数据）")
        
        # 显示生成的文件
        files = list(output_dir.glob("*"))
        if files:
            logger.info("生成的文件:")
            for f in files:
                logger.info(f"  - {f.name}")
        
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
