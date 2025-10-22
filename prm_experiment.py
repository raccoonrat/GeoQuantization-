#!/usr/bin/env python3
"""
GeoQuantization PRM-FSGD Experiment
基于几何意义的LLM量化离群点分析实验

Author: GeoQuantization Team
Date: 2024
Description: 实现基于几何意义的功能型vs敏感型离群点分离实验
"""

import os
import math
import json
import random
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# External libs
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration Management
# -----------------------------
@dataclass
class ExperimentConfig:
    """实验配置类"""
    # Model settings
    model_name: str = "facebook/opt-125m"
    device: str = "auto"
    max_length: int = 256
    batch_size: int = 8
    
    # Dataset settings
    calib_name: str = "wikitext"
    calib_subset: str = "wikitext-2-raw-v1"
    calib_samples: int = 200
    eval_samples: int = 200
    
    # Experiment parameters
    seed: int = 42
    topk_eigenvectors: int = 50
    noise_sigmas: List[float] = None
    repeats: int = 3
    output_dir: str = "prm_outputs"
    
    # Geometric analysis
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    
    # Thresholds
    func_cos_min: float = 0.7
    func_sparsity_max: float = 0.5
    sens_cos_max: float = 0.3
    sens_sparsity_min: float = 0.9
    
    def __post_init__(self):
        if self.noise_sigmas is None:
            self.noise_sigmas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
        
        # Auto device selection
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

def load_config(config_path: str = "experiment_config.yaml") -> ExperimentConfig:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract values from nested structure
        model_config = config_dict.get('model', {})
        dataset_config = config_dict.get('dataset', {})
        experiment_config = config_dict.get('experiment', {})
        geometry_config = config_dict.get('geometry', {})
        
        # Build flat config
        flat_config = {
            'model_name': model_config.get('name', 'facebook/opt-125m'),
            'device': model_config.get('device', 'auto'),
            'max_length': model_config.get('max_length', 256),
            'batch_size': model_config.get('batch_size', 8),
            'calib_name': dataset_config.get('calib_name', 'wikitext'),
            'calib_subset': dataset_config.get('calib_subset', 'wikitext-2-raw-v1'),
            'calib_samples': dataset_config.get('calib_samples', 200),
            'eval_samples': dataset_config.get('eval_samples', 200),
            'seed': experiment_config.get('seed', 42),
            'topk_eigenvectors': experiment_config.get('topk_eigenvectors', 50),
            'noise_sigmas': experiment_config.get('noise_sigmas', [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
            'repeats': experiment_config.get('repeats', 3),
            'output_dir': experiment_config.get('output_dir', 'prm_outputs'),
            'umap_n_neighbors': geometry_config.get('umap', {}).get('n_neighbors', 15),
            'umap_min_dist': geometry_config.get('umap', {}).get('min_dist', 0.1),
            'dbscan_eps': geometry_config.get('clustering', {}).get('eps', 0.5),
            'dbscan_min_samples': geometry_config.get('clustering', {}).get('min_samples', 5),
            'func_cos_min': geometry_config.get('thresholds', {}).get('func_cos_min', 0.7),
            'func_sparsity_max': geometry_config.get('thresholds', {}).get('func_sparsity_max', 0.5),
            'sens_cos_max': geometry_config.get('thresholds', {}).get('sens_cos_max', 0.3),
            'sens_sparsity_min': geometry_config.get('thresholds', {}).get('sens_sparsity_min', 0.9),
        }
        
        return ExperimentConfig(**flat_config)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return ExperimentConfig()

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(obj: Any, path: str):
    """保存JSON文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_csv(df: pd.DataFrame, path: str):
    """保存CSV文件"""
    df.to_csv(path, index=False, encoding='utf-8')

# -----------------------------
# Data Loading
# -----------------------------
def get_calib_texts(dataset_name: Tuple[str, str], n: int) -> List[str]:
    """获取校准文本数据"""
    logger.info(f"Loading calibration dataset: {dataset_name}")
    try:
        ds = load_dataset(dataset_name[0], dataset_name[1], split='train')
        samples = []
        for ex in ds:
            text = ex.get('text') or ex.get('article') or ex.get('content')
            if text and len(text.split()) > 5:
                samples.append(text)
            if len(samples) >= n:
                break
        logger.info(f"Loaded {len(samples)} calibration samples")
        return samples
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        # Fallback to dummy data
        return [f"Sample text {i} for calibration purposes." for i in range(n)]

# -----------------------------
# Model Wrapper
# -----------------------------
class ModelWrapper:
    """模型包装器，用于参数分析和评估"""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        logger.info(f"Loading model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        self.model.to(device)
        self.device = device
        self.model.eval()
        
        # Collect parameter groups
        self.param_groups = self._collect_param_groups()
        logger.info(f"Collected {len(self.param_groups)} parameter groups")
    
    def _collect_param_groups(self) -> List[Tuple[str, torch.Tensor]]:
        """收集参数组"""
        groups = []
        for name, p in self.model.named_parameters():
            if 'weight' in name and p.ndim >= 2:
                groups.append((name, p))
        return groups
    
    def tokenize(self, texts: List[str], max_length: int = 256) -> Dict[str, torch.Tensor]:
        """文本分词"""
        return self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
    
    @torch.no_grad()
    def compute_ppl(self, texts: List[str], batch_size: int = 8) -> float:
        """计算困惑度"""
        tokenized = self.tokenize(texts)
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        nll = 0.0
        total = 0
        
        for i in range(0, input_ids.size(0), batch_size):
            ids = input_ids[i:i+batch_size]
            att = attention_mask[i:i+batch_size]
            
            outputs = self.model(ids, attention_mask=att, labels=ids)
            batch_nll = outputs.loss.item() * ids.numel()
            nll += batch_nll
            total += ids.numel()
        
        ppl = math.exp(nll / total)
        return ppl

# -----------------------------
# Geometric Analysis
# -----------------------------
def approx_top_eigvecs(model: ModelWrapper, calib_texts: List[str], 
                      topk: int = 50, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """近似计算Hessian的top-k特征向量"""
    logger.info("Computing approximate top eigenvectors via gradient PCA")
    
    Gs = []
    model.model.zero_grad()
    
    for text in tqdm(calib_texts[:min(20, len(calib_texts))], desc='Collecting gradients'):
        try:
            tok = model.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            input_ids = tok['input_ids'].to(device)
            att = tok['attention_mask'].to(device)
            
            out = model.model(input_ids, attention_mask=att, labels=input_ids)
            loss = out.loss
            
            grads = torch.autograd.grad(
                loss, [p for (_, p) in model.param_groups], 
                retain_graph=False, create_graph=False, allow_unused=True
            )
            
            # Flatten and concatenate gradients
            grad_flat = torch.cat([
                g.detach().flatten() if g is not None else torch.zeros_like(p).flatten() 
                for ((_, p), g) in zip(model.param_groups, grads)
            ])
            Gs.append(grad_flat.cpu().numpy())
            
        except Exception as e:
            logger.warning(f"Error processing text: {e}")
            continue
        finally:
            model.model.zero_grad()
    
    if not Gs:
        logger.error("No gradients collected")
        return np.array([]), np.array([])
    
    G = np.stack(Gs, axis=0)
    G_mean = G.mean(axis=0, keepdims=True)
    Gc = G - G_mean
    
    # SVD
    u, s, vh = np.linalg.svd(Gc, full_matrices=False)
    vecs = vh[:topk]
    vals = s[:topk]
    
    logger.info(f"Computed {vecs.shape[0]} eigenvectors")
    return vecs, vals

def compute_activation_sparsity(model: ModelWrapper, texts: List[str], 
                               device: str = 'cpu') -> Dict[str, float]:
    """计算激活稀疏度"""
    logger.info("Computing activation sparsity")
    
    activ_counts = {}
    activ_nonzero = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            a = out.detach().cpu()
            nonzero = (a.abs() > 1e-5).sum().item()
            total = a.numel()
            activ_nonzero[name] = activ_nonzero.get(name, 0) + nonzero
            activ_counts[name] = activ_counts.get(name, 0) + total
        return hook
    
    # Attach hooks to linear modules
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run forward passes
    tokenizer = model.tokenizer
    for text in tqdm(texts, desc='Computing sparsity'):
        try:
            tok = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            input_ids = tok['input_ids'].to(device)
            att = tok['attention_mask'].to(device)
            
            with torch.no_grad():
                _ = model.model(input_ids, attention_mask=att)
        except Exception as e:
            logger.warning(f"Error in forward pass: {e}")
            continue
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Calculate sparsity
    sparsity = {}
    for k in activ_counts:
        frac = 1.0 - (activ_nonzero[k] / activ_counts[k])
        sparsity[k] = frac
    
    logger.info(f"Computed sparsity for {len(sparsity)} modules")
    return sparsity

# -----------------------------
# Clustering and Partitioning
# -----------------------------
def build_param_feature_matrix(model: ModelWrapper, eigvecs: np.ndarray, 
                              sparsity: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    """构建参数特征矩阵"""
    logger.info("Building parameter feature matrix")
    
    feats = []
    names = []
    
    for name, p in model.param_groups:
        names.append(name)
        arr = p.detach().cpu().numpy().flatten()
        
        # Projection onto eigenvectors
        if eigvecs.size > 0:
            proj = arr.dot(eigvecs.T[:min(eigvecs.shape[0], 20)].T)
        else:
            proj = np.zeros(20)
        
        # Curvature proxy (L2 norm)
        curvature = np.linalg.norm(arr)
        sp = sparsity.get(name, 0.0)
        
        vec = np.concatenate([proj.flatten(), [curvature, sp]])
        feats.append(vec)
    
    X = np.stack(feats, axis=0)
    logger.info(f"Feature matrix shape: {X.shape}")
    return names, X

def cluster_and_label(names: List[str], X: np.ndarray, 
                     n_neighbors: int = 15, min_dist: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float]:
    """UMAP降维和DBSCAN聚类"""
    logger.info("Performing UMAP dimensionality reduction and clustering")
    
    # UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    emb = reducer.fit_transform(X)
    
    # DBSCAN clustering
    db = DBSCAN(eps=0.5, min_samples=5).fit(emb)
    labels = db.labels_
    
    # Silhouette score
    sil = -1.0
    try:
        if len(set(labels)) > 1:
            sil = silhouette_score(emb, labels)
    except Exception as e:
        logger.warning(f"Could not compute silhouette score: {e}")
    
    logger.info(f"UMAP embedding shape: {emb.shape}, Silhouette score: {sil:.3f}")
    return emb, labels, sil

def partition_rule(names: List[str], X: np.ndarray, eigvecs: np.ndarray, 
                  sparsity: Dict[str, float], config: ExperimentConfig) -> Dict[str, str]:
    """根据规则划分参数类型"""
    logger.info("Partitioning parameters into functional/sensitive/mixed types")
    
    labels = {}
    for i, name in enumerate(names):
        sp = sparsity.get(name, 0.0)
        
        # Cosine alignment proxy
        if eigvecs.size > 0:
            arr = X[i, :eigvecs.shape[0]]
            cos = np.mean(np.abs(arr))  # Proxy for cosine similarity
        else:
            cos = 0.0
        
        # Apply partition rules
        if cos >= config.func_cos_min and sp < config.func_sparsity_max:
            labels[name] = 'Wfunc'
        elif cos <= config.sens_cos_max and sp >= config.sens_sparsity_min:
            labels[name] = 'Wsens'
        else:
            labels[name] = 'Wboth'
    
    # Count labels
    label_counts = {}
    for label in labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info(f"Partition results: {label_counts}")
    return labels

# -----------------------------
# Noise Injection and Evaluation
# -----------------------------
def inject_noise_and_eval(model: ModelWrapper, name_to_mask: Dict[str, str], 
                         sigma: float, texts_eval: List[str]) -> Tuple[float, Optional[float]]:
    """注入噪声并评估"""
    original = {}
    device = model.device
    
    # Store original parameters
    for name, p in model.model.named_parameters():
        if name in name_to_mask:
            original[name] = p.data.clone()
            noise = torch.randn_like(p.data) * sigma
            p.data.add_(noise)
    
    # Evaluate
    ppl = model.compute_ppl(texts_eval, batch_size=8)
    mia_auc = None  # TODO: Implement MIA evaluation
    
    # Restore original parameters
    for name, p in model.model.named_parameters():
        if name in original:
            p.data.copy_(original[name])
    
    return ppl, mia_auc

# -----------------------------
# Main Experiment Functions
# -----------------------------
def pilot_run(config: ExperimentConfig) -> str:
    """运行pilot实验"""
    logger.info("Starting pilot experiment")
    
    # Set seed
    set_seed(config.seed)
    
    # Load model
    mw = ModelWrapper(config.model_name, config.device)
    
    # Load data
    calib = get_calib_texts((config.calib_name, config.calib_subset), config.calib_samples)
    evals = get_calib_texts((config.calib_name, config.calib_subset), config.eval_samples)
    
    # Baseline PPL
    baseline_ppl = mw.compute_ppl(evals[:50], batch_size=config.batch_size)
    logger.info(f"Baseline PPL: {baseline_ppl:.3f}")
    
    # Compute eigenvectors
    vecs, vals = approx_top_eigvecs(mw, calib, topk=config.topk_eigenvectors, device=config.device)
    logger.info(f"Eigenvectors shape: {vecs.shape}")
    
    # Compute activation sparsity
    sparsity = compute_activation_sparsity(mw, calib, device=config.device)
    logger.info(f"Computed sparsity for {len(sparsity)} modules")
    
    # Build feature matrix
    names, X = build_param_feature_matrix(mw, vecs, sparsity)
    
    # UMAP and clustering
    emb, clabels, sil = cluster_and_label(names, X, config.umap_n_neighbors, config.umap_min_dist)
    logger.info(f"UMAP embedding shape: {emb.shape}, Silhouette score: {sil:.3f}")
    
    # Partition parameters
    name_labels = partition_rule(names, X, vecs, sparsity, config)
    
    # Group parameters by label
    groups = {'Wfunc': [], 'Wsens': [], 'Wboth': []}
    for n, lab in name_labels.items():
        groups[lab].append(n)
    
    logger.info(f"Group sizes: { {k: len(v) for k, v in groups.items()} }")
    
    # Run noise injection experiments
    results = []
    logger.info("Starting noise injection experiments")
    
    for lab in ['Wfunc', 'Wsens', 'Wboth']:
        target_names = set(groups[lab])
        if not target_names:
            continue
            
        mask = {n: lab for n in target_names}
        
        for sigma in config.noise_sigmas:
            for r in range(config.repeats):
                try:
                    ppl, mia = inject_noise_and_eval(mw, mask, sigma, evals[:50])
                    results.append({
                        'label': lab,
                        'sigma': sigma,
                        'repeat': r,
                        'ppl': ppl,
                        'mia': mia,
                        'delta_ppl': ppl - baseline_ppl
                    })
                    logger.info(f"{lab} σ={sigma:.1e} r={r} PPL={ppl:.3f} ΔPPL={ppl-baseline_ppl:.3f}")
                except Exception as e:
                    logger.error(f"Error in noise injection: {e}")
                    continue
    
    # Save results
    out_path = os.path.join(config.output_dir, 'pilot_results.json')
    save_json(results, out_path)
    logger.info(f'Saved results to {out_path}')
    
    # Save UMAP plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=clabels, cmap='Spectral', s=20, alpha=0.7)
    plt.colorbar(scatter)
    plt.title('UMAP Parameter Embedding (Pilot)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.savefig(os.path.join(config.output_dir, 'pilot_umap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save results as CSV
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(config.output_dir, 'pilot_results.csv')
        save_csv(df, csv_path)
        logger.info(f'Saved CSV results to {csv_path}')
    
    return out_path

def main():
    """主函数"""
    logger.info("Starting GeoQuantization PRM-FSGD Experiment")
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration: {config}")
    
    try:
        # Run pilot experiment
        result_path = pilot_run(config)
        logger.info(f"Pilot experiment completed successfully. Results saved to: {result_path}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == '__main__':
    main()
