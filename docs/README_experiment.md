# GeoQuantization PRM-FSGD 实验

基于几何意义的LLM量化离群点分析实验框架

## 实验概述

本实验实现了基于几何意义的功能型vs敏感型离群点分离方法，通过以下技术路径：

1. **几何探测（FSGD）**：使用Hessian特征向量和激活稀疏度分析参数几何结构
2. **扰动响应映射（PRM）**：通过噪声注入分析精度-隐私权衡
3. **统计验证**：使用UMAP降维和DBSCAN聚类验证几何可分离性

## 快速开始

### 1. 环境设置

```bash
# 克隆或下载项目
# 运行环境设置脚本
python setup_experiment.py
```

### 2. 运行实验

```bash
# 运行pilot实验（小规模验证）
python prm_experiment.py

# 查看结果
ls prm_outputs/
```

### 3. 结果分析

实验结果将保存在 `prm_outputs/` 目录中：

- `pilot_results.json` - 详细实验结果
- `pilot_results.csv` - 结果表格
- `pilot_umap.png` - 参数空间可视化
- `experiment.log` - 实验日志

## 实验配置

### 模型设置
- **默认模型**: `facebook/opt-125m` (用于pilot实验)
- **推荐模型**: `meta-llama/Llama-2-7b-hf` (完整实验)
- **设备**: 自动检测CUDA/CPU

### 数据集
- **校准集**: WikiText-2 (200样本)
- **评估集**: WikiText-2 (200样本)
- **最大长度**: 256 tokens

### 几何分析参数
- **特征向量数**: 50
- **UMAP邻居数**: 15
- **DBSCAN参数**: eps=0.5, min_samples=5

### 噪声注入
- **噪声强度**: [0, 1e-6, 1e-5, 1e-4, 1e-3]
- **重复次数**: 3次
- **评估指标**: ∆PPL, ∆AUC

## 实验流程

### 阶段1: 数据与模型准备
1. 加载预训练模型
2. 准备校准和评估数据集
3. 测量基线精度（PPL）

### 阶段2: 几何探测与子空间划分
1. **Hessian近似**: 通过梯度PCA计算主成分
2. **激活稀疏度**: 分析参数激活模式
3. **UMAP降维**: 可视化参数空间结构
4. **参数分类**: 划分为Wfunc/Wsens/Wboth三类

### 阶段3: 扰动响应映射
1. **噪声注入**: 对每类参数施加不同强度噪声
2. **精度评估**: 计算∆PPL变化
3. **隐私评估**: 计算∆AUC变化（待实现）
4. **相图绘制**: 绘制PRM轨迹

## 关键指标

### 几何指标
- **主成分对齐度**: cos ≥ 0.7 (功能型)
- **激活稀疏度**: ≥ 0.9 (敏感型)
- **曲率指标**: L2范数

### 统计指标
- **∆PPL**: 困惑度变化
- **∆AUC**: 隐私泄露变化
- **轮廓系数**: 聚类质量

### 阈值设置
- **功能型**: cos ≥ 0.7 且 稀疏度 < 0.5
- **敏感型**: cos ≤ 0.3 且 稀疏度 ≥ 0.9
- **混合型**: 其他情况

## 输出文件说明

### 结果文件
- `pilot_results.json`: 完整实验结果（JSON格式）
- `pilot_results.csv`: 结果表格（CSV格式）

### 可视化文件
- `pilot_umap.png`: 参数空间UMAP可视化
- `prm_phase_diagram.png`: PRM相图（待生成）

### 日志文件
- `experiment.log`: 详细实验日志

## 扩展实验

### 完整实验（待实现）
```python
# 修改配置使用更大模型
config.model_name = "meta-llama/Llama-2-7b-hf"
config.calib_samples = 1000
config.eval_samples = 1000

# 运行完整实验
python full_experiment.py
```

### MIA评估（待实现）
- 实现成员推断攻击评估
- 计算∆AUC指标
- 生成隐私-精度权衡曲线

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 使用CPU模式
   export CUDA_VISIBLE_DEVICES=""
   python prm_experiment.py
   ```

2. **模型下载失败**
   ```bash
   # 设置HuggingFace镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **依赖包冲突**
   ```bash
   # 创建虚拟环境
   python -m venv geo_env
   source geo_env/bin/activate  # Linux/Mac
   # 或
   geo_env\Scripts\activate  # Windows
   ```

### 性能优化

1. **减少内存使用**
   - 降低batch_size
   - 使用gradient checkpointing
   - 启用混合精度训练

2. **加速计算**
   - 使用多GPU并行
   - 启用编译优化
   - 减少重复计算

## 引用

如果您使用了本实验框架，请引用相关论文：

```bibtex
@article{geo_quantization_2024,
  title={基于几何意义的LLM量化离群点分析},
  author={GeoQuantization Team},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue到GitHub仓库
- 发送邮件至项目维护者

---

**注意**: 本实验需要大量计算资源，建议在GPU服务器上运行完整实验。
