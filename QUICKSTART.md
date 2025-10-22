# 🚀 GeoQuantization 快速启动指南

## 立即开始实验

### Windows 用户
```cmd
# 双击运行或在命令行执行
run_experiment.bat
```

### Linux/Mac 用户
```bash
# 运行启动脚本
./run_experiment.sh
```

### 手动运行
```bash
# 1. 设置环境
python setup_experiment.py

# 2. 运行实验
python prm_experiment.py

# 3. 查看结果
ls prm_outputs/
```

## 实验文件说明

### 核心文件
- `prm_experiment.py` - 主实验脚本
- `experiment_config.yaml` - 配置文件
- `requirements.txt` - Python依赖
- `setup_experiment.py` - 环境设置

### 启动脚本
- `run_experiment.bat` - Windows批处理
- `run_experiment.sh` - Linux/Mac脚本
- `run_experiment.py` - 跨平台Python启动器

### 输出目录
- `prm_outputs/` - 实验结果
- `prm_outputs/plots/` - 可视化图表
- `prm_outputs/results/` - 数据文件
- `prm_outputs/logs/` - 日志文件

## 实验参数调整

### 修改模型
编辑 `experiment_config.yaml`:
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # 更大模型
```

### 调整样本数
```yaml
dataset:
  calib_samples: 1000  # 增加校准样本
  eval_samples: 1000   # 增加评估样本
```

### 修改设备
```yaml
model:
  device: "cuda"  # 强制使用GPU
```

## 预期结果

### 输出文件
- `pilot_results.json` - 详细结果
- `pilot_results.csv` - 结果表格
- `pilot_umap.png` - 参数可视化
- `experiment.log` - 实验日志

### 关键指标
- **∆PPL**: 精度变化
- **∆AUC**: 隐私变化（待实现）
- **聚类质量**: 轮廓系数
- **参数分类**: Wfunc/Wsens/Wboth

## 故障排除

### 常见问题
1. **CUDA内存不足** → 使用CPU模式
2. **模型下载失败** → 检查网络连接
3. **依赖冲突** → 创建虚拟环境

### 性能优化
- 使用GPU加速
- 调整batch_size
- 启用混合精度

## 下一步

1. **查看结果**: 分析生成的CSV和图片
2. **调整参数**: 修改配置文件
3. **扩展实验**: 使用更大模型
4. **实现MIA**: 添加隐私评估

---

**注意**: 首次运行会自动下载模型和依赖包，请确保网络连接正常。
