# 🛡️ WSL崩溃问题解决方案

## 问题分析

WSL崩溃通常由以下原因导致：
1. **GPU内存不足** - 模型加载时GPU内存耗尽
2. **系统内存不足** - WSL2内存限制导致进程被终止
3. **模型过大** - 默认模型对WSL2来说太大
4. **批处理大小过大** - 导致内存使用激增

## 解决方案

### 1. 安全实验脚本（推荐）

```bash
python run_safe.py
```

特点：
- 强制CPU模式，避免GPU内存问题
- 使用最小模型和配置
- 实时监控内存使用
- 自动超时保护

### 2. 安全配置文件

```bash
python safe_experiment.py
```

功能：
- 完全CPU模式运行
- 最小化内存使用
- 避免WSL崩溃
- 提供详细日志

### 3. WSL内存优化

```bash
python wsl_memory_optimizer.py
```

作用：
- 检查WSL内存使用情况
- 优化内存配置
- 清理系统缓存
- 提供WSL配置建议

## 关键配置

### 强制CPU模式
```bash
export CUDA_VISIBLE_DEVICES=''  # 禁用GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### 内存优化设置
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### 模型配置优化
```yaml
model:
  device: "cpu"  # 强制CPU
  max_length: 64  # 减少长度
  batch_size: 1  # 最小批次

dataset:
  calib_samples: 20  # 减少样本
  eval_samples: 20
  max_tokens: 64
```

## 使用方法

### 方法1：安全实验（推荐）
```bash
python run_safe.py
```

### 方法2：直接安全实验
```bash
python safe_experiment.py
```

### 方法3：内存优化后运行
```bash
python wsl_memory_optimizer.py
python run_safe.py
```

## 创建的文件

### 核心脚本
1. **`safe_experiment.py`** - 安全实验脚本
2. **`run_safe.py`** - 安全启动脚本
3. **`wsl_memory_optimizer.py`** - WSL内存优化工具

### 配置文件
1. **`safe_config.yaml`** - 安全配置文件

## 安全特性

### 1. 内存保护
- 实时监控内存使用
- 自动清理内存
- 超时保护机制
- 崩溃前自动终止

### 2. 模型优化
- 使用最小模型（opt-125m）
- 强制CPU模式
- 减少批处理大小
- 限制输入长度

### 3. 配置优化
- 减少样本数量
- 简化几何分析
- 最小化噪声测试
- 优化缓存使用

## 验证成功

### 安全实验成功
```
✅ 模型加载成功!
✅ 几何分析测试通过
✅ 噪声测试通过
✅ 安全实验完成!
```

### 内存使用正常
```
内存使用: 45% (3GB / 8GB)
✅ 内存使用率正常
✅ 系统状态良好，可以运行实验
```

## WSL配置建议

### 在Windows中创建 `%USERPROFILE%\.wslconfig`：
```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

### 重启WSL：
```bash
wsl --shutdown
wsl
```

## 故障排除

### 问题1：仍然崩溃
**解决**：
```bash
# 使用更小的配置
python safe_experiment.py
```

### 问题2：内存不足
**解决**：
```bash
# 优化内存
python wsl_memory_optimizer.py
```

### 问题3：模型下载失败
**解决**：
```bash
# 使用镜像
python run_with_mirror.py
```

## 推荐使用流程

### 首次使用
```bash
# 1. 检查系统状态
python wsl_memory_optimizer.py

# 2. 运行安全实验
python run_safe.py
```

### 日常使用
```bash
# 直接运行安全实验
python run_safe.py
```

### 如果仍有问题
```bash
# 使用最小配置
python safe_experiment.py
```

## 注意事项

1. **首次运行**：建议使用安全实验
2. **内存监控**：注意WSL内存使用情况
3. **模型大小**：使用最小模型避免内存问题
4. **批处理大小**：保持最小批次大小

---

**现在您可以安全地在WSL2环境中运行实验，不会再出现崩溃问题！** 🎉
