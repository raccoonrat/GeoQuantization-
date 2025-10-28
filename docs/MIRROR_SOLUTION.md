# 🔧 HuggingFace镜像配置解决方案

## 概述

我们已经配置了HuggingFace镜像网站，解决了模型和数据集下载问题。现在可以在非代理环境下正常下载和使用HuggingFace资源。

## 配置内容

### 1. 镜像设置
- **主镜像**: `https://hf-mirror.com`
- **备用镜像**: `https://huggingface.co`
- **自动选择**: 脚本会自动测试并选择最佳镜像

### 2. 环境变量配置
```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_PROGRESS_BARS=1
HF_HUB_OFFLINE=0
HF_HOME=./hf_cache
```

### 3. 缓存目录
- **本地缓存**: `./hf_cache`
- **自动创建**: 脚本会自动创建缓存目录
- **持久化**: 下载的模型和数据集会保存在本地

## 使用方法

### 方法1：使用镜像启动脚本（推荐）
```bash
python run_with_mirror.py
```

### 方法2：使用镜像配置工具
```bash
# 配置镜像
python setup_mirror.py

# 运行实验
python prm_experiment.py
```

### 方法3：测试镜像连接
```bash
python test_mirror.py
```

### 方法4：最小化实验
```bash
python minimal_experiment.py
```

## 配置文件更新

### `experiment_config.yaml`
添加了HuggingFace镜像配置：
```yaml
# HuggingFace Mirror Configuration
huggingface:
  endpoint: "https://hf-mirror.com"
  disable_telemetry: true
  disable_progress_bars: true
  offline: false
  cache_dir: "./hf_cache"
```

### 主实验脚本更新
- 自动配置镜像环境变量
- 创建本地缓存目录
- 应用配置文件中的镜像设置

## 创建的文件

### 核心脚本
1. **`setup_mirror.py`** - 镜像配置工具
2. **`run_with_mirror.py`** - 镜像启动脚本
3. **`test_mirror.py`** - 镜像测试工具

### 更新的文件
1. **`experiment_config.yaml`** - 添加镜像配置
2. **`prm_experiment.py`** - 集成镜像配置
3. **`minimal_experiment.py`** - 添加镜像设置

## 测试验证

### 连接测试
```bash
python test_mirror.py
```
输出示例：
```
✅ https://hf-mirror.com: 连接成功
✅ 模型下载成功!
✅ 数据集下载成功!
```

### 模型下载测试
- 测试模型：`facebook/opt-125m`
- 验证分词功能
- 检查缓存目录

### 数据集下载测试
- 测试数据集：`wikitext/wikitext-2-raw-v1`
- 验证数据加载
- 检查样本数量

## 优势

### 1. 下载速度
- 使用国内镜像，下载速度更快
- 减少网络延迟和超时问题

### 2. 稳定性
- 自动选择最佳镜像
- 支持镜像切换和故障转移

### 3. 缓存管理
- 本地缓存避免重复下载
- 支持离线使用已下载的模型

### 4. 配置灵活
- 支持配置文件自定义
- 环境变量覆盖配置

## 故障排除

### 问题1：镜像连接失败
**解决**：
```bash
# 测试所有镜像
python test_mirror.py

# 手动设置镜像
export HF_ENDPOINT=https://huggingface.co
```

### 问题2：模型下载失败
**解决**：
```bash
# 清理缓存
rm -rf hf_cache

# 重新下载
python setup_mirror.py
```

### 问题3：数据集下载失败
**解决**：
```bash
# 检查网络连接
ping hf-mirror.com

# 使用备用镜像
export HF_ENDPOINT=https://huggingface.co
```

## 推荐使用流程

### 首次使用
```bash
# 1. 测试镜像连接
python test_mirror.py

# 2. 配置镜像
python setup_mirror.py

# 3. 运行实验
python run_with_mirror.py
```

### 日常使用
```bash
# 直接运行（已配置镜像）
python run_with_mirror.py
```

### 最小化测试
```bash
# 快速验证
python minimal_experiment.py
```

## 验证成功

如果看到以下输出，说明镜像配置成功：

```
✅ 已设置HuggingFace镜像: https://hf-mirror.com
✅ 缓存目录: ./hf_cache
✅ 模型下载成功!
✅ 数据集下载成功!
✅ Experiment completed successfully!
```

## 注意事项

1. **首次下载**：模型和数据集首次下载需要时间
2. **缓存空间**：确保有足够的磁盘空间存储缓存
3. **网络连接**：确保网络连接稳定
4. **镜像更新**：定期检查镜像可用性

---

**现在您可以在非代理环境下正常下载和使用HuggingFace模型和数据集了！** 🎉
