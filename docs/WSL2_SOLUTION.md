# 🐧 WSL2环境解决方案

## 问题分析

WSL2环境下的闪退和终止问题通常由以下原因导致：

1. **内存不足** - WSL2内存限制导致进程被终止
2. **编码问题** - Windows和Linux编码不兼容
3. **代理配置** - SOCKS5代理在WSL2中的兼容性问题
4. **依赖冲突** - Python包版本冲突

## 解决方案

### 1. 环境检查

首先运行环境检查脚本：

```bash
# Linux/WSL2
python check_wsl2.py

# Windows
python check_wsl2.py
```

### 2. 安全运行方式

#### 方式1：最小化测试（推荐）
```bash
# Linux/WSL2
python minimal_experiment.py

# Windows
python minimal_experiment.py
```

#### 方式2：WSL2专用启动器
```bash
# Linux/WSL2
python run_wsl2.py

# Windows
python run_wsl2.py
```

#### 方式3：交互式启动器
```bash
# Linux/WSL2
./run_wsl2_safe.sh

# Windows
run_wsl2_safe.bat
```

### 3. 内存优化配置

#### 环境变量设置
```bash
# 设置内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 设置编码
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

#### WSL2内存限制调整
在Windows中创建或编辑 `%USERPROFILE%\.wslconfig`：
```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
```

### 4. 代理配置优化

#### 自动代理转换
脚本会自动检测并转换SOCKS5代理：
- `socks5h://` → `socks5://`
- 设置HTTP/HTTPS代理环境变量

#### 镜像配置
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
```

## 推荐使用步骤

### 步骤1：环境检查
```bash
python check_wsl2.py
```

### 步骤2：选择运行方式

#### 首次使用（推荐最小化测试）
```bash
python minimal_experiment.py
```

#### 完整实验
```bash
python run_wsl2.py
```

#### 交互式选择
```bash
# Linux/WSL2
./run_wsl2_safe.sh

# Windows
run_wsl2_safe.bat
```

### 步骤3：查看结果
```bash
ls prm_outputs/
cat prm_outputs/minimal_results.json
```

## 故障排除

### 问题1：WSL2被终止
**原因**: 内存不足
**解决**: 
1. 调整WSL2内存限制
2. 使用最小化配置
3. 强制使用CPU模式

### 问题2：编码错误
**原因**: Windows/Linux编码不兼容
**解决**:
```bash
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### 问题3：代理连接失败
**原因**: SOCKS5协议不兼容
**解决**: 脚本自动转换协议格式

### 问题4：模型下载失败
**原因**: 网络连接问题
**解决**: 使用镜像地址
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 性能优化建议

### 1. 内存优化
- 使用CPU模式避免GPU内存问题
- 减少batch_size和max_length
- 启用内存清理

### 2. 网络优化
- 使用镜像地址
- 配置代理（如果需要）
- 禁用遥测和进度条

### 3. 计算优化
- 限制线程数
- 使用较小的模型
- 减少实验参数

## 文件说明

### 核心脚本
- `check_wsl2.py` - WSL2环境检查
- `minimal_experiment.py` - 最小化实验
- `run_wsl2.py` - WSL2专用启动器

### 启动脚本
- `run_wsl2_safe.sh` - Linux/WSL2启动脚本
- `run_wsl2_safe.bat` - Windows启动脚本

### 配置文件
- `wsl2_config.yaml` - WSL2最小化配置

## 验证成功

如果看到以下输出，说明配置成功：

```
✅ 环境检查通过
✅ 模型加载成功!
✅ 几何分析测试通过
✅ 最小化实验完成!
```

## 注意事项

1. **首次运行**：建议使用最小化测试
2. **内存监控**：注意WSL2内存使用情况
3. **网络连接**：确保代理配置正确
4. **编码设置**：避免编码冲突

---

**推荐使用顺序**：
1. `python check_wsl2.py` - 检查环境
2. `python minimal_experiment.py` - 最小化测试
3. `python run_wsl2.py` - 完整实验（如果最小化测试成功）
