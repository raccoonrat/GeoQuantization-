# 🔧 SOCKS5代理问题解决方案

## 问题描述

您遇到的错误：
```
unsupported scheme socks5h
```

这个错误表明系统正在尝试使用`socks5h://`协议，但某些HTTP客户端不支持这种协议格式。

## 解决方案

### 1. 自动修复脚本

我已经创建了多个脚本来解决这个问题：

#### `setup_proxy.py` - 完整代理配置工具
```bash
python setup_proxy.py
```
- 自动检测SOCKS5代理设置
- 将`socks5h://`转换为`socks5://`
- 配置HuggingFace镜像
- 测试网络连接

#### `test_proxy.py` - 代理测试工具
```bash
python test_proxy.py
```
- 测试代理连接
- 验证HuggingFace访问
- 检查模型下载功能

#### `run_with_proxy.py` - 带代理支持的启动脚本
```bash
python run_with_proxy.py
```
- 自动配置代理环境
- 运行实验
- 处理所有代理相关问题

### 2. 手动修复方法

#### 方法1：环境变量设置
```bash
# 检查当前代理设置
echo $SOCKS_PROXY
echo $ALL_PROXY

# 如果使用socks5h，转换为socks5
export HTTP_PROXY="socks5://your-proxy:port"
export HTTPS_PROXY="socks5://your-proxy:port"
```

#### 方法2：使用镜像
```bash
# 设置HuggingFace镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 禁用遥测
export HF_HUB_DISABLE_TELEMETRY=1
```

### 3. 代码级修复

主实验脚本`prm_experiment.py`已经集成了代理处理：

```python
def configure_proxy():
    """配置代理设置"""
    # 自动检测并转换socks5h为socks5
    # 设置HuggingFace配置
    # 配置镜像地址
```

## 推荐使用步骤

### 快速解决
```bash
# 1. 运行代理配置工具
python setup_proxy.py

# 2. 测试代理连接
python test_proxy.py

# 3. 运行实验
python run_with_proxy.py
```

### 详细诊断
```bash
# 1. 检查代理设置
python -c "import os; print('SOCKS_PROXY:', os.environ.get('SOCKS_PROXY'))"

# 2. 测试网络连接
python -c "import requests; print(requests.get('https://huggingface.co').status_code)"

# 3. 测试模型下载
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('facebook/opt-125m')"
```

## 常见问题

### Q1: 仍然出现socks5h错误
**A**: 确保所有环境变量都使用`socks5://`而不是`socks5h://`

### Q2: 模型下载很慢
**A**: 使用镜像地址：
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### Q3: 代理认证失败
**A**: 在代理URL中包含用户名和密码：
```bash
export HTTP_PROXY="socks5://username:password@proxy:port"
```

## 验证修复

运行以下命令验证修复是否成功：

```bash
# 测试代理配置
python test_proxy.py

# 运行实验
python run_with_proxy.py
```

如果看到以下输出，说明修复成功：
```
✅ 模型下载成功!
✅ Experiment completed successfully!
```

## 技术细节

### 协议转换
- `socks5h://` → `socks5://`
- 原因：某些HTTP客户端不支持`socks5h`协议

### 环境变量映射
```bash
SOCKS_PROXY → HTTP_PROXY, HTTPS_PROXY
socks5h:// → socks5://
```

### HuggingFace配置
```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_PROGRESS_BARS=1
```

---

**注意**: 如果问题仍然存在，请检查您的代理服务器配置，确保它支持HTTP/HTTPS流量通过SOCKS5协议。
