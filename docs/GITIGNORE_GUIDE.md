# 📁 .gitignore 使用指南

## 概述

我已经为您创建了一个完整的`.gitignore`文件，用于忽略日志、中间结果、模型文件、数据集等文件。

## 文件内容

### 主要忽略类别

1. **Python相关**
   - `__pycache__/`
   - `*.pyc`
   - `*.pyo`
   - `build/`
   - `dist/`
   - `*.egg-info/`

2. **虚拟环境**
   - `venv/`
   - `env/`
   - `.venv/`
   - `.env/`

3. **IDE和编辑器**
   - `.vscode/`
   - `.idea/`
   - `*.swp`
   - `*.swo`

4. **实验输出和结果**
   - `prm_outputs/`
   - `outputs/`
   - `results/`
   - `logs/`
   - `*.log`
   - `*.json`
   - `*.csv`
   - `*.png`
   - `*.pdf`

5. **模型文件和权重**
   - `hf_cache/`
   - `models/`
   - `checkpoints/`
   - `*.bin`
   - `*.safetensors`
   - `*.pt`
   - `*.pth`
   - `*.ckpt`
   - `*.pkl`
   - `*.h5`

6. **数据集文件**
   - `data/`
   - `datasets/`
   - `*.csv`
   - `*.tsv`
   - `*.json`
   - `*.parquet`
   - `*.arrow`

7. **LaTeX编译文件**
   - `*.aux`
   - `*.log`
   - `*.out`
   - `*.toc`
   - `*.fdb_latexmk`
   - `*.fls`
   - `*.synctex.gz`
   - `*.pytxcode`
   - `*.bbl`
   - `*.blg`

8. **操作系统文件**
   - `.DS_Store`
   - `Thumbs.db`
   - `*.tmp`
   - `*.temp`

## 使用方法

### 1. 检查.gitignore效果

```bash
python check_gitignore.py
```

这个脚本会：
- 检查Git状态
- 显示被忽略的文件
- 检查特定文件是否被忽略
- 显示.gitignore文件摘要

### 2. 清理已跟踪的文件

```bash
# 预览模式（推荐先运行）
python cleanup_git.py

# 实际删除文件
python cleanup_git.py --execute
```

这个脚本会：
- 查找应该被忽略但已经被跟踪的文件
- 从Git中删除这些文件
- 提供预览模式避免误删

### 3. 手动检查

```bash
# 检查Git状态
git status

# 检查被忽略的文件
git status --ignored

# 检查特定文件是否被忽略
git check-ignore filename
```

## 常见问题

### Q1: 某些文件仍然被跟踪
**A**: 这些文件可能已经被Git跟踪，需要手动删除：
```bash
git rm --cached filename
git commit -m "Remove tracked file"
```

### Q2: 想要保留某些被忽略的文件
**A**: 在.gitignore文件中添加例外规则：
```gitignore
# 忽略所有.log文件
*.log

# 但保留important.log
!important.log
```

### Q3: 检查特定文件是否被忽略
**A**: 使用Git命令：
```bash
git check-ignore filename
```

## 项目特定配置

### 当前项目忽略的文件
- 实验方案PDF: `实验方案.pdf`
- 特定日志: `test_echo.log`
- 测试文件: `test_chinese.*`, `test_packages.*`

### 建议保留的文件
- 源代码文件: `*.py`
- 配置文件: `*.yaml`, `*.yml`
- 文档文件: `*.md`
- 依赖文件: `requirements.txt`

## 验证配置

### 检查Git状态
```bash
git status
```

应该只显示源代码文件，不显示：
- 日志文件
- 编译文件
- 模型文件
- 数据集文件

### 检查被忽略的文件
```bash
git status --ignored
```

应该显示所有被忽略的文件。

## 最佳实践

1. **定期检查**: 使用`check_gitignore.py`定期检查
2. **清理跟踪**: 使用`cleanup_git.py`清理已跟踪的文件
3. **更新规则**: 根据需要更新.gitignore文件
4. **团队协作**: 确保团队成员都使用相同的.gitignore文件

## 注意事项

1. **备份重要文件**: 在清理前确保重要文件已备份
2. **检查依赖**: 确保被忽略的文件不是项目必需的
3. **团队同步**: 更新.gitignore后及时提交到仓库
4. **测试配置**: 在提交前测试.gitignore配置

---

**现在您的项目已经配置了完整的.gitignore文件，可以避免提交不必要的文件！** 🎉
