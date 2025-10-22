#!/usr/bin/env python3
"""
Git清理脚本
清理应该被忽略但已经被跟踪的文件
"""

import os
import subprocess
import sys
from pathlib import Path

def check_git_repo():
    """检查是否是Git仓库"""
    try:
        result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_tracked_files():
    """获取已跟踪的文件"""
    try:
        result = subprocess.run(['git', 'ls-files'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError:
        return []

def should_be_ignored(filename):
    """检查文件是否应该被忽略"""
    # 检查文件是否匹配.gitignore规则
    try:
        result = subprocess.run(['git', 'check-ignore', filename], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def find_files_to_clean():
    """查找需要清理的文件"""
    print("查找需要清理的文件...")
    
    tracked_files = get_tracked_files()
    files_to_clean = []
    
    # 常见的应该被忽略的文件模式
    ignore_patterns = [
        '*.log',
        '*.aux',
        '*.out',
        '*.toc',
        '*.fdb_latexmk',
        '*.fls',
        '*.synctex.gz',
        '*.pytxcode',
        '*.bbl',
        '*.blg',
        '*.idx',
        '*.ind',
        '*.ilg',
        '*.lof',
        '*.lot',
        '*.nav',
        '*.snm',
        '*.vrb',
        '*.pdf',
        '*.png',
        '*.jpg',
        '*.jpeg',
        '*.gif',
        '*.svg',
        '*.csv',
        '*.json',
        '*.pkl',
        '*.pickle',
        '*.bin',
        '*.pt',
        '*.pth',
        '*.ckpt',
        '*.h5',
        '*.hdf5',
        '*.onnx',
        '*.tflite',
        '*.pb',
        '*.exe',
        '*.dll',
        '*.so',
        '*.dylib',
        '*.a',
        '*.lib',
        '*.zip',
        '*.tar',
        '*.tar.gz',
        '*.tar.bz2',
        '*.rar',
        '*.7z',
        '*.db',
        '*.sqlite',
        '*.sqlite3',
        '*.tmp',
        '*.temp',
        '*.bak',
        '*.backup',
        '*.old',
        '*.orig',
        '*.large',
        '*.big',
        '*.huge',
        '*.exe',
        '*.dll',
        '*.so',
        '*.dylib',
        '*.a',
        '*.lib',
        '*.exe',
        '*.dll',
        '*.so',
        '*.dylib',
        '*.a',
        '*.lib'
    ]
    
    for file in tracked_files:
        if not file:
            continue
            
        # 检查文件是否应该被忽略
        if should_be_ignored(file):
            files_to_clean.append(file)
            continue
            
        # 检查文件扩展名
        file_path = Path(file)
        if file_path.suffix.lower() in ['.log', '.aux', '.out', '.toc', '.fdb_latexmk', 
                                       '.fls', '.synctex.gz', '.pytxcode', '.bbl', 
                                       '.blg', '.idx', '.ind', '.ilg', '.lof', '.lot', 
                                       '.nav', '.snm', '.vrb', '.pdf', '.png', '.jpg', 
                                       '.jpeg', '.gif', '.svg', '.csv', '.json', '.pkl', 
                                       '.pickle', '.bin', '.pt', '.pth', '.ckpt', '.h5', 
                                       '.hdf5', '.onnx', '.tflite', '.pb', '.exe', '.dll', 
                                       '.so', '.dylib', '.a', '.lib', '.zip', '.tar', 
                                       '.tar.gz', '.tar.bz2', '.rar', '.7z', '.db', 
                                       '.sqlite', '.sqlite3', '.tmp', '.temp', '.bak', 
                                       '.backup', '.old', '.orig', '.large', '.big', '.huge']:
            files_to_clean.append(file)
    
    return files_to_clean

def clean_files(files_to_clean, dry_run=True):
    """清理文件"""
    if not files_to_clean:
        print("✅ 没有需要清理的文件")
        return True
    
    print(f"\n找到 {len(files_to_clean)} 个需要清理的文件:")
    for file in files_to_clean:
        print(f"  - {file}")
    
    if dry_run:
        print("\n⚠️  这是预览模式，没有实际删除文件")
        print("要实际删除文件，请运行: python cleanup_git.py --execute")
        return True
    
    # 确认删除
    choice = input(f"\n确定要删除这 {len(files_to_clean)} 个文件吗? (y/n): ").lower().strip()
    if choice != 'y':
        print("已取消删除")
        return False
    
    # 删除文件
    success_count = 0
    for file in files_to_clean:
        try:
            # 从Git中删除文件
            subprocess.run(['git', 'rm', '--cached', file], check=True)
            print(f"✅ 已从Git中删除: {file}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ 删除失败: {file} - {e}")
    
    print(f"\n✅ 成功删除 {success_count}/{len(files_to_clean)} 个文件")
    return True

def main():
    """主函数"""
    print("🧹 Git清理工具")
    print("=" * 50)
    
    # 检查是否是Git仓库
    if not check_git_repo():
        print("❌ 当前目录不是Git仓库")
        return 1
    
    # 检查参数
    dry_run = '--execute' not in sys.argv
    
    if dry_run:
        print("🔍 预览模式 - 不会实际删除文件")
    else:
        print("⚠️  执行模式 - 将实际删除文件")
    
    # 查找需要清理的文件
    files_to_clean = find_files_to_clean()
    
    # 清理文件
    success = clean_files(files_to_clean, dry_run)
    
    if success and not dry_run:
        print("\n建议运行以下命令提交更改:")
        print("git add .")
        print("git commit -m 'Clean up ignored files'")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
