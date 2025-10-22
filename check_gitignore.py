#!/usr/bin/env python3
"""
检查.gitignore文件效果
"""

import os
import subprocess
import sys
from pathlib import Path

def check_git_status():
    """检查Git状态"""
    print("检查Git状态...")
    
    try:
        # 检查Git状态
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print("当前Git状态:")
            print(result.stdout)
        else:
            print("✅ 工作目录干净，没有未跟踪的文件")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git状态检查失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ Git未安装或不在PATH中")
        return False

def check_ignored_files():
    """检查被忽略的文件"""
    print("\n检查被忽略的文件...")
    
    try:
        # 检查被忽略的文件
        result = subprocess.run(['git', 'status', '--ignored', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print("被忽略的文件:")
            for line in result.stdout.strip().split('\n'):
                if line.startswith('!!'):
                    print(f"  {line[3:]}")
        else:
            print("✅ 没有文件被忽略")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 检查忽略文件失败: {e}")
        return False

def check_specific_files():
    """检查特定文件是否被忽略"""
    print("\n检查特定文件...")
    
    # 检查常见的应该被忽略的文件
    test_files = [
        '*.log',
        '*.aux',
        '*.pdf',
        'prm_outputs/',
        'hf_cache/',
        '__pycache__/',
        '.DS_Store',
        'Thumbs.db'
    ]
    
    for pattern in test_files:
        try:
            result = subprocess.run(['git', 'check-ignore', pattern], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {pattern} - 被忽略")
            else:
                print(f"⚠️  {pattern} - 未被忽略")
                
        except Exception as e:
            print(f"❌ 检查 {pattern} 失败: {e}")

def show_gitignore_summary():
    """显示.gitignore文件摘要"""
    print("\n.gitignore文件摘要:")
    
    try:
        with open('.gitignore', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计不同类型的忽略规则
        lines = content.split('\n')
        categories = {}
        current_category = "其他"
        
        for line in lines:
            line = line.strip()
            if line.startswith('# ==========================================='):
                continue
            elif line.startswith('# ') and not line.startswith('# ='):
                current_category = line[2:]
                categories[current_category] = 0
            elif line and not line.startswith('#'):
                categories[current_category] = categories.get(current_category, 0) + 1
        
        print("忽略规则统计:")
        for category, count in categories.items():
            if count > 0:
                print(f"  {category}: {count} 条规则")
        
    except Exception as e:
        print(f"❌ 读取.gitignore文件失败: {e}")

def main():
    """主函数"""
    print("🔍 .gitignore文件检查工具")
    print("=" * 50)
    
    # 检查Git状态
    git_ok = check_git_status()
    
    # 检查被忽略的文件
    if git_ok:
        check_ignored_files()
        check_specific_files()
    
    # 显示.gitignore摘要
    show_gitignore_summary()
    
    print("\n" + "=" * 50)
    print("检查完成!")
    
    if git_ok:
        print("✅ Git状态正常")
    else:
        print("❌ Git状态异常，请检查Git配置")

if __name__ == "__main__":
    main()
