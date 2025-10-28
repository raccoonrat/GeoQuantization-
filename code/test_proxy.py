#!/usr/bin/env python3
"""
代理测试脚本
"""

import os
import sys
import requests
from urllib.parse import urlparse

def test_proxy_connection():
    """测试代理连接"""
    print("测试代理连接...")
    
    # 检查代理设置
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                  'SOCKS_PROXY', 'socks_proxy', 'ALL_PROXY', 'all_proxy']
    
    print("\n当前代理设置:")
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}: {value}")
    
    # 测试连接
    test_urls = [
        'https://huggingface.co',
        'https://hf-mirror.com',
        'https://www.google.com'
    ]
    
    print("\n测试网络连接:")
    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"  ✅ {url}: {response.status_code}")
        except Exception as e:
            print(f"  ❌ {url}: {e}")

def fix_socks5_proxy():
    """修复SOCKS5代理问题"""
    print("\n修复SOCKS5代理问题...")
    
    # 查找SOCKS5代理
    socks_proxy = None
    for var in ['SOCKS_PROXY', 'socks_proxy', 'ALL_PROXY', 'all_proxy']:
        value = os.environ.get(var)
        if value and 'socks5' in value.lower():
            socks_proxy = value
            print(f"找到SOCKS5代理: {value}")
            break
    
    if socks_proxy:
        # 转换socks5h为socks5
        if 'socks5h://' in socks_proxy:
            new_proxy = socks_proxy.replace('socks5h://', 'socks5://')
            print(f"转换代理协议: {socks_proxy} -> {new_proxy}")
            
            # 设置HTTP代理
            os.environ['HTTP_PROXY'] = new_proxy
            os.environ['HTTPS_PROXY'] = new_proxy
            os.environ['http_proxy'] = new_proxy
            os.environ['https_proxy'] = new_proxy
            
            print("已设置HTTP/HTTPS代理环境变量")
    else:
        print("未找到SOCKS5代理设置")

def test_huggingface_access():
    """测试HuggingFace访问"""
    print("\n测试HuggingFace访问...")
    
    try:
        from transformers import AutoTokenizer
        print("正在测试模型下载...")
        
        # 尝试下载一个小模型
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        print("✅ 模型下载成功!")
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        
        # 尝试使用镜像
        print("尝试使用镜像...")
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
            print("✅ 使用镜像下载成功!")
        except Exception as e2:
            print(f"❌ 镜像下载也失败: {e2}")

def main():
    """主函数"""
    print("🔧 代理测试工具")
    print("=" * 50)
    
    # 修复SOCKS5代理
    fix_socks5_proxy()
    
    # 测试连接
    test_proxy_connection()
    
    # 测试HuggingFace
    test_huggingface_access()
    
    print("\n" + "=" * 50)
    print("测试完成!")

if __name__ == "__main__":
    main()
