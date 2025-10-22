#!/usr/bin/env python3
"""
代理配置脚本
解决SOCKS5代理相关问题
"""

import os
import sys
import subprocess
import requests
from urllib.parse import urlparse

def check_proxy_settings():
    """检查当前代理设置"""
    print("检查当前代理设置...")
    
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 
                  'ALL_PROXY', 'all_proxy', 'SOCKS_PROXY', 'socks_proxy']
    
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var}: 未设置")

def configure_proxy_for_huggingface():
    """为HuggingFace配置代理"""
    print("\n配置HuggingFace代理设置...")
    
    # 检查是否有SOCKS5代理
    socks_proxy = None
    for var in ['SOCKS_PROXY', 'socks_proxy', 'ALL_PROXY', 'all_proxy']:
        value = os.environ.get(var)
        if value and 'socks5' in value.lower():
            socks_proxy = value
            break
    
    if socks_proxy:
        print(f"检测到SOCKS5代理: {socks_proxy}")
        
        # 解析SOCKS5代理
        parsed = urlparse(socks_proxy)
        if parsed.scheme == 'socks5h':
            # 将socks5h转换为socks5
            new_proxy = socks_proxy.replace('socks5h://', 'socks5://')
            print(f"转换代理协议: {socks_proxy} -> {new_proxy}")
            
            # 设置环境变量
            os.environ['HTTP_PROXY'] = new_proxy
            os.environ['HTTPS_PROXY'] = new_proxy
            os.environ['http_proxy'] = new_proxy
            os.environ['https_proxy'] = new_proxy
            
            print("已设置HTTP/HTTPS代理环境变量")
    
    # 设置HuggingFace特定配置
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    print("已配置HuggingFace设置")

def test_connection():
    """测试网络连接"""
    print("\n测试网络连接...")
    
    test_urls = [
        'https://huggingface.co',
        'https://hf-mirror.com',
        'https://www.google.com'
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"✅ {url}: {response.status_code}")
        except Exception as e:
            print(f"❌ {url}: {e}")

def setup_mirror():
    """设置HuggingFace镜像"""
    print("\n设置HuggingFace镜像...")
    
    # 设置镜像环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("已设置HF_ENDPOINT为镜像地址")
    
    # 创建HuggingFace配置目录
    hf_cache_dir = os.path.expanduser('~/.cache/huggingface')
    os.makedirs(hf_cache_dir, exist_ok=True)
    
    # 创建配置文件
    config_file = os.path.join(hf_cache_dir, 'hub', 'config.json')
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    config_content = {
        "endpoint": "https://hf-mirror.com",
        "local_files_only": False,
        "use_auth_token": False
    }
    
    import json
    with open(config_file, 'w') as f:
        json.dump(config_content, f, indent=2)
    
    print(f"已创建配置文件: {config_file}")

def disable_proxy_for_experiment():
    """为实验禁用代理（如果需要）"""
    print("\n为实验配置网络设置...")
    
    # 保存原始代理设置
    original_proxies = {}
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    
    for var in proxy_vars:
        original_proxies[var] = os.environ.get(var)
    
    # 询问用户是否要禁用代理
    print("当前代理设置:")
    for var, value in original_proxies.items():
        if value:
            print(f"  {var}: {value}")
    
    choice = input("\n是否要为实验禁用代理? (y/n): ").lower().strip()
    
    if choice == 'y':
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
        print("已禁用代理设置")
    else:
        print("保持当前代理设置")

def main():
    """主函数"""
    print("🔧 代理配置工具")
    print("=" * 50)
    
    # 检查当前设置
    check_proxy_settings()
    
    # 配置代理
    configure_proxy_for_huggingface()
    
    # 设置镜像
    setup_mirror()
    
    # 测试连接
    test_connection()
    
    # 询问是否禁用代理
    disable_proxy_for_experiment()
    
    print("\n✅ 代理配置完成!")
    print("\n现在可以运行实验:")
    print("python prm_experiment.py")

if __name__ == "__main__":
    main()
