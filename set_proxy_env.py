#!/usr/bin/env python3
"""
设置代理环境变量
"""

import os
import sys

def set_proxy_environment():
    """设置代理环境变量"""
    print("设置代理环境变量...")
    
    # 检查当前代理设置
    current_proxy = os.environ.get('SOCKS_PROXY') or os.environ.get('socks_proxy') or os.environ.get('ALL_PROXY') or os.environ.get('all_proxy')
    
    if current_proxy:
        print(f"当前代理: {current_proxy}")
        
        # 如果是socks5h，转换为socks5
        if 'socks5h://' in current_proxy:
            new_proxy = current_proxy.replace('socks5h://', 'socks5://')
            print(f"转换代理: {current_proxy} -> {new_proxy}")
            
            # 设置HTTP代理环境变量
            os.environ['HTTP_PROXY'] = new_proxy
            os.environ['HTTPS_PROXY'] = new_proxy
            os.environ['http_proxy'] = new_proxy
            os.environ['https_proxy'] = new_proxy
            
            print("已设置HTTP/HTTPS代理环境变量")
        else:
            print("代理协议无需转换")
    else:
        print("未检测到代理设置")
    
    # 设置HuggingFace配置
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    # 设置镜像
    if not os.environ.get('HF_ENDPOINT'):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("已设置HuggingFace镜像")
    
    print("环境变量设置完成!")

def main():
    """主函数"""
    set_proxy_environment()
    
    # 显示当前环境变量
    print("\n当前环境变量:")
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'HF_ENDPOINT']
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}: {value}")

if __name__ == "__main__":
    main()
