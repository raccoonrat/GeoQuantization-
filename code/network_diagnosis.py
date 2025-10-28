#!/usr/bin/env python3
"""
网络诊断脚本
检查网络连接和镜像可用性
"""

import os
import sys
import socket
import requests
import time
from urllib.parse import urlparse

def test_basic_connectivity():
    """测试基本网络连接"""
    print("测试基本网络连接...")
    
    test_hosts = [
        ("google.com", 80),
        ("baidu.com", 80),
        ("hf-mirror.com", 443),
        ("huggingface.co", 443)
    ]
    
    results = {}
    
    for host, port in test_hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"✅ {host}:{port} - 连接成功")
                results[host] = True
            else:
                print(f"❌ {host}:{port} - 连接失败")
                results[host] = False
                
        except Exception as e:
            print(f"❌ {host}:{port} - 异常: {e}")
            results[host] = False
    
    return results

def test_http_requests():
    """测试HTTP请求"""
    print("\n测试HTTP请求...")
    
    test_urls = [
        "https://hf-mirror.com/api/models",
        "https://huggingface.co/api/models",
        "https://hf-mirror.com/facebook/opt-125m/resolve/main/tokenizer.json",
        "https://huggingface.co/facebook/opt-125m/resolve/main/tokenizer.json"
    ]
    
    results = {}
    
    for url in test_urls:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                print(f"✅ {url} - 成功 ({response.status_code}) - {end_time-start_time:.2f}s")
                results[url] = True
            else:
                print(f"⚠️  {url} - 状态码: {response.status_code}")
                results[url] = False
                
        except requests.exceptions.Timeout:
            print(f"⏰ {url} - 超时")
            results[url] = False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ {url} - 连接错误: {e}")
            results[url] = False
        except Exception as e:
            print(f"❌ {url} - 异常: {e}")
            results[url] = False
    
    return results

def test_dns_resolution():
    """测试DNS解析"""
    print("\n测试DNS解析...")
    
    hosts = [
        "hf-mirror.com",
        "huggingface.co",
        "google.com",
        "baidu.com"
    ]
    
    results = {}
    
    for host in hosts:
        try:
            ip = socket.gethostbyname(host)
            print(f"✅ {host} -> {ip}")
            results[host] = ip
        except Exception as e:
            print(f"❌ {host} - DNS解析失败: {e}")
            results[host] = None
    
    return results

def check_proxy_settings():
    """检查代理设置"""
    print("\n检查代理设置...")
    
    proxy_vars = [
        'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
        'SOCKS_PROXY', 'socks_proxy', 'ALL_PROXY', 'all_proxy'
    ]
    
    proxy_found = False
    
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}: {value}")
            proxy_found = True
    
    if not proxy_found:
        print("  未检测到代理设置")
    
    return proxy_found

def test_transformers_import():
    """测试transformers导入"""
    print("\n测试transformers导入...")
    
    try:
        import transformers
        print(f"✅ transformers版本: {transformers.__version__}")
        
        # 测试环境变量
        print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
        print(f"  HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', '未设置')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ transformers导入失败: {e}")
        return False

def suggest_solutions(connectivity_results, http_results):
    """建议解决方案"""
    print("\n" + "=" * 50)
    print("诊断结果和建议:")
    
    # 检查镜像可用性
    mirror_ok = any(url in http_results and http_results[url] 
                   for url in http_results if 'hf-mirror.com' in url)
    official_ok = any(url in http_results and http_results[url] 
                     for url in http_results if 'huggingface.co' in url)
    
    if mirror_ok:
        print("✅ 推荐使用镜像: https://hf-mirror.com")
        print("   运行: python force_mirror.py")
    elif official_ok:
        print("⚠️  镜像不可用，但官方站点可用")
        print("   运行: python setup_mirror.py")
    else:
        print("❌ 网络连接有问题")
        print("   建议:")
        print("   1. 检查网络连接")
        print("   2. 检查防火墙设置")
        print("   3. 尝试使用代理")
        print("   4. 使用离线模式: python offline_mode.py")

def main():
    """主函数"""
    print("🔍 网络诊断工具")
    print("=" * 50)
    
    # 基本连接测试
    connectivity_results = test_basic_connectivity()
    
    # HTTP请求测试
    http_results = test_http_requests()
    
    # DNS解析测试
    dns_results = test_dns_resolution()
    
    # 代理设置检查
    proxy_found = check_proxy_settings()
    
    # transformers导入测试
    transformers_ok = test_transformers_import()
    
    # 建议解决方案
    suggest_solutions(connectivity_results, http_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
