#!/usr/bin/env python3
"""
测试配置文件加载
"""

import yaml
import os

def test_config_loading():
    """测试配置文件加载"""
    print("Testing configuration loading...")
    
    # 测试配置文件是否存在
    config_path = "experiment_config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Config file {config_path} not found")
        return False
    
    # 测试YAML解析
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        print("SUCCESS: YAML parsing successful")
    except Exception as e:
        print(f"ERROR: YAML parsing failed: {e}")
        return False
    
    # 测试配置结构
    required_sections = ['model', 'dataset', 'experiment', 'geometry']
    for section in required_sections:
        if section not in config_dict:
            print(f"ERROR: Missing section: {section}")
            return False
        print(f"SUCCESS: Found section: {section}")
    
    # 测试关键字段
    model_config = config_dict.get('model', {})
    if 'name' not in model_config:
        print("ERROR: Missing model.name")
        return False
    print(f"SUCCESS: Model name: {model_config['name']}")
    
    print("SUCCESS: Configuration test passed!")
    return True

if __name__ == "__main__":
    test_config_loading()
