#!/usr/bin/env python3
"""
动态生成数据集配置文件，自动适应当前工作目录
"""

import yaml
from pathlib import Path
import os

def create_dataset_config():
    """创建自适应的数据集配置文件"""
    
    # 获取当前工作目录
    current_dir = Path.cwd()
    dataset_path = current_dir / "dataset"
    
    print(f"当前工作目录: {current_dir}")
    print(f"数据集路径: {dataset_path}")
    
    # 检查数据集目录是否存在
    if not dataset_path.exists():
        print("❌ 错误: dataset目录不存在")
        return False
    
    # 创建配置
    config = {
        'path': str(dataset_path),  # 使用绝对路径
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 15,
        'names': {
            0: 'plane',
            1: 'ship', 
            2: 'storage-tank',
            3: 'baseball-diamond',
            4: 'tennis-court',
            5: 'basketball-court',
            6: 'ground-track-field',
            7: 'harbor',
            8: 'bridge',
            9: 'large-vehicle',
            10: 'small-vehicle',
            11: 'helicopter',
            12: 'roundabout',
            13: 'soccer-ball-field',
            14: 'swimming-pool'
        }
    }
    
    # 保存配置文件
    config_file = current_dir / "dota_dataset.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✓ 配置文件已创建: {config_file}")
    
    # 验证路径
    print("\n验证数据集路径:")
    for split in ['train', 'val', 'test']:
        if split in config:
            img_path = dataset_path / config[split]
            label_path = dataset_path / config[split].replace('images', 'labels')
            
            img_exists = img_path.exists()
            label_exists = label_path.exists()
            
            print(f"  {split}集:")
            print(f"    图像: {img_path} - {'✓' if img_exists else '✗'}")
            print(f"    标签: {label_path} - {'✓' if label_exists else '✗'}")
            
            if img_exists:
                img_count = len(list(img_path.glob('*.png')) + list(img_path.glob('*.jpg')))
                print(f"    图像数量: {img_count}")
            
            if label_exists:
                label_count = len(list(label_path.glob('*.txt')))
                print(f"    标签数量: {label_count}")
    
    return True

if __name__ == "__main__":
    create_dataset_config()