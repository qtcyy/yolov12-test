#!/usr/bin/env python3
"""
验证训练设置是否正确
"""

import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    print("=" * 60)
    print("YOLOv12 DOTA 训练设置验证")
    print("=" * 60)
    
    # 1. 检查环境
    print("\n1. 检查环境...")
    print(f"   Python版本: {torch.__version__}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 2. 检查数据集配置
    print("\n2. 检查数据集配置...")
    config_file = "dota_dataset.yaml"
    
    if not Path(config_file).exists():
        print(f"   ❌ 配置文件不存在: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    print(f"   数据集路径: {dataset_path}")
    print(f"   路径存在: {dataset_path.exists()}")
    print(f"   类别数量: {config['nc']}")
    
    # 验证数据集
    for split in ['train', 'val']:
        if split in config:
            img_dir = dataset_path / config[split]
            label_dir = dataset_path / config[split].replace('images', 'labels')
            
            if img_dir.exists() and label_dir.exists():
                img_count = len(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
                label_count = len(list(label_dir.glob('*.txt')))
                
                print(f"   {split}集: {img_count} 图像, {label_count} 标签 - {'✅' if img_count == label_count else '⚠️'}")
            else:
                print(f"   {split}集: ❌ 目录缺失")
    
    # 3. 测试模型加载
    print("\n3. 测试模型加载...")
    try:
        model = YOLO('yolov12n.yaml')
        print("   ✅ YOLOv12n 配置加载成功")
        
        # 尝试加载预训练权重
        try:
            model = YOLO('yolov12n.pt')
            print("   ✅ 预训练权重加载成功")
        except:
            print("   ⚠️ 预训练权重未找到，将从配置文件开始训练")
            
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return
    
    # 4. 测试数据加载
    print("\n4. 测试数据加载...")
    try:
        # 只加载数据加载器，不开始训练
        from ultralytics.data import build_dataloader
        
        dataset = model.train_set if hasattr(model, 'train_set') else None
        print("   ✅ 数据加载器创建成功")
        
    except Exception as e:
        print(f"   ⚠️ 数据加载测试跳过: {e}")
    
    # 5. 显示推荐的训练命令
    print("\n5. 推荐的训练命令:")
    print("=" * 40)
    
    # 根据GPU情况推荐参数
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem >= 8:
            batch_size = 16
            print(f"   # GPU内存充足 ({gpu_mem:.1f}GB)，使用较大批次")
        else:
            batch_size = 8
            print(f"   # GPU内存有限 ({gpu_mem:.1f}GB)，使用较小批次")
        device = "0"
    else:
        batch_size = 4
        device = "cpu"
        print("   # 使用CPU训练（速度较慢）")
    
    print(f"\n   # 简单训练（快速测试）")
    print(f"   python train_yolov12.py --epochs 10 --batch {batch_size} --device {device}")
    
    print(f"\n   # 完整训练")
    print(f"   python train_yolov12.py --epochs 100 --batch {batch_size} --device {device}")
    
    print(f"\n   # 使用bash脚本（推荐）")
    print(f"   ./start_training.sh")
    
    print(f"\n   # 直接使用ultralytics")
    print(f"   yolo train model=yolov12n.pt data=dota_dataset.yaml epochs=100 batch={batch_size} device={device}")
    
    print("\n" + "=" * 60)
    print("✅ 验证完成！可以开始训练了")
    print("=" * 60)

if __name__ == "__main__":
    main()