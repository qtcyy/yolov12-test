#!/usr/bin/env python3
"""
DOTA数据集准备脚本 - 转换DOTA格式到YOLO格式
"""

import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def reorganize_dataset_structure():
    """重组数据集目录结构以符合YOLO格式要求"""
    
    print("1. 创建YOLO格式目录结构...")
    
    # 创建必要的目录
    base_path = Path("dataset")
    
    # 图像目录
    (base_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base_path / "images" / "val").mkdir(parents=True, exist_ok=True) 
    (base_path / "images" / "test").mkdir(parents=True, exist_ok=True)
    
    # 标签目录
    (base_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (base_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (base_path / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    print("✓ 目录结构创建完成")
    
    # 移动图像文件
    print("\n2. 移动图像文件...")
    
    # 训练集图像
    src_train = base_path / "train" / "images" / "images"
    dst_train = base_path / "images" / "train"
    if src_train.exists():
        train_images = list(src_train.glob("*.png")) + list(src_train.glob("*.jpg"))
        print(f"   发现 {len(train_images)} 张训练图像")
        for img in tqdm(train_images, desc="   移动训练集图像"):
            shutil.copy2(img, dst_train / img.name)
    
    # 验证集图像
    src_val = base_path / "val" / "images" / "images"
    dst_val = base_path / "images" / "val"
    if src_val.exists():
        val_images = list(src_val.glob("*.png")) + list(src_val.glob("*.jpg"))
        print(f"   发现 {len(val_images)} 张验证图像")
        for img in tqdm(val_images, desc="   移动验证集图像"):
            shutil.copy2(img, dst_val / img.name)
    
    # 测试集图像
    src_test = base_path / "test" / "images" / "images"
    dst_test = base_path / "images" / "test"
    if src_test.exists():
        test_images = list(src_test.glob("*.png")) + list(src_test.glob("*.jpg"))
        print(f"   发现 {len(test_images)} 张测试图像")
        for img in tqdm(test_images, desc="   移动测试集图像"):
            shutil.copy2(img, dst_test / img.name)
    
    print("✓ 图像文件移动完成")
    return True


def convert_dota_to_yolo():
    """转换DOTA标注格式到YOLO格式"""
    
    print("\n3. 转换DOTA标注到YOLO格式...")
    
    # DOTA类别映射
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14
    }
    
    def convert_split(split_name):
        """转换特定数据集分割（train/val/test）"""
        base_path = Path("dataset")
        
        # 标签源目录 - 检查两个版本
        label_dirs = [
            base_path / split_name / "labelTxt-v1.0" / "labelTxt",
            base_path / split_name / "labelTxt-v1.5" / "labelTxt"
        ]
        
        label_dir = None
        for dir_path in label_dirs:
            if dir_path.exists():
                label_dir = dir_path
                break
        
        if not label_dir:
            print(f"   ⚠️ 未找到{split_name}集标签目录")
            return 0
        
        # 图像和标签目录
        img_dir = base_path / "images" / split_name
        save_dir = base_path / "labels" / split_name
        
        # 获取所有标签文件
        label_files = list(label_dir.glob("*.txt"))
        converted_count = 0
        
        for label_file in tqdm(label_files, desc=f"   转换{split_name}集标注"):
            img_name = label_file.stem
            
            # 查找对应的图像文件
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                temp_path = img_dir / f"{img_name}{ext}"
                if temp_path.exists():
                    img_path = temp_path
                    break
            
            if not img_path:
                continue
            
            # 获取图像尺寸
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # 读取并转换标注
            save_path = save_dir / f"{img_name}.txt"
            
            with open(label_file, 'r') as f, open(save_path, 'w') as g:
                lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    
                    # 跳过头部信息行
                    if len(parts) < 9 or parts[0] in ['imagesource:', 'gsd:']:
                        continue
                    
                    # 提取类别和坐标
                    try:
                        # DOTA格式: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                        coords = [float(p) for p in parts[:8]]
                        class_name = parts[8]
                        
                        if class_name not in class_mapping:
                            continue
                        
                        class_idx = class_mapping[class_name]
                        
                        # 转换为水平边界框 (获取最小外接矩形)
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]
                        
                        x_min = min(x_coords)
                        x_max = max(x_coords)
                        y_min = min(y_coords)
                        y_max = max(y_coords)
                        
                        # 转换为YOLO格式 (中心点坐标和宽高，归一化)
                        x_center = (x_min + x_max) / 2 / img_width
                        y_center = (y_min + y_max) / 2 / img_height
                        width = (x_max - x_min) / img_width
                        height = (y_max - y_min) / img_height
                        
                        # 写入YOLO格式
                        g.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        
                    except (ValueError, IndexError):
                        continue
            
            converted_count += 1
        
        return converted_count
    
    # 转换各个数据集分割
    train_count = convert_split("train")
    print(f"   ✓ 转换了 {train_count} 个训练集标注文件")
    
    val_count = convert_split("val")
    print(f"   ✓ 转换了 {val_count} 个验证集标注文件")
    
    test_count = convert_split("test")
    if test_count > 0:
        print(f"   ✓ 转换了 {test_count} 个测试集标注文件")
    
    print("✓ 标注转换完成")
    return train_count, val_count, test_count


def verify_dataset():
    """验证数据集完整性"""
    
    print("\n4. 验证数据集完整性...")
    
    base_path = Path("dataset")
    
    # 统计文件数量
    stats = {}
    for split in ["train", "val", "test"]:
        img_dir = base_path / "images" / split
        label_dir = base_path / "labels" / split
        
        if img_dir.exists():
            img_count = len(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        else:
            img_count = 0
            
        if label_dir.exists():
            label_count = len(list(label_dir.glob("*.txt")))
        else:
            label_count = 0
        
        stats[split] = {"images": img_count, "labels": label_count}
        
        print(f"   {split}集: {img_count} 张图像, {label_count} 个标注文件")
        
        if img_count > 0 and label_count > 0 and img_count != label_count:
            print(f"   ⚠️ 警告: {split}集图像和标注数量不匹配")
    
    print("✓ 数据集验证完成")
    return stats


def main():
    """主函数"""
    
    print("=" * 60)
    print("DOTA数据集准备工具")
    print("=" * 60)
    
    # 检查数据集目录是否存在
    if not Path("dataset").exists():
        print("❌ 错误: 未找到dataset目录")
        print("   请确保数据集已放置在dataset文件夹中")
        return
    
    # 执行数据准备步骤
    try:
        # 1. 重组目录结构
        reorganize_dataset_structure()
        
        # 2. 转换标注格式
        convert_dota_to_yolo()
        
        # 3. 验证数据完整性
        stats = verify_dataset()
        
        print("\n" + "=" * 60)
        print("✅ 数据集准备完成！")
        print("=" * 60)
        
        print("\n下一步:")
        print("1. 运行训练命令: python train_yolov12.py")
        print("2. 或使用命令行: yolo train model=yolov12n.pt data=dota_dataset.yaml")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()