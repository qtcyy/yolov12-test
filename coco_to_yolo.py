#!/usr/bin/env python3
"""
COCO格式数据集转换为YOLO格式脚本
适用于将COCO JSON标注转换为YOLO TXT标注
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm


def create_yolo_structure():
    """创建YOLO标准目录结构"""
    print("1. 创建YOLO格式目录结构...")
    
    base_path = Path("dataset_yolo")
    
    # 创建目录结构
    for split in ["train", "val"]:
        (base_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    print("✓ YOLO目录结构创建完成")
    return base_path


def copy_images(base_path):
    """复制图像文件到YOLO目录结构"""
    print("\n2. 复制图像文件...")
    
    src_base = Path("dataset")
    
    for split in ["train", "val"]:
        src_dir = src_base / split
        dst_dir = base_path / "images" / split
        
        if src_dir.exists():
            # 获取所有图像文件
            image_files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
            print(f"   复制 {len(image_files)} 张{split}图像...")
            
            for img_file in tqdm(image_files, desc=f"   {split}集"):
                shutil.copy2(img_file, dst_dir / img_file.name)
        
    print("✓ 图像文件复制完成")


def convert_coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    转换COCO格式bbox到YOLO格式
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (归一化)
    """
    x_min, y_min, width, height = bbox
    
    # 计算中心点坐标
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    
    # 归一化
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]


def convert_annotations(base_path):
    """转换COCO标注到YOLO格式"""
    print("\n3. 转换标注格式...")
    
    src_base = Path("dataset")
    
    # 类别ID映射 (COCO category_id -> YOLO class_id)
    category_mapping = {
        1: 0,  # ship -> 0
        2: 1,  # people -> 1
        3: 2,  # car -> 2
        4: 3   # motor -> 3
    }
    
    for split in ["train", "val"]:
        annotation_file = src_base / "annotations" / f"{split}.json"
        
        if not annotation_file.exists():
            print(f"⚠️ 标注文件不存在: {annotation_file}")
            continue
            
        print(f"   处理{split}集标注...")
        
        # 加载COCO标注
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # 创建图像ID到文件名的映射
        image_info = {img['id']: img for img in coco_data['images']}
        
        # 按图像分组标注
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # 生成YOLO格式标注文件
        labels_dir = base_path / "labels" / split
        processed_count = 0
        
        # 遍历所有图像（而不是只遍历有标注的图像）
        for img_info in tqdm(coco_data['images'], desc=f"   转换{split}标注"):
            image_id = img_info['id']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # 创建对应的标注文件名
            img_filename = img_info['file_name']
            label_filename = Path(img_filename).stem + '.txt'
            label_path = labels_dir / label_filename
            
            # 获取这张图像的所有标注
            annotations = annotations_by_image.get(image_id, [])
            
            # 创建标注文件（如果没有标注则创建空文件）
            with open(label_path, 'w') as f:
                for ann in annotations:
                    category_id = ann['category_id']
                    if category_id not in category_mapping:
                        continue
                    
                    yolo_class_id = category_mapping[category_id]
                    bbox = ann['bbox']
                    
                    # 转换为YOLO格式
                    yolo_bbox = convert_coco_to_yolo_bbox(bbox, img_width, img_height)
                    
                    # 写入文件
                    line = f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}\n"
                    f.write(line)
            
            processed_count += 1
        
        print(f"   ✓ {split}集: 处理了 {processed_count} 张图像的标注")
    
    print("✓ 标注转换完成")


def create_dataset_yaml(base_path):
    """创建YOLO数据集配置文件"""
    print("\n4. 创建数据集配置文件...")
    
    yaml_content = f"""# COCO-4类 目标检测数据集配置
# 用于YOLOv12训练

# 数据集路径
path: {base_path.absolute()}  # 数据集根目录

# 数据集分割
train: images/train  # 训练集图像（相对于path）
val: images/val      # 验证集图像（相对于path）

# 类别数量
nc: 4

# 类别名称
names:
  0: ship      # 船舶
  1: people    # 人
  2: car       # 汽车
  3: motor     # 摩托车

# 数据增强参数（针对4类小数据集优化）
augment: True
hsv_h: 0.015     # 图像HSV-Hue增强（分数）
hsv_s: 0.7       # 图像HSV-Saturation增强（分数）
hsv_v: 0.4       # 图像HSV-Value增强（分数）
degrees: 0.0     # 图像旋转（+/- deg）
translate: 0.1   # 图像平移（+/- 分数）
scale: 0.5       # 图像缩放（+/- gain）
shear: 0.0       # 图像剪切（+/- deg）
perspective: 0.0 # 图像透视（+/- 分数）
flipud: 0.0      # 图像上下翻转（概率）
fliplr: 0.5      # 图像左右翻转（概率）
mosaic: 1.0      # 图像马赛克（概率）
mixup: 0.1       # 图像混合（概率）
copy_paste: 0.1  # 段复制粘贴（概率）
"""
    
    config_file = Path("coco_dataset.yaml")
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✓ 配置文件已创建: {config_file}")


def validate_conversion(base_path):
    """验证转换结果"""
    print("\n5. 验证转换结果...")
    
    all_good = True
    
    for split in ["train", "val"]:
        img_dir = base_path / "images" / split
        label_dir = base_path / "labels" / split
        
        if not img_dir.exists() or not label_dir.exists():
            print(f"❌ {split}集目录不存在")
            all_good = False
            continue
        
        # 统计文件数量
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        label_files = list(label_dir.glob("*.txt"))
        
        print(f"   {split}集: {len(img_files)} 张图像, {len(label_files)} 个标注文件")
        
        if len(img_files) != len(label_files):
            print(f"   ⚠️ {split}集: 图像和标注文件数量不匹配")
            all_good = False
        else:
            # 统计空标注文件数量
            empty_labels = 0
            for label_file in label_files[:10]:  # 只检查前10个以节省时间
                if label_file.stat().st_size == 0:
                    empty_labels += 1
            if empty_labels > 0:
                print(f"   ℹ️ {split}集: 检查发现有空标注文件（对应无目标图像，这是正常的）")
        
        # 检查一些标注文件的格式
        if label_files:
            sample_file = label_files[0]
            try:
                with open(sample_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"   ⚠️ 标注格式错误: {sample_file}")
                        all_good = False
                        break
                    
                    # 检查类别ID和坐标范围
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    if class_id < 0 or class_id > 3:
                        print(f"   ⚠️ 类别ID超出范围: {sample_file}")
                        all_good = False
                        break
                    
                    if any(c < 0 or c > 1 for c in coords):
                        print(f"   ⚠️ 坐标未正确归一化: {sample_file}")
                        all_good = False
                        break
            
            except Exception as e:
                print(f"   ⚠️ 读取标注文件失败: {sample_file}, 错误: {e}")
                all_good = False
    
    if all_good:
        print("✅ 转换验证通过！")
    else:
        print("❌ 转换验证失败，请检查上述问题")
    
    return all_good


def main():
    """主函数"""
    print("=" * 60)
    print("COCO to YOLO 数据集转换工具")
    print("=" * 60)
    
    # 检查源数据集是否存在
    src_path = Path("dataset")
    if not src_path.exists():
        print("❌ 错误: dataset目录不存在")
        print("   请确保COCO格式数据集在 'dataset' 目录中")
        return
    
    # 检查标注文件
    for split in ["train", "val"]:
        ann_file = src_path / "annotations" / f"{split}.json"
        if not ann_file.exists():
            print(f"❌ 错误: {ann_file} 不存在")
            return
    
    try:
        # 执行转换步骤
        base_path = create_yolo_structure()
        copy_images(base_path)
        convert_annotations(base_path)
        create_dataset_yaml(base_path)
        
        # 验证转换结果
        if validate_conversion(base_path):
            print("\n" + "=" * 60)
            print("✅ 转换完成！")
            print("=" * 60)
            print(f"\n转换后的数据集位置: {base_path}")
            print(f"配置文件: coco_dataset.yaml")
            print("\n下一步:")
            print("python train_yolov12.py --data coco_dataset.yaml")
        
    except Exception as e:
        print(f"\n❌ 转换过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()