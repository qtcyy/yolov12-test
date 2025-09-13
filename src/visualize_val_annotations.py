#!/usr/bin/env python3
"""
验证集标注可视化脚本
将验证集中的标准标注数据在图像中框出，保存到指定文件夹

使用方法:
    python src/visualize_val_annotations.py [--max_images 100] [--show_stats]
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="验证集标注可视化")

    # 输入输出参数
    parser.add_argument(
        "--images_dir",
        type=str,
        default="../dataset_yolo/images/val",
        help="验证集图像目录"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="../dataset_yolo/labels/val",
        help="验证集标注目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../dataset_yolo/val_standard",
        help="可视化输出目录"
    )

    # 处理选项
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="最大处理图像数量（用于测试）"
    )
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="显示详细统计信息"
    )
    parser.add_argument(
        "--box_thickness",
        type=int,
        default=2,
        help="边界框线条粗细"
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=0.6,
        help="标签字体大小"
    )

    return parser.parse_args()


def load_class_names():
    """加载类别名称（与coco_dataset.yaml一致）"""
    return {
        0: "ship",
        1: "people",
        2: "car",
        3: "motor"
    }


def get_class_colors():
    """获取各类别的颜色（BGR格式）"""
    return {
        0: (255, 255, 0),  # ship - 青色
        1: (255, 0, 255),  # people - 紫色
        2: (0, 255, 255),  # car - 黄色
        3: (0, 255, 0)  # motor - 绿色
    }


def read_yolo_annotation(label_path, img_width, img_height):
    """
    读取YOLO格式的标注文件

    Args:
        label_path: 标注文件路径
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        List[dict]: 边界框信息列表，每个元素包含class_id, x1, y1, x2, y2
    """
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 转换为像素坐标
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            width_px = width * img_width
            height_px = height * img_height

            # 计算边界框坐标
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            annotations.append({
                'class_id': class_id,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })

    return annotations


def draw_annotations(image, annotations, class_names, class_colors, box_thickness=2, font_scale=0.6):
    """
    在图像上绘制标注

    Args:
        image: 输入图像
        annotations: 标注列表
        class_names: 类别名称字典
        class_colors: 类别颜色字典
        box_thickness: 边界框线条粗细
        font_scale: 字体大小

    Returns:
        绘制后的图像
    """
    img_draw = image.copy()

    for ann in annotations:
        class_id = ann['class_id']
        x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']

        # 获取类别信息
        class_name = class_names.get(class_id, f"unknown_{class_id}")
        color = class_colors.get(class_id, (128, 128, 128))  # 默认灰色

        # 绘制边界框
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, box_thickness)

        # 准备标签文本
        label = f"{class_name}"

        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, 2
        )

        # 绘制文本背景
        label_x1 = x1
        label_y1 = y1 - text_height - 10
        label_x2 = x1 + text_width + 8
        label_y2 = y1

        # 确保标签在图像内
        if label_y1 < 0:
            label_y1 = y2
            label_y2 = y2 + text_height + 10

        cv2.rectangle(img_draw, (label_x1, label_y1), (label_x2, label_y2), color, -1)

        # 绘制文本
        text_x = label_x1 + 4
        text_y = label_y2 - 5
        cv2.putText(
            img_draw, label, (text_x, text_y), font, font_scale, (255, 255, 255), 2
        )

    return img_draw


def process_single_image(image_path, label_path, output_path, class_names, class_colors, args):
    """
    处理单张图像

    Args:
        image_path: 图像文件路径
        label_path: 标注文件路径
        output_path: 输出文件路径
        class_names: 类别名称字典
        class_colors: 类别颜色字典
        args: 命令行参数

    Returns:
        dict: 处理结果统计
    """
    try:
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            return {'success': False, 'error': 'Cannot read image', 'annotations': 0}

        img_height, img_width = image.shape[:2]

        # 读取标注
        annotations = read_yolo_annotation(label_path, img_width, img_height)

        # 绘制标注
        if annotations:
            image_with_annotations = draw_annotations(
                image, annotations, class_names, class_colors,
                args.box_thickness, args.font_scale
            )
        else:
            # 如果没有标注，直接复制原图
            image_with_annotations = image.copy()

        # 保存结果
        cv2.imwrite(str(output_path), image_with_annotations)

        return {
            'success': True,
            'annotations': len(annotations),
            'classes': [ann['class_id'] for ann in annotations]
        }

    except Exception as e:
        return {'success': False, 'error': str(e), 'annotations': 0}


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("验证集标注可视化")
    print("=" * 60)

    # 检查输入目录
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    if not images_dir.exists():
        print(f"❌ 错误: 图像目录不存在 {images_dir}")
        return

    if not labels_dir.exists():
        print(f"❌ 错误: 标注目录不存在 {labels_dir}")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载类别信息
    class_names = load_class_names()
    class_colors = get_class_colors()

    # 获取图像文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
        image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))

    image_files = sorted(image_files)

    if args.max_images:
        image_files = image_files[:args.max_images]

    print(f"\n找到 {len(image_files)} 张图像")

    if len(image_files) == 0:
        print("❌ 未找到任何图像文件")
        return

    # 显示处理配置
    print(f"\n处理配置:")
    print(f"  图像目录: {images_dir}")
    print(f"  标注目录: {labels_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  边界框粗细: {args.box_thickness}")
    print(f"  字体大小: {args.font_scale}")
    print(f"  显示统计: {args.show_stats}")

    # 统计信息
    stats = {
        'total_images': len(image_files),
        'processed_images': 0,
        'failed_images': 0,
        'total_annotations': 0,
        'class_counts': defaultdict(int),
        'processing_times': [],
        'start_time': time.time()
    }

    # 开始处理
    print(f"\n开始处理...")
    print("-" * 60)

    # 处理每张图像
    for image_path in tqdm(image_files, desc="处理进度"):
        start_time = time.time()

        # 构建路径
        image_name = image_path.stem
        label_path = labels_dir / f"{image_name}.txt"
        output_path = output_dir / f"{image_name}.jpg"

        # 处理图像
        result = process_single_image(
            image_path, label_path, output_path,
            class_names, class_colors, args
        )

        # 更新统计
        if result['success']:
            stats['processed_images'] += 1
            stats['total_annotations'] += result['annotations']

            # 统计各类别数量
            for class_id in result.get('classes', []):
                class_name = class_names.get(class_id, f"unknown_{class_id}")
                stats['class_counts'][class_name] += 1
        else:
            stats['failed_images'] += 1
            if args.show_stats:
                print(f"\n⚠️ 处理失败 {image_path.name}: {result.get('error', 'Unknown error')}")

        # 记录处理时间
        processing_time = time.time() - start_time
        stats['processing_times'].append(processing_time)

    # 计算总体统计
    total_time = time.time() - stats['start_time']
    avg_time = sum(stats['processing_times']) / len(stats['processing_times']) if stats['processing_times'] else 0
    fps = 1 / avg_time if avg_time > 0 else 0

    # 显示结果
    print("\n" + "=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)

    print(f"\n📊 处理统计:")
    print(f"  总图像数量: {stats['total_images']}")
    print(f"  成功处理: {stats['processed_images']}")
    print(f"  处理失败: {stats['failed_images']}")
    print(f"  总标注数量: {stats['total_annotations']}")
    print(f"  平均每张图像: {stats['total_annotations'] / stats['processed_images']:.2f} 个标注" if stats[
                                                                                                        'processed_images'] > 0 else "  平均每张图像: 0 个标注")

    print(f"\n⏱️ 处理时间:")
    print(f"  总处理时间: {total_time:.2f} 秒")
    print(f"  平均处理时间: {avg_time:.3f} 秒/张")
    print(f"  处理速度: {fps:.2f} FPS")

    print(f"\n📋 各类别统计:")
    for class_name, count in sorted(stats['class_counts'].items()):
        percentage = (count / stats['total_annotations']) * 100 if stats['total_annotations'] > 0 else 0
        color_info = ""
        for class_id, name in class_names.items():
            if name == class_name:
                color = class_colors.get(class_id, (128, 128, 128))
                color_info = f" (BGR: {color})"
                break
        print(f"  {class_name}: {count} ({percentage:.1f}%){color_info}")

    print(f"\n📁 输出信息:")
    print(f"  输出目录: {output_dir}")
    print(f"  输出图像: {stats['processed_images']} 张")

    if args.show_stats:
        print(f"\n💡 使用说明:")
        print(f"  - 青色边界框: ship (船舶)")
        print(f"  - 紫色边界框: people (人)")
        print(f"  - 黄色边界框: car (汽车)")
        print(f"  - 绿色边界框: motor (摩托车)")


if __name__ == "__main__":
    main()
