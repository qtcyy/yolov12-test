import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv12验证集预测")

    # 模型和数据参数
    parser.add_argument(
        "--model", type=str, default="yolov12n.pt", help="预训练模型路径"
    )
    parser.add_argument(
        "--val_dir", type=str, default="dataset_yolo/images/val", help="验证集图像目录"
    )
    parser.add_argument(
        "--output_dir", type=str, default="predictions", help="输出结果目录"
    )

    # 预测参数
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像大小")
    parser.add_argument(
        "--device", type=str, default="cpu", help="推理设备 (0,1,2,3,cpu,mps)"
    )

    # 输出选项
    parser.add_argument("--save_images", action="store_true", help="保存带标注的图像")
    parser.add_argument(
        "--save_json", action="store_true", help="保存JSON格式的预测结果"
    )
    parser.add_argument("--save_txt", action="store_true", help="保存TXT格式的预测结果")
    parser.add_argument(
        "--max_images", type=int, default=None, help="最大处理图像数量（用于测试）"
    )
    parser.add_argument(
        "--filter_classes",
        action="store_true",
        help="只显示目标类别（person,car,motorcycle,boat）",
    )

    return parser.parse_args()


def setup_output_dirs(output_dir):
    """创建输出目录"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dirs = {
        "images": output_path / "images",
        "json": output_path / "json",
        "txt": output_path / "txt",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)

    return dirs


def draw_predictions(
    image, results, class_names, target_classes=None, filter_classes=False, coco_to_custom=None
):
    """在图像上绘制预测框"""
    img_draw = image.copy()

    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            # 获取坐标和信息
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())

            # 如果启用类别过滤或者类别不在目标类别中，跳过
            if cls not in target_classes:
                if filter_classes:
                    continue
                else:
                    # 如果不过滤，显示为unknown
                    class_name = f"unknown_{cls}"
            else:
                # 转换为自定义类别名称
                if coco_to_custom:
                    custom_id = coco_to_custom[cls]
                    class_name = class_names[custom_id]
                else:
                    class_name = f"class_{cls}"

            # 绘制边界框
            color = get_class_color(cls)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                img_draw,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                img_draw,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    return img_draw


def get_class_color(class_id):
    """获取类别颜色"""
    colors = [
        (0, 255, 255),  # ship - 青色
        (255, 0, 255),  # people - 紫色
        (255, 255, 0),  # car - 黄色
        (0, 255, 0),  # motor - 绿色
    ]
    return colors[class_id % len(colors)]


def results_to_json(
    results, image_name, class_names, target_classes=None, filter_classes=False, coco_to_custom=None
):
    """将预测结果转换为JSON格式"""
    predictions = []

    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            # 如果启用类别过滤或者类别不在目标类别中，跳过
            if cls not in target_classes:
                if filter_classes:
                    continue
                else:
                    class_name = f"unknown_{cls}"
                    custom_id = cls  # 保持原始ID
            else:
                # 转换为自定义类别
                if coco_to_custom:
                    custom_id = coco_to_custom[cls]
                    class_name = class_names[custom_id]
                else:
                    custom_id = cls
                    class_name = f"class_{cls}"

            prediction = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
                "class_id": custom_id,  # 使用转换后的类别ID
                "class_name": class_name,
                "original_class_id": cls,  # 保留原始COCO类别ID
            }
            predictions.append(prediction)

    return {"image": image_name, "predictions": predictions}


def results_to_txt(results, target_classes=None, filter_classes=False, coco_to_custom=None):
    """将预测结果转换为YOLO TXT格式"""
    lines = []

    if results[0].boxes is not None:
        boxes = results[0].boxes
        img_height, img_width = results[0].orig_shape

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            # 如果启用类别过滤，只保存目标类别
            if cls not in target_classes:
                if filter_classes:
                    continue
                else:
                    custom_id = cls  # 保持原始ID
            else:
                # 转换为自定义类别ID
                if coco_to_custom:
                    custom_id = coco_to_custom[cls]
                else:
                    custom_id = cls

            # 转换为YOLO格式（中心点坐标，归一化）
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            line = f"{custom_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}"
            lines.append(line)

    return lines


def auto_select_device(device_arg):
    """自动选择最佳设备"""
    import torch

    if device_arg != "cpu":
        # 用户指定了具体设备，直接使用
        return device_arg

    # 检查是否为Apple Silicon Mac并支持MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🍎 检测到Apple Silicon Mac，使用MPS加速")
        return "mps"

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("🚀 检测到CUDA GPU，使用GPU加速")
        return "0"

    print("💻 使用CPU运行")
    return "cpu"


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("YOLOv12 验证集预测")
    print("=" * 60)

    # 自动选择设备
    args.device = auto_select_device(args.device)

    # 检查输入目录
    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        print(f"❌ 错误: 验证集目录不存在 {val_dir}")
        return

    # 检查模型文件
    if not Path(args.model).exists():
        print(f"❌ 错误: 模型文件不存在 {args.model}")
        return

    # 创建输出目录
    output_dirs = setup_output_dirs(args.output_dir)

    # 加载模型
    print(f"\n加载模型: {args.model}")
    try:
        model = YOLO(args.model)
        print(f"✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # COCO类别到您的数据集类别的映射
    coco_to_custom = {
        0: 0,   # person -> people
        2: 1,   # car -> car  
        3: 2,   # motorcycle -> motor
        8: 3    # boat -> ship
    }
    
    # 您的4类数据集的类别名称
    class_names = {
        0: "people",
        1: "car", 
        2: "motor",
        3: "ship"
    }
    
    # 目标类别（COCO中您关心的类别ID）
    target_classes = {0, 2, 3, 8}  # person, car, motorcycle, boat

    # 获取图像文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(val_dir.glob(f"*{ext}")))
        image_files.extend(list(val_dir.glob(f"*{ext.upper()}")))

    image_files = sorted(image_files)

    if args.max_images:
        image_files = image_files[: args.max_images]

    print(f"\n找到 {len(image_files)} 张图像")

    if len(image_files) == 0:
        print("❌ 未找到任何图像文件")
        return

    # 显示预测配置
    print(f"\n预测配置:")
    print(f"  置信度阈值: {args.conf}")
    print(f"  IoU阈值: {args.iou}")
    print(f"  图像大小: {args.imgsz}")
    print(f"  设备: {args.device}")
    print(f"  保存图像: {args.save_images}")
    print(f"  保存JSON: {args.save_json}")
    print(f"  保存TXT: {args.save_txt}")
    print(f"  类别过滤: {args.filter_classes}")

    # 开始预测
    print(f"\n开始预测...")
    print("-" * 60)

    # 统计信息
    stats = {
        "total_images": len(image_files),
        "total_detections": 0,
        "class_counts": defaultdict(int),
        "processing_times": [],
        "start_time": time.time(),
    }

    # 存储所有JSON结果
    all_results = []

    # 处理每张图像
    for image_path in tqdm(image_files, desc="预测进度"):
        start_time = time.time()

        try:
            # 执行预测
            results = model(
                str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            image_name = image_path.name
            stem_name = image_path.stem

            # 统计检测结果
            if results[0].boxes is not None:
                boxes = results[0].boxes

                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())

                    # 如果启用类别过滤，只统计目标类别  
                    if cls not in target_classes:
                        if args.filter_classes:
                            continue
                        else:
                            class_name = f"unknown_{cls}"
                    else:
                        # 转换为自定义类别名称
                        custom_id = coco_to_custom[cls]
                        class_name = class_names[custom_id]

                    stats["total_detections"] += 1
                    stats["class_counts"][class_name] += 1

            # 保存带标注的图像
            if args.save_images:
                image = cv2.imread(str(image_path))
                if image is not None:
                    annotated_image = draw_predictions(
                        image, results, class_names, target_classes, args.filter_classes, coco_to_custom
                    )
                    output_path = output_dirs["images"] / f"{stem_name}_pred.jpg"
                    cv2.imwrite(str(output_path), annotated_image)

            # 保存JSON结果
            if args.save_json:
                json_result = results_to_json(
                    results,
                    image_name,
                    class_names,
                    target_classes,
                    args.filter_classes,
                    coco_to_custom
                )
                all_results.append(json_result)

            # 保存TXT结果
            if args.save_txt:
                txt_lines = results_to_txt(results, target_classes, args.filter_classes, coco_to_custom)
                txt_path = output_dirs["txt"] / f"{stem_name}.txt"
                with open(txt_path, "w") as f:
                    f.write("\n".join(txt_lines))

            # 记录处理时间
            processing_time = time.time() - start_time
            stats["processing_times"].append(processing_time)

        except Exception as e:
            print(f"\n⚠️ 处理 {image_path.name} 时出错: {e}")
            import traceback

            print(f"详细错误信息: {traceback.format_exc()}")
            continue

    # 保存JSON结果文件
    if args.save_json and all_results:
        json_output_path = output_dirs["json"] / "predictions.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 计算统计信息
    total_time = time.time() - stats["start_time"]
    avg_time = (
        sum(stats["processing_times"]) / len(stats["processing_times"])
        if stats["processing_times"]
        else 0
    )
    fps = 1 / avg_time if avg_time > 0 else 0

    # 显示结果统计
    print("\n" + "=" * 60)
    print("✅ 预测完成！")
    print("=" * 60)

    print(f"\n📊 统计信息:")
    print(f"  处理图像数量: {stats['total_images']}")
    print(f"  总检测数量: {stats['total_detections']}")
    print(
        f"  平均每张图像: {stats['total_detections']/stats['total_images']:.2f} 个目标"
    )
    print(f"  总处理时间: {total_time:.2f} 秒")
    print(f"  平均处理时间: {avg_time:.3f} 秒/张")
    print(f"  处理速度: {fps:.2f} FPS")

    print(f"\n📋 各类别检测数量:")
    for class_name, count in stats["class_counts"].items():
        percentage = (
            (count / stats["total_detections"]) * 100
            if stats["total_detections"] > 0
            else 0
        )
        print(f"  {class_name}: {count} ({percentage:.1f}%)")

    print(f"\n📁 输出文件:")
    print(f"  输出目录: {args.output_dir}")
    if args.save_images:
        print(f"  标注图像: {output_dirs['images']} ({len(image_files)} 张)")
    if args.save_json:
        print(f"  JSON结果: {output_dirs['json']}/predictions.json")
    if args.save_txt:
        print(f"  TXT标注: {output_dirs['txt']} ({len(image_files)} 个文件)")

    print(f"\n🎯 平均置信度统计:")
    if all_results:
        all_confidences = []
        for result in all_results:
            for pred in result["predictions"]:
                all_confidences.append(pred["confidence"])

        if all_confidences:
            avg_conf = sum(all_confidences) / len(all_confidences)
            max_conf = max(all_confidences)
            min_conf = min(all_confidences)
            print(f"  平均置信度: {avg_conf:.3f}")
            print(f"  最高置信度: {max_conf:.3f}")
            print(f"  最低置信度: {min_conf:.3f}")


if __name__ == "__main__":
    main()
