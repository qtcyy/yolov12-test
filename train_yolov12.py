#!/usr/bin/env python3
"""
YOLOv12 DOTA数据集训练脚本
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练YOLOv12模型")

    # 模型相关参数
    parser.add_argument(
        "--model",
        type=str,
        default="yolov12n.pt",
        help="预训练模型路径或配置文件 (yolov12n/s/m/l/x.pt)",
    )

    # 数据集参数
    parser.add_argument(
        "--data", type=str, default="coco_dataset.yaml", help="数据集配置文件路径"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="训练图像大小")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="训练设备 (0,1,2,3 为CUDA GPU, mps 为Apple GPU, cpu 为CPU)",
    )

    # 优化器参数
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument(
        "--lrf", type=float, default=0.01, help="最终学习率 (lr0 * lrf)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.937, help="SGD动量/Adam beta1"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="权重衰减")

    # 数据增强参数
    parser.add_argument(
        "--scale", type=float, default=0.5, help="图像缩放范围 (+/- gain)"
    )
    parser.add_argument("--mosaic", type=float, default=1.0, help="马赛克增强概率")
    parser.add_argument("--mixup", type=float, default=0.0, help="混合增强概率")
    parser.add_argument(
        "--copy_paste", type=float, default=0.1, help="复制粘贴增强概率"
    )

    # 其他参数
    parser.add_argument(
        "--project", type=str, default="runs/train", help="保存项目路径"
    )
    parser.add_argument("--name", type=str, default="coco_yolov12", help="实验名称")
    parser.add_argument("--exist_ok", action="store_true", help="是否覆盖已存在的项目")
    parser.add_argument(
        "--pretrained", type=bool, default=True, help="是否使用预训练权重"
    )
    parser.add_argument(
        "--resume", type=bool, default=False, help="是否从最后一个检查点恢复训练"
    )
    parser.add_argument(
        "--save_period", type=int, default=10, help="每N个epoch保存检查点"
    )
    parser.add_argument("--cache", type=bool, default=False, help="是否缓存图像到内存")
    parser.add_argument("--workers", type=int, default=8, help="数据加载的工作线程数")
    parser.add_argument(
        "--amp", type=bool, default=True, help="是否使用自动混合精度训练"
    )
    parser.add_argument("--patience", type=int, default=50, help="早停的耐心值")

    return parser.parse_args()


def check_and_fix_dataset(data_path):
    """检查并修复数据集配置"""

    print("\n检查数据集...")

    # 加载数据集配置
    try:
        with open(data_path, "r") as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 配置文件不存在 {data_path}")
        print("   请先运行: python coco_to_yolo.py 生成配置文件")
        return False

    # 验证数据集路径
    dataset_path = Path(data_config["path"])
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集路径不存在 {dataset_path}")
        print("   请先运行: python coco_to_yolo.py 转换数据集")
        return False

    print(f"\n验证数据集路径: {dataset_path}")

    all_good = True
    for split in ["train", "val"]:
        if split in data_config:
            img_dir = dataset_path / data_config[split]
            label_dir = dataset_path / data_config[split].replace("images", "labels")

            img_exists = img_dir.exists()
            label_exists = label_dir.exists()

            if img_exists:
                img_count = len(
                    list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
                )
                print(f"  {split}集: {img_count} 张图像", end="")

                if label_exists:
                    label_count = len(list(label_dir.glob("*.txt")))
                    print(f", {label_count} 个标注")

                    if img_count != label_count:
                        print(f"    ⚠️ 图像和标注数量不匹配")
                        all_good = False
                else:
                    print(f" - ⚠️ 未找到标注目录")
                    all_good = False
            else:
                print(f"  {split}集: ❌ 未找到图像目录")
                all_good = False

    return all_good


def train():
    """训练主函数"""

    args = parse_args()

    print("=" * 60)
    print("YOLOv12 COCO-4类数据集训练")
    print("=" * 60)

    # 显示设备信息
    if args.device == "mps":
        # Apple Silicon GPU (Metal Performance Shaders)
        if torch.backends.mps.is_available():
            print(f"\n使用Apple Silicon GPU (MPS)训练")
            import platform

            print(f"  芯片: Apple {platform.processor()}")
            print(f"  架构: {platform.machine()}")
        else:
            print("\n⚠️ MPS不可用，将使用CPU训练")
            args.device = "cpu"
    elif args.device != "cpu":
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"\n使用NVIDIA GPU训练:")
            for i in range(device_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("\n⚠️ CUDA不可用，将使用CPU训练")
            args.device = "cpu"
    else:
        print("\n使用CPU训练")

    # 检查并修复数据集配置
    if not check_and_fix_dataset(args.data):
        print("\n❌ 数据集检查失败，请先运行 python coco_to_yolo.py")
        return

    # 选择模型规模的训练参数
    model_params = {
        "yolov12n.pt": {"scale": 0.5, "mixup": 0.0, "copy_paste": 0.1},
        "yolov12s.pt": {"scale": 0.9, "mixup": 0.05, "copy_paste": 0.15},
        "yolov12m.pt": {"scale": 0.9, "mixup": 0.15, "copy_paste": 0.4},
        "yolov12l.pt": {"scale": 0.9, "mixup": 0.15, "copy_paste": 0.5},
        "yolov12x.pt": {"scale": 0.9, "mixup": 0.2, "copy_paste": 0.6},
    }

    # 获取模型特定参数
    if args.model in model_params:
        params = model_params[args.model]
        print(f"\n使用{args.model}推荐参数:")
        print(f"  scale: {params['scale']}")
        print(f"  mixup: {params['mixup']}")
        print(f"  copy_paste: {params['copy_paste']}")

        # 更新参数（如果用户没有明确指定）
        args.scale = params["scale"]
        args.mixup = params["mixup"]
        args.copy_paste = params["copy_paste"]

    print(f"\n训练配置:")
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.data}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch}")
    print(f"  图像大小: {args.imgsz}")
    print(f"  学习率: {args.lr0}")
    print(f"  设备: {args.device}")
    print(f"  混合精度: {args.amp}")

    # 初始化模型
    print("\n加载模型...")
    try:
        # 尝试加载预训练模型
        model = YOLO(args.model)
        print(f"✓ 成功加载预训练模型: {args.model}")
    except:
        # 如果预训练模型不存在，使用配置文件
        model_config = args.model.replace(".pt", ".yaml")
        if Path(model_config).exists():
            model = YOLO(model_config)
            print(f"✓ 从配置文件创建模型: {model_config}")
        else:
            # 使用默认的n配置
            model = YOLO("yolov12n.yaml")
            print("✓ 使用默认yolov12n配置")

    # 开始训练
    print("\n开始训练...")
    print("-" * 60)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scale=args.scale,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=args.pretrained,
        resume=args.resume,
        save_period=args.save_period,
        cache=args.cache,
        workers=args.workers,
        amp=args.amp,
        patience=args.patience,
        cos_lr=True,  # 使用余弦学习率调度
        close_mosaic=10,  # 最后10个epoch关闭马赛克增强
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print("=" * 60)

    # 评估模型
    print("\n评估模型性能...")
    metrics = model.val()

    print(f"\n验证集指标:")
    print(f"  mAP@0.5: {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")

    # 保存最终模型
    save_path = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n最佳模型保存在: {save_path}")

    # 测试推理
    print("\n测试推理...")
    
    # 从配置文件获取验证集路径
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = Path(data_config['path'])
    test_img_dir = dataset_path / data_config['val']
    test_images = (list(test_img_dir.glob("*.png")) + list(test_img_dir.glob("*.jpg")))[:10]

    if test_images:
        print(f"  在{len(test_images)}张图像上测试...")
        for img in test_images:
            results = model(img)
            print(f"  ✓ {img.name}: 检测到 {len(results[0].boxes)} 个目标")

    print("\n下一步:")
    print(f"1. 查看训练结果: {args.project}/{args.name}/")
    print(f"2. 使用模型进行推理: yolo predict model={save_path} source=your_image.jpg")
    print(f"3. 导出模型: yolo export model={save_path} format=onnx")


if __name__ == "__main__":
    train()
