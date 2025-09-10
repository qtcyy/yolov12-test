#!/bin/bash

# YOLOv12 DOTA训练快速启动脚本

echo "============================================================"
echo "YOLOv12 DOTA数据集训练快速启动"
echo "============================================================"

# 检查环境
echo "检查环境..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装"
    exit 1
fi

# 检查必要的库
python -c "import torch, ultralytics" 2>/dev/null || {
    echo "❌ 缺少必要的依赖库，请先运行："
    echo "   pip install -r requirements.txt"
    exit 1
}

# 检查GPU
echo "检查GPU可用性..."
python -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "发现 $gpu_count 个GPU"
else
    echo "⚠️ 未检测到GPU，将使用CPU训练（速度较慢）"
fi

# 检查数据集
echo ""
echo "检查数据集..."
if [ ! -d "dataset/images/train" ] || [ ! -d "dataset/labels/train" ]; then
    echo "⚠️ 数据集未准备完成，正在准备数据集..."
    python prepare_dota_dataset.py
else
    echo "✓ 数据集已准备完成"
fi

# 选择模型大小
echo ""
echo "选择模型大小:"
echo "1) YOLOv12n (轻量级，速度快)"
echo "2) YOLOv12s (平衡型)"
echo "3) YOLOv12m (中等精度)"
echo "4) YOLOv12l (高精度)"
echo "5) YOLOv12x (最高精度，速度慢)"

read -p "请选择 [1-5] (默认: 1): " choice
choice=${choice:-1}

case $choice in
    1) model="yolov12n.pt"; batch=16 ;;
    2) model="yolov12s.pt"; batch=12 ;;
    3) model="yolov12m.pt"; batch=8 ;;
    4) model="yolov12l.pt"; batch=6 ;;
    5) model="yolov12x.pt"; batch=4 ;;
    *) model="yolov12n.pt"; batch=16 ;;
esac

echo "选择的模型: $model, 批次大小: $batch"

# 选择训练轮数
read -p "训练轮数 (默认: 100): " epochs
epochs=${epochs:-100}

# 选择设备
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    read -p "使用的GPU编号 (默认: 0): " device
    device=${device:-0}
else
    device="cpu"
fi

echo ""
echo "开始训练..."
echo "配置信息:"
echo "  模型: $model"
echo "  训练轮数: $epochs" 
echo "  批次大小: $batch"
echo "  设备: $device"
echo ""

# 启动训练
python train_yolov12.py \
    --model $model \
    --data dota_dataset.yaml \
    --epochs $epochs \
    --batch $batch \
    --device $device \
    --project runs/train \
    --name dota_yolov12_$(date +%Y%m%d_%H%M%S) \
    --exist_ok

echo ""
echo "训练完成！查看结果："
echo "  训练日志: runs/train/"
echo "  最佳模型: runs/train/dota_yolov12_*/weights/best.pt"