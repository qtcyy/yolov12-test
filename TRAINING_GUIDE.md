# YOLOv12 DOTA数据集训练指南

## ✅ 设置完成状态

### 数据集状态
- **训练集**: 469张图像，469个标注文件 ✅
- **验证集**: 458张图像，458个标注文件 ✅  
- **测试集**: 468张图像（无标注，用于最终评估）
- **类别数量**: 15个DOTA航空目标类别

### 文件结构
```
yolov12/
├── dataset/
│   ├── images/
│   │   ├── train/     # 训练图像
│   │   ├── val/       # 验证图像
│   │   └── test/      # 测试图像
│   └── labels/
│       ├── train/     # 训练标注（YOLO格式）
│       └── val/       # 验证标注（YOLO格式）
├── dota_dataset.yaml  # 数据集配置
├── train_yolov12.py   # 训练脚本
├── start_training.sh  # 快速启动脚本
└── verify_setup.py    # 设置验证脚本
```

## 🚀 开始训练

### 方式1: 交互式启动（推荐新手）
```bash
./start_training.sh
```
- 自动检查环境和数据集
- 交互式选择模型大小和参数
- 适合初学者

### 方式2: Python脚本（推荐进阶用户）
```bash
# Apple Silicon GPU训练（推荐）
python train_yolov12.py --epochs 100 --batch 8 --device mps

# NVIDIA GPU训练
python train_yolov12.py --epochs 100 --batch 16 --device 0

# CPU训练
python train_yolov12.py --epochs 100 --batch 4 --device cpu

# 快速测试（10轮）
python train_yolov12.py --epochs 10 --batch 8 --device mps
```

### 方式3: 直接使用YOLO命令
```bash
# Apple Silicon GPU
yolo train model=yolov12n.pt data=dota_dataset.yaml epochs=100 batch=8 device=mps

# NVIDIA GPU
yolo train model=yolov12n.pt data=dota_dataset.yaml epochs=100 batch=16 device=0

# CPU
yolo train model=yolov12n.pt data=dota_dataset.yaml epochs=100 batch=4 device=cpu
```

## 📊 模型选择建议

| 模型 | 参数量 | 推荐批次大小 | 训练时间 | 精度 |
|------|--------|-------------|----------|------|
| YOLOv12n | 2.5M | 16 | 最快 | 基准 |
| YOLOv12s | 9.1M | 12 | 快 | 中等 |
| YOLOv12m | 19.6M | 8 | 中等 | 较高 |
| YOLOv12l | 26.5M | 6 | 慢 | 高 |
| YOLOv12x | 59.3M | 4 | 最慢 | 最高 |

## ⚙️ 训练参数说明

### 基础参数
- `--model`: 模型大小（yolov12n/s/m/l/x.pt）
- `--epochs`: 训练轮数（建议100-300）
- `--batch`: 批次大小（根据内存调整）
- `--device`: 设备
  - `mps`: Apple Silicon GPU
  - `0,1,2,3`: NVIDIA GPU编号
  - `cpu`: CPU训练

### 优化参数
- `--lr0`: 初始学习率（默认0.01）
- `--weight_decay`: 权重衰减（默认0.0005）
- `--amp`: 混合精度训练（默认True，加速训练）

### 数据增强参数
- `--scale`: 图像缩放范围（默认0.5）
- `--mosaic`: 马赛克增强概率（默认1.0）
- `--mixup`: 图像混合概率（n:0.0, s:0.05, m:0.15, l:0.15, x:0.2）

## 📈 监控训练

### 实时监控
```bash
# 使用TensorBoard
tensorboard --logdir runs/train

# 查看日志
tail -f runs/train/dota_yolov12_*/train.log
```

### 结果文件
- **模型权重**: `runs/train/*/weights/best.pt`
- **训练日志**: `runs/train/*/results.csv`
- **训练图表**: `runs/train/*/results.png`

## 🎯 训练完成后

### 模型评估
```bash
# 在验证集上评估
yolo val model=runs/train/dota_yolov12_*/weights/best.pt data=dota_dataset.yaml

# 在测试集上推理
yolo predict model=runs/train/dota_yolov12_*/weights/best.pt source=dataset/images/test
```

### 模型导出
```bash
# 导出ONNX格式
yolo export model=runs/train/dota_yolov12_*/weights/best.pt format=onnx

# 导出TensorRT格式（需要GPU）
yolo export model=runs/train/dota_yolov12_*/weights/best.pt format=engine half=true
```

## 🐛 常见问题解决

### 1. 内存不足
```bash
# 减小批次大小
--batch 4

# 减小图像尺寸
--imgsz 416
```

### 2. 训练速度慢
```bash
# 启用混合精度
--amp True

# 增加工作线程
--workers 8

# 使用缓存
--cache ram
```

### 3. 过拟合
```bash
# 增强数据增强
--scale 0.9 --mixup 0.15

# 增加正则化
--weight_decay 0.001

# 早停
--patience 50
```

### 4. 欠拟合
```bash
# 增加训练轮数
--epochs 300

# 调整学习率
--lr0 0.02

# 减少数据增强
--mosaic 0.5
```

## 🎯 性能优化建议

### Apple Silicon GPU (MPS) 训练优化 🍎
- **推荐模型**: YOLOv12n/s（轻量级）
- **批次大小**: 8-12（根据内存调整）
- **设备设置**: `--device mps`
- **混合精度**: 支持但可能需要调整
- **注意事项**: 
  - MPS在某些操作上可能比CPU慢，建议先小规模测试
  - 不支持多GPU并行训练
  - 适合中小规模数据集和快速原型验证

#### Apple Silicon 专用命令
```bash
# 使用Apple GPU训练（推荐）
python train_yolov12.py --epochs 100 --batch 8 --device mps

# 快速测试（10轮）
python train_yolov12.py --epochs 10 --batch 8 --device mps

# 使用交互式脚本（自动检测Apple GPU）
./start_training.sh
```

### CPU训练优化
- 使用较小的模型（YOLOv12n/s）
- 减小批次大小（4-8）
- 启用OpenMP多线程

### NVIDIA GPU训练优化
- 使用混合精度训练（--amp True）
- 适当增大批次大小
- 使用多GPU并行（--device 0,1,2,3）

### 数据增强调优
- 航空图像特点：物体较小，方向多变
- 建议保持较高的mosaic概率
- 适当使用旋转和翻转增强

## 📝 预期结果

根据DOTA数据集的特点，预期训练结果：

- **YOLOv12n**: mAP@0.5 ~35-40%
- **YOLOv12s**: mAP@0.5 ~40-45% 
- **YOLOv12m**: mAP@0.5 ~45-50%
- **YOLOv12l**: mAP@0.5 ~50-55%
- **YOLOv12x**: mAP@0.5 ~55%+

训练时间（CPU）：
- 100 epochs: 约6-12小时（取决于硬件）
- 建议先用10 epochs快速测试

---

🎉 **恭喜！您的YOLOv12 DOTA训练环境已完全配置就绪，可以开始训练了！**