# YOLOv12 数据集说明文档

本文档详细说明了 YOLOv12 项目中使用的数据集结构、标注格式和使用方法。

## 📁 数据集结构

```
dataset_yolo/
├── images/                 # 图像文件目录
│   ├── train/             # 训练集图像 (4757张)
│   └── val/               # 验证集图像 (2033张)
├── labels/                 # 标注文件目录
│   ├── train/             # 训练集标注 (4757个.txt文件)
│   └── val/               # 验证集标注 (2033个.txt文件)
├── images_test/           # 测试图像目录 (备用)
├── labels_test/           # 测试标注目录 (备用)
├── val_standard/          # 验证集可视化结果 (由脚本生成)
└── README.md              # 本文档
```

## 🏷️ 类别定义

本数据集包含 **4个目标类别**，每个类别都有对应的数字ID：

| 类别ID | 类别名称 | 英文名称 | 描述 |
|--------|----------|----------|------|
| 0 | 船舶 | ship | 各种类型的船只 |
| 1 | 人 | people | 行人、人群 |
| 2 | 汽车 | car | 各种汽车 |
| 3 | 摩托车 | motor | 摩托车、电动车等两轮车辆 |

## 📝 标注格式说明

### YOLO 格式标注

每个图像都有一个对应的 `.txt` 标注文件，文件名与图像文件名相同（仅扩展名不同）。

#### 标注文件格式

每行代表一个目标对象，格式为：
```
class_id x_center y_center width height
```

#### 参数说明

- **class_id**: 目标类别ID（0-3，对应上表中的类别）
- **x_center**: 边界框中心点x坐标（归一化到0-1）
- **y_center**: 边界框中心点y坐标（归一化到0-1）
- **width**: 边界框宽度（归一化到0-1）
- **height**: 边界框高度（归一化到0-1）

> **注意**: 所有坐标值都是相对于图像尺寸的归一化值，范围在 0.0 到 1.0 之间。

#### 标注示例

```
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.1 0.3
2 0.2 0.8 0.15 0.25
```

**解释**:
- 第1行：船舶，中心点在图像的(50%, 30%)位置，宽度占图像20%，高度占图像40%
- 第2行：人，中心点在图像的(70%, 60%)位置，宽度占图像10%，高度占图像30%
- 第3行：汽车，中心点在图像的(20%, 80%)位置，宽度占图像15%，高度占图像25%

### 坐标转换

如果需要将归一化坐标转换为像素坐标，可以使用以下公式：

```python
# 已知图像尺寸 (img_width, img_height) 和YOLO标注
x_center_px = x_center * img_width
y_center_px = y_center * img_height
width_px = width * img_width
height_px = height * img_height

# 计算边界框的四个角点坐标
x1 = x_center_px - width_px / 2    # 左上角x
y1 = y_center_px - height_px / 2   # 左上角y
x2 = x_center_px + width_px / 2    # 右下角x
y2 = y_center_px + height_px / 2   # 右下角y
```

## 📊 数据集统计

### 基本信息
- **总图像数量**: 6,790张
- **训练集**: 4,757张图像 (70.1%)
- **验证集**: 2,033张图像 (29.9%)
- **图像格式**: JPEG (.jpg)
- **标注格式**: YOLO TXT格式

### 类别分布
基于验证集的统计结果：
- **people (人)**: 约89.0% - 主要类别
- **car (汽车)**: 约5.8%
- **motor (摩托车)**: 约5.2%
- **ship (船舶)**: 少量

## 🛠️ 使用方法

### 1. 训练模型

```bash
# 使用YOLOv12训练
yolo train model=yolov12n.yaml data=coco_dataset.yaml epochs=600 batch=256
```

### 2. 验证模型

```bash
# 验证模型性能
yolo val model=best.pt data=coco_dataset.yaml
```

### 3. 预测

```bash
# 单张图像预测
yolo predict model=best.pt source=path/to/image.jpg

# 批量预测
python predict_val.py --model best.pt --save_images
```

### 4. 可视化标注

```bash
# 可视化验证集的真实标注
python src/visualize_val_annotations.py --show_stats

# 处理部分图像进行测试
python src/visualize_val_annotations.py --max_images 10 --show_stats
```

## 📁 相关配置文件

### coco_dataset.yaml
数据集配置文件，定义了：
- 数据集路径
- 训练/验证集分割
- 类别数量和名称
- 数据增强参数

### predict_val.py
验证集预测脚本，支持：
- 自定义模型预测
- 结果可视化
- 性能评估
- 多种输出格式

### src/visualize_val_annotations.py
标注可视化脚本，用于：
- 显示真实标注框
- 类别标签展示
- 批量处理验证集
- 统计信息输出

## 🎨 可视化效果

使用可视化脚本生成的图像中，不同类别使用不同颜色的边界框：

- 🟦 **船舶 (ship)**: 青色边界框
- 🟣 **人 (people)**: 紫色边界框
- 🟡 **汽车 (car)**: 黄色边界框
- 🟢 **摩托车 (motor)**: 绿色边界框

## 🔧 工具和脚本

### 数据预处理
- 标注格式转换工具
- 数据集划分脚本
- 数据增强配置

### 模型训练
- YOLOv12模型配置
- 训练参数优化
- 损失函数设置

### 评估和可视化
- 预测结果可视化
- 性能指标计算
- 错误分析工具

## 📋 注意事项

1. **文件命名**: 确保图像文件和标注文件名称一致（仅扩展名不同）
2. **坐标格式**: 所有坐标都必须是0-1范围内的归一化值
3. **类别ID**: 类别ID必须在0-3范围内，对应预定义的4个类别
4. **文件编码**: 标注文件使用UTF-8编码
5. **路径设置**: 相对路径基于数据集根目录

## 🚀 快速开始

1. **环境准备**
   ```bash
   pip install ultralytics opencv-python tqdm
   ```

2. **验证数据集**
   ```bash
   python src/visualize_val_annotations.py --max_images 5 --show_stats
   ```

3. **开始训练**
   ```bash
   yolo train model=yolov12n.yaml data=coco_dataset.yaml epochs=100
   ```

4. **评估模型**
   ```bash
   python predict_val.py --model runs/detect/train/weights/best.pt --evaluate
   ```

## 📞 技术支持

如果在使用过程中遇到问题，请检查：
- 数据集路径配置是否正确
- 标注文件格式是否符合YOLO规范
- 图像和标注文件是否一一对应
- 类别ID是否在有效范围内

---

*最后更新时间: 2024年9月*