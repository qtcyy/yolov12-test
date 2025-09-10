# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv12 is an attention-centric real-time object detection framework that leverages attention mechanisms while maintaining competitive speed. This repository is a fork/implementation of YOLOv12 based on the Ultralytics framework.

## Common Development Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# For flash attention support (required for YOLOv12)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_python.py
pytest tests/test_engine.py
pytest tests/test_exports.py

# Run tests with coverage
pytest --cov=ultralytics

# Run tests with specific markers
pytest -m "not slow"
```

### Model Usage
```bash
# CLI commands
yolo predict model=yolov12n.pt source=path/to/image.jpg
yolo train model=yolov12n.yaml data=coco.yaml epochs=600 batch=256
yolo val model=yolov12n.pt data=coco.yaml
yolo export model=yolov12n.pt format=onnx

# Run demo application
python app.py  # Launches Gradio interface at http://127.0.0.1:7860
```

### Code Quality Tools
```bash
# Format code with yapf (column limit: 120)
yapf -i -r ultralytics/

# Lint with ruff (line length: 120)
ruff check ultralytics/

# Check for typos
codespell
```

## Architecture Overview

### Core Components

1. **Model Configuration**: `ultralytics/cfg/models/v12/yolov12.yaml`
   - Defines YOLOv12 architecture with attention-centric modules (A2C2f)
   - Supports multiple scales: n, s, m, l, x
   - Two versions: turbo (default, faster) and v1.0

2. **Backbone**: Uses A2C2f (Attention-Centric) modules
   - Replaces traditional CNN blocks with attention mechanisms
   - Maintains competitive inference speed

3. **Training Pipeline**: Based on Ultralytics framework
   - Entry point: `ultralytics.cfg:entrypoint` 
   - Main training logic in `ultralytics/engine/`
   - Model definitions in `ultralytics/nn/`

4. **Key Modules**:
   - `ultralytics/nn/tasks.py`: Core model building logic
   - `ultralytics/models/yolo/`: YOLO-specific implementations
   - `ultralytics/data/`: Data loading and augmentation
   - `ultralytics/utils/`: Utility functions

### Model Variants
- **Detection**: yolov12{n/s/m/l/x}.pt
- **Segmentation**: yolov12{n/s/m/l/x}-seg.pt (in Seg branch)
- **Classification**: yolov12{n/s/m/l/x}-cls.pt (in Cls branch)

### Important Training Parameters
- `scale`: 0.5 for n-variant, 0.9 for s/m/l/x
- `mosaic`: 1.0 for all variants
- `mixup`: 0.0 for n, increases for larger models (up to 0.2 for x)
- `copy_paste`: 0.1 for n, increases for larger models (up to 0.6 for x)

## Key Dependencies
- PyTorch 2.2.2 with TorchVision 0.17.2
- Flash Attention (critical for YOLOv12 performance)
- timm 1.0.14 (for vision transformers)
- albumentations 2.0.4 (for augmentations)
- Gradio 4.44.1 (for demo interface)