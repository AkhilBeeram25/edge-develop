# Colab Training Quickstart for Ultralytics Micro

This repository vendors a modified Ultralytics package under `ULTRALYTICS_MICRO/`.
In Colab, install that package in editable mode so `import ultralytics` and the
`yolo` command resolve to the micro-object fork, not the stock PyPI package.

## 1. Start a GPU Runtime

In Colab, select:

```text
Runtime -> Change runtime type -> GPU
```

Then verify CUDA:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

## 2. Clone and Install This Repository

Replace the URL with the Git remote that contains these committed changes.

```bash
!git clone <YOUR_REPOSITORY_URL> /content/ak
%cd /content/ak
!python -m pip install -U pip
!python -m pip install -e /content/ak/ULTRALYTICS_MICRO
```

Verify that Colab imports the modified package:

```python
import ultralytics

print(ultralytics.__file__)
```

Expected path:

```text
/content/ak/ULTRALYTICS_MICRO/ultralytics/__init__.py
```

## 3. Prepare Dataset YAML

Use normal Ultralytics YOLO dataset layout and labels. Example `data.yaml`:

```yaml
path: /content/datasets/my_dataset
train: images/train
val: images/val

names:
  0: object
```

For Google Drive datasets:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Then point `path`, `train`, and `val` at your Drive dataset paths.

## 4. Train the Micro Architecture

Start with the YOLOv8-compatible micro config because it can partially warm-start
from `yolov8n.pt`:

```bash
!yolo detect train \
  model=/content/ak/ULTRALYTICS_MICRO/ultralytics/cfg/models/v8/yolov8-micro.yaml \
  pretrained=yolov8n.pt \
  data=/content/data.yaml \
  epochs=100 \
  imgsz=960 \
  batch=4 \
  mosaic=0.2 \
  scale=0.25 \
  degrees=0 \
  perspective=0 \
  close_mosaic=20
```

For very small GPUs, reduce `batch` first. If training is still out of memory,
reduce `imgsz` to `768` or `640`.

## 5. Alternative Python API

```python
from ultralytics import YOLO

model = YOLO("/content/ak/ULTRALYTICS_MICRO/ultralytics/cfg/models/v8/yolov8-micro.yaml")
results = model.train(
    data="/content/data.yaml",
    pretrained="yolov8n.pt",
    epochs=100,
    imgsz=960,
    batch=4,
    mosaic=0.2,
    scale=0.25,
    degrees=0,
    perspective=0,
    close_mosaic=20,
)
```

## 6. Train the Latest YOLO26-Style Micro Config

Use this when you want the latest vendored Ultralytics architecture style. It is
less directly comparable to `yolov8n.pt`, so treat pretraining transfer as more
limited.

```bash
!yolo detect train \
  model=/content/ak/ULTRALYTICS_MICRO/ultralytics/cfg/models/26/yolo26-micro.yaml \
  data=/content/data.yaml \
  epochs=100 \
  imgsz=960 \
  batch=4 \
  mosaic=0.2 \
  scale=0.25 \
  degrees=0 \
  perspective=0 \
  close_mosaic=20
```

## Notes for 2-5 Pixel Objects

- Use the highest native image size your GPU can handle.
- Avoid aggressive perspective, rotation, and scale augmentation because tiny
  objects can disappear after transforms.
- Compare against a baseline such as `yolo11n.pt` or `yolov8n.pt` with identical
  data splits.
- Review predictions visually. Aggregate mAP can hide whether 2-5 pixel recall
  improved.
