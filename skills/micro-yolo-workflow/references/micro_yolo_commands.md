# Micro YOLO Commands

## Colab Install

For a public repo:

```bash
!git clone https://github.com/AkhilBeeram25/edge-develop.git /content/ak
%cd /content/ak
!python -m pip install -e /content/ak/ULTRALYTICS_MICRO
```

For a private repo, use a token through `getpass`, not a literal token in notebook text.

Verify package path:

```python
import ultralytics
print(ultralytics.__file__)
```

Expected:

```text
/content/ak/ULTRALYTICS_MICRO/ultralytics/__init__.py
```

## Main Training Command

```bash
!yolo detect train \
  model=/content/ak/ULTRALYTICS_MICRO/ultralytics/cfg/models/v8/yolov8-micro.yaml \
  pretrained=yolov8n.pt \
  data=/content/MPI-crack-1/data.yaml \
  epochs=150 \
  imgsz=960 \
  batch=4 \
  mosaic=0.2 \
  scale=0.25 \
  degrees=0 \
  perspective=0 \
  close_mosaic=20
```

If AMP startup fails, append:

```text
amp=False
```

## Confirm Micro Architecture

The training log must show:

```text
model=/content/ak/ULTRALYTICS_MICRO/ultralytics/cfg/models/v8/yolov8-micro.yaml
ultralytics.nn.modules.micro.MicroC2f
ultralytics.nn.modules.micro.SPDConv
ultralytics.nn.modules.micro.MicroFPNFusion
ultralytics.nn.modules.micro.MicroSPPF
ultralytics.nn.modules.micro.MicroDetect
```

This line means `yolov8n.pt` is only a partial warm-start:

```text
Transferred 28/821 items from pretrained weights
```

## Colab Output To Preserve

At the end of a run, record:

```text
/content/ak/runs/detect/train/args.yaml
/content/ak/runs/detect/train/results.csv
/content/ak/runs/detect/train/weights/best.pt
/content/ak/runs/detect/train/weights/last.pt
```

If there are repeated runs, the directory may be `train2`, `train3`, etc.
