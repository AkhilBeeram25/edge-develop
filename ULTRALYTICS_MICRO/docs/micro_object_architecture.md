# YOLO Micro-Object Architecture

This vendored Ultralytics tree adds micro-object detection models for objects around 2x2 pixels while keeping the normal
Ultralytics training, validation, prediction, and export entry points.

## Added Model Configs

- `ultralytics/cfg/models/26/yolo26-micro.yaml`: latest YOLO26-style model with end-to-end detection and P1-P5 heads.
- `ultralytics/cfg/models/v8/yolov8-micro.yaml`: YOLOv8-compatible model with C2f blocks, DFL regression, and P1-P5 heads.

Use them like any other Ultralytics model:

```bash
PYTHONPATH=ULTRALYTICS_MICRO .venv/bin/python -m ultralytics train \
  model=ULTRALYTICS_MICRO/ultralytics/cfg/models/26/yolo26-micro.yaml \
  data=/path/to/dataset.yaml \
  imgsz=1024 \
  batch=8 \
  device=0
```

## Architectural Changes

### Native and P1 Detail Preservation

Standard YOLO models usually begin with stride-2 convolution and detect at P3/8 or, in small-object variants, P2/4.
A 2x2-pixel object is already compressed to roughly one cell at P1/2 and becomes sub-cell evidence by P2/4. The micro
models therefore:

- keep a stride-1 native-resolution stem,
- add a P1/2 detection head,
- keep P2/4, P3/8, P4/16, and P5/32 heads for normal-size objects.

This improves supervision coverage for tiny targets without removing standard multi-scale detection.

### SPDConv Downsampling

`SPDConv` replaces stride-2 convolution in pyramid transitions. It uses `nn.PixelUnshuffle` to move each 2x2 spatial
phase into channels, then applies a stride-1 convolution. This preserves all input samples before learning a reduction.
The design follows the SPD-Conv finding that strided convolution and pooling can lose fine-grained information in
low-resolution or small-object settings.

### Pooling-Free Context

`MicroSPPF` replaces max-pool SPPF with parallel depthwise dilated convolutions. It expands receptive field without
pooling away small activation peaks. This keeps global context useful for classification while reducing destructive
aggregation of micro-object evidence.

### MicroC2f Refinement

`MicroC2f` keeps the Ultralytics C2f-style gradient path but replaces regular bottlenecks with `MicroDilatedBlock`.
Each block uses:

- a channel-reduced branch,
- depthwise dilated 3x3 convolutions with dilation 1, 2, and 3,
- CBAM attention for lightweight channel and spatial recalibration,
- residual addition for stable gradients.

The intent is to expose a larger local context around a 2x2 object without further downsampling.

### Weighted Multi-Resolution Fusion

`MicroFPNFusion` is a BiFPN-style fusion block. Each input is projected to the target channel count, resized to the
highest requested resolution, and combined with positive normalized learnable weights. This replaces raw concatenation
in the micro head where high-resolution maps must not be dominated by coarse semantic maps.

### Tiny-Aware Detect Head

`MicroDetect` subclasses Ultralytics `Detect`, so it remains compatible with the existing model, loss, inference, and
export code. It changes two defaults:

- `max_det = 1000` to avoid discarding dense tiny-object candidates too early,
- less-suppressed class bias on stride <= 4 heads, improving early positive gradients for P1/P2 candidates.

`MicroDetect` also enables `TinyObjectTaskAlignedAssigner` inside the standard detection loss. The assigner keeps normal
task-aligned assignment for regular boxes, but for boxes up to 4 pixels it combines IoU with center similarity so exact
2x2 targets still receive positive candidates when random initial boxes have zero IoU.

### 2px-Safe Transform Filtering

Ultralytics' perspective transform candidate filter previously required augmented boxes to be strictly larger than 2
pixels in both width and height. This tree relaxes that default to keep boxes at least 1 pixel wide/high, so exact 2x2
objects are not silently removed before loss calculation.

## Training Pipeline Notes

Use the normal Ultralytics dataset YAML format. For tiny objects:

- train at the highest native resolution that fits memory,
- avoid augmentations that shrink or blur 2x2 objects until baseline convergence is proven,
- use tiling or slicing for very high-resolution imagery,
- verify labels remain visible after resize and mosaic transforms,
- monitor size-sliced metrics, especially 2-to-5-pixel recall and false positives per megapixel.

Synthetic smoke training is available:

```bash
PYTHONPATH=ULTRALYTICS_MICRO .venv/bin/python ULTRALYTICS_MICRO/examples/micro_object_train_2px.py \
  --model ULTRALYTICS_MICRO/ultralytics/cfg/models/26/yolo26-micro.yaml \
  --image-size 64 \
  --object-size 2 \
  --epochs 1 \
  --work-dir /tmp/ultralytics_micro_2px
```

CPU smoke validation completed with both `yolo26-micro.yaml` and `yolov8-micro.yaml` on a 4-image train / 2-image val
synthetic 2x2 dataset at 64x64. Both runs kept 2 instances per batch, produced nonzero box/class supervision, saved
`last.pt` and `best.pt`, and reported 0 mAP after one epoch. The 0 mAP is not a convergence result; it only confirms the
pipeline runs and that exact 2x2 boxes reach the loss.

## Limits

No architecture can recover information that was never sampled. If an object is physically below one sensor pixel or is
destroyed by compression, demosaicing, blur, resizing, or augmentation, model changes cannot reconstruct it reliably.
This implementation is designed to preserve and supervise the smallest rasterized evidence that actually exists in the
training image.

## Research Anchors

- Feature Pyramid Networks: https://arxiv.org/abs/1612.03144
- EfficientDet / BiFPN: https://arxiv.org/abs/1911.09070
- SPD-Conv for small objects and low-resolution images: https://arxiv.org/abs/2208.03641
- Normalized Wasserstein Distance for tiny object detection: https://arxiv.org/abs/2110.13389
- CBAM attention: https://arxiv.org/abs/1807.06521
- SAHI slicing/fine-tuning: https://arxiv.org/abs/2202.06934
- Ultralytics model YAML customization: https://docs.ultralytics.com/guides/model-yaml-config/
