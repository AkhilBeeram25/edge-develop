# YOLO UPDATE

`YOLO_UPDATE/` is the standalone training directory for the modified YOLO-family architecture. It is intentionally structured like a practical YOLO repo: configs, model package, training engine, scripts, tests, and documentation live under one directory.

This is not a random model. It keeps the YOLO design contract:

- staged convolutional backbone with C2f-style blocks and SPPF context,
- multi-scale feature pyramid/PAN fusion,
- dense single-pass prediction,
- anchor-free box/objectness/class heads,
- distribution-style box regression,
- model scale variants: `micro_s`, `micro_b`, `micro_l`.

The modifications are targeted at micro-object detection:

- P1 detail branch and P2-primary detection,
- anti-aliased early downsampling and space-to-depth first reduction,
- MicroFusion for injecting semantic context into high-resolution features,
- tiny-box weighting, NWD loss, and subpixel center offsets,
- optional P1 detector accuracy mode,
- mask prototypes and region embeddings for later multi-task/open-vocabulary work.

## Install

From the repository root:

```bash
.venv/bin/python -m pip install --no-build-isolation -e YOLO_UPDATE
```

Or from this directory:

```bash
python3 -m pip install -e ".[dev]"
```

## Smoke Train

Run a CPU-safe synthetic training check:

```bash
.venv/bin/python YOLO_UPDATE/scripts/smoke_train.py --variant micro_s --image-size 64 --steps 2
```

## Architecture Validation

```bash
.venv/bin/python YOLO_UPDATE/scripts/validate_architecture.py
```

## Train On YOLO Labels

Prepare a dataset like:

```text
dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

Label rows use normalized YOLO format:

```text
class_id x_center y_center width height
```

Then edit `configs/data/dataset.example.yaml` and run:

```bash
.venv/bin/python YOLO_UPDATE/scripts/train.py \
  --data YOLO_UPDATE/configs/data/dataset.example.yaml \
  --model YOLO_UPDATE/configs/models/yolo_update_micro_s.yaml \
  --train YOLO_UPDATE/configs/train/default.yaml
```

Training writes `last.pt` and `best.pt` under `save_dir`, stores EMA weights in each checkpoint, and reports validation loss plus decoded detection metrics such as `val/det/map`, precision, recall, and 2-to-5-pixel recall.

Resume from a checkpoint with:

```bash
.venv/bin/python YOLO_UPDATE/scripts/train.py \
  --data YOLO_UPDATE/configs/data/dataset.example.yaml \
  --model YOLO_UPDATE/configs/models/yolo_update_micro_s.yaml \
  --train YOLO_UPDATE/configs/train/default.yaml \
  --resume runs/train/yolo_update/last.pt
```

Validate a checkpoint with:

```bash
.venv/bin/python YOLO_UPDATE/scripts/validate.py \
  --data YOLO_UPDATE/configs/data/dataset.example.yaml \
  --model YOLO_UPDATE/configs/models/yolo_update_micro_s.yaml \
  --train YOLO_UPDATE/configs/train/default.yaml \
  --checkpoint runs/train/yolo_update/best.pt
```

## Validate Architecture

See `docs/YOLO_ARCHITECTURE_VALIDATION.md` for the YOLO-lineage audit and the exact modifications.
See `docs/CHANGES_BENEFITS_AND_MICRO_OBJECT_MATH.md` for the detailed change log, benefits, major YOLO differences, and equations used for 2x2-pixel object handling.
