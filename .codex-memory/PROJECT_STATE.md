# Project State

## Workspace

- Path: `/home/open/ak`
- Purpose: persistent Codex memory and resumable development on an ARM64 SoC device that may shut down unexpectedly.

## Important Commands

- Check workspace state: `git status`
- View recent checkpoints: `git log --oneline -5`
- Read last session: `cat .codex-memory/LAST_SESSION.md`
- Manual local Git checkpoint, when intentionally needed: `savecodex`
- Start live local memory watcher: `setsid -f /home/open/ak/.venv/bin/python /home/open/ak/skills/micro-yolo-workflow/scripts/memory_watch.py --workspace /home/open/ak --interval 1 >/tmp/ak-memory-watch.log 2>&1 < /dev/null`
- Check live memory status: `cat .codex-memory/live/STATUS.md`
- Create a tracked memory snapshot: `.venv/bin/python skills/micro-yolo-workflow/scripts/snapshot_memory.py --workspace /home/open/ak --note "<status>"`

## Current Setup Status

- Git repository initialized for durable checkpoints.
- `.codex-memory/` created for persistent project memory.
- Legacy checkpoint script remains installed at `/home/open/.local/bin/codex-checkpoint`, but it is not scheduled and should not be used for automatic status saving.
- Cron Git autosave for `/home/open/ak` was removed on 2026-06-25.
- Live local memory watcher added at `skills/micro-yolo-workflow/scripts/memory_watch.py`; it writes `.codex-memory/live/` once per second when running and never commits or pushes to Git.
- Manual shell helper `savecodex` configured in `/home/open/.bashrc` when that file exists.
- First durable checkpoint commit created.
- Codex CLI updated to `0.142.2` on 2026-06-25 using the official `https://chatgpt.com/codex/install.sh` installer.

## Current Project Artifacts

- `docs/YOLO_TECHNICAL_ROADMAP.md` contains the comprehensive technical roadmap for the proprietary YOLO-codenamed unified vision model, including micro-object architecture, multi-task heads, zero/few-shot strategy, edge inference, phased implementation, risks, and acceptance metrics.
- `src/yolo_micro/` contains the initial Phase 1 implementation scaffold:
  - P1/P2-preserving PyTorch model modules for YOLO-Micro backbone, neck, unified heads, mask prototypes, and open-vocabulary prototype scoring.
  - Tiny-object assignment, NWD/tiny-box loss helpers, and multi-task balancing modules.
  - Dependency-light native tiling, augmentation guards, few-shot episode sampling, micro-object metrics, domain slicing, and weighted box fusion.
  - ONNX, TensorRT command, and QAT deployment entry points.
- `configs/yolo_micro_b.yaml` is the baseline Micro-B configuration.
- `tests/` contains standard-library unit tests for dependency-light utilities.
- `scripts/smoke_torch_pipeline.py` runs a PyTorch smoke pipeline covering model forward, decode, tiny detection loss, multi-task loss combination, and backward pass.
- `YOLO_UPDATE/` is the self-contained trainable model directory for the modified YOLO-family architecture. It includes:
  - `yolo_update/` package with backbone, neck, heads, losses, inference, dataset loading, training criterion, and trainer.
  - `configs/models/`, `configs/data/`, and `configs/train/` YAML files.
  - `scripts/train.py`, `scripts/validate.py`, `scripts/smoke_pipeline.py`, `scripts/smoke_train.py`, and `scripts/validate_architecture.py`.
  - `docs/YOLO_ARCHITECTURE_VALIDATION.md` documenting why the model is a YOLO-family modification rather than a random architecture.
  - `docs/CHANGES_BENEFITS_AND_MICRO_OBJECT_MATH.md` documenting what changed, why the changes help, major differences from YOLO, and the equations used for 2x2-pixel object handling.
  - `tests/` for architecture and synthetic training validation.
- `YOLO_UPDATE` trainer now supports production-oriented basics:
  - checkpoint resume through config or `scripts/train.py --resume`,
  - `last.pt` and `best.pt` checkpoint writing,
  - EMA model weights saved as `ema_model`,
  - decoded validation detection metrics including `val/det/map`, precision, recall, and 2-to-5-pixel recall,
  - optional horizontal flip augmentation for YOLO-format training labels.
- `ULTRALYTICS_MICRO/` is a vendored clone of the official Ultralytics repository from `main` commit `974dda2`, modified for 2x2-pixel micro-object detection while preserving the standard Ultralytics training/inference API. It includes:
  - `ultralytics/nn/modules/micro.py` with `SPDConv`, `MicroC2f`, `MicroSPPF`, `MicroFPNFusion`, and `MicroDetect`.
  - parser integration in `ultralytics/nn/tasks.py` and module exports in `ultralytics/nn/modules/__init__.py`.
  - tiny-aware assignment via `TinyObjectTaskAlignedAssigner` in `ultralytics/utils/tal.py`, activated only by `MicroDetect` through `ultralytics/utils/loss.py`.
  - a 2px-safe transform guard in `ultralytics/data/augment.py` so exact 2x2 boxes are not filtered before loss assignment.
  - model configs `ultralytics/cfg/models/26/yolo26-micro.yaml` and `ultralytics/cfg/models/v8/yolov8-micro.yaml`.
  - synthetic 2x2 training demo `examples/micro_object_train_2px.py`.
  - documentation `docs/micro_object_architecture.md`.
  - Colab setup and training quickstart `docs/colab_micro_training.md`.
  - focused tests in `tests/test_micro_architecture.py`.
  - default Ultralytics assets `ultralytics/assets/bus.jpg` and `ultralytics/assets/zidane.jpg` are tracked because CUDA AMP startup checks reference `bus.jpg`.
- `skills/micro-yolo-workflow/` is a portable project Codex skill that accelerates future sessions by recovering from `.codex-memory`, preserving Colab/micro-YOLO commands, creating non-Git memory snapshots, and running a one-second local memory watcher.
- The same skill is installed for auto-discovery at `/home/open/.codex/skills/micro-yolo-workflow` and has been synchronized with the live-memory/no-auto-push workflow.
- `.codex-memory/LATEST_SNAPSHOT.md` and `.codex-memory/snapshots/*.md` provide non-Git recovery snapshots generated by `skills/micro-yolo-workflow/scripts/snapshot_memory.py`.
- `.codex-memory/live/` is ignored by Git and provides high-frequency local heartbeat/status files from `memory_watch.py`.

## Local Python Environment

- `.venv/` was created with `virtualenv` because system `python3 -m venv` is unavailable without the missing Debian `python3.11-venv` package.
- Installed local validation dependencies in `.venv`: `torch==2.12.1`, `pytest==9.1.1`, `numpy==2.4.6`, Pillow, PyYAML, and Ultralytics runtime dependencies needed for local validation (`opencv-python-headless`, `matplotlib`, `requests`, `scipy`, `psutil`, `polars`, `ultralytics-thop`, and `torchvision`).
- PyPI resolved a GPU-capable ARM64 PyTorch build (`torch==2.12.1+cu130`) with CUDA runtime packages; CUDA is not available on this host (`torch.cuda.is_available() == False`).
- `.venv/` is ignored and is not committed.

## Validation Status

- `.venv/bin/python -m compileall -q src tests scripts` passes.
- `.venv/bin/python -m pytest -q` passes: 11 tests.
- `.venv/bin/python scripts/smoke_torch_pipeline.py --variant micro_s --image-size 64 --num-classes 3` passes.
- `.venv/bin/python scripts/smoke_torch_pipeline.py --variant micro_b --image-size 128 --num-classes 5` passes.
- `.venv/bin/python scripts/smoke_torch_pipeline.py --variant micro_s --image-size 64 --num-classes 3 --p1-detector` passes.
- `.venv/bin/python YOLO_UPDATE/scripts/validate_architecture.py` passes and reports that YOLO UPDATE is a YOLO-family architecture with targeted micro-object modifications.
- `.venv/bin/python -m pytest -q YOLO_UPDATE/tests` passes: 4 tests.
- Documentation sanity check for non-ASCII characters in the new YOLO UPDATE docs passes.
- `.venv/bin/python YOLO_UPDATE/scripts/smoke_pipeline.py --variant micro_s --image-size 64 --num-classes 3` passes.
- `.venv/bin/python YOLO_UPDATE/scripts/smoke_train.py --variant micro_s --image-size 64 --steps 2 --save-dir /tmp/yolo_update_resume_smoke` passes and writes checkpoints under `/tmp`.
- `PYTHONPATH=ULTRALYTICS_MICRO .venv/bin/python -m pytest -q ULTRALYTICS_MICRO/tests/test_micro_architecture.py` passes: 5 tests.
- `PYTHONPATH=ULTRALYTICS_MICRO .venv/bin/python ULTRALYTICS_MICRO/examples/micro_object_train_2px.py --model ULTRALYTICS_MICRO/ultralytics/cfg/models/26/yolo26-micro.yaml --image-size 64 --object-size 2 --train-samples 4 --val-samples 2 --epochs 1 --batch 2 --work-dir /tmp/ultralytics_micro_yolo26_2px_augfix` passes, keeps 2 instances per batch, produces nonzero box/class loss, saves `last.pt` and `best.pt`, and reports 0 mAP after one epoch.
- `PYTHONPATH=ULTRALYTICS_MICRO .venv/bin/python ULTRALYTICS_MICRO/examples/micro_object_train_2px.py --model ULTRALYTICS_MICRO/ultralytics/cfg/models/v8/yolov8-micro.yaml --image-size 64 --object-size 2 --train-samples 4 --val-samples 2 --epochs 1 --batch 2 --work-dir /tmp/ultralytics_micro_v8_2px_augfix` passes, keeps 2 instances per batch, produces nonzero box/class/DFL loss, saves `last.pt` and `best.pt`, and reports 0 mAP after one epoch.
- ONNX export, TensorRT build, QAT conversion, and real native-tile hardware benchmarking have not been run yet.
- A Colab-reported AMP startup crash caused by missing `ultralytics/assets/bus.jpg` was fixed by tracking the default assets and making the AMP check skip rather than crash on `FileNotFoundError`.
- Colab re-test after the fix reached active training with the modified micro architecture:
  - code path: `/content/ak/ULTRALYTICS_MICRO`,
  - model config: `/content/ak/ULTRALYTICS_MICRO/ultralytics/cfg/models/v8/yolov8-micro.yaml`,
  - dataset: `/content/MPI-crack-1/data.yaml`,
  - classes: `nc=3`,
  - GPU: Tesla T4,
  - command settings observed: `epochs=150`, `imgsz=960`, `batch=4`, `pretrained=yolov8n.pt`,
  - startup evidence: custom modules `MicroC2f`, `SPDConv`, `MicroFPNFusion`, `MicroSPPF`, and `MicroDetect` were printed in the model summary,
  - pretraining evidence: `Transferred 28/821 items from pretrained weights`, so `yolov8n.pt` is only a partial warm-start and not the architecture,
  - AMP check passed after downloading `yolo26n.pt`,
  - training started and reached epoch `1/150`, batch progress around `109/809`, with GPU memory around `9.45G`.
- `micro-yolo-workflow` skill validation passes for both the repo copy and the auto-discovered installed copy using `.venv/bin/python /home/open/.codex/skills/.system/skill-creator/scripts/quick_validate.py`.
