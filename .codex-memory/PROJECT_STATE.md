# Project State

## Workspace

- Path: `/home/open/ak`
- Purpose: persistent Codex memory and resumable development on an ARM64 SoC device that may shut down unexpectedly.

## Important Commands

- Check workspace state: `git status`
- View recent checkpoints: `git log --oneline -5`
- Read last session: `cat .codex-memory/LAST_SESSION.md`
- Manual checkpoint: `savecodex`
- Autosave checkpoint script: `/home/open/.local/bin/codex-checkpoint /home/open/ak`

## Current Setup Status

- Git repository initialized for durable checkpoints.
- `.codex-memory/` created for persistent project memory.
- Autosave checkpoint script installed at `/home/open/.local/bin/codex-checkpoint`.
- Cron autosave configured to run once per minute for `/home/open/ak`.
- Manual shell helper `savecodex` configured in `/home/open/.bashrc` when that file exists.
- First durable checkpoint commit created.

## Current Project Artifacts

- `docs/YOLO_TECHNICAL_ROADMAP.md` contains the comprehensive technical roadmap for the proprietary YOLO-codenamed unified vision model, including micro-object architecture, multi-task heads, zero/few-shot strategy, edge inference, phased implementation, risks, and acceptance metrics.
- `src/yolo_micro/` contains the initial Phase 1 implementation scaffold:
  - P1/P2-preserving PyTorch model modules for YOLO-Micro backbone, neck, unified heads, mask prototypes, and open-vocabulary prototype scoring.
  - Tiny-object assignment, NWD/tiny-box loss helpers, and multi-task balancing modules.
  - Dependency-light native tiling, augmentation guards, few-shot episode sampling, micro-object metrics, domain slicing, and weighted box fusion.
  - ONNX, TensorRT command, and QAT deployment entry points.
- `configs/yolo_micro_b.yaml` is the baseline Micro-B configuration.
- `tests/` contains standard-library unit tests for dependency-light utilities.

## Validation Status

- `python3 -m compileall -q src tests` passes.
- `PYTHONPATH=src python3 -m unittest discover -s tests` passes: 8 tests.
- PyTorch is not installed in this workspace, so model forward-pass, training-loss tensor execution, ONNX export, and QAT conversion have not been run locally.
