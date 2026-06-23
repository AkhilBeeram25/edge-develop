# Decisions

## Durable Checkpoints

- Use Git commits as durable checkpoints so work can be resumed after unexpected shutdown.

## Persistent Memory

- Use `.codex-memory/` as project memory for Codex state, tasks, decisions, and last-session recovery notes.

## Secrets

- Do not store secrets, tokens, private keys, passwords, or credentials in memory files, logs, or commits.

## YOLO Roadmap Documentation

- Store the proprietary YOLO-codenamed vision model roadmap as `docs/YOLO_TECHNICAL_ROADMAP.md`.
- Treat native-resolution tiling, P1/P2 feature preservation, anchor-free assignment, class-agnostic objectness, and mixed-precision edge deployment as the core architecture recommendations for 2 to 5 pixel object detection.
- Treat open-vocabulary embeddings and few-shot prototype adaptation as extensions built after tiny-object detection is stable, not as the first implementation step.

## Phase 1 Implementation

- Use a `src/yolo_micro/` Python package with PyTorch required only for model, loss, decoder, and deployment paths.
- Keep native tiling, augmentation guards, few-shot sampling, metrics, assignment level selection, and weighted box fusion dependency-light so they can run on edge and data-QA machines without PyTorch.
- Keep assignment separate from `TinyDetectionLoss`; the trainer should pass matched positive tensors after size-aware dynamic assignment.
- Export should wrap the model into a flat tuple of tensors for ONNX instead of relying on nested dictionaries.

## Runtime Architecture Adjustment

- Default YOLO-Micro detection levels are P2-P5. P1 remains a detail/refinement path by default.
- A full P1 detection head is supported only via `include_p1_head=True` / `--p1-detector` accuracy mode because runtime validation showed the high-resolution head is costly on CPU and the roadmap recommends P2 as the primary micro-object detection level.
- Include NumPy in the `torch` extra because PyTorch emits runtime warnings and some tensor interop paths are degraded without it.

## YOLO UPDATE Standalone Directory

- Treat `YOLO_UPDATE/` as the standalone trainable directory for the modified YOLO-family model, similar in spirit to a YOLO project directory with configs, package code, scripts, docs, and tests in one place.
- Do not describe YOLO UPDATE as a random architecture. The validation verdict is that it preserves YOLO-family backbone/neck/head structure and adds targeted micro-object modifications.
- Use `YOLO_UPDATE/scripts/validate_architecture.py` as the reproducible architecture-lineage check.
- Use `YOLO_UPDATE/scripts/smoke_train.py` as the minimal trainability check until a real dataset is wired in.

## YOLO UPDATE Trainer Conventions

- Save both `last.pt` and `best.pt` during training.
- Keep EMA weights in checkpoint payloads as `ema_model` and use EMA for validation when enabled.
- Treat decoded validation metrics (`det/map`, precision, recall, and 2-to-5-pixel recall) as the first trainer-level detection-quality signal beyond loss.
- Keep augmentation conservative until real data is available; horizontal flip is enabled through `horizontal_flip_prob` and updates YOLO labels in pixel space.

## Ultralytics Micro-Object Fork

- Keep the cloned Ultralytics code vendored under `ULTRALYTICS_MICRO/` rather than as a nested Git repository or submodule, so checkpoints contain the modified source.
- Preserve Ultralytics training, validation, inference, and export entry points; implement micro-object support through standard modules, YAML configs, parser integration, and loss/assigner hooks.
- Use P1-P5 detection for the micro configs. P1/2 is required so exact 2x2 raster evidence has a detection level before it collapses below one cell.
- Replace stride-2 pyramid transitions in micro configs with `SPDConv` to preserve all 2x2 spatial phases before channel mixing.
- Use `TinyObjectTaskAlignedAssigner` only for `MicroDetect` so upstream model behavior remains unchanged.
- Keep exact 2x2 boxes through transform filtering by allowing candidate boxes with width/height at least 1 pixel.
- In Colab, install `/content/ak/ULTRALYTICS_MICRO` in editable mode and verify `ultralytics.__file__` points inside the cloned repository before training.
