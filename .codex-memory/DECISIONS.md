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
