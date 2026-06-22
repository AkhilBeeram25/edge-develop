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
