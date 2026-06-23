---
name: micro-yolo-workflow
description: Project-specific productivity workflow for the /home/open/ak micro-object YOLO workspace. Use when Codex is asked to resume work, save status, recover context without relying on Git, train or debug the ULTRALYTICS_MICRO fork, handle Colab setup/training, inspect micro-object YOLO architecture changes, or continue the MPI crack dataset experiment.
---

# Micro YOLO Workflow

## Operating Rule

Use `.codex-memory/` as the source of truth for project state. Treat Git as a checkpoint transport, not as the only memory system.

At task start in `/home/open/ak`, read:

```text
AGENTS.md
.codex-memory/PROJECT_STATE.md
.codex-memory/TODO.md
.codex-memory/DECISIONS.md
.codex-memory/LAST_SESSION.md
```

Then use Git only for safety checks required by `AGENTS.md` or for publishing checkpoints. If Git is unavailable, continue from `.codex-memory` and write a snapshot before ending the turn.

## Fast Recovery

When resuming after reset, answer these before editing:

- What is the current objective?
- What files or commands are already validated?
- What exact external run is in progress, especially Colab training?
- What is the next irreversible or expensive action?

For this project, the most important active context is:

- `ULTRALYTICS_MICRO/` is the vendored modified Ultralytics fork.
- Main micro config: `ULTRALYTICS_MICRO/ultralytics/cfg/models/v8/yolov8-micro.yaml`.
- Latest confirmed Colab run used `/content/MPI-crack-1/data.yaml`, `epochs=150`, `imgsz=960`, `batch=4`, Tesla T4, and partial warm-start from `yolov8n.pt`.
- The log confirmed micro modules: `MicroC2f`, `SPDConv`, `MicroFPNFusion`, `MicroSPPF`, and `MicroDetect`.
- `yolov8n.pt` is only a partial weight source, not the architecture.

Read `references/micro_yolo_commands.md` when the user asks for training, Colab, validation, or troubleshooting commands.

## Status Saving

When the user says “save status” or when meaningful work is done:

1. Update `.codex-memory/LAST_SESSION.md` with concrete actions, commands, validation, and next step.
2. Update `.codex-memory/PROJECT_STATE.md`, `.codex-memory/TODO.md`, and `.codex-memory/DECISIONS.md` when facts, pending work, or decisions changed.
3. Run `scripts/snapshot_memory.py --workspace /home/open/ak --note "<short status>"` from this skill directory, or from the repo copy at `skills/micro-yolo-workflow/scripts/snapshot_memory.py`.
4. Do not store secrets, tokens, private keys, passwords, or credential-bearing URLs.
5. If Git is available and project rules require it, make a small checkpoint commit after the memory update. The snapshot is still the non-Git recovery path.

## Working On Code

Follow the repo’s existing engineering rules:

- Run `git status` before edits when Git is available.
- Use `rg` for search.
- Use `apply_patch` for manual file edits.
- Preserve user changes.
- Validate narrowly for small changes and broadly for shared behavior.
- For Ultralytics micro changes, run the focused tests:

```bash
PYTHONPATH=ULTRALYTICS_MICRO .venv/bin/python -m pytest -q ULTRALYTICS_MICRO/tests/test_micro_architecture.py
```

## Training Guidance

Prefer `yolov8-micro.yaml` for the current Colab workflow because it can partially transfer from `yolov8n.pt`.

Use conservative augmentation for very small targets when accuracy matters:

```text
mosaic=0.2 scale=0.25 degrees=0 perspective=0 close_mosaic=20
```

If a user log shows default `mosaic=1.0` and `scale=0.5`, mention that the run is still valid micro training but may be aggressive for tiny crack-like targets.
