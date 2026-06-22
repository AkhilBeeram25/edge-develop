# TODO

## Pending Tasks

- Use small Git commits as durable checkpoints after meaningful changes.
- After restart, confirm the latest commit includes any work completed before shutdown.
- Install PyTorch on a training or edge-validation machine and run a YOLO-Micro-B forward-pass smoke test.
- Implement the full training loop that connects size-aware assignment to `TinyDetectionLoss`.
- Add decoder-to-candidate thresholding and hardware benchmark scripts once PyTorch/TensorRT are available.

## Next Steps After Restart

- Run `cd /home/open/ak`.
- Run `git status`.
- Read `.codex-memory/LAST_SESSION.md`.
- Continue from the exact next step listed there.

## Unresolved Questions

- Target class taxonomy, sensor resolution, edge hardware, and dataset schema are not yet specified.
