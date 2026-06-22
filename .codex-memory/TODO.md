# TODO

## Pending Tasks

- Use small Git commits as durable checkpoints after meaningful changes.
- After restart, confirm the latest commit includes any work completed before shutdown.
- Run a larger native-tile smoke benchmark once hardware memory/thermal limits are acceptable.
- Implement the full training loop that connects size-aware assignment to `TinyDetectionLoss`.
- Add decoder-to-candidate thresholding and hardware benchmark scripts once PyTorch/TensorRT are available.
- Test ONNX export and TensorRT build on an edge device with the actual accelerator stack.

## Next Steps After Restart

- Run `cd /home/open/ak`.
- Run `git status`.
- Read `.codex-memory/LAST_SESSION.md`.
- Continue from the exact next step listed there.

## Unresolved Questions

- Target class taxonomy, sensor resolution, edge hardware, dataset schema, and latency budget are not yet specified.
