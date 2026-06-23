# TODO

## Pending Tasks

- Use small Git commits as durable checkpoints after meaningful changes.
- After restart, confirm the latest commit includes any work completed before shutdown.
- Replace `YOLO_UPDATE/configs/data/dataset.example.yaml` with the real dataset paths and class names before real training.
- Review `YOLO_UPDATE/docs/CHANGES_BENEFITS_AND_MICRO_OBJECT_MATH.md` with the model team before freezing the public architecture explanation.
- Run a larger native-tile smoke benchmark once hardware memory/thermal limits are acceptable.
- Extend the `YOLO_UPDATE` trainer with remaining production features: distributed training, LR scheduling, richer micro-object augmentations, structured metric logging, and export-time EMA selection.
- Add decoder-to-candidate thresholding and hardware benchmark scripts once PyTorch/TensorRT are available.
- Test ONNX export and TensorRT build on an edge device with the actual accelerator stack.

## Next Steps After Restart

- Run `cd /home/open/ak`.
- Run `git status`.
- Read `.codex-memory/LAST_SESSION.md`.
- Continue from the exact next step listed there.

## Unresolved Questions

- Target class taxonomy, sensor resolution, edge hardware, dataset schema, and latency budget are not yet specified.
