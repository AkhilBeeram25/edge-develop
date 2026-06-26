# TODO

## Pending Tasks

- Use `.codex-memory/` and the live memory watcher for automatic status persistence; never auto-push status to Git.
- Use small local Git commits only as deliberate checkpoints when they are useful or explicitly requested.
- After restart, confirm the latest commit includes any work completed before shutdown.
- Replace `YOLO_UPDATE/configs/data/dataset.example.yaml` with the real dataset paths and class names before real training.
- Review `YOLO_UPDATE/docs/CHANGES_BENEFITS_AND_MICRO_OBJECT_MATH.md` with the model team before freezing the public architecture explanation.
- Run a larger native-tile smoke benchmark once hardware memory/thermal limits are acceptable.
- Extend the `YOLO_UPDATE` trainer with remaining production features: distributed training, LR scheduling, richer micro-object augmentations, structured metric logging, and export-time EMA selection.
- Add decoder-to-candidate thresholding and hardware benchmark scripts once PyTorch/TensorRT are available.
- Test ONNX export and TensorRT build on an edge device with the actual accelerator stack.
- For `ULTRALYTICS_MICRO/`, run a real convergence experiment on a non-synthetic micro-object dataset and compare against upstream `yolo26-p2.yaml` / `yolov8-p2.yaml`.
- For `ULTRALYTICS_MICRO/`, add size-sliced validation reporting for 2-to-5-pixel objects in the standard Ultralytics validator.
- For `ULTRALYTICS_MICRO/`, benchmark P1-P5 latency and memory at target native sensor resolution before deployment.
- Monitor the active Colab run on `/content/MPI-crack-1/data.yaml`; save final `results.csv`, `args.yaml`, and best/last checkpoint paths when training completes.
- If tiny/crack recall is weak, rerun with conservative augmentation overrides: `mosaic=0.2 scale=0.25 degrees=0 perspective=0 close_mosaic=20`.
- Use `$micro-yolo-workflow` at the start of future sessions to recover context without relying on Git history.
- Keep `skills/micro-yolo-workflow/scripts/memory_watch.py` running during long work sessions when one-second local recovery status is desired.
- Run `skills/micro-yolo-workflow/scripts/snapshot_memory.py` after meaningful status updates so tracked `.codex-memory/snapshots/` remains current even if Git is unavailable.

## Proprietary YOLO Development Queue

- Use `docs/PROPRIETARY_YOLO_DEVELOPMENT_PLAYBOOK.md` as the task breakdown for future YOLO architecture, benchmark, and Rockchip/RKNN work.
- When the user asks for a YOLO architecture change, start with a small design note: target problem, changed files, expected metric impact, tests, and benchmark gate.
- Collect final artifacts from the current Colab run: `results.csv`, `args.yaml`, `weights/best.pt`, `weights/last.pt`, plots, and any validation images.
- Add size-sliced validation reporting inside `ULTRALYTICS_MICRO/`, especially 2-to-5-pixel recall and 6-to-16-pixel recall.
- Build a benchmark manifest script that records commit SHA, model YAML, dataset YAML, weights path, image size, batch size, epochs, seed, metrics, and hardware.
- Run comparable baselines on the same dataset/settings: upstream YOLO, P2 small-object baseline when available, current `yolov8-micro.yaml`, and each candidate model.
- Add ONNX export parity checks before attempting RKNN conversion.
- Pin the target Rockchip board, RKNN-Toolkit2 version, and RKNN Model Zoo commit before making edge performance claims.
- Create RKNN conversion and board-inference scripts only after ONNX parity passes.
- Promote a candidate as the proprietary baseline only after reproducible accuracy and efficiency gains are documented.

## Next Steps After Restart

- Run `cd /home/open/ak`.
- Run `git status`.
- Read `.codex-memory/LAST_SESSION.md`.
- If present, read `.codex-memory/live/STATUS.md` for the latest local heartbeat.
- Continue from the exact next step listed there.

## Unresolved Questions

- Target class taxonomy, sensor resolution, edge hardware, dataset schema, and latency budget are not yet specified.
- Target Rockchip board, RKNN-Toolkit2 version, and RKNN Model Zoo revision are not yet pinned.
