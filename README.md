# YOLO-Micro

YOLO-Micro is the first implementation scaffold for the proprietary YOLO-codenamed vision model described in `docs/YOLO_TECHNICAL_ROADMAP.md`.

The current build focuses on Phase 1 foundations:

- Native-resolution tiling and micro-object evaluation utilities.
- PyTorch model modules for a P1/P2-preserving YOLO-Micro backbone.
- Weighted BiFPN/PAN neck with micro-feature fusion.
- Unified anchor-free heads for objectness, box distributions, masks, closed-set classes, and region embeddings.
- Loss helpers for tiny boxes, normalized Wasserstein distance, and multi-task balancing.
- Deployment stubs for ONNX, TensorRT, and quantization-aware training.

## Environment

The package metadata keeps PyTorch optional so standard-library utilities and tests can run on lightweight edge devices. Model training and forward-pass validation require PyTorch:

```bash
python3 -m pip install -e ".[torch,dev]"
```

On ARM64, PyPI currently resolves a GPU-capable PyTorch distribution with large CUDA runtime packages. Use a workspace-local virtual environment and expect several gigabytes of disk use.

## Smoke Checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
python3 -m py_compile $(find src tests -name "*.py")
.venv/bin/python scripts/smoke_torch_pipeline.py --variant micro_s --image-size 64 --num-classes 3
```

The default Micro-B architecture uses P1 as a detail/refinement path and detects on P2-P5. Enable a P1 detection head only for accuracy-mode experiments where the hardware budget can absorb the high-resolution head cost.

## Phase 1 Build Target

The immediate engineering target is a working YOLO-Micro-B detector at native tile resolution:

1. Validate the tile planner and metrics on real sensor dimensions.
2. Install PyTorch on the target training machine.
3. Run a forward-pass smoke test with `YOLOMicroModel`.
4. Wire the assignment/loss path into the training loop.
5. Benchmark FP16 export on the edge target.
