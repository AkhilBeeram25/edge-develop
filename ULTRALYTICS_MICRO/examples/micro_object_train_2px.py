#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO-Micro on a synthetic 2x2-pixel object dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def _write_split(root: Path, split: str, count: int, image_size: int, object_size: int, seed: int) -> None:
    """Write one YOLO-format synthetic split."""
    rng = random.Random(seed)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    margin = max(2, object_size + 1)
    for index in range(count):
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        image += rng.randint(0, 12)
        x0 = rng.randint(margin, image_size - margin - object_size)
        y0 = rng.randint(margin, image_size - margin - object_size)
        x1 = x0 + object_size
        y1 = y0 + object_size
        image[y0:y1, x0:x1] = np.array([255, 255, 255], dtype=np.uint8)
        Image.fromarray(image).save(image_dir / f"{split}_{index:04d}.png")

        xc = ((x0 + x1) / 2.0) / image_size
        yc = ((y0 + y1) / 2.0) / image_size
        w = object_size / image_size
        h = object_size / image_size
        (label_dir / f"{split}_{index:04d}.txt").write_text(
            f"0 {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}\n",
            encoding="utf-8",
        )


def build_dataset(root: Path, train_count: int, val_count: int, image_size: int, object_size: int, seed: int) -> Path:
    """Build a tiny synthetic dataset and return the data YAML path."""
    _write_split(root, "train", train_count, image_size, object_size, seed)
    _write_split(root, "val", val_count, image_size, object_size, seed + 10_000)
    data_yaml = root / "micro_2px.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: micro_target",
                "nc: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def main() -> None:
    """Generate data and run a normal Ultralytics training job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "ultralytics/cfg/models/26/yolo26-micro.yaml",
        help="Path to a micro-object model YAML.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ultralytics_micro_2px"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--object-size", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data_yaml = build_dataset(
        args.work_dir / "dataset",
        train_count=args.train_samples,
        val_count=args.val_samples,
        image_size=args.image_size,
        object_size=args.object_size,
        seed=args.seed,
    )

    model = YOLO(str(args.model))
    results = model.train(
        data=str(data_yaml),
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(args.work_dir / "runs"),
        name="micro_2px",
        exist_ok=True,
        pretrained=False,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.9,
        warmup_epochs=0.0,
        patience=0,
        amp=False,
        plots=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )
    print(f"data_yaml={data_yaml}")
    print(f"run_dir={args.work_dir / 'runs' / 'micro_2px'}")
    print(f"metrics={getattr(results, 'results_dict', results)}")


if __name__ == "__main__":
    main()
