#!/usr/bin/env python3
"""Run a small synthetic YOLO UPDATE training smoke test."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_update.config import TrainConfig
from yolo_update.data.synthetic import SyntheticMicroObjectDataset
from yolo_update.engine import YOLOUpdateTrainer
from yolo_update.models import YOLOUpdateConfig
from yolo_update.utils.seed import set_seed


def build_model_config(variant: str, num_classes: int, p1_detector: bool) -> YOLOUpdateConfig:
    factories = {
        "micro_s": YOLOUpdateConfig.micro_s,
        "micro_b": YOLOUpdateConfig.micro_b,
        "micro_l": YOLOUpdateConfig.micro_l,
    }
    config = factories[variant](num_classes=num_classes)
    return replace(config, include_p1_head=p1_detector)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["micro_s", "micro_b", "micro_l"], default="micro_s")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--p1-detector", action="store_true")
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()

    set_seed(7)
    save_dir = args.save_dir or Path(tempfile.mkdtemp(prefix="yolo_update_smoke_"))
    train_config = TrainConfig(
        epochs=1,
        batch_size=2,
        image_size=args.image_size,
        learning_rate=1e-4,
        save_dir=str(save_dir),
        log_interval=1,
        device="cpu",
    )
    model_config = build_model_config(args.variant, args.num_classes, args.p1_detector)
    dataset = SyntheticMicroObjectDataset(
        length=max(4, args.steps * train_config.batch_size),
        image_size=args.image_size,
        num_classes=args.num_classes,
    )
    trainer = YOLOUpdateTrainer(model_config, train_config, dataset, dataset)
    trainer.fit(max_steps_per_epoch=args.steps)
    print(f"smoke_checkpoint={save_dir / 'last.pt'}")


if __name__ == "__main__":
    main()

