#!/usr/bin/env python3
"""Run loss validation for YOLO UPDATE on a YOLO-format validation split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_update.config import TrainConfig, model_config_from_yaml
from yolo_update.data.yolo_dataset import YOLOFormatDetectionDataset
from yolo_update.engine import YOLOUpdateTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    train_config = TrainConfig.from_yaml(args.train)
    model_config = model_config_from_yaml(args.model)
    val_dataset = YOLOFormatDetectionDataset(args.data, split="val", image_size=train_config.image_size)
    trainer = YOLOUpdateTrainer(model_config, train_config, val_dataset, val_dataset)
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint, load_optimizer=False)
    print(trainer.validate(max_steps=args.max_steps))


if __name__ == "__main__":
    main()
