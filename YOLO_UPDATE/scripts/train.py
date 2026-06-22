#!/usr/bin/env python3
"""Train YOLO UPDATE on a YOLO-format dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_update.config import TrainConfig, model_config_from_yaml
from yolo_update.data.yolo_dataset import YOLOFormatDetectionDataset
from yolo_update.engine import YOLOUpdateTrainer
from yolo_update.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps-per-epoch", type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    train_config = TrainConfig.from_yaml(args.train)
    model_config = model_config_from_yaml(args.model)
    train_dataset = YOLOFormatDetectionDataset(args.data, split="train", image_size=train_config.image_size)
    val_dataset = YOLOFormatDetectionDataset(args.data, split="val", image_size=train_config.image_size)
    trainer = YOLOUpdateTrainer(model_config, train_config, train_dataset, val_dataset)
    trainer.fit(max_steps_per_epoch=args.max_steps_per_epoch)


if __name__ == "__main__":
    main()

