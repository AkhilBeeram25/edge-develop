#!/usr/bin/env python3
"""Forward/decode/loss smoke test for YOLO UPDATE."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_update.inference.decoder import decode_predictions
from yolo_update.models import YOLOUpdateConfig, YOLOUpdateModel
from yolo_update.training import YOLOUpdateDetectionCriterion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["micro_s", "micro_b", "micro_l"], default="micro_s")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--p1-detector", action="store_true")
    args = parser.parse_args()

    config = getattr(YOLOUpdateConfig, args.variant)(num_classes=args.num_classes)
    config = replace(config, include_p1_head=args.p1_detector)
    model = YOLOUpdateModel(config).eval()
    images = torch.zeros(1, 3, args.image_size, args.image_size)
    targets = [torch.tensor([[0.0, 10.0, 10.0, 13.0, 13.0]])]
    criterion = YOLOUpdateDetectionCriterion(num_classes=args.num_classes, reg_max=config.reg_max)
    start = time.perf_counter()
    outputs = model(images)
    decoded = decode_predictions(outputs["predictions"], reg_max=config.reg_max)
    losses = criterion(outputs, targets)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"elapsed_ms={elapsed_ms:.2f}")
    print(f"levels={sorted(outputs['predictions'].keys())}")
    print({level: tuple(data["boxes"].shape) for level, data in decoded.items()})
    print({name: round(float(value.detach()), 6) for name, value in losses.items()})


if __name__ == "__main__":
    main()

