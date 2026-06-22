#!/usr/bin/env python3
"""Validate that YOLO UPDATE preserves the YOLO-family architecture contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo_update.models import YOLOUpdateConfig, YOLOUpdateModel


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    config = YOLOUpdateConfig.micro_s(num_classes=3)
    model = YOLOUpdateModel(config).eval()

    require(hasattr(model, "backbone"), "Missing YOLO-style backbone.")
    require(hasattr(model, "neck"), "Missing YOLO-style neck.")
    require(hasattr(model, "head"), "Missing YOLO-style dense head.")
    for name in ("c2", "c3", "c4", "c5", "c6"):
        require(hasattr(model.backbone, name), f"Missing staged backbone block {name}.")
    for level in ("p1", "p2", "p3", "p4", "p5"):
        require(level in model.neck.out_channels, f"Missing pyramid level {level}.")

    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))
    require(sorted(outputs["predictions"].keys()) == ["p2", "p3", "p4", "p5"], "Default detection must be P2-P5.")
    p2 = outputs["predictions"]["p2"]
    expected_box_channels = 4 * (config.reg_max + 1)
    require(
        p2["box_dist_logits"].shape[1] == expected_box_channels,
        "Box distribution branch does not match DFL-style channel count.",
    )
    require(p2["quality_logits"].shape[1] == 1, "Missing objectness/quality branch.")
    require(p2["class_logits"].shape[1] == config.num_classes, "Missing closed-set class branch.")

    p1_config = replace(config, include_p1_head=True)
    p1_model = YOLOUpdateModel(p1_config).eval()
    with torch.no_grad():
        p1_outputs = p1_model(torch.zeros(1, 3, 64, 64))
    require("p1" in p1_outputs["predictions"], "P1 accuracy mode did not enable P1 detector.")

    print("PASS: YOLO UPDATE is a YOLO-family architecture with targeted micro-object modifications.")


if __name__ == "__main__":
    main()

