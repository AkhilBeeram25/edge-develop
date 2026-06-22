from __future__ import annotations

import torch

from yolo_update.models import YOLOUpdateConfig, YOLOUpdateModel


def test_default_detection_is_p2_to_p5() -> None:
    model = YOLOUpdateModel(YOLOUpdateConfig.micro_s(num_classes=3)).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))
    assert sorted(outputs["predictions"].keys()) == ["p2", "p3", "p4", "p5"]


def test_p1_accuracy_mode_is_explicit() -> None:
    config = YOLOUpdateConfig.micro_s(num_classes=3)
    config = YOLOUpdateConfig(
        num_classes=config.num_classes,
        width_mult=config.width_mult,
        depth_mult=config.depth_mult,
        include_p1_head=True,
    )
    model = YOLOUpdateModel(config).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))
    assert "p1" in outputs["predictions"]

