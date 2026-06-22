from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")


def test_micro_s_forward_decode_default_uses_p2_to_p5() -> None:
    import torch

    from yolo_micro.inference.decoder import decode_predictions
    from yolo_micro.models import YOLOMicroConfig, YOLOMicroModel

    model = YOLOMicroModel(YOLOMicroConfig.micro_s(num_classes=3)).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))
        decoded = decode_predictions(outputs["predictions"], reg_max=model.config.reg_max)

    assert sorted(outputs["predictions"]) == ["p2", "p3", "p4", "p5"]
    assert tuple(outputs["mask_prototypes"].shape) == (1, 32, 32, 32)
    assert tuple(outputs["image_logits"].shape) == (1, 3)
    assert tuple(decoded["p2"]["boxes"].shape) == (1, 4, 32, 32)


def test_micro_s_forward_decode_accuracy_mode_includes_p1() -> None:
    import torch

    from yolo_micro.inference.decoder import decode_predictions
    from yolo_micro.models import YOLOMicroConfig, YOLOMicroModel

    config = YOLOMicroConfig.micro_s(num_classes=3)
    config = YOLOMicroConfig(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        embedding_dim=config.embedding_dim,
        mask_dim=config.mask_dim,
        reg_max=config.reg_max,
        width_mult=config.width_mult,
        depth_mult=config.depth_mult,
        include_p1_head=True,
    )
    model = YOLOMicroModel(config).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))
        decoded = decode_predictions(outputs["predictions"], reg_max=model.config.reg_max)

    assert "p1" in outputs["predictions"]
    assert tuple(decoded["p1"]["boxes"].shape) == (1, 4, 64, 64)


def test_tiny_detection_loss_backward() -> None:
    import torch

    from yolo_micro.losses.multitask_balancer import FixedLossWeights
    from yolo_micro.losses.tiny_box_loss import TinyDetectionLoss

    pred_boxes = torch.tensor(
        [[10.0, 10.0, 12.0, 12.0], [20.0, 20.0, 36.0, 36.0]],
        requires_grad=True,
    )
    target_boxes = torch.tensor([[10.5, 10.5, 12.5, 12.5], [21.0, 20.0, 35.0, 37.0]])
    quality_logits = torch.zeros(2, requires_grad=True)

    losses = TinyDetectionLoss()(pred_boxes, target_boxes, quality_logits)
    total = FixedLossWeights().combine({"detection": losses["total"]})
    total.backward()

    assert float(losses["total"].detach()) > 0.0
    assert pred_boxes.grad is not None
    assert quality_logits.grad is not None

