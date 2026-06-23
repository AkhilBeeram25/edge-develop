# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Micro-object architecture tests."""

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.data.augment import RandomPerspective
from ultralytics.nn.modules import MicroDetect, MicroFPNFusion, SPDConv
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]


def test_spdconv_preserves_all_downsample_phases() -> None:
    """SPDConv should move every 2x2 phase into channels before convolution."""
    block = SPDConv(1, 4, k=1, s=2, act=False)
    x = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    y = block.rearrange(x)
    assert y.shape == (1, 4, 2, 2)
    assert torch.equal(y[0, :, 0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))


def test_micro_fpn_fusion_projects_and_resizes() -> None:
    """MicroFPNFusion should combine unequal channel counts at the highest requested resolution."""
    fusion = MicroFPNFusion([8, 16], 12)
    y = fusion([torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8)])
    assert y.shape == (1, 12, 16, 16)


def test_exact_2px_boxes_survive_transform_candidate_filter() -> None:
    """Exact 2x2-pixel boxes must not be dropped before loss assignment."""
    box = torch.tensor([[10.0], [10.0], [12.0], [12.0]]).numpy()
    assert RandomPerspective.box_candidates(box, box).tolist() == [True]


def test_yolo26_micro_builds_p1_to_p5_detect_head() -> None:
    """The latest micro model should expose P1-P5 strides for 2x2-pixel objects."""
    model = DetectionModel(ROOT / "ultralytics/cfg/models/26/yolo26-micro.yaml", ch=3, nc=3, verbose=False)
    assert isinstance(model.model[-1], MicroDetect)
    assert model.stride.tolist() == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_yolov8_micro_builds_p1_to_p5_detect_head() -> None:
    """The YOLOv8-compatible micro model should keep DFL and expose P1-P5 detection."""
    model = DetectionModel(ROOT / "ultralytics/cfg/models/v8/yolov8-micro.yaml", ch=3, nc=3, verbose=False)
    assert isinstance(model.model[-1], MicroDetect)
    assert model.model[-1].reg_max == 16
    assert model.stride.tolist() == [2.0, 4.0, 8.0, 16.0, 32.0]
