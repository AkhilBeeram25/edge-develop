# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Micro-object architecture tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import RandomPerspective
from ultralytics.engine.trainer import _metric_keys_for_training
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules import MicroDetect, MicroFPNFusion, SPDConv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.metrics import DetMetrics


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


def test_validator_records_native_pixel_target_diameter(tmp_path) -> None:
    """Size-sliced validation should use native image pixels, not letterboxed input pixels."""
    validator = DetectionValidator(save_dir=tmp_path, args={"task": "detect"})
    validator.device = torch.device("cpu")
    batch = {
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "ori_shape": [(100, 100)],
        "img": torch.zeros(1, 3, 200, 200),
        "ratio_pad": [((2.0, 2.0), (0.0, 0.0))],
        "im_file": ["image.jpg"],
    }

    pbatch = validator._prepare_batch(0, batch)

    assert torch.allclose(pbatch["target_size"], torch.tensor([20.0]))


def test_detmetrics_reports_micro_size_slice_recall() -> None:
    """DetMetrics should expose 2-to-5px and 6-to-16px target recall slices."""
    metrics = DetMetrics(names={0: "target"})
    metrics.update_stats(
        {
            "tp": np.zeros((0, 10), dtype=bool),
            "conf": np.zeros(0),
            "pred_cls": np.zeros(0),
            "target_cls": np.array([0, 0]),
            "target_img": np.array([0]),
            "target_size": np.array([2.0, 10.0]),
            "target_matched": np.array([True, False]),
            "im_name": "image.jpg",
        }
    )

    metrics.process()
    results = metrics.results_dict

    assert results["metrics/recall_2_to_5_px(B)"] == 1.0
    assert results["metrics/recall_6_to_16_px(B)"] == 0.0


def test_detmetrics_size_slice_results_keep_display_keys_separate() -> None:
    """Size-sliced metrics should not expand the per-class display metric list."""
    metrics = DetMetrics(names={0: "target"})
    result_keys = list(metrics.results_dict)

    assert metrics.keys == ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    assert "metrics/recall_2_to_5_px(B)" not in metrics.keys
    assert "metrics/recall_6_to_16_px(B)" not in metrics.keys
    assert "metrics/recall_2_to_5_px(B)" in result_keys
    assert "metrics/recall_6_to_16_px(B)" in result_keys


def test_trainer_initial_metric_keys_include_size_slices_without_fitness() -> None:
    """Initial train CSV headers should match final validation metrics without adding fitness."""
    metrics = DetMetrics(names={0: "target"})
    keys = _metric_keys_for_training(metrics, ["val/box_loss"])

    assert keys == [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/recall_2_to_5_px(B)",
        "metrics/recall_6_to_16_px(B)",
        "val/box_loss",
    ]
    assert "fitness" not in keys


def test_detmetrics_keeps_size_slice_fields_optional() -> None:
    """Non-micro validators may update DetMetrics without target size fields."""
    metrics = DetMetrics(names={0: "target"})
    metrics.update_stats(
        {
            "tp": np.zeros((0, 10), dtype=bool),
            "conf": np.zeros(0),
            "pred_cls": np.zeros(0),
            "target_cls": np.array([0]),
            "target_img": np.array([0]),
            "im_name": "image.jpg",
        }
    )

    metrics.process()
    results = metrics.results_dict

    assert results["metrics/recall_2_to_5_px(B)"] == 0.0
    assert results["metrics/recall_6_to_16_px(B)"] == 0.0
