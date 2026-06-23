"""Config loading helpers for YOLO UPDATE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from yolo_update.models import YOLOUpdateConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}.")
    return data


def model_config_from_yaml(path: str | Path) -> YOLOUpdateConfig:
    data = load_yaml(path)
    return YOLOUpdateConfig(
        num_classes=int(data["num_classes"]),
        input_channels=int(data.get("input_channels", 3)),
        embedding_dim=int(data.get("embedding_dim", 256)),
        mask_dim=int(data.get("mask_dim", 32)),
        reg_max=int(data.get("reg_max", 16)),
        width_mult=float(data.get("width_mult", 0.75)),
        depth_mult=float(data.get("depth_mult", 0.75)),
        include_p1_head=bool(data.get("include_p1_head", False)),
    )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 2
    image_size: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    num_workers: int = 0
    device: str = "cpu"
    save_dir: str = "runs/train/yolo_update"
    log_interval: int = 10
    negative_quality_weight: float = 0.25
    max_grad_norm: float = 10.0
    ema_decay: float = 0.999
    resume_checkpoint: str = ""
    horizontal_flip_prob: float = 0.0
    validation_score_threshold: float = 0.05
    validation_max_detections: int = 100
    validation_iou_threshold: float = 0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        data = load_yaml(path)
        return cls(
            epochs=int(data.get("epochs", cls.epochs)),
            batch_size=int(data.get("batch_size", cls.batch_size)),
            image_size=int(data.get("image_size", cls.image_size)),
            learning_rate=float(data.get("learning_rate", cls.learning_rate)),
            weight_decay=float(data.get("weight_decay", cls.weight_decay)),
            num_workers=int(data.get("num_workers", cls.num_workers)),
            device=str(data.get("device", cls.device)),
            save_dir=str(data.get("save_dir", cls.save_dir)),
            log_interval=int(data.get("log_interval", cls.log_interval)),
            negative_quality_weight=float(
                data.get("negative_quality_weight", cls.negative_quality_weight)
            ),
            max_grad_norm=float(data.get("max_grad_norm", cls.max_grad_norm)),
            ema_decay=float(data.get("ema_decay", cls.ema_decay)),
            resume_checkpoint=str(data.get("resume_checkpoint", cls.resume_checkpoint) or ""),
            horizontal_flip_prob=float(
                data.get("horizontal_flip_prob", cls.horizontal_flip_prob)
            ),
            validation_score_threshold=float(
                data.get("validation_score_threshold", cls.validation_score_threshold)
            ),
            validation_max_detections=int(
                data.get("validation_max_detections", cls.validation_max_detections)
            ),
            validation_iou_threshold=float(
                data.get("validation_iou_threshold", cls.validation_iou_threshold)
            ),
        )
