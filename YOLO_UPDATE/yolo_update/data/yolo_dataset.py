"""YOLO-format detection dataset."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from yolo_update.config import load_yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    return sorted(file for file in path.rglob("*") if file.suffix.lower() in IMAGE_EXTENSIONS)


def _read_yolo_labels(label_path: Path, image_width: int, image_height: int) -> Tensor:
    rows: list[list[float]] = []
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label row in {label_path}: {line}")
        class_id, xc, yc, width, height = map(float, parts[:5])
        x0 = (xc - width / 2.0) * image_width
        y0 = (yc - height / 2.0) * image_height
        x1 = (xc + width / 2.0) * image_width
        y1 = (yc + height / 2.0) * image_height
        rows.append([class_id, x0, y0, x1, y1])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


class YOLOFormatDetectionDataset(Dataset[tuple[Tensor, Tensor, dict[str, Any]]]):
    """Load images and normalized YOLO txt labels."""

    def __init__(
        self,
        data_yaml: str | Path,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        self.data_yaml = Path(data_yaml)
        self.data = load_yaml(self.data_yaml)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.horizontal_flip_prob = max(0.0, min(1.0, horizontal_flip_prob))
        root = Path(self.data.get("path", ".")).expanduser()
        if not root.is_absolute():
            root = (self.data_yaml.parent / root).resolve()
        image_key = "train" if split == "train" else "val"
        label_key = "train_labels" if split == "train" else "val_labels"
        self.image_dir = root / str(self.data[image_key])
        self.label_dir = root / str(self.data.get(label_key, str(self.data[image_key]).replace("images", "labels")))
        self.images = _list_images(self.image_dir)
        self.names = list(self.data.get("names", []))
        self.num_classes = int(self.data.get("nc", len(self.names)))

    def __len__(self) -> int:
        return len(self.images)

    def _label_path_for(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.image_dir)
        return (self.label_dir / relative).with_suffix(".txt")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict[str, Any]]:
        image_path = self.images[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            labels = _read_yolo_labels(self._label_path_for(image_path), original_width, original_height)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        scale_x = self.image_size / float(original_width)
        scale_y = self.image_size / float(original_height)
        if labels.numel():
            labels[:, [1, 3]] *= scale_x
            labels[:, [2, 4]] *= scale_y
            labels[:, 1:] = labels[:, 1:].clamp(min=0.0, max=float(self.image_size))
        flipped = False
        if self.augment and self.horizontal_flip_prob > 0.0:
            flipped = random.random() < self.horizontal_flip_prob
            if flipped:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if labels.numel():
                    old_x0 = labels[:, 1].clone()
                    old_x1 = labels[:, 3].clone()
                    labels[:, 1] = float(self.image_size) - old_x1
                    labels[:, 3] = float(self.image_size) - old_x0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if flipped:
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        meta = {
            "image_path": str(image_path),
            "original_size": (original_width, original_height),
            "flipped": flipped,
        }
        return tensor, labels, meta


def collate_detection_batch(
    batch: list[tuple[Tensor, Tensor, dict[str, Any]]],
) -> tuple[Tensor, list[Tensor], list[dict[str, Any]]]:
    images, targets, metas = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets), list(metas)
