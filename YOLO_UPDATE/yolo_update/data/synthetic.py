"""Synthetic datasets for smoke-training YOLO UPDATE."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticMicroObjectDataset(Dataset[tuple[Tensor, Tensor, dict[str, object]]]):
    """Generate simple images with tiny square targets."""

    def __init__(
        self,
        length: int = 16,
        image_size: int = 64,
        num_classes: int = 3,
        object_size: int = 3,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.object_size = object_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict[str, object]]:
        generator = torch.Generator().manual_seed(index)
        image = torch.rand((3, self.image_size, self.image_size), generator=generator) * 0.05
        margin = max(4, self.object_size + 1)
        x0 = int(torch.randint(margin, self.image_size - margin, (1,), generator=generator))
        y0 = int(torch.randint(margin, self.image_size - margin, (1,), generator=generator))
        x1 = min(self.image_size - 1, x0 + self.object_size)
        y1 = min(self.image_size - 1, y0 + self.object_size)
        class_id = index % self.num_classes
        image[:, y0:y1, x0:x1] = 1.0
        target = torch.tensor([[float(class_id), float(x0), float(y0), float(x1), float(y1)]])
        return image, target, {"synthetic_index": index}

