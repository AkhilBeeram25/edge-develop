"""Guards for augmentation policies that could erase tiny objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .native_tile_dataset import Box


@dataclass(frozen=True)
class ObjectSizeGuard:
    """Validate that augmentation keeps micro objects physically visible."""

    min_side_pixels: float = 2.0
    min_area_pixels: float = 4.0

    def scaled_box_is_visible(self, box: Box, scale_x: float, scale_y: float | None = None) -> bool:
        scale_y = scale_x if scale_y is None else scale_y
        x0, y0, x1, y1 = box
        width = max(0.0, x1 - x0) * scale_x
        height = max(0.0, y1 - y0) * scale_y
        area = width * height
        return width >= self.min_side_pixels and height >= self.min_side_pixels and area >= self.min_area_pixels

    def valid_scale(self, boxes: Iterable[Box], scale_x: float, scale_y: float | None = None) -> bool:
        return all(self.scaled_box_is_visible(box, scale_x, scale_y) for box in boxes)


def filter_visible_boxes(
    boxes: Iterable[Box],
    scale_x: float,
    scale_y: float | None = None,
    guard: ObjectSizeGuard | None = None,
) -> list[Box]:
    guard = guard or ObjectSizeGuard()
    return [box for box in boxes if guard.scaled_box_is_visible(box, scale_x, scale_y)]


def valid_scales_for_boxes(
    boxes: Iterable[Box],
    candidate_scales: Iterable[float],
    guard: ObjectSizeGuard | None = None,
) -> list[float]:
    guard = guard or ObjectSizeGuard()
    boxes = list(boxes)
    return [scale for scale in candidate_scales if guard.valid_scale(boxes, scale)]

