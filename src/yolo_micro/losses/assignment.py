"""Size-aware assignment policy for YOLO-Micro training."""

from __future__ import annotations

from dataclasses import dataclass

from yolo_micro.data.native_tile_dataset import Box


@dataclass(frozen=True)
class LevelAssignment:
    levels: tuple[str, ...]
    center_radius_px: float


def box_size(box: Box) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0), max(0.0, y1 - y0)


class SizeAwareAssigner:
    """Choose pyramid levels and center radius from image-space object size."""

    def __init__(self, min_radius: float = 1.0, max_radius: float = 4.0) -> None:
        self.min_radius = min_radius
        self.max_radius = max_radius

    def assign_box(self, box: Box) -> LevelAssignment:
        width, height = box_size(box)
        max_side = max(width, height)
        min_side = max(min(width, height), 1e-6)

        if max_side <= 5.0:
            levels = ("p1", "p2")
        elif max_side <= 16.0:
            levels = ("p2", "p3")
        elif max_side <= 64.0:
            levels = ("p3", "p4")
        else:
            levels = ("p4", "p5")

        radius = max(self.min_radius, min(self.max_radius, 0.75 * min_side))
        return LevelAssignment(levels=levels, center_radius_px=radius)

