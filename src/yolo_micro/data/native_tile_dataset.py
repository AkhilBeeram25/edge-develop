"""Native-resolution tiling utilities.

These utilities are dependency-free by design so they can run in data QA,
annotation tooling, and edge pre/post-processing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Tile:
    """A rectangular crop in image coordinates."""

    index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    image_width: int
    image_height: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def origin(self) -> tuple[int, int]:
        return self.x0, self.y0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def to_local_box(self, box: Box, clip: bool = True) -> Box | None:
        """Convert an image-space box into tile-local coordinates."""

        x0, y0, x1, y1 = box
        if clip:
            x0 = max(x0, self.x0)
            y0 = max(y0, self.y0)
            x1 = min(x1, self.x1)
            y1 = min(y1, self.y1)
            if x1 <= x0 or y1 <= y0:
                return None
        return (x0 - self.x0, y0 - self.y0, x1 - self.x0, y1 - self.y0)

    def to_image_box(self, local_box: Box) -> Box:
        x0, y0, x1, y1 = local_box
        return (x0 + self.x0, y0 + self.y0, x1 + self.x0, y1 + self.y0)


class NativeTilePlanner:
    """Plan overlapping high-resolution tiles for a native sensor frame."""

    def __init__(self, tile_size: int = 1536, overlap: int = 128) -> None:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if overlap >= tile_size:
            raise ValueError("overlap must be smaller than tile_size.")
        self.tile_size = tile_size
        self.overlap = overlap
        self.step = tile_size - overlap

    def axis_starts(self, length: int) -> list[int]:
        if length <= 0:
            raise ValueError("Image dimensions must be positive.")
        if length <= self.tile_size:
            return [0]

        starts: list[int] = []
        pos = 0
        while pos + self.tile_size < length:
            starts.append(pos)
            pos += self.step
        last = max(0, length - self.tile_size)
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    def tiles(self, image_width: int, image_height: int) -> list[Tile]:
        xs = self.axis_starts(image_width)
        ys = self.axis_starts(image_height)
        tiles: list[Tile] = []
        index = 0
        for row, y0 in enumerate(ys):
            for col, x0 in enumerate(xs):
                tiles.append(
                    Tile(
                        index=index,
                        row=row,
                        col=col,
                        x0=x0,
                        y0=y0,
                        x1=min(x0 + self.tile_size, image_width),
                        y1=min(y0 + self.tile_size, image_height),
                        image_width=image_width,
                        image_height=image_height,
                    )
                )
                index += 1
        return tiles

    def assign_boxes(
        self,
        boxes: Sequence[Box],
        image_width: int,
        image_height: int,
        min_visible_fraction: float = 0.2,
    ) -> dict[int, list[tuple[int, Box]]]:
        """Assign image-space boxes to all tiles that contain enough area."""

        assignments: dict[int, list[tuple[int, Box]]] = {}
        for tile in self.tiles(image_width, image_height):
            local: list[tuple[int, Box]] = []
            for box_index, box in enumerate(boxes):
                clipped = tile.to_local_box(box, clip=True)
                if clipped is None:
                    continue
                if box_area(clipped) / max(box_area(box), 1e-6) >= min_visible_fraction:
                    local.append((box_index, clipped))
            if local:
                assignments[tile.index] = local
        return assignments


def box_area(box: Box) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_center(box: Box) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def boxes_touch(box: Box, tile: Tile) -> bool:
    x0, y0, x1, y1 = box
    return not (x1 <= tile.x0 or x0 >= tile.x1 or y1 <= tile.y0 or y0 >= tile.y1)


def iter_tiles_for_box(box: Box, tiles: Iterable[Tile]) -> list[Tile]:
    return [tile for tile in tiles if boxes_touch(box, tile)]

