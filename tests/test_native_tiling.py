from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_micro.data.native_tile_dataset import NativeTilePlanner, box_area


class NativeTilePlannerTest(unittest.TestCase):
    def test_tiles_cover_frame_edges(self) -> None:
        planner = NativeTilePlanner(tile_size=1536, overlap=128)
        tiles = planner.tiles(image_width=4096, image_height=2160)

        self.assertEqual(tiles[0].origin, (0, 0))
        self.assertEqual(tiles[-1].x1, 4096)
        self.assertEqual(tiles[-1].y1, 2160)
        self.assertTrue(any(tile.contains_point(4095.5, 2159.5) for tile in tiles))

    def test_assigns_tiny_box_to_overlapping_tiles(self) -> None:
        planner = NativeTilePlanner(tile_size=100, overlap=20)
        boxes = [(78.0, 10.0, 82.0, 14.0)]
        assignments = planner.assign_boxes(boxes, image_width=180, image_height=100)

        assigned_box_count = sum(len(items) for items in assignments.values())
        self.assertGreaterEqual(assigned_box_count, 1)
        for items in assignments.values():
            for _, local_box in items:
                self.assertGreater(box_area(local_box), 0.0)


if __name__ == "__main__":
    unittest.main()

