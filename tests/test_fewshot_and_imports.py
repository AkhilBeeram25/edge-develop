from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yolo_micro
from yolo_micro.data.fewshot_episode_sampler import FewShotEpisodeSampler
from yolo_micro.data.tiny_object_augment import ObjectSizeGuard, valid_scales_for_boxes


class FewShotAndImportTest(unittest.TestCase):
    def test_top_level_import_does_not_require_torch(self) -> None:
        self.assertEqual(yolo_micro.__version__, "0.1.0")

    def test_episode_sampler_is_reproducible_shape(self) -> None:
        sampler = FewShotEpisodeSampler(
            {"bolt": tuple(range(10)), "washer": tuple(range(10, 20))},
            shots=2,
            queries=3,
            seed=7,
        )

        episode = sampler.sample()

        self.assertEqual(len(episode.support), 2)
        self.assertEqual(len(episode.query), 3)
        self.assertNotEqual(set(episode.support), set(episode.query))

    def test_size_guard_rejects_erased_micro_object(self) -> None:
        guard = ObjectSizeGuard(min_side_pixels=2.0)
        boxes = [(0.0, 0.0, 4.0, 4.0)]

        self.assertFalse(guard.valid_scale(boxes, 0.25))
        self.assertEqual(valid_scales_for_boxes(boxes, [0.25, 0.5, 1.0], guard), [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()

