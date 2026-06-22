from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_micro.eval.micro_object_metrics import BoxRecord, summarize_detections_by_size
from yolo_micro.inference.merge import DetectionCandidate, weighted_box_fusion
from yolo_micro.losses.assignment import SizeAwareAssigner


class MetricsAssignmentMergeTest(unittest.TestCase):
    def test_micro_center_match_counts_recall(self) -> None:
        ground_truths = [BoxRecord(image_id="frame_1", box=(10.0, 10.0, 12.0, 12.0), label="target")]
        detections = [BoxRecord(image_id="frame_1", box=(10.5, 10.5, 12.5, 12.5), score=0.2, label="target")]

        summary = summarize_detections_by_size(
            detections,
            ground_truths,
            image_area_pixels=1000 * 1000,
        )

        self.assertEqual(summary["2_to_5_px"]["gt"], 1.0)
        self.assertEqual(summary["2_to_5_px"]["matched"], 1.0)
        self.assertEqual(summary["2_to_5_px"]["recall"], 1.0)

    def test_size_aware_assigner_routes_micro_boxes_to_p1_p2(self) -> None:
        assignment = SizeAwareAssigner().assign_box((20.0, 20.0, 22.0, 23.0))

        self.assertEqual(assignment.levels, ("p1", "p2"))
        self.assertGreaterEqual(assignment.center_radius_px, 1.0)

    def test_weighted_box_fusion_merges_compatible_candidates(self) -> None:
        fused = weighted_box_fusion(
            [
                DetectionCandidate(box=(0.0, 0.0, 10.0, 10.0), score=0.9, label="part"),
                DetectionCandidate(box=(1.0, 1.0, 11.0, 11.0), score=0.6, label="part"),
                DetectionCandidate(box=(50.0, 50.0, 60.0, 60.0), score=0.7, label="part"),
            ],
            iou_threshold=0.5,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].label, "part")
        self.assertAlmostEqual(fused[0].box[0], 0.4, places=3)


if __name__ == "__main__":
    unittest.main()

