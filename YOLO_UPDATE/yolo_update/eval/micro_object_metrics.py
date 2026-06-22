"""Micro-object detection metrics and size-sliced summaries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from yolo_update.data.native_tile_dataset import Box, box_area, box_center


@dataclass(frozen=True)
class SizeBin:
    name: str
    min_diameter: float
    max_diameter: float

    def contains(self, diameter: float) -> bool:
        return self.min_diameter <= diameter < self.max_diameter


DEFAULT_SIZE_BINS: tuple[SizeBin, ...] = (
    SizeBin("2_to_5_px", 2.0, 6.0),
    SizeBin("6_to_16_px", 6.0, 17.0),
    SizeBin("17_to_64_px", 16.0, 64.0),
    SizeBin("gt_64_px", 64.0, float("inf")),
)


@dataclass(frozen=True)
class BoxRecord:
    image_id: str
    box: Box
    score: float = 1.0
    label: str | None = None


def box_diameter(box: Box) -> float:
    return math.sqrt(max(box_area(box), 0.0))


def box_iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    intersection = box_area((ix0, iy0, ix1, iy1))
    union = box_area(a) + box_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_distance(a: Box, b: Box) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)


def _coerce_record(record: BoxRecord | Mapping[str, object]) -> BoxRecord:
    if isinstance(record, BoxRecord):
        return record
    return BoxRecord(
        image_id=str(record["image_id"]),
        box=tuple(record["box"]),  # type: ignore[arg-type]
        score=float(record.get("score", 1.0)),
        label=record.get("label"),  # type: ignore[arg-type]
    )


def _bin_for_box(box: Box, bins: Sequence[SizeBin]) -> str:
    diameter = box_diameter(box)
    for size_bin in bins:
        if size_bin.contains(diameter):
            return size_bin.name
    return "unbinned"


def summarize_detections_by_size(
    detections: Iterable[BoxRecord | Mapping[str, object]],
    ground_truths: Iterable[BoxRecord | Mapping[str, object]],
    image_area_pixels: int,
    bins: Sequence[SizeBin] = DEFAULT_SIZE_BINS,
    iou_threshold: float = 0.3,
    micro_center_tolerance_px: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Compute recall and false-positive density by target size bin."""

    detections_by_image: dict[str, list[BoxRecord]] = {}
    for det in map(_coerce_record, detections):
        detections_by_image.setdefault(det.image_id, []).append(det)
    for dets in detections_by_image.values():
        dets.sort(key=lambda item: item.score, reverse=True)

    gt_records = list(map(_coerce_record, ground_truths))
    stats = {
        size_bin.name: {"gt": 0.0, "matched": 0.0, "false_positive": 0.0, "loc_error_px": 0.0}
        for size_bin in bins
    }
    stats["unbinned"] = {"gt": 0.0, "matched": 0.0, "false_positive": 0.0, "loc_error_px": 0.0}

    matched_detection_ids: set[tuple[str, int]] = set()

    for gt in gt_records:
        bin_name = _bin_for_box(gt.box, bins)
        stats[bin_name]["gt"] += 1.0
        best: tuple[int, BoxRecord, float, float] | None = None
        for det_index, det in enumerate(detections_by_image.get(gt.image_id, [])):
            if (gt.image_id, det_index) in matched_detection_ids:
                continue
            if gt.label is not None and det.label is not None and gt.label != det.label:
                continue
            iou = box_iou(gt.box, det.box)
            distance = center_distance(gt.box, det.box)
            micro_match = box_diameter(gt.box) < 5.0 and distance <= micro_center_tolerance_px
            if iou >= iou_threshold or micro_match:
                rank_score = iou - 0.01 * distance
                if best is None or rank_score > best[2]:
                    best = (det_index, det, rank_score, distance)
        if best is not None:
            matched_detection_ids.add((gt.image_id, best[0]))
            stats[bin_name]["matched"] += 1.0
            stats[bin_name]["loc_error_px"] += best[3]

    for image_id, dets in detections_by_image.items():
        for det_index, det in enumerate(dets):
            if (image_id, det_index) in matched_detection_ids:
                continue
            bin_name = _bin_for_box(det.box, bins)
            stats.setdefault(
                bin_name,
                {"gt": 0.0, "matched": 0.0, "false_positive": 0.0, "loc_error_px": 0.0},
            )
            stats[bin_name]["false_positive"] += 1.0

    megapixels = max(float(image_area_pixels) / 1_000_000.0, 1e-9)
    for values in stats.values():
        gt_count = values["gt"]
        matched = values["matched"]
        values["recall"] = matched / gt_count if gt_count else 0.0
        values["false_positives_per_megapixel"] = values["false_positive"] / megapixels
        values["mean_loc_error_px"] = values["loc_error_px"] / matched if matched else 0.0
    return stats
