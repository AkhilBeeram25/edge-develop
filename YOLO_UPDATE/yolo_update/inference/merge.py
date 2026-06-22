"""Post-processing helpers for tile and frame-level candidate merge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from yolo_update.data.native_tile_dataset import Box
from yolo_update.eval.micro_object_metrics import box_iou


@dataclass(frozen=True)
class DetectionCandidate:
    box: Box
    score: float
    label: str | None = None
    source: str | None = None


def _compatible(a: DetectionCandidate, b: DetectionCandidate) -> bool:
    return a.label == b.label or a.label is None or b.label is None


def _fuse_cluster(cluster: list[DetectionCandidate]) -> DetectionCandidate:
    total = sum(max(candidate.score, 1e-6) for candidate in cluster)
    coords = [0.0, 0.0, 0.0, 0.0]
    for candidate in cluster:
        weight = max(candidate.score, 1e-6) / total
        for idx, value in enumerate(candidate.box):
            coords[idx] += weight * value
    label_counts: dict[str | None, float] = {}
    for candidate in cluster:
        label_counts[candidate.label] = label_counts.get(candidate.label, 0.0) + candidate.score
    label = max(label_counts.items(), key=lambda item: item[1])[0]
    return DetectionCandidate(
        box=(coords[0], coords[1], coords[2], coords[3]),
        score=max(candidate.score for candidate in cluster),
        label=label,
        source="weighted_box_fusion",
    )


def weighted_box_fusion(
    candidates: Iterable[DetectionCandidate],
    iou_threshold: float = 0.55,
) -> list[DetectionCandidate]:
    """Merge overlapping candidates while preserving low-confidence micro hits."""

    remaining = sorted(candidates, key=lambda item: item.score, reverse=True)
    fused: list[DetectionCandidate] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        keep: list[DetectionCandidate] = []
        for candidate in remaining:
            if _compatible(seed, candidate) and box_iou(seed.box, candidate.box) >= iou_threshold:
                cluster.append(candidate)
            else:
                keep.append(candidate)
        fused.append(_fuse_cluster(cluster))
        remaining = keep
    return fused

