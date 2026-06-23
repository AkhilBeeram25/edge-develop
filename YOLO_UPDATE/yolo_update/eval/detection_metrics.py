"""Detection AP metrics for decoded YOLO UPDATE candidates."""

from __future__ import annotations

from collections.abc import Iterable

from yolo_update.eval.micro_object_metrics import BoxRecord, box_iou


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    total = 0.0
    for point in range(101):
        threshold = point / 100.0
        best_precision = 0.0
        for recall, precision in zip(recalls, precisions, strict=True):
            if recall >= threshold:
                best_precision = max(best_precision, precision)
        total += best_precision
    return total / 101.0


def _score_label(
    detections: list[BoxRecord],
    ground_truths: list[BoxRecord],
    iou_threshold: float,
) -> dict[str, float]:
    total_gt = len(ground_truths)
    if total_gt == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "true_positive": 0.0,
            "false_positive": float(len(detections)),
            "ground_truth": 0.0,
        }

    matched: set[int] = set()
    true_positive = 0.0
    false_positive = 0.0
    recalls: list[float] = []
    precisions: list[float] = []

    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(ground_truths):
            if index in matched or detection.image_id != target.image_id:
                continue
            iou = box_iou(detection.box, target.box)
            if iou > best_iou:
                best_index = index
                best_iou = iou
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add(best_index)
            true_positive += 1.0
        else:
            false_positive += 1.0
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / total_gt
        precisions.append(precision)
        recalls.append(recall)

    return {
        "ap": _average_precision(recalls, precisions),
        "precision": true_positive / max(true_positive + false_positive, 1.0),
        "recall": true_positive / total_gt,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "ground_truth": float(total_gt),
    }


def mean_average_precision(
    detections: Iterable[BoxRecord],
    ground_truths: Iterable[BoxRecord],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute class-mean AP plus aggregate precision/recall."""

    detections = list(detections)
    ground_truths = list(ground_truths)
    labels = sorted({target.label for target in ground_truths})
    if not labels:
        return {
            "map": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_detections": float(len(detections)),
            "num_targets": 0.0,
        }

    ap_values: list[float] = []
    true_positive = 0.0
    false_positive = 0.0
    total_gt = 0.0
    for label in labels:
        label_detections = [item for item in detections if item.label == label]
        label_targets = [item for item in ground_truths if item.label == label]
        scores = _score_label(label_detections, label_targets, iou_threshold)
        ap_values.append(scores["ap"])
        true_positive += scores["true_positive"]
        false_positive += scores["false_positive"]
        total_gt += scores["ground_truth"]

    return {
        "map": sum(ap_values) / max(len(ap_values), 1),
        "precision": true_positive / max(true_positive + false_positive, 1.0),
        "recall": true_positive / max(total_gt, 1.0),
        "num_detections": float(len(detections)),
        "num_targets": total_gt,
    }
