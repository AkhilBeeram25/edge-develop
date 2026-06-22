"""Tiny-object detection loss components."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "Tiny box losses require PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc

from .nwd_loss import normalized_wasserstein_similarity


def box_area(boxes: Tensor) -> Tensor:
    width = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0)
    height = (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)
    return width * height


def small_object_weight(
    target_boxes: Tensor,
    reference_area: float = 256.0,
    max_weight: float = 6.0,
) -> Tensor:
    """Area-aware positive weighting for tiny localization targets."""

    weights = torch.sqrt(torch.as_tensor(reference_area, device=target_boxes.device) / box_area(target_boxes).clamp(min=1.0))
    return weights.clamp(max=max_weight)


def box_iou(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    x0 = torch.maximum(pred_boxes[..., 0], target_boxes[..., 0])
    y0 = torch.maximum(pred_boxes[..., 1], target_boxes[..., 1])
    x1 = torch.minimum(pred_boxes[..., 2], target_boxes[..., 2])
    y1 = torch.minimum(pred_boxes[..., 3], target_boxes[..., 3])
    intersection = box_area(torch.stack([x0, y0, x1, y1], dim=-1))
    union = box_area(pred_boxes) + box_area(target_boxes) - intersection
    return intersection / union.clamp(min=1e-6)


@dataclass(frozen=True)
class DetectionLossWeights:
    box: float = 7.5
    dfl: float = 1.5
    quality: float = 1.0
    closed_cls: float = 0.5
    nwd_micro: float = 2.0
    subpixel: float = 0.5


class TinyDetectionLoss(nn.Module):
    """Loss wrapper for already-assigned positive samples.

    Assignment remains deliberately outside this class. The trainer should pass
    matched prediction/target tensors after size-aware dynamic assignment.
    """

    def __init__(
        self,
        weights: DetectionLossWeights | None = None,
        micro_area_threshold: float = 25.0,
    ) -> None:
        super().__init__()
        self.weights = weights or DetectionLossWeights()
        self.micro_area_threshold = micro_area_threshold

    def forward(
        self,
        pred_boxes: Tensor,
        target_boxes: Tensor,
        quality_logits: Tensor,
        quality_targets: Tensor | None = None,
        class_logits: Tensor | None = None,
        class_targets: Tensor | None = None,
        pred_subpixel: Tensor | None = None,
        target_subpixel: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if pred_boxes.numel() == 0:
            zero = quality_logits.sum() * 0.0
            return {"total": zero, "box": zero, "nwd_micro": zero, "quality": zero}

        pos_weights = small_object_weight(target_boxes).detach()
        normalizer = pos_weights.sum().clamp(min=1.0)

        iou = box_iou(pred_boxes, target_boxes)
        box_loss = ((1.0 - iou) * pos_weights).sum() / normalizer

        micro_mask = box_area(target_boxes) <= self.micro_area_threshold
        if micro_mask.any():
            nwd = 1.0 - normalized_wasserstein_similarity(
                pred_boxes[micro_mask],
                target_boxes[micro_mask],
            )
            nwd_loss_value = (nwd * pos_weights[micro_mask]).sum() / pos_weights[micro_mask].sum().clamp(min=1.0)
        else:
            nwd_loss_value = box_loss * 0.0

        if quality_targets is None:
            quality_targets = (0.5 * iou + 0.5 * normalized_wasserstein_similarity(pred_boxes, target_boxes)).detach()
        quality_loss = F.binary_cross_entropy_with_logits(
            quality_logits.reshape_as(quality_targets),
            quality_targets,
            reduction="none",
        )
        quality_loss = (quality_loss * pos_weights).sum() / normalizer

        total = (
            self.weights.box * box_loss
            + self.weights.nwd_micro * nwd_loss_value
            + self.weights.quality * quality_loss
        )

        losses = {
            "box": box_loss,
            "nwd_micro": nwd_loss_value,
            "quality": quality_loss,
        }

        if class_logits is not None and class_targets is not None:
            class_loss = F.binary_cross_entropy_with_logits(class_logits, class_targets, reduction="mean")
            total = total + self.weights.closed_cls * class_loss
            losses["closed_cls"] = class_loss

        if pred_subpixel is not None and target_subpixel is not None:
            subpixel_loss = F.smooth_l1_loss(pred_subpixel, target_subpixel, reduction="mean")
            total = total + self.weights.subpixel * subpixel_loss
            losses["subpixel"] = subpixel_loss

        losses["total"] = total
        return losses

