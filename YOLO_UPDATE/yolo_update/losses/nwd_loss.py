"""Normalized Wasserstein distance helpers for tiny boxes."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "NWD losses require PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    width = (x1 - x0).clamp(min=0.0)
    height = (y1 - y0).clamp(min=0.0)
    return torch.stack([cx, cy, width, height], dim=-1)


def normalized_wasserstein_distance(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    """Approximate NWD for boxes represented as diagonal Gaussians."""

    pred = xyxy_to_cxcywh(pred_boxes)
    target = xyxy_to_cxcywh(target_boxes)
    center_dist = (pred[..., 0] - target[..., 0]).pow(2) + (pred[..., 1] - target[..., 1]).pow(2)
    size_dist = ((pred[..., 2] - target[..., 2]).pow(2) + (pred[..., 3] - target[..., 3]).pow(2)) / 4.0
    return center_dist + size_dist


def normalized_wasserstein_similarity(
    pred_boxes: Tensor,
    target_boxes: Tensor,
    constant: float = 12.8,
) -> Tensor:
    """Convert NWD to a similarity score in [0, 1]."""

    distance = normalized_wasserstein_distance(pred_boxes, target_boxes)
    return torch.exp(-torch.sqrt(distance.clamp(min=0.0) + 1e-9) / constant)


def nwd_loss(pred_boxes: Tensor, target_boxes: Tensor, constant: float = 12.8) -> Tensor:
    return 1.0 - normalized_wasserstein_similarity(pred_boxes, target_boxes, constant=constant)

