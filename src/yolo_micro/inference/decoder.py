"""Torch decoding helpers for raw YOLO-Micro head outputs."""

from __future__ import annotations

from typing import Dict, Iterable

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "YOLO-Micro decoding requires PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc


DEFAULT_STRIDES = {"p1": 1, "p2": 2, "p3": 4, "p4": 8, "p5": 16}


def make_grid(height: int, width: int, stride: int, device: torch.device) -> Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    grid = torch.stack([x, y], dim=-1).float()
    return (grid + 0.5) * float(stride)


def distribution_to_distance(box_dist_logits: Tensor, reg_max: int) -> Tensor:
    """Project distribution-focal logits to continuous LTRB distances."""

    b, _, h, w = box_dist_logits.shape
    logits = box_dist_logits.view(b, 4, reg_max + 1, h, w)
    probs = F.softmax(logits, dim=2)
    bins = torch.arange(reg_max + 1, device=box_dist_logits.device, dtype=probs.dtype)
    return (probs * bins.view(1, 1, -1, 1, 1)).sum(dim=2)


def decode_level(
    level_outputs: Dict[str, Tensor],
    stride: int,
    reg_max: int,
) -> Dict[str, Tensor]:
    """Decode one pyramid level to image-space boxes and scores."""

    dist = distribution_to_distance(level_outputs["box_dist_logits"], reg_max) * float(stride)
    b, _, h, w = dist.shape
    grid = make_grid(h, w, stride, dist.device).permute(2, 0, 1).unsqueeze(0)
    offset = level_outputs["subpixel"] * float(stride)
    center = grid + offset
    left, top, right, bottom = dist.unbind(dim=1)
    cx, cy = center[:, 0], center[:, 1]
    boxes = torch.stack([cx - left, cy - top, cx + right, cy + bottom], dim=1)
    return {
        "boxes": boxes,
        "quality": level_outputs["quality_logits"].sigmoid(),
        "class_scores": level_outputs["class_logits"].sigmoid(),
        "region_embedding": level_outputs["region_embedding"],
        "mask_coefficients": level_outputs["mask_coefficients"],
        "box_log_variance": level_outputs["box_log_variance"],
    }


def decode_predictions(
    predictions: Dict[str, Dict[str, Tensor]],
    reg_max: int,
    levels: Iterable[str] = ("p1", "p2", "p3", "p4", "p5"),
    strides: Dict[str, int] | None = None,
) -> Dict[str, Dict[str, Tensor]]:
    strides = strides or DEFAULT_STRIDES
    return {
        level: decode_level(predictions[level], stride=strides[level], reg_max=reg_max)
        for level in levels
        if level in predictions
    }

