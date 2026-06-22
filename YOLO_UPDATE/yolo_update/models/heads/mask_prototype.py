"""Prototype mask branch for unified detection and segmentation."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "MaskPrototypeHead requires PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc

from ..blocks import ConvBNAct, DepthwiseSeparableConv


class MaskPrototypeHead(nn.Module):
    """Build high-resolution mask prototypes from P2 with P3 semantic support."""

    def __init__(self, p2_channels: int, p3_channels: int, mask_dim: int = 32) -> None:
        super().__init__()
        self.p2_proj = ConvBNAct(p2_channels, p2_channels, kernel_size=3)
        self.p3_proj = ConvBNAct(p3_channels, p2_channels, kernel_size=1)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(p2_channels, p2_channels),
            DepthwiseSeparableConv(p2_channels, p2_channels),
            nn.Conv2d(p2_channels, mask_dim, kernel_size=1),
        )

    def forward(self, p2: Tensor, p3: Tensor) -> Tensor:
        p2 = self.p2_proj(p2)
        p3 = self.p3_proj(p3)
        p3 = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        return self.refine(p2 + p3)

