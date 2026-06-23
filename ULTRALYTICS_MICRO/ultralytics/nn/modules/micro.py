# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Micro-object modules for high-resolution YOLO detection heads."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import CBAM, Conv, DWConv
from .head import Detect

__all__ = (
    "MicroC2f",
    "MicroDetect",
    "MicroDilatedBlock",
    "MicroFPNFusion",
    "MicroSPPF",
    "SPDConv",
)


def _as_tuple(values: Iterable[int] | int) -> tuple[int, ...]:
    """Normalize YAML-provided dilation values."""
    if isinstance(values, int):
        return (values,)
    return tuple(int(v) for v in values)


class SPDConv(nn.Module):
    """Space-to-depth downsampling followed by a stride-1 convolution.

    A stride-2 convolution samples only one phase of a 2x2 neighborhood. This block preserves all four phases by moving
    them into channels before convolution, which is better suited to 1-2 pixel evidence.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p=None, g: int = 1, d: int = 1, act=True):
        """Initialize SPDConv.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Convolution kernel size after space-to-depth.
            s (int): Downsampling factor. Use 2 for YOLO pyramid transitions.
            p (int, optional): Padding.
            g (int): Convolution groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.s = int(s)
        if self.s < 1:
            raise ValueError("SPDConv downsampling factor must be >= 1.")
        self.rearrange = nn.PixelUnshuffle(self.s) if self.s > 1 else nn.Identity()
        self.conv = Conv(c1 * self.s * self.s, c2, k, 1, p, g, d, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lossless spatial rearrangement then a stride-1 convolution."""
        return self.conv(self.rearrange(x))


class MicroDilatedBlock(nn.Module):
    """Lightweight residual block with depthwise dilated branches and CBAM attention."""

    def __init__(
        self,
        c1: int,
        shortcut: bool = True,
        dilations: Iterable[int] | int = (1, 2, 3),
        e: float = 0.5,
    ):
        """Initialize the micro-feature refinement block."""
        super().__init__()
        dilations = _as_tuple(dilations)
        c_ = max(8, int(c1 * e))
        self.reduce = Conv(c1, c_, 1, 1)
        self.branches = nn.ModuleList(DWConv(c_, c_, 3, d=dilation) for dilation in dilations)
        self.mix = Conv(c_ * (len(dilations) + 1), c1, 1, 1)
        self.attn = CBAM(c1, kernel_size=7)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine local pixel evidence while expanding receptive field without downsampling."""
        y = self.reduce(x)
        y = torch.cat([y, *(branch(y) for branch in self.branches)], dim=1)
        y = self.attn(self.mix(y))
        return x + y if self.add else y


class MicroC2f(nn.Module):
    """C2f-style block that replaces bottlenecks with micro-object dilated refinement."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        dilations: Iterable[int] | int = (1, 2, 3),
    ):
        """Initialize MicroC2f with Ultralytics-compatible C2f arguments."""
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(MicroDilatedBlock(self.c, shortcut, dilations=dilations) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run C2f feature aggregation with micro-refinement branches."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))


class MicroSPPF(nn.Module):
    """SPPF alternative that uses dilated depthwise context instead of max pooling."""

    def __init__(
        self,
        c1: int,
        c2: int,
        dilations: Iterable[int] | int = (1, 2, 3),
        shortcut: bool = True,
    ):
        """Initialize pooling-free spatial pyramid context."""
        super().__init__()
        dilations = _as_tuple(dilations)
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.context = nn.ModuleList(DWConv(c_, c_, 3, d=dilation) for dilation in dilations)
        self.cv2 = Conv(c_ * (len(dilations) + 1), c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate context without pooling away tiny-object responses."""
        y = self.cv1(x)
        out = self.cv2(torch.cat([y, *(layer(y) for layer in self.context)], dim=1))
        return x + out if self.add else out


class MicroFPNFusion(nn.Module):
    """Weighted BiFPN-style fusion with per-input projection and automatic resizing."""

    def __init__(self, channels: list[int], c2: int, eps: float = 1e-4):
        """Initialize learnable multi-resolution feature fusion.

        Args:
            channels (list[int]): Input channels for each source feature.
            c2 (int): Output channels after projection and fusion.
            eps (float): Numerical stability constant for normalized weights.
        """
        super().__init__()
        self.proj = nn.ModuleList(Conv(c, c2, 1, 1) for c in channels)
        self.weights = nn.Parameter(torch.ones(len(channels), dtype=torch.float32))
        self.eps = eps
        self.out = Conv(c2, c2, 3, 1)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Project, resize, and fuse feature maps with positive normalized weights."""
        target_size = xs[0].shape[-2:]
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        fused = None
        for x, proj, weight in zip(xs, self.proj, weights, strict=True):
            y = proj(x)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode="nearest")
            fused = y * weight if fused is None else fused + y * weight
        return self.out(fused)


class MicroDetect(Detect):
    """Detect head with higher candidate budget and less-suppressed P1/P2 class priors."""

    max_det = 1000

    def __init__(self, nc: int = 80, micro_prior: float = 12.0, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize a tiny-aware detection head.

        Args:
            nc (int): Number of classes.
            micro_prior (float): Bias prior for stride <= 4 detection layers.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Input channels for detection feature maps.
        """
        self.micro_prior = float(micro_prior)
        self.tiny_assign = True
        self.tiny_assign_wh = 4.0
        self.tiny_assign_sigma = 2.0
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)

    def bias_init(self):
        """Initialize object class priors with extra coverage for P1/P2 micro-object layers."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0
            prior = self.micro_prior if self.stride[i] <= 4 else 5.0
            b[-1].bias.data[: self.nc] = math.log(prior / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0
                prior = self.micro_prior if self.stride[i] <= 4 else 5.0
                b[-1].bias.data[: self.nc] = math.log(prior / self.nc / (640 / self.stride[i]) ** 2)
