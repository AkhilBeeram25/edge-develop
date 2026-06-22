"""Reusable PyTorch blocks for YOLO Update."""

from __future__ import annotations

import math
from typing import Iterable

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without torch.
    raise ImportError(
        "YOLO Update model modules require PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc


def make_divisible(value: float, divisor: int = 8) -> int:
    """Round channel counts to hardware-friendly multiples."""

    if value <= 0:
        return divisor
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


def scale_channels(base: int, width_mult: float) -> int:
    return make_divisible(base * width_mult)


def scale_depth(base: int, depth_mult: float) -> int:
    return max(1, int(math.ceil(base * depth_mult)))


class ConvBNAct(nn.Module):
    """Fusable Conv-BN-Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise convolution followed by pointwise projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class SpaceToDepthConv(nn.Module):
    """Downsample by moving a 2x2 neighborhood into channels before mixing."""

    def __init__(self, in_channels: int, out_channels: int, block_size: int = 2) -> None:
        super().__init__()
        self.block_size = block_size
        self.proj = ConvBNAct(in_channels * block_size * block_size, out_channels, kernel_size=1)
        self.mix = ConvBNAct(out_channels, out_channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pixel_unshuffle(x, self.block_size)
        return self.mix(self.proj(x))


class AntiAliasDownsample(nn.Module):
    """Low-pass downsample before convolution to reduce aliasing on tiny objects."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)
        return self.proj(x)


class EdgePreserveBlock(nn.Module):
    """Narrow high-resolution block with a learned edge/detail gate."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = DepthwiseSeparableConv(channels, channels)
        self.edge_gate = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        detail = self.local(x)
        return x + self.edge_gate(x) * detail


class C2fBlock(nn.Module):
    """Compact CSP-style block similar to modern YOLO C2f modules."""

    def __init__(self, in_channels: int, out_channels: int, repeats: int) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 8)
        self.entry = ConvBNAct(in_channels, hidden * 2, kernel_size=1)
        self.blocks = nn.ModuleList(
            DepthwiseSeparableConv(hidden, hidden) for _ in range(max(1, repeats))
        )
        self.exit = ConvBNAct(hidden * (2 + len(self.blocks)), out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        left, right = self.entry(x).chunk(2, dim=1)
        features = [left, right]
        y = right
        for block in self.blocks:
            y = block(y)
            features.append(y)
        return self.exit(torch.cat(features, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling fast block."""

    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 5) -> None:
        super().__init__()
        hidden = max(in_channels // 2, 8)
        self.entry = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.exit = ConvBNAct(hidden * 4, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.entry(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.exit(torch.cat([x, y1, y2, y3], dim=1))


class ConvStack(nn.Module):
    """Small stack of fusable convolution blocks."""

    def __init__(self, channels: Iterable[int]) -> None:
        super().__init__()
        channels = list(channels)
        pairs = list(zip(channels, channels[1:]))
        self.layers = nn.Sequential(*(ConvBNAct(a, b) for a, b in pairs))

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
