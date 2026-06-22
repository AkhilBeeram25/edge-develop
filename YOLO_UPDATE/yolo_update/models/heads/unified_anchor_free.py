"""Unified anchor-free prediction head."""

from __future__ import annotations

from typing import Dict, Iterable

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "UnifiedAnchorFreeHead requires PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc

from ..blocks import ConvBNAct, DepthwiseSeparableConv, scale_channels


class _HeadBranch(nn.Module):
    def __init__(self, channels: int, out_channels: int, depth: int = 2) -> None:
        super().__init__()
        blocks = [DepthwiseSeparableConv(channels, channels) for _ in range(depth)]
        blocks.append(nn.Conv2d(channels, out_channels, kernel_size=1))
        self.branch = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.branch(x)


class UnifiedAnchorFreeHead(nn.Module):
    """Per-level multi-task head for detection, masks, and embeddings."""

    def __init__(
        self,
        in_channels: Dict[str, int],
        num_classes: int,
        levels: Iterable[str] = ("p1", "p2", "p3", "p4", "p5"),
        width_mult: float = 0.75,
        head_channels: int | None = None,
        reg_max: int = 16,
        embedding_dim: int = 256,
        mask_dim: int = 32,
    ) -> None:
        super().__init__()
        self.levels = tuple(levels)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.embedding_dim = embedding_dim
        self.mask_dim = mask_dim
        c_head = head_channels or scale_channels(160, width_mult)

        self.stems = nn.ModuleDict()
        self.quality = nn.ModuleDict()
        self.box_dist = nn.ModuleDict()
        self.subpixel = nn.ModuleDict()
        self.box_uncertainty = nn.ModuleDict()
        self.closed_cls = nn.ModuleDict()
        self.embedding = nn.ModuleDict()
        self.mask_coeff = nn.ModuleDict()

        for level in self.levels:
            in_ch = in_channels[level]
            self.stems[level] = nn.Sequential(
                ConvBNAct(in_ch, c_head, kernel_size=3),
                ConvBNAct(c_head, c_head, kernel_size=3),
            )
            self.quality[level] = _HeadBranch(c_head, 1)
            self.box_dist[level] = _HeadBranch(c_head, 4 * (reg_max + 1))
            self.subpixel[level] = _HeadBranch(c_head, 2)
            self.box_uncertainty[level] = _HeadBranch(c_head, 4)
            self.closed_cls[level] = _HeadBranch(c_head, num_classes)
            self.embedding[level] = _HeadBranch(c_head, embedding_dim)
            self.mask_coeff[level] = _HeadBranch(c_head, mask_dim)

        self._init_biases()

    def _init_biases(self) -> None:
        for branch in self.quality.values():
            conv = branch.branch[-1]
            nn.init.constant_(conv.bias, -4.5)
        for branch in self.closed_cls.values():
            conv = branch.branch[-1]
            nn.init.constant_(conv.bias, -4.5)

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        outputs: Dict[str, Dict[str, Tensor]] = {}
        for level in self.levels:
            x = self.stems[level](features[level])
            embedding = F.normalize(self.embedding[level](x), dim=1)
            outputs[level] = {
                "quality_logits": self.quality[level](x),
                "box_dist_logits": self.box_dist[level](x),
                "subpixel": self.subpixel[level](x).tanh(),
                "box_log_variance": self.box_uncertainty[level](x).clamp(min=-8.0, max=8.0),
                "class_logits": self.closed_cls[level](x),
                "region_embedding": embedding,
                "mask_coefficients": self.mask_coeff[level](x),
            }
        return outputs

