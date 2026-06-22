"""Weighted high-resolution BiFPN/PAN neck for YOLO-Micro."""

from __future__ import annotations

from typing import Dict, Iterable

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "YOLOMicroNeck requires PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc

from ..blocks import AntiAliasDownsample, ConvBNAct, DepthwiseSeparableConv, scale_channels


class WeightedFusion(nn.Module):
    """Learned non-negative feature fusion."""

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: Iterable[Tensor]) -> Tensor:
        tensors = list(inputs)
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        out = tensors[0] * weights[0]
        for tensor, weight in zip(tensors[1:], weights[1:]):
            out = out + tensor * weight
        return out


class MicroFusion(nn.Module):
    """Inject semantic context into a narrow P1 detail map without blurring it."""

    def __init__(self, detail_channels: int, semantic_channels: int, out_channels: int) -> None:
        super().__init__()
        self.detail_proj = ConvBNAct(detail_channels, out_channels, kernel_size=1)
        self.semantic_proj = ConvBNAct(semantic_channels, out_channels, kernel_size=1)
        self.gate = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, kernel_size=3),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, detail: Tensor, semantic: Tensor) -> Tensor:
        detail = self.detail_proj(detail)
        semantic = self.semantic_proj(semantic)
        semantic = F.interpolate(semantic, size=detail.shape[-2:], mode="nearest")
        gate = self.gate(torch.cat([detail, semantic], dim=1))
        return self.refine(detail + gate * semantic)


class YOLOMicroNeck(nn.Module):
    """Bi-directional pyramid neck with explicit P1/P2 micro-object support."""

    def __init__(
        self,
        in_channels: Dict[str, int],
        width_mult: float = 0.75,
        fpn_channels: int | None = None,
        p1_channels: int | None = None,
    ) -> None:
        super().__init__()
        c_fpn = fpn_channels or scale_channels(192, width_mult)
        c_p1 = p1_channels or scale_channels(64, width_mult)

        self.lateral = nn.ModuleDict(
            {
                "c2": ConvBNAct(in_channels["c2"], c_fpn, kernel_size=1),
                "c3": ConvBNAct(in_channels["c3"], c_fpn, kernel_size=1),
                "c4": ConvBNAct(in_channels["c4"], c_fpn, kernel_size=1),
                "c5": ConvBNAct(in_channels["c5"], c_fpn, kernel_size=1),
                "c6": ConvBNAct(in_channels["c6"], c_fpn, kernel_size=1),
            }
        )

        self.fuse5 = WeightedFusion(2)
        self.fuse4 = WeightedFusion(2)
        self.fuse3 = WeightedFusion(2)
        self.fuse2 = WeightedFusion(2)

        self.refine5 = DepthwiseSeparableConv(c_fpn, c_fpn)
        self.refine4 = DepthwiseSeparableConv(c_fpn, c_fpn)
        self.refine3 = DepthwiseSeparableConv(c_fpn, c_fpn)
        self.refine2 = DepthwiseSeparableConv(c_fpn, c_fpn)

        self.p1_fusion = MicroFusion(in_channels["p1_detail"], c_fpn, c_p1)

        self.down2 = AntiAliasDownsample(c_fpn, c_fpn)
        self.down3 = AntiAliasDownsample(c_fpn, c_fpn)
        self.down4 = AntiAliasDownsample(c_fpn, c_fpn)
        self.pan3 = WeightedFusion(2)
        self.pan4 = WeightedFusion(2)
        self.pan5 = WeightedFusion(2)
        self.pan_refine3 = DepthwiseSeparableConv(c_fpn, c_fpn)
        self.pan_refine4 = DepthwiseSeparableConv(c_fpn, c_fpn)
        self.pan_refine5 = DepthwiseSeparableConv(c_fpn, c_fpn)

        self.out_channels = {
            "p1": c_p1,
            "p2": c_fpn,
            "p3": c_fpn,
            "p4": c_fpn,
            "p5": c_fpn,
        }

    @staticmethod
    def _upsample_like(x: Tensor, ref: Tensor) -> Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        c2 = self.lateral["c2"](features["c2"])
        c3 = self.lateral["c3"](features["c3"])
        c4 = self.lateral["c4"](features["c4"])
        c5 = self.lateral["c5"](features["c5"])
        c6 = self.lateral["c6"](features["c6"])

        p5 = self.refine5(self.fuse5([c5, self._upsample_like(c6, c5)]))
        p4 = self.refine4(self.fuse4([c4, self._upsample_like(p5, c4)]))
        p3 = self.refine3(self.fuse3([c3, self._upsample_like(p4, c3)]))
        p2 = self.refine2(self.fuse2([c2, self._upsample_like(p3, c2)]))
        p1 = self.p1_fusion(features["p1_detail"], p2)

        p3 = self.pan_refine3(self.pan3([p3, self.down2(p2)]))
        p4 = self.pan_refine4(self.pan4([p4, self.down3(p3)]))
        p5 = self.pan_refine5(self.pan5([p5, self.down4(p4)]))

        return {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5}

