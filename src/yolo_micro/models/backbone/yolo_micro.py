"""P1/P2-preserving YOLO-Micro backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    from torch import Tensor, nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "YOLOMicroBackbone requires PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc

from ..blocks import (
    AntiAliasDownsample,
    C2fBlock,
    ConvBNAct,
    EdgePreserveBlock,
    SPPF,
    SpaceToDepthConv,
    scale_channels,
    scale_depth,
)


@dataclass(frozen=True)
class BackboneFeatures:
    """Backbone feature maps keyed by their effective stride."""

    p1_detail: Tensor
    c2: Tensor
    c3: Tensor
    c4: Tensor
    c5: Tensor
    c6: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "p1_detail": self.p1_detail,
            "c2": self.c2,
            "c3": self.c3,
            "c4": self.c4,
            "c5": self.c5,
            "c6": self.c6,
        }


class YOLOMicroBackbone(nn.Module):
    """Backbone that preserves native-resolution detail for micro objects."""

    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 0.75,
        depth_mult: float = 0.75,
        detail_channels: int | None = None,
    ) -> None:
        super().__init__()
        c_stem_a = scale_channels(32, width_mult)
        c_stem_b = scale_channels(48, width_mult)
        c_detail = detail_channels or scale_channels(24, width_mult)
        c2 = scale_channels(96, width_mult)
        c3 = scale_channels(160, width_mult)
        c4 = scale_channels(256, width_mult)
        c5 = scale_channels(384, width_mult)
        c6 = scale_channels(512, width_mult)

        self.stem_a = ConvBNAct(in_channels, c_stem_a, kernel_size=3, stride=1)
        self.stem_b = ConvBNAct(c_stem_a, c_stem_b, kernel_size=3, stride=1)
        self.detail_proj = ConvBNAct(c_stem_b, c_detail, kernel_size=1)
        self.detail_blocks = nn.Sequential(
            *(EdgePreserveBlock(c_detail) for _ in range(scale_depth(2, depth_mult)))
        )

        self.c2_down = SpaceToDepthConv(c_stem_b, c2)
        self.c2 = C2fBlock(c2, c2, repeats=scale_depth(3, depth_mult))

        self.c3_down = AntiAliasDownsample(c2, c3)
        self.c3 = C2fBlock(c3, c3, repeats=scale_depth(4, depth_mult))

        self.c4_down = AntiAliasDownsample(c3, c4)
        self.c4 = C2fBlock(c4, c4, repeats=scale_depth(6, depth_mult))

        self.c5_down = AntiAliasDownsample(c4, c5)
        self.c5 = nn.Sequential(
            C2fBlock(c5, c5, repeats=scale_depth(6, depth_mult)),
            SPPF(c5, c5),
        )

        self.c6_down = AntiAliasDownsample(c5, c6)
        self.c6 = C2fBlock(c6, c6, repeats=scale_depth(3, depth_mult))

        self.out_channels = {
            "p1_detail": c_detail,
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
            "c6": c6,
        }

    def forward(self, x: Tensor) -> BackboneFeatures:
        stem = self.stem_b(self.stem_a(x))
        p1_detail = self.detail_blocks(self.detail_proj(stem))

        c2 = self.c2(self.c2_down(stem))
        c3 = self.c3(self.c3_down(c2))
        c4 = self.c4(self.c4_down(c3))
        c5 = self.c5(self.c5_down(c4))
        c6 = self.c6(self.c6_down(c5))
        return BackboneFeatures(p1_detail=p1_detail, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6)

