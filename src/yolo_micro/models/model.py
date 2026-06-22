"""Top-level YOLO-Micro model assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    from torch import Tensor, nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "YOLOMicroModel requires PyTorch. Install with "
        '`python3 -m pip install -e ".[torch]"`.'
    ) from exc

from .backbone import YOLOMicroBackbone
from .heads import MaskPrototypeHead, UnifiedAnchorFreeHead
from .neck import YOLOMicroNeck


@dataclass(frozen=True)
class YOLOMicroConfig:
    """Model configuration for the YOLO-Micro family."""

    num_classes: int
    input_channels: int = 3
    embedding_dim: int = 256
    mask_dim: int = 32
    reg_max: int = 16
    width_mult: float = 0.75
    depth_mult: float = 0.75
    include_p1_head: bool = True

    @classmethod
    def micro_s(cls, num_classes: int) -> "YOLOMicroConfig":
        return cls(num_classes=num_classes, width_mult=0.50, depth_mult=0.50)

    @classmethod
    def micro_b(cls, num_classes: int) -> "YOLOMicroConfig":
        return cls(num_classes=num_classes, width_mult=0.75, depth_mult=0.75)

    @classmethod
    def micro_l(cls, num_classes: int) -> "YOLOMicroConfig":
        return cls(num_classes=num_classes, width_mult=1.00, depth_mult=1.00)


class YOLOMicroModel(nn.Module):
    """Unified model with detection, segmentation, and image classification outputs."""

    def __init__(self, config: YOLOMicroConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = YOLOMicroBackbone(
            in_channels=config.input_channels,
            width_mult=config.width_mult,
            depth_mult=config.depth_mult,
        )
        self.neck = YOLOMicroNeck(
            self.backbone.out_channels,
            width_mult=config.width_mult,
        )
        levels = ("p1", "p2", "p3", "p4", "p5") if config.include_p1_head else ("p2", "p3", "p4", "p5")
        self.head = UnifiedAnchorFreeHead(
            self.neck.out_channels,
            num_classes=config.num_classes,
            levels=levels,
            width_mult=config.width_mult,
            reg_max=config.reg_max,
            embedding_dim=config.embedding_dim,
            mask_dim=config.mask_dim,
        )
        self.mask_head = MaskPrototypeHead(
            self.neck.out_channels["p2"],
            self.neck.out_channels["p3"],
            mask_dim=config.mask_dim,
        )
        self.image_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.neck.out_channels["p5"], config.num_classes),
        )

    def forward(self, x: Tensor, return_features: bool = False) -> Dict[str, object]:
        backbone_features = self.backbone(x)
        pyramid = self.neck(backbone_features.as_dict())
        predictions = self.head(pyramid)
        out: Dict[str, object] = {
            "predictions": predictions,
            "mask_prototypes": self.mask_head(pyramid["p2"], pyramid["p3"]),
            "image_logits": self.image_classifier(pyramid["p5"]),
        }
        if return_features:
            out["backbone_features"] = backbone_features
            out["pyramid_features"] = pyramid
        return out

