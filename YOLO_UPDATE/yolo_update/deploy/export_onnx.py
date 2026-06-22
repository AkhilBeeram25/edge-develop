"""ONNX export entry point for YOLO Update."""

from __future__ import annotations

import argparse
from pathlib import Path


def export_onnx(
    output_path: Path,
    num_classes: int,
    image_size: int = 1536,
    variant: str = "micro_b",
    opset: int = 17,
) -> None:
    try:
        import torch
        from torch import nn
        from yolo_update.models import YOLOUpdateConfig, YOLOUpdateModel
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("ONNX export requires PyTorch.") from exc

    config_factory = {
        "micro_s": YOLOUpdateConfig.micro_s,
        "micro_b": YOLOUpdateConfig.micro_b,
        "micro_l": YOLOUpdateConfig.micro_l,
    }[variant]
    model = YOLOUpdateModel(config_factory(num_classes=num_classes)).eval()

    class ExportWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, images):  # type: ignore[no-untyped-def]
            out = self.inner(images)
            tensors = [out["mask_prototypes"], out["image_logits"]]
            for level in ("p1", "p2", "p3", "p4", "p5"):
                if level not in out["predictions"]:
                    continue
                level_out = out["predictions"][level]
                tensors.extend(
                    [
                        level_out["quality_logits"],
                        level_out["box_dist_logits"],
                        level_out["subpixel"],
                        level_out["box_log_variance"],
                        level_out["class_logits"],
                        level_out["region_embedding"],
                        level_out["mask_coefficients"],
                    ]
                )
            return tuple(tensors)

    exportable = ExportWrapper(model).eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        exportable,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["images"],
        output_names=None,
        dynamic_axes=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=1536)
    parser.add_argument("--variant", choices=["micro_s", "micro_b", "micro_l"], default="micro_b")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_onnx(args.output, args.num_classes, args.image_size, args.variant, args.opset)


if __name__ == "__main__":
    main()
