#!/usr/bin/env python3
"""Run a small YOLO-Micro torch pipeline smoke test."""

from __future__ import annotations

import argparse
import time

import torch

from yolo_micro.inference.decoder import decode_predictions
from yolo_micro.losses.multitask_balancer import FixedLossWeights
from yolo_micro.losses.tiny_box_loss import TinyDetectionLoss
from yolo_micro.models import YOLOMicroConfig, YOLOMicroModel


def build_config(variant: str, num_classes: int, p1_detector: bool) -> YOLOMicroConfig:
    factories = {
        "micro_s": YOLOMicroConfig.micro_s,
        "micro_b": YOLOMicroConfig.micro_b,
        "micro_l": YOLOMicroConfig.micro_l,
    }
    config = factories[variant](num_classes=num_classes)
    return YOLOMicroConfig(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        embedding_dim=config.embedding_dim,
        mask_dim=config.mask_dim,
        reg_max=config.reg_max,
        width_mult=config.width_mult,
        depth_mult=config.depth_mult,
        include_p1_head=p1_detector,
    )


def run(args: argparse.Namespace) -> None:
    torch.set_grad_enabled(False)
    config = build_config(args.variant, args.num_classes, args.p1_detector)
    model = YOLOMicroModel(config).eval()
    images = torch.zeros(args.batch_size, 3, args.image_size, args.image_size)

    start = time.perf_counter()
    outputs = model(images)
    decoded = decode_predictions(outputs["predictions"], reg_max=config.reg_max)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    print(f"variant={args.variant} p1_detector={args.p1_detector} elapsed_ms={elapsed_ms:.2f}")
    print(f"mask_prototypes={tuple(outputs['mask_prototypes'].shape)}")
    print(f"image_logits={tuple(outputs['image_logits'].shape)}")
    print("decoded_boxes=" + str({level: tuple(data["boxes"].shape) for level, data in decoded.items()}))

    torch.set_grad_enabled(True)
    pred_boxes = torch.tensor([[10.0, 10.0, 12.0, 12.0], [20.0, 20.0, 36.0, 36.0]], requires_grad=True)
    target_boxes = torch.tensor([[10.5, 10.5, 12.5, 12.5], [21.0, 20.0, 35.0, 37.0]])
    quality_logits = torch.zeros(2, requires_grad=True)
    losses = TinyDetectionLoss()(pred_boxes, target_boxes, quality_logits)
    total = FixedLossWeights().combine({"detection": losses["total"]})
    total.backward()
    print("losses=" + str({name: round(float(value.detach()), 6) for name, value in losses.items()}))
    print(f"backward_ok={pred_boxes.grad is not None and quality_logits.grad is not None}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["micro_s", "micro_b", "micro_l"], default="micro_s")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--p1-detector", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

