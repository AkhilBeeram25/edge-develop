"""Training criterion that connects YOLO UPDATE predictions to targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from yolo_update.inference.decoder import DEFAULT_STRIDES, decode_predictions
from yolo_update.losses.assignment import SizeAwareAssigner
from yolo_update.losses.nwd_loss import normalized_wasserstein_similarity
from yolo_update.losses.tiny_box_loss import TinyDetectionLoss


@dataclass(frozen=True)
class AssignmentRecord:
    batch_index: int
    level: str
    y_index: int
    x_index: int
    class_id: int
    box: tuple[float, float, float, float]


class YOLOUpdateDetectionCriterion(nn.Module):
    """First trainable detector criterion for YOLO UPDATE."""

    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        negative_quality_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.negative_quality_weight = negative_quality_weight
        self.assigner = SizeAwareAssigner()
        self.tiny_loss = TinyDetectionLoss()

    def _assign_targets(
        self,
        predictions: dict[str, dict[str, Tensor]],
        targets: list[Tensor],
    ) -> list[AssignmentRecord]:
        assignments: list[AssignmentRecord] = []
        occupied: set[tuple[int, str, int, int]] = set()
        available_levels = set(predictions)
        for batch_index, target_rows in enumerate(targets):
            for row in target_rows.detach().cpu().tolist():
                class_id = int(row[0])
                box = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
                for level in self.assigner.assign_box(box).levels:
                    if level not in available_levels:
                        continue
                    stride = DEFAULT_STRIDES[level]
                    _, _, height, width = predictions[level]["quality_logits"].shape
                    cx = 0.5 * (box[0] + box[2])
                    cy = 0.5 * (box[1] + box[3])
                    x_index = max(0, min(width - 1, int(cx / stride)))
                    y_index = max(0, min(height - 1, int(cy / stride)))
                    key = (batch_index, level, y_index, x_index)
                    if key in occupied:
                        continue
                    occupied.add(key)
                    assignments.append(
                        AssignmentRecord(
                            batch_index=batch_index,
                            level=level,
                            y_index=y_index,
                            x_index=x_index,
                            class_id=class_id,
                            box=box,
                        )
                    )
        return assignments

    def forward(
        self,
        outputs: dict[str, object],
        targets: list[Tensor],
    ) -> dict[str, Tensor]:
        predictions = outputs["predictions"]
        if not isinstance(predictions, dict):
            raise TypeError("outputs['predictions'] must be a prediction dictionary.")
        decoded = decode_predictions(predictions, reg_max=self.reg_max)
        assignments = self._assign_targets(predictions, targets)

        device = next(iter(predictions.values()))["quality_logits"].device
        quality_map_loss = torch.zeros((), device=device)
        for level, level_outputs in predictions.items():
            target_map = torch.zeros_like(level_outputs["quality_logits"])
            for record in assignments:
                if record.level == level:
                    target_map[record.batch_index, 0, record.y_index, record.x_index] = 1.0
            quality_map_loss = quality_map_loss + F.binary_cross_entropy_with_logits(
                level_outputs["quality_logits"],
                target_map,
                reduction="mean",
            )

        if not assignments:
            total = self.negative_quality_weight * quality_map_loss
            return {
                "total": total,
                "positive": total * 0.0,
                "quality_map": quality_map_loss,
                "num_assignments": torch.zeros((), device=device),
            }

        pred_boxes: list[Tensor] = []
        target_boxes: list[Tensor] = []
        quality_logits: list[Tensor] = []
        class_logits: list[Tensor] = []
        class_targets: list[Tensor] = []

        for record in assignments:
            level_decoded = decoded[record.level]
            level_raw = predictions[record.level]
            pred_boxes.append(
                level_decoded["boxes"][
                    record.batch_index,
                    :,
                    record.y_index,
                    record.x_index,
                ]
            )
            target_boxes.append(torch.tensor(record.box, device=device, dtype=torch.float32))
            quality_logits.append(
                level_raw["quality_logits"][
                    record.batch_index,
                    0,
                    record.y_index,
                    record.x_index,
                ]
            )
            class_logits.append(
                level_raw["class_logits"][
                    record.batch_index,
                    :,
                    record.y_index,
                    record.x_index,
                ]
            )
            one_hot = torch.zeros(self.num_classes, device=device)
            if 0 <= record.class_id < self.num_classes:
                one_hot[record.class_id] = 1.0
            class_targets.append(one_hot)

        pred_box_tensor = torch.stack(pred_boxes)
        target_box_tensor = torch.stack(target_boxes)
        quality_tensor = torch.stack(quality_logits)
        class_logit_tensor = torch.stack(class_logits)
        class_target_tensor = torch.stack(class_targets)
        iou_nwd_quality = normalized_wasserstein_similarity(
            pred_box_tensor,
            target_box_tensor,
        ).detach()

        positive_losses = self.tiny_loss(
            pred_box_tensor,
            target_box_tensor,
            quality_tensor,
            quality_targets=iou_nwd_quality,
            class_logits=class_logit_tensor,
            class_targets=class_target_tensor,
        )
        total = positive_losses["total"] + self.negative_quality_weight * quality_map_loss
        positive_losses["positive"] = positive_losses["total"]
        positive_losses["quality_map"] = quality_map_loss
        positive_losses["num_assignments"] = torch.tensor(float(len(assignments)), device=device)
        positive_losses["total"] = total
        return positive_losses

