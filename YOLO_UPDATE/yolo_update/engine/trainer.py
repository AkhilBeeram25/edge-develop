"""Minimal training engine for YOLO UPDATE."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Dataset

from yolo_update.config import TrainConfig
from yolo_update.data.yolo_dataset import collate_detection_batch
from yolo_update.eval.detection_metrics import mean_average_precision
from yolo_update.eval.micro_object_metrics import BoxRecord, summarize_detections_by_size
from yolo_update.inference.decoder import decode_predictions
from yolo_update.models import YOLOUpdateConfig, YOLOUpdateModel
from yolo_update.training.criterion import YOLOUpdateDetectionCriterion


class ModelEMA:
    """Exponential moving average weights for validation and export."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


class YOLOUpdateTrainer:
    def __init__(
        self,
        model_config: YOLOUpdateConfig,
        train_config: TrainConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.device = torch.device(train_config.device)
        self.model = YOLOUpdateModel(model_config).to(self.device)
        self.criterion = YOLOUpdateDetectionCriterion(
            num_classes=model_config.num_classes,
            reg_max=model_config.reg_max,
            negative_quality_weight=train_config.negative_quality_weight,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.ema = ModelEMA(self.model, train_config.ema_decay) if train_config.ema_decay > 0.0 else None
        self.start_epoch = 1
        self.best_metric = float("-inf")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers,
            collate_fn=collate_detection_batch,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=train_config.num_workers,
                collate_fn=collate_detection_batch,
            )

    def _move_targets(self, targets: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        return [target.to(self.device) for target in targets]

    def _image_ids_for_batch(self, metas: list[dict[str, object]]) -> list[str]:
        image_ids: list[str] = []
        for fallback_index, meta in enumerate(metas):
            if "image_path" in meta:
                image_ids.append(str(meta["image_path"]))
            elif "synthetic_index" in meta:
                image_ids.append(f"synthetic:{meta['synthetic_index']}")
            else:
                image_ids.append(f"batch_item:{fallback_index}")
        return image_ids

    def _targets_to_records(self, targets: list[Tensor], image_ids: list[str]) -> list[BoxRecord]:
        records: list[BoxRecord] = []
        for image_id, target_rows in zip(image_ids, targets, strict=True):
            for row in target_rows.detach().cpu().tolist():
                records.append(
                    BoxRecord(
                        image_id=image_id,
                        box=(float(row[1]), float(row[2]), float(row[3]), float(row[4])),
                        score=1.0,
                        label=str(int(row[0])),
                    )
                )
        return records

    def _detections_from_outputs(
        self,
        outputs: dict[str, object],
        image_ids: list[str],
    ) -> list[BoxRecord]:
        predictions = outputs["predictions"]
        if not isinstance(predictions, dict):
            raise TypeError("outputs['predictions'] must be a prediction dictionary.")
        decoded = decode_predictions(predictions, reg_max=self.model_config.reg_max)
        records: list[BoxRecord] = []
        max_detections = max(1, self.train_config.validation_max_detections)
        score_threshold = self.train_config.validation_score_threshold

        for batch_index, image_id in enumerate(image_ids):
            image_candidates: list[BoxRecord] = []
            for level_outputs in decoded.values():
                boxes = level_outputs["boxes"][batch_index]
                quality = level_outputs["quality"][batch_index, 0]
                class_scores = level_outputs["class_scores"][batch_index]
                scores = class_scores * quality.unsqueeze(0)
                num_classes, height, width = scores.shape
                flat_scores = scores.reshape(-1)
                top_k = min(max_detections, flat_scores.numel())
                top_scores, top_indices = torch.topk(flat_scores, k=top_k)
                cells_per_class = height * width
                for score_tensor, index_tensor in zip(top_scores, top_indices, strict=True):
                    score = float(score_tensor.detach().cpu())
                    if score < score_threshold:
                        continue
                    flat_index = int(index_tensor.detach().cpu())
                    class_index = flat_index // cells_per_class
                    cell_index = flat_index % cells_per_class
                    if class_index >= num_classes:
                        continue
                    y_index = cell_index // width
                    x_index = cell_index % width
                    box_tensor = boxes[:, y_index, x_index].detach().cpu()
                    image_candidates.append(
                        BoxRecord(
                            image_id=image_id,
                            box=(
                                float(box_tensor[0]),
                                float(box_tensor[1]),
                                float(box_tensor[2]),
                                float(box_tensor[3]),
                            ),
                            score=score,
                            label=str(class_index),
                        )
                    )
            image_candidates.sort(key=lambda item: item.score, reverse=True)
            records.extend(image_candidates[:max_detections])
        return records

    def train_one_epoch(self, epoch: int, max_steps: int | None = None) -> dict[str, float]:
        self.model.train()
        running: dict[str, float] = {}
        steps = 0
        for step, (images, targets, _) in enumerate(self.train_loader, start=1):
            images = images.to(self.device)
            targets = self._move_targets(targets)
            outputs = self.model(images)
            losses = self.criterion(outputs, targets)
            loss = losses["total"]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.train_config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)

            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + float(value.detach().cpu())
            steps += 1
            if step % self.train_config.log_interval == 0:
                print(f"epoch={epoch} step={step} loss={float(loss.detach().cpu()):.4f}")
            if max_steps is not None and steps >= max_steps:
                break
        return {name: value / max(steps, 1) for name, value in running.items()}

    @torch.no_grad()
    def validate(self, max_steps: int | None = None) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        eval_model = self.ema.module if self.ema is not None else self.model
        eval_model.eval()
        running: dict[str, float] = {}
        detections: list[BoxRecord] = []
        ground_truths: list[BoxRecord] = []
        image_area_pixels = 0
        steps = 0
        for images, targets, metas in self.val_loader:
            outputs = eval_model(images.to(self.device))
            losses = self.criterion(outputs, self._move_targets(targets))
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + float(value.detach().cpu())
            image_ids = self._image_ids_for_batch(metas)
            detections.extend(self._detections_from_outputs(outputs, image_ids))
            ground_truths.extend(self._targets_to_records(targets, image_ids))
            image_area_pixels += int(images.shape[0] * images.shape[-2] * images.shape[-1])
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        metrics = {name: value / max(steps, 1) for name, value in running.items()}
        detection_metrics = mean_average_precision(
            detections,
            ground_truths,
            iou_threshold=self.train_config.validation_iou_threshold,
        )
        metrics.update({f"det/{name}": value for name, value in detection_metrics.items()})
        size_summary = summarize_detections_by_size(
            detections,
            ground_truths,
            image_area_pixels=max(image_area_pixels, 1),
            iou_threshold=min(self.train_config.validation_iou_threshold, 0.3),
        )
        if "2_to_5_px" in size_summary:
            metrics["det/recall_2_to_5_px"] = size_summary["2_to_5_px"]["recall"]
            metrics["det/fp_per_mpix_2_to_5_px"] = size_summary["2_to_5_px"][
                "false_positives_per_megapixel"
            ]
        return metrics

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_config": self.model_config.__dict__,
            "train_config": self.train_config.__dict__,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
        }
        if self.ema is not None:
            payload["ema_model"] = self.ema.module.state_dict()
        torch.save(
            payload,
            path,
        )

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.ema is not None:
            ema_payload = payload.get("ema_model", payload["model"])
            self.ema.module.load_state_dict(ema_payload)
        epoch = int(payload.get("epoch", 0))
        self.start_epoch = epoch + 1
        self.best_metric = float(payload.get("best_metric", float("-inf")))
        return epoch

    def fit(self, max_steps_per_epoch: int | None = None) -> None:
        save_dir = Path(self.train_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.train_config.resume_checkpoint:
            loaded_epoch = self.load_checkpoint(self.train_config.resume_checkpoint)
            print(f"resumed_checkpoint={self.train_config.resume_checkpoint} epoch={loaded_epoch}")

        for epoch in range(self.start_epoch, self.train_config.epochs + 1):
            train_metrics = self.train_one_epoch(epoch, max_steps=max_steps_per_epoch)
            val_metrics = self.validate(max_steps=max_steps_per_epoch)
            metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            print(f"epoch={epoch} metrics={metrics}")
            best_candidate = metrics.get("val/det/map", -metrics.get("val/total", metrics["train/total"]))
            is_best = best_candidate >= self.best_metric
            if is_best:
                self.best_metric = best_candidate
            self.save_checkpoint(save_dir / "last.pt", epoch=epoch, metrics=metrics)
            if is_best:
                self.save_checkpoint(save_dir / "best.pt", epoch=epoch, metrics=metrics)
