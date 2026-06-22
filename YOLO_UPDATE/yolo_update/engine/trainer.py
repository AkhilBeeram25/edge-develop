"""Minimal training engine for YOLO UPDATE."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from yolo_update.config import TrainConfig
from yolo_update.data.yolo_dataset import collate_detection_batch
from yolo_update.models import YOLOUpdateConfig, YOLOUpdateModel
from yolo_update.training.criterion import YOLOUpdateDetectionCriterion


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
        self.model.eval()
        running: dict[str, float] = {}
        steps = 0
        for images, targets, _ in self.val_loader:
            outputs = self.model(images.to(self.device))
            losses = self.criterion(outputs, self._move_targets(targets))
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + float(value.detach().cpu())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return {name: value / max(steps, 1) for name, value in running.items()}

    def save_checkpoint(self, path: str | Path, epoch: int, metrics: dict[str, float]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_config": self.model_config.__dict__,
                "train_config": self.train_config.__dict__,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def fit(self, max_steps_per_epoch: int | None = None) -> None:
        save_dir = Path(self.train_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, self.train_config.epochs + 1):
            train_metrics = self.train_one_epoch(epoch, max_steps=max_steps_per_epoch)
            val_metrics = self.validate(max_steps=max_steps_per_epoch)
            metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            print(f"epoch={epoch} metrics={metrics}")
            self.save_checkpoint(save_dir / "last.pt", epoch=epoch, metrics=metrics)

