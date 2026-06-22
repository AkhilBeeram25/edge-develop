from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from yolo_update.config import TrainConfig
from yolo_update.data.synthetic import SyntheticMicroObjectDataset
from yolo_update.engine import YOLOUpdateTrainer
from yolo_update.models import YOLOUpdateConfig


def test_synthetic_train_step_writes_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        train_config = TrainConfig(
            epochs=1,
            batch_size=2,
            image_size=64,
            learning_rate=1e-4,
            save_dir=tmpdir,
            device="cpu",
        )
        dataset = SyntheticMicroObjectDataset(length=4, image_size=64, num_classes=3)
        trainer = YOLOUpdateTrainer(YOLOUpdateConfig.micro_s(num_classes=3), train_config, dataset, dataset)
        trainer.fit(max_steps_per_epoch=1)
        checkpoint = Path(tmpdir) / "last.pt"
        assert checkpoint.exists()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["epoch"] == 1
        assert "model" in payload

