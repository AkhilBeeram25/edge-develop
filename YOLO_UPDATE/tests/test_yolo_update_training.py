from __future__ import annotations

from pathlib import Path
import tempfile

from PIL import Image
import torch

from yolo_update.config import TrainConfig
from yolo_update.data.synthetic import SyntheticMicroObjectDataset
from yolo_update.data.yolo_dataset import YOLOFormatDetectionDataset
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
        assert (Path(tmpdir) / "best.pt").exists()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["epoch"] == 1
        assert "model" in payload
        assert "ema_model" in payload
        assert "val/det/map" in payload["metrics"]

        resumed_config = TrainConfig(
            epochs=2,
            batch_size=2,
            image_size=64,
            learning_rate=1e-4,
            save_dir=tmpdir,
            device="cpu",
            resume_checkpoint=str(checkpoint),
        )
        resumed = YOLOUpdateTrainer(YOLOUpdateConfig.micro_s(num_classes=3), resumed_config, dataset, dataset)
        resumed.fit(max_steps_per_epoch=1)
        resumed_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert resumed_payload["epoch"] == 2


def test_yolo_dataset_horizontal_flip_updates_labels() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images" / "train"
        label_dir = root / "labels" / "train"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (root / "images" / "val").mkdir(parents=True)
        (root / "labels" / "val").mkdir(parents=True)

        Image.new("RGB", (10, 10), color=(0, 0, 0)).save(image_dir / "sample.jpg")
        (label_dir / "sample.txt").write_text("0 0.3 0.5 0.2 0.2\n", encoding="utf-8")
        data_yaml = root / "dataset.yaml"
        data_yaml.write_text(
            "\n".join(
                [
                    "path: .",
                    "train: images/train",
                    "val: images/val",
                    "names: [target]",
                    "nc: 1",
                ]
            ),
            encoding="utf-8",
        )

        dataset = YOLOFormatDetectionDataset(
            data_yaml,
            split="train",
            image_size=10,
            augment=True,
            horizontal_flip_prob=1.0,
        )
        _, labels, meta = dataset[0]
        assert meta["flipped"] is True
        assert torch.allclose(labels[0, 1:], torch.tensor([6.0, 4.0, 8.0, 6.0]))
