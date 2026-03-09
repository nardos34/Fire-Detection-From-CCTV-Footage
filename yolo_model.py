from __future__ import annotations

import os
import shutil
from pathlib import Path

from ultralytics import YOLO

from yolo_model_labels import generate_labels


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _dataset_root() -> Path:
    return _project_root() / "Version 2 - Final Model" / "Dataset"


def _config_path() -> Path:
    return _project_root() / "yolo_model_config.yaml"


def _write_dataset_config() -> None:
    dataset_root = _dataset_root()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    missing = [str(p) for p in required_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset directories:\n" + "\n".join(missing)
        )

    config_text = (
        f"path: {dataset_root.as_posix()}\n\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: fire\n"
    )

    _config_path().write_text(config_text, encoding="utf-8")


def main() -> None:
    project_root = _project_root()
    output_dir = project_root / "Version 2 - Final Model" / "train"

    generate_labels()
    _write_dataset_config()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    model = YOLO(str(project_root / "yolov8n.pt"))
    model.train(
        data=str(_config_path()),
        epochs=int(os.getenv("YOLO_EPOCHS", "80")),
        imgsz=int(os.getenv("YOLO_IMGSZ", "640")),
        batch=int(os.getenv("YOLO_BATCH", "16")),
        project=str(project_root / "Version 2 - Final Model"),
        name="train",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()