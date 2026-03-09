from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _dataset_root() -> Path:
    return _project_root() / "Version 2 - Final Model" / "Dataset"


def _images_root() -> Path:
    return _dataset_root() / "images"


def _labels_root() -> Path:
    return _dataset_root() / "labels"


def _csv_path() -> Path:
    return _project_root() / "image_coordinates_final.csv"


def _iter_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])


def _image_lookup(split: str) -> Dict[str, Path]:
    split_dir = _images_root() / split
    return {image_path.stem: image_path for image_path in _iter_images(split_dir)}


def _normalized_box(x_min: float, y_min: float, x_max: float, y_max: float, img_width: int, img_height: int) -> str:
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + (box_width / 2)
    y_center = y_min + (box_height / 2)

    x_center /= img_width
    y_center /= img_height
    box_width /= img_width
    box_height /= img_height

    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def generate_labels() -> None:
    df = pd.read_csv(_csv_path())
    df['img_name'] = df['img_name'].astype(str)
    fire_groups = df.groupby('img_name', sort=True)

    labels_root = _labels_root()
    if labels_root.exists():
        shutil.rmtree(labels_root)
    labels_root.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'val'):
        fire_dir = _images_root() / split / 'Fire'
        for image_path in _iter_images(fire_dir):
            if image_path.stem not in fire_groups.groups:
                raise ValueError(f"Fire image {image_path.name} is missing from image_coordinates_final.csv")

            with Image.open(image_path) as img:
                img_width, img_height = img.size

            label_path = labels_root / split / 'Fire' / f"{image_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for _, row in fire_groups.get_group(image_path.stem).iterrows():
                lines.append(
                    _normalized_box(
                        float(row['Xmin']),
                        float(row['Ymin']),
                        float(row['Xmax']),
                        float(row['Ymax']),
                        img_width,
                        img_height,
                    )
                )
            label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

        for category in ('Control', 'Smoke'):
            category_dir = _images_root() / split / category
            for image_path in _iter_images(category_dir):
                label_path = labels_root / split / category / f"{image_path.stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text('', encoding='utf-8')

    for cache_file in labels_root.glob('*.cache'):
        cache_file.unlink()


if __name__ == '__main__':
    generate_labels()
