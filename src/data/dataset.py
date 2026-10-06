from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class RetinalRecord:
    dataset: str
    split: str
    image_path: Path
    vessel_gt_path: Optional[Path]
    fov_mask_path: Optional[Path]


def read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def read_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    mask = np.array(image)

    return (mask > 0).astype(np.uint8)


def _chase_image_paths(root: Path):
    return sorted(root.glob("Image_*.jpg"))


def _chase_subject_id(image_path: Path) -> int:
    # Image_01L.jpg -> 1
    name = image_path.stem
    return int(name.split("_")[1][:-1])


def build_training_records(
    drive_dir: str,
    chase_dir: str,
    chase_train_subjects: list[int],
) -> list[RetinalRecord]:

    drive_root = Path(drive_dir)
    chase_root = Path(chase_dir)

    records = []

    # ---------------------------------------------------------
    # DRIVE: 20 training images
    # ---------------------------------------------------------

    drive_images = sorted(
        (drive_root / "training" / "images").glob("*.tif")
    )

    for image_path in drive_images:

        stem = image_path.stem
        image_id = stem.split("_")[0]

        vessel_gt = (
            drive_root
            / "training"
            / "1st_manual"
            / f"{image_id}_manual1.gif"
        )

        fov_mask = (
            drive_root
            / "training"
            / "mask"
            / f"{stem}_mask.gif"
        )

        if not vessel_gt.exists():
            raise FileNotFoundError(
                f"DRIVE vessel GT not found: {vessel_gt}"
            )

        if not fov_mask.exists():
            raise FileNotFoundError(
                f"DRIVE FOV mask not found: {fov_mask}"
            )

        records.append(
            RetinalRecord(
                dataset="DRIVE",
                split="train",
                image_path=image_path,
                vessel_gt_path=vessel_gt,
                fov_mask_path=fov_mask,
            )
        )

    # ---------------------------------------------------------
    # CHASE_DB1: 20 images
    # ---------------------------------------------------------

    for image_path in _chase_image_paths(chase_root):

        subject_id = _chase_subject_id(image_path)

        if subject_id not in chase_train_subjects:
            continue

        stem = image_path.stem

        vessel_gt = (
            chase_root / f"{stem}_1stHO.png"
        )

        if not vessel_gt.exists():
            raise FileNotFoundError(
                f"CHASE_DB1 vessel GT not found: {vessel_gt}"
            )

        records.append(
            RetinalRecord(
                dataset="CHASE_DB1",
                split="train",
                image_path=image_path,
                vessel_gt_path=vessel_gt,
                fov_mask_path=None,
            )
        )

    if len(records) != 40:
        raise RuntimeError(
            f"Expected 40 training images, got {len(records)}."
        )

    return records


def build_test_records(
    drive_dir: str,
    chase_dir: str,
    chase_test_subjects: list[int],
) -> list[RetinalRecord]:

    drive_root = Path(drive_dir)
    chase_root = Path(chase_dir)

    records = []

    # ---------------------------------------------------------
    # DRIVE TEST
    # ---------------------------------------------------------

    drive_images = sorted(
        (drive_root / "test" / "images").glob("*.tif")
    )

    for image_path in drive_images:

        stem = image_path.stem
        image_id = stem.split("_")[0]

        fov_mask = (
            drive_root
            / "test"
            / "mask"
            / f"{stem}_mask.gif"
        )

        # Ground truth vessel belum tersedia pada folder user.
        possible_gt = (
            drive_root
            / "test"
            / "1st_manual"
            / f"{image_id}_manual1.gif"
        )

        vessel_gt = (
            possible_gt
            if possible_gt.exists()
            else None
        )

        records.append(
            RetinalRecord(
                dataset="DRIVE",
                split="test",
                image_path=image_path,
                vessel_gt_path=vessel_gt,
                fov_mask_path=fov_mask,
            )
        )

    # ---------------------------------------------------------
    # CHASE_DB1 TEST
    # ---------------------------------------------------------

    for image_path in _chase_image_paths(chase_root):

        subject_id = _chase_subject_id(image_path)

        if subject_id not in chase_test_subjects:
            continue

        stem = image_path.stem

        vessel_gt = (
            chase_root / f"{stem}_1stHO.png"
        )

        records.append(
            RetinalRecord(
                dataset="CHASE_DB1",
                split="test",
                image_path=image_path,
                vessel_gt_path=vessel_gt,
                fov_mask_path=None,
            )
        )

    if len(records) != 28:
        raise RuntimeError(
            f"Expected 28 test images, got {len(records)}."
        )

    return records