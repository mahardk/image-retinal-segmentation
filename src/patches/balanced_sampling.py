from dataclasses import dataclass

import numpy as np

from src.data.dataset import RetinalRecord, read_mask


@dataclass
class PatchLocation:
    record_index: int
    y: int
    x: int
    label_type: str


def _valid_coordinates(mask: np.ndarray, patch_size: int):
    """
    Generate valid top-left coordinates for a patch.
    """
    h, w = mask.shape

    for y in range(0, h - patch_size + 1):
        for x in range(0, w - patch_size + 1):
            yield y, x


def build_balanced_patch_index(
    records: list[RetinalRecord],
    patch_size: int = 48,
    patches_per_image: int = 1000,
    seed: int = 42,
):
    """
    Build a balanced random patch index.

    Based on the reference paper:
    - patch size = 48x48
    - 40,000 patches from 40 training images
    - balanced random sampling

    The paper does not specify the exact balancing algorithm.
    Therefore, this implementation uses a median split of vessel
    coverage as an explicit implementation choice.

    For each image:
    - collect valid 48x48 patches
    - calculate vessel-pixel ratio
    - split candidates around the median ratio
    - randomly sample half from the lower group
    - randomly sample half from the upper group
    """

    rng = np.random.default_rng(seed)
    all_locations = []

    patches_per_group = patches_per_image // 2

    if patches_per_image % 2 != 0:
        raise ValueError(
            "patches_per_image must be even for balanced sampling."
        )

    for record_index, record in enumerate(records):

        if record.vessel_gt_path is None:
            raise ValueError(
                f"Missing vessel ground truth for {record.image_path}"
            )

        vessel = read_mask(record.vessel_gt_path)

        if record.fov_mask_path is not None:
            fov = read_mask(record.fov_mask_path)
        else:
            fov = np.ones_like(vessel, dtype=np.uint8)

        candidates = []

        for y, x in _valid_coordinates(fov, patch_size):

            fov_patch = fov[
                y:y + patch_size,
                x:x + patch_size
            ]

            # Keep patches completely inside the valid field of view.
            if fov_patch.mean() < 1.0:
                continue

            vessel_patch = vessel[
                y:y + patch_size,
                x:x + patch_size
            ]

            vessel_ratio = float(vessel_patch.mean())

            candidates.append((y, x, vessel_ratio))

        if len(candidates) < patches_per_image:
            raise RuntimeError(
                f"Not enough valid patches in {record.image_path}. "
                f"Available: {len(candidates)}, "
                f"required: {patches_per_image}"
            )

        # Sort candidates by vessel coverage.
        candidates.sort(key=lambda item: item[2])

        # Median split.
        midpoint = len(candidates) // 2

        lower_group = candidates[:midpoint]
        upper_group = candidates[midpoint:]

        if len(lower_group) == 0 or len(upper_group) == 0:
            raise RuntimeError(
                f"Unable to create balanced groups for "
                f"{record.image_path}"
            )

        lower_indices = rng.choice(
            len(lower_group),
            size=patches_per_group,
            replace=len(lower_group) < patches_per_group,
        )

        upper_indices = rng.choice(
            len(upper_group),
            size=patches_per_group,
            replace=len(upper_group) < patches_per_group,
        )

        for index in lower_indices:
            y, x, _ = lower_group[index]

            all_locations.append(
                PatchLocation(
                    record_index=record_index,
                    y=y,
                    x=x,
                    label_type="lower_vessel_ratio",
                )
            )

        for index in upper_indices:
            y, x, _ = upper_group[index]

            all_locations.append(
                PatchLocation(
                    record_index=record_index,
                    y=y,
                    x=x,
                    label_type="higher_vessel_ratio",
                )
            )

    expected = len(records) * patches_per_image

    if len(all_locations) != expected:
        raise RuntimeError(
            f"Expected {expected} patches, "
            f"got {len(all_locations)}."
        )

    rng.shuffle(all_locations)

    return all_locations