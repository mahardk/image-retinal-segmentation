import numpy as np

from src.preprocessing.grayscale import rgb_to_grayscale
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.gamma import apply_gamma


SCENARIOS = (
    "grayscale",
    "clahe",
    "clahe_gamma",
)


def preprocess_image(
    rgb_image: np.ndarray,
    scenario: str,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
    gamma: float = 0.8,
) -> np.ndarray:

    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Choose from {SCENARIOS}."
        )

    gray = rgb_to_grayscale(rgb_image)

    if scenario == "grayscale":
        return gray

    clahe = apply_clahe(
        gray,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_grid_size,
    )

    if scenario == "clahe":
        return clahe

    return apply_gamma(
        clahe,
        gamma=gamma,
    )