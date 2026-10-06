import cv2
import numpy as np


def apply_clahe(
    gray_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to a grayscale image.

    The paper specifies CLAHE as the second preprocessing scenario,
    but does not provide OpenCV-specific parameter values.
    Therefore clip_limit and tile_grid_size are configurable.
    """

    if gray_image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image with shape (H, W), got {gray_image.shape}"
        )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    return clahe.apply(gray_image)