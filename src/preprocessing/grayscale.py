import numpy as np


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using the weighted formula
    described in the reference paper.

    Formula:
        I = 0.299R + 0.587G + 0.114B
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {image.shape}"
        )

    image = image.astype(np.float32)

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    return np.clip(gray, 0, 255).astype(np.uint8)