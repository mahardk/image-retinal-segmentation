import numpy as np


def apply_gamma(
    image: np.ndarray,
    gamma: float = 0.8,
) -> np.ndarray:
    """
    Apply gamma correction.

    Formula:
        Iout = (Iin / 255)^gamma * 255

    Gamma < 1 makes darker intensity values brighter.
    """

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image with shape (H, W), got {image.shape}"
        )

    if gamma <= 0:
        raise ValueError(
            f"Gamma must be greater than 0, got {gamma}"
        )

    image_float = image.astype(np.float32) / 255.0

    corrected = np.power(
        image_float,
        gamma,
    ) * 255.0

    return np.clip(
        corrected,
        0,
        255,
    ).astype(np.uint8)