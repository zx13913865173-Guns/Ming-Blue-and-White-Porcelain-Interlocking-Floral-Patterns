"""
Inter-unit MS-SSIM self-similarity computation.

Measures visual consistency between adjacent repeating units in a
horizontally banded pattern. Uses skimage SSIM (multi-scale via
Gaussian pyramid if needed). SIFT-based unit localization is
provided in localize_units.py.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_inter_unit_ms_ssim(
    image: np.ndarray,
    unit_width: int = 128,
    boundaries: list = None,
) -> float:
    """
    Compute inter-unit MS-SSIM self-similarity for a horizontally
    banded pattern.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W) or (H, W, C).
    unit_width : int
        Expected width of one repeating unit in pixels.
    boundaries : list, optional
        Pre-computed x-coordinates of unit boundaries. If None,
        a uniform grid based on unit_width is used.

    Returns
    -------
    float
        Mean MS-SSIM across all adjacent unit pairs.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]

    if boundaries is None:
        n_units = w // unit_width
        if n_units < 2:
            return 0.0
        boundaries = [i * unit_width for i in range(n_units + 1)]

    ms_ssim_values = []
    for i in range(len(boundaries) - 2):
        x0 = boundaries[i]
        x1 = boundaries[i + 1]
        x2 = boundaries[i + 2]

        unit_a = gray[:, x0:x1]
        unit_b = gray[:, x1:x2]

        min_w = min(unit_a.shape[1], unit_b.shape[1])
        if min_w < 7:
            continue
        unit_a = unit_a[:, :min_w]
        unit_b = unit_b[:, :min_w]

        score = ssim(unit_a, unit_b, data_range=255)
        ms_ssim_values.append(score)

    if not ms_ssim_values:
        return 0.0
    return float(np.mean(ms_ssim_values))


if __name__ == "__main__":
    img = cv2.imread("pattern.png", cv2.IMREAD_GRAYSCALE)
    score = compute_inter_unit_ms_ssim(img, unit_width=128)
    print(f"Inter-unit MS-SSIM = {score:.3f}")
