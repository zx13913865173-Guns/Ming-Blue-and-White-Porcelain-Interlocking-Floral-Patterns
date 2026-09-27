"""
Texture Periodicity Index (TPI) computation.

Pipeline:
1. Vertical accumulation of the edge map -> 1D signal s(x).
2. Autocorrelation R(k) of s(x).
3. Locate first three local maxima (excluding zero lag).
4. Define sigma_base as the standard deviation of R(k) in non-peak regions.
5. TPI = (R1 + R2 + R3) / (3 * sigma_base).

Higher TPI indicates stronger periodic regularity.
"""

import numpy as np
import cv2
from scipy.signal import find_peaks


def compute_tpi(edge_map: np.ndarray, peak_count: int = 3) -> float:
    """
    Compute the Texture Periodicity Index (TPI) from an edge map.

    Parameters
    ----------
    edge_map : np.ndarray
        Binary or grayscale edge map (H, W).
    peak_count : int
        Number of autocorrelation peaks to use (default 3).

    Returns
    -------
    float
        TPI value. Higher = stronger periodic regularity.
    """
    if edge_map.ndim == 3:
        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)

    # Step 1: Vertical accumulation
    s = np.sum(edge_map, axis=0).astype(np.float64)
    if np.std(s) == 0:
        return 0.0
    s = (s - np.mean(s)) / np.std(s)

    # Step 2: Autocorrelation
    R = np.correlate(s, s, mode="full")
    R = R[len(R) // 2:]  # keep non-negative lags
    R = R[: len(R) // 2]  # limit to width/2

    if len(R) < 10:
        return 0.0

    # Step 3: Locate peaks (exclude zero lag)
    peaks, properties = find_peaks(
        R, distance=5, prominence=0.1 * np.max(R)
    )
    if len(peaks) < peak_count:
        return 0.0

    # Sort peaks by prominence and take top peak_count
    sorted_idx = np.argsort(properties["prominences"])[::-1]
    top_peaks = peaks[sorted_idx[:peak_count]]
    R_vals = R[top_peaks]

    # Step 4: Baseline noise
    mask = np.ones_like(R, dtype=bool)
    for p in peaks:
        mask[max(0, p - 2): min(len(R), p + 3)] = False
    sigma_base = np.std(R[mask]) if np.any(mask) else 1e-6

    # Step 5: TPI
    tpi = np.sum(R_vals) / (peak_count * sigma_base)
    return float(tpi)


if __name__ == "__main__":
    edge_map = cv2.imread("edges.png", cv2.IMREAD_GRAYSCALE)
    tpi = compute_tpi(edge_map)
    print(f"TPI = {tpi:.2f}")
