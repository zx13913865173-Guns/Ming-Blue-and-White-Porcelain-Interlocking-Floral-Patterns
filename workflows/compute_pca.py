"""
PCA intrinsic dimensionality analysis with bootstrapping.

Images are resized to 128x128, flattened, standardized, and PCA is
performed over 100 bootstrap iterations. The number of components
retaining 90% variance is recorded.
"""

import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_pca_dimensionality(
    image_dir: str,
    variance_threshold: float = 0.90,
    n_bootstrap: int = 100,
    image_size: int = 128,
) -> tuple:
    """
    Compute the intrinsic dimensionality of a dataset via PCA with
    bootstrapping.

    Parameters
    ----------
    image_dir : str
        Directory containing training images.
    variance_threshold : float
        Fraction of variance to retain (default 0.90).
    n_bootstrap : int
        Number of bootstrap resamples.
    image_size : int
        Resize images to (image_size, image_size).

    Returns
    -------
    (mean_n_components, std_n_components) : tuple
    """
    images = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(image_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (image_size, image_size))
            images.append(img.flatten())

    if len(images) < 2:
        raise ValueError("Not enough images for PCA.")

    X = np.array(images, dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components_list = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
        X_boot = X_scaled[idx]
        pca = PCA().fit(X_boot)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_comp = int(np.argmax(cumsum >= variance_threshold) + 1)
        n_components_list.append(n_comp)

    return float(np.mean(n_components_list)), float(np.std(n_components_list))


if __name__ == "__main__":
    mean_n, std_n = compute_pca_dimensionality("dataset/train")
    print(f"Intrinsic dimensionality: {mean_n:.1f} ± {std_n:.1f}")
