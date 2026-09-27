"""
Objective metric evaluation: SSIM, PSNR, LPIPS, FID.

- SSIM and PSNR: skimage
- LPIPS: lpips library
- FID: clean-fid
"""

import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_ssim_psnr(gen_path: str, ref_path: str) -> tuple:
    gen = cv2.imread(gen_path, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if gen is None or ref is None:
        raise FileNotFoundError("Image not found.")
    if gen.shape != ref.shape:
        gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]))
    s = ssim(gen, ref, data_range=255)
    p = psnr(ref, gen, data_range=255)
    return float(s), float(p)


def compute_lpips(gen_dir: str, ref_dir: str) -> float:
    import torch
    import lpips

    loss_fn = lpips.LPIPS(net="alex")
    files = sorted(
        f for f in os.listdir(gen_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    values = []
    for fname in files:
        gen_path = os.path.join(gen_dir, fname)
        ref_path = os.path.join(ref_dir, fname)
        if not os.path.exists(ref_path):
            continue
        gen = lpips.im2tensor(lpips.load_image(gen_path))
        ref = lpips.im2tensor(lpips.load_image(ref_path))
        with torch.no_grad():
            d = loss_fn(gen, ref)
        values.append(float(d.item()))
    return float(np.mean(values)) if values else 0.0


def compute_fid(gen_dir: str, ref_dir: str) -> float:
    from cleanfid import fid

    score = fid.compute_fid(gen_dir, ref_dir, mode="clean")
    return float(score)


if __name__ == "__main__":
    gen_dir = "outputs/group_E"
    ref_dir = "dataset/test"
    print("LPIPS:", compute_lpips(gen_dir, ref_dir))
    print("FID:", compute_fid(gen_dir, ref_dir))
