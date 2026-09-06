"""red_ratio_stats.py — calibrate the app's not-food gate (RED_RATIO_MIN).

Computes the red-dominant pixel ratio over the whole test split using EXACTLY
the app's pipeline (resize short side to 224, center-crop, threshold
r>90 & r>1.35g & r>1.1b — see app/src/ml/preprocess.ts).

The gate must sit BELOW the in-distribution floor: if e.g. the weakest chili
image has ratio 0.18, a threshold of 0.12 keeps every real sample analyzable
while still rejecting hands/surfaces (which measure near 0).

Run:  .venv/Scripts/python.exe red_ratio_stats.py
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from data_prep_redchili import build_file_lists

SIZE = 224


def red_ratio(path: str) -> float:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = SIZE / min(w, h)
    rw = max(SIZE, round(w * scale))
    rh = max(SIZE, round(h * scale))
    img = img.resize((rw, rh))
    left = (rw - SIZE) // 2
    top = (rh - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    a = np.asarray(img, dtype=np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 90) & (r > g * 1.35) & (r > b * 1.1)
    return float(mask.mean())


def main() -> None:
    splits, _ = build_file_lists()
    paths, labels = splits["test"]
    ratios = np.array([red_ratio(p) for p in paths])
    print(f"n={len(ratios)} images (test split)")
    print(
        "min={:.4f}  p0.5={:.4f}  p1={:.4f}  p5={:.4f}  median={:.4f}".format(
            ratios.min(),
            np.percentile(ratios, 0.5),
            np.percentile(ratios, 1),
            np.percentile(ratios, 5),
            np.median(ratios),
        )
    )
    for t in (0.08, 0.10, 0.12, 0.15, 0.20):
        print(f"fraction below {t:.2f}: {float((ratios < t).mean()):.4f}")


if __name__ == "__main__":
    main()
