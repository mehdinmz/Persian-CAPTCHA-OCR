"""Augment the crop dataset: add rotations/shifts for class 2 specifically,
and mild augmentation for all classes to improve generalization."""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "captcha_crops_byekan"
OUT_DIR = BASE / "data" / "captcha_crops_byekan_aug"

out = Path(OUT_DIR)
for i in range(10):
    (out / str(i)).mkdir(parents=True, exist_ok=True)

# Copy existing + add augmented versions
rng = np.random.default_rng(42)
total = 0
for cls in range(10):
    src = DATA_DIR / str(cls)
    files = sorted(src.glob("*.png"))
    # copy originals
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(out / str(cls) / f.name), img)
        total += 1
        # augment: 4 extra versions for class 2 (the problem class), 2 for others
        n_aug = 4 if cls == 2 else 2
        for k in range(n_aug):
            # random small rotation + shift
            angle = rng.uniform(-8, 8)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            aug = cv2.warpAffine(img, M, (28, 28), borderValue=0)
            dx, dy = rng.integers(-2, 3), rng.integers(-2, 3)
            M2 = np.float32([[1, 0, dx], [0, 1, dy]])
            aug = cv2.warpAffine(aug, M2, (28, 28), borderValue=0)
            cv2.imwrite(str(out / str(cls) / f"{f.stem}_a{k}.png"), aug)
            total += 1

print(f"Total crops (original + augmented): {total}")
for i in range(10):
    n = len(list((out / str(i)).glob("*.png")))
    print(f"  class {i}: {n}")
