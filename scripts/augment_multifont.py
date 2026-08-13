"""Augment the multi-font dataset with strong augmentation (rotation, scale, shift, noise).

Usage:
    python src/augment_multifont.py [--src data/captcha_crops_multifont] [--out data/captcha_crops_multifont_aug] [--per 3]
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.abspath('..')))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def augment_image(img, rng):
    """Apply random rotation, shift, scale, and noise to a 28x28 digit crop."""
    h, w = img.shape
    # random rotation
    angle = rng.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(img, M, (w, h), borderValue=0)
    # random shift
    dx, dy = rng.integers(-3, 4), rng.integers(-3, 4)
    M2 = np.float32([[1, 0, dx], [0, 1, dy]])
    aug = cv2.warpAffine(aug, M2, (w, h), borderValue=0)
    # random scale
    scale = rng.uniform(0.85, 1.15)
    M3 = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    aug = cv2.warpAffine(aug, M3, (w, h), borderValue=0)
    # slight erosion/dilation for stroke width variation
    if rng.random() < 0.3:
        k = np.ones((2, 2), np.uint8)
        aug = cv2.dilate(aug, k) if rng.random() < 0.5 else cv2.erode(aug, k)
    # noise
    noise = rng.normal(0, 15, aug.shape).astype(np.float32)
    aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return aug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(BASE / "data" / "captcha_crops_multifont"))
    parser.add_argument("--out", default=str(BASE / "data" / "captcha_crops_multifont_aug"))
    parser.add_argument("--per", type=int, default=3, help="Augmented copies per original")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    for i in range(10):
        (out / str(i)).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    total = 0
    for cls in range(10):
        files = sorted((src / str(cls)).glob("*.png"))
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            # copy original
            cv2.imwrite(str(out / str(cls) / f.name), img)
            total += 1
            # augmented copies
            for k in range(args.per):
                aug = augment_image(img, rng)
                cv2.imwrite(str(out / str(cls) / f"{f.stem}_a{k}.png"), aug)
                total += 1
    print(f"Total crops (orig + {args.per}x aug): {total} -> {out}")
    for i in range(10):
        n = len(list((out / str(i)).glob("*.png")))
        print(f"  class {i}: {n}")


if __name__ == "__main__":
    main()
