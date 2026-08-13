"""Generate a multi-font crop dataset: render captchas with several Persian fonts,
segment them (ground truth known), save 28x28 crops per class.

Usage:
    python src/build_multifont_dataset.py [--n 300] [--out data/captcha_crops_multifont] [--fonts ...]
"""
import argparse
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.abspath('..')))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.preprocessing import preprocess_before_seg  # noqa: E402

DIGITS = "۰۱۲۳۴۵۶۷۸۹"
WIDTH, HEIGHT = 200, 80
FONT_SIZE = 42

# Fonts that render all digits fully (from check_fonts.py: OK list)
GOOD_FONTS = [
    "BYekan.ttf",
    "BTitrBd.ttf",
    "BHoma.ttf",
    "BHelal.ttf",
    "BBardiya.ttf",
    "BNikoo.ttf",
    "BSahar.ttf",
    "BShiraz.ttf",
    "BTabssom.ttf",
    "BLotusBd.ttf",
]


def make_captcha(text, font_path, seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    for _ in range(8):
        draw.line(
            (rng.randint(0, WIDTH), rng.randint(0, HEIGHT),
             rng.randint(0, WIDTH), rng.randint(0, HEIGHT)),
            fill=(rng.randint(50, 180), rng.randint(50, 180), rng.randint(50, 180)),
            width=2,
        )
    for _ in range(80):
        x, y = rng.randint(0, WIDTH), rng.randint(0, HEIGHT)
        draw.ellipse((x, y, x + 2, y + 2),
                     fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    font = ImageFont.truetype(str(font_path), FONT_SIZE)
    x = 20
    for digit in text:
        y = rng.randint(15, 25)
        draw.text((x, y), digit, font=font, fill="black")
        x += 32
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Captchas per font")
    parser.add_argument("--out", default="data/captcha_crops_multifont")
    parser.add_argument("--fonts", nargs="+", default=None)
    args = parser.parse_args()

    fonts = args.fonts or GOOD_FONTS
    out_dir = BASE / args.out
    for i in range(10):
        (out_dir / str(i)).mkdir(parents=True, exist_ok=True)

    total_saved = 0
    seg_failures = 0
    rng = random.Random(42)

    for fi, font_name in enumerate(fonts):
        font_path = BASE / "fonts" / "persian" / font_name
        font_crops = 0
        for i in range(args.n):
            text = "".join(rng.choice(DIGITS) for _ in range(5))
            seed = 10000 * fi + i
            img = make_captcha(text, font_path, seed)
            tmp = BASE / "data" / "_mf_tmp.png"
            img.save(tmp)
            try:
                thresh = preprocess_before_seg(str(tmp))
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
                boxes = [b for b in boxes if b[2] >= 5 and b[3] >= 5]
                if len(boxes) != 5:
                    seg_failures += 1
                    continue
                for pos, (x, y, w, h) in enumerate(boxes):
                    crop = thresh[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4]
                    if crop.size == 0:
                        continue
                    c28 = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
                    digit = text[pos]
                    cls = DIGITS.index(digit)
                    fname = f"{font_name.replace('.ttf', '')}_{i:04d}_{pos}.png"
                    cv2.imwrite(str(out_dir / str(cls) / fname), c28)
                    total_saved += 1
                    font_crops += 1
            finally:
                tmp.unlink(missing_ok=True)
        print(f"  {font_name}: {font_crops} crops")

    print(f"\nTotal crops saved: {total_saved} -> {out_dir}")
    print(f"Segmentation failures: {seg_failures}")
    for i in range(10):
        n = len(list((out_dir / str(i)).glob("*.png")))
        print(f"  class {i}: {n}")


if __name__ == "__main__":
    main()
