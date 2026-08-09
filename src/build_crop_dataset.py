"""Build a training dataset from real BYekan captcha crops (segmented by position)."""
import csv
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.preprocessing import preprocess_before_seg

# Use the 200 test captchas + generate more
SRC_CSV = BASE / "data" / "captcha_train_yekan" / "labels.csv"
SRC_IMAGES = BASE / "data" / "captcha_train_yekan" / "images"
OUT_DIR = BASE / "data" / "captcha_crops_byekan"

out = Path(OUT_DIR)
for i in range(10):
    (out / str(i)).mkdir(parents=True, exist_ok=True)

DIGITS = "۰۱۲۳۴۵۶۷۸۹"

rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8-sig")))
count = 0
failed = 0
for row in rows:
    filename = row["filename"]
    true_label = row["label"].strip()
    try:
        thresh = preprocess_before_seg(SRC_IMAGES / filename)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
        boxes = [b for b in boxes if b[2] >= 5 and b[3] >= 5]
        if len(boxes) != 5:
            failed += 1
            continue
        # crop with padding, resize to 28x28 (matching training style)
        for pos, (x, y, w, h) in enumerate(boxes):
            crop = thresh[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4]
            if crop.size == 0:
                continue
            c28 = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
            digit = true_label[pos]
            idx = DIGITS.index(digit)
            cv2.imwrite(str(out / str(idx) / f"{filename[:-4]}_{pos}.png"), c28)
            count += 1
    except Exception:
        failed += 1

print(f"Saved {count} crops to {OUT_DIR}")
print(f"Failed (segmentation issues): {failed}")
# Count per class
for i in range(10):
    n = len(list((out / str(i)).glob("*.png")))
    print(f"  class {i}: {n}")
