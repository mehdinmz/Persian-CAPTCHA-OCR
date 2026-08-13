"""Create a LinkedIn-ready image: real Persian handwritten digits with model predictions.

Grid: 10 rows (classes 0-9) x 6 cols (samples), each cell shows the digit,
the true label, the prediction, and a green/red check.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.abspath('..')))

import cv2
import numpy as np

from src.pipeline import model

SAMPLE_DIR = BASE / "data" / "dataset_farsi_sample"
OUT = BASE / "linkedin_demo.png"

DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ROWS = 10
COLS = 6
CELL = 140  # px per cell
PAD = 8

total = correct = 0
cell_imgs = []
for cls in range(10):
    files = sorted((SAMPLE_DIR / str(cls)).glob("*.png"))[:COLS]
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        # upscale to CELL with white background
        cell = np.full((CELL, CELL), 255, dtype=np.uint8)
        h, w = img.shape
        scale = (CELL - 2 * PAD) / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        y0 = (CELL - nh) // 2
        x0 = (CELL - nw) // 2
        cell[y0 : y0 + nh, x0 : x0 + nw] = resized
        # predict (model expects 28x28)
        c28 = cv2.resize(img, (28, 28)).astype(np.float32) / 255.0
        p = model.predict(c28.reshape(1, 28, 28, 1), verbose=0)[0]
        pred = int(np.argmax(p))
        total += 1
        ok = pred == cls
        correct += ok
        # green or red border + label strip
        border_color = (60, 179, 113) if ok else (220, 20, 60)
        cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), border_color, 4)
        label = f"{DIGITS[cls]} -> {DIGITS[pred]}"
        cv2.putText(
            cell, label, (8, CELL - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            border_color, 2, cv2.LINE_AA,
        )
        cell_imgs.append(cell)

grid = np.vstack(
    [np.hstack(cell_imgs[r * COLS : (r + 1) * COLS]) for r in range(ROWS)]
)
# header bar
hdr = np.full((60, grid.shape[1], 3), 245, dtype=np.uint8)
cv2.putText(
    hdr, f"Persian Handwritten Digit Recognition - {correct}/{total} correct ({correct/total:.1%})",
    (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA,
)
out = np.vstack([hdr, grid])
cv2.imwrite(str(OUT), out)
print(f"Saved {OUT} | {correct}/{total} = {correct/total:.1%}")
