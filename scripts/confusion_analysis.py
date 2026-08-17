"""
Fast confusion-matrix analysis for Persian-Digit-OCR.

Same analysis as confusion_analysis.py, but performs BATCH prediction
(one model.predict() call per batch of digit crops) so it runs much
faster on CPU.
"""
import csv
import random
import sys
from pathlib import Path
from collections import Counter

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.preprocessing import preprocess_before_seg  # noqa: E402
from src.segmentation import find_digits, crop_digits  # noqa: E402
from src.pipeline import model, DIGITS  # noqa: E402

LABELS = BASE / "data" / "captcha_generated" / "labels.csv"
IMAGES = BASE / "data" / "captcha_generated" / "images"

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
BATCH = 128


def preprocess_crop(crop):
    """Replicate preprocess_digit from preprocessing.py (returns 28x28 normalized)."""
    import cv2
    if crop is None or crop.size == 0:
        return None
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (28, 28))
    crop = crop.astype(np.float32) / 255.0
    return np.expand_dims(crop, axis=-1)


def main():
    random.seed(42)
    rows = list(csv.DictReader(open(LABELS, encoding="utf-8-sig")))
    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

    digit_cm = Counter()
    digit_total = Counter()
    captcha_correct = 0
    errors = []

    # Phase 1: segment all images, collect crops
    print(f"Segmenting {len(sample)} CAPTCHAs...")
    jobs = []  # (index, true_label, filename)
    for i, row in enumerate(sample):
        filename = row["filename"]
        true_label = row["label"].strip()
        try:
            thresh = preprocess_before_seg(IMAGES / filename)
            boxes = find_digits(thresh)
            if len(boxes) == 0:
                errors.append((filename, true_label, "<no digits>", -1))
                continue
            boxes = sorted(boxes, key=lambda b: b[0])
            crops = crop_digits(thresh, boxes)
            jobs.append((i, true_label, filename, crops))
        except Exception as e:
            errors.append((filename, true_label, f"<seg error: {e}>", -1))

    # Phase 2: batch predict all crops
    print(f"Predicting {sum(len(j[3]) for j in jobs)} digit crops in batches of {BATCH}...")
    all_inputs = []
    for _, _, _, crops in jobs:
        for c in crops:
            p = preprocess_crop(c)
            if p is not None:
                all_inputs.append(p)

    predictions = []
    for start in range(0, len(all_inputs), BATCH):
        batch = np.stack(all_inputs[start:start + BATCH])
        probs = model.predict(batch, verbose=0)
        predictions.extend(int(np.argmax(p)) for p in probs)

    # Phase 3: reassemble predictions
    pos = 0
    for i, true_label, filename, crops in jobs:
        n = len(crops)
        preds = predictions[pos:pos + n]
        pos += n
        predicted_label = "".join(DIGITS[p] for p in preds)
        if predicted_label == true_label:
            captcha_correct += 1
        else:
            wrong_pos = [p for p in range(len(true_label)) if p >= len(predicted_label) or true_label[p] != predicted_label[p]]
            errors.append((filename, true_label, predicted_label, wrong_pos))
        for p, t in enumerate(true_label):
            ti = DIGITS.index(t) if t in DIGITS else -1
            pi = preds[p] if p < len(preds) else -1
            digit_total[ti] += 1
            digit_cm[(ti, pi)] += 1

    total_digits = sum(digit_total.values())
    correct_digits = sum(digit_cm[(i, i)] for i in range(10))
    digit_acc = correct_digits / total_digits if total_digits else 0
    captcha_acc = captcha_correct / len(sample)

    print()
    print("=" * 52)
    print("SUMMARY")
    print("=" * 52)
    print(f"CAPTCHA accuracy : {captcha_acc:.1%}  ({captcha_correct}/{len(sample)})")
    print(f"Digit accuracy   : {digit_acc:.1%}  ({correct_digits}/{total_digits})")
    print(f"Segmentation failures: {sum(1 for e in errors if e[2].startswith('<'))}")
    print()

    print("=" * 52)
    print("PER-DIGIT ACCURACY (Persian digits ۰-۹)")
    print("=" * 52)
    print("class |  count | accuracy | top confusions (true->predicted)")
    print("-" * 52)
    for i in range(10):
        cnt = digit_total[i]
        acc = digit_cm[(i, i)] / cnt if cnt else 0
        confs = [(digit_cm[(i, j)], j) for j in range(10) if j != i and digit_cm[(i, j)] > 0]
        confs.sort(reverse=True)
        top = ", ".join(f"{DIGITS[j]}→{DIGITS[j]}:{c}" for c, j in confs[:3])
        print(f"  {DIGITS[i]}   | {cnt:5d} | {acc:8.1%} | {top or '—'}")

    print()
    print("=" * 52)
    print("TOP-15 ERROR PAIRS (true→predicted)")
    print("=" * 52)
    pairs = [(c, i, j) for (i, j), c in digit_cm.items() if i != j]
    pairs.sort(reverse=True)
    for c, i, j in pairs[:15]:
        pct = c / digit_total[i] if digit_total[i] else 0
        print(f"  {DIGITS[i]} → {DIGITS[j]} : {c:4d}  ({pct:.1%} of true {DIGITS[i]})")

    print()
    print("=" * 52)
    print("ERRORS BY POSITION (0=leftmost)")
    print("=" * 52)
    pos_counts = Counter()
    pos_total = Counter()
    for _, true_label, predicted_label, wrong_pos in errors:
        if wrong_pos == -1:
            continue
        for p in wrong_pos:
            pos_counts[p] += 1
        for p in range(len(true_label)):
            pos_total[p] += 1
    for p in range(5):
        n = pos_total[p]
        print(f"  position {p}: {pos_counts[p]:4d} errors / {n} digits ({pos_counts[p]/n:.1%}" if n else f"  position {p}: 0/0")

    print()
    print("=" * 52)
    print("SAMPLE FAILURES (first 15)")
    print("=" * 52)
    shown = 0
    for filename, true_label, predicted_label, wrong_pos in errors:
        if wrong_pos == -1:
            continue
        print(f"  {filename}: true={true_label}  pred={predicted_label}  wrong@ {wrong_pos}")
        shown += 1
        if shown >= 15:
            break

    out = BASE / "confusion_analysis.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Sample: {len(sample)} CAPTCHAs\n")
        f.write(f"CAPTCHA acc: {captcha_acc:.1%} | Digit acc: {digit_acc:.1%}\n\n")
        f.write("Per-digit confusion (row=true, col=predicted):\n")
        f.write("      " + "".join(f"{DIGITS[j]:>4}" for j in range(10)) + "\n")
        for i in range(10):
            f.write(f"  {DIGITS[i]}  " + "".join(f"{digit_cm[(i,j)]:4d}" for j in range(10)) + "\n")
        f.write("\nAll failures:\n")
        for filename, true_label, predicted_label, wrong_pos in errors:
            f.write(f"  {filename} true={true_label} pred={predicted_label} wrong@{wrong_pos}\n")
    print(f"\nFull details saved to {out}")


if __name__ == "__main__":
    main()
