"""Evaluate the NEW fine-tuned model on test captchas (BYekan, unseen)."""
import csv
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import tensorflow as tf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.preprocessing import preprocess_before_seg

# Load the new model
model = tf.keras.models.load_model(str(BASE / "models" / "digit_classifier_captcha_v2.keras"))
DIGITS = "۰۱۲۳۴۵۶۷۸۹"

# Test set: the ORIGINAL 200 BYekan captchas (not used in training)
LABELS = BASE / "data" / "captcha_test_yekan" / "labels.csv"
IMAGES = BASE / "data" / "captcha_test_yekan" / "images"


def predict_captcha(image_path):
    thresh = preprocess_before_seg(image_path)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] >= 5 and b[3] >= 5]
    if len(boxes) > 5:
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:5]
    boxes = sorted(boxes, key=lambda b: b[0])
    preds = []
    for (x, y, w, h) in boxes:
        crop = thresh[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
        if crop.size == 0:
            continue
        c28 = cv2.resize(crop, (28, 28)).astype(np.float32) / 255.0
        p = model.predict(c28.reshape(1, 28, 28, 1), verbose=0)[0]
        preds.append(int(np.argmax(p)))
    return "".join(DIGITS[i] for i in preds)


rows = list(csv.DictReader(open(LABELS, encoding="utf-8-sig")))
digit_cm = Counter()
digit_total = Counter()
captcha_correct = 0
length_mismatch = 0

print(f"Evaluating {len(rows)} unseen BYekan captchas with NEW model...")
for row in rows:
    filename = row["filename"]
    true_label = row["label"].strip()
    try:
        pred = predict_captcha(IMAGES / filename)
    except Exception:
        continue
    if len(pred) != len(true_label):
        length_mismatch += 1
        continue
    if pred == true_label:
        captcha_correct += 1
    for t, pr in zip(true_label, pred):
        ti = DIGITS.index(t)
        pi = DIGITS.index(pr)
        digit_total[ti] += 1
        digit_cm[(ti, pi)] += 1

total = sum(digit_total.values())
correct = sum(digit_cm[(i, i)] for i in range(10))
print(f"\nCAPTCHA accuracy: {captcha_correct}/{len(rows)} = {captcha_correct/len(rows):.1%}")
print(f"Digit accuracy  : {correct}/{total} = {correct/total:.1%}")
print(f"Length mismatches: {length_mismatch}")
print("\nPer-digit accuracy:")
for i in range(10):
    cnt = digit_total[i]
    acc = digit_cm[(i, i)] / cnt if cnt else 0
    print(f"  {DIGITS[i]}: {acc:.1%}  ({cnt} samples)")
print("\nTop error pairs:")
pairs = [(c, i, j) for (i, j), c in digit_cm.items() if i != j]
pairs.sort(reverse=True)
for c, i, j in pairs[:10]:
    print(f"  {DIGITS[i]} -> {DIGITS[j]}: {c}")
