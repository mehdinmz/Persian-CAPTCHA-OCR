"""Evaluate the handwritten-digit model on a real dataset.

Usage:
    python src/evaluate.py [--n 50]

Uses the 600-image sample (60 per class) shipped on HuggingFace
(Mehdinmz/persian-handwritten-digits → farsi_digits_sample.zip).
The sample is downloaded once and cached under data/dataset_farsi_sample/.
"""
import argparse
import os
import random
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2
import numpy as np

from src.predictor import model  # handwritten-digit model

DIGITS = "۰۱۲۳۴۵۶۷۸۹"
HF_REPO = "Mehdinmz/persian-handwritten-digits"
ZIP_FILE = "farsi_digits_sample.zip"
SAMPLE_DIR = BASE / "data" / "dataset_farsi_sample"


def _get_sample_dir() -> Path:
    """Return cached sample dir; download+extract once if missing."""
    if (SAMPLE_DIR / "0").exists() and any((SAMPLE_DIR / "0").iterdir()):
        return SAMPLE_DIR
    print("Sample dataset not found — downloading from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")
    zip_path = hf_hub_download(HF_REPO, ZIP_FILE, repo_type="dataset")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(SAMPLE_DIR)
    return SAMPLE_DIR


def evaluate(data_dir: Path, n_per_class: int) -> None:
    random.seed(42)
    total, correct = 0, 0
    per_class = {}
    for d in range(10):
        files = os.listdir(data_dir / str(d))
        random.shuffle(files)
        files = files[:n_per_class]
        ok = 0
        for f in files:
            img = cv2.imread(str(data_dir / str(d) / f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            c28 = cv2.resize(img, (28, 28)).astype(np.float32) / 255.0
            p = model.predict(c28.reshape(1, 28, 28, 1), verbose=0)[0]
            pred = int(np.argmax(p))
            total += 1
            if pred == d:
                ok += 1
                correct += 1
        per_class[DIGITS[d]] = ok / max(len(files), 1)
    print(f"\nOverall accuracy: {correct}/{total} = {correct / max(total, 1):.1%}")
    print("Per-class accuracy:")
    for d in range(10):
        print(f"  {DIGITS[d]}: {per_class[DIGITS[d]]:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the handwritten-digit model")
    parser.add_argument("--n", type=int, default=50, help="images per class (max 60)")
    args = parser.parse_args()
    data_dir = _get_sample_dir()
    evaluate(data_dir, min(args.n, 60))


if __name__ == "__main__":
    main()
