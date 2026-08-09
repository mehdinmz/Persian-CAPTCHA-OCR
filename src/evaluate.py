"""Evaluate the handwritten-digit model on a real dataset.

Usage:
    python src/evaluate.py [--n 500] [--data data/dataset_farsi]

Downloads the dataset from HuggingFace if not present locally, then reports
per-class and overall accuracy.
"""
import argparse
import os
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2
import numpy as np

from src.predictor import model  # handwritten-digit model

DIGITS = "۰۱۲۳۴۵۶۷۸۹"
HF_REPO = "Mehdinmz/persian-handwritten-digits"


def _get_data_dir(data_arg: str) -> Path:
    data_dir = Path(data_arg)
    if (data_dir / "0").exists() and any((data_dir / "0").iterdir()):
        return data_dir
    print("Local dataset not found — downloading from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(data_dir))
    return data_dir


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
    parser.add_argument("--data", default=str(BASE / "data" / "dataset_farsi"))
    parser.add_argument("--n", type=int, default=500, help="images per class")
    args = parser.parse_args()
    data_dir = _get_data_dir(args.data)
    evaluate(data_dir, args.n)


if __name__ == "__main__":
    main()
