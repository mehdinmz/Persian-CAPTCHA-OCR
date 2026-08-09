"""End-to-end tests for the Persian handwritten digit OCR pipeline.

Run:  python -m pytest tests/ -v
(or:   python tests/test_pipeline.py)

The test data (600 images: 60 per class) is downloaded once from HuggingFace
as a single zip file and cached under data/dataset_farsi_sample/.
"""
import os
import random
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.pipeline import model  # noqa: E402

DIGITS = "۰۱۲۳۴۵۶۷۸۹"
SAMPLE_DIR = BASE / "data" / "dataset_farsi_sample"
ZIP_URL_REPO = "Mehdinmz/persian-handwritten-digits"
ZIP_FILE = "farsi_digits_sample.zip"


def _get_sample_dir() -> Path:
    """Return cached sample dir; download+extract once if missing."""
    if (SAMPLE_DIR / "0").exists() and any((SAMPLE_DIR / "0").iterdir()):
        return SAMPLE_DIR
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "sample dataset missing and `huggingface_hub` is not installed. "
            "Run: pip install huggingface_hub"
        )
    zip_path = hf_hub_download(ZIP_URL_REPO, ZIP_FILE, repo_type="dataset")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(SAMPLE_DIR)
    return SAMPLE_DIR


def _predict_digit_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"cannot read {path}"
    c28 = cv2.resize(img, (28, 28)).astype(np.float32) / 255.0
    p = model.predict(c28.reshape(1, 28, 28, 1), verbose=0)[0]
    return int(np.argmax(p))


def test_single_digit_each_class():
    """Predict one image from each of the 10 classes."""
    data_dir = _get_sample_dir()
    for d in range(10):
        files = sorted((data_dir / str(d)).glob("*.png"))
        assert files, f"no files for class {d}"
        pred = _predict_digit_image(files[0])
        assert pred == d, f"class {d}: predicted {pred}"


def test_random_sample_accuracy():
    """Sample 50 images per class; require >= 97% overall accuracy."""
    data_dir = _get_sample_dir()
    random.seed(42)
    total, correct = 0, 0
    for d in range(10):
        files = os.listdir(data_dir / str(d))
        random.shuffle(files)
        for f in files[:50]:
            pred = _predict_digit_image(data_dir / str(d) / f)
            total += 1
            if pred == d:
                correct += 1
    acc = correct / total
    print(f"\nHandwritten digit accuracy: {correct}/{total} = {acc:.1%}")
    assert acc >= 0.97, f"accuracy too low: {acc:.1%}"


if __name__ == "__main__":
    test_single_digit_each_class()
    print("test_single_digit_each_class: OK")
    test_random_sample_accuracy()
    print("test_random_sample_accuracy: OK")
