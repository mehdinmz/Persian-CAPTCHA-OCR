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

from src.predictor import model  # noqa: E402  (handwritten-digit model)

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


def test_captcha_multifont():
    """Full CAPTCHA solving with the multi-font model must be correct."""
    from PIL import Image, ImageDraw, ImageFont
    from src.pipeline import predict_captcha

    font_path = BASE / "fonts" / "persian" / "BYekan.ttf"
    if not font_path.exists():
        return  # fonts not shipped in this checkout; skip

    img = Image.new("RGB", (200, 80), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path), 42)
    x = 20
    text = "۵۲۶۰۱"
    for ch in text:
        d.text((x, 18), ch, font=font, fill="black")
        x += 32
    tmp = BASE / "data" / "_captcha_test.png"
    img.save(tmp)
    try:
        pred = predict_captcha(tmp)
        assert pred == text, f"captcha: expected {text}, got {pred}"
        print(f"\nCAPTCHA (multi-font model): {pred} -> OK")
    finally:
        tmp.unlink(missing_ok=True)


def test_predict_digit_accepts_path():
    """predict_digit must accept both a file path and a numpy array."""
    from src.predictor import predict_digit

    data_dir = _get_sample_dir()
    sample = sorted((data_dir / "5").glob("*.png"))[0]

    # path (str/Path)
    d1, _ = predict_digit(str(sample))
    assert d1 == "۵", f"path: expected ۵, got {d1}"

    # numpy array
    img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    d2, _ = predict_digit(img)
    assert d2 == "۵", f"array: expected ۵, got {d2}"

    print("\npredict_digit accepts str path + numpy array: OK")


def test_find_digits_filters_noise_and_texture():
    """find_digits must keep clean digits while rejecting noise and
    textured (busy) regions."""
    import numpy as np
    from src.segmentation import find_digits

    # A clean synthetic digit-like blob: 40x50 solid rectangle
    clean = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(clean, (10, 25), (50, 75), 255, -1)  # 40x50 solid

    # A textured region: 40x50 with many small specks
    textured = np.zeros((100, 100), dtype=np.uint8)
    rng = np.random.default_rng(42)
    speck = rng.integers(0, 2, size=(40, 50), dtype=np.uint8) * 255
    textured[10:50, 10:60] = speck

    boxes_clean = find_digits(clean)
    boxes_textured = find_digits(textured)

    # Clean solid shape should be kept (it is a plausible digit)
    assert len(boxes_clean) >= 1, "clean digit-like region was dropped"

    # Textured busy region should be rejected
    assert len(boxes_textured) == 0, "textured region should be rejected"


if __name__ == "__main__":
    test_single_digit_each_class()
    print("test_single_digit_each_class: OK")
    test_random_sample_accuracy()
    print("test_random_sample_accuracy: OK")
    test_captcha_multifont()
    print("test_captcha_multifont: OK")
    test_predict_digit_accepts_path()
    print("test_predict_digit_accepts_path: OK")
