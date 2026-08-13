"""Prepare dataset_farsi as a HuggingFace ImageFolder dataset with README."""
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data" / "dataset_farsi"
OUT = BASE / "data" / "hf_farsi_digits"

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# Copy each class folder (ImageFolder layout: class_dir/images)
for i in range(10):
    src_dir = SRC / str(i)
    dst_dir = OUT / str(i)
    dst_dir.mkdir(parents=True)
    for f in sorted(src_dir.glob("*.png"))[:8000]:
        shutil.copy2(f, dst_dir / f.name)

# Dataset card (README.md for the HF dataset)
readme = """---
license: mit
language:
  - fa
tags:
  - persian
  - digits
  - handwritten
  - ocr
pretty_name: Persian Handwritten Digits (Farsi)
---

# Persian Handwritten Digits (Farsi)

80,000 grayscale images of **handwritten Persian (Farsi) digits** — `۰۱۲۳۴۵۶۷۸۹` —
organized as an ImageFolder dataset with 10 classes (0–9), 8,000 images per class.

Each image is a **28×28** grayscale PNG of a single digit.

## Classes

| Class | Count |
|-------|-------|
| 0 (۰) | 8,000 |
| 1 (۱) | 8,000 |
| 2 (۲) | 8,000 |
| 3 (۳) | 8,000 |
| 4 (۴) | 8,000 |
| 5 (۵) | 8,000 |
| 6 (۶) | 8,000 |
| 7 (۷) | 8,000 |
| 8 (۸) | 8,000 |
| 9 (۹) | 8,000 |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("Mehdinmz/persian-handwritten-digits", split="train")
```

## License

MIT
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

# Count
total = sum(len(list((OUT / str(i)).glob("*.png"))) for i in range(10))
print(f"Prepared {total} images -> {OUT}")
for i in range(10):
    print(f"  class {i}: {len(list((OUT / str(i)).glob('*.png')))}")
