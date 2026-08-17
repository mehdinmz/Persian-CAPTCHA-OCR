# Persian-Digit-OCR

A robust, end-to-end OCR pipeline for Persian handwritten digits and CAPTCHAs.

## Features
- 🧠 **Handwritten digit recognition** — CNN with **99.8% accuracy** on 600 real handwritten digits
- 🤖 **CAPTCHA pipeline** — image preprocessing → segmentation → multi-font digit recognition (100% on test set)
- 🔤 **Multi-font support** — 80+ Persian fonts for synthetic CAPTCHA generation
- 📦 **HuggingFace integration** — model & sample dataset published on [Mehdinmz/persian-handwritten-digits](https://huggingface.co/datasets/Mehdinmz/persian-handwritten-digits)
- 🧪 **CI & tests** — pytest suite + GitHub Actions

## Prerequisites
- Python 3.10+
- `pip` and `venv` (recommended)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mehdinmz/Persian-Digit-OCR.git
   cd Persian-Digit-OCR
   ```

2. **Setup virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > It also installs `jupyter` for running the notebooks and `matplotlib` / `seaborn` for visualizations.

## Usage

### 1. Prediction (CLI)
Recognize a handwritten digit or a CAPTCHA image:

```bash
python src/predictor.py --image path/to/image.png
```

You can also pass the path as a positional argument:

```bash
python src/predictor.py path/to/image.png
```

Add `--conf` to show per-digit confidence:

```bash
python src/predictor.py path/to/image.png --conf
```

**Example output:**
```
۴
  ۴: 94.5%
```

### 2. As a Python API

```python
from src.predictor import predict_digit

digit, confidence = predict_digit("path/to/digit.png")
print(digit, confidence)
```

### 3. Evaluate the model
Evaluate the handwritten-digit model on the real sample dataset (auto-downloads from HuggingFace):

```bash
python scripts/evaluate.py --n 60
```

**Example output:**
```
Overall accuracy: 599/600 = 99.8%
```

### 4. Train the model
Train a fresh CNN on the handwritten dataset:

```bash
python src/train_handwritten.py --epochs 25
```

### 5. Run the notebooks
The notebooks document the full pipeline (exploration → training → fine-tuning → evaluation):

```bash
jupyter notebook notebooks/06_evaluation.ipynb
```

> **Tip:** run Jupyter via the venv executable if `jupyter` is not on your PATH:
> ```bash
> .venv/bin/jupyter-notebook --no-browser --port=8888
> ```

## Project Structure
- `src/`: Core logic (predictor, pipeline, preprocessing, segmentation, model).
- `scripts/`: Helper scripts (evaluation, dataset building, augmentation, fonts check).
- `notebooks/`: Analytical & training notebooks.
- `data/`: Dataset storage (git-ignored).
- `models/`: Trained model checkpoints.
- `tests/`: Unit tests (pytest).

## Tests
```bash
python -m pytest tests/ -v
```

## License
[MIT](LICENSE) © Mehdi Namazian