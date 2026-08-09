from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Project root: src/ is one level below the repo root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "digit_classifier_handwritten.keras"

IMAGE_SIZE = (28, 28)


# Persian digits
DIGITS = "۰۱۲۳۴۵۶۷۸۹"


# =====================================
# Load Model
# =====================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =====================================
# Prepare Image
# =====================================

def prepare_image(image):
    """
    Prepare one segmented digit for the CNN.

    Input:
        image -> grayscale numpy array

    Output:
        shape -> (1, 28, 28, 1)
        values -> normalized to [0, 1]
    """

    # Make sure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    image = image.astype(
        np.float32
    ) / 255.0

    # Add channel dimension
    image = np.expand_dims(
        image,
        axis=-1
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# =====================================
# Predict
# =====================================

def predict_digit(image):
    """
    Predict one Persian digit.

    Accepts either a file path (str/Path) or a numpy array (grayscale/BGR).

    Returns:
        digit      -> Persian digit
        confidence -> prediction confidence
    """

    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        image = img

    image = prepare_image(image)

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_index]
    )

    digit = DIGITS[class_index]

    return digit, confidence


# =====================================
# Predict From Image Path
# =====================================

def predict_from_path(image_path):
    """
    Load a digit image and predict it.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    return predict_digit(image)


# =====================================
# CLI
# =====================================

def main():
    """CLI: captcha-ocr <image_path> — print the predicted CAPTCHA text."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Recognize Persian digit CAPTCHAs")
    parser.add_argument("image", help="Path to CAPTCHA image")
    parser.add_argument("--conf", action="store_true", help="Show per-digit confidence")
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
        from src.pipeline import predict_captcha, predict_digit
        from src.preprocessing import preprocess_before_seg
        from src.segmentation import find_digits, crop_digits

        # Segment the image
        thresh = preprocess_before_seg(args.image)
        boxes = sorted(find_digits(thresh), key=lambda b: b[0])
        crops = crop_digits(thresh, boxes)

        if not crops:
            raise ValueError("No digits were detected.")

        # Decide: single digit vs full CAPTCHA.
        # A CAPTCHA has several separated digit blobs; a single digit may
        # fragment into several small contours, so we merge them and check
        # the overall aspect ratio.
        if len(crops) > 1:
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[0] + b[2] for b in boxes)
            y1 = max(b[1] + b[3] for b in boxes)
            merged = thresh[y0:y1, x0:x1]
            h, w = merged.shape[:2]
            # digits are roughly square-ish (h/w between 0.5 and 2.5);
            # a full CAPTCHA is much wider than tall
            is_single = (h / max(w, 1)) > 0.4
        else:
            merged = crops[0]
            is_single = True

        if is_single:
            # Single digit image → handwritten model
            from src.predictor import predict_digit as predict_digit_hw
            digit, conf = predict_digit_hw(merged)
            print(digit)
            if args.conf:
                print(f"  {digit}: {conf:.1%}")
            return

        # Full CAPTCHA (multiple digits) → pipeline (multi-font model)
        text = predict_captcha(args.image)
        print(text)
        if args.conf:
            for d, crop in zip(text, crops):
                if crop is None or crop.size == 0:
                    continue
                digit, conf = predict_digit(crop)
                print(f"  {digit}: {conf:.1%}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()