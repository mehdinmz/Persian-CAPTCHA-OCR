from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Project root: src/ is one level below the repo root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "digit_classifier_captcha_v2.keras"

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

    Returns:
        digit      -> Persian digit
        confidence -> prediction confidence
    """

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
        from src.pipeline import predict_captcha
        text = predict_captcha(args.image)
        print(text)
        if args.conf:
            # per-digit confidence via predictor
            import cv2
            from src.preprocessing import preprocess_before_seg
            from src.segmentation import find_digits, crop_digits
            thresh = preprocess_before_seg(args.image)
            boxes = sorted(find_digits(thresh), key=lambda b: b[0])
            crops = crop_digits(thresh, boxes)
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