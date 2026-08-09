from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing import (
    preprocess_before_seg,
    preprocess_digit,
)

from src.segmentation import (
    find_digits,
    crop_digits,
)


# =====================================
# Settings
# =====================================

# Project root: src/ is one level below the repo root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "digit_classifier_handwritten.keras"

DIGITS = "۰۱۲۳۴۵۶۷۸۹"


# =====================================
# Load Model
# =====================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =====================================
# Predict One Digit
# =====================================

def predict_digit(image):
    """
    Predict one segmented digit.
    """

    processed = preprocess_digit(image)

    # Add channel dimension
    processed = np.expand_dims(
        processed,
        axis=-1
    )

    # Add batch dimension
    processed = np.expand_dims(
        processed,
        axis=0
    )

    prediction = model.predict(
        processed,
        verbose=0
    )

    class_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[0][class_index]
    )

    return (
        DIGITS[class_index],
        confidence
    )


# =====================================
# Predict CAPTCHA
# =====================================

def predict_captcha(image_path, verbose=False):
    """
    Run the complete CAPTCHA prediction pipeline.

    Returns:
        predicted_text
    """

    # -------------------------
    # Preprocessing
    # -------------------------

    thresh = preprocess_before_seg(
        image_path
    )

    # -------------------------
    # Segmentation
    # -------------------------

    boxes = find_digits(
        thresh
    )

    if len(boxes) == 0:
        raise ValueError(
            "No digits were detected."
        )

    # -------------------------
    # Sort left → right
    # -------------------------

    boxes = sorted(
        boxes,
        key=lambda box: box[0]
    )

    # -------------------------
    # Crop
    # -------------------------

    crops = crop_digits(
        thresh,
        boxes
    )

    # -------------------------
    # Prediction
    # -------------------------

    predicted_digits = []

    for crop in crops:

        digit, confidence = predict_digit(
            crop
        )

        predicted_digits.append(
            digit
        )
        if verbose:
            print(
                f"{digit} "
                f"({confidence:.2%})"
            )

    # -------------------------
    # Final CAPTCHA
    # -------------------------

    predicted_text = "".join(
        predicted_digits
    )

    return predicted_text


# =====================================
# Main
# =====================================

if __name__ == "__main__":

    image_path = Path(
        "../data/captcha_generated/images/09991.png"
    )

    result = predict_captcha(
        image_path
    )

    print(
        f"\nPredicted CAPTCHA: {result}"
    )