from pathlib import Path
import sys

import cv2
import numpy as np
import tensorflow as tf

# Project root: src/ is one level below the repo root
BASE_DIR = Path(__file__).resolve().parent.parent

# Allow running as a script: python src/predictor.py ...
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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

def _looks_like_captcha(thresh, boxes):
    """Heuristic: synthetic CAPTCHAs are small images with compact boxes.

    Real-world screenshots are large (>300px wide) and have bigger,
    more spaced-out boxes. Synthetic captchas are typically ~200px wide
    with small (10-35px) digit boxes.
    """
    h, w = thresh.shape[:2]
    if w > 300:
        return False
    if not boxes:
        return False
    avg_h = sum(b[3] for b in boxes) / len(boxes)
    return avg_h <= 40


def _geometric_digit(crop, hw_digit, pf_digit):
    """Tie-break between the handwritten and multi-font models using
    simple geometry when the two models disagree.

    - A tall, narrow blob (aspect ratio > 2.2) with no holes is a ۱.
    - A wide blob (aspect < 1.2) with an enclosed hole is a ۰.
    Otherwise fall back to the handwritten model (better on handwriting).
    """
    import cv2
    gray = crop if len(crop.shape) == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Ensure binary (0/255) orientation: white digits on black
    if gray.mean() > 127:
        gray = cv2.bitwise_not(gray)
    # Morphology to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    h, w = gray.shape[:2]
    aspect = h / max(w, 1)

    contours, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hier is not None:
        holes = sum(1 for hh in hier[0] if hh[3] != -1)

    if aspect > 2.2 and holes == 0:
        # Tall and narrow with no enclosed area → definitely ۱
        return "۱"
    if aspect < 1.2 and holes >= 1:
        # Wide with an enclosed hole → definitely ۰
        return "۰"
    return hw_digit


def main():
    """CLI: captcha-ocr <image_path> — print the predicted CAPTCHA text."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Recognize Persian digit CAPTCHAs")
    parser.add_argument("image", nargs="?", help="Path to CAPTCHA image")
    parser.add_argument("--image", dest="image_path", help="Path to CAPTCHA image")
    parser.add_argument("--conf", action="store_true", help="Show per-digit confidence")
    parser.add_argument("--auto", action="store_true",
                        help="Use adaptive (Otsu) binarization for real-world images")
    args = parser.parse_args()

    image_path = args.image_path or args.image
    if not image_path:
        parser.error("the following arguments are required: image or --image")

    try:
        import cv2
        import numpy as np
        from src.pipeline import predict_captcha
        from src.predictor import predict_digit
        from src.preprocessing import preprocess_before_seg
        from src.segmentation import find_digits, crop_digits

        # Segment the image
        thresh = preprocess_before_seg(image_path, auto_otsu=args.auto)
        boxes = sorted(find_digits(thresh), key=lambda b: b[0])
        crops = crop_digits(thresh, boxes)

        if not crops:
            raise ValueError("No digits were detected.")

        # Decide: single digit vs full CAPTCHA.
        # A CAPTCHA (multi-font synthetic) is handled by src.pipeline
        # (multi-font model, trained on synthetic crops). Real-world
        # screenshots are handled by the handwritten model, which
        # generalizes better. We choose based on whether the image
        # looks like a synthetic captcha: synthetic captchas are small
        # (<= 220px wide) and have compact boxes.
        is_captcha = len(crops) > 1 and _looks_like_captcha(thresh, boxes)

        if is_captcha:
            # Full CAPTCHA (multiple digits) → pipeline (multi-font model)
            text = predict_captcha(image_path)
            print(text)
            if args.conf:
                from src.pipeline import predict_digit as captcha_digit
                for d, crop in zip(text, crops):
                    if crop is None or crop.size == 0:
                        continue
                    digit, conf = captcha_digit(crop)
                    print(f"  {digit}: {conf:.1%}")
            return

        # Single digit (or real-world screenshot) → handwritten model
        # with a geometric tie-breaker when the two models disagree.
        from src.pipeline import predict_digit as captcha_digit

        digits_out = []
        for crop in crops:
            hw_digit, hw_conf = predict_digit(crop)
            pf_digit, pf_conf = captcha_digit(crop)
            if hw_digit == pf_digit:
                digit = hw_digit
            else:
                digit = _geometric_digit(crop, hw_digit, pf_digit)
            digits_out.append((digit, max(hw_conf, pf_conf)))

        text = "".join(d for d, _ in digits_out)
        print(text)
        if args.conf:
            for d, conf in digits_out:
                print(f"  {d}: {conf:.1%}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()