from pathlib import Path

import pandas as pd

from src.pipeline import predict_captcha


# =====================================
# Settings
# =====================================

LABELS_PATH = Path(
    "../data/captcha_generated/labels.csv"
)

IMAGE_DIR = Path(
    "../data/captcha_generated/images"
)


# =====================================
# Evaluation
# =====================================

def evaluate():

    # -------------------------
    # Load labels
    # -------------------------

    df = pd.read_csv(
        LABELS_PATH
    )

    total_captchas = len(df)

    correct_captchas = 0
    correct_digits = 0
    total_digits = 0

    failed_images = []

    print(
        f"Evaluating {total_captchas} CAPTCHA images...\n"
    )

    # -------------------------
    # Run Pipeline
    # -------------------------

    for index, row in df.iterrows():

        filename = row["filename"]
        true_label = str(row["label"])

        image_path = IMAGE_DIR / filename

        try:

            predicted_label = predict_captcha(
                image_path
            )

        except Exception as error:

            print(
                f"[ERROR] {filename}: {error}"
            )

            failed_images.append(
                filename
            )

            continue

        # -------------------------
        # CAPTCHA Accuracy
        # -------------------------

        if predicted_label == true_label:

            correct_captchas += 1

        # -------------------------
        # Digit Accuracy
        # -------------------------

        for true_digit, predicted_digit in zip(
            true_label,
            predicted_label
        ):

            total_digits += 1

            if true_digit == predicted_digit:

                correct_digits += 1

        # -------------------------
        # Progress
        # -------------------------

        if (index + 1) % 500 == 0:

            print(
                f"Processed "
                f"{index + 1}/{total_captchas}"
            )

    # =====================================
    # Results
    # =====================================

    digit_accuracy = (
        correct_digits / total_digits
        if total_digits > 0
        else 0
    )

    captcha_accuracy = (
        correct_captchas / total_captchas
        if total_captchas > 0
        else 0
    )

    print("\n")
    print("=" * 45)
    print("FINAL EVALUATION")
    print("=" * 45)

    print(
        f"Total CAPTCHA:       {total_captchas}"
    )

    print(
        f"Correct CAPTCHA:     {correct_captchas}"
    )

    print(
        f"Wrong CAPTCHA:       "
        f"{total_captchas - correct_captchas}"
    )

    print(
        f"Digit Accuracy:      "
        f"{digit_accuracy:.2%}"
    )

    print(
        f"CAPTCHA Accuracy:    "
        f"{captcha_accuracy:.2%}"
    )

    print(
        f"Failed Images:       "
        f"{len(failed_images)}"
    )

    print("=" * 45)


# =====================================
# Main
# =====================================

if __name__ == "__main__":

    evaluate()
