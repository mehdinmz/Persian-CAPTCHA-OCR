import cv2
import numpy as np
import tensorflow as tf


def preprocess_before_seg(image_path, auto_otsu=False):
    """Binarize an image for digit segmentation.

    The default mode keeps near-black pixels (threshold < 20), which works
    for clean CAPTCHAs. When ``auto_otsu=True`` (or when the default mask
    yields no foreground at all), an adaptive Otsu threshold on the
    grayscale image is used instead, so real-world screenshots/photos with
    gray or colored text also work.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}. Check that the file exists."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)

    # فقط پیکسل‌های نزدیک به سیاه را نگه دار
    mask = (b < 20) & (g < 20) & (r < 20)
    thresh = np.zeros_like(b)
    thresh[mask] = 255

    # اگر حالت خودکار فعال بود یا ماسک چیزی پیدا نکرد، Otsu را امتحان کن
    if auto_otsu or (thresh > 0).sum() < 50:
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

    # حذف نویزهای خیلی کوچک
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # کمی پر کردن شکستگی داخل اعداد
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh


def preprocess_digit(image):
    """Prepare a single digit image for the CNN (28x28, normalized)."""

    # اگر رنگی بود تبدیل به grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize
    image = cv2.resize(image, (28, 28))

    # normalize
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    image = normalization_layer(image)

    # اضافه کردن channel
    image = np.expand_dims(image, axis=-1)

    return image