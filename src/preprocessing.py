import cv2
import numpy as np
import tensorflow as tf
def preprocess_before_seg(image_path):

    img = cv2.imread(image_path)
    # جدا کردن کانال‌ها
    b, g, r = cv2.split(img)

    # فقط پیکسل‌های نزدیک به سیاه را نگه دار
    mask = (
        (b < 20) &
        (g < 20) &
        (r < 20)
    )

    thresh = np.zeros_like(b)

    thresh[mask] = 255
    

    # حذف نویزهای خیلی کوچک
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 1)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )


    # کمی پر کردن شکستگی داخل اعداد
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )
    return thresh

def preprocess_digit(image):

    # اگر رنگی بود تبدیل به grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # resize
    image = cv2.resize(
        image,
        (28, 28)
    )

    # normalize
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    image = normalization_layer(image)

    # اضافه کردن channel
    image = np.expand_dims(
        image,
        axis=-1
    )

    return image

