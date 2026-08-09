import random
from pathlib import Path
import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageFilter,
    ImageOps
)

# =====================================
# Configuration
# =====================================

OUTPUT_DIR = Path("../data/synthetic_dataset")

FONTS_DIR = Path("../fonts/persian")

IMAGE_SIZE = (28, 28)

CANVAS_SIZE = (60, 60)

SAMPLES_PER_DIGIT = 4000

FONT_SIZE = (40, 46)

DIGITS = "۰۱۲۳۴۵۶۷۸۹"

NOISE_STD = 6

ROTATION = (-5, 5)

SHIFT = (-2, 2)

BLUR_PROBABILITY = 0.20

# =====================================

fonts = list(FONTS_DIR.glob("*.ttf"))
print(fonts)
if len(fonts) == 0:
    raise FileNotFoundError(
        f"No fonts found inside {FONTS_DIR}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(f"Found {len(fonts)} fonts.")


# =====================================
# Dataset Generation
# =====================================
file_num = 0
for digit in DIGITS:
    class_dir = OUTPUT_DIR / str(file_num)

    class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Generating {digit} ...")

    for i in range(SAMPLES_PER_DIGIT):

        # -------------------------
        # Random Font
        # -------------------------

        font = ImageFont.truetype(
            str(random.choice(fonts)),
            random.randint(*FONT_SIZE)
        )

        # -------------------------
        # White Background
        # -------------------------

        image = Image.new(
            "L",
            CANVAS_SIZE,
            255
        )

        draw = ImageDraw.Draw(image)

        bbox = draw.textbbox(
            (0, 0),
            digit,
            font=font
        )

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        x = (
            (CANVAS_SIZE[0] - width) // 2
            + random.randint(*SHIFT)
        )

        y = (
            (CANVAS_SIZE[1] - height) // 2
            + random.randint(*SHIFT)
        )

        # -------------------------
        # Thickness Augmentation
        # -------------------------

        if random.random() < 0.5:

            offsets = [(0, 0)]

        else:

            offsets = [
                (0, 0),
                (1, 0)
            ]

        for dx, dy in offsets:

            draw.text(
                (x + dx, y + dy),
                digit,
                font=font,
                fill=0
            )
        # -------------------------
        # Rotation
        # -------------------------

        image = image.rotate(
            random.uniform(*ROTATION),
            fillcolor=255
        )

        # -------------------------
        # Blur
        # -------------------------

        if random.random() < BLUR_PROBABILITY:

            image = image.filter(
                ImageFilter.GaussianBlur(
                    random.uniform(0.3, 0.8)
                )
            )
        
        # -------------------------
        # Resize
        # -------------------------

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )
        image = ImageOps.invert(image)  # Invert colors to have black digits on white background

        # -------------------------
        # Gaussian Noise
        # -------------------------

        img = np.array(image).astype(np.float32)

        noise = np.random.normal(
            0,
            NOISE_STD,
            img.shape
        )

        img = np.clip(
            img + noise,
            0,
            255
        ).astype(np.uint8)

        image = Image.fromarray(img)
        # -------------------------
        # Save
        # -------------------------
        image.save(
            class_dir / f"{i:05d}.png"
        )
    file_num +=1
print("\n===================================")
print("Synthetic Dataset Generated Successfully!")
print("===================================")