import random
from pathlib import Path
import sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# =====================
# Settings
# =====================

NUM_CAPTCHAS = 10000

WIDTH = 200
HEIGHT = 80

FONT_SIZE = 42
FONT_PATH = BASE_DIR / "fonts" / "persian" / "BYekan.ttf"
OUTPUT_DIR = BASE_DIR / "data" / "captcha_generated"
DIGITS = "۰۱۲۳۴۵۶۷۸۹"


# =====================
# Create folders
# =====================

output_dir = Path(OUTPUT_DIR)
image_dir = output_dir / "images"

image_dir.mkdir(parents=True, exist_ok=True)

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

labels = []


# =====================
# Generate
# =====================

for i in range(NUM_CAPTCHAS):

    text = "".join(random.choice(DIGITS) for _ in range(5))

    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(img)

    # خطوط نویز
    for _ in range(8):

        draw.line(
            (
                random.randint(0, WIDTH),
                random.randint(0, HEIGHT),
                random.randint(0, WIDTH),
                random.randint(0, HEIGHT),
            ),
            fill=(
                random.randint(50, 180),
                random.randint(50, 180),
                random.randint(50, 180),
            ),
            width=2,
        )

    # نقطه‌های نویز
    for _ in range(80):

        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)

        draw.ellipse(
            (x, y, x + 2, y + 2),
            fill=(
                random.randint(0,255),
                random.randint(0,255),
                random.randint(0,255),
            ),
        )

    # نوشتن ارقام
    x = 20

    for digit in text:

        y = random.randint(15, 25)

        draw.text(
            (x, y),
            digit,
            font=font,
            fill="black"
        )

        x += 32

    filename = f"{i:05d}.png"

    img.save(image_dir / filename)

    labels.append(
        {
            "filename": filename,
            "label": text
        }
    )

    if i % 1000 == 0:
        print(i)


# =====================
# Save labels
# =====================

pd.DataFrame(labels).to_csv(
    output_dir / "labels.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Done.")