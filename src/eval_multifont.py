"""Evaluate multi-font model on fresh captchas per font (seeds not seen in training)."""
import random
import sys
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.pipeline import predict_captcha  # noqa: E402

DIGITS = "۰۱۲۳۴۵۶۷۸۹"
WIDTH, HEIGHT = 200, 80
FONT_SIZE = 42

FONTS = ["BYekan.ttf", "BTitrBd.ttf", "BHoma.ttf", "BHelal.ttf", "BNikoo.ttf",
         "BSahar.ttf", "BShiraz.ttf", "BTabssom.ttf", "BLotusBd.ttf", "BBadrBd.ttf"]


def make_captcha(text, font_path, seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    for _ in range(8):
        draw.line((rng.randint(0, WIDTH), rng.randint(0, HEIGHT),
                   rng.randint(0, WIDTH), rng.randint(0, HEIGHT)),
                  fill=(rng.randint(50, 180), rng.randint(50, 180), rng.randint(50, 180)), width=2)
    for _ in range(80):
        x, y = rng.randint(0, WIDTH), rng.randint(0, HEIGHT)
        draw.ellipse((x, y, x + 2, y + 2),
                     fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    font = ImageFont.truetype(str(font_path), FONT_SIZE)
    x = 20
    for digit in text:
        y = rng.randint(15, 25)
        draw.text((x, y), digit, font=font, fill="black")
        x += 32
    return img


N_PER_FONT = 40
tmp = BASE / "data" / "_eval_mf.png"
tmp.parent.mkdir(exist_ok=True)

print("Per-font captcha accuracy on fresh seeds (seeds 50000+, unseen):")
for fname in FONTS:
    font_path = BASE / "fonts" / "persian" / fname
    correct = 0
    total = N_PER_FONT
    for i in range(N_PER_FONT):
        rng = random.Random(50000 + i)
        text = "".join(rng.choice(DIGITS) for _ in range(5))
        img = make_captcha(text, font_path, 50000 + i)
        img.save(tmp)
        pred = predict_captcha(str(tmp))
        if pred == text:
            correct += 1
    print(f"  {fname:<16} {correct}/{total} = {correct/total:.0%}")
tmp.unlink(missing_ok=True)
