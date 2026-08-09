"""Generate a demo grid: CAPTCHAs from multiple Persian fonts, solved by the pipeline.

Usage:
    python src/demo.py [--out demo.png] [--rows 3]
"""
import argparse
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.pipeline import predict_captcha  # noqa: E402

DIGITS = "۰۱۲۳۴۵۶۷۸۹"
WIDTH, HEIGHT = 200, 80
FONT_SIZE = 42


def pick_fonts():
    """Return a handful of Persian fonts (best renderers for digits)."""
    fonts_dir = BASE / "fonts" / "persian"
    # Known-good fonts for digit rendering (0 renders as a full ring)
    candidates = [
        "BYekan.ttf",
        "BTitrBd.ttf",
        "BHoma.ttf",
        "BArshia.ttf",
        "BBadrBd.ttf",
        "BLotus.ttf",
        "BNazanin.ttf",
        "BZar.ttf",
        "BTabassom.ttf",
        "BEsfehanBd.ttf",
    ]
    found = [f for f in candidates if (fonts_dir / f).exists()]
    # fall back to any ttf in the folder
    if not found:
        found = sorted(p.name for p in fonts_dir.glob("*.ttf"))
    return found


def make_captcha(text, font_path, seed=None):
    """Render a captcha like src/captcha_generator.py but with given font."""
    rng = random.Random(seed)
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    # noise lines
    for _ in range(6):
        draw.line(
            (rng.randint(0, WIDTH), rng.randint(0, HEIGHT),
             rng.randint(0, WIDTH), rng.randint(0, HEIGHT)),
            fill=(rng.randint(60, 170), rng.randint(60, 170), rng.randint(60, 170)),
            width=2,
        )
    # noise dots
    for _ in range(60):
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


def main():
    parser = argparse.ArgumentParser(description="Create a demo grid of solved CAPTCHAs")
    parser.add_argument("--out", default=str(BASE / "demo.png"), help="Output image path")
    parser.add_argument("--rows", type=int, default=4, help="Number of rows")
    args = parser.parse_args()

    fonts = pick_fonts()
    if len(fonts) < args.rows:
        args.rows = len(fonts)

    rows = []
    statuses = []
    rng = random.Random(7)
    for fi in range(args.rows):
        font_path = BASE / "fonts" / "persian" / fonts[fi]
        text = "".join(rng.choice(DIGITS) for _ in range(5))
        captcha = make_captcha(text, font_path, seed=100 + fi)
        # save temp file for pipeline (it reads from path)
        tmp = BASE / "data" / f"_demo_{fi}.png"
        tmp.parent.mkdir(exist_ok=True)
        captcha.save(tmp)
        pred = predict_captcha(str(tmp))
        tmp.unlink()
        ok = pred == text
        statuses.append((fonts[fi], text, pred, ok))
        rows.append(captcha)

    # Build grid: each row = captcha | truth -> pred (green/red)
    cell_w = WIDTH
    cell_h = HEIGHT + 24  # room for the caption line
    grid = Image.new("RGB", (cell_w, args.rows * cell_h), "white")
    draw = ImageDraw.Draw(grid)
    for i, (font_name, truth, pred, ok) in enumerate(statuses):
        y = i * cell_h
        grid.paste(rows[i], (0, y))
        color = (34, 139, 34) if ok else (220, 20, 60)
        label = f"{font_name}: true={truth}  pred={pred}  {'OK' if ok else 'FAIL'}"
        draw.text((8, y + HEIGHT + 4), label, fill=color)
    grid.save(args.out)
    print(f"Saved demo grid -> {args.out}")
    for font_name, truth, pred, ok in statuses:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {font_name}: true={truth} pred={pred}")


if __name__ == "__main__":
    main()
