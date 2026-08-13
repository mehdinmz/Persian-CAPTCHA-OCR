"""Check which Persian fonts render all digits fully (esp. 0 as a ring)."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

fonts_dir = Path("fonts/persian")
DIGITS = "۰۱۲۳۴۵۶۷۸۹"

print(f"{'font':<16} {'0-area':>7} {'min-area':>9} {'min-digit':>9}  status")
for font_path in sorted(fonts_dir.glob("*.ttf")):
    try:
        font = ImageFont.truetype(str(font_path), 42)
    except Exception as e:
        print(f"{font_path.name:<16} ERROR {e}")
        continue
    areas = {}
    for d in DIGITS:
        img = Image.new("L", (60, 80), 255)
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), d, font=font, fill=0)
        # count dark pixels
        pixels = img.load()
        dark = sum(1 for x in range(60) for y in range(80) if pixels[x, y] < 100)
        areas[d] = dark
    zero_area = areas["۰"]
    min_area = min(areas.values())
    min_digit = min(areas, key=areas.get)
    # criteria: zero area should be a reasonable ring (> 40 px), and no digit too tiny
    ok = zero_area >= 40 and min_area >= 30
    print(f"{font_path.name:<16} {zero_area:>7} {min_area:>9} {min_digit!r:>9}  {'OK' if ok else '--'}")
