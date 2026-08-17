from src.preprocessing import preprocess_before_seg
import cv2

#import matplotlib.pyplot as plt



def find_digits(thresh):
    """Find digit contours in a binary image.

    Filters:
      - minimum size (h >= 10, w >= 4): removes noise specks while
        keeping thin CAPTCHA digits (a ۱ can be only 4-5 px wide)
      - aspect ratio h/w in [0.5, 6.0]: digits are roughly square; the
        tall Persian ۱ (and ۷) are narrow, while wide bars / tall page
        strips (UI elements) are still rejected
      - relative area: regions smaller than 30% of the median area are
        dropped (outliers far below the typical digit size)
      - internal complexity: a real digit has at most a couple of nested
        contours (holes for 0/4/6/8/9). Textured regions (UI icons, small
        text, busy backgrounds) produce dozens of contours and are dropped
    """

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if h < 10 or w < 4:
            continue  # too small (noise)

        aspect = h / float(w)
        if not (0.5 <= aspect <= 6.0):
            continue  # not digit-shaped (UI element / edge)

        # Internal complexity filter: a clean digit has <= 3 nested
        # contours (outer + 1-2 holes). Textures have dozens.
        region = thresh[y:y + h, x:x + w]
        inner, hier = cv2.findContours(
            region,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        n_contours = len(inner)
        if n_contours > 5:
            continue  # textured / busy region, not a digit

        boxes.append((x, y, w, h))

    if not boxes:
        return boxes

    # Relative-area filter: drop boxes much smaller than the typical digit
    import statistics
    median_area = statistics.median([w * h for (_, _, w, h) in boxes])
    boxes = [
        (x, y, w, h) for (x, y, w, h) in boxes
        if w * h >= 0.3 * median_area
    ]

    return sorted(
        boxes,
        key=lambda box: box[0]
    )

def crop_digits(thresh, boxes):
    digits = []
    for i, (x,y,w,h) in enumerate(boxes):
        crop = thresh[y-10:y+h+10, x-10:x+w+10]
        digits.append(crop)
    return digits

def main(image_path):
    digits=[]
    thresh = preprocess_before_seg(image_path)
    boxes = find_digits(thresh)
    digits = crop_digits(thresh, boxes)

    # for x,y,w,h in boxes:
    #     cv2.rectangle(
    #         thresh,
    #         (x-5,y-5),
    #         (x+w+5,y+h+5),
    #         (255,255,255),
    #         1
    # )

    print(f"Found {len(digits)} digits.")
    for i, digit in enumerate(digits):
        cv2.imshow(f"Digit {i}", digit)

        cv2.waitKey(0)
if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/captcha.png"
    main(image_path)