from src.preprocessing import preprocess_before_seg
import cv2

#import matplotlib.pyplot as plt



def find_digits(thresh):

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if h > 1 and w > 1:
            boxes.append((x,y,w,h))

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