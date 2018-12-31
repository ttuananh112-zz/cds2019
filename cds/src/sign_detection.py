import cv2
import numpy as np

OFFSET = 0

def detect_sign(image_np):
    img = image_np[20:, :]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = cv2.inRange(hsv, (0, 100, 100), (20, 255, 255))
    upper_red = cv2.inRange(hsv, (150, 100, 100), (179, 255, 255))
    red = cv2.bitwise_or(lower_red, upper_red)


    blue = cv2.inRange(hsv, (90, 100, 100), (110, 255, 255))
    # combined = cv2.bitwise_or(red, blue)
    combined = blue
    rev = cv2.bitwise_not(combined)
    cv2.imshow("Thresholding", rev)
    cntr_frame, contours, hierarchy = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    rect = None
    (height, width, channel) = img.shape
    print(img.shape)
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        # x -= int(OFFSET * RATIO)
        x -= OFFSET
        if x < 0:
            x = 0

        y -= OFFSET
        if y < 0:
            y = 0

        # x2 = int(x + w + 2 * OFFSET * RATIO)
        x2 = x + w + 2 * OFFSET
        if x2 > width:
            x2 = width

        y2 = y + h + 2 * OFFSET
        if y2 > height:
            y2 = height

        if w >= 5 and h >= 5 and (0.7 <= h / w <= 1.0 / 0.7):
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 1)
    
    return img