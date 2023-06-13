import cv2
import random


def rotate(image):
    (h, w) = image.shape[:2]
    angle = random.uniform(-15, 15)  # max 61

    # provide the type of transformation, here: rotation on center in (w/2,h/2) by angle in degrees and with scale=1
    M = cv2.getRotationMatrix2D(center=(w/2, h/2), angle=angle, scale=1)
    return cv2.warpAffine(src=image, M=M, dsize=(w, h), borderMode=cv2.BORDER_REPLICATE)     # cv2.BORDER_REFLECT