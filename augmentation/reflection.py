import cv2
import random

def reflect(img):
    value = random.randrange(1)
    return cv2.flip(img, value)
