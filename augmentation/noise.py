import cv2
import skimage
import random
import numpy as np
from PIL import ImageEnhance
from PIL import Image


def plotnoise(img):
    img = img / 255.0
    variance = random.uniform(0, 0.9) / 100.0
    mean = random.uniform(0, 0.2) / 100.0
    gimg = skimage.util.random_noise(img, mode="gaussian", mean=mean, var=variance)
    return gimg*255.0


def gaussianBlur(img):
    sigmaX = random.choice(range(10))
    if sigmaX % 2 == 0:
        sigmaX = sigmaX + 1

    sigmaY = random.choice(range(10))
    if sigmaY % 2 == 0:
        sigmaY = sigmaY + 1

    return cv2.GaussianBlur(img, (sigmaX, sigmaY), cv2.BORDER_CONSTANT)


def change_brightness(img):
    value = random.randrange(50)   # max 50
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if random.random() < 0.5:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def change_saturation(img):
    var = random.uniform(0.4, 1.7)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    converter = ImageEnhance.Color(im_pil)
    im_pil = converter.enhance(var)
    im_np = np.asarray(im_pil)
    img = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
    return img
