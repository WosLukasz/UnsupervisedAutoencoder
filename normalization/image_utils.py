import cv2
from PIL import Image
import numpy as np


def cv2ToPILConvert(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(im)


def PILToCv2Convert(img):
    open_cv_image = np.array(img)
    return open_cv_image[:, :, ::-1].copy()


def getColorOfPixel(img, x , y): # img is PIL format
    pix = img.load()
    return pix[x,y]


def getCornerrPixelPosition(img, corrner): # img is PIL format, corrner : 0->left top, 1-> right top, 2->left bottom, 3->right bottom
    w, h = img.size
    padding = 100
    x = 0
    y = 0

    if corrner is 0 or corrner is 2:
        x = padding
    if corrner is 1 or corrner is 3:
        x = w - padding
    if corrner is 0 or corrner is 1:
        y = padding
    if corrner is 2 or corrner is 3:
        y = h - padding

    return x, y


def getMeanColor(img):
    mean = (0, 0, 0)
    for i in range(4):
        pos = getCornerrPixelPosition(img, i)
        col = getColorOfPixel(img, pos[0], pos[1])
        if isWhiteColor(col):
            mean = (mean[0] + col[0], mean[1] + col[1], mean[2] + col[2])
    mean = (int(mean[0]/4), int(mean[1]/4), int(mean[2]/4), 0)
    return mean


def isWhiteColor(color):
    Y = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    return Y > 128


def crop_image(img, x, y, x1, y1):
    return img[y:y1, x:x1]


def crop_image_vott(img, x, y, w, h):
    return img[y:y+h, x:x+w]