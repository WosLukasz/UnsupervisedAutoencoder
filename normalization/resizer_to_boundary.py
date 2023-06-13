import cv2
from PIL import Image
import normalization.image_utils as image_utils

default_image_size = 200

def resize_old(src, dim_min, dim_max):
    ratio = max(len(src[0]), len(src)) / dim_max
    ratio = min(
        ratio,
        min(len(src[0]), len(src)) / dim_min
    )
    return cv2.resize(src, (int(len(src[0])*1.0 / ratio), int(len(src)*1.0 / ratio)))


def make_square(im, min_size=default_image_size, fill_color=(255, 255, 255, 0)):
    im = image_utils.cv2ToPILConvert(im)
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return image_utils.PILToCv2Convert(new_im)


def make_scale_square(im, autoColor=True, end_size=default_image_size, fill_color=(255, 255, 255, 0)):
    im = image_utils.cv2ToPILConvert(im)
    x, y = im.size
    size = max(x, y)
    if autoColor:
        fill_color = image_utils.getMeanColor(im)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    img = image_utils.PILToCv2Convert(new_im)
    return cv2.resize(img, (end_size, end_size))


def resize(im, size=default_image_size):
    return cv2.resize(im, (size, size))


def make_square_fill_by_replicate(im, desired_size=default_image_size):
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE)