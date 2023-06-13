# !! INSTRUCTION !!
# PUT ALL DATA IN "./dataset". FOLDER STRUCTURE SHOULD LOOKS LIKE THIS:
# ./dataset
#       /n0
#            -n1023.jpg
#            -n1312.jpg
#            -n1313.jpg
#            -n1323.jpg
#            -n1023.jpg
#       /n1
#            -n2023.jpg
#            -n2013.jpg
#            -n2223.jpg
#            -n2053.jpg
#       /...
#
# OUTPUT SHOULD BE IN "./augmented_dataset" FOLDER

import os
import shutil
import random
import cv2
import string
import augmentation.reflection as reflection
import augmentation.noise as noise
import augmentation.rotation as rotation
import copyUtils.copyDirectoryUtils as copy_utils
from normalization.resizer_to_boundary import resize_old as resize_old
from enum import IntEnum
import params as params
import argparse


class AugmTrans(IntEnum):
    REFLECTION = 0,
    ROTATION = 1,
    BLUR = 2,
    BRIGHTNESS = 3,
    SATURATION = 4,
    NOISE = 5

args = {
"source": 'C:/Projekty/UnsupervisedCNN/data/selected8-easy-2',
"destination": 'C:/Projekty/UnsupervisedCNN/data/selected8-easy-augmented',
"min": str(params.dim_min),
"max": str(params.dim_max),
"force": True,
"number": '1000'
}

main_data_directory = args["source"]
main_augmented_data_directory = args["destination"]
images_size = int(args["number"])

dim_min = int(args["min"])
dim_max = int(args["max"])
transformations_probabilities = {

    AugmTrans.REFLECTION: 0.5,
    AugmTrans.ROTATION: 0.9,
    AugmTrans.BLUR: 0.3,
    AugmTrans.BRIGHTNESS: 0.6,
    AugmTrans.SATURATION: 0.6,
    AugmTrans.NOISE: 0.6
}


def getFileNames(path):
    file_names = os.listdir(path)
    return list(filter(lambda x: x.endswith(".png"), file_names))


def toTransform(probs):
    for i in AugmTrans:
        if probs[i] < transformations_probabilities[i]:
            return True
    return False


def transform(img, probs):
    # Transformations first
    if probs[AugmTrans.REFLECTION] < transformations_probabilities[AugmTrans.REFLECTION]:
        img = reflection.reflect(img)
    if probs[AugmTrans.ROTATION] < transformations_probabilities[AugmTrans.ROTATION]:
        img = rotation.rotate(img)

    # Quality reduction second
    if probs[AugmTrans.BLUR] < transformations_probabilities[AugmTrans.BLUR]:
        img = noise.gaussianBlur(img)

    # Per pixel operations third
    if probs[AugmTrans.BRIGHTNESS] < transformations_probabilities[AugmTrans.BRIGHTNESS]:
        img = noise.change_brightness(img)
    if probs[AugmTrans.SATURATION] < transformations_probabilities[AugmTrans.SATURATION]:
        img = noise.change_saturation(img)

    # Additional noise fourth
    if probs[AugmTrans.NOISE] < transformations_probabilities[AugmTrans.NOISE]:
        img = noise.plotnoise(img)

    return img


copy_utils.copy_directory(main_data_directory, main_augmented_data_directory, dim_min, dim_max, args["force"])
dirs = os.listdir(main_data_directory)

print("[Start] Augment dataset...")
for dir in dirs:
    source_dir_path = os.path.join(main_data_directory, dir).__str__()
    destination_dir_path = os.path.join(main_augmented_data_directory, dir).__str__()
    print("Proccesing " + source_dir_path + "...")
    original_file_names = getFileNames(source_dir_path)
    augmented_file_names = getFileNames(destination_dir_path)
    while images_size > len(augmented_file_names):
        image_name = random.choice(original_file_names)
        img = cv2.imread(os.path.join(source_dir_path, image_name).__str__())
        probs = {i: random.random() for i in AugmTrans}
        if not toTransform(probs):
            continue
        new_image = transform(img, probs)

        # resize to target input
        new_image = resize_old(new_image, dim_min, dim_max)

        new_file_name = random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + random.choice(string.ascii_letters) + image_name
        cv2.imwrite(os.path.join(destination_dir_path, new_file_name).__str__(), new_image)
        augmented_file_names = getFileNames(destination_dir_path)

print("[Stop] Augment dataset...")