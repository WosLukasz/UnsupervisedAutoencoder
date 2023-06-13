import numpy as np
import cv2
import math


def map_to_model_input_range_n1_1(x, y):
    """ normalizuje HSV / HSL do przedzialu od -1 do 1"""

    for i in range(len(y)):
        for j in range(len(y[0])):
            _alpha = math.radians(y[i][j][0])
            x[0, i, j] = [
               np.cos(_alpha),
               np.sin(_alpha),
               y[i][j][1] * 2. - 1.,
               y[i][j][2] * 2. - 1.
           ]


def array_to_model_input_range_n1_1(target, source):
    """ normalizuje HSV / HSL do przedzialu od -1 do 1"""
    target[0, :, :, 0] = np.cos(np.radians(source[:, :, 0]))
    target[0, :, :, 1] = np.sin(np.radians(source[:, :, 0]))
    target[0, :, :, 2] = source[:, :, 1] * 2 - 1
    target[0, :, :, 3] = source[:, :, 2] * 2 - 1


def map_to_model_input_range_0_1(x, y):
    """ normalizuje HSV / HSL do przedzialu od 0 do 1"""
    for i in range(len(y)):
        for j in range(len(y[0])):
            _alpha = math.radians(y[i][j][0])
            x[0, i, j] = [
                (np.cos(_alpha) + 1.) / 2.,
                (np.sin(_alpha) + 1.) / 2.,
                y[i][j][1],
                y[i][j][2]
            ]


def array_to_model_input_range_0_1(target, source):
    """ normalizuje HSV / HSL do przedzialu od -1 do 1"""
    target[0, :, :, 0] = (np.cos(np.radians(source[:, :, 0]))+1.0)/2.0
    target[0, :, :, 1] = (np.sin(np.radians(source[:, :, 0]))+1.0)/2.0
    target[0, :, :, 2] = source[:, :, 1]
    target[0, :, :, 3] = source[:, :, 2]


def convert_raw_data_to_model_input(raw_input, cv_color_conversion, mapping_function):
    """
        rawInput = obrazek typu image w BGR dla uint8
        rawLabel = numer odpowiadający etykiecie
        cv_color_conversion = indeks dla cv2.cvtColor
            ex. cv2.COLOR_BGR2HSV_FULL
                cv2.COLOR_BGR2HLS_FULL
        mapping_function = funkcja konwertująca efekt koncowy na wejscie sieci
                            musi uwzgledniac normalizacje wejscia
            ex. map_to_model_input_range_n1_1
                map_to_model_input_range_0_1
    """
    #scaling RGB colors to to 0-1
    _float_raw_input = np.float32(raw_input) * 1. / 255
    _converted_input = cv2.cvtColor(_float_raw_input, cv_color_conversion)
    _mapped_hsv_input = np.ndarray(shape=(1, len(raw_input), len(raw_input[0]), 4), dtype=np.float)  # 1 is because of batchsize == 1

    mapping_function(_mapped_hsv_input, _converted_input)
    return _mapped_hsv_input

