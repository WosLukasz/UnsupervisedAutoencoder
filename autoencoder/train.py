import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os

import copyUtils.copyDirectoryUtils as copy_utils
import params
from model import get_autoencoder_models
from params import images_shape_x, images_shape_y, images_main_directory, get_file
from utils.gpu import gpu_fix

from keras.datasets import mnist


def train(paths):
    # (x_train, train_y), (x_val, test_y) = mnist.load_data()

    # x_train = x_train.astype('float32') / 255.
    # x_val = x_val.astype('float32') / 255.
    # x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    # x_val = np.reshape(x_val, (len(x_val), 28, 28, 1))
    # x_train = np.expand_dims(x_train, axis=3)
    # x_val = np.expand_dims(x_val, axis=3)

    x = load_images(paths)
    x_train, x_val = train_test_split(x, test_size=0.2, random_state=42)
    model, _ = get_autoencoder_models()
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    logdir = os.path.join('logs') #'D:/dataset/models/logs'
    model.fit(x_train, x_train, batch_size=256, epochs=2000, validation_data=(x_val, x_val), callbacks=[
        EarlyStopping(monitor='val_loss', patience=300, verbose=0, mode='min'),
        ModelCheckpoint(params.autoencoder_weights_path, save_best_only=True, monitor='val_loss', mode='min'),
        TensorBoard(log_dir=logdir)
    ])
    model.save_weights(params.autoencoder_weights_path)


def test(paths):
    x = load_images(paths)
    _, x_val = train_test_split(x, test_size=0.2, random_state=42)

    # (x_train, train_y), (x_val, test_y) = mnist.load_data()
    # x_train = np.expand_dims(x_train, axis=3)
    # x_val = np.expand_dims(x_val, axis=3)

    model, _ = get_autoencoder_models()
    model.load_weights(params.autoencoder_weights_path)
    y = model.predict(x_val)
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 2
    for i in range(0, 3):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow((x_val[i]).reshape(images_shape_x, images_shape_y))
        fig.add_subplot(rows, columns, i + 1 + columns)
        plt.imshow((y[i]).reshape(images_shape_x, images_shape_y))
    plt.show()


def load_images(paths):
    return np.array(list(map(lambda path: get_file(path, (images_shape_x, images_shape_y)), paths)))


# tf.debugging.set_log_device_placement(True)

gpu_fix()

train_images_array, _ = copy_utils.get_images_list(images_main_directory)
train(train_images_array)
#test(train_images_array)

# model, _ = get_autoencoder_models()
# model.summary()

# get_autoencoder_models()[0].summary()

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# from keras.models import Model
# model, encoder = get_autoencoder_models()
# model.load_weights(autoencoder_weights_path)
# x = load_images(train_images_array[0:1])
# compressed = encoder.predict(x)
# y = Model(inputs=model.layers[8].input, outputs=model.layers[13].output, name='encoder').predict(compressed)
# plt.imshow((y[0]).reshape(images_shape_x, images_shape_y))
# plt.show()

#wytrenować na rozwiązaniu piotra na rela15 ! i sprawdzić czy też sobie tak kiepsko radzi jak to rozwiązanie moje