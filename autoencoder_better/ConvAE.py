from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model


# import keras
# import pydot as pyd
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
#
# keras.utils.vis_utils.pydot = pyd
#
#
# def visualize_model(model):
#   return SVG(model_to_dot(model).create(prog='dot', format='svg'))


def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]): #[32, 64, 128, 256] <- spróbować z takimi parametrami  https://arxiv.org/ftp/arxiv/papers/1710/1710.08961.pdf
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))
    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu'))
    model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))
    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))
    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    #plot_model(model, to_file='C:/Users/wolukasz/Desktop/PG/II_stopien/Praca_Magisterska/REPORTS/2021-05-05/autoencoder/model_plot.png', show_shapes=True, show_layer_names=True, rankdir="LR")
    return model