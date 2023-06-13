from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, \
    Dropout, BatchNormalization
from tensorflow.python.keras.models import Sequential, Model

import params


# def get_autoencoder_models():
#     dims = [params.images_shape_x * params.images_shape_y, 2000, 1000, 400]
#     act = 'relu'
#     init = 'glorot_uniform'
#
#     n_stacks = len(dims) - 1
#     input_img = Input(shape=(dims[0],), name='input')
#     x = input_img
#     for i in range(n_stacks - 1):
#         x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
#
#     encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)
#
#     x = encoded
#     for i in range(n_stacks - 1, 0, -1):
#         x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)
#
#     x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
#     decoded = x
#
#     return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


# def get_autoencoder_models():
#     model = Sequential()
#
#     # 1st convolution layer
#     model.add(Conv2D(16, (4, 4), padding='same', input_shape=(params.images_shape_x, params.images_shape_y, 1)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
#
#     # 2nd convolution layer
#     model.add(Conv2D(8, (4, 4), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
#
#     # -------------------------
#
#     # 3rd convolution layer
#     model.add(Conv2D(8, (4, 4), padding='same'))
#     model.add(Activation('relu'))
#     model.add(UpSampling2D((3, 3)))
#
#     # 4rd convolution layer
#     model.add(Conv2D(16, (4, 4), padding='same'))
#     model.add(Activation('relu'))
#     model.add(UpSampling2D((3, 3)))
#
#     # -------------------------
#     model.add(Conv2D(1, (5, 5), padding='same'))
#     model.add(Activation('sigmoid'))
#
#     return model, Model(inputs=model.layers[0].input, outputs=model.layers[7].output, name='encoder')

# autoencoder_model_100, autoencoder_model_1000
def get_autoencoder_models():
    input_shape = (params.images_shape_x, params.images_shape_y, 1)
    filters = [16, 32, 64, 100] # [32, 64, 128, 256] <- spróbować z takimi parametrami  https://arxiv.org/ftp/arxiv/papers/1710/1710.08961.pdf

    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 4, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu'))

    model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 3, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 4, strides=2, padding='same', name='deconv1'))

    return model, Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)


# def get_autoencoder_models():
#     input_shape = (params.images_shape_x, params.images_shape_y, 1)
#     filters = [8, 16, 32, 64, 500]
#
#     model = Sequential()
#     if input_shape[0] % 8 == 0:
#         pad3 = 'same'
#     else:
#         pad3 = 'valid'
#     model.add(Conv2D(filters[0], 4, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))
#     model.add(Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='conv2'))
#     model.add(Conv2D(filters[2], 3, strides=2, padding='same', activation='relu', name='conv3'))
#     model.add(Conv2D(filters[3], 3, strides=2, padding=pad3, activation='relu', name='conv4'))
#
#     model.add(Flatten())
#     model.add(Dense(units=filters[4], name='embedding'))
#     model.add(Dense(units=filters[3]*int(input_shape[0]/16)*int(input_shape[0]/16), activation='relu'))
#
#     model.add(Reshape((int(input_shape[0]/16), int(input_shape[0]/16), filters[3])))
#     model.add(Conv2DTranspose(filters[2], 3, strides=2, padding=pad3, activation='relu', name='deconv4'))
#     model.add(Conv2DTranspose(filters[1], 3, strides=2, padding='same', activation='relu', name='deconv3'))
#     model.add(Conv2DTranspose(filters[0], 3, strides=2, padding='same', activation='relu', name='deconv2'))
#     model.add(Conv2DTranspose(input_shape[2], 4, strides=2, padding='same', name='deconv1'))
#
#     return model, Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)


def load_trained_encoder():
    model, _ = get_autoencoder_models()
    model.load_weights(params.autoencoder_weights_path)

    encoder = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)


    encoder.summary()

    return encoder
