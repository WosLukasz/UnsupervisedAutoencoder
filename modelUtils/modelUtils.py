from keras.applications.nasnet import NASNetLarge
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.models import Model

import params


def get_model(architecture='VGG16', layer='fc2'):
    """Keras Model of the VGG16 network, with the output layer set to `layer`.
    The default layer is the second-to-last fully connected layer 'fc2' of
    shape (4096,).
    Parameters
    ----------
    layer : str
        which layer to extract (must be of shape (None, X)),
         For:
         - VGG16:'fc2', 'fc1' or 'flatten'
         - VGG19:'fc2', 'fc1' or 'flatten'
         - Xception:'avg_pool'
         - NASNetLarge: 'global_average_pooling2d_1'
    """
    base_model = None
    if architecture == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=True)
    elif architecture == 'Xception':
        base_model = Xception(weights='imagenet', include_top=True)
    elif architecture == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=True)
    elif architecture == 'NASNetLarge':
        base_model = NASNetLarge(weights='imagenet', include_top=True)

    base_model.summary()

    return Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)


def feature_vectors(imgs_dict, model, architecture='VGG16'):
    f_vect = {}
    for fn, img in imgs_dict.items():
        f_vect[fn] = params.feature_vector(img, model, architecture)
    return f_vect


def convert_to_numeric_labels(x):
    d = {}
    count = 0
    for i in x:
        if i not in d:
            d[i] = count
            count += 1

    new_x = [d[i] for i in x]

    return new_x