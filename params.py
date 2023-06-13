import cv2
import numpy as np
from keras.applications.nasnet import preprocess_input as nasnet_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input
import autoencoder.model
import matplotlib.pyplot as plt

from modelUtils import modelUtils

images_main_directory = 'D:/final_datasets/REAL_15_CLASSES_augmented' # 1  path to train data direcotory
autoencoder_model = 'D:/dataset/autoencoderTest/models/autoencoder_model'
extracted_features_directory = "D:/dataset/autoencoderTest/autoencoderFeatures" # 2
predicted_clusters_directory = "D:/dataset/autoencoderTest/autoencoderPredicted" # 4
models_direcory = "D:/dataset/autoencoderTest/models" # path to direcotory with models
test_dataset_direcory = "D:/final_datasets/REAL_15_CLASSES_augmented/" # path to test data direcotory
autoencoder_weights_path = 'D:/dataset/autoencoderTest/models/ae_weights.h5'
batchSize = 100 # batchSize > clusters && batchSize > selected_features_number # batches in k_means and PCA
dim_min = 200 # scaling images
dim_max = 500 # scaling images
clusters = 10 # classes / clusters count
selected_features_number = 80 # min(n_features_in, n_samples) using in PCA
# images_shape_x = 299 # VGG16/VGG 19: 224, Xception: 299, NASNetLarge: 331
# images_shape_y = 299
images_shape_x = 128 # 128
images_shape_y = 128 # 128
use_augmented_dataset = True
pca_model_name = "autoencoderPca.model"
k_means_model_name ="autoencoderKmeans.model"
model_architecture = "Xception"
model_layer = "avg_pool"
test_on_data = "train" # train or test
fourier_transform = False


# def get_model():
#     return modelUtils.get_model(architecture=model_architecture, layer=model_layer)
#
#
# def get_file(path, size):
#     return cv2.resize(cv2.imread(path), size)
#
#
# def feature_vector(img_arr, model, architecture='VGG16'):
#     if img_arr.shape[2] == 1:
#         img_arr = img_arr.repeat(3, axis=2)
#
#     arr4d = np.expand_dims(img_arr, axis=0)
#     if architecture == 'VGG16':
#         arr4d_pp = vgg16_preprocess_input(arr4d)
#     elif architecture == 'Xception':
#         arr4d_pp = xception_preprocess_input(arr4d)
#     elif architecture == 'VGG19':
#         arr4d_pp = vgg19_preprocess_input(arr4d)
#     elif architecture == 'NASNetLarge':
#         arr4d_pp = nasnet_preprocess_input(arr4d)
#
#     return model.predict(arr4d_pp)[0,:]

# def get_model():
#     return autoencoder.model.load_trained_encoder()
#
#
# def get_file(path, size):
#     img = cv2.resize(cv2.imread(path), size)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img.reshape(images_shape_x, images_shape_y, 1) / 255
#
#
def feature_vector(img_arr, model, architecture='VGG16'):
    features = model.predict(np.expand_dims(img_arr, axis=0)).reshape(-1)
    return features


def get_model():
    return autoencoder.model.load_trained_encoder()


# def get_file(path, size):
#     img = cv2.resize(cv2.imread(path), size)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img.reshape(-1) / 255


# def feature_vector(img_arr, model, architecture='VGG16'):
#     return model.predict(np.expand_dims(img_arr, axis=0))[0]


# def get_model():
#     return None


def get_file(path, size):
    if fourier_transform is not True:
        img = cv2.resize(cv2.imread(path), size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.reshape(images_shape_x, images_shape_y, 1) / 255
    if fourier_transform is True:
        img = cv2.resize(cv2.imread(path, 0), size)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        #plt.imshow(magnitude_spectrum)
        #plt.imshow(magnitude_spectrum, cmap='gray')
        #plt.show()

        return magnitude_spectrum.reshape(images_shape_x, images_shape_y, 1) / 255


        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # f = np.fft.fft2(img)
        # fshift = np.fft.fftshift(f)
        # #img = 20 * np.log(np.abs(fshift))
       # new_img = fshift.reshape(images_shape_x, images_shape_y, 1) / 255

        # print(np.shape(fshift))
        # print(type(fshift))
        # print(type(fshift[0]))
        # print(type(fshift[0][0]))
        # print(fshift)
        # print(np.shape(new_img))
        # print(type(new_img))
        # print(type(new_img[0]))
        # print(type(new_img[0][0]))
        # print(new_img)
        #
        #
        #
        # plt.imshow(fshift)
        # plt.show()
        #return new_img

        # f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # f_shift = np.fft.fftshift(f)
        # f_complex = f_shift[:, :, 0] + 1j * f_shift[:, :, 1]
        # f_abs = np.abs(f_complex) + 1  # lie between 1 and 1e6
        # f_bounded = 20 * np.log(f_abs)
        # f_img = 255 * f_bounded / np.max(f_bounded)
        # img = f_img.astype(np.uint8)
        # return img

        # img_float32 = np.float32(img)
        # dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        # dft_shift = np.fft.fftshift(dft)
        # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        # return magnitude_spectrum.reshape(images_shape_x, images_shape_y, 1) / 255



# def feature_vector(img_arr, model, architecture='VGG16'):
#     return img_arr


##### NASNetLarge ####### kmeans = MiniBatchKMeans(n_clusters=clusters, n_init=500, max_iter=9000, verbose=0, batch_size=batchSize)

# images_main_directory = base_path + 'data/selected8-easy/' # 1  path to train data direcotory
# extracted_features_directory = base_path + "NASNetLargeFeatures" # 2
# predicted_clusters_directory = base_path + "NASNetLargePredicted" # 4
# models_direcory = base_path + "models" # path to direcotory with models
# test_dataset_direcory = base_path + "selected8-easy/" # path to test data direcotory
# batchSize = 100 # batchSize > clusters && batchSize > selected_features_number # batches in k_means and PCA
# dim_min = 200 # scaling images
# dim_max = 500 # scaling images
# clusters = 8 # classes / clusters count
# selected_features_number = 100 # min(n_features_in, n_samples) using in PCA
# images_shape_x = 331 # VGG16/VGG19: 224, Xception: 299, NASNetLarge: 331
# images_shape_y = 331
# use_augmented_dataset = True
# pca_model_name = "NASNetLargePca.model"
# k_means_model_name ="NASNetLargeKmeans.model"
# model_architecture = "NASNetLarge"
# model_layer = "global_average_pooling2d_1"
# test_on_data = "train" # train or test