import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
import copyUtils.copyDirectoryUtils as copy_utils
import modelUtils.modelUtils as modelUtils
from sklearn.utils import shuffle
import joblib
import params as params
from utils.gpu import gpu_fix

from keras.datasets import mnist

path_to_training_files = params.images_main_directory
images_shape_x = params.images_shape_x
images_shape_y = params.images_shape_y
batch_size = params.batchSize
feature_vector_length = params.selected_features_number
main_data_out_directory = params.models_direcory
pca_model_path = os.path.join(main_data_out_directory, params.pca_model_name).__str__()

gpu_fix()

# Preparing arrays with images paths and images categories
# train_images_array, _ = copy_utils.get_images_list(path_to_training_files)
# train_images_array = shuffle(np.array(train_images_array))

(train_images_array, train_y), (x_val, test_y) = mnist.load_data()
train_images_array = np.expand_dims(train_images_array, axis=3)


print("[Start] Start Initializing models...")
# Get CNN model with proper output layer
model = params.get_model()
pca = IncrementalPCA(n_components=feature_vector_length)  # n_components < images.length &&

print("[Stop] Stop Initializing models...")

rest = len(train_images_array) % batch_size
epochs = int(len(train_images_array) / batch_size)
if rest != 0:
    epochs = epochs + 1
features_vectors = []

print("[Start] Start getting features...")
for i in range(epochs):
    print("Batch [" + str(i) + "/" + str(epochs - 1) + "]")
    start = i * batch_size
    end = start + batch_size
    if len(train_images_array) < end:
        end = len(train_images_array)

    imgs_dict = train_images_array[start:end]
    #imgs_dict = copy_utils.get_files(train_images_array=bath_images, size=(images_shape_x, images_shape_y))
    #img_feature_vector = modelUtils.feature_vectors(imgs_dict, model, architecture=params.model_architecture)
    ii = 0
    img_feature_vector = {}
    for img in imgs_dict:
        img_feature_vector[ii] = params.feature_vector(img, model, params.model_architecture)
        ii = ii + 1



    images = list(img_feature_vector.values())
    if len(features_vectors) == 0:
        features_vectors = images
    else:
        features_vectors = np.concatenate((features_vectors, images), axis=0)
    if np.shape(images)[0] < feature_vector_length:  # jest na to bug zgloszony: https://github.com/scikit-learn/scikit-learn/issues/12234 pomyslec moze nad czyms innym
        print("Batch is to small " + np.shape(images)[0].__str__())
        continue
    pca.partial_fit(images)

print("[Stop] Stop getting features...")

copy_utils.create_directory_if_not_exists(main_data_out_directory)
copy_utils.create_directory_if_not_exists(params.extracted_features_directory)

joblib.dump(pca, pca_model_path)

print("[Start] Start selecting features...")
for i in range(epochs):
    print("Batch [" + str(i) + "/" + str(epochs - 1) + "]")
    start = i * batch_size
    end = start + batch_size
    if len(train_images_array) < end:
        end = len(train_images_array)

    bath_vectors = features_vectors[start:end]
    vectors_transformed = pca.transform(bath_vectors)

    filename = i.__str__() + '.pkl'
    file_path = os.path.join(params.extracted_features_directory, filename).__str__()
    joblib.dump(vectors_transformed, file_path)

print("[Stop] Stop selecting features...")