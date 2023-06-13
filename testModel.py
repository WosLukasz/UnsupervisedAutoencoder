import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import copyUtils.copyDirectoryUtils as copy_utils
import modelUtils.modelUtils as modelUtils
import params as params
from utils.gpu import gpu_fix

from keras.datasets import mnist

gpu_fix()

models_direcory = params.models_direcory
predicted_clusters_directory = params.predicted_clusters_directory
pca_path = os.path.join(models_direcory, params.pca_model_name).__str__()
kmeans_path = os.path.join(models_direcory, params.k_means_model_name).__str__()

model = params.get_model()
pca = joblib.load(pca_path)
k_means = joblib.load(kmeans_path)
images_shape_x = params.images_shape_x
images_shape_y = params.images_shape_y


test_dataset_direcory = params.test_dataset_direcory
dirs = os.listdir(test_dataset_direcory)

test_images_array = []
test_features = []
test_images_category_array = []

(x_val, test_images_category_array), (xxx, yyyy) = mnist.load_data()
x_val = np.expand_dims(x_val, axis=3)

for file in x_val:
    features = params.feature_vector(file, model, architecture=params.model_architecture)
    test_features.append(features)


# for dir in dirs:
#     if copy_utils.is_system_file(dir):
#         continue
#     print("Getting data paths from class " + dir.__str__())
#     source_dir_path = os.path.join(test_dataset_direcory, dir).__str__()
#     test_directory_file_names = copy_utils.get_file_names(source_dir_path)
#
#     test_directory_path = os.path.join(source_dir_path, params.test_on_data).__str__()
#     test_directory_file_names = copy_utils.get_file_names(test_directory_path)
#
#
#     for file in test_directory_file_names:
#         path_to_file = os.path.join(test_directory_path, file).__str__()
#         test_images_array.append(path_to_file)
#         test_images_category_array.append(dir.__str__())
#         img = params.get_file(path_to_file, (images_shape_x, images_shape_y))
#         features = params.feature_vector(img, model, architecture=params.model_architecture)
#         test_features.append(features)


test_features = np.array(test_features)
test_features_selected = pca.transform(test_features)
lab = k_means.labels_
test_predict = k_means.predict(test_features_selected)

# print(test_images_category_array)
# print(test_predict)

print("[-1, 1] (Best 1) Adjusted Rand index: ", metrics.adjusted_rand_score(test_images_category_array, test_predict))
print("[-1, 1] (Best 1) Mutual Information based scores: ", metrics.adjusted_mutual_info_score(test_images_category_array, test_predict))
print("[-1, 1] (Best 1) V-measure: ", metrics.v_measure_score(test_images_category_array, test_predict))
# print("[-1, 1] (Best 1) Silhouette Coefficient: ", metrics.silhouette_score(test_features_selected, lab, metric='euclidean'))

#copy_utils.save_predicted_clusters(predicted_clusters_directory, test_predict, test_images_array)

numeric_categories = modelUtils.convert_to_numeric_labels(test_images_category_array)

sns.set(font_scale=3)
confusion_matrix = confusion_matrix(numeric_categories, test_predict)

plt.figure(figsize=(16, 14))
# sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
# plt.title("Confusion matrix", fontsize=30)
# plt.ylabel('True label', fontsize=25)
# plt.xlabel('Clustering label', fontsize=25)
# plt.savefig(os.path.join(predicted_clusters_directory, 'conf_matrix.png').__str__())
# # plt.show()