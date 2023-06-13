import os
import cv2
import numpy as np
import params
from normalization.resizer_to_boundary import resize_old as resize_old
import shutil


def get_file_names(path):
    file_names = os.listdir(path)
    return list(filter(lambda x: x.endswith(('.png', '.jpg')), file_names))


def is_system_file(filename):
    return filename.endswith('.db')


def copy_directory(source_directory, destination_directory, dim_min, dim_max, force=False):
    if not os.path.isdir(destination_directory) or force:
        if os.path.isdir(destination_directory) and force:
            shutil.rmtree(destination_directory)
        print("[Start] Copy dataset...")
        original_dirs = os.listdir(source_directory)
        os.mkdir(destination_directory)
        for dir in original_dirs:
            new_dir_path =  os.path.join(destination_directory, dir).__str__()
            os.mkdir(new_dir_path)
            original_dir_path = os.path.join(source_directory, dir).__str__()
            new_file_names = get_file_names(original_dir_path)
            for new_file_name in new_file_names:
                img = cv2.imread(os.path.join(original_dir_path, new_file_name).__str__())
                new_image = resize_old(img, dim_min, dim_max)
                cv2.imwrite(os.path.join(new_dir_path, new_file_name).__str__(), new_image)
        print("[Stop] Copy dataset...")
    else:
        print("[INFO] Dataset already exist. Skipped.")


def create_directory_if_not_exists(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print('Error: Creating directory of data')


def copy_file(src_path, dest_path):
    shutil.copy(src_path, dest_path)


def save_predicted_clusters(main_out_data_directory, test_predicted, image_array):
    create_directory_if_not_exists(main_out_data_directory)
    for i in range(0, len(test_predicted)):
        file_path = image_array[i]
        label = test_predicted[i]
        out_class_directory = os.path.join(main_out_data_directory, str(label)).__str__()
        create_directory_if_not_exists(out_class_directory)
        copy_file(file_path, out_class_directory)


def save_predicted_clusters_mnist(main_out_data_directory, test_predicted, image_array):
    image_array = np.array(image_array, dtype=np.uint8)
    image_array = image_array.reshape((-1, 28, 28))
    create_directory_if_not_exists(main_out_data_directory)
    for i in range(0, len(test_predicted)):
        file = image_array[i]
        label = test_predicted[i]
        out_class_directory = os.path.join(main_out_data_directory, str(label)).__str__()
        create_directory_if_not_exists(out_class_directory)
        cv2.imwrite(os.path.join(out_class_directory, i.__str__() + '.jpg').__str__(), file)


def get_images_list(path_to_files):
    print("[Start] Start getting data paths...")
    dirs = os.listdir(path_to_files)
    train_images_array = []
    train_images_category_array = []

    for dir in dirs:
        if is_system_file(dir):
            continue
        print("Getting data paths from class " + dir.__str__())
        source_dir_path = os.path.join(path_to_files, dir).__str__()
        train_directory_path = os.path.join(source_dir_path, "train").__str__()
        train_directory_file_names = get_file_names(train_directory_path)

        for file in train_directory_file_names:
            path_to_file = os.path.join(train_directory_path, file).__str__()
            train_images_array.append(path_to_file)
            train_images_category_array.append(dir.__str__())

    print("[Stop] Stop getting data paths...")

    return train_images_array, train_images_category_array


def get_files(train_images_array, size):
    fn_imgs = []
    for file in train_images_array:
        img = params.get_file(file, size)
        fn_imgs.append([file, img])
    return dict(fn_imgs)




