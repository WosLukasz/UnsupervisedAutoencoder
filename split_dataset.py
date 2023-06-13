import os
import random
import shutil
from sklearn.model_selection import train_test_split
import copyUtils.copyDirectoryUtils as copy_utils

main_data_directory = "C:/Projekty/UnsupervisedCNN/data/selected8-easy-augmented2"
train_size = 0.9
test_size = 0.1
dirs = os.listdir(main_data_directory)

print("[Start] Spliting into train and test datasets...")
for dir in dirs:
    print("[Start] Start processing dir:" + dir.__str__())
    source_dir_path = os.path.join(main_data_directory, dir).__str__()
    category_file_names = copy_utils.get_file_names(source_dir_path)

    train_directory_path = os.path.join(source_dir_path, "train").__str__()
    test_directory_path = os.path.join(source_dir_path, "test").__str__()

    if not os.path.exists(train_directory_path):
        os.makedirs(train_directory_path)
    if not os.path.exists(test_directory_path):
        os.makedirs(test_directory_path)

    random.shuffle(category_file_names)
    training_dataset_files, test_dataset_files = train_test_split(category_file_names, train_size=train_size, test_size=test_size)

    for file in training_dataset_files:
        shutil.move(os.path.join(source_dir_path, file).__str__(), os.path.join(train_directory_path, file).__str__())

    for file in test_dataset_files:
        shutil.move(os.path.join(source_dir_path, file).__str__(), os.path.join(test_directory_path, file).__str__())


print("[Stop] Spliting into train and test datasets...")