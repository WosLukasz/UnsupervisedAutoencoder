import copyUtils.copyDirectoryUtils as copy_utils
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False, nargs='?', default=".\\dataset",	help="path to source folder")
ap.add_argument("-d", "--destination", required=False, nargs='?', default=".\\augmented_dataset",	help="path to destination folder")
ap.add_argument("-m", "--min", required=False, default="64",	help="min ratio")
ap.add_argument("-M", "--max", required=False, default="128",	help="max ratio")
ap.add_argument("-f", "--force", required=False, action='store_true', help="overwrites the dataset")
args = vars(ap.parse_args())

#copy_utils.copy_directory(args["source"], args["destination"], int(args["min"]), int(args["max"]), args["force"])

source = 'D:/dataset/selected15-easy'
destination = 'D:/dataset/destination'
min = '64'
max = '128'
force = True

copy_utils.copy_directory(source, destination, int(min), int(max), force)



