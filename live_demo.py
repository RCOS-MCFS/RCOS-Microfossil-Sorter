# For live classification
from approaches.approach import Multiclass_Logistic_Regression, Perceptron, Sklearn_SVM

import comp_vis.calibration_tools as ct
import comp_vis.data_tools as dt
import comp_vis.img_tools as it
import comp_vis.live_tools as lt
import numpy as np
import os
import random
import sys
import time

already_cropped = True

# Check the correct number of arguments have been provided
if len(sys.argv) < 3 or len(sys.argv) > 5:
    sys.stderr.write("Error: Invalid argument configuration.\n" +
                     "Possible argument configurations examples: \n" +
                     "live_demo.py perceptron [camera_num] [bones_folder_path] [rocks_folder_path]\n" +
                     "live_demo.py perceptron [camera_num] [location to previous weights]")
    exit()

# The first command line argument describes the ML approach to be used (ex: perceptron)
# The second argument (int) specifies which camera to use.

approach_name = sys.argv[1]
camera_num = int(sys.argv[2])

try:
    model = dt.string_to_model(approach_name)
except ValueError:
    exit()

# If there are two additional arguments, then these arguments should be a folder containing images of rocks,
# as well as a folder containing images of bones to be used in the training process.
if len(sys.argv) == 4:
    # Then load weight from location
    if not os.path.isfile(sys.argv[3]):
        sys.stderr.write("Error: No file found at path " + sys.argv[3])
        exit()
    model.load_weights(sys.argv[4])
elif len(sys.argv) == 5:
    # Load bones and rocks
    bones_path = sys.argv[3]
    rocks_path = sys.argv[4]

    if not os.path.exists(bones_path):
        sys.stderr.write("Error: Invalid path " + bones_path)
        exit()
    if not os.path.exists(rocks_path):
        sys.stderr.write("Error: Invalid path " + rocks_path)
        exit()

    # Load images from selected path
    bone_images_whole = it.load_images(bones_path)
    rock_images_whole = it.load_images(rocks_path)

    # Translate images to the key data we'll train on
    bone_data = dt.images_to_data(bone_images_whole, 1)
    rock_data = dt.images_to_data(rock_images_whole, 0)

    # Combine datasets
    data = np.append(bone_data, rock_data, 0)

    training_data, testing_data = dt.training_and_testing_sep(data, 0.50)

    model.train(training_data)

    print("Accuracy on loaded data: " + str(model.assess_accuracy(testing_data)))

thresh_s = ct.calibrate_thressholding()

lt.live_labeling(model, camera_no=camera_num, threshold_settings=thresh_s)
