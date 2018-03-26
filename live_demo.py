# For live classification
from approaches.approach import Multiclass_Logistic_Regression, Perceptron, Sklearn_SVM

import cv2
import comp_vis.img_tools as it
import os
import random
import sys


# Check the correct number of arguments have been provided
if len(sys.argv) < 3 or len(sys.argv) > 5:
    sys.stderr.write("Error: Invalid argument configuration.\n" +
                     "Possible argument configurations examples: \n" +
                     "live_demo.py perceptron [camera_num] [bones_folder_path] [rocks_folder_path]\n" +
                     "live_demo.py perceptron [camera_num] [location to previous weights]")
    exit()

# The first command line argument describes the ML approach to be used.
approach_name = sys.argv[2]
if approach_name == "perceptron":
    model = Perceptron()
elif approach_name == "multiclass":
    model = Multiclass_Logistic_Regression()
elif approach_name == "sklearn_svm":
    model = Multiclass_Logistic_Regression()
else:
    sys.stderr.write("Error: Invalid model name " + sys.argv[2] + " given.")
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

    # Crop the images appropriately
    bone_images = []
    for img in bone_images_whole:
        bone_images += it.crop_image_all(img)
    rock_images = []
    for img in rock_images_whole:
        rock_images += it.crop_image_all(img)
    if len(bone_images) < 30 or len(rock_images) < 30:
        sys.stderr.write("Warning: Only " + str(len(bone_images)) + " bones and " + str(len(rock_images)) +
                         "rocks detected in given folders. Try gathering more training data to improve accuracy.")

    # Get average colors for these images
    bone_avgs = [it.average_color(img) for img in bone_images]
    rock_avgs = [it.average.color(img) for img in rock_images]

    # Add on the size of these objects. Might be unnecessary down the road, but presently it does increase accuracy.
    bone_dims = it.get_images_dimensions(bone_images, ordered=True)
    rock_dims = it.get_images_dimensions(rock_images, ordered=True)
    assert(len(bone_dims) == len(bone_avgs) and len(rock_dims) == len(rock_avgs))
    # We will also label these values as long as we've got them here.
    bone_data = [bone_avgs[i] + bone_dims[i] + [1] for i in range(len(bone_images))]
    rock_data = [rock_avgs[i] + rock_dims[i] + [0] for i in range(len(rock_images))]




else:
    sys.stderr.write("Error: Too many command-line arguments given.\n" 
                     "Possible argument configurations examples: \n" +
                     "live_demo.py perceptron [Camera_num] [bones_folder_path] [rocks_folder_path]\n" +
                     "live_demo.py perceptron [Camera_num] [location to previous weights]")

# Variables used in output styling later on
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (2, 400)
fontScale = 1
lineType = 2
color_bad = (0, 0, 255)
color_good = (0, 255, 0)

capture = cv2.VideoCapture(0)

display_text = ""
display_color = (0, 0, 0)

while True:
    ret, frame = capture.read()
    # Todo: Analysis
    if it.contains_object(frame):
        _ = 1
    else:
        display_text = "No object detected"
        display_color = color_bad
    cv2.putText(frame, display_text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                display_color,
                lineType)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()