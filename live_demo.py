# For live classification
from approaches.approach import Multiclass_Logistic_Regression, Perceptron, Sklearn_SVM

import cv2
import comp_vis.img_tools as it
import os
import sys

print(sys.argv)

# Check the correct number of arguments have been provided
if len(sys.argv) < 2 or len(sys.argv) > 4:
    sys.stderr.write("Error: Invalid argument configuration.\n" +
                     "Possible argument configurations examples: \n" +
                     "live_demo.py perceptron [bones_folder_path] [rocks_folder_path]\n" +
                     "live_demo.py perceptron [location to previous weights]")
    exit()

# The first command line argument describes the ML approach to be used.
approach_name = sys.argv[1]
if approach_name == "perceptron":
    model = Perceptron()
elif approach_name == "multiclass":
    model = Multiclass_Logistic_Regression()
elif approach_name == "sklearn_svm":
    model = Multiclass_Logistic_Regression()

# If there are two additional arguments, then these arguments should be a folder containing images of rocks,
# as well as a folder containing images of bones to be used in the training process.
if len(sys.argv) == 3:
    # Then load weight from location
elif len(sys.argv) == 4:
    # Then load the models and train o these models.

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
    # Todo: Analaysis
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