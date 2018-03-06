from approaches.approach import Approach
from approaches.multiclass_logistic_regression import Multiclass_Logistic_Regression
from approaches.perceptron import Perceptron
from approaches.sklearn_svm import Sklearn_SVM

import comp_vis.img_tools as it
import numpy as np
import random
import sys


def generate_avg_color_data():
    if len(sys.argv) < 3:
        raise ValueError("Arguments must specify directories for bone images and rock images.\n" +
                         "\t(i.e. python3 example.py sample_images/multi_images/multi_bones sample_images/multi_images/multi_rocks)")
    # Dictates how the training and testing data will be split
    training_percentage = 0.50

    print("Pulling and cropping bone images...")
    path_bones = sys.argv[1]
    multi_images_bones = it.load_images(path_bones)
    images_bones = []
    for img in multi_images_bones:
        images_bones += it.generate_cropped_from_multi(img)

    print("Pulling and cropping rock images...")
    path_rocks = sys.argv[2]
    multi_images_rocks = it.load_images(path_rocks)
    images_rocks = []
    for img in multi_images_rocks:
        images_rocks += it.generate_cropped_from_multi(img)

    print("Discovered " + str(len(images_bones)) + " bones and " + str(len(images_rocks)) + " rocks.")

    # Combine these two sets of images into a matrix of their average color, with the last value representing
    # If it is a bone or a rock, 1 and 0 respectively.
    average_colors = []
    for image in images_bones:
        average_colors.append(it.average_color(image) + [1])
    for image in images_rocks:
        average_colors.append(it.average_color(image) + [0])

    # Select training set from this matix.
    training_set_size = int(np.shape(average_colors)[0] * training_percentage)

    train_set = random.sample(average_colors, training_set_size)
    test_set = np.array(average_colors)
    train_set = np.array(train_set)

    return train_set, test_set

def main():
    training_data, testing_data = generate_avg_color_data()

    perceptron = Approach(Perceptron)
    perceptron.train(training_data)
    print(perceptron.assess_accuracy(testing_data))

    multiclass = Approach(Multiclass_Logistic_Regression)
    multiclass.train(training_data)
    print(multiclass.assess_accuracy(testing_data))

if __name__ == "__main__":
    main()