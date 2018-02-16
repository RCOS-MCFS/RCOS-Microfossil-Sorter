from approaches.perceptron import create_perceptron, test_accuracy
import numpy as np
import comp_vis.img_tools as it
import random
import sys


def perceptron_example(training_percentage):
    '''
    :param training_percentage: Float representing the percentage of the total set to be used for training.
    :return: None
    '''
    assert(training_percentage <= 1.00 and training_percentage > 0.00)

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

    training_set = random.sample(average_colors, training_set_size)
    test_set = np.array(average_colors)
    training_set = np.array(training_set)

    weights = create_perceptron(training_set)
    print(test_accuracy(weights, test_set))


def main():
    if len(sys.argv) < 3:
        print("ERROR: At command line arguments required.")
        print(
            "\t(i.e. python3 example.py sample_images/multi_images/multi_bones sample_images/multi_images/multi_rocks)")
        exit()
    perceptron_example(.50)

main()