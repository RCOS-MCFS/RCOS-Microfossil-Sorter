# A collection of common functions used in the creation and manipulation of the data for this project.
from approaches.approach import Multiclass_Logistic_Regression, Perceptron, Sklearn_SVM

import comp_vis.img_tools as it
import numpy as np
import sys


def images_to_data(images, label, already_cropped=True):
    '''
    # TODO: Should this be fixed or modular? I think just returning the most successful arrangement of data from our tets is best, but who knows!
    :param images: A list of OpenCV images
    :param already_cropped: if these images have already been cropped. If not, they will be.
    :param label: An integer label for the data.
    :return: A numpy matrix of the data, with the label if any.
    '''

    # Crop the images appropriately
    cropped_images = []
    if already_cropped:
        cropped_images = images
    else:
        for image in images:
            cropped_img = it.crop_image(image)
            if np.shape(cropped_img) != np.shape(image):
                cropped_images += [cropped_img]
        if len(cropped_images) == 0:
            sys.stderr.write('Error: No objects detected in images.\n(Did you mistakenly set already_cropped to false?)')

    img_avgs = [it.average_color(image) for image in cropped_images]
    img_dims = it.get_images_dimensions(cropped_images, ordered=True)

    data = [img_avgs[i] + list(img_dims[i]) + [label] for i in range(len(images))]
    data = np.array([np.array(x) for x in data])
    return data


def string_to_model(approach_name):
    '''
    :param approach_name: The string name of the model to be returned
    :return: The model (subset of the Approach class) with a name matching approach_name
    :raises ValueError: Raises if approach_name not recognized
    '''
    if approach_name == "perceptron":
        return Perceptron()
    elif approach_name == "multiclass":
        return  Multiclass_Logistic_Regression()
    elif approach_name == "sklearn_svm":
        return Sklearn_SVM()
    else:
        raise ValueError('Model type ' + approach_name + ' not recognized.')


def training_and_testing_sep(data, training_fraction):
    '''
    :param data: Numpy matrix of data
    :param training_fraction: Float between 0.00 and 1.00 denoting the size of the training set
    :return: A training and testing set.
    '''

    # Randomly shuffle the data
    np.random.shuffle(data)
    training_size = int(training_fraction*np.shape(data)[0])
    training_data, testing_data = data[0:training_size], data[training_size:]

    return training_data, testing_data

