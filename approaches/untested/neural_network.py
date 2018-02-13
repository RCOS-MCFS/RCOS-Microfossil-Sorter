# RCOS-MCFS
# Though it may later assume some more important function, currently this file is just a testing 
# ground for different approaches to the mose base level of this project.

from random import shuffle

import copy
import cv2
import math
import numpy as np
import os
import random
import sys

# Shorthand for displaying images in a common way.
def show(img):
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the images into a list
def load_images(path):
    assert(os.path.isdir(path))
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.resize(img, (24, 24))
        if img is not None:
            images.append(img)
    return images

# Translates images in the specified folders into matricies representing the RGB values of those images. The last value
# in this each image's is the type label for the data in question, with (1, 0, 0) representing rock and (0, 1, 0) representing bone.
def created_labeled_matrices():
    # Checks that the correct arugments have been provided. 
    if len(sys.argv) < 3:
        print("ERROR: This program requires the following two folders as arguments:")
        print("\tThe location of the images of rocks to be used in testing.")
        print("\tThe location of the images of bone to be used in testing.")
        exit()

    images_rock = load_images(sys.argv[1])
    images_bone = load_images(sys.argv[2])

    # Checks that images dimensions are the same across both sets.
    sizes = set([np.shape(image) for image in images_rock] + [np.shape(image) for image in images_bone])
    assert(len(sizes) == 1)

    # We translate these images into a 1-D array to make it slightly easier with numpy, and
    # it allows us to add a column for labels.
    y, x, z = np.shape(images_rock[0])    
    contrast_img = copy.copy(images_rock[0])
    for i, image in enumerate(images_rock):
        images_rock[i] = np.reshape(image, (y*x*z))

    y, x, z = np.shape(images_bone[0])
    contrast_img = copy.copy(images_bone[0])
    for i, image in enumerate(images_bone):
        images_bone[i] = np.reshape(image, (y*x*z))

    rock_type = np.array([1, 0, 0])
    bone_type = np.array([0, 1, 0])
    labels = [rock_type for image in images_rock] + [bone_type for image in images_bone]

    full_labeled_image_set = images_rock + images_bone
    full_labeled_image_set = np.array([np.array(x) for x in full_labeled_image_set])
    return np.array(full_labeled_image_set), np.array(labels)

def distance(x, y):
    assert (np.shape(x) == np.shape(y))
    diffVec = x - y
    return math.sqrt(np.dot(diffVec, diffVec))

def sigmoid_logistic_function(net):
    if (-net > 100):
        return 0
    val = (1.0 / (1.0 + math.pow(math.e, -net)))
    if val > 0.9999:
        return 1
    if val < 0.0001:
        return 0
    return val

def neural_network(x, y, num_hidden, eta, epochs, activation_type):

    n, d = np.shape(x)
    random_i_order = list(range(n))
    num_classes = 3
    y_hat = np.zeros((n, num_classes))
    y_vector = y

    # All weights are initialized to random values between 0.1 and -0.1
    # Weights connecting original nodes to the hidden layer, with d+1 for the bias.

    # The weights go from each original node (and the bias) to each hidden node, so there are num_hidden*d+1 weights
    first_layer_weights = [[random.uniform(-0.1, 0.1) for i in range(num_hidden)] for j in range(d+1)]
    # The weights go from the hidden layer to the num_classes nodes for the response.
    hidden_layer_weights = [[random.uniform(-0.1, 0.1) for i in range(num_classes)] for j in range(num_hidden)]
    for e in range(epochs):
        shuffle(random_i_order)
        for i in random_i_order:
            first_layer_nodes = list(x[i])
            first_layer_nodes.append(1)
            hidden_layer_nodes = [sigmoid_logistic_function(sum([first_layer_nodes[i] * first_layer_weights[i][h]
                                         for i in range(d+1)])) for h in range(num_hidden)]
            final_layer_nodes = [sigmoid_logistic_function(sum([hidden_layer_nodes[i] * hidden_layer_weights[i][h]
                                        for i in range(num_hidden)])) for h in range(num_classes)]

            y_hat[i] = [0 for i in range(num_classes)]
            # The max value is set to 1
            y_hat[i][final_layer_nodes.index(max(final_layer_nodes))] = 1
            E = 0.5*math.pow(distance(y_hat[i], y_vector[i]), 2)
            if E > 0:
                # If E > 0, start the backprogopogation process
                # k is used instead of i as i is used in the entry to this loop.
                # Compute the lowercase_delta, used in both layers of backpropogation.
                lowercase_delta = []
                for j in range(len(y_vector[i])):
                    o_j = final_layer_nodes[j]
                    t_j = y_vector[i][j]
                    lowercase_delta.append((o_j - t_j) * o_j * (1-o_j))
                for k, w_i in enumerate(hidden_layer_weights):
                    for j, w_ij in enumerate(w_i):
                        o_i = hidden_layer_nodes[k]
                        capital_delta_w_ij = o_i * lowercase_delta[j]
                        new_w_ij = w_ij - eta*capital_delta_w_ij
                        hidden_layer_weights[k][j] = new_w_ij

                # Second layer of backpropogation
                lowercase_delta_2 = []
                for m in range(len(first_layer_nodes)):
                    for j in range(len(hidden_layer_nodes)):
                        o_j = hidden_layer_nodes[j]
                        lowercase_delta_2.append((o_j*(1-o_j)*sum([lowercase_delta[k] * hidden_layer_weights[j][k]
                                                                   for k in range(len(final_layer_nodes))])))
                for m, w_i in enumerate(first_layer_weights):
                    for j, w_ij in enumerate(w_i):
                        o_i = first_layer_nodes[m]
                        capital_delta_w_ij = o_i * lowercase_delta_2[j]
                        new_w_ij = w_ij - eta*capital_delta_w_ij
                        first_layer_weights[m][j] = new_w_ij
    return (first_layer_weights, hidden_layer_weights)

def test_neural_network_accuracy(x, y, first_layer_weights, hidden_layer_weights):
    n, d = np.shape(x)
    num_classes = len(y[0])
    y_hat = np.zeros((n, num_classes))
    y_vector = y
    num_hidden = len(hidden_layer_weights)
    for i, data in enumerate(x):
        first_layer_nodes = list(data)
        # append bias
        first_layer_nodes.append(1)
        hidden_layer_nodes = [sigmoid_logistic_function(sum([first_layer_nodes[i] * first_layer_weights[i][h]
                                                             for i in range(d + 1)])) for h in range(num_hidden)]

        final_layer_nodes = [sigmoid_logistic_function(sum([hidden_layer_nodes[i] * hidden_layer_weights[i][h]
                                                            for i in range(num_hidden)])) for h in range(num_classes)]
        y_hat[i] = [0 for i in range(num_classes)]
        # The max value is set to 1
        y_hat[i][final_layer_nodes.index(max(final_layer_nodes))] = 1

    accurate_count = 0
    for i, y in enumerate(y_vector):
        if list(y) == list(y_hat[i]):
            accurate_count += 1
    print("The accuracy is:")
    print(str((accurate_count/len(y_vector))*100) + "%")

x, y = created_labeled_matrices()
first_layer_weights, hidden_layer_weights = neural_network(x, y, 10, .05, 100, 'sigmoid')
test_neural_network_accuracy(x, y, first_layer_weights, hidden_layer_weights)


