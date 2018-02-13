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
        img = cv2.resize(img, (16, 16))
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

x, y = created_labeled_matrices()



