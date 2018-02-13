
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
        if img is not None:
            images.append(img)
    return images

# Takes in an image containing multiple subjects and returns a list of each individual image.
# Gaus describes the intensity of the blur.
# min_crop_size is the smallest image that could be cropped
def generate_images_from_multi(img, gaus=25, min_crop_size = 7):
    # Apply gaussian blur to help later remove the background and keep
    # background noise from forming edges in the later edgemap.
    blur = cv2.GaussianBlur(img,(gaus,gaus),0)
    edges = cv2.Canny(blur, 50, 50)
    # Break the edgemap into columns. Save the y coordinates which delinate these rows.
    y, x = np.shape(edges)
    blank_xs = []
    for pos_x in range(x):
        if sum(edges[:,pos_x]) == 0: blank_xs.append(pos_x)
    x_coords = []
    for i in range(1, len(blank_xs)):
        if blank_xs[i]-blank_xs[i-1] > min_crop_size:
            x_coords.append((blank_xs[i-1], blank_xs[i]+1))
    x_column_images_edges = [edges[:, x_1:x_2] for x_1, x_2 in x_coords]
    x_column_images = [img[:, x_1:x_2] for x_1, x_2 in x_coords]
    # Preform the same operation on these columns to form more cropped images.
    y_cropped_images_edges = []
    y_cropped_images = []
    for j, col_img in enumerate(x_column_images_edges):
        blank_ys = []
        for pos_y in range(y):
            if sum(col_img[pos_y]) == 0: blank_ys.append(pos_y)
        y_coords = []
        for i in range(1, len(blank_ys)):
            if blank_ys[i]-blank_ys[i-1] > min_crop_size:
                y_coords.append((blank_ys[i-1], blank_ys[i]))
        for y_1, y_2 in y_coords:
            y_cropped_images.append(x_column_images[j][y_1:y_2])
            y_cropped_images_edges.append(col_img[y_1:y_2])
    cropped_images = []
    # And one final pass.
    for k, y_crop_img_edges in enumerate(y_cropped_images_edges):
        y, x = np.shape(y_crop_img_edges)
        blank_xs = []
        for pos_x in range(x):
            if sum(y_crop_img_edges[:,pos_x]) == 0: blank_xs.append(pos_x)
        blank_xs.append(x)
        x_coords = []
        for i in range(1, len(blank_xs)):
            if blank_xs[i]-blank_xs[i-1] > min_crop_size:
                x_coords.append((blank_xs[i-1], blank_xs[i]))
        cropped_images += [y_cropped_images[k][:,x_1:x_2] for x_1, x_2 in x_coords]
    return cropped_images

def main():
    if len(sys.argv) < 2:
        print("ERROR: Must provide a path to the folder containing images to be split.")
        exit()
    path = sys.argv[1]
    images = load_images(path)
    images = generate_images_from_multi(images[0])
    for img in images:
        show(img)

main()