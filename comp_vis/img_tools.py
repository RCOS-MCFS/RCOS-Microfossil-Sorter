# Tools for the analysis, loading, and alteration of images.

import cv2
import numpy as np
import os
import random

def average_color(image):
    '''
    :param image: Image to be analyzed
    :return: A list containing the averages for each color channel.
    '''
    return [image[:, :, i].mean() for i in range(image.shape[-1])]


def crop_image_all(img, gaus=25, min_crop_size=7):
    '''
    :param img: Numpy matrix representing the image to be broken into cropped images.
    :param gaus: The level of gaussian blur, used to reduce noise in edges.
    :param min_crop_size: Smallest possible image size. Used to prevent edge noise form being cropped as its own image.
    :return: A list containing the cropped images from the source image.
    '''
    # Apply gaussian blur to help later remove the background and keep
    # background noise from forming edges in the later edgemap.
    blur = cv2.GaussianBlur(img, (gaus, gaus), 0)
    edges = cv2.Canny(blur, 50, 50)
    # Break the edgemap into columns. Save the y coordinates which delinate these rows.
    y, x = np.shape(edges)
    blank_xs = []
    for pos_x in range(x):
        if sum(edges[:, pos_x]) == 0: blank_xs.append(pos_x)
    x_coords = []
    for i in range(1, len(blank_xs)):
        if blank_xs[i] - blank_xs[i - 1] > min_crop_size:
            x_coords.append((blank_xs[i - 1], blank_xs[i] + 1))
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
            if blank_ys[i] - blank_ys[i - 1] > min_crop_size:
                y_coords.append((blank_ys[i - 1], blank_ys[i]))
        for y_1, y_2 in y_coords:
            y_cropped_images.append(x_column_images[j][y_1:y_2])
            y_cropped_images_edges.append(col_img[y_1:y_2])
    cropped_images = []
    # And one final pass.
    for k, y_crop_img_edges in enumerate(y_cropped_images_edges):
        y, x = np.shape(y_crop_img_edges)
        blank_xs = []
        for pos_x in range(x):
            if sum(y_crop_img_edges[:, pos_x]) == 0: blank_xs.append(pos_x)
        blank_xs.append(x)
        x_coords = []
        for i in range(1, len(blank_xs)):
            if blank_xs[i] - blank_xs[i - 1] > min_crop_size:
                x_coords.append((blank_xs[i - 1], blank_xs[i]))
        cropped_images += [y_cropped_images[k][:, x_1:x_2] for x_1, x_2 in x_coords]
    return cropped_images

def crop_image_single(image, gaus=25, min_crop_size=7):
    '''
    The above function is meant for the cropping of multiple objects from a single photo
    This is for refining a single image.

    If no images are detected, or too many are detected, returns type None, which
    should be interpreted as an error by the receiving function.
    '''
    crops = crop_image_all(image, gaus, min_crop_size)
    if len(crops) == 0:
        print("ERROR: No cropable area found in image. Try adjusting parameters.")
        return None
    elif len(crops) > 1:
        print("ERROR: Too many images found in image. Try adjusting parameters.")
        return None
    else:
        return crops[0]

def get_images_dimensions(images, normalized=False, ordered=False):
    '''
    :param images: List of images for which the dimensions are returned
    :param normalized: If true, returns the dimensions unordered as a ratio of one to the other, i.e. a square would be
                     (1, 1), whilst a rectangle with the wide edge twice the length of the short would be (.5, 2).
                     If working with objects of disparate sizes in which oblong-ness is an important feature, this might
                     be a valuable piece of data
    :param ordered: Returns the larger of the two values first, such that the same rectangle would return the same
                    dimensions even if it was on it's side or standing up.
    :return: A list of tuples representing the height and width dimensions. (Z values, if present, are ignored.)
    '''
    ret_list = []
    for image in images:
        a, b = np.shape(image)[0:2]
        if ordered:
            a, b = max(a, b), min(a, b)
        if normalized:
            a, b = a/b, b/a
        ret_list.append((a, b))
    return ret_list

def normalize_image_sizes(images):
    '''
    Normalizes the size of images, potentially useful for certain machine learning functions which mandate the
    same size for all input vectors.
    :param images: A vector containing multiple images
    :return: A vector of the same images, but with the all the same size with black bars being added to the smaller
            images evenly on both sides in order to bring them up to the desired size.
    '''
    new_size = max(np.shape(image) for image in images)
    print(new_size)
    # TODO: Complete function

def load_images(path):
    '''
    Load the images contained within the folder path into a list of numpy matrices representing these images.
    :param path: Path to the folder containing images.
    :return: A list containing the loaded images.
    '''
    assert (os.path.isdir(path))
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images

def show(img, title='frame'):
    '''
    Shorthand for displaying images in a common way.
    :param img: Image to be displayed
    :return: None
    '''
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def take_image():
    '''
    So short it might not even be kept, just takes a picture! Probably not going to be used in actuality, since
    there is a computational cost associated with the creation and release of captures, but for testing is decent.
    :return: Returns an image taken with the primary webcam(if present) from the main camera.
    '''
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    capture.release()
    return frame