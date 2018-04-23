# Tools for the analysis, loading, and alteration of images.

import cv2
import numpy as np
import os
import sys

def average_color(image):
    '''
    :param image: Image to be analyzed
    :return: A list containing the averages for each color channel.
    '''
    return [image[:, :, i].mean() for i in range(image.shape[-1])]

def coordinates_from_contour(contour):
    '''
    :param contour: the contour to be analyzed
    :return: A tuple containing two tuples, for the top-left and bottom-right of the area covered by the contour.
    '''
    top_left = ((min([point[0][0] for point in contour])), (min([point[0][1] for point in contour])))
    bottom_right = ((max([point[0][0] for point in contour])), (max([point[0][1] for point in contour])))
    return top_left, bottom_right

def crop_image(image):
    '''
    :param image: Image to be cropped
    :return: Return a cropped image of the largest object in this image, or the whole image if no object detected.
    This is a bit redundant, but it's here for clarity in the demo functions.
    '''
    _, contour = get_largest_object(image)
    if contour is not None:
        return crop_to_contour(image, contour)
    return image

def crop_image_from_edges(img, gaus=25, min_crop_size=7):
    # TODO: Remove, deprecated function still used in some older tests or demonstrations, kept for posterity.
    '''
    :param img: Numpy matrix representing the image to be broken into cropped images.
    :param gaus: The level of gaussian blur, used to reduce noise in edges.
    :param min_crop_size: Smallest possible image size. Used to prevent edge noise form being cropped as its own image.
    :return: A list containing the cropped images from the source image.
    '''
    # Print warning about the deprecated nature of this function
    sys.stderr.write("Warning: This functino has been deprecated and will be removed in upcoming iterations. Please"
                     "update function referencing this accordingly.")

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

def crop_to_contour(img, contour):
    # TODO: REDO SO THAT IT ONLY HAS THE CONTENTS OF THE CONTOUR, RATHER THAN RECTANGLE AROUND IT
    coordinates = coordinates_from_contour(contour)
    cropped = img[coordinates[0][1]:coordinates[1][1], coordinates[0][0]:coordinates[1][0]]
    return cropped

def get_edges(img, gaus=25):
    '''
    DEPREECATED
    '''
    blur = cv2.GaussianBlur(img, (gaus, gaus), 0)
    edges = cv2.Canny(blur, 50, 50)
    return edges

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
    if type(images) != type([1,2,3]):
        return get_images_dimensions(list([images]), normalized, ordered)[0]

    ret_list = []
    for image in images:
        a, b = np.shape(image)[0:2]
        if ordered:
            a, b = max(a, b), min(a, b)
        if normalized:
            a, b = a/b, b/a
        ret_list.append((a, b))
    return ret_list

def get_largest_object(img, discount_out_of_bounds=True, min_contour_area = 750, threshhold_settings=(0, 128, 1, 2)):
    '''
    :param img: An RGB OpenCV image.
    :param discount_out_of_bounds: If true, objects touching the edges of the image are ignored. This setting is
                                   usually on, as it can keep lighting gradients or uneven backgrounds from being
                                   detected as an object.
    :param min_contour_area: The smallest area of a contour we'll accept, keeping small imperfections in the background
                             material or minor artifacts due to lighting condition from registering as objects.
    :param threshhold_settings: A tuple containing upper_limit, erode_iter, and kernel_size to be used in thressholding.
    :return:
    '''
    lower_limit, upper_limit, erode_iter, kernel_size = threshhold_settings
    thresh = threshhold(img, lower_limit, upper_limit, erode_iter, kernel_size)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x, y, _ = np.shape(img)

    def touches_edge(contour):
        '''
        :param contour: Contour to examined
        :return: True if the contour touches the edge fo the image, false otherwise
        '''
        coordinates = coordinates_from_contour(contour)
        return 0 in coordinates[0] or coordinates[1][0] == x or coordinates[1][1] == y

    if discount_out_of_bounds:
        contours = [contour for contour in contours if not touches_edge(contour)]

    if contours:
        max_area = np.shape(img)[0]*np.shape(img)[1]
        max_area *= .9
        # Get the areas for all the contours, and label them with the contour they relate to
        areas = [(cv2.contourArea(contour), i) for i, contour in enumerate(contours)]
        # Weed out those areas small enough to be meaningless noise or large enough to be glitches
        areas = [area for area in areas if area[0] < max_area and area[0] > min_contour_area]
        if areas:
            largest_contour = contours[max(areas)[1]]
            return thresh, largest_contour

    return thresh, None

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

def save_images(images, path):
    '''
    :param path: Folder to which the images in the list will be saved.
    :param images:
    :return: None
    '''
    if not os.path.isdir(path):
        sys.stderr.write("ERROR: Path " + path + " does not exist.\nCreating folder ...")
        try:
            os.makedirs(path)
        except OSError as e:
            sys.stderr.write("ERROR: Path " + path + " could not be creating! Exiting.")
            exit()

    for i, image in enumerate(images):
        cv2.imwrite(path + '/' + str(i) + '.png', image)
    print("Images saved successfully.")


def threshhold(img, lower_limit, upper_limit, iter, kernel_size):
    '''
    Given an image and parameters, returns the binary thresshold of that image. In the context of this project,
    when given proper parameters, the white part of the returned threshhold will the the rock or bone in the image,
    while the black will be its black backdrop.

    :param img: An RGB OpenCV image
    :param upper_limit: Integer [0..255] upper limit of brightness to be displayed in the thresshold
    :param iter: Integer [0..] NUmber of iterations of erosion to occur on the image. Common ranges are 0 to 6,
                 though it is partially dependent upon image resolution. Used in
    :param kernel_size: Integer [0..] The size of the kernel in generating the thresshold. Higher values result
                        in a less noisy image, but too high a kernel size can lead to a blockier image lacking in
                        key details. 2 is a common value.
    :return: OpenCV binary image.
    '''
    y, x = np.shape(img)[0:2]
    # Convert to greyscale if not already
    if len(np.shape(img)) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply the lower limit, it's not baked into the opencv thresshold, but can be
    # useful. Anything below this level is supplanted by a 0, whilst anything
    # higher than the upper is 255. Useful for some imperfect backgrounds.
    gray[gray <= lower_limit] = 0
    ret, thresh = cv2.threshold(gray, upper_limit, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=iter)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing
