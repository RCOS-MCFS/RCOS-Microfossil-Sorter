'''
Functions to be used in the generation of proper parameters for image isolation and classification to be used
in this experiment.
'''

from comp_vis.img_tools import threshhold

import cv2
import numpy as np

# Create a black image
cv2.namedWindow('image')


def calibrate_thressholding(upper_limit=128, erode_iter=1, kernel_size=2):
    '''
    :param upper_limit:
    :param erode_iter:
    :param kernel_size:
    :return: A tuple containing (upper_limit, erode_iter, and kernel_size)
    '''

    def nil():
        '''
        This is a junk method used to satisfy the requirements of the create
        trackbar function.
        '''
        pass

    # Trackbars to be used in value adjustments
    cv2.createTrackbar('upper limit','image',upper_limit,255, nil)
    cv2.createTrackbar('erosion_lvl','image',erode_iter, 16, nil)
    cv2.createTrackbar('kernel size','image',kernel_size, 16, nil)

    capture = cv2.VideoCapture(0)

    while(1):
        _, img = capture.read()
        img = threshhold(img, upper_limit, erode_iter, kernel_size)
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        upper_limit = cv2.getTrackbarPos('upper limit','image')
        iter = cv2.getTrackbarPos('erosion_lvl','image')
        kernel_size = cv2.getTrackbarPos('kernel size','image')

    cv2.destroyAllWindows()

    return upper_limit, erode_iter, kernel_size

if __name__ == '__main__':
    print("Testing threshholding calibration..")
    calibrate_thressholding()