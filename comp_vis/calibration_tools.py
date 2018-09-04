'''
Functions to be used in the generation of proper parameters for image isolation and classification to be used
in this experiment.
'''

import comp_vis.img_tools as it

import cv2
import numpy as np


cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)


def calibrate_thressholding(lower_limit=0, upper_limit=128, erode_iter=1, kernel_size=2):
    '''
    :param upper_limit:
    :param erode_iter:
    :param kernel_size:
    :return: A tuple containing (upper_limit, erode_iter, and kernel_size)
    '''

    def nil(_):
        '''
        This is a junk method used to satisfy the requirements of the create
        trackbar function.
        '''
        pass

    # Trackbars to be used in value adjustments
    cv2.createTrackbar('lower_limit', 'image', lower_limit, 255, nil)
    cv2.createTrackbar('upper limit','image',upper_limit,255, nil)
    cv2.createTrackbar('erosion_lvl','image',erode_iter, 16, nil)
    cv2.createTrackbar('kernel size','image',kernel_size, 16, nil)

    capture = cv2.VideoCapture(0)

    while(1):
        _, img = capture.read()
        thresh_settings = (lower_limit, upper_limit, erode_iter, kernel_size)
        thresh, contour = it.get_largest_object(img, threshhold_settings=thresh_settings)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        if contour is not None:
            img = cv2.drawContours(img, [contour], 0, (255, 0, 0), 2)

        final = np.concatenate((img, thresh), axis=1)
        cv2.imshow('image',final)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        lower_limit = cv2.getTrackbarPos('lower_limit', 'image')
        upper_limit = cv2.getTrackbarPos('upper limit','image')
        erode_iter = cv2.getTrackbarPos('erosion_lvl','image')
        kernel_size = cv2.getTrackbarPos('kernel size','image')

    cv2.destroyAllWindows()

    return lower_limit, upper_limit, erode_iter, kernel_size

if __name__ == '__main__':
    print("Testing threshholding calibration..")
    calibrate_thressholding()