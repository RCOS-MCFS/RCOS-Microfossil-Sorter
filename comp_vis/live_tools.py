# Tools related to the use of the webcam in our project

import comp_vis.img_tools as it
import cv2
import numpy as np

def live_labeling(model, camera_no=0, threshold_settings=(128, 1, 2)):
    '''
    One of the key functions used in the demo, this acts as a live demonstration of the
    utilty's capability for labeling, using the provided model to display classifiations on
    objects passing in front of the camera.

    TODO: Redo contouring, have set of contour parameters be as argument, increase range of usable lighting envts.

    :param model: The model, a subclass of the Approach class, being used for classification.
    :param camera_no: The number of the webcam to be used. Typically 0 on the raspberry pi, but
                      often 1 if you're using a USB webcam on a computer with a webcam.
    :return: None
    '''

    # Variables used in output styling later on
    font_scale = 1
    line_type = 2

    error_position = (160, 450)
    error_color = (0, 0, 255)

    font = cv2.FONT_HERSHEY_SIMPLEX

    capture = cv2.VideoCapture(camera_no)

    label = {1: 'bone',
             0: 'rock', }

    while True:
        ret, frame = capture.read()

        _, contour = it.get_largest_object(frame, threshhold_settings=threshold_settings)
        if contour is not None:
            # If object detected, label that object, outline it, and draw the label
            # over the outline.
            cropped_img = it.crop_to_contour(frame, contour)
            a, b = it.get_images_dimensions(cropped_img, normalized=True, ordered=True)
            contour_coordinates = it.coordinates_from_contour(contour)
            cropped_img_avg = it.average_color(cropped_img)
            datum = cropped_img_avg + [float(a), float(b)]
            datum = np.array(datum)
            response = model.classify(datum)

            response_text = label[response]
            display_coord = ((contour_coordinates[0][0] + contour_coordinates[1][0]) / 2,
                             (contour_coordinates[0][1] + contour_coordinates[1][1]) / 2)
            display_coord = (int(display_coord[0]) - 10, int(display_coord[1]))
            if response_text == 'bone':
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            else:
                cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
            cv2.putText(frame, response_text,
                        display_coord,
                        font,
                        font_scale,
                        (255, 255, 255),
                        line_type)
        else:
            cv2.putText(frame, "No object detected",
                        error_position,
                        font,
                        font_scale,
                        error_color,
                        line_type)
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    capture.release()

def data_gathering(thresh_settings=(0, 128, 1, 2), camera_num=0):
    '''
    :param camera_num:
    :return: A tuple containing a list of gathered rock images, and a list of gathered bone images.
    '''
    # Variables used in output styling later on
    font = cv2.FONT_HERSHEY_SIMPLEX
    instruction_text = "'r' = rock - 'b' = Bone"

    capture = cv2.VideoCapture(camera_num)

    # Numbers corresponding to various key presses.
    q = 113
    r = 114
    b = 98

    # Used to save objects
    rock_counter = 0
    bone_counter = 0

    # Number of frames between presses
    time_till_next_press = 0
    frames_between_presses = 10

    rock_images = []
    bone_images = []

    while True:
        # Take in the current frame of the video
        ret, frame = capture.read()
        # check for a clearly defined object in the current frame
        thresh, contour = it.get_largest_object(frame, threshhold_settings=thresh_settings)

        # If a contour is found, we display it for use in debugging.
        if contour is not None:
            original = frame.copy()
            cv2.drawContours(frame, [contour], 0, (255, 0, 0), 2)

        cv2.putText(frame, instruction_text, (10, 30), font, 1, (255, 255, 255), 0, cv2.LINE_AA)
        if time_till_next_press > 0:
            cv2.putText(frame, 'Saved!', (10, 300), font, 1, (0, 255, 0), 0, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        # cv2.imshow('threshhold', thresh)
        key = cv2.waitKey(30)

        if key == q:
            break
        elif contour is not None and time_till_next_press == 0:
            if key == r:
                cropped_image = it.crop_to_contour(original, contour)
                rock_images.append(cropped_image)
                time_till_next_press = frames_between_presses
                rock_counter += 1
            elif key == b:
                cropped_image = it.crop_to_contour(original, contour)
                bone_images.append(cropped_image)
                time_till_next_press = frames_between_presses
                bone_counter += 1

        time_till_next_press = max(time_till_next_press-1, 0)

    capture.release()

    return bone_images, rock_images

