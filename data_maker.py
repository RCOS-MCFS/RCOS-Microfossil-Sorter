
import comp_vis.img_tools as it
import cv2
import os
import sys

def main():
    # Check that requisite arguments have been provided
    if len(sys.argv) < 4:
        sys.stderr.write("ERROR: Not all arguments provided. Should follow format:\n" +
                         "data_maker.py [camera_num (usually 0)] [bones_write_path] [rocks_write_path]")
        exit(1)

    camera_num  = int(sys.argv[1])
    bones_write_path = sys.argv[2]
    rocks_write_path = sys.argv[3]

    # Create directories if they do not yet exist.
    if not os.path.isdir(bones_write_path):
        os.makedirs(bones_write_path)
    if not os.path.isdir(rocks_write_path):
        os.makedirs(rocks_write_path)

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

    # Todo: Add parameter so that existing files won't be overwritten

    while True:
        # Take in the current frame of the video
        ret, frame = capture.read()
        # check for a clearly defined object in the current frame
        thresh, contour = it.get_largest_object(frame)

        # If a contour is found, we display it for use in debugging.
        if contour is not None:
            original = frame.copy()
            cv2.drawContours(frame, [contour], 0, (255, 0, 0), 2)

        cv2.putText(frame, instruction_text, (10, 30), font, 1, (255, 255, 255), 0, cv2.LINE_AA)
        if time_till_next_press > 0:
            cv2.putText(frame, 'Saved!', (10, 300), font, 1, (0, 255, 0), 0, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        # cv2.imshow('threshhold', thresh)
        key = cv2.waitKey(1)

        if key == q:
            break
        elif contour is not None and time_till_next_press == 0:
            if key == r:
                cropped_image = it.crop_to_contour(original, contour)
                cv2.imwrite(rocks_write_path + '/rock_' + str(rock_counter) + '.png', cropped_image)
                time_till_next_press = frames_between_presses
                rock_counter += 1
            elif key == b:
                cropped_image = it.crop_to_contour(original, contour)
                cv2.imwrite(bones_write_path + '/bone_' + str(bone_counter) + '.png', cropped_image)
                time_till_next_press = frames_between_presses
                bone_counter += 1

        time_till_next_press = max(time_till_next_press-1, 0)

    capture.release()

main()