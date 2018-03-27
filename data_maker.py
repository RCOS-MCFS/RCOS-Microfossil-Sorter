
import comp_vis.img_tools as it
import cv2

def main():
    # Variables used in output styling later on
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (2, 400)
    fontScale = 1
    lineType = 2
    color_bad = (0, 0, 255)
    color_good = (0, 255, 0)
    display_text = ""
    display_color = (0, 0, 0)

    capture = cv2.VideoCapture(0)

    while True:
        # Take in the current frame of the video
        ret, frame = capture.read()
        # check for a clearly defined object in the current frame
        cropped_img, coordinates = it.get_largest_object(frame)
        cv2.rectangle(frame, coordinates[0], coordinates[1], (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()

main()