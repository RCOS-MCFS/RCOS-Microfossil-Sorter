
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

    capture = cv2.VideoCapture(1)

    display_text = ""
    display_color = (0, 0, 0)

    while True:
        ret, frame = capture.read()
        # Todo: Analysis
        if False:
            _ = 1
        else:
            display_text = "No object detected"
            display_color = color_bad
        cv2.putText(frame, display_text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    display_color,
                    lineType)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()

main()