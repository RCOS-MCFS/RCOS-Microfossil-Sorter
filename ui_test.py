
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import comp_vis.img_tools as it
import cv2
import numpy as np


# these two functions are modified to update frame-by-frame for incorporation into a single program through
# a UI.



def live_update(model, camera_no=0, threshold_settings=(128, 1, 2)):
    '''
        One of the key functions used in the demo, this acts as a live demonstration of the
        utility's capability for labeling, using the provided model to display classifications on
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
    capture.release()
    return frame
    #cv2.imshow('frame', frame)
    #if cv2.waitKey(30) & 0xFF == ord('q'):
    #    break












class MyDialog(QDialog):
    def __init__(self, parent=None, dispmode=0):
        super(MyDialog, self).__init__(parent)

        # Determine which mode we're in
        if 0 == dispmode:  # Data make
            self.frame = cv2.imread(r'cat.jpg')
        else:  # Live demo
            self.frame = cv2.imread(r'cat.jpg')


        height, width, byteValue = self.frame.shape
        byteValue = byteValue * width

        cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB, self.frame)

        self.mQImage = QImage(self.frame, width, height, byteValue, QImage.Format_RGB888)

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self.mQImage)
        painter.end()

    def keyPressEvent(self, QKeyEvent):  # Add checker to see if we're in data-gather mode
        super(MyDialog, self).keyPressEvent(QKeyEvent)
        if 'b' == QKeyEvent.text():
            cv2.imwrite("cat2.png", self.frame)  # Change to appropriate directory
        elif 'r' == QKeyEvent.text():
            cv2.imwrite("cat2.png", self.frame)  # Change to appropriate directory
        else:
            app.exit(1)


if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    w = MyDialog(1)
    w.resize(600, 400)
    w.show()
    app.exec_()
