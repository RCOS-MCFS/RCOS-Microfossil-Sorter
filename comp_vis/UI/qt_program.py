# Thanks to Ben Hoff at http://benhoff.net/face-detection-opencv-pyqt.html for providing a tutorial on integrating
# OpenCV with PyQt5, not seen elsewhere on the internet. Much of the code related to displaying camera data is
# based off of his provided at the link above.


from PyQt5 import QtCore, QtWidgets,  QtGui
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication
from PyQt5.QtGui import QIcon

import cv2
import numpy as np
import sys


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=1, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class ObjectDetectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    def image_data_slot(self, image_data):
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.object_detection_widget = ObjectDetectionWidget()

        self.record_video = RecordVideo()

        image_data_slot = self.object_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.object_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')

        comboBox = QtWidgets.QComboBox(self)
        comboBox.addItem("SVM")
        comboBox.addItem("Perceptron")
        comboBox.addItem("Logistic Regressor")
        layout.addWidget(comboBox)

        layout.addWidget(self.run_button)
        self.run_button.clicked.connect(self.record_video.start_recording)
        self.setLayout(layout)

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)

        self.setGeometry(660, 660, 660, 660)
        self.setWindowTitle('Simple menu')
        self.show()

def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = Example()
    main_widget = MainWidget()
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()