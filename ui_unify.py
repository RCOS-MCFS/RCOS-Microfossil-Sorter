# Should maybe swap to PyQt5?

import sys
import os
from PySide2.QtWidgets import *
from PySide2.QtCore import *
qt_app = QApplication(sys.argv)


class UI_Unify(QLabel):


    def __init__(self):
        # Initialize the object as a QLabel
        QLabel.__init__(self)

        # Label for the method chooser
        self.method_lbl = QLabel('Method:', self)
        self.method_lbl.move(5, 5)  # offset the first control 5px
        # from top and left
        self.methods = ['Perceptron',
                        'SVM',
                        'Broken things',
                        ]
        # Create and fill the combo box to choose the method
        self.method = QComboBox(self)
        self.method.addItems(self.methods)

        # Allow 100px for the label and 5px each for borders at the
        # far left, between the label and the combobox, and at the far
        # right
        self.method.setMinimumWidth(285)
        # Place it five pixels to the right of the end of the label
        self.method.move(110, 5)




        # The label for getting pickle file
        self.pickleget_lbl = QLabel('Pickle:', self)
        # 5 pixel indent, 25 pixels lower than last pair of widgets
        self.pickleget_lbl.move(5, 30)

        self.pickleget = QLineEdit(self)
        # Add some ghost text to indicate what sort of thing to enter
        self.pickleget.setPlaceholderText('Enter location of pickle file here')
        # Same width as the method choose box
        self.pickleget.setMinimumWidth(285)
        # Same indent as method but 25 pixels lower
        self.pickleget.move(110, 30)





        # Set the size, alignment, and title
        self.setMinimumSize(QSize(400, 200))
        self.setAlignment(Qt.AlignCenter)
        self.setWindowTitle('Microfossil Concentrate Fossil Sorter')

        # The build button is a push button
        self.discrim_button = QPushButton('classify', self)

        # Place it at the bottom right, narrower than
        # the other interactive widgets
        self.discrim_button.setMinimumWidth(145)
        self.discrim_button.move(250, 55)

        self.discrim_button.clicked.connect(self.discrim_click)




        self.rockget_lbl = QLabel('Rocks:', self)
        self.rockget_lbl.move(5, 105)

        self.rockget = QLineEdit(self)
        self.rockget.setPlaceholderText('Enter location of rock folder here')
        self.rockget.setMinimumWidth(285)
        self.rockget.move(110, 105)





        self.boneget_lbl = QLabel('Bones:', self)
        self.boneget_lbl.move(5, 130)

        self.boneget = QLineEdit(self)
        self.boneget.setPlaceholderText('Enter location of bone folder here')
        self.boneget.setMinimumWidth(285)
        self.boneget.move(110, 130)




        # The build button is a push button
        self.data_button = QPushButton('gather data', self)

        # Place it at the bottom right, narrower than
        # the other interactive widgets
        self.data_button.setMinimumWidth(145)
        self.data_button.move(250, 155)

        self.data_button.clicked.connect(self.data_click)


    def run(self):
        ''' Show the application window and start the main event loop '''
        self.show()
        qt_app.exec_()

    # Connect buttons?
    # Figure out way of determining pickle versus file directory
    def data_click(self):
        ''' Tell when the button is clicked. '''
        print('Click! Data tool will break!')
        print("Input: \"" + self.boneget.text() + "\" and \"" + self.rockget.text() + "\"")
        os.system("py data_maker.py 0 perceptron " + self.boneget.text() + " " + self.rockget.text()) # broke?

    def discrim_click(self):
        ''' Tell when the button is clicked. '''
        print('Click! Discrimination will break!')
        print("Input: \"" + self.boneget.text() + "\" and \"" + self.rockget.text() + "\"")
        os.system("py live_demo.py 0 " + self.boneget.text() + " " + self.rockget.text())





# Create an instance of the application and run it
UI_Unify().run()




#cv2.namedWindow("MICROFOSSIL SORTER")
# Two file open things for directories
#b1 = cv2.createButton("Gather Data", buttonType=cv2.QT_PUSH_BUTTON);
# One file open thing for pickle file
# Drop down kinds of algorithms with QMenu
#b2 = cv2.createButton("Identify", buttonType=cv2.QT_PUSH_BUTTON);




# https://stackoverflow.com/questions/9251644/simple-file-browser-file-chooser-in-python-program-with-qt-gui
# File browser