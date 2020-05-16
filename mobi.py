import cv2
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import sys

width = 256
height = 256

class MainWindow(QWidget):
    def __init__(self, argv):
        # Qt window
        QWidget.__init__(self)
        self.setWindowTitle('MapGenerator')
        self.setFixedSize(width + 20, height + 250)

        # QLabel with video
        self.video = QLabel('', self)
        self.video.setFixedSize(width, height)

        # Something for the logs
        self.logs = QTextEdit('', self)
        self.logs.setReadOnly(True)
        self.logs.setFixedSize(width, 200)
        self.logs.ensureCursorVisible()
        self.logs.setText("")

        self.save_button = QPushButton('Save', self)

        self.end_button = QPushButton('End', self)

        self.prev_button = QPushButton('Previous', self)
        self.prev_button.setEnabled(False)

        self.pause_button = QPushButton('Pause', self)

        self.next_button = QPushButton('Next', self)
        self.next_button.setEnabled(False)

        self.fast_button = QPushButton('Fast', self)

        # Main layout = vbox, containing two hbox'es
        vbox = QVBoxLayout(self)
        hbox1 = QHBoxLayout(self)
        hbox2 = QHBoxLayout(self)
        vbox4 = QVBoxLayout(self)
        iconbox = QHBoxLayout(self)
        playbox = QHBoxLayout(self)
        buttonbox = QHBoxLayout(self)

        # Put everything together
        playbox.addWidget(self.prev_button)
        playbox.addWidget(self.pause_button)
        playbox.addWidget(self.next_button)
        playbox.addWidget(self.fast_button)
        buttonbox.addWidget(self.save_button)
        buttonbox.addWidget(self.end_button)
        vbox4.addLayout(playbox)
        vbox4.addLayout(iconbox)
        vbox4.addLayout(buttonbox)
        hbox1.addWidget(self.video)
        hbox2.addWidget(self.logs)
        hbox2.addLayout(vbox4)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

        self.setLayout(vbox)
        self.show()


    def on_quit(self):
        QCoreApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(sys.argv)
    sys.exit(app.exec_())
