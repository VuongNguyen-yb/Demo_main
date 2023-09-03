import sys
# pip install pyqt5
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic.properties import QtCore
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5 import QtGui

from mainwindow import Ui_MainWindow
from PyQt5.QtCore import QTimer
from recognition4 import FaceRecognition
from PyQt5.QtCore import QObject, pyqtSignal

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.btn_run.clicked.connect(self.start_capture_video)
        self.uic.btn_open.clicked.connect(self.receiver)
        self.thread = {}
        self.timer = QTimer()
        self.fr = FaceRecognition()
        self.fr.signal2.connect(self.show_wedcam)

    def receiver(self):
        self.fr.update_data()
        self.fr.signal1.connect(self.handle_data)

    @pyqtSlot(str)
    def handle_data(self, data):
        print(data)

    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        self.timer.stop()
        self.fr.stop_recognition()

    def start_capture_video(self):
        self.timer.timeout.connect(self.process_recognition)
        self.timer.start(10)  # Change 100 to the desired display frequency in milliseconds

   # @pyqtSlot()
    def process_recognition(self):
        self.fr.run_recognition()

    @pyqtSlot(bool, np.ndarray)
    def show_wedcam(self, bool_value, image_path):
        if bool_value:
            qt_img = self.convert_cv_qt(image_path)
            self.uic.scr_vieo.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap and resize to fit Scr_Vieo"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        #image = QtGui.QImage(rgb_image, w, h, ch * w, QtGui.QImage.Format_RGB888).scaled(self.uic.scr_vieo.width(), self.uic.scr_vieo.height(), Qt.KeepAspectRatio )
        print(self.uic.scr_vieo.height())
        image = QtGui.QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QtGui.QImage.Format_RGB888).scaled(self.uic.scr_vieo.width(), self.uic.scr_vieo.height(), Qt.KeepAspectRatio)
        print(self.uic.scr_vieo.height())
        pixmap = QtGui.QPixmap.fromImage(image)

        return pixmap


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())