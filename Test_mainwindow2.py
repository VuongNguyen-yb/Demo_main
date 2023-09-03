import sys
# pip install pyqt5
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from mainwindow import Ui_MainWindow
from PyQt5.QtCore import QTimer
from recognition4 import FaceRecognition
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtCore import QTimer

class MainWindow(QMainWindow, QObject):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.btn_run.clicked.connect(self.start_capture_video)
        self.uic.btn_open.clicked.connect(self.receiver)
        self.thread = {}
        self.timer = QTimer()
        self.fr = FaceRecognition()

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
        self.thread[1].stop()

    def start_capture_video(self):
        self.thread[1] = capture_video(index=1)
        self.timer.timeout.connect(self.thread[1].start)
        self.timer.start(100)  # Thay đổi 100 thành số milliseconds bạn muốn cho tần suất hiển thị
        self.thread[1].signal.connect(self.show_wedcam)

    def show_wedcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.scr_vieo.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        qimage = pixmap.toImage()
        return QPixmap.fromImage(qimage)


class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)

    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super(capture_video, self).__init__()
        self.fr = FaceRecognition()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_recognition)
        self.timer.start(10)
        self.fr.signal2.connect(self.run_frame)

    def process_recognition(self):
        self.fr.run_recognition()

    def run(self):
       self.run_frame()

    def stop(self):
        print("stop threading", self.index)
        self.terminate()

    @pyqtSlot(bool, np.ndarray)
    def run_frame(self,bool_value, image_path):
        while True:
            ret = bool_value
            cv_img = image_path
            print(cv_img)
            if ret:
                self.signal.emit(cv_img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())