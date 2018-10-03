import sys

import cv2
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi


class Image(QDialog):
    def __init__(self):
        super(Image, self).__init__()
        loadUi("presentation/Webcam.ui", self)
        self.image = None
        self.processed_image = None
        self.btn_start.clicked.connect(self.start_webcam)
        self.btn_stop.clicked.connect(self.stop_webcam)
        self.btn_canny.toggled.connect(self.canny_webcam)
        self.btn_canny.setCheckable(True)
        self.canny_enabled = False
        self.btn_detect.setCheckable(True)
        self.btn_detect.toggled.connect(self.detect_webcam_face)
        self.face_enabled = False

        self.face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

    def detect_webcam_face(self, status):
        if status:
            self.face_enabled = True
            self.btn_detect.setText("Stop face detection")
        else:
            self.face_enabled = False
            self.btn_detect.setText("Detect")

    def canny_webcam(self, status):
        if status:
            self.canny_enabled = True
            self.btn_canny.setText("Stop Canny")
        else:
            self.canny_enabled = False
            self.btn_canny.setText("Canny")


    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)

        if self.canny_enabled:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) >= 3 else self.image
            self.processed_image = cv2.Canny(gray, 100, 200)
            self.display_image(self.processed_image, 2)

        if self.face_enabled:
            detected_face = self.detect_face(self.image)
            self.display_image(detected_face, 1)
        else:
            self.display_image(self.image, 1)

    def detect_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return img

    def stop_webcam(self):
        self.timer.stop()

    def display_image(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=row ,[1]=cols, [2]=Channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR>>RGB
        out_image = out_image.rgbSwapped()
        if window == 1:
            self.lbl_webcamfeed.setPixmap(QPixmap.fromImage(out_image))
            self.lbl_webcamfeed.setScaledContents(True)

        else:
            self.lbl_processed_webcamfeed.setPixmap(QPixmap.fromImage(out_image))
            self.lbl_processed_webcamfeed.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Image()
    window.setWindowTitle("Start Webcam using Qt GUI")
    window.show()
    sys.exit(app.exec_())
