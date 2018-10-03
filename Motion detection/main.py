import sys

import cv2
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi


class Image(QDialog):
    def __init__(self):
        super(Image, self).__init__()
        loadUi("presentation/surveillance.ui", self)
        self.image = None
        self.btn_start.clicked.connect(self.start_webcam)
        self.btn_stop.clicked.connect(self.stop_webcam)
        self.btn_motion_image.clicked.connect(self.set_motion_ref_image)
        self.btn_detect.toggled.connect(self.detect_webcam_motion)
        self.btn_detect.setCheckable(True)
        self.detect_motion_enabled = False
        self.motion_frame = None

    def detect_webcam_motion(self, status):
        if status:
            self.detect_motion_enabled = True
            self.btn_detect.setText("Stop motion detection")
        else:
            self.detect_motion_enabled = False
            self.btn_detect.setText("Detect motion")

    def set_motion_ref_image(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        self.motion_frame = gray
        self.display_image(self.motion_frame, 2)

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
        if(self.detect_motion_enabled):
            detected_motion = self.detect_motion(self.image.copy())
            self.display_image(detected_motion, 1)
        else:
            self.display_image(self.image, 1)

    def detect_motion(self, input_img):
        self.text = "No Motion"
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_diff = cv2.absdiff(self.motion_frame, gray)
        threshld = cv2.threshold(frame_diff, 40, 255, cv2.THRESH_BINARY)[1]

        threshld = cv2.dilate(threshld, None, iterations=5)

        im2, cnts, hierarchy = cv2.findContours(threshld.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        height, width, channels = input_img.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        for contour, hier in zip(cnts, hierarchy):
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)

        if max_x - min_x > 80 and max_y - min_y > 80:
            cv2.rectangle(input_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
            self.text = "Motion Detected"

        cv2.putText(input_img, "Motion status: {}".format(self.text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return input_img

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

        if window == 2:
            self.lbl_motion.setPixmap(QPixmap.fromImage(out_image))
            self.lbl_motion.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Image()
    window.setWindowTitle("Start Webcam using Qt GUI")
    window.show()
    sys.exit(app.exec_())
