import sys

import numpy as np
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
        self.btn_start.clicked.connect(self.start_webcam)
        self.btn_stop.clicked.connect(self.stop_webcam)
        self.btn_track.setCheckable(True)
        self.btn_track.toggled.connect(self.track_webcam_color)
        self.track_enabled = False

        self.btn_set_color_1.clicked.connect(self.set_color_1)

    def track_webcam_color(self, status):
        if status:
            self.track_enabled = True
            self.btn_track.setText("Stop Tracking")
        else:
            self.track_enabled = False
            self.btn_track.setText("Track Color")

    def set_color_1(self):
        self.color1_lower = np.array([self.sldr_h_min.value(), self.sldr_s_min.value(), self.sldr_v_min.value()], np.uint8)
        self.color1_upper = np.array([self.sldr_h_max.value(), self.sldr_s_max.value(), self.sldr_v_max.value()], np.uint8)

        self.lbl_color_1_value.setText("Min: " + str(self.color1_lower) + " Max: " + str(self.color1_upper))

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
        self.display_image(self.image, 1)

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        color_lower = np.array([self.sldr_h_min.value(), self.sldr_s_min.value(), self.sldr_v_min.value()], np.uint8)
        color_upper = np.array([self.sldr_h_max.value(), self.sldr_s_max.value(), self.sldr_v_max.value()], np.uint8)

        color_mask = cv2.inRange(hsv, color_lower, color_upper)
        self.display_image(color_mask, 2)

        if self.track_enabled and self.ckb_color_1.isChecked():
            tracked_image = self.track_colored_object(self.image.copy())
            self.display_image(tracked_image, 1)
        else:
            self.display_image(self.image, 1)

    def track_colored_object(self, img):
        blur = cv2.blur(img, (3, 3))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.color1_lower, self.color1_upper)

        erode = cv2.erode(color_mask, None, iterations=2)
        dilate = cv2.dilate(erode, None, iterations=10)

        kernel_open = np.ones((5, 5))
        kernel_close = np.ones((20, 20))

        mask_open = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel_open)
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

        (_, contours, hierarchy) = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "Object Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

        if window == 2:
            self.lbl_processed_webcamfeed.setPixmap(QPixmap.fromImage(out_image))
            self.lbl_processed_webcamfeed.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Image()
    window.setWindowTitle("Start Webcam using Qt GUI")
    window.show()
    sys.exit(app.exec_())
