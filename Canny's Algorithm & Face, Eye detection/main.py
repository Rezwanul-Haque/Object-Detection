import sys

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMessageBox
from PyQt5.uic import loadUi


class Image(QDialog):
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    
    def __init__(self):
        super(Image, self).__init__()
        loadUi("presentation/basicGUI.ui", self)
        self.image = None
        self.processed_image = None
        self.btn_load.clicked.connect(self.load_clicked)
        self.btn_save.clicked.connect(self.save_clicked)
        self.btn_canny.clicked.connect(self.canny_clicked)
        self.btn_detect.clicked.connect(self.detect_clicked)
        self.sldr_canny_threshold.valueChanged.connect(self.canny_display)
        self.dial_value.valueChanged.connect(self.rotate_image)
        self.rotate_value.returnPressed.connect(self.update_image)

    def update_image(self):
        angle = int(self.rotate_value.text())
        self.btn_load.setDefault(False)
        self.btn_load.setAutoDefault(False)
        if(0 <= angle <= 360):
            self.rotate_image(angle)
            self.dial_value.setValue(angle)
        else:
            QMessageBox.information(self, "Error", "Please enter value between 0 to 360")

    def rotate_image(self, angle, scale=1.):
        w = self.image.shape[1]
        h = self.image.shape[0]
        rangle = np.deg2rad(int(angle))  # angle in radians

        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale

        # ask OpenCV for the rotate matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)

        # calculate the move from the old center to the new center combined with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w) * 0.5, (nh-h) * 0.5, 0]))

        # the move only offsets the translation, so update the translation part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]

        self.processed_image = cv2.warpAffine(self.image, rot_mat, (int(np.math.ceil(nw)), int(np.math.ceil(nh))))
        self.rotate_value.setText(str(angle))
        self.display_image(2)
    
    @pyqtSlot()
    def detect_clicked(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) > 3 else self.image
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            if self.ckb_face.isChecked():
                cv2.rectangle(self.processed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                self.processed_image = self.image.copy()
            roi_gray = gray[y: y+h, x: x + w]
            roi_color = self.processed_image[y: y+h, x: x+w]
            if self.ckb_eye.isChecked():
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            else:
                self.processed_image[y: y+h, x: x+w] = self.image[y: y+h, x: x+w].copy()
            
        self.display_image(2)

    @pyqtSlot()
    def canny_display(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) > 3 else self.image
        self.processed_image = cv2.Canny(gray, self.sldr_canny_threshold.value(), self.sldr_canny_threshold.value() * 3)
        self.display_image(2)
    
    @pyqtSlot()
    def canny_clicked(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) > 3 else self.image
        self.processed_image = cv2.Canny(gray, 100, 200)
        self.display_image(2)

    @pyqtSlot()
    def load_clicked(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open file', "E:\\", "Image Files(*.jpg)")
        if fname:
            self.load_image(fname)
        else:
            print("Invalid Image!!")

    @pyqtSlot()
    def save_clicked(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save file', "E:\\", "Image Files(*.jpg)")
        if fname:
            cv2.imwrite(fname, self.processed_image)
        else:
            print("Error")

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        self.processed_image = self.image.copy()
        self.display_image(1)

    def display_image(self, window=1):
        qformat = QImage.Format_Indexed8
        
        if len(self.processed_image.shape) == 3:  # rows[0], cols[1], channels[2]
            if (self.processed_image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        
        img = QImage(self.processed_image, self.processed_image.shape[1], self.processed_image.shape[0], self.processed_image.strides[0], qformat)
        
        # color BGR > RGB
        img = img.rgbSwapped()
        if window == 1:
            self.lbl_image.setPixmap(QPixmap.fromImage(img))
            self.lbl_image.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        else:
            self.lbl_processed.setPixmap(QPixmap.fromImage(img))
            self.lbl_processed.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        
    
app = QApplication(sys.argv)
window = Image()
window.setWindowTitle("Load Image to Qt GUI")
window.show()
sys.exit(app.exec_())
