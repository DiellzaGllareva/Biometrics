import pickle

import numpy as np
from PyQt5.QtCore import QRect
from keras_preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QMovie
import cv2
import imutils
import pyrebase
import sys
import os
import pytesseract
from train import Train_Model

print("[INFO] loading face detector...")
protoPath = os.path.join("face_detector", "deploy.prototxt")
modelPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# loading the liveness detector model and label encoder
print("[INFO] loading liveness detector...")
model = load_model("liveness.model")
le = pickle.loads(open("le.pickle", "rb").read())
config = {
    "apiKey": "AIzaSyBVYsYBPafzmC5ppwzNHhs43Juv_CYC8d0",
    "authDomain": "diploma-2022-8d60b.firebaseapp.com",
    "databaseURL": "gs://diploma-2022-8d60b.appspot.com",
    "projectId": "diploma-2022-8d60b",
    "storageBucket": "diploma-2022-8d60b.appspot.com",
    "messagingSenderId": "891048948646",
    "appId": "1:891048948646:web:d8377fe12651a83f735567"
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
y = 0
z = 255
real = "real"
fake = "fake"
data = []
labels = []
match = bool(True)
dismatch = bool(False)
my_image = "firebaseImages/snapshot.png"
croped_image = "firebaseImages/faces_detected.jpg"
method = cv2.TM_SQDIFF_NORMED
small_image = cv2.imread('schema.png')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

stylesheet = """
QPixmap{
    width: 20px;
    height: 20px;
}
QPushButton{
    font-size: 40px;
    font-family: Marcellus SC;
}
"""


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        '''MainWindow.setStyleSheet("background-color: #6d679e;")'''
        MainWindow.resize(1300, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(45, 20, 491, 160))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 20, 491, 160))
        self.label_2.setObjectName("label2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(45, 20, 491, 160))
        self.label_3.setObjectName("label3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(45, 20, 491, 160))
        self.label_4.setObjectName("label4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(45, 20, 491, 250))
        self.label_5.setObjectName("label5")
        self.label_5.setWordWrap(bool(True))
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(45, 20, 491, 250))
        self.label_6.setObjectName("label5")
        self.label_6.setWordWrap(bool(True))
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(25, 25, 200, 200))
        self.label_7.setMinimumSize(QtCore.QSize(250, 250))
        self.label_7.setMaximumSize(QtCore.QSize(250, 250))
        self.label_7.setObjectName("label7")
        # Loading the GIF
        self.movie = QMovie("loader.gif")
        self.label_7.setMovie(self.movie)
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(50, 105, 670, 650))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap("designImages/background.png"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        self.photo_2 = QtWidgets.QLabel(self.centralwidget)
        self.photo_2.setGeometry(QtCore.QRect(5, 50, 200, 200))
        self.photo_2.setText("")
        self.photo_2.setPixmap(QtGui.QPixmap("designImages/idVerification.png"))
        self.photo_2.setScaledContents(True)
        self.photo_2.setObjectName("photo2")
        self.photo_3 = QtWidgets.QLabel(self.centralwidget)
        self.photo_3.setGeometry(QtCore.QRect(600, 570, 200, 200))
        self.photo_3.setText("")
        self.photo_3.setPixmap(QtGui.QPixmap("designImages/faceVerification.png"))
        self.photo_3.setScaledContents(True)
        self.photo_3.setObjectName("photo3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(180, 950, 190, 80))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.move(700, 1850)
        self.pushButton.setStyleSheet("border: 1px solid black;")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 1000, 460, 80))
        self.pushButton_2.setObjectName("pushButton")
        self.pushButton_2.move(760, 450)
        self.pushButton_2.setStyleSheet("border: 1px solid black;")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(180, 950, 190, 80))
        self.pushButton_3.setObjectName("pushButton")
        self.pushButton_3.setStyleSheet("border: 1px solid black;")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(180, 950, 190, 80))
        self.pushButton_4.setObjectName("pushButton4")
        self.pushButton_4.setStyleSheet("border: 1px solid black;")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton_3.clicked.connect(self.loadImage2)
        self.pushButton_4.clicked.connect(self.loadImage3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        Train_Model.startTrain(self, data, labels)
        # Added code here
        self.filename = 'firebaseImages/snapshot.png'  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display

    def postImage(self):
        storage.child(my_image).put(croped_image)

    def deleteImage(self):
        storage.child(my_image).delete(my_image)

    def readImage(self, img):
        keywords = ['KOSOVË', 'REPLUBLIKA', 'LETËRNJOFTIM', 'KOSOVO', 'K0C0B0', 'REPUBLIC', 'KAPTA', 'IDENTITY', 'CARD',
                    'KOSOVAR']
        for keyword in keywords:
            '''if keyword in text:
                print("Match found!")'''
        if self.checkSchema(img) and keyword in keywords:
            global y, z
            y = 255
            z = 0
            self.label.move(800, 450)
            self.label.setStyleSheet("font-size:80px;color: #3c9150;font-family: Marcellus SC;")
            self.pushButton_3.setEnabled(bool(True))
            self.label.setText("Id verified!")
        else:
            y = 0
            z = 255
            self.label.setStyleSheet("font-size:80px;color: #993131;font-family: Marcellus SC;")
            self.label.move(800, 450)
            self.label.setText("Try again!")

    def getImage(self):
        storage.child(my_image).download(my_image)

    def checkSchema(self, img):
        result = cv2.matchTemplate(small_image, img, method)
        mn, _, mnLoc, _ = cv2.minMaxLoc(result)
        print(mn)
        if mn < 0.009 or mn > 0.12:
            return dismatch
        else:
            return match

    def loadImage(self):
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """
        self.photo.clear()
        self.photo_2.clear()
        self.photo_3.clear()
        self.label.setText("")
        self.label_2.setGeometry(40, -20, 1220, 85)
        self.label_3.setText("")
        self.label_4.setText("Instructions")
        self.label_4.setStyleSheet("font-size: 42px;font-family: Marcellus SC;")
        self.label_4.move(850, 60)
        self.label_5.setText("Step 1: ID Card Verification.          "
                             " Please place the ID card into the red frame and click Save button. You will be informed if the ID card was verified or not.")
        self.label_5.move(760, 180)
        self.label_5.setStyleSheet("font-size:26px;font-family: Century Gothic; text-align:center;")
        self.pushButton.move(750, 688)
        self.pushButton_3.move(1070, 688)
        self.pushButton_2.move(10, 7800)
        self.pushButton_3.setEnabled(bool(True))
        self.pushButton.clicked.connect(self.savePhoto)
        cam = True  # True for webcam
        if cam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture('video.mp4')  # place path to your video file here

        while (vid.isOpened()):
            QtWidgets.QApplication.processEvents()
            img, self.image = vid.read()
            self.image = imutils.resize(self.image)

            cv2.rectangle(self.image, (50, 40), (600, 400), (0, y, z), 2)
            self.update()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def loadImage2(self):
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """

        self.photo.clear()
        self.label.setText("")
        self.label_5.setText("")
        self.pushButton_3.move(10700, 688)
        self.pushButton_4.move(1070, 688)
        self.pushButton.setEnabled(bool(False))
        self.pushButton_4.setEnabled(bool(False))
        self.label_6.setText("Step 2: Liveness check.                    "
                             " Please move in front of the camera and let the algorithm to detect if you are real or not. You will be informed and if you are real please click save to save photo and move in other steps.")
        self.label_6.move(760, 180)
        self.label_6.setStyleSheet("font-size:26px;font-family: Century Gothic; text-align:center;")
        cam = True  # True for webcam
        if cam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture('video.mp4')  # place path to your video file here

        while (vid.isOpened()):
            QtWidgets.QApplication.processEvents()
            img, self.image = vid.read()
            self.image = imutils.resize(self.image)
            # grab frame and have a maximum width of 600 pixels

            # grab the frame dimensions and convert it to a blob
            (h, w) = self.image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the detected bounding box does fall outside the dimensions of the frame
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # get the face region
                    face = self.image[startY:endY, startX:endX]
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    # pass the face ROI through the trained liveness detector model to determine if the face is "real" or "fake"
                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]

                    # draw the label and bounding box on the frame
                    label = "{}: {:.4f}".format(label, preds[j])
                    if fake in label:
                        self.pushButton.setEnabled(bool(False))
                        self.pushButton_4.setEnabled(bool(False))
                        cv2.putText(self.image, label, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(self.image, (startX, startY), (endX, endY),
                                      (0, 0, 255), 2)
                    if real in label:
                        self.pushButton.setEnabled(bool(True))
                        self.pushButton_4.setEnabled(bool(True))
                        cv2.putText(self.image, label, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(self.image, (startX, startY), (endX, endY),
                                      (0, 255, 0), 2)
            self.update()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def loadImage3(self):
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """
        """ This function will load the camera device, obtain the image
            and set it to label using the setPhoto function
        """
        self.photo.clear()
        self.photo_2.clear()
        self.photo_3.clear()
        self.label.setText("")
        self.label_2.setGeometry(40, -20, 1220, 85)
        self.label_3.setText("")
        self.label_5.setText("")
        self.label_5.move(70060, 180)
        self.label_5.setStyleSheet("font-size:26px;font-family: Century Gothic; text-align:center;")
        self.label_6.setText("")
        self.pushButton.move(700500, 688)
        self.pushButton_3.move(107000, 688)
        self.pushButton_2.move(100000, 7800)
        self.pushButton_4.move(10000, 10)
        cam = True  # True for webcam
        if cam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture('video.mp4')  # place path to your video file here

        while (vid.isOpened()):
            QtWidgets.QApplication.processEvents()
            img, self.image = vid.read()
            self.image = imutils.resize(self.image)
            self.photo.setGeometry(QtCore.QRect(0, 0, 10, 10))
            self.label_4.setText("Processing data")
            self.label_4.setStyleSheet("font-size: 42px;font-family: Marcellus SC;")
            self.label_4.move(500, 60)
            self.label_4.setGeometry(QRect(500, 60, 500, 100))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def setPhoto(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.tmp = image
        '''self.label.setText("")'''
        image = imutils.resize(image)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.photo.setPixmap(QtGui.QPixmap.fromImage(image))
        self.photo.setGeometry(QtCore.QRect(20, 450, 700, 700))
        self.photo.move(10, 100)
        self.pushButton_3.setText("Next")
        self.pushButton_4.setText("Next")

    def fixImageColours(self, img):
        """ This function will take an image (img) and the brightness
            value. It will perform the brightness change using OpenCv
            and after split, will merge the img and return it.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255
        v[v > lim] = 255
        v[v <= lim] += 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def update(self):
        """ This function will update the photo according to the
            current values of blur and brightness and set it to photo label.
        """
        img = self.fixImageColours(self.image)
        self.setPhoto(img)

    def savePhoto(self):
        """ This function will save the image"""
        self.filename = 'firebaseImages/snapshot.png'
        cv2.imwrite(self.filename, self.tmp)
        print('Image saved as:', self.filename)
        self.readImage(self.tmp)
        self.cropImage()
        self.postImage()

    def saveDetecionPhoto(self):
        """ This function will save the image"""
        self.filename = 'firebaseImages/snapshot.png'
        cv2.imwrite(self.filename, self.tmp)
        print('Image saved as:', self.filename)
        self.readImage(self.tmp)
        self.cropImage()
        self.postImage()

    def cropImage(self):
        image = cv2.imread('firebaseImages/snapshot.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_Cascade = cv2.CascadeClassifier("haarcascade/frontalface.xml")
        faces = face_Cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )

        print("[INFO] Found {0} Faces!".format(len(faces)))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            cv2.imwrite('firebaseImages/faces_detected.jpg', roi_color)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Check Me !"))
        self.label.setText(_translate("MainWindow", "CHECK "))
        self.label.setStyleSheet("font-size: 130px;font-family: Marcellus SC;")
        self.label.setGeometry(750, 130, 540, 110)
        self.label_3.setText(_translate("MainWindow", "ME"))
        self.label_3.setStyleSheet("font-size: 130px;font-family: Marcellus SC;")
        self.label_3.setGeometry(880, 260, 540, 110)
        self.label_2.setText(_translate("MainWindow",
                                        "__________________________________________________________________________________"))
        self.label_2.setGeometry(60, -20, 1220, 85)
        self.label_2.setStyleSheet("font-size: 94px;font-family: Saber;color: #bab8b8;")
        self.pushButton.setText(_translate("MainWindow", "Save"))
        self.pushButton_2.setText(_translate("MainWindow", "Verify"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
