from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy
import os
import numpy as np
import cv2

#https://github.com/gitshanks/fer2013/blob/master/fertestcustom.py
#Loads model
model = load_model(r"Models\model_64_40Epochs.h5")

#sets variables
WIDTH = 48
HEIGHT = 48
x=None
y=None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#Load image
image = cv2.imread("test.jpg")

#Resize image if it is too large
if image.shape[0] > 1000 or image.shape[1] > 2000:
    scale_percent = 25 #percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#CV2 Stuff for face reconition
gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face.detectMultiScale(gray, 1.3  , 10)

#CV2 Ditects face and our model predicts emotion on face
for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #predicting the emotion
        yhat= model.predict(cropped_img)
        cv2.putText(image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        print("Emotion: "+labels[int(np.argmax(yhat))])

#Shows image
cv2.imshow('Emotion', image)
cv2.waitKey()