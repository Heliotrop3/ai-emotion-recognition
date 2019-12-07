import numpy as np
import pandas as pd
import sys, os, getpass
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from fastai.vision import *


BASEPATH = r'C:\Users\{}\Documents\GitHub\ai-emotion-recognition'.format(getpass.getuser())
data = pd.read_csv(r'{}\Data\fer2013.csv'.format(BASEPATH))
model = load_model(r"Models\model_64_40Epochs.h5")

pixels = data['pixels'].tolist() # 1

width, height = 48, 48
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')] # 2
    face = np.asarray(face).reshape(width, height) # 3
    
    # There is an issue for normalizing images. Just comment out 4 and 5 lines until when I found the solution.
    # face = face / 255.0 # 4
    # face = cv2.resize(face.astype('uint8'), (width, height)) # 5
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1) # 6

emotions = pd.get_dummies(data['emotion']).as_matrix() # 7

batch_size = 64

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)


scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)

print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))

model.fit(1)
preds,y,losses = model.get_preds(with_loss=True)
interp = ClassificationInterpretation(model, preds, y, losses)

interp = ClassificationInterpretation.from_learner(learn)

#print(confusion_matrix(y_test, y_val))