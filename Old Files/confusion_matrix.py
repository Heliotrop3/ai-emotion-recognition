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
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

#Defines Paths
BASEPATH = r'C:\Users\{}\Documents\GitHub\ai-emotion-recognition'.format(getpass.getuser())
data = pd.read_csv(r'{}\DataSet\fer2013.csv'.format(BASEPATH))

#Loads Model
model = load_model(r"Models\model_64_40Epochs.h5")

pixels = data['pixels'].tolist() #Converts pixels to a list of each row
#Converts pixels in to RGB values in an array
batch_size = 64
width, height = 48, 48
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1) #Expands dimention of each image

emotions = pd.get_dummies(data['emotion']).as_matrix() #Converts sediment labels to a matrix

#Splits Dataset into testing, training, and validation
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

#Predicts values in test data set and saves predictions into list
y_pred = model.predict_classes(X_test)

#Gets y_test into right format
new_y_test = []
value_list = []
for row_list in y_test:
    for  i in range(len(row_list)):
        value_list = [(i, row_list[i])] #Sets index value for each real sediment
        if row_list[i] == 1:
            new_y_test.append(i) #Puts the sediment value into a list 

    ###print([list((i, row_list[i])) for i in range(len(row_list))]) 
    
#Sediment Labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#Confusion Matrix
confusion = confusion_matrix(new_y_test, y_pred) #Creates binary array matrix for pedicted and acutal values
fig, ax = plot_confusion_matrix(conf_mat=confusion,colorbar=True,class_names=labels) #Crates Plot
ax.set_ylim(len(confusion)-0.5, -0.5)
print(confusion) #Prints binary matrix
plt.show() #Shows Plot


###Prints our loss and accuracy numbers after training and testing
###scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
###print("Loss: " + str(scores[0]))
###print("Accuracy: " + str(scores[1]))