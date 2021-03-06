import sys, os, getpass
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model

#Adapted from:
#https://medium.com/@birdortyedi_23820/deep-learning-lab-episode-3-fer2013-c38f2e052280

#Sets up file paths
BASEPATH = r'C:\Users\{}\Documents\GitHub\ai-emotion-recognition'.format(getpass.getuser())
sys.path.insert(0, BASEPATH)
os.chdir(BASEPATH)
MODELPATH = './models/model.h5'

#Defines Variables
num_features = 64
num_labels = 7
batch_size = 64
epochs = 40
width, height = 48, 48  #Size of images in fer2013.csv

#Extracts data from csv and prints the last 5 values
data = pd.read_csv(r'{}\Data\fer2013.csv'.format(BASEPATH))
print(data.tail())


pixels = data['pixels'].tolist() #Converts pixels to a list of each row

#Converts pixels in to RGB values in an array
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


#Model with Relu as activation function
model = Sequential()
#Conv2D layer added
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
#Conv2D layer added
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
#Conv2D layer added
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
#Conv2D layer added
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
#Conv2D layer added
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
#Conv2D layer added
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
#Conv2D layer added
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
#Conv2D layer added
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
#Takes the above shape and turns it into a one dimensional tensor
model.add(Flatten())    
#Dense layer added
model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
#Dense layer added
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
#Dense layer added
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))
#Dense layer added
model.add(Dense(num_labels, activation='softmax'))

print(model.summary())

#Compiles model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

#Improves the loss of our data
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

#Creates log files while training
tensorboard = TensorBoard(log_dir=r'{}\Logs'.format(BASEPATH))

#Stops training if there is no change in the loss
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

#Saves model
checkpointer = ModelCheckpoint(MODELPATH, monitor='val_loss', verbose=1, save_best_only=True)

#Trains model
model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_test), np.array(y_test)),
          shuffle=True,
          callbacks=[lr_reducer, tensorboard, early_stopper, checkpointer])

#Prints our loss and accuracy numbers after training and testing
scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))