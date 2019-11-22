import os
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#Define the paths to the respective data sets
training_folder = r"\Data\Train"
validation_folder  = r"\Data\Test"

#Define the path to the sentiment
path_to_sentiment = r"\Data\csv_file.csv"
#Grab the first row of the csv file

#Find the number of training and validation images we have
num_training_imgs = len(os.listdir(training_folder))
num_of_valid_imgs = len(os.listdir(validation_folder))
print("Total Training Images  : ",num_training_imgs)
print("Total Validation Images: ",num_training_imgs)

batch_size = 128
epochs = 15
IMG_HEIGHT = 48
IMG_WIDTH = 48

#Create generators to process the data such that it is tensor ready
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

#Read in the data from disk and put into tensor ready form
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=training_folder,shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH),class_mode='binary')
val_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_folder,shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH),class_mode='binary')

#Show some of the training data
sample_training_images, _ = next(train_data_gen)

#The actual neural net with dropout
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
#Compile the above neural net
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Take a look at the overview of the model
model.summary()

#Train the network
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#View the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()