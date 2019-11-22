import sys, os
import pandas as pd
import numpy as np
import cv2
from nltk import word_tokenize
import matplotlib.pyplot as plt
import time

#Adapted From:
#https://www.kaggle.com/gemyhamed/facial-expression-of-emotions

#Imports Data
df = pd.read_csv('C:\\Users\\natet\\Desktop\\Test\\fer2013.csv')
print(df.head())

#Gets the pixels out of the dataset
pixels = df.loc[:,'pixels'].values
print(pixels.shape)
print(type(pixels))

#Transforms pixels to right format
px = []
for x in pixels : 
    x = word_tokenize(x)
    x = [float(t) for t in x]
    px.append(x)

#Sets x values
x = np.array(px)
print(x.shape)

#Sets y values
y = df.loc[:, 'emotion'].values
print(y.shape)
print(type(y))

#Loops through entire dataset and picks out each rows pixels and saves to am image
startTime = time.time()
for ix in range(len(x)-1):
    ###For sentiment, read the pics into a list, read the sentiments into a list, 
    ### and use their position in the list as a mapping
    image = plt.figure(ix)
    ax = plt.Axes(image, [0., 0., 1., 1.])
    ax.set_axis_off()
    image.add_axes(ax)
    ax.imshow(x[ix].reshape((48, 48)), interpolation='none', cmap='gray')
    #Saves Train Data
    if ix <= 28710:
        image.savefig('Train_Data\\figure_{}.png'.format(ix))
        plt.close()
    #Saves Test Data
    else:
        image.savefig('Test_Data\\figure_{}.png'.format(ix))
        plt.close()
totalTime = time.time() - startTime
print(totalTime)