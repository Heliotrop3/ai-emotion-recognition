from fastai import *
from fastai.text import *
from fastai.core import *
import tensorflow
import torch
import cv2
import csv
import pandas as pd

#path = 'C:\\Users\\%USERNAME\\Documents\\GitHub\\ai-emotion-recognition\\Data'

#Reads csv data files
dataTrain = pd.read_csv("Data\\faceexp-comparison-data-train-public.csv", error_bad_lines=False)
dataTest = pd.read_csv("Data\\faceexp-comparison-data-test-public.csv", error_bad_lines=False)

fiveDataTrain = dataTrain.head()
print(fiveDataTrain)

fiveDataTest = dataTest.head()
print(fiveDataTest)