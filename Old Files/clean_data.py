import cv2, dlib, csv, urllib, os
import pandas as pd
import numpy as np
from skimage import io

'''
Checks whether or not a face is present in the image

Source: https://gist.github.com/arunponnusamy/7013c8617f97937250cb2c2de57c9b11
'''
def face_exists(url):
    try:
        image = io.imread(url)      #Grab the image located at the url
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Convert the image to the proper color schema
       
        #If the image is not greyscale then convert it
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image is None:
            print("Could not read input image")
            return False

        #Initialize cnn based face detector with the weights
        cnn_face_detector = dlib.cnn_face_detection_model_v1(r"\mmod_human_face_detector.dat")

        #Apply face detection (cnn)
        faces_cnn = cnn_face_detector(image, 1)

        #If there exists a face
        if faces_cnn:
            print("There exists a face!")
            return True
        else:
            print("There exists no faces :(")
            return False
    except urllib.error.HTTPError: #In the case the url leads to a dead page
        print("Error: Dead Link")
        return False               #Then there exists no face at the link

'''
Save any modifications made to the data set in a new file into the github folder
'''
def save_csv(filename,data):
    path = r"Data\[Cleaned]{}.csv".format(filename)    #Define the path
    print("Saving {}.csv to {}".format(filename,path))
    data.to_csv(path, index = None, header=True)       #Save the data at path. Might have to use a try and except to catch if the file already exists
    print("File Saved")


def MCP():
    #Reads csv data files
    dataTrain = pd.read_csv("Data\\faceexp-comparison-data-train-public.csv", error_bad_lines=False)
    #dataTest = pd.read_csv("Data\\faceexp-comparison-data-test-public.csv", error_bad_lines=False)
    dataTrain = clean_data("[Cleaned] Training Data",dataTrain)
    #dataTest = clean_data(dataTest)
    save_csv("TrainData",dataTrain)
MCP()