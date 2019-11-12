from skimage import io
import cv2
import csv
import pandas as pd

#Reads csv data files
dataTrain = pd.read_csv("Data\\faceexp-comparison-data-train-public.csv", error_bad_lines=False)
dataTest = pd.read_csv("Data\\faceexp-comparison-data-test-public.csv", error_bad_lines=False)


def CheckForFace(url):
    image = io.imread(url)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('URL Image', image)
    #cv2.waitKey()

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    if ((len(faces)) == 0):
        print("FASLE")
        #return False
    else:
        print("TRUE")
        #return True


#print(dataTrain.iloc[:,[0,5,10]])

url = "http://farm6.staticflickr.com/5033/14414157667_98383c7f1c_b.jpg"

CheckForFace(url)