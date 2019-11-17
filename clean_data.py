import cv2, csv, urllib,  os
import pandas as pd
import numpy as np
from skimage import io

'''
Checks whether or not a face is present in the image
'''
def CheckForFace(url):
    try:
        image = io.imread(url)                         #Grab the image located at the url

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
            ###print("FASLE")     ###DEBUGGING: See the results of the face detection
            return False
        else:
            #print("TRUE")        ###DEBUGGING: See the results of the face detection
            return True
    except urllib.error.HTTPError: #In the case the url leads to a dead page
        return False               #Then there exists no face at the link
    except cv2.error:              #An unknown error in image reconition
        print("cv2 Error")
        return False               #so we assume there is no face at this image

'''
Given the FEC dataset, check whether the URL contains a face and if not remove the row from the csv
'''
def clean_data(data):
    '''
    Some image links appear in more then one row. In order to optimize the time taken in determining
    whether to delete a row we build a cache to store the result of CheckForFace where the url is the
    key and a boolean is the value.
   
    Currently cleaning the entire training dataset would take just under 2  weeks. We may or may not decide
    to split the dataset into smaller chunks and have each team member responsible for cleaning a portion of
    the data. For now, we simply clean the first 5000 rows which gives us 15000 pictures to train on.
    '''
    i = 0   ### Initialize a counter to keep track of the number 
    numDeleted = 0                        #Initialize a counter to kep track of the number of deleted row
    data = data.iloc[:,[0,5,10]]          #Grab the columns where the urls for the image are stored
    link_dict = {}                        #Create the cache
    for index, row in data.iterrows():    #Iterate over all the rows in the csv 
        for url in row:                   #Iterate over the links in each row
            if url not in link_dict.keys():        #If the url is not in the cache
                link_dict[url] = CheckForFace(url) #Add it to the cache
        ###print(row[0], row[1], row[2])           ###DEBUGGING: Print each link
        if (link_dict[row[0]] and link_dict[row[1]] and link_dict[row[2]]) != True:  #If any of the urls do not contain a face
            data = data.drop([data.index[index]])  #Delete the row in the CSV
            print("Row {} Deleted".format(index))
            numDeleted += 1                        #Increment the deleted row counter by 1
        else:                            ###If all rows contain a face
            i += 1                       ###Increase the valid counter by 1 
        '''
        Once the dataset has 5000 valid rows of data we drop all rows after row 5000
        and save the new test dataset to the project's data folder
        '''
        if i == 5000:
            print("Last Row: {}".format(index+1))
            print("Number of Rows Deleted: {}".format(numDeleted))
            ###for j in range(26, len(data.index)):
             ###   data.drop(data.index[j])
             ###   print("Row {} Deleted".format(j))
            break
        
    return data #Return the cleaned csv data

'''
Save any modifications made to the data set in a new file into the github folder
'''
def save_csv(filename,data):
    path = r"Data\[Cleaned]{}.csv".format(filename)    #Define the path
    print("Saving {}.csv to {}".format(filename,path))
    data.to_csv(path, index = None, header=True)       #Save the data at path
    print("File Saved")


def MCP():
    #Reads csv data files
    dataTrain = pd.read_csv("Data\\faceexp-comparison-data-train-public.csv", error_bad_lines=False)
    dataTest = pd.read_csv("Data\\faceexp-comparison-data-test-public.csv", error_bad_lines=False)
    dataTrain = clean_data(dataTrain)
    #dataTest = clean_data(dataTest)
    save_csv("TrainData",dataTrain)
MCP()