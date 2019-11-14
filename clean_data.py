from skimage import io
import cv2
import csv
import pandas as pd
import urllib
import getpass

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

'''
Given the FEC dataset, check whether the URL contains a face and if not remove the row from the csv
'''
def clean_data(data):
    AFK
    '''
    Some image links appear in more then one row. In order to optimize the time taken in determining
    whether to delete a row we build a cache to store the result of CheckForFace where the url is the
    key and a boolean is the value.
    '''
    link_dict = {}                                 #Create the cache
    for index, row in data.iterrows():             #Iterate over all the rows in the csv 
        for url in row:                            #Iterate over the links in each row
            if url not in link_dict.keys():        #If the url is not in the cache
                link_dict[url] = CheckForFace(url) #Add it to the cache
        ###print(row[0], row[1], row[2])           ###DEBUGGING: Check
        if (link_dict[row[0]] and link_dict[row[1]] and link_dict[row[2]]) != True:  #If any of the urls do not contain a face
            data = data.drop([data.index[index]])  #Delete the row
            print("Row {} Deleted".format(index))

    return data #Return the cleaned csv data

#export_csv = df.to_csv (r'C:\Users\{}\Desktop\[Cleaned]text_emotions.csv'.format(getpass.getuser()), index = None, header=True)


def MCP():
    #Reads csv data files
    dataTrain = pd.read_csv("Data\\faceexp-comparison-data-train-public.csv", error_bad_lines=False)
    dataTest = pd.read_csv("Data\\faceexp-comparison-data-test-public.csv", error_bad_lines=False)
    ###Grab the urls to the images
    ###df = dataTrain.iloc[:,[0,5,10]]
    '''
    The problem with passing clean_data "dataTrain.iloc[:,[0,5,10]]" is it won't modify dataTrain accordingly.
    We need to pass it the full dataset and then reference columns A,F, and K.
    '''
    #dataTrain = clean_data(training_data)

MCP()