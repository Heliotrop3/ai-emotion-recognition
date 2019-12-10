from tkinter import *
import re, ctypes, cv2, sys, dlib, imutils
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#import aspect

#Try to import the filedialog lib and catch for older versions of python
try :from tkinter import filedialog
except ImportError: import tkFileDialog as filedialog

#A Placeholder function that does nothing...
def donothing():
   filewin = Toplevel(root)
   button = Button(filewin, text="Do nothing button")
   button.pack()

'''
Given the path to the file, return the name of the file
'''
def find_file_name(file_path):
   filename = re.split('/', file_path)[-1]
   extension = filename.split('.',1)[1]
   ###print(extension)
   return filename, extension

'''
Query the user to open a text file.
If the user attempts to open anything other than a text file throw an error and try again
'''
def message_box(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

'''
Prompt the user to select a file from their hard drive. If the file is valid then process the file
otherwise keep them trapped in the file prompt until they play by the rules or exit the program
'''
def open_file():
    #Restrict the filetype to accept only texts
    ftypes = [('JPG', '*.jpg'),('PNG','*.png')]
    #Grab a list of valid file extensions
    image_types = [ftypes[i][1][2:] for i in range(len(ftypes))]
    ##DEBUGGING: See the files the program has identified as valid
    ###print(image_types)
    is_valid = False
    #Trap the user in a loop until they either choose a valid file or exit the gui prompt
    while(is_valid == False):
        file_path = filedialog.askopenfilename(filetypes=ftypes)
        '''
        If filePath is null then the user has hit cancel and we want to break
        free from the while loop
        '''
        if (file_path == ''):
           is_valid = True
        else:
            '''
            ftypes still allow other input such as internet shortcuts so we're
            going to have to check the file extension and throw an error for
            any file that doesn't end with a .txt
            '''
            #Grab the extension of the file
            filename,extenstion = find_file_name(file_path)
            ###DEBUGGING: See the file extension
            ###print(extenstion)
            '''
            If the extension is not one of the valid extensions we attempt to read the file in.
            We catch the case where the user requires elevated privliges to upload a given file.

            If the file is not of a valid image type, display a popup box letting the user know as much
            and forcing them to choose again.
            '''
            if extenstion not in image_types:
               message_box("Error: Invalid File Type","Chosen file is not a text file\nPlease try again. ",0)
            else:
                try:
                    with open(file_path, 'r+', errors='ignore') as f:
                        text = f.read()     
                        print("File Uploaded!")           
                    is_valid = True
                except PermissionError:
                    print("Error: Operation requires elevated privilege\nTry runnin the program as an admin")
                    sys.exit(0)
    return detect_emotion_picture(file_path)

'''
Use a pre-trained CNN to detect whether or not a face exists
'''
def face_exists(path_to_image):
    image = Image.open(open(path_to_image,'rb'))   #Open the image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  #Convert the image to the proper color schema
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #If the image is not greyscale then convert it
    if image is None:                              #If no image exists then return False
        print("Could not read input image")
        return False

    #Initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1(r"C:\Users\T\Desktop\test\mmod_human_face_detector.dat")
    faces_cnn = cnn_face_detector(image, 1) #Apply face detection (cnn)

    if faces_cnn:                           #Return True if a face exists and false if no face exists
        print("There exists a face!")
        return True
    else:
        print("There exists no faces :(")
        return False

#Adapted from:
#https://github.com/gitshanks/fer2013/blob/master/fertestcustom.py
def detect_emotion_picture(file_path):
    print("Working...")
    model = load_model(r"Models\model_64_40Epochs.h5")  #Load the model
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    image = cv2.imread(file_path)                       #Load image

    #Resize image if it is too large
    if image.shape[0] > 1000 or image.shape[1] > 2000:
        scale_percent = 25                                 #Scale the image x% of its original size
        width = int(image.shape[1] * scale_percent / 100)  #Define the width
        height = int(image.shape[0] * scale_percent / 100) #Define the height
        dim = (width, height)                              #Set the dimension
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #Resize the image according to the above definitions

    #Use CV2 to check for a face
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)                         #Convert the image to grayscale
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Use CV2's classifier to determine whether a face exists in the photo
    faces = face.detectMultiScale(gray, 1.3  , 10)                      #Grab all detected faces

    #Draw a square around the face 
    for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #Predict the emotion
            yhat= model.predict(cropped_img)
            #Label the emotion
            cv2.putText(image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            print("Emotion: "+labels[int(np.argmax(yhat))])

    #Shows image
    cv2.imshow('{} Picture Emotion'.format(file_path), image)
    cv2.waitKey()

'''
Use the model to predict emotion of multiple faces in real time
'''
def detect_emotion_webcam():
    print("Opening Webcam...")
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"} #Define the possible emotions
    model = load_model(r"Models\model_64_40Epochs.h5")   #Load the model              
    cap = cv2.VideoCapture(0)                            #Grab the webcam device
    hasOpened = False
    '''
    Capture the input from the webcam until the user exits out of the program
    '''
    while True:
        ret, frame = cap.read()                         #Grab the frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Convert to grayscale

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Define where the faces are on the image

        #Draw a square around the boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            #Predict the emotion
            prediction = model.predict(cropped_img)
            #Put the predicted emotion text into the proper box
            cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        if not hasOpened:
            print("Webcam Opened!")
            print("To close webcam press the q key on your keyboard!")
            hasOpened = True

        cv2.imshow('Webcam Emotion Face Recognition     Use Q on Keyboard to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Webcam Closed!")
            break

    cap.release()
    cv2.destroyAllWindows()

#######################################
root = Tk()               #Create the instnace of tkinter
root.geometry("700x700")  #Set the size of the window on startup
root.resizable(0,0)       #Make the user unable to resize tkinter window
backgroundColor='#89cff0' #Define the background color
root.configure(background=backgroundColor) #Set the background color

menubar = Menu(root)                        #Create the menu object
main_window = PanedWindow(orient=VERTICAL)  #Make canvas scalable
main_window.configure(background=backgroundColor) #Set the background function for the main window
main_window.pack()                                

'''
This section adds the text, buttons, and other garnishing for the gui
'''
#Adds the title of the project
title = Label(main_window, text="Aspect 1.1",wraplength=650,justify=LEFT,background=backgroundColor,font=("arial",24))
main_window.add(title)

#Add the authors of the project
authors = Label(main_window,background=backgroundColor, text="Nate, Tyler, Terry, Josh A., Josh S.",wraplength=650,justify=LEFT,font=("arial",10))
main_window.add(authors)

#Add a space between the authors and the description
space = Label(main_window,background=backgroundColor, text="",wraplength=650,justify=LEFT,font=("arial",14))
main_window.add(space)

#Add the description of the project
description = Label(main_window,background=backgroundColor, text="Aspect 1.1 uses CV2's face recognition to detect a face and then uses our trained model from the fer2013 dataset to detect emotion present in the faces.  Aspect 1.1 will take an image or video feed and draw a box around the face (CV2) and then will show the emotion of the faces present.  To use this app you can either upload your own image or use your computers webcam to get a live video feed with the emotion detection.", wraplength=650,justify=LEFT ,font=("arial",12))
main_window.add(description)

#Add a space between the description and the picture label
space = Label(main_window,background=backgroundColor, text="",wraplength=650,justify=LEFT,font=("arial",14))
main_window.add(space)

#Add text for the phot upload button
pic_label = Label(main_window,background=backgroundColor, text="Click the button below to upload your own photo:",wraplength=650,justify=LEFT,font=("arial",15))
main_window.add(pic_label)

#Add the button for uploading a photo
top = Button(main_window,background='#89bda0', text="Upload Photo", width=4, height=4,justify=CENTER, font=("arial",20), command = open_file)
main_window.add(top)

#Add the text for the button for using the webcam
webcam_label = Label(main_window,background=backgroundColor, text="Click the button below to use your webcam:",wraplength=650,justify=LEFT,font=("arial",15))
main_window.add(webcam_label)

#Add the button for utilizing the webcam for real time emotion recognition
bottom = Button(main_window,background='#c482ad', text="Webcam",width=4, height=4,justify=CENTER,font=("arial",20),command = detect_emotion_webcam)
main_window.add(bottom)

#Add a space between the webcam button and the closing note
space = Label(main_window,background=backgroundColor, text="",wraplength=650,justify=LEFT,font=("arial",14))
main_window.add(space)

#Add a hobbit size note regarding about user's privacy (i.e We are not storing their face)
note = Label(main_window,background=backgroundColor, text="**No user data is saved in using this app**",wraplength=650,justify=LEFT,font=("arial",10))
main_window.add(note)

'''
Create a file menu button that will allow the user the same functionality as the main program
The labels for the add_command function are pretty self-explanatory...
'''
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Upload Photo", command=open_file)
filemenu.add_separator()
filemenu.add_command(label="Use Webcam", command=detect_emotion_webcam)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="Menu", menu=filemenu)
########################################

root.title("Aspect 1.1")  #Set the title for the window
root.config(menu=menubar) #Add the menu bar
root.mainloop()           #Start the window

