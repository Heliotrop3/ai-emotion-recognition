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

#Placeholder Function
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

def open_file():
    #Restrict the filetype to accept only texts
    ftypes = [('JPG', '*.jpg'),('PNG','*.png')]
    #Grab a list of valid file extensions
    image_types = [ftypes[i][1][2:] for i in range(len(ftypes))]
    ###print(image_types)
    is_valid = False
    while(is_valid == False):
        file_path = filedialog.askopenfilename(filetypes=ftypes)
        '''
        If filePath is null then the user has hit cancel and we want to break
        free from the while loop
        '''
        if (file_path == ''):
            ###print("Filepath is false")
            is_valid = True
        else:
            ###print(file_path)
            '''
            ftypes still allow other input such as internet shortcuts so we're
            going to have to check the file extension and throw an error for
            any file that doesn't end with a .txt
            '''
            filename,extenstion = find_file_name(file_path)
            ###print(extenstion)
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
    image = Image.open(open(path_to_image,'rb'))

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
    #Loads model
    model = load_model(r"Models\model_64_40Epochs.h5")

    #sets variables
    WIDTH = 48
    HEIGHT = 48
    x=None
    y=None
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    #Load image
    image = cv2.imread(file_path)   ######"test.jpg"

    #Resize image if it is too large
    if image.shape[0] > 1000 or image.shape[1] > 2000:
        scale_percent = 25 #percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    #CV2 Stuff for face reconition
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, 1.3  , 10)

    #CV2 Ditects face and our model predicts emotion on face
    for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #predicting the emotion
            yhat= model.predict(cropped_img)
            cv2.putText(image, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            print("Emotion: "+labels[int(np.argmax(yhat))])

    #Shows image
    cv2.imshow('{} Picture Emotion'.format(file_path), image)
    cv2.waitKey()

def detect_emotion_webcam():
    print("Opening Webcam...")
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    model = load_model(r"Models\model_64_40Epochs.h5")
    run = True
    cap = cv2.VideoCapture(0)
    hasOpened = False
    while run:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            prediction = model.predict(cropped_img)
            cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        opened = True
        if opened and not hasOpened:
            print("Webcam Opened!")
            hasOpened = True

        cv2.imshow('Webcam Emotion Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            run = False
            print("Webcam Closed!")

    cap.release()
    cv2.destroyAllWindows()

#######################################
#Create the instnace of tkinter
root = Tk()
menubar = Menu(root)
#Make canvas scalable
main_window = PanedWindow(orient=VERTICAL)
main_window.pack(fill=BOTH, expand=1)

top = Button(main_window, text="Upload Photo", command = open_file)
main_window.add(top)

bottom = Button(main_window, text="Webcam",command = detect_emotion_webcam)
main_window.add(bottom)

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Upload Photo", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Use Webcam", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
filemenu.add_separator()
menubar.add_cascade(label="Aspect 1.1", menu=filemenu)
########################################


#Name the title bar
root.title("Aspect")
root.config(menu=menubar)
#Start the loop
root.mainloop()

