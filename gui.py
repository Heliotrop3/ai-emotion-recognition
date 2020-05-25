import re
import ctypes
import cv2
import sys
import dlib
import imutils
import numpy as np
from tkinter import *
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model

#Try to import the filedialog lib and catch for older versions of python
try :
    from tkinter import filedialog
except ImportError:
    import tkFileDialog as filedialog

def donothing() -> None:
   """A placeholder function that does absolutely nothing"""
   filewin = Toplevel(root)
   button = Button(filewin, text="Do nothing button")
   button.pack()

def find_file_extension(file_path: str) -> str:
   """
   Returns the extension of the object at the end of the file path

   Parameters
   ----------
   file_path : str
      A string representation of the location of the object the user
      is attempting to upload

   Returns
   -------
   str
      The string of characters after the final
      period of the passed the passed string 
      
   """
   filename = re.split('/', file_path)[-1].lower()
   extension = filename.split('.',1)[1]
   ###print(extension)
   return extension

def message_box(title: str,
                text : str,
                style: int) -> int:
   """
   Display a message in a pop-up box to the user

   Parameters
   ----------
   title : str
      The message to be displayed above the menu bar

   text : str
      The message to be displayed in the main body of the message box

   style : int
      The number corresponding with a layout of buttons to present
      to the user to deal with the pop-up.
      
   Returns
   -------
   int
      Returns the integer value of the associated return code for the
      button selected by the user.

   Note
   ----
   See the related Microsoft documentation for more information
   about the MessageBoxW function.
   https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-messageboxw

   """
   return ctypes.windll.user32.MessageBoxW(0, text, title, style)


def open_file() -> None:
    """
    Passes the path to an image of a valid image type to the nueral net

    """
    # Restrict the filetype to accept only certain file types.
    # 
    ftypes = [('JPG', '*.jpg'),('PNG','*.png')]
    
    #Grab a list of valid file extensions
    image_types = [ftypes[i][1][2:] for i in range(len(ftypes))]
    ##DEBUGGING: See the files the program has identified as valid
    ##print(image_types)
    
    # Trap the user in a loop until they either
    # choose a valid file or exit the gui prompt
    is_valid = False
    while(is_valid == False):
        file_path = filedialog.askopenfilename(filetypes=ftypes)
        
        # If filePath is null then the user has hit cancel
        # and we want to break free from the while loop
        if (file_path == ''):
           is_valid = True
        else:

            # Our ftypes list still allow other input such as internet
            # shortcuts.  To counter this we check the file extension
            # and throw a popup box letting the user know if their
            # selected file is not a valid file type.
            #Grab the extension of the file
            extenstion = find_file_extension(file_path)
            ###DEBUGGING: See the file extension
            ###print(extenstion)
        
            if extenstion not in image_types:
               message_box("Error: Invalid File Type",
                           ("Uploaded files must either be a jpg or png"
                            "\n Please try again. "
                            ),
                           0 # Only give the user the "Ok" button
                           )
            else:
                try:
                        print("File Uploaded!")
                        # Pass the image off to the neural net
                        detect_emotion_picture(file_path)
                        
                        # Break from the while loop
                        is_valid = True
                        

                # Catch the case where the user requires elevated privliges
                # to upload a given file.  
                except PermissionError:
                    print("Permission Error: Access Denied")
                    message_box("Error: Permission Denied",
                                ("Operation requires elevated privilege"
                                 "\nTry runnin the program as an admin"
                                 ),
                                0
                                )
                    sys.exit(0)

def face_exists(path_to_image: str) -> bool:
    """
    Returns True if a face is present otherwise returns False

    Parameters
    ----------
    path_to_image : str
        The file path to the location of where the image is stored

    Returns
    -------
    bool
        True if the image contains a face which is detectable by the
        pre-trained convolutional nueral net (CNN)

    Note
    ----
    A pre-trained CNN is used to detect whether or not an image contains
    a face.  The CNN file is mmod_human_face_detector.dat.

    """
    image = Image.open(open(path_to_image,'rb'))

    # Convert the image to the proper color schema
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # If the image is not greyscale then convert it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # If no image exists then return False.  This shouldn't happen
    # but it's not impossible.
    if image is None:
        print("Could not read input image")
        return False

    #Initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    #Apply face detection (cnn)
    faces_cnn = cnn_face_detector(image, 1)
    
    if faces_cnn:
        print("There exists a face!")
        return True
    else:
        print("There exists no faces :(")
        return False

#Adapted from:
#https://github.com/gitshanks/fer2013/blob/master/fertestcustom.py
def detect_emotion_picture(file_path):
    """

    """
    print("Working...")
    #Load the model
    model = load_model(r"Models\model_64_40Epochs.h5")
    labels = [
              'Angry',
              'Disgust',
              'Fear',
              'Happy',
              'Sad',
              'Surprise',
              'Neutral'
              ]
    
    #Load image
    image = cv2.imread(file_path)

    #Resize image if it is too large
    if image.shape[0] > 1000 or image.shape[1] > 2000:
        
        #Scale the image x% of its original size
        scale_percent = 25
        width = int(image.shape[1]
                    * scale_percent
                    / 100)
        

        height = int(image.shape[0]
                     * scale_percent
                     / 100
                     )
        dim = (width,
               height)                              #Set the dimension
        
        #Resize the image according to the above definitions
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    #Convert the image to grayscale
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #Use CV2's classifier to determine whether a face exists in the photo
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

    #Grab all detected faces
    faces = face.detectMultiScale(gray,
                                  1.3,
                                  10
                                  )                      

    #Draw a square around the face 
    for (x, y, w, h) in faces:
        
            roi_gray = gray[y:y + h, x:x + w]
            
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img,
                          cropped_img,
                          alpha=0,
                          beta=1,
                          norm_type=cv2.NORM_L2,
                          dtype=cv2.CV_32F
                          )
            
            cv2.rectangle(image,
                          (x, y
                           ),
                          (x + w,
                           y + h
                           ),
                          (0, 255, 0
                           ),
                          1
                          )
            
            #Predict the emotion
            yhat= model.predict(cropped_img)

            #Label the emotion
            cv2.putText(image,
                        labels[
                               int(np.argmax(yhat)
                                   )
                               ],
                        (x, y
                         ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0
                         ),
                        1,
                        cv2.LINE_AA
                        )
            
            print("Emotion: "+labels[int(np.argmax(yhat))])

    #Shows image
    cv2.imshow('{} Picture Emotion'.format(file_path),
               image)
    
    cv2.waitKey()


def detect_emotion_webcam() -> None:
    """
    Opens the webcam and provides real-time emotion recognition
    
    """
    
    # Define the range of emotions 
    emotion_dict = {0: "Angry",
                    1: "Disgust",
                    2: "Fear",
                    3: "Happy",
                    4: "Sad",
                    5: "Surprise",
                    6: "Neutral"
                    }

    # Load the emotion recognition model
    model = load_model(r"Models\model_64_40Epochs.h5")

    # Grab the webcam device
    cap = cv2.VideoCapture(0)
   
    hasOpened = False
    # Capture the input from the webcam until
    # the user exits out of the program
    while True:

       if not hasOpened:
          print("Webcam Opened!")
          print("To close webcam press the q key on your keyboard!")
          hasOpened = True

       ret, frame = cap.read()
       # Convert to grayscale
       gray = cv2.cvtColor(frame,
                           cv2.COLOR_BGR2GRAY)
       #Detect the faces
       face_cascade = cv2.CascadeClassifier(("haarcascade_"
                                            "frontalface_"
                                            "default.xml"
                                             )
                                            )

       # Define where the faces are on the image
       faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      
       #Draw a square around the boxes
       for (x, y, w, h) in faces:
          
           cv2.rectangle(frame,
                         (x, y),
                         (x + w,
                          y + h),
                         (0, 255, 0),
                         1
                         )
          
           roi_gray = gray[
                           y : y
                               + h,
                           x : x
                               + w
                           ]

           # Expand what the actual **** is going on with this line.
           # Too condensed.
           cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
          
           cv2.normalize(cropped_img,
                         cropped_img,
                         alpha=0,
                         beta=1,
                         norm_type=cv2.NORM_L2,
                         dtype=cv2.CV_32F
                         )
           
           #Predict the emotion
           prediction = model.predict(cropped_img)
            
           #Put the predicted emotion text into the proper box
           cv2.putText(frame,
                       emotion_dict[
                                    int(np.argmax(prediction)
                                        )
                                    ],
                       (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8,
                       (0,
                        255,
                        0
                        ),
                       1,
                       cv2.LINE_AA
                       )

           cv2.imshow(("Webcam Emotion Face Recognition ("
                       "Use Q on Keyboard to Quit)"
                       ),
                       frame
                       )
               
           if cv2.waitKey(1) & 0xFF == ord('q'):
               cap.release()
               cv2.destroyAllWindows()
               print("Webcam Closed!")
               return
    

#Create the lowest view instnace of tkinter
root = Tk()

#Set the size of the window on startup
root.geometry("700x700")

#Make the user unable to resize tkinter window
root.resizable(0,0)

#Define the background color
backgroundColor='#89cff0'
#Set the background color
root.configure(background=backgroundColor)

# Add a view on the root instance
main_window = PanedWindow(orient=VERTICAL)
# Set the background function for the main window
main_window.configure(background=backgroundColor)
# 
main_window.pack()                                


# This section adds the text, buttons, and other garnishing for the gui

# Define and add the title of the project to the main window
title = Label(main_window,
              text="Aspect 1.1",
              wraplength=650,
              justify=LEFT,
              background=backgroundColor,
              font=("arial",
                    24
                    )
              )
main_window.add(title)

# Define and add the authors of the project underneath the title
authors = Label(main_window,
                background=backgroundColor,
                text="Nate, Tyler, Terry, Josh A., Josh S.",
                wraplength=650,
                justify=LEFT,
                font=("arial",
                      10
                      )
                )
main_window.add(authors)

# Add a space between the authors and the description
space = Label(main_window,
              background=backgroundColor,
              text="",
              wraplength=650,
              justify=LEFT,
              font=("arial",
                    14
                    )
              )
main_window.add(space)

# Define and add the description of the project
description = Label(main_window,
                    background=backgroundColor,
                    text=("Aspect 1.1 uses CV2's face recognition to "
                          "detect a face and then uses our trained "
                          "model from the fer2013 dataset to detect "
                          "emotion present in the faces.  Aspect 1.1 "
                          "will take an image or video feed and draw "
                          "a box around the face (CV2) and then will "
                          "show the emotion of the faces present. To "
                          "use this app you can either upload your own "
                          "image or use your computers webcam to get a "
                          "live video feed with the emotion detection."
                          ),
                          wraplength=650,
                          justify=LEFT,
                          font=("arial",
                                12
                                )
                          )
main_window.add(description)

# Add a space between the description and the picture label
space = Label(main_window,
              background=backgroundColor,
              text="",
              wraplength=650,
              justify=LEFT,
              font=("arial",
                    14
                    )
              )
main_window.add(space)

# Add text for the photo upload button
pic_label = Label(main_window,
                  background=backgroundColor,
                  text=("Click the button below to "
                        "upload your own photo:"
                        ),
                  wraplength=650,
                  justify=LEFT,
                  font=("arial",
                        15
                        )
                  )
main_window.add(pic_label)

# Add the button for uploading a photo
top = Button(main_window,
             background='#89bda0',
             text="Upload Photo",
             width=4, height=4,
             justify=CENTER,
             font=("arial",
                   20
                   ),
             command = open_file
             )
main_window.add(top)

# Add the text for the button for using the webcam
webcam_label = Label(main_window,
                     background=backgroundColor,
                     text="Click the button below to use your webcam:",
                     wraplength=650,
                     justify=LEFT,
                     font=("arial",
                           15
                           )
                     )
main_window.add(webcam_label)

# Add the button for utilizing the webcam for real time emotion recognition
bottom = Button(main_window,
                background='#c482ad',
                text="Webcam",
                width=4,
                height=4,
                justify=CENTER,
                font=("arial",
                      20
                      ),
                command = detect_emotion_webcam
                )
main_window.add(bottom)

# Add a space between the webcam button and the closing note
space = Label(main_window,
              background=backgroundColor,
              text="",
              wraplength=650,
              justify=LEFT,
              font=("arial",
                    14
                    )
              )
main_window.add(space)

# Add a hobbit size note regarding about user's
# privacy (i.e We are not storing their face)
note = Label(main_window,
             background=backgroundColor,
             text="**No user data is saved in using this app**",
             wraplength=650,
             justify=LEFT,
             font=("arial",
                   10
                   )
             )
main_window.add(note)


# Create a file menu button that will allow the
# user the same functionality as the main program.
# The labels should explain whats happening.

# Create the file menu object
menubar = Menu(root)
filemenu = Menu(menubar,
                tearoff=0
                )

# Add the commands to the file menu object
filemenu.add_command(label="Upload Photo",
                     command=open_file
                     )
filemenu.add_separator()

filemenu.add_command(label="Use Webcam",
                     command=detect_emotion_webcam
                     )
filemenu.add_separator()

filemenu.add_command(label="Exit",
                     command=root.quit
                     )

menubar.add_cascade(label="Menu",
                    menu=filemenu
                    )

root.title("Aspect 1.1")  #Set the title for the window
root.config(menu=menubar) #Add the menu bar
root.mainloop()           #Start the window

