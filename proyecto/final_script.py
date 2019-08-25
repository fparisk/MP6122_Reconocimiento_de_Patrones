import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as k
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

from IPython.display import Image

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import sys
import time

import cv2

import warnings
warnings.filterwarnings('ignore')

# globals
text_coordinates = (380,440)
font = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (255,255,255)
dim = (48,48)
emotion_classes = ['Sad','Happy','Neutral']
emo_status = 'Neutral'

# load model (.h5 format)
adaptNet_model = models.load_model('adaptNet_model_v02.h5')

# init gstreamer
def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=820, display_height=616, framerate=21/1, flip_method=2):
    return('nvarguscamerasrc !'
    'video/x-raw(memory:NVMM),'
    'width=(int)%d, height=(int)%d,'
    'format=(string)NV12, framerate=(fraction)%d/1 !'
    'nvvidconv flip-method=%d !'
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert !'
    'video/x-raw, format=(string)BGR ! appsink' % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

# Returns input (X) and respective labels (Y) from dataset
def get_in_out_data(dataset):
    x = list(dataset["pixels"])
    X = []
    
    for i in range(len(x)):
        each_pixel = [int(num) for num in x[i].split()]
        X.append(each_pixel)
        
    X = np.array(X)
    X = X.reshape(X.shape[0], 48, 48,1)  # reshape to a 48x48 grayscale (1 pixel) format
    X = X.astype("float32")
    X /= 255 # normalize
    
    Y = dataset.emotion.values
    Y = np_utils.to_categorical(Y)
    
    return X,Y  

# runs model
def classufyImage(x_toClassify):
    print ("\n")
    print ("---> Classification started.........")
    print ("\n")

    # classify emotion using loaded model and captured data
    predicted_data = adaptNet_model.predict(x_toClassify)

    print ("\n")
    print ("Classification results: \n")
    print(predicted_data)
    print ("\n")

    if (predicted_data.size > 0):
        indx = np.where(predicted_data[0] == np.amax(predicted_data[0]))
        max_indx = int(indx[0])
        return emotion_classes[max_indx] 
    else:
        return 'Neutral'   
    

# returns region of interes
def getRegionOfInterest(img):
    roi = img[120:360, 200:440]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, dim, interpolation = cv2.INTER_AREA)
    return roi

# converts cv Mat to a 48x48x1 Numpy array 
def cvMatToNumpyArray(cvMat):
    npArray = np.asarray(cvMat)
    npArray = npArray[np.newaxis, :, :, np.newaxis]
    npArray = npArray.astype("float32")
    npArray /= 255 # normalize

    print ("\n")
    print(npArray.shape)
    print ("\n")

    return npArray
      

# read camera data
def read_cam():
    emo_status = 'Neutral'

    try:
        # Modified
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0,
                                                  capture_width=640, 
                                                  capture_height=480, 
                                                  display_width=640, 
                                                  display_height=480), 
                                cv2.CAP_GSTREAMER)

        if cap.isOpened():
            window_handle = cv2.namedWindow('Maestria Electronica', cv2.WINDOW_AUTOSIZE)

            while cv2.getWindowProperty('Maestria Electronica',0)>=0:
                # read camera dataframe
                ret_val, img = cap.read()

                # add status label and "hot zone" boundaries
                cv2.putText(img, emo_status, text_coordinates, font, 1, fontColor, 2)
                img = cv2.rectangle(img,(200,120),(440,360),(0,255,0),3)

                # display image
                cv2.imshow('demo',img)
                
                keyCode = cv2.waitKey(30) & 0xff

                # capture analysis zone if space is pressed and classify it
                if keyCode == 32:
                    roi = getRegionOfInterest(img)
                    #cv2.imwrite('/home/tec/MP6122_Reconocimiento_de_Patrones/proyecto/test2.png', roi)                    
                    x_toClasify = cvMatToNumpyArray(roi)
                    emo_status = classufyImage(x_toClasify)
                
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
                
        else:
            #print(cv2.getBuildInformation())
            print("camera open failed")
            cv2.destroyAllWindows()

    except Exception as e:
        print(str(e))
        exit(1)

if __name__ == '__main__':
    print("----> START .............")
    read_cam()    
    #show_camera()
