from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
classifier=load_model('/home/ayush/ball_detect_cnn/my_model11.h5')
from keras.preprocessing import image
import numpy as np
import cv2
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    if(_):
        frame = cv2.resize(frame,(256,128))
        left_frame = np.split(frame,2,axis=1)[0]
        right_frame = np.split(frame,2,axis=1)[1]
        cv2.imwrite('left.jpg',left_frame)
        cv2.imwrite('right.jpg',right_frame)
        #cv2.imshow('Main',frame)
        #cv2.imshow('Left',left_frame)
        #cv2.imshow('Right',right_frame)
        test_image1 = image.load_img('left.jpg', target_size = (128, 128))
        test_image2 = image.load_img('right.jpg', target_size = (128, 128))
        test_image1 = image.img_to_array(test_image1)
        test_image2 = image.img_to_array(test_image2)
        test_image1 = np.expand_dims(test_image1, axis = 0)
        test_image2 = np.expand_dims(test_image2, axis = 0)
        resultL = classifier.predict(test_image1)
        resultR = classifier.predict(test_image2)
        #training_set.class_indices
        if resultL[0][0] == 0 and resultR[0][0] == 0:
            print('Center')
        elif resultL[0][0] == 0:
            print('Right')
        elif resultR[0][0] == 0:
            print('Left')
        else:
            print('No Ball')
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break
    else:
        break
cap.release()
