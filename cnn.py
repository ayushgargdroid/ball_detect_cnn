#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:44:58 2017

@author: ayush
"""
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

#Part - 1 Initializing the classifier
classifier = Sequential()

#Part - 2  Adding Convolutional layer to the classifier
classifier.add(Conv2D(32,(3,3),input_shape=(128, 128, 3),activation='relu'))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

#Part - 3 Max Pooling layer to the classifier
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding for more acc
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Part - 4 Add Flattening layer to the classifier
classifier.add(Flatten())

#Part - 5 Full Connection - Classic ANN
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

#Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the CNN we built above to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset2/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset2/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=342,
        epochs=3,
        validation_data=test_set,
        validation_steps=150)

classifier.save('new_model2.h5') 
del classifier



classifier=load_model('new_model5.h5')
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
        cv2.imshow('Main',frame)
        cv2.imshow('Left',left_frame)
        cv2.imshow('Right',right_frame)
        test_image1 = image.load_img('left.jpg', target_size = (128, 128))
        test_image2 = image.load_img('right.jpg', target_size = (128, 128))
        test_image1 = image.img_to_array(test_image1)
        test_image2 = image.img_to_array(test_image2)
        test_image1 = np.expand_dims(test_image1, axis = 0)
        test_image2 = np.expand_dims(test_image2, axis = 0)
        resultL = classifier.predict(test_image1)
        resultR = classifier.predict(test_image2)
        #training_set.class_indices
        if resultL[0][0] == 1:
            predictionL = 'Not A Ball'
        else:
            predictionL = 'Ball'
        if resultR[0][0] == 1:
            predictionR = 'Not A Ball'
        else:
            predictionR = 'Ball'
        print(predictionL+' '+predictionR)
        if(cv2.waitKey(1) & 0xFF==ord('q')):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

