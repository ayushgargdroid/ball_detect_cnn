#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:36:26 2017

@author: ayush
"""

import threading
import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.python.framework import ops

#Load dataset
dataset_name = 'dataset5'
os.chdir('/home/ayush/ball_detect_cnn/'+dataset_name)
sub1 = os.listdir(os.curdir)
x_train = np.uint8([])
y_train = np.uint8([])
x_train1 = np.uint8([])
y_train1 = np.uint8([])
x_train2 = np.uint8([])
y_train2 = np.uint8([])
x_test = np.uint8([])
y_test = np.uint8([])

class DataThread(threading.Thread):
    def __init__(self,sub,positive):
        threading.Thread.__init__(self)
        self.sub = sub
        self.positive = positive
        
    def run(self):
        getData(self.sub,self.positive)
        
def getData(dataset_name,sub,positive):
    global x_train1, y_train1, x_test, y_test,x_train2
    path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub
    if(sub=='training_set'):
        sub2 = os.listdir(path)
        for inn in sub2:
            if(positive==1 and inn=='noball'):
                continue
            elif(positive==0 and inn=='ball'):
                continue
            path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                print('train '+i)
                if(i.find('jpg')==-1):
                    continue
                if(i.find('no')==-1 and positive==1):
                    y_train1 = np.append(y_train1,1)
                elif(i.find('no')==-1 and positive==0):
                    y_train2 = np.append(y_train2,1)
                elif(positive==1):
                    y_train1 = np.append(y_train1,0)
                elif(positive==0):
                    y_train2 = np.append(y_train2,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR).flatten()
                print(img.shape)
                if(len(x_train1.shape) is not 1):
                    img.shape = (1,691200)
                if(len(x_train2.shape) is not 1):
                    img.shape = (1,691200)
                print(img.shape)
                if(positive==1):
                    x_train1 = np.append(x_train1,img,axis=0)
                else:
                    x_train2 = np.append(x_train2,img,axis=0)
                if(len(x_train1.shape) is 1):
                    x_train1.shape = (1,x_train1.shape[0])
                if(len(x_train2.shape) is 1):
                    x_train2.shape = (1,x_train2.shape[0])
        if(positive==1):
            y_train1.shape = (y_train1.shape[0],1)
        else:
            y_train2.shape = (y_train2.shape[0],1)
    
    if(sub=='test_set'):
        path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub
        sub2 = os.listdir(path)
        for inn in sub2:
            path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                print('test '+i)
                if(i.find('jpg')==-1):
                    continue
                if(i.find('no')==-1):
                    y_test = np.append(y_test,1)
                else:
                    y_test = np.append(y_test,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR).flatten()
                if(len(x_test.shape) is not 1):
                    img.shape = (1,img.shape[0])
                #print(img.shape)
                #print(x_test.shape)
                x_test = np.append(x_test,img,axis=0)
                if(len(x_test.shape) is 1):
                    x_test.shape = (1,x_test.shape[0])
        y_test.shape = (y_test.shape[0],1)
        print('X_test: ',x_test.shape)
        print('Y_test: ',y_test.shape)    

train_thread1 = DataThread('training_set',1)
train_thread2 = DataThread('training_set',0)
test_thread = DataThread('test_set',0)

try:
    train_thread1.start()
    train_thread2.start()
    test_thread.start()
    test_thread.join()
    train_thread1.join()
    train_thread2.join()
except:
    print('Error')
    
x_train = x_train1
x_train = np.append(x_train1,x_train2,axis=0)
y_train = y_train1
y_train = np.append(y_train1,y_train2,axis=0)
print('X_train: ',x_train.shape)
print('Y_train: ',y_train.shape)
print('X_test: ',x_test.shape)
print('Y_test: ',y_test.shape)
            
            
            