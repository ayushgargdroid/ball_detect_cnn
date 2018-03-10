import cv2
import numpy as np
path = '/home/mrmai/Ayush/ball_detect_cnn/dataset3/'
x_train = np.load(path+'x_train.npy')
y_train = np.load(path+'y_train.npy')
for i in range(x_train.shape[0]):
    cv2.imshow('asdas',x_train[i])
    print(y_train[i])
    cv2.waitKey(0)

