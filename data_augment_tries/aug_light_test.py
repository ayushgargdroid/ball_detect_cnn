from imgaug import augmenters as iaa
import os
import numpy as np
import cv2
os.chdir('/home/mrmai/Ayush/ball_detect_cnn/labelled_dataset1')
x = np.load('x_train.npy')

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])
images_aug = seq.augment_images(x)

for i in images_aug:
    cv2.imshow('Image',i)
    cv2.waitKey(0)
    
