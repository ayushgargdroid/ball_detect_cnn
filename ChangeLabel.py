import cv2
import numpy as np
import os
path='dataset_cust/training_set/balls'
images = os.listdir('/home/mrmai/Desktop/positives/train')
# images = [i.strip() for i in images]
pic_num = 346
for i in images:
    print pic_num
    img = cv2.imread("/home/mrmai/Desktop/positives/train/"+i, cv2.IMREAD_COLOR)
    # should be larger than samples / pos pic (so we can place our image on it)
    try:
        resized_image = cv2.resize(img, (640, 480))
        cv2.imwrite("/home/mrmai/Ayush/ball_detect_cnn/"+path+"/a." + str(pic_num) + ".jpg", resized_image)
        pic_num += 1

    except Exception as e:
    	print e
