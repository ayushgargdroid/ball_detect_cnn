import cv2
import numpy as np

images = open('/home/ayush/ball_detect_cnn/test/TOC.txt','r')
images = [i.strip() for i in images]
pic_num = 163
for i in images:
    print pic_num
    img = cv2.imread("/home/ayush/ball_detect_cnn/test/"+i, cv2.IMREAD_COLOR)
    # should be larger than samples / pos pic (so we can place our image on it)
    resized_image = cv2.resize(img, (200, 200))
    cv2.imwrite("/home/ayush/ball_detect_cnn/dataset2/training_set/balls/ball." + str(pic_num) + ".jpg", resized_image)
    pic_num += 1