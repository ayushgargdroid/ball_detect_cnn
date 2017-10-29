import cv2
import numpy as np
path='dataset5/test_set/noball'
images = open('/home/ayush/ball_detect_cnn/'+path+'/TOC.txt','r')
images = [i.strip() for i in images]
pic_num = 1
for i in images:
    print pic_num
    img = cv2.imread("/home/ayush/ball_detect_cnn/"+path+"/"+i, cv2.IMREAD_COLOR)
    # should be larger than samples / pos pic (so we can place our image on it)
    try:
        resized_image = cv2.resize(img, (480, 480))
        cv2.imwrite("/home/ayush/ball_detect_cnn/"+path+"/noball." + str(pic_num) + ".jpg", resized_image)
        pic_num += 1

    except Exception as e:
    	print e
