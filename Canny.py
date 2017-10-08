import cv2
import numpy as np

images = open('/home/ayush/ball_detect_cnn/dataset/test_set/noball/TOC.txt','r')
images = [i.strip() for i in images]
pic_num = 1
for i in images:
    print pic_num
    img = cv2.imread("/home/ayush/ball_detect_cnn/dataset/test_set/noball/"+i, 0)
    # plate = np.zeros(img.shape,np.uint8)
    # frame = img.copy()
    # tt = frame.copy()
    # med = cv2.medianBlur(frame,5)
    # canny = cv2.Canny(med, 50, 255)
    # _,contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(plate, contours, -1, 255, 1)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)
    # canny = cv2.Canny(img,50,255)
    resized_image = cv2.resize(img, (200, 200))
    cv2.imwrite("/home/ayush/ball_detect_cnn/dataset3/test_set/noball/noball." + str(pic_num) + ".jpg", resized_image)
    pic_num += 1
