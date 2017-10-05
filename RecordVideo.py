import cv2
import numpy as np

cap = cv2.VideoCapture(0)

pic_num=196
while cap.isOpened():
    _, frame = cap.read()
    if(_):
        cv2.imshow('frame',frame)
        cv2.imwrite(str(pic_num)+'.jpg',frame)
        pic_num+=1
    else:
        break
    if(cv2.waitKey(1) & 0xFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()