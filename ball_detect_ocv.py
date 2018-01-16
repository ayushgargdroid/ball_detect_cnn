import numpy as np
import time
import cv2
timeout = 0
cv_cnt2 = 0
cv_cnt1 = 0
rot = 0
test = 0
prev_radius = 0
prev_cx = 0
prev_cy = 0
path = '/home/ayush/ball_detect_cnn/dataset2/test_set/balls/'
images = open(path+'TOC.txt','r')
images = [i.strip() for i in images]
k=0
cv2.namedWindow('Orignal Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Detected Balls', cv2.WINDOW_NORMAL)
for j in images:
    b = cv2.imread(path+j,1)
    gym = np.copy(b)
    cv2.imshow('Orignal Image', b)
    c = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    t1 = np.array([30, 100, 100])
    t2 = np.array([45 , 255, 255])
    m = cv2.inRange(c, t1, t2)
    f = cv2.bitwise_and(b, b, mask = m)
    md = cv2.medianBlur(f,15)
    gray = cv2.cvtColor(md, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)
    edges = cv2.Canny(mask, 100, 200)
    #mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
    dil = cv2.dilate(edges,kernel=(3,3),iterations=1)
    #dil = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel=(3,3))
    _,co,hierarchy = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in co:
        ((dx), (dy)), d = cv2.minEnclosingCircle(i)
        M = cv2.moments(i)
        if (M["m00"] != 0):
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            first = (i[0][0][0]-cx)*(i[0][0][0]-cx)
            second = (i[0][0][1] - cy) * (i[0][0][1] - cy)
            Radius = cv2.contourArea(i) / (np.pi)
            eqn = (first+second)-Radius
            eqn = np.int(eqn)
            area_m = cv2.contourArea(i)
            area_a = (np.pi)*Radius*Radius
            #cv2.circle(b, (dx, dy), d, (0, 0, 255), -1)
            if (eqn <= 30 and eqn >=(-30)) :
                prev_cx = cx=np.int(cx)
                prev_cy = cy=np.int(cy)
                prev_radius = Radius = np.int(np.sqrt(Radius))
                # cv2.drawContours(b, i, 0, (0, 0, 255), -1)
                cv2.circle(b,(cx,cy),Radius,(255,0,0),-1)
                # cv_cnt1 = cv_cnt1 + 1
                # if cv_cnt1 >= 3:
                #     #ser.write(s)
                #     rot = 1
                #     cv_cnt2 = cv_cnt2 + 1
                #     if cv_cnt2 == 50:
                #         timeout = 0
                #         cv_cnt2 = 0
                #         x = 27
                # else:
                #     rot = 0
            else :
                # cv2.circle(b, (int(cx), int(cy)), int(Radius), (255, 0, 0), -1)
                # print 'yolo'
                pass
        else :
            cx=prev_cx
            cy=prev_cy
            Radius=prev_radius
        # cv2.drawContours(b, co, 0, (0, 255, 0),1)
    #can = cv2.Canny(mask,150,150)
    cv2.imshow('Detected Balls',b)
    cv2.waitKey(0)
    #cv2.imshow('win2', md)
    #cv2.imshow('win3', f)
    cv2.destroyAllWindows()
print 'found'