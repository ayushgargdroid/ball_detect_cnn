import cv2
import numpy as np

cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)

while True:
    _,frame = cap.read()
    cv2.imshow('Orignal',frame)
    plate = np.zeros(frame.shape,np.uint8)
    frameC = frame.copy()
    tt = frame.copy()
    med = cv2.medianBlur(frame,5)
    canny = cv2.Canny(med, 50, 255)
    cv2.imshow('Canny', canny)
    _,contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(plate, contours, -1, 255, 3)
    # if(len(contours)!=0):
    #     c = max(contours, key=cv2.contourArea)
    #     ((x, y), radius) = cv2.minEnclosingCircle(c)
    #     M = cv2.moments(c)
    #     peri = np.pi * 2 * radius
    #     actual_peri = cv2.arcLength(c,True)
    #     try:
    #         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #         if radius > 10 and peri >= actual_peri - 50 and peri <= actual_peri + 50:# and ((np.pi * radius * radius)-cv2.contourArea(c)<=4000):
    #             cv2.circle(frameC, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    #             cv2.circle(frameC, center, 5, (0, 0, 255), -1)
    #             print(str(np.pi * radius * radius)+' - '+str(cv2.contourArea(c)))
    #     except ZeroDivisionError:
    #         pass
    # for i in contours:
    #     if(cv2.contourArea(i)>1000):
    #         c = max(contours, key=cv2.contourArea)
    #         ((x,y),radius) = cv2.minEnclosingCircle(c)
    #         M = cv2.moments(c)
    #         peri = np.pi * 2 * radius
    #         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #         if radius > 10:
    #             cv2.circle(frameC, (int(x), int(y)), int(radius),
    #                        (0, 255, 255), 2)
    #             cv2.circle(frameC, center, 5, (0, 0, 255), -1)
    #         diam = np.sqrt(4*cv2.contourArea(i)/np.pi)
    #         diam = int(diam)
    #         peri = np.pi * diam
    #         print 'Diameter: '+str(diam)
    #         print 'Perimeter: '+str(peri)
    #         print 'Actual Perimeter: '+str(cv2.arcLength(i,True))
    #         if(cv2.arcLength(i,True)<=peri+50 and cv2.arcLength(i,True)>=peri-50):
    #             print cv2.contourArea(i)
    #             cv2.drawContours(frameC,[i],0,(0,255,0),3)
    # cv2.drawContours(frameC, contours, 1, (0, 255, 0), 3)
    # cv2.imshow('Mask',mask)
    cv2.imshow('Final',frameC)
    cv2.imshow('Contours',plate)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

