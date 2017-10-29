import cv2
import numpy as np

images = open('/home/ayush/ball_detect_cnn/dataset/test_set/balls/TOC.txt','r')
images = [i.strip() for i in images]
pic_num = 1
for i in images:
    print pic_num
    img = cv2.imread("/home/ayush/ball_detect_cnn/dataset/test_set/balls/"+i, 0)
    plate = np.zeros(img.shape,np.uint8)
    frame = img.copy()
    tt = frame.copy()
    med = cv2.medianBlur(frame,5)
    canny = cv2.Canny(med, 50, 255)
    _,contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(plate, contours, -1, 255, 1)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)
    # canny = cv2.Canny(img,50,255)
    resized_image = cv2.resize(plate, (200, 200))
    cv2.imwrite("/home/ayush/ball_detect_cnn/dataset3/test_set/balls/ball." + str(pic_num) + ".jpg", resized_image)
    pic_num += 1

# import cv2
# import numpy as np

# images = open('/home/ayush/ball_detect_cnn/dataset/test_set/balls/TOC.txt','r')
# images = [i.strip() for i in images]
# pic_num = 1
# kernel = np.ones((5,5),np.uint8)

# for i in images:
#     img = cv2.imread("/home/ayush/ball_detect_cnn/dataset/test_set/balls/"+i, cv2.IMREAD_COLOR)

#     frameC = img.copy()
#     med = cv2.medianBlur(img, 5)
#     lower_green = np.array([40, 80, 80])
#     upper_green = np.array([80, 255, 255])
#     mask = cv2.inRange(med, lower_green, upper_green)
#     mask = cv2.erode(mask, kernel, iterations=2)
#     mask = cv2.dilate(mask, kernel, iterations=2)
#     canny = cv2.Canny(mask, 50, 255)
#     _,contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if(len(contours)!=0):
#         c = max(contours, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(c)
#         M = cv2.moments(c)
#         peri = np.pi * 2 * radius
#         actual_peri = cv2.arcLength(c, True)
#         try:
#             center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#             if radius > 10 and peri >= actual_peri - 50 and peri <= actual_peri + 50:
#                 cv2.circle(frameC, (int(x), int(y)), int(radius), (0, 255, 255), 2)
#                 cv2.circle(frameC, center, 5, (0, 0, 255), -1)
#         except ZeroDivisionError:
#             pass
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
    # resized_image = cv2.resize(frameC, (200, 200))
    # cv2.imwrite("/home/ayush/ball_detect_cnn/dataset3/test_set/balls/ball." + str(pic_num) + ".jpg", resized_image)
    # pic_num+=1