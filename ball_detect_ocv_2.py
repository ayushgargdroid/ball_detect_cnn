import cv2
import numpy as np
import time
import math
import os

data = None

def draw_box(event,x,y,flags,param):
    global cx,cy,lx,ly,center_click,global_image,left_click,data, delete_img
    if event==cv2.EVENT_LBUTTONDOWN and not center_click and not left_click:
        cx,cy = x,y
        center_click = not center_click
        # print('X: '+str(x)+' Y: '+str(y))
    elif event==cv2.EVENT_LBUTTONDOWN and center_click and not left_click:
        lx,ly = x,y
        left_click = not left_click
        # print('X: '+str(x)+' Y: '+str(y))
    
    elif event==cv2.EVENT_LBUTTONDOWN and center_click and left_click:
        tx,ty = x,y
        # print('X: '+str(x)+' Y: '+str(y))
        left_click = not left_click
        center_click = not center_click
        left = int(math.sqrt((lx - cx)*(lx - cx) + (ly - cy)*(ly - cy)))
        top = int(math.sqrt((tx - cx)*(tx - cx) + (ty - cy)*(ty - cy)))
        left_corner_x = cx-left
        left_corner_y = cy-top
        right_corner_x = cx+left
        right_corner_y = cy+top
        if(left_corner_x<0):
            left_corner_x = 0
        if(left_corner_y<0):
            left_corner_y = 0
        if(right_corner_x<0):
            right_corner_x = 0
        if(right_corner_y<0):
            right_corner_y = 0
        if(left_corner_x>global_image.shape[1]):
            left_corner_x = global_image.shape[1]
        if(left_corner_y>global_image.shape[0]):
            left_corner_y = global_image.shape[0]
        if(right_corner_x>global_image.shape[1]):
            right_corner_x = global_image.shape[1]
        if(right_corner_y>global_image.shape[0]):
            right_corner_y = global_image.shape[0]
        left_corner = (left_corner_x,left_corner_y)
        right_corner = (right_corner_x,right_corner_y)
        # print(left_corner)
        # print(right_corner)
        # print(str(float(right_corner_x-left_corner_x)/global_image.shape[1]))
        cv2.rectangle(global_image,left_corner,right_corner,(255,0,0),3)
        cv2.imshow('Orignal Image',global_image)
        if data is None:
            data = [[1,cx/float(global_image.shape[1]),cy/float(global_image.shape[0]),float(right_corner_x-left_corner_x)/global_image.shape[1],float(right_corner_y - left_corner_y)/global_image.shape[0]]]
        else:
            t = [1,cx/float(global_image.shape[1]),cy/float(global_image.shape[0]),float(right_corner_x-left_corner_x)/global_image.shape[1],float(right_corner_y - left_corner_y)/global_image.shape[0]]
            data.append(t)
    elif event==cv2.EVENT_MBUTTONDBLCLK:
        print 'yolo'
        os.remove(path+global_image_name)
        delete_img = True
        print 'Image Delelted'

def getAllNames(start,end):
    if start is end:
        return []
    all_names = np.load('names'+str(start)+'.npy')
    for i in range(start+1,end):
        all_names = np.append(all_names,np.load('names'+str(i)+'.npy'))
    # print(str(len(all_names)))
    return all_names

def getAllDimens(start,end):
    if start is end:
        return []
    all_dimens = np.load('y'+str(start)+'.npy')
    if end-start is 1:
        return all_dimens
    for i in range(start+1,end):
        all_dimens = np.append(all_dimens,np.load('y'+str(i)+'.npy'))
    all_dimens = np.resize(all_dimens,(len(all_dimens)/5,5))
    return all_dimens

def map(all_names,all_dimens):
    print(str(len(all_names)))
    print(str(len(all_dimens)))
    # all_names = np.delete(all_names,26,0)
    for i in range(len(all_names)):
        name = all_names[i]
        dimens = all_dimens[i]
        image = cv2.imread(path+name,1)
        left_corner_x = (dimens[1]*200) - (dimens[3]/2)*200
        left_corner_y = (dimens[2]*200) - (dimens[4]/2)*200
        right_corner_x = (dimens[1]*200) + (dimens[3]/2)*200
        right_corner_y = (dimens[2]*200) + (dimens[4]/2)*200
        left_corner = (int(left_corner_x),int(left_corner_y))
        right_corner = (int(right_corner_x),int(right_corner_y))
        cv2.rectangle(image,left_corner,right_corner,(255,0,0),3)
        cv2.imshow('Image',image)
        print i
        cv2.waitKey(0)
    cv2.destroyAllWindows()

path = '/home/ayush/ball_detect_cnn/labelled_dataset1/training_set/balls/'
map(np.load('names.npy'),np.load('y.npy'))
# np.save('names_test',getAllNames(0,5))
# np.save('y_test',getAllDimens(0,5))
# cv2.namedWindow('Orignal Image', cv2.WINDOW_NORMAL)
# cv2.setMouseCallback('Orignal Image',draw_box)
# center_click = False
# left_click = False
# delete_img = False
# cx,cy,lx,ly = 0,0,0,0
# global_image_name = ''
# global_image = None
# images_in_order = []
# count = 0
# iter_var = 0
# all_names = getAllNames(0,iter_var)
# print(len(os.listdir(path)))
# for image_name in os.listdir(path):
#     if '.jpg' not in image_name:
#         continue
#     if image_name in all_names:
#         continue
#     print("Image no. "+str(count))
#     image = cv2.imread(path+image_name,1)
#     global_image = image
#     global_image_name = image_name
#     cv2.imshow('Orignal Image',image)
#     cv2.waitKey(0)
#     print(len(data))
#     if not delete_img:
#         images_in_order.append(image_name)
#         count+=1
#     else:
#         delete_img = False
#     center_click = False
#     left_click = False
#     delete_img = False
#     cx,cy,lx,ly = 0,0,0,0
#     if count == 50:
#         np.save('names'+str(iter_var),images_in_order)
#         np.save('y'+str(iter_var),data)
#         count = 0
#         data = None
#         images_in_order = []
#         iter_var += 1
#         print 'Saving...'
#         all_names = getAllNames(0,iter_var)
#     #     map(getAllNames(iter_var-1,iter_var),getAllDimens(iter_var-1,iter_var))
#     # print data

# np.save('names'+str(iter_var),images_in_order)
# np.save('y'+str(iter_var),data)
# count = 0
# data = None
# images_in_order = []
# iter_var += 1
# print 'Saving...'
# cv2.destroyAllWindows()


# from gps3 import gps3
# import socket
# import sys
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind(('localhost',5050))
# gps_socket = gps3.GPSDSocket()
# data_stream = gps3.DataStream()
# gps_socket.connect()
# gps_socket.watch()
# s.listen(1)
# conn, addr = s.accept()
# for new_data in gps_socket:
#     try:
#         if new_data:
#             data_stream.unpack(new_data)
#             print('Latitude = ', data_stream.TPV['lat'])
#             print('longitude= ',data_stream.TPV['lon'])
#             print(type(data_stream.TPV['lon']))
#             print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#             conn.send(data_stream.TPV['lat'])
#             conn.send(data_stream.TPV['lon'])
#     except KeyboardInterrupt:  
#         conn.close()
#         break

# conn.close()

        
