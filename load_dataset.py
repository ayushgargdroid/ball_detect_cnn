import multiprocessing as mp
import numpy as np
import os
import cv2

#Load dataset
dataset_name = 'dataset3'
if(os.name=='posix'):
    os.chdir(os.curdir+'/'+dataset_name)
else:
    os.chdir('C:\\Users\\Ayush\\')
# sub1 = os.listdir(os.curdir)
x_train = np.uint8([])
y_train = np.uint8([])
x_train1 = np.uint8([])
y_train1 = np.uint8([])
x_train2 = np.uint8([])
y_train2 = np.uint8([])
x_test = np.uint8([])
y_test = np.uint8([])

def getData(dataset_name,sub,positive):
    path = '/home/mrmai/Ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'
    x_train = np.uint8([])
    y_train = np.uint8([])
    x_test = np.uint8([])
    y_test = np.uint8([])
    count=0
    if(sub=='training_set'):
        sub2 = os.listdir(path)
        for inn in sub2:
            print(inn)
            if(positive==1 and inn=='noball'):
                continue
            elif(positive==0 and inn=='balls'):
                continue
            path = '/home/mrmai/Ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                # print('train '+i)
                if(i.find('jpg')==-1):
                    continue
                if(inn.find('no')==-1):
                    y_train = np.append(y_train,1)
                else:
                    y_train = np.append(y_train,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR)
                img = cv2.resize(img,(200,200))
                if(len(x_train.shape) is 1):
                    x_train = np.uint8([img])
                else:
                    x_train = np.append(x_train,[img],axis=0)
                count+=1
                # print(count)
        y_train.shape = (y_train.shape[0],1)
        if(positive==1):
            np.save('x1',x_train)
            np.save('y1',y_train)
        else:
            np.save('x2',x_train)
            np.save('y2',y_train)
    
    elif(sub=='test_set'):
        path = '/home/mrmai/Ayush/ball_detect_cnn/'+dataset_name+'/'+sub
        sub2 = os.listdir(path)
        for inn in sub2:
            path = '/home/mrmai/Ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                if(i.find('jpg')==-1):
                    continue
                if(inn.find('no')==-1):
                    y_test = np.append(y_test,1)
                else:
                    y_test = np.append(y_test,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR)
                img = cv2.resize(img,(200,200))
                if(len(x_test.shape) is 1):
                    x_test = np.uint8([img])
                else:
                    x_test = np.append(x_test,[img],axis=0)
        y_test.shape = (y_test.shape[0],1)
        np.save('x3',x_test)
        np.save('y3',y_test)


#Shuffle Data        
def getShuffled(x,y):
    shuffler = np.arange(x.shape[0])
    np.random.shuffle(shuffler)
    x = np.uint8([x[shuffler[i]] for i in range(x.shape[0])])
    y = np.uint8([y[shuffler[i]] for i in range(y.shape[0])])
    del shuffler
    return x,y


train1_process = mp.Process(target=getData,args=(dataset_name,'training_set',1))
train2_process = mp.Process(target=getData,args=(dataset_name,'training_set',0))
test_process = mp.Process(target=getData,args=(dataset_name,'test_set',2))

train1_process.start()
train2_process.start()
test_process.start()
train1_process.join()
train2_process.join()
test_process.join()

x_train1 = np.load('x1.npy')
y_train1 = np.load('y1.npy')
x_train2 = np.load('x2.npy')
y_train2 = np.load('y2.npy')
x_test = np.load('x3.npy')
y_test = np.load('y3.npy')

x_train = np.append(x_train1,x_train2,axis=0)
y_train = np.append(y_train1,y_train2,axis=0)
del x_train1,x_train2,y_train1,y_train2

x_train,y_train = getShuffled(x_train,y_train)
x_test,y_test = getShuffled(x_test,y_test)
x_train,y_train = getShuffled(x_train,y_train)
x_test,y_test = getShuffled(x_test,y_test)
x_train,y_train = getShuffled(x_train,y_train)
x_test,y_test = getShuffled(x_test,y_test)
x_train,y_train = getShuffled(x_train,y_train)
x_test,y_test = getShuffled(x_test,y_test)
np.save('x_test.npy',x_test)
np.save('x_train.npy',x_train)
np.save('y_test.npy',y_test)
np.save('y_train.npy',y_train)