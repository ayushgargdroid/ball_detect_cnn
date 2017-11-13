import multiprocessing as mp
import numpy as np
import os
import tensorflow as tf
import cv2

#Load dataset
dataset_name = 'dataset'
if(os.name=='posix'):
    os.chdir('/home/ayush/ball_detect_cnn/'+dataset_name)
else:
    os.chdir('C:\\Users\\Ayush\\')
sub1 = os.listdir(os.curdir)
x_train = np.uint8([])
y_train = np.uint8([])
x_train1 = np.uint8([])
y_train1 = np.uint8([])
x_train2 = np.uint8([])
y_train2 = np.uint8([])
x_test = np.uint8([])
y_test = np.uint8([])
        
def getDataFlatten(dataset_name,sub,positive):
    path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub
    x_train = np.uint8([])
    y_train = np.uint8([])
    x_test = np.uint8([])
    y_test = np.uint8([])
    count=0
    if(sub=='training_set'):
        sub2 = os.listdir(path)
        for inn in sub2:
            if(positive==1 and inn=='noball'):
                continue
            elif(positive==0 and inn=='ball'):
                continue
            path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                print('train '+i)
                if(i.find('jpg')==-1):
                    continue
                if(i.find('no')==-1):
                    y_train = np.append(y_train,1)
                else:
                    y_train = np.append(y_train,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR).flatten()
                if(len(x_train.shape) is not 1):
                    img.shape = (1,img.shape[0])
                print(img.shape)
                x_train = np.append(x_train,img,axis=0)
                if(len(x_train.shape) is 1):
                    x_train.shape = (1,x_train.shape[0])
                count+=1
                print(count)
        y_train.shape = (y_train.shape[0],1)
        print('sdas')
        if(positive==1):
            np.save('x1',x_train)
            np.save('y1',y_train)
        else:
            np.save('x2',x_train)
            np.save('y2',y_train)
    
    elif(sub=='test_set'):
        path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub
        sub2 = os.listdir(path)
        for inn in sub2:
            path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                print('test '+i)
                if(i.find('jpg')==-1):
                    continue
                if(i.find('no')==-1):
                    y_test = np.append(y_test,1)
                else:
                    y_test = np.append(y_test,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR).flatten()
                if(len(x_test.shape) is not 1):
                    img.shape = (1,img.shape[0])
                #print(img.shape)
                #print(x_test.shape)
                x_test = np.append(x_test,img,axis=0)
                if(len(x_test.shape) is 1):
                    x_test.shape = (1,x_test.shape[0])
        y_test.shape = (y_test.shape[0],1)
        np.save('x3',x_test)
        np.save('y3',y_test)

def getData(dataset_name,sub,positive):
    path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub
    x_train = np.uint8([])
    y_train = np.uint8([])
    x_test = np.uint8([])
    y_test = np.uint8([])
    count=0
    if(sub=='training_set'):
        sub2 = os.listdir(path)
        for inn in sub2:
            if(positive==1 and inn=='noball'):
                continue
            elif(positive==0 and inn=='ball'):
                continue
            path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                print('train '+i)
                if(i.find('jpg')==-1):
                    continue
                if(i.find('no')==-1):
                    y_train = np.append(y_train,1)
                else:
                    y_train = np.append(y_train,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR)
                if(len(x_train.shape) is 1):
                    x_train = np.uint8([img])
                else:
                    x_train = np.append(x_train,[img],axis=0)
                count+=1
                print(count)
                print(x_train.shape)
        y_train.shape = (y_train.shape[0],1)
        if(positive==1):
            np.save('x1',x_train)
            np.save('y1',y_train)
        else:
            np.save('x2',x_train)
            np.save('y2',y_train)
    
    elif(sub=='test_set'):
        path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub
        sub2 = os.listdir(path)
        for inn in sub2:
            path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                print('test '+i)
                if(i.find('jpg')==-1):
                    continue
                if(i.find('no')==-1):
                    y_test = np.append(y_test,1)
                else:
                    y_test = np.append(y_test,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR)
                if(len(x_test.shape) is 1):
                    x_test = np.uint8([img])
                else:
                    x_test = np.append(x_test,[img],axis=0)
        y_test.shape = (y_test.shape[0],1)
        np.save('x3',x_test)
        np.save('y3',y_test)
        
def getDataWindows(dataset_name,sub,positive):
    path = 'C:\\Users\\Ayush\\'+dataset_name+'\\'+sub
    x_train = np.uint8([])
    y_train = np.uint8([])
    x_test = np.uint8([])
    y_test = np.uint8([])
    count=0
    if(sub=='training_set'):
        sub2 = os.listdir(path)
        for inn in sub2:
            if(positive==1 and inn=='noball'):
                continue
            elif(positive==0 and inn=='ball'):
                continue
            path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                print('train '+i)
                if(i.find('jpg')==-1):
                    continue
                if(i.find('no')==-1):
                    y_train = np.append(y_train,1)
                else:
                    y_train = np.append(y_train,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR)
                if(len(x_train.shape) is 1):
                    x_train = np.uint8([img])
                else:
                    x_train = np.append(x_train,[img],axis=0)
                count+=1
                print(count)
                print(x_train.shape)
        y_train.shape = (y_train.shape[0],1)
        if(positive==1):
            np.save('x1',x_train)
            np.save('y1',y_train)
        else:
            np.save('x2',x_train)
            np.save('y2',y_train)
    
    elif(sub=='test_set'):
        path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub
        sub2 = os.listdir(path)
        for inn in sub2:
            path = '/home/ayush/ball_detect_cnn/'+dataset_name+'/'+sub+'/'+inn
            imgs = os.listdir(path)
            for i in imgs:
                print('test '+i)
                if(i.find('jpg')==-1):
                    continue
                if(i.find('no')==-1):
                    y_test = np.append(y_test,1)
                else:
                    y_test = np.append(y_test,0)
                img = cv2.imread(path+'/'+i,cv2.IMREAD_COLOR)
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

def getWeight(shape):
    initial = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    return initial

def getBias(shape):
    initial = tf.Variable(tf.constant(0.1,shape=shape))
    return initial

def getWeights(a,b,i):
    W = tf.get_variable('W'+str(i),[b,a],initializer=tf.contrib.layers.xavier_initializer())
    c = tf.get_variable('b'+str(i),[b,1],initializer=tf.zeros_initializer())
    return W,c

def trainNN(X,Y):
    print('Training started...')
    W1,b1 = getWeight([5,5,3,32]),getBias([32])
    print('Received Weights 1...')
    conv1 = tf.nn.relu(tf.nn.conv2d(X,W1,strides=[1, 1, 1, 1], padding='SAME')+b1)
    print(W1)
    print('Computed Conv 1...')
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    print('Computed Pool 1...')
    W2,b2 = getWeight([5,5,32,64]),getBias([64])
    print('Received Weights 2...')
    conv2 = tf.nn.relu(tf.nn.conv2d(pool1,W2,strides=[1, 1, 1, 1], padding='SAME')+b2)
    print('Computed Conv 2...')
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    print('Computed Pool 2...')
    layer_shape = pool2.get_shape()
    num_feat = layer_shape[1:4].num_elements()
    X_t = tf.reshape(pool2,[-1,num_feat])
    print('Flattened...')
    W3,b3 = getWeight([num_feat,128]),getBias([128])
    X_tt = tf.nn.relu(tf.add(tf.matmul(X_t,W3),b3))
    print('Computed Dense 1...')
    W4,b4 = getWeight([128,1]),getBias([1])
    X_ft = tf.matmul(X_tt,W4)+b4
    X_f = tf.nn.sigmoid(X_ft)
    return X_f

if(os.name=='posix'):
    train1_process = mp.Process(target=getData,args=(dataset_name,'training_set',1))
    train2_process = mp.Process(target=getData,args=(dataset_name,'training_set',0))
    test_process = mp.Process(target=getData,args=(dataset_name,'test_set',2))
else:
    train1_process = mp.Process(target=getDataWindows,args=(dataset_name,'training_set',1))
    train2_process = mp.Process(target=getDataWindows,args=(dataset_name,'training_set',0))
    test_process = mp.Process(target=getDataWindows,args=(dataset_name,'test_set',2))

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

np.save('x1',x_train1)
np.save('y1',y_train1)
np.save('x2',x_train2)
np.save('y2',y_train2)
np.save('x3',x_test)
np.save('y3',y_test)

x_train = np.append(x_train1,x_train2,axis=0)
y_train = np.append(y_train1,y_train2,axis=0)
del x_train1,x_train2,y_train1,y_train2

x_train,y_train = getShuffled(x_train,y_train)
x_test,y_test = getShuffled(x_test,y_test)

print('X_train: ',x_train.shape)
print('Y_train: ',y_train.shape)
print('X_test: ',x_test.shape)
print('Y_test: ',y_test.shape)

minibatch_size = 32
m = x_train.shape[0]
dimen = x_train.shape[1:]
batches = m/minibatch_size
batches = 5

with tf.Session() as sess:

    X = tf.placeholder(tf.float32,shape=[minibatch_size,dimen[0],dimen[1],dimen[2]],name='X')
    Y = tf.placeholder(tf.float32,shape=[minibatch_size,1],name='Y')
    
    Z_f = trainNN(X,Y)
        
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(Z_f), labels = tf.transpose(Y)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(cost)
    init = tf.global_variables_initializer()
    sess.run(init)
    epochs = 2
    for epoch in range(epochs):
        epoch_cost = 0
        for i in range(batches):
            print(i)
            (miniX,miniY) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size]
            print(miniX.shape)
            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: miniX, Y: miniY})
            print(minibatch_cost)
            epoch_cost += minibatch_cost/batches
            del miniX,miniY
        print('Cost after epoch %i is %f' % (epoch+1,epoch_cost))


    
    
    
            