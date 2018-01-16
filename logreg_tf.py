import cv2
import os
import numpy as np
import tensorflow as tf

dataset_name = 'labelled_dataset1'
os.chdir('/home/ayush/ball_detect_cnn')

def gettraindataset():
    os.chdir('/home/ayush/ball_detect_cnn')
    x_train = np.uint8([])
    names = np.load('names.npy')
    y_train = np.load('y.npy')
    print 'reading training ball images'
    for i in names:
        print(i)
        img = cv2.imread('./'+dataset_name+'/training_set/balls/'+i,cv2.IMREAD_COLOR).flatten()
        if(len(x_train.shape) is not 1):
            img.shape = (1,img.shape[0])
        x_train = np.append(x_train,img,axis=0)
        if(len(x_train.shape) is 1):
            x_train.shape = (1,x_train.shape[0])

    for i in os.listdir(os.curdir+'/'+dataset_name+'/training_set/noball/'):
        print(i)
        img = cv2.imread('./'+dataset_name+'/training_set/noball/'+i,cv2.IMREAD_COLOR).flatten()
        if(len(x_train.shape) is not 1):
            img.shape = (1,img.shape[0])
        x_train = np.append(x_train,img,axis=0)
        if(len(x_train.shape) is 1):
            x_train.shape = (1,x_train.shape[0])
        y_train = np.append(y_train,[[0,0,0,0,0]],axis=0)
    print x_train.shape
    print y_train.shape
    os.chdir(os.curdir+'/'+dataset_name)
    np.save('x_train',x_train)
    np.save('y_train',y_train)
    print y_train

def gettestdataset():
    os.chdir('/home/ayush/ball_detect_cnn')
    x_train = np.uint8([])
    names = np.load('names_test.npy')
    y_train = np.load('y_test.npy')
    print 'reading test ball images'
    for i in names:
        print(i)
        img = cv2.imread('./'+dataset_name+'/test_set/balls/'+i,cv2.IMREAD_COLOR).flatten()
        if(len(x_train.shape) is not 1):
            img.shape = (1,img.shape[0])
        x_train = np.append(x_train,img,axis=0)
        if(len(x_train.shape) is 1):
            x_train.shape = (1,x_train.shape[0])

    for i in os.listdir(os.curdir+'/'+dataset_name+'/test_set/noball/'):
        print(i)
        img = cv2.imread('./'+dataset_name+'/test_set/noball/'+i,cv2.IMREAD_COLOR).flatten()
        if(len(x_train.shape) is not 1):
            img.shape = (1,img.shape[0])
        x_train = np.append(x_train,img,axis=0)
        if(len(x_train.shape) is 1):
            x_train.shape = (1,x_train.shape[0])
        y_train = np.append(y_train,[[0,0,0,0,0]],axis=0)
    print x_train.shape
    print y_train.shape
    os.chdir(os.curdir+'/'+dataset_name)
    np.save('x_test',x_train)
    np.save('y_test',y_train)
    print y_train


# gettestdataset()
# gettraindataset()

os.chdir('/home/ayush/ball_detect_cnn/'+dataset_name)

def getShuffled(x,y):
    shuffler = np.arange(x.shape[0])
    np.random.shuffle(shuffler)
    x = np.uint8([x[shuffler[i]] for i in range(x.shape[0])])
    t = np.uint8([y[shuffler[0]]])
    for i in range(1,y.shape[0]):
        t = np.append(t,[y[shuffler[i]]],axis=0)
    del shuffler
    return x,t

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
# x_train, y_train = getShuffled(x_train,y_train)
# x_test, y_test = getShuffled(x_test,y_test)
# np.save('x_test',x_test)
# np.save('y_test',y_test)
# np.save('x_train',x_train)
# np.save('y_train',y_train)

def map(x_train,y_train):
    for i in range(len(x_train)):
        print(x_train[i])
        image = x_train[i].reshape([200,200,3])
        dimens = y_train[i]
        left_corner_x = (dimens[1]*200) - (dimens[3]/2)*200
        left_corner_y = (dimens[2]*200) - (dimens[4]/2)*200
        right_corner_x = (dimens[1]*200) + (dimens[3]/2)*200
        right_corner_y = (dimens[2]*200) + (dimens[4]/2)*200
        left_corner = (int(left_corner_x),int(left_corner_y))
        right_corner = (int(right_corner_x),int(right_corner_y))
        cv2.rectangle(image,left_corner,right_corner,(255,0,0),3)
        cv2.imshow('Image',image)
        print y_train[i]
        cv2.waitKey(0)
    cv2.destroyAllWindows()

map(x_train,y_train)

# learning_rate = 0.01
# epochs = 5
# minibatch_size = 32

# m = x_train.shape[0]
# dimen = x_train.shape[1:]
# batches = m/minibatch_size

# X = tf.placeholder(tf.float32,shape=[None,120000],name='X')
# Y = tf.placeholder(tf.float32,shape=[None,5],name='Y')

# W1 = tf.get_variable('W1',[120000,128],initializer=tf.contrib.layers.xavier_initializer())
# b1 = tf.get_variable('b1',[128],initializer=tf.zeros_initializer())

# W2 = tf.get_variable('W2',[128,128],initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.get_variable('b2',[128],initializer=tf.zeros_initializer())

# W3 = tf.get_variable('W3',[128,5],initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.get_variable('b3',[5],initializer=tf.zeros_initializer())

# # pred = tf.add(tf.matmul(X,W),b)
# # pred = tf.matmul(X,W)

# l1 = tf.add(tf.matmul(X,W1),b1)
# l1 = tf.nn.relu(l1)

# l2 = tf.add(tf.matmul(l1,W2),b2)
# l2 = tf.nn.relu(l2)

# output = tf.add(tf.matmul(l2,W3),b3)

# cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = output,labels = Y))
# l2_loss = 0.1 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) +tf.nn.l2_loss(W3))
# cost = tf.add(cost,l2_loss)

# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(epochs):
#         # sess.run(optimizer,feed_dict={X:x_train,Y:y_train})
#         # c = sess.run(cost,feed_dict={X:x_train,Y:y_train})
#         # print batches
#         for i in range(batches):
#             (miniX,miniY) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size] 
#             sess.run(optimizer,feed_dict={X:miniX,Y:miniY})
#         c = sess.run(cost,feed_dict={X:x_train,Y:y_train})
#         print c
          