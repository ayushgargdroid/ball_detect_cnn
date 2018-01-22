import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import os
import cv2

dataset_name = 'labelled_dataset1'
os.chdir('/home/mrmai/Ayush/ball_detect_cnn/'+dataset_name)

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

img = x_train[0]
img = cv2.resize(img,(60,60))
t = np.uint8([img])

for i in range(1,len(x_train)):
    img = x_train[i]
    img = cv2.resize(img,(60,60))
    t=np.append(t,[img],0)
x_train = t
    
minibatch_size = 64
m = x_train.shape[0]
dimen = x_train.shape[1:]
batches = m/minibatch_size

y_train = np.uint8([y_train[i][0] for i in range(len(y_train))])
y_train.shape = (m,1)
y_train = np.insert(y_train,1,0,1)
for i in range(len(y_train)):
    if y_train[i][0] == 1:
        y_train[i][1] = 1
        y_train[i][0] = 0
    else:
        y_train[i][0] = 1
        y_train[i][1] = 0
y_true_cls = tf.argmax(y_train, dimension=1)
y_test = np.uint8([y_test[i][0] for i in range(len(y_test))])
y_test.shape = (y_test.shape[0],1)
y_test = np.insert(y_test,1,0,1)
for i in range(len(y_test)):
    if y_test[i][0] == 1:
        y_test[i][1] = 1
        y_test[i][0] = 0
    else:
        y_test[i][0] = 1
        y_test[i][1] = 0
        
print('X_train: ',x_train.shape)
print('Y_train: ',y_train.shape)
print('X_test: ',x_test.shape)
print('Y_test: ',y_test.shape)

y_train_cls = np.argmax(y_train, axis=1)
y_test_cls = np.argmax(y_test, axis=1)

def map(X,Y):
    for i in range(len(X)):
        cv2.imshow('Image',X[i])
        print Y[i]
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#map(x_train[:10],y_train_cls[:10])

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,shape,use_pooling=True):  
    weights = new_weights(shape=shape)
    biases = new_biases(length=shape[-1])
    print weights.get_shape()
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],padding='SAME')
    layer += biases
    print layer.get_shape()
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,num_inputs,num_outputs,use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

img_size = dimen[0]
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 2

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x,shape=[5,5,3,32],use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,shape=[5,5,32,32],use_pooling=False)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,shape=[5,5,32,32],use_pooling=True)
layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,shape=[5,5,32,64],use_pooling=False)
layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4,shape=[5,5,64,64],use_pooling=True)
layer_conv6, weights_conv6 = new_conv_layer(input=layer_conv5,shape=[5,5,64,64],use_pooling=False)
layer_conv7, weights_conv7 = new_conv_layer(input=layer_conv6,shape=[5,5,64,64],use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv7)

layer_fc1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=128,use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=128,num_outputs=128,use_relu=True)
layer_fc3 = new_fc_layer(input=layer_fc1,num_inputs=128,num_outputs=128,use_relu=False)

y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, axis=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Counter for total number of iterations performed so far.

def optimize(num_iterations):
    start_time = time.time()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(num_iterations):
            acc_tot = 0
            for i in range(batches):
                (miniX,miniY) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size]
                print miniX.shape
                session.run(optimizer, feed_dict={x: miniX, y_true: miniY})
                del miniX,miniY
            for i in range(batches):
                (miniX,miniY) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size]
                print miniX.shape
                acc = session.run(accuracy, feed_dict={x: miniX, y_true: miniY})
                del miniX,miniY
                acc_tot += acc
            acc_tot /= batches
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.4%}"
            print(msg.format(i + 1, acc))

    end_time = time.time()

    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations=10)