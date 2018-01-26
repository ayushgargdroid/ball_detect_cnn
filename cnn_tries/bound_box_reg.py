import tensorflow as tf
import numpy as np
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

resuts_file = open('results.txt','a')

# img = x_train[0]
# img = cv2.resize(img,(128,128))
# t = np.uint8([img])

# for i in range(1,len(x_train)):
#     img = x_train[i]
#     img = cv2.resize(img,(128,128))
#     t=np.append(t,[img],0)
# x_train = t

# img = x_test[0]
# img = cv2.resize(img,(128,128))
# t = np.uint8([img])

# for i in range(1,len(x_test)):
#     img = x_test[i]
#     img = cv2.resize(img,(128,128))
#     t=np.append(t,[img],0)
# x_test = t
    
minibatch_size = 32
m = x_train.shape[0]
dimen = x_train.shape[1:]
batches = m/minibatch_size

y_train_box = np.float32([y_train[i][1:] for i in range(len(y_train))])
y_train = np.uint8([y_train[i][0] for i in range(len(y_train))])
y_train_box.shape = (m,4)
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
y_test_box = np.float32([y_test[i][1:] for i in range(len(y_test))])
y_test = np.uint8([y_test[i][0] for i in range(len(y_test))])
y_test_box.shape = (y_test.shape[0],4)
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

def live_map(X,Y):
    dimens = Y
    image = X
    left_corner_x = (dimens[0]*200) - (dimens[2]/2)*200
    left_corner_y = (dimens[1]*200) - (dimens[3]/2)*200
    right_corner_x = (dimens[0]*200) + (dimens[2]/2)*200
    right_corner_y = (dimens[1]*200) + (dimens[3]/2)*200
    left_corner = (int(left_corner_x),int(left_corner_y))
    right_corner = (int(right_corner_x),int(right_corner_y))
    cv2.rectangle(image,left_corner,right_corner,(255,0,0),3)
    cv2.imshow('Image',image)

    
#map(x_train[:10],y_train_cls[:10])

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,shape,use_pooling=True):  
    weights = new_weights(shape=shape)
    biases = new_biases(length=shape[-1])
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],padding='SAME')
    layer += biases
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
    return layer,weights,biases

img_size = dimen[0]
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 2

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_box = tf.placeholder(tf.float32, shape=[None, 4], name='y_true_box')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x,shape=[5,5,3,32],use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,shape=[5,5,32,32],use_pooling=False)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,shape=[5,5,32,32],use_pooling=True)
layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,shape=[5,5,32,128],use_pooling=False)
layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4,shape=[5,5,128,128],use_pooling=True)
layer_conv6, weights_conv6 = new_conv_layer(input=layer_conv5,shape=[5,5,128,128],use_pooling=False)
layer_conv7, weights_conv7 = new_conv_layer(input=layer_conv6,shape=[5,5,128,128],use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv7)

with tf.device('/device:GPU:0'):
    layer_fc1, weights_fc_1, bias_fc_1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=128,use_relu=True)
    layer_fc2, weights_fc_2, bias_fc_2 = new_fc_layer(input=layer_fc1,num_inputs=128,num_outputs=128,use_relu=True)
    layer_fc3, weights_fc_3, bias_fc_3 = new_fc_layer(input=layer_fc2,num_inputs=128,num_outputs=2,use_relu=False)

    y_pred = tf.nn.softmax(layer_fc3)
    y_pred_cls = tf.argmax(y_pred, axis=1)

    cost_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,labels=y_true))
    optimizer_cls = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_cls)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.device('/device:GPU:1'):
    layer_reg_fc1, weights_reg_fc_1, bias_reg_fc_1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=128,use_relu=True)
    layer_reg_fc2, weights_reg_fc_2, bias_reg_fc_2 = new_fc_layer(input=layer_reg_fc1,num_inputs=128,num_outputs=128,use_relu=True)
    layer_reg, weights_reg_fc_3, bias_reg_fc_3 = new_fc_layer(input=layer_reg_fc2,num_inputs=128,num_outputs=4,use_relu=False)

    cost_reg = tf.reduce_mean(tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(layer_reg,y_true_box)),reduction_indices=1)))
    optimizer_reg = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost_reg,var_list=[weights_reg_fc_1,weights_reg_fc_2,weights_reg_fc_3,bias_reg_fc_1,bias_reg_fc_2,bias_reg_fc_3])

saver = tf.train.Saver()

# Counter for total number of iterations performed so far.

def optimize(num_iterations):
    start_time = time.time()
    acc_cls_tot = 0
    acc_cls_test = 0
    acc_box_tot = 0
    acc_box_test = 0
    with tf.Session() as session:
        # saver.restore(session, "/home/mrmai/Ayush/ball_detect_cnn/cnn_tries/bound_box_reg.ckpt")
        # print 'Restored'
        session.run(tf.global_variables_initializer())
        for epoch in range(num_iterations):
            for i in range(batches):
                (miniX,miniY,miniBox) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size],y_train_box[i*minibatch_size:(i+1)*minibatch_size]
                if acc_cls_test*100 >=90:
                    session.run([optimizer_reg], feed_dict={x: miniX, y_true: miniY, y_true_box:miniBox})
                else:
                    session.run([optimizer_cls,optimizer_reg], feed_dict={x: miniX, y_true: miniY, y_true_box:miniBox})
                del miniX,miniY

            acc_cls_tot = 0
            acc_cls_test = 0
            acc_box_tot = 0
            acc_box_test = 0

            for i in range(batches):
                (miniX,miniY,miniBox) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size],y_train_box[i*minibatch_size:(i+1)*minibatch_size]
                acc,acc1 = session.run([accuracy_cls,cost_reg], feed_dict={x: miniX, y_true: miniY, y_true_box:miniBox})
                del miniX,miniY
                acc_cls_tot += acc
                acc_box_tot += acc1

            for i in range(x_test.shape[0]/minibatch_size):
                (miniX,miniY,miniBox) = x_test[i*minibatch_size:(i+1)*minibatch_size],y_test[i*minibatch_size:(i+1)*minibatch_size],y_test_box[i*minibatch_size:(i+1)*minibatch_size]
                acc,acc1 = session.run([accuracy_cls,cost_reg], feed_dict={x: miniX, y_true: miniY, y_true_box:miniBox})
                del miniX,miniY
                acc_cls_test += acc
                acc_box_test += acc1

            acc_cls_tot /= batches
            acc_cls_test /= (x_test.shape[0]/minibatch_size)
            acc_box_tot /= batches
            acc_box_test /= (x_test.shape[0]/minibatch_size)
            msg = "Optimization Iteration: {0:>3}, Training Accuracy_cls: {1:>6.4%}, Test Accuracy_cls: {2:>6.4%}, Training Accuracy_box: {3:>6.4}, Test Accuracy_box: {4:>6.4}"
            print(msg.format(epoch+1, acc_cls_tot,acc_cls_test,acc_box_tot,acc_box_test))
            resuts_file.write(msg.format(epoch+1, acc_cls_tot,acc_cls_test,acc_box_tot,acc_box_test))

        resuts_file.close()
        save_path = saver.save(session, "/home/mrmai/Ayush/ball_detect_cnn/cnn_tries/bound_box_reg.ckpt")
        print("Model saved in file: %s" % save_path)
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        # cap = cv2.VideoCapture(0)
        # while True:
        #     _, frame = cap.read()
        #     if _:
        #         frame = cv2.resize(frame,(200,200))
        #         t = frame.copy()
        #         frame.shape = (1,200,200,3)
        #         pred,bb = session.run([y_pred_cls,layer_reg],feed_dict={x:frame})
        #         print t.shape
        #         print pred
        #         live_map(t,bb.T)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             cv2.destroyAllWindows()
        #             cap.release()
        #             session.close()
        #             break

optimize(num_iterations=10)