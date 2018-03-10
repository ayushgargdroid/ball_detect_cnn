from imgaug import augmenters as iaa
import tensorflow as tf
from datetime import timedelta
import numpy as np
import time
import os
import numpy as np
import cv2
os.chdir('/home/mrmai/Ayush/ball_detect_cnn/labelled_dataset1')

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)

images_aug = seq.augment_images(x_train)
x_train = images_aug.copy()

minibatch_size = 32
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

def new_fc_layer(input,num_inputs,num_outputs,use_relu,keep_prob):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    layer = tf.nn.dropout(layer,keep_prob=keep_prob)
    return layer

img_size = dimen[0]
num_channels = 3

# Number of classes, one class for each of 10 digits.
num_classes = 2

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

layer_conv1, weights_conv1 = new_conv_layer(input=x,shape=[5,5,3,32],use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,shape=[5,5,32,32],use_pooling=False)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,shape=[5,5,32,32],use_pooling=True)
layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,shape=[5,5,32,128],use_pooling=False)
layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4,shape=[5,5,128,128],use_pooling=True)
layer_conv6, weights_conv6 = new_conv_layer(input=layer_conv5,shape=[5,5,128,128],use_pooling=False)
layer_conv7, weights_conv7 = new_conv_layer(input=layer_conv6,shape=[5,5,128,128],use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv7)

layer_fc1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=128,use_relu=True,keep_prob=keep_prob)
layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=128,num_outputs=128,use_relu=True,keep_prob=keep_prob)
layer_fc3 = new_fc_layer(input=layer_fc2,num_inputs=128,num_outputs=2,use_relu=False,keep_prob=keep_prob)

y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, axis=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# Counter for total number of iterations performed so far.

def optimize(num_iterations):
    start_time = time.time()
    with tf.Session() as session:
        saver.restore(session, "/home/mrmai/Ayush/ball_detect_cnn/data_augment_tries/aug_mid_test.ckpt")
        print 'Restored'
        # session.run(tf.global_variables_initializer())
        # for epoch in range(num_iterations):
        #     acc_tot = 0
        #     acc_test = 0
        #     for i in range(batches):
        #         (miniX,miniY) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size]
        #         session.run(optimizer, feed_dict={x: miniX, y_true: miniY,keep_prob:0.5})
        #         del miniX,miniY

        #     for i in range(batches):
        #         (miniX,miniY) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size]
        #         acc = session.run(accuracy, feed_dict={x: miniX, y_true: miniY,keep_prob:1.0})
        #         del miniX,miniY
        #         acc_tot += acc

        #     for i in range(x_test.shape[0]/minibatch_size):
        #         (miniX,miniY) = x_test[i*minibatch_size:(i+1)*minibatch_size],y_test[i*minibatch_size:(i+1)*minibatch_size]
        #         acc = session.run(accuracy, feed_dict={x: miniX, y_true: miniY,keep_prob:1.0})
        #         del miniX,miniY
        #         acc_test += acc
        #     acc_tot /= batches
        #     acc_test /= (x_test.shape[0]/minibatch_size)
        #     msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.4%}, Test Accuracy: {2:>6.4%}"
        #     print(msg.format(epoch+1, acc_tot,acc_test))

        # save_path = saver.save(session, "/home/mrmai/Ayush/ball_detect_cnn/data_augment_tries/aug_mid_test.ckpt")
        # print("Model saved in file: %s" % save_path)
        # end_time = time.time()
        # time_dif = end_time - start_time
        # print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        cap = cv2.VideoCapture(0)
        load = np.zeros([77,200,200,3])
        check = np.zeros([32])
        count = 0
        while True:
            _, frame = cap.read()
            if _:
                frame = cv2.resize(frame,(200,200))
                t = frame.copy()
                frame.shape = (1,200,200,3)
                pred = session.run(y_pred_cls,feed_dict={x:frame,keep_prob:1.0})
                cv2.imshow('Image',t)
                cv2.waitKey(1)
                print pred
                # cv2.imshow('Image',frame)
                # cv2.waitKey(1)
                # rows,cols = frame.shape[0],frame.shape[1]
                # i,j=0,0
                # span = 40
                # print 'Got new'
                # while((i+200)<rows):
                #     if(j+200>=cols):
                #         j = 0
                #         i += span
                #     tframe = frame[i:i+200,j:j+200,:]
                #     j+=span
                #     # print(tframe.shape)
                #     load[count] = tframe
                #     count+=1
                #     if(count>=77):
                #         count = 0
                #         pred = session.run(y_pred_cls,feed_dict={x:load,keep_prob:1.0})
                #         print(pred)
                #         # if(np.array_equal(pred,check)==True):
                #         #     print('Not Found')
                #         # else:
                #         #     print('Found')

optimize(num_iterations=50)
    
