import numpy as np
import cv2
import tensorflow as tf
import os

dataset_name = 'labelled_dataset1'
os.chdir('/home/mrmai/Ayush/ball_detect_cnn/'+dataset_name)

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

minibatch_size = 32
m = x_train.shape[0]
dimen = x_train.shape[1:]
batches = m/minibatch_size

y_train = np.uint8([y_train[i][0] for i in range(len(y_train))])
y_train.shape = (m,1)
y_train = np.insert(y_train,1,1,1)
for i in range(len(y_train)):
    if y_train[i][0] == 1:
        y_train[i][1] = 0
y_true_cls = tf.argmax(y_train, dimension=1)
y_test = np.uint8([y_test[i][0] for i in range(len(y_test))])
y_test.shape = (y_test.shape[0],1)
y_test = np.insert(y_test,1,0,1)
for i in range(len(y_test)):
    if y_test[i][0] == 0:
        y_test[i][1] = 1

print('X_train: ',x_train.shape)
print('Y_train: ',y_train.shape)
print('X_test: ',x_test.shape)
print('Y_test: ',y_test.shape)

def map(X,Y):
    for i in range(len(X)):
        # image = np.reshape(X[i],(200,200,3))
        cv2.imshow('Image',X[i])
        print Y[i]
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# map(x_test,y_test)

def pre_process_image(image, training):
    if training:
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,target_height=img_size_cropped,target_width=img_size_cropped)

    return image

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def create_bias(size):
    return tf.Variable(tf.constant(0.05,shape=[size]))

def conv(back,filter,input_channels,output_channels):
    W = create_weights([filter[0],filter[1],input_channels,output_channels])
    b = create_bias(output_channels)
    layer = tf.nn.conv2d(back,strides=[1,1,1,1],filter=W,padding='SAME')
    layer += b
    layer = tf.nn.max_pool(layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    layer = tf.nn.relu(layer)
    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(input,num_inputs,num_outputs,use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_bias(num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

X = tf.placeholder(tf.float32, shape=[None, dimen[0], dimen[1], dimen[2]], name='X')
Y = tf.placeholder(tf.float32, shape=[None,2], name='Y')

layer_conv1 = conv(X,[5,5],3,32)
layer_conv2 = conv(layer_conv1,[5,5],32,32)
layer_conv3 = conv(layer_conv2,[5,5],32,64)
layer_conv4 = conv(layer_conv3,[5,5],64,64)
          
layer_flat = create_flatten_layer(layer_conv4)

layer_fc1 = create_fc_layer(layer_flat,layer_flat.get_shape()[1:4].num_elements(),128,True) 
layer_fc2 = create_fc_layer(layer_fc1,128,2,False)
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    epochs = 5
    for epoch in range(epochs):
        epoch_cost = 0
        for i in range(batches):
            (miniX,miniY) = x_train[i*minibatch_size:(i+1)*minibatch_size],y_train[i*minibatch_size:(i+1)*minibatch_size]
            _ , minibatch_cost, y = sess.run([optimizer, cost,y_pred], feed_dict={X: miniX, Y: miniY})
            epoch_cost += minibatch_cost/batches
            del miniX,miniY
            print y
            # print('Y shape: '+str(y.shape))
        # acc = sess.run(accuracy,feed_dict={X:x_train,Y:y_train})
        # valid_acc = sess.run(accuracy,feed_dict={X:x_test,Y:y_test})
        print('Cost after epoch %i is %f' % (epoch+1,epoch_cost))
        # print('Accuracy: '+str(acc)+' Validate Accuracy: '+str(valid_acc))
    for i in range(batches):
        (miniX,miniY) = x_test[i*minibatch_size:(i+1)*minibatch_size],y_test[i*minibatch_size:(i+1)*minibatch_size]
        y = sess.run(y_pred, feed_dict={X: miniX, Y: miniY})
        for j in range(minibatch_size):
            print 'Should be'
            print(y_test[j])
            print 'Is'
            print(y[j])
        del miniX,miniY
        break


