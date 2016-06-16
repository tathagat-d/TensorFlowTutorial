#!/usr/bin/python

'''
Convolutional Neural Network design.
1. Input Layer,
2. Two Convolutional Layers,
3. One Dense Layer,
4. Output Layer

step size   : 1
patch size  : 2 * 2
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Data structure containing all the input
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# Input and Output
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Initializing internal data structure to tensorflow. [ weights and biases ]
#=============================================================================#
def initializeWeight(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def initializeBias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
#=============================================================================#

#Convolution and Pooling
#=============================================================================#
def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2X2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME')
#=============================================================================#

# Network Design
#=============================================================================#
# Transforming input data for first layer
x_image = tf.reshape(x, [-1, 28, 28, 1])

#Weights and Biases window for first Layer
# Dimension: 5 X 5. Input: 1 Output: 32
W_conv1 = initializeWeight([5, 5, 1, 32])
b_conv1 = initializeBias([32])

# Getting the output ready for next Layer
h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1) + b_conv1)
h_pool1 = maxPool2X2(h_conv1)

#Weights and Biases window for Second Layer
# Dimension: 5 X 5. Input: 32 Output: 64
W_conv2 = initializeWeight([5, 5, 32, 64])
b_conv2 = initializeBias([64])

# Getting the output ready for next Layer
h_conv2 = tf.nn.relu(conv2D(x_image, W_conv2) + b_conv2)
h_pool2 = maxPool2X2(h_conv2)

# Final Layer
W_final1 = initializeWeight([7 * 7 * 64, 1024])
b_final1 =initializeBias([1024])

# Reshaping input Layer for the final layer
h_pool = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_final1 = tf.nn.relu(tf.matmul(h_pool, W_final1) + b_final1)
#=============================================================================#

# Avoiding Overfitting
keep_prob = tf.placeholder(tf.float32)
h_final2 = tf.nn.dropout(h_final1, keep_prob)

#Applying softmax to generate probabilities
W_final2 = initializeWeight([1024, 10])
b_final2 = initializeBias([10])

y_conv = tf.nn.softmax(tf.matmul(h_final2, W_final2) + b_final2)
#=============================================================================#

# Training the model
# Cost Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#=============================================================================#

session = tf.Session()
session.run(tf.initialize_all_variables())

#Iterating 20,000 times with hope that we will converge
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if not i %100:
        train_accuracy = session.run(accuracy, feed_dict= {
            x:batch[0], y_ : batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    session.run(train_step, feed_dict={ x : batch[0], y_ : batch[1], keep_prob: 0.5})

print("test_accuracy %g" % session.run(accuracy, feed_dict= {
    x : mnist.test.images, y_ : mnist.test.labels, keep_prob: 1.0}))
#=============================================================================#
