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

# Load Data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Data structure containing all the input
mnist = input_data.read_data_sets("MNIST", one_hot=True)

# Network Design
#=============================================================================#
# Input and Output
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Transforming input data for first layer
x_image = tf.reshape(x, [-1, 28, 28, 1])

#Weights and Biases window for first Layer
# Dimension: 5 X 5. Input: 1 Output: 32
W_conv1 = initializeWeight([5, 5, 1, 32])
b_conv1 = initializeBias([32])

# Getting the output ready for next Layer
# Dimension: [None, 28, 28, 32]
h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1) + b_conv1)
# Dimension: [None, 14, 14, 32]
h_pool1 = maxPool2X2(h_conv1)

#Weights and Biases window for Second Layer
# Dimension: 5 X 5. Input: 32 Output: 64
W_conv2 = initializeWeight([5, 5, 32, 64])
b_conv2 = initializeBias([64])

# Getting the output ready for next Layer
# Dimension: [None, 14, 14, 64]
h_conv2 = tf.nn.relu(conv2D(h_pool1, W_conv2) + b_conv2)
# Dimension: [None, 7, 7, 64]
h_pool2 = maxPool2X2(h_conv2)

# Final Layer
# Dimension: [3136, 1024]
W_final1 = initializeWeight([7 * 7 * 64, 1024])
b_final1 =initializeBias([1024])

# Reshaping input Layer for the final layer
# Dimension: [None, 3136]
h_pool = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# Dimension: [None, 1024]
h_final1 = tf.nn.relu(tf.matmul(h_pool, W_final1) + b_final1)
#=============================================================================#

# Avoiding Overfitting
keep_prob = tf.placeholder(tf.float32)
# Dimension: [None, 1024]
h_final2 = tf.nn.dropout(h_final1, keep_prob)

#Applying softmax to generate probabilities
# Dimension: [1024, 10]
W_final2 = initializeWeight([1024, 10])
b_final2 = initializeBias([10])

y_conv = tf.nn.softmax(tf.matmul(h_final2, W_final2) + b_final2)
#=============================================================================#

# Training the model
# Cost Function
cross_entropy       = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
                        reduction_indices=[1]))
train_step          = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction  = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
score               = tf.cast(correct_prediction, tf.float32)
accuracy            = tf.reduce_mean(score)

#=============================================================================#

def train(session):
    #Iterating 20,000 times with hope that we will converge
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if not i %100:
            train_accuracy = session.run(accuracy, feed_dict= {
                x:batch[0], y_ : batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        session.run(train_step,
                feed_dict = { x : batch[0], y_ : batch[1], keep_prob: 0.5 })

def test(session):
    count = list()
    for i in range(1000):
        count.extend(session.run(score, feed_dict = {
            x : mnist.test.images[i * 100: i * 100 + 100],
            y_ : mnist.test.labels[i*100: i * 100 + 100],
            keep_prob: 1.0}
        ))
    count = tf.reduce_mean(count)
    print("test accuracy %g" % session.run(count))

session = tf.Session()
session.run(tf.initialize_all_variables())
train(session)
test(session)
#=============================================================================#
session.close()
