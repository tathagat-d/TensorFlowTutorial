#!/usr/bin/python

'''
We basically are computing probabiltiy of image belonging to each 
class using neural network.
y = softmax(Wx + b)
There are no hidden layer in this model. Just one input and one
output layer.
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Download the data and put it into mnist data structure.
mnist = input_data.read_data_sets("MNIST/", one_hot = True)

# variable x represents input images.
# 'None' corresponds to unknown number of images with 784 columns.
x = tf.placeholder(tf.float32, [None, 784])

# Each pixel contributing[positively or negatively] to 10 classes.
# Initialized to zero.
W = tf.Variable(tf.zeros(shape = [784, 10]))

# Contribution independent of the input image to a class.
b = tf.Variable(tf.zeros([10]))

# Hypothesis
y = tf.nn.softmax(tf.matmul(x, W) + b)

'''
Training:
Cant used mean square error, because sigmoid/logistic function is non-linear.
using cross-entropy.
'''
# Correct prediction for each image. [Handling any number of images]
y_data = tf.placeholder(tf.float32, [None, 10])

# reduction_indice tells where to put the reduced sum. 
cost = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y), reduction_indices = [1]))

alpha = 0.5
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

#===================== End Of Graph Declaration ===============================#
'''
Execution:
Supplying data to placeholders and executing the DAG in tensorflow
'''
init = tf.initialize_all_variables()
session = tf.Session()
# All variables are now initialized.
session.run(init)
# Execute the training:
# Note: Number of iteration is through experimentation.
# Choosing stochastic gradient descent over batch gradient descent.
for i in range(1000):
    # Conquering the task, 100 examples at a time.[for 1000 times]
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train, feed_dict = {x: batch_xs, y_data: batch_ys})

#========================= Model Evaluation ===================================#
'''
Ranking:
Effeciency of our model 
'''
# Number of images correctly classified
# tf.argmax gives the index where the probability was maximum. Then we match it
# with the index of known result. Count of such matches are correct predictions.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_data, 1))
# Converting boolean to 1's and 0's
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

# Printing the accuracy on test data set
print 'Accuracy obtained on test data:',
print session.run(accuracy, feed_dict = {x : mnist.test.images, y_data : mnist.test.labels})
#============================ E O Program  ====================================#
