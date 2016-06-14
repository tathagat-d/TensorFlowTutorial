#!/usr/bin/python

import tensorflow as tf
import matplotlib.pyplot as plt

#--------------------GET DATA------------------------#
def load(fname):
    '''Read data one line at a time into a list '''
    fhand = open(fname, 'r')
    x_data= list()
    y_data= list()

    # For Each Training Example
    for line in fhand:
        line = line.strip().split(',')
        # Processing of features
        for i in range(len(line)):
            line[i] = float(line[i])
        # data formatting
        temp = list()
        temp.append(1)
        temp.extend(line[:-1])
        x_data.append(temp)
        y_data.append(line[-1])
    # End of all Training Examples

    return x_data, y_data
#--------------------GET DATA------------------------#

#--------------------FEATURE SCALE------------------------#
def featureScale(x_data):
    mean_x= [0, 0, 0]
    max_x = [None, None, None]
    min_x = [None, None, None]
    denom = [None, None, None]
    samples = 0

    for item in x_data:
        samples += 1
        for i in range(len(mean_x)):
            mean_x[i] += item[i]
            if max_x[i] == None or max_x[i] < item[i]:
                max_x[i] = item[i]
            if min_x[i] == None or min_x[i] > item[i]:
                min_x[i] = item[i]

    for i in range(len(mean_x)):
        mean_x[i] = float(mean_x[i]) / samples
        denom[i]    = max_x[i] - min_x[i]

    for index in range(len(x_data)):
        for i in range(1, len(x_data[index])):
            x_data[index][i] -= mean_x[i]
            x_data[index][i] /= denom[i]

#-----------------E-O-FEATURE SCALE------------------------#

# get data
x_data, y_data = load('data2.txt')
featureScale(x_data)

#------------------INITIALIZATION--------------------#
theta     = tf.Variable([[1.0, 2.0, 3.0]])
features  = tf.Variable(x_data)
y_data    = tf.Variable([list(y_data)])

# Hypothesis: We assume that our data represents a straight line
y = tf.matmul(theta, features, transpose_a = False, transpose_b = True)

# Cost function
loss = tf.reduce_mean(tf.square(y - y_data))

# Learning rate alpha is result of experimentation
# Learning rate is scary low
alpha = 0.000003
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(loss)

#-----------------E-O-INITIALIZATION------------------#
session = tf.Session()
session.run(tf.initialize_all_variables())

cost = list()
for x in range(51):
    session.run(train)
    cost.append(session.run(loss))

print session.run(theta)
#---------------E-O-EXECUTION-------------------------#
plt.title('Convergence Graph')
plt.xlabel('iteration')
plt.ylabel('cost')
iteration = range(51)
plt.plot(iteration, cost, 'r')
plt.show()
#---------------E-O-SESSION----------------------------#
# Using Normal Equation as the data size is small
theta = tf.matmul(
            tf.matrix_inverse(tf.matmul(features, features, transpose_a = True)),
            tf.matmul(features, y_data, transpose_a = True, transpose_b = True))
print session.run(theta)
session.close()
