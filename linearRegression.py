#!/usr/bin/python

import tensorflow as tf
import matplotlib.pyplot as plt

'''
Help:
    a. Assumption: that data is best represented as a straight line.
    b. We compute the outcome(y) and compare it with known outcome(y_data)
       to identify errors.
    c. Convex function: least squared error is used to represent error.
    d. Errors are minimized using greedy algorithm: GradientDescentOptimizer
'''

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
        x_data.append(line[0])
        y_data.append(line[1])
    # End of all Training Examples

    return x_data, y_data
#--------------------GET DATA------------------------#

# get data
x_data, y_data = load('data1.txt')
# Spread of data in two dimension
plt.title('Data Spread')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_data, y_data, 'go')
plt.show()

#------------------INITIALIZATION--------------------#
m = tf.Variable(0.0)
c = tf.Variable(0.0)

# Hypothesis: We assume that our data represents a
# straight line
y = m * x_data + c

# Cost function
loss = tf.reduce_mean(tf.square(y - y_data))

# Learning rate alpha is result of experimentation
alpha = 0.008
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(loss)
#-----------------E-O-INITIALIZATION------------------#


#-------------------EXECUTION-------------------------#
session = tf.Session()
session.run(tf.initialize_all_variables())

cost = list()
for x in range(1500):
    session.run(train)
    cost.append(session.run(loss))

#---------------E-O-EXECUTION-------------------------#
plt.title('Best Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_data, y_data, 'go')
plt.plot(x_data, session.run(y))
plt.text(15, -4, 'slope = %.2f\nconst = %.2f'
        %(session.run(m),session.run(c)))
plt.show()

plt.title('Convergence Graph')
plt.xlabel('iteration')
plt.ylabel('cost')
iteration = range(1500)
plt.plot(iteration, cost, 'r')
plt.show()

#---------------E-O-SESSION----------------------------#
session.close()
