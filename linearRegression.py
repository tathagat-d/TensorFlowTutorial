import tensorflow as tf
import matplotlib.pyplot as plt

#--------------------GET DATA------------------------#
def getData(fname):
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
x_data, y_data = getData('data1.txt')

# Spread of data in two dimension
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_data, y_data, 'go')
plt.show()

#------------------INITIALIZATION--------------------#
m = tf.Variable(0.0)
c = tf.Variable(0.0)

# Hypothesis
y = m * x_data + c

# Cost function
loss = tf.reduce_mean(tf.square(y - y_data))

alpha = 0.008
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(loss)
#-----------------E-O-INITIALIZATION------------------#

session = tf.Session()
session.run(tf.initialize_all_variables())
# Initial State
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_data, y_data, 'go')
plt.plot(x_data, session.run(y))
plt.text(15, -4, 'slope = %.2f\nconst = %.2f'%(session.run(m),session.run(c)))
plt.show()

for x in range(1000):
    session.run(train)
    if not x % 100:
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x_data, y_data, 'go')
        plt.plot(x_data, session.run(y))
        plt.text(15, -4, 'slope = %.2f\nconst = %.2f'
                %(session.run(m),session.run(c)))
        plt.show()
