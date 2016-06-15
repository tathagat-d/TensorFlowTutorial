import tensorflow as tf

'''
Note: We basically are computing probabiltiy of image belonging to
each class using neural network.
y = softmax(Wx + b)
'''

# None corresponds to unknown number of images with 784 columns.
# Each pixel is represented with type float.
x = tf.placeholder(tf.float32, [None, 784])

# Hypothesis/Model
# Each pixel contributing[positively or negatively] to 10 classes.
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
init = tf.initialize_all_variables()
session = tf.Session()
# All variables are now initialized.
session = tf.run(init)
# Execute the training:
# Note: Number of iteration is through experimentation.
# Choosing stochastic gradient descent over batch gradient descent.
for i in range(1000):
    # Conquering the task, 100 examples at a time.[for 1000 times]
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train, feed_dict = {x: batch_xs, y_data: batch_ys})
