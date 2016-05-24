#!/usr/bin/python
import tensorflow as tf

#=============================================================
# Session One : VECTOR ADDITION
# PART 1: Decalaration
var2 = tf.Variable(range(1,5))
var3 = tf.Variable([5, 6, 4, 1])

# PART 2: Instantiation
# Adding two Tensors
session = tf.Session()
# All the variables are assigned values now.
session.run(tf.initialize_all_variables())
print session.run(var2), '+', session.run(var3),
# Operation : add
print '=', session.run(tf.add(var2, var3))
session.close()

# 1 X 2 Matrix
a = [[3, 3]]
# 2 X 1 Matrix
b = [[2], [2]]
# PART 1: Decalaration
matrix1 = tf.constant(a)
matrix2 = tf.constant(b)
# PART 2: Instantiation
# Multiplying two Tensors
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    product = tf.matmul(matrix1, matrix2)
    print a, 'X', b, '=',
    print session.run(product)

#=============================================================
# Session Two : SCALAR X VECTOR
# Non Tensor part of the program
number = int(raw_input('Enter the number: '))

# PART 1: Decalaration
var1 = tf.constant(number)
# PART 2: Instantiation
session = tf.Session()
session.run(tf.initialize_all_variables())
print session.run(var1), 'X', session.run(var2),
product = tf.mul(var1, var2)
print '=', session.run(product)
session.close()

#=============================================================
# Session Three : TENSOR X PYTHON VARIABLE
# NOTE: i is not a tensor here, but still can be used
# Can always use number in place of var1 and it will work
session = tf.Session()
print 'TABLE of', session.run(var1)
for i in range(1, 11):
    print number, 'X', i, '=', session.run(tf.mul(var1, i))
session.close()
#=============================================================
