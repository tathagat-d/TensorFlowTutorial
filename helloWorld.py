#!/usr/bin/python

import tensorflow as tf

'''
    TensorFlow is like Object Oriented Programming.
    There are two parts. Declaration and Instantiation.
    Declaration is prototype. [ blueprint of a program ]
    Compare it with class declaration in OOPS.
    Instantiation is creating obj.
'''
# PART 1: Declaration
greeting = tf.constant('Hello World!')

# PART 2: Instantiation
session  = tf.Session()
# Session class has a run method that executes programs on CPU/GPU.
print session.run(greeting)
session.close()
