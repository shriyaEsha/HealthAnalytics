'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Training Data
""" Reading Testing data """
data = pd.read_csv('Xtrain.csv', sep=',', header=None)
dataset = data.values
header = dataset[0,1:dataset.shape[1]]
dataset = dataset[1:dataset.shape[0],:]
train_X = dataset[:,1:dataset.shape[1]]
train_Y = dataset[:,0]
 
train_X = np.ndarray(shape=(train_X.shape[0], train_X.shape[1]), dtype=np.float32)
train_Y = np.ndarray(shape=(train_Y.shape[0], 1), dtype=np.float32)

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.mul(X, W), b)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    # plt.plot(train_X, train_Y, 'ro', label='Original data')
    # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()

    # Testing example, as requested (Issue #2)
    test_X = np.ndarray(shape=(1, train_X.shape[1]), dtype=np.float32)
    test_Y = np.ndarray(shape=(1, 1), dtype=np.float32)

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    # plt.plot(test_X, test_Y, 'bo', label='Testing data')
    # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()
    print(pred,Y)
