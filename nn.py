""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

# Import MNIST data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 100
display_step = 100

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
num_input = 2 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def train(X_train, Y_train, x_test, y_test):

    # Construct model
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        for iter in range(200):
            for step in range(0, len(X_train), batch_size):
                batch_x, batch_y = X_train[step:step+batch_size], Y_train[step:step+batch_size]
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Iter "+str(iter)+", Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        preds = sess.run(prediction, feed_dict={X: x_test,
                                      Y: y_test})

        # print(preds)

        grids = [preds[:,0]]
        lev   = np.linspace(0,1,11)  
        titles = ['Neural Network for Classification']
        for title, grid in zip(titles, grids):
            plt.figure(figsize=(8,6))
            plt.contourf(X1,X2,np.reshape(grid,(n_grid,n_grid)),
                         levels = lev,cmap=cm.coolwarm)
            p = 0.5
            # print(y_test)
            X_train = np.asarray(X_train)
            Y_train = np.asarray(Y_train)
            Y_train = Y_train[:,0]
            plt.plot(X_train[Y_train>=p,0],X_train[Y_train>=p,1],"ro", markersize = 3)
            plt.plot(X_train[Y_train<p,0],X_train[Y_train<p,1],"bo", markersize = 3)
            plt.colorbar()
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("y")
            # plt.savefig("./plot.png")
            plt.show()

def f(x):
    # if np.sum(x) == 0:
    #     return -1
    if x[1] - 5*np.sin(x[0]) - 0.9 >= 0:
        return [1,0]
    else:
        return [0,1]

def generate_classification_toy_data(n_samples=500):

    np.random.seed(0)
    d = 2
    # n_samples = 500
    lims = [10,10]
    x = np.zeros((n_samples, d))
    y = np.zeros((n_samples, 1))

    y = []
    for i in range(len(x)):
        x[i][0] = np.random.uniform(-lims[0],lims[0])
        x[i][1] = np.random.uniform(-lims[1],lims[1])
        y.append(f(x[i]))
    x = np.asarray(x)
    x = x.astype(np.float32)
    y = np.asarray(y)
    y = y.astype(np.float32)
    return x, y


def plot_binary_data(X_train, y_train):
    plt.plot(X_train[0, np.argwhere(y_train == 1)], X_train[1, np.argwhere(y_train == 1)], 'bo')
    plt.plot(X_train[0, np.argwhere(y_train == -1)], X_train[1, np.argwhere(y_train == -1)], 'ro')



X_train, y_train = generate_classification_toy_data(500)
x = X_train
n_grid = 100
max_x      = np.max(x,axis = 0)
min_x      = np.min(x,axis = 0)
X1         = np.linspace(min_x[0]-5,max_x[0]+5,n_grid)
X2         = np.linspace(min_x[1]-5,max_x[1]+5,n_grid)
x1,x2      = np.meshgrid(X1,X2)
x_test      = np.zeros([n_grid**2,2])
x_test[:,0] = np.reshape(x1,(n_grid**2,))
x_test[:,1] = np.reshape(x2,(n_grid**2,))
y_test = []
for i in range(len(x_test)):
    y_test.append(f(x_test[i]))
print(x_test, y_test)
train(X_train, y_train, x_test, y_test)