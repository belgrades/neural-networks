import numpy as np
import tensorflow as tf
from util import generation_train_test

# TODO: Modify generation_train_test to retrieve test_y with proper size
# TODO: Think about better way to define parameters.
# TODO: Separate hyperparameters from normal parameters.
# TODO: Create functions separating utilities.
# TODO: Add stop by convergence of gradient.
# TODO: Find out how to measure the number of times we call the gradient.
# TODO: Add comments in every function.
# TODO: Possibly add doc.

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def activation(t, sigma=0.01):
    return tf.exp(-tf.square(t)/sigma)


def main():
    tf.sub
    INPUT_UNITS = 2     # Number of input nodes: 4 features and 1 bias
    HIDDEN_UNITS = 30              # Number of hidden nodes
    OUTPUT_UNITS = 1   # Number of outcomes (3 iris flowers)
    LEARNING_RATE = 0.001
    SAMPLE_SIZE = 1000
    TRAIN_TEST_PROPORTION = 0.7
    RHO_CONSTANT = 0.0001
    SIGMA_CONSTANT = 4
    GENERATE_IMAGES = False
    EPOCHS = 20000

    # Creating dataset
    train_X, test_X, train_y, test_y = generation_train_test(size=SAMPLE_SIZE, p=TRAIN_TEST_PROPORTION)
    test_y = np.reshape(test_y, newshape=(int(SAMPLE_SIZE * (1 - TRAIN_TEST_PROPORTION)), OUTPUT_UNITS))
    train_y = np.reshape(train_y, newshape=(int(SAMPLE_SIZE * TRAIN_TEST_PROPORTION), OUTPUT_UNITS))

    # Writing placeholders
    X = tf.placeholder("float", shape=[None, INPUT_UNITS])
    y = tf.placeholder("float", shape=[None, OUTPUT_UNITS])

    # Defining omega
    omega = {
        'w': tf.ones(shape=[INPUT_UNITS, HIDDEN_UNITS]),
        'c': init_weights([INPUT_UNITS, HIDDEN_UNITS]),
        'v': init_weights([HIDDEN_UNITS, OUTPUT_UNITS])
    }

    # Defining pi
    pi = {
        'rho': tf.constant(RHO_CONSTANT)
    }

    # Definition functions
    input_layer = tf.matmul(X, omega['w'])
    print(input_layer.shape)
    hidden_layer = activation(tf.norm(tf.subtract(input_layer, omega['c'])))
    print(hidden_layer.shape)
    y_hat = tf.matmul(hidden_layer, omega['v'])
    cost = tf.reduce_sum(y_hat)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(EPOCHS):
        _, loss_training = sess.run([optimizer, cost], feed_dict={X: train_X, y: train_y})
        loss_test = sess.run([cost], feed_dict={X: test_X, y: test_y})

        print("[info] loss_training: {} loss_test: {}".format(loss_training, loss_test))

if __name__ == '__main__':
    main()