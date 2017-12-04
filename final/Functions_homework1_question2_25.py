from numpy import exp
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from time import time

def franke2d_tensor(x):
    t1 = 0.75*exp(-(9*x[0]-2)**2/4-(9*x[1]-2)**2/4)
    t2 = 0.75*exp(-(9*x[0]+1)**2/49 -(9*x[1]+1)/10)
    t3 = 0.5*exp(-(9*x[0]-7)**2/4-(9*x[1]-3)**2/4)
    t4 = -0.2*exp(-(9*x[0]-4)**2-(9*x[1]-7)**2)
    return t1+t2+t3+t4


def g(t, sigma=4.0):
    return (1-tf.exp(-sigma*t))/(1+tf.exp(-sigma*t))


def generate_train_test(n=100, p=0.7, a=-1e-1, b=1e-1, seed=1743734):
    np.random.seed(seed)

    # Generate the Columns x1 x2 from uniform([0,1]x[0,1])
    X = np.random.uniform(size=[n, 2],
                          low=0,
                          high=1)

    # Generate 1 as franke2d(x1, x2)
    y = np.reshape(list(map(franke2d_tensor, X)), n)

    # Generate noise shape (100, 1) from uniform(a, b)
    noise = np.random.uniform(size=n,
                              low=a,
                              high=b)

    # Adding noise to y
    y = y + noise

    return train_test_split(X, y, train_size=p)


def init_weights(shape, trainable=True, seed=174373):
    """ Weight initialization """
    # weights = tf.random_normal(shape, mean=0.0847, stddev=0.4737, seed=seed
    weights = tf.truncated_normal(shape, seed=seed)
    return tf.Variable(weights, trainable=trainable)


def generate_extreme_mlp_variables(INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, RHO_CONSTANT, SIGMA_CONSTANT):
    X = tf.placeholder("float", shape=[None, INPUT_UNITS])
    y = tf.placeholder("float", shape=[None, OUTPUT_UNITS])

    # Weight initializations
    omega = {
        'w': init_weights([INPUT_UNITS, HIDDEN_UNITS], trainable=False),
        'v': init_weights([HIDDEN_UNITS, OUTPUT_UNITS]),
        'b': init_weights([HIDDEN_UNITS], trainable=False)
    }

    pi = {
        'rho': tf.constant(RHO_CONSTANT),
        'sigma': SIGMA_CONSTANT
    }

    con = tf.concat([tf.reshape(omega['w'], shape=[1, INPUT_UNITS * HIDDEN_UNITS]),
                     tf.reshape(omega['v'], shape=[1, HIDDEN_UNITS * OUTPUT_UNITS]),
                     tf.reshape(omega['b'], shape=[1, HIDDEN_UNITS])], 1)

    return X, y, omega, pi, con


def generate_mlp_y_hat_function(X, omega, pi):
    h = g((tf.matmul(X, omega['w']) - omega['b']), sigma=pi['sigma'])
    y_hat = tf.matmul(h, omega['v'])

    return y_hat


def generate_mlp_loss_functions(y_hat, y, pi, con, SAMPLE_SIZE=100, TRAIN_TEST_PROPORTION=0.7):
    regularization = pi['rho'] * tf.reduce_sum(tf.square(tf.norm(con)))
    # Forward propagation

    train_loss = tf.reduce_sum(tf.square(tf.norm(tf.subtract(y_hat, y)))) / (SAMPLE_SIZE * TRAIN_TEST_PROPORTION) * 2.0 + regularization
    test_loss = tf.reduce_sum(tf.square(tf.norm(tf.subtract(y_hat, y)))) / 2.0

    return train_loss, test_loss


def generate_mlp_optimizer(loss_function, LEARNING_RATE=0.01):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_function)


def init_tensorflow_session():
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess


def train_mlp(sess, optimizer, omega, X, y,  train_X, train_y, EPOCHS=10000, COMPUTE_TIME=True):
    tic = time()
    for epoch in range(EPOCHS):
        # Train with each example
        _ = sess.run([optimizer], feed_dict={X: train_X, y: train_y})
    toc = time()-tic

    return toc


def predict_mlp(sess, y_hat, X, test_X):
    return sess.run([y_hat], feed_dict={X: test_X})


def compute_mse_test(sess, test_loss, X, y, test_X, test_y):
    return sess.run([test_loss], feed_dict={X: test_X, y: test_y})


