import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from util import generate_data_bias, plot_pred_real, plot_3d_data

def franke2d_tensor(x):
    from numpy import exp
    t1 = 0.75*exp(-(9*x[0]-2)**2/4-(9*x[1]-2)**2/4)
    t2 = 0.75*exp(-(9*x[0]+1)**2/49 -(9*x[1]+1)/10)
    t3 = 0.5*exp(-(9*x[0]-7)**2/4-(9*x[1]-3)**2/4)
    t4 = -0.2*exp(-(9*x[0]-4)**2-(9*x[1]-7)**2)
    return t1+t2+t3+t4


def generation_train_test(size, p, a=-1e-1, b=1e-1, seed=1743734):
    np.random.seed(seed)
    x = np.random.random([size,2])
    noise = np.random.uniform(a, b, size=size)
    y = np.array([franke2d_tensor(x[i])+noise[i] for i in range(0,size)])
    return train_test_split(x, y, train_size=p)


def init_weights(shape, trainable=True, seed=1743734):
    """ Weight initialization """
    # weights = tf.random_normal(shape, mean=0.0847, stddev=0.4737, seed=seed
    weights = tf.truncated_normal(shape, seed=seed)
    return tf.Variable(weights, trainable=trainable)


def activation(t, sigma=0.1):
    return tf.exp(-tf.square(t)/(sigma*sigma))



from itertools import product

sigma_range = [0.01, 0.05, 0.1, 0.5, 1]
rho_range = [0.0001, 0.001, 0.01]
N_range = [5, 20, 30, 50, 75]

data = pd.DataFrame([])
for N, sigma, rho in product(N_range, sigma_range, rho_range):

    INPUT_UNITS = 2
    HIDDEN_UNITS = N            # Number of hidden nodes
    OUTPUT_UNITS = 1
    LEARNING_RATE = 0.001
    SAMPLE_SIZE = 1000
    TRAIN_TEST_PROPORTION = 0.7
    RHO_CONSTANT = rho
    SIGMA_CONSTANT = sigma
    GENERATE_IMAGES = False
    EPOCHS = 1000

    # Creating dataset
    train_X, test_X, train_y, test_y = generation_train_test(size=SAMPLE_SIZE, p=TRAIN_TEST_PROPORTION)
    test_y = np.reshape(test_y, newshape=(int(SAMPLE_SIZE * (1 - TRAIN_TEST_PROPORTION)), OUTPUT_UNITS))
    train_y = np.reshape(train_y, newshape=(int(SAMPLE_SIZE * TRAIN_TEST_PROPORTION), OUTPUT_UNITS))

    # Writing placeholders
    X = tf.placeholder("float", shape=[None, INPUT_UNITS])
    y = tf.placeholder("float", shape=[None, OUTPUT_UNITS])

    # Defining omega
    omega = {
        'w': tf.ones(shape=[HIDDEN_UNITS, INPUT_UNITS]),
        'c': init_weights([HIDDEN_UNITS, INPUT_UNITS]),
        'v': init_weights([HIDDEN_UNITS, OUTPUT_UNITS])
    }

    # Defining pi
    pi = {
        'rho': tf.constant(RHO_CONSTANT)
    }

    m = []
    for cj in tf.unstack(omega['c']):
        resta = tf.subtract(X, cj)
        norma = tf.norm(resta, axis=1)
        phi = activation(norma)
        m.append(phi)
    PHI = tf.stack(m, axis=1)
    y_hat = tf.matmul(PHI, omega['v'])

    c = omega['c']
    v = omega['v']

    c_v = tf.concat([c, v], 1)

    reg = RHO_CONSTANT*tf.square(tf.norm(c_v))


    loss_function = (tf.reduce_sum(tf.square(tf.norm(tf.subtract(y_hat, y)))) / 2.0) + reg
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_function)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(EPOCHS):
        _, cost_3 = sess.run([optimizer, loss_function], feed_dict={X: train_X, y: train_y})
        loss_training = sess.run([loss_function], feed_dict={X: test_X, y:test_y})
        print("\r epoch: {} loss_training: {} loss_test: {}".format(epoch, cost_3, loss_training), end="")

    print("centers ", sess.run(omega['c']))
    print("v", sess.run(omega['v']))

    y_real = sess.run([y_hat], feed_dict={X: train_X, y: train_y})
    y_pred = sess.run([y_hat], feed_dict={X: test_X, y: test_y})

    #plot_pred_real(train_y, y_real)
    #plot_pred_real(test_y, y_pred)

    #plot_3d_data(train_X, train_y)
    y_pred = np.array(y_pred).reshape(test_y.shape)
    #plot_3d_data(test_X, y_pred)
    #plot_3d_data(test_X, test_y)

    data = data.append(pd.DataFrame({'N': N, 'rho': rho, 'sigma': sigma, 'Test Error': cost_3},
                                    index = [0]), ignore_index = True)

    print(N, rho, sigma, cost_3)