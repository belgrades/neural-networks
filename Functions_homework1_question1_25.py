from numpy import exp
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


def franke2d_tensor(x):
    t1 = 0.75*exp(-(9*x[0]-2)**2/4-(9*x[1]-2)**2/4)
    t2 = 0.75*exp(-(9*x[0]+1)**2/49 -(9*x[1]+1)/10)
    t3 = 0.5*exp(-(9*x[0]-7)**2/4-(9*x[1]-3)**2/4)
    t4 = -0.2*exp(-(9*x[0]-4)**2-(9*x[1]-7)**2)
    return t1+t2+t3+t4


def g(t, sigma=1.0):
    return (1-tf.exp(-sigma*t))/(1+tf.exp(-sigma*t))


def activation(t, sigma=0.1):
    return tf.exp(-tf.square(t)/(sigma*sigma))


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


def generate_rbf_variables(train_X, INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, RHO_CONSTANT, SIGMA_CONSTANT):
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

    con = tf.concat([tf.reshape(omega['w'], shape=[1, INPUT_UNITS * HIDDEN_UNITS]),
                     tf.reshape(omega['c'], shape=[1, HIDDEN_UNITS * INPUT_UNITS]),
                     tf.reshape(omega['v'], shape=[1, HIDDEN_UNITS])], 1)

    return X, y, omega, pi, con


def generate_mlp_variables(INPUT_UNITS, OUTPUT_UNITS, HIDDEN_UNITS, RHO_CONSTANT, SIGMA_CONSTANT):
    X = tf.placeholder("float", shape=[None, INPUT_UNITS])
    y = tf.placeholder("float", shape=[None, OUTPUT_UNITS])

    # Weight initializations
    omega = {
        'w': init_weights([INPUT_UNITS, HIDDEN_UNITS]),
        'v': init_weights([HIDDEN_UNITS, OUTPUT_UNITS]),
        'b': init_weights([HIDDEN_UNITS])
    }

    pi = {
        'rho': tf.constant(RHO_CONSTANT),
        'sigma': SIGMA_CONSTANT
    }

    con = tf.concat([tf.reshape(omega['w'], shape=[1, INPUT_UNITS * HIDDEN_UNITS]),
                     tf.reshape(omega['v'], shape=[1, HIDDEN_UNITS * OUTPUT_UNITS]),
                     tf.reshape(omega['b'], shape=[1, HIDDEN_UNITS])], 1)

    return X, y, omega, pi, con


def generate_rbf_y_hat_function(X, omega, pi):
    m = []
    for cj in tf.unstack(omega['c']):
        sub = tf.subtract(X, cj)
        norma = tf.norm(sub, axis=1)
        phi = activation(norma)
        m.append(phi)
    PHI = tf.stack(m, axis=1)
    y_hat = tf.matmul(PHI, omega['v'])

    return y_hat


def generate_mlp_y_hat_function(X, omega, pi):
    h = g((tf.matmul(X, omega['w']) - omega['b']), sigma=pi['sigma'])
    y_hat = tf.matmul(h, omega['v'])

    return y_hat


def generate_generic_loss_functions(y_hat, y, pi, con, SAMPLE_SIZE=100, TRAIN_TEST_PROPORTION=0.7):
    regularization = pi['rho'] * tf.reduce_sum(tf.square(tf.norm(con)))
    # Forward propagation

    train_loss = tf.reduce_sum(tf.square(tf.norm(tf.subtract(y_hat, y)))) / ((SAMPLE_SIZE * TRAIN_TEST_PROPORTION) * 2.0 )+ regularization
    test_loss = tf.reduce_sum(tf.square(tf.norm(tf.subtract(y_hat, y)))) / (SAMPLE_SIZE * TRAIN_TEST_PROPORTION) * 2.0

    return train_loss, test_loss


def generate_mse_functions(y_hat, y, SAMPLE_SIZE, TRAIN_TEST_PROPORTION):
    mse_test = tf.reduce_sum(tf.square(tf.subtract(y_hat, y))) / (SAMPLE_SIZE * (1 - TRAIN_TEST_PROPORTION))
    mse_training = tf.reduce_sum(tf.square(tf.subtract(y_hat, y))) / (SAMPLE_SIZE * TRAIN_TEST_PROPORTION)

    return mse_test, mse_training


def generate_generic_optimizer(loss_function, LEARNING_RATE=0.01):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_function)


def init_tf_session():
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    return sess


def train_generic_model(sess, optimizer, train_loss, omega, X, y, train_X, train_y,
                        EPOCHS=10000, COMPUTE_TIME=True, tol=1e-9, target="MLP"):
    toc = None
    if COMPUTE_TIME:
        tic = time()

    train_loss_actual = 0
    for epoch in range(1, EPOCHS+1):
        # Train with each example
        _, train_loss_new = sess.run([optimizer, train_loss],
                                     feed_dict={X: train_X, y: train_y})
        if abs(train_loss_actual-train_loss_new) < tol:
            break
        else:
            train_loss_actual = train_loss_new

    if COMPUTE_TIME:
        toc = time()-tic

    if target == "MLP":
        w_opt, b_opt, v_opt = sess.run([omega['w'], omega['b'], omega['v']])
        return w_opt, b_opt, v_opt, epoch, toc
    else:
        w_opt, c_opt, v_opt = sess.run([omega['w'], omega['c'], omega['v']])
        return w_opt, c_opt, v_opt, epoch, toc


def predict_mlp(sess, y_hat, X, test_X):
    return sess.run([y_hat], feed_dict={X: test_X})


def predict_mlp_w_b_v(sess, y_hat, w_opt, b_opt, v_opt, X, test_X):
    return sess.run([y_hat], feed_dict={X: test_X})


def compute_mse(sess, mse_function, X, y, X_input, y_input):
    return sess.run([mse_function], feed_dict={X: X_input, y: y_input})


def compute_loss(sess, loss_function, X, y, input_X, input_y):
    return sess.run([loss_function], feed_dict={X: input_X, y: input_y})


def plot_pred_real(y, y_pred,
                   x_lim=(-0.2,1.2), y_lim=(-0.2, 1.2), xlab="", ylab="",title=""):
    plt.scatter(np.array(y), np.array(y_pred))
    plt.plot([x_lim[0], x_lim[1]], [y_lim[0], y_lim[1]], 'k-', lw=3)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()
    plt.close()


def plot_3d_data(train_x1_x2, pred_y, test_y=None, save=False, name=None, title=""):
    train_x1, train_x2 = np.split(train_x1_x2,[1], 1)
    n = train_x1.shape[0]

    train_x1 = train_x1.reshape([n])
    train_x2 = train_x2.reshape([n])
    pred_y = pred_y.reshape([n])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(train_x1,
                    train_x2,
                    pred_y,
                    linewidth=0.2,
                    antialiased=True,
                    cmap=cm.coolwarm)

    ax.scatter(train_x1, train_x2, pred_y)
    plt.title(title)

    if save:
        plt.savefig('C:\\Users\\Giulia\\fernando\\optimization\\neural-networks\\img\\{}.png'.format(name))
    else:
        plt.show()
    plt.close()


