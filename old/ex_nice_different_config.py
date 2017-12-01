# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from util import generate_data_bias, plot_pred_real, plot_3d_data, plot_3d_both_functions, generation_train_test
from util import plot_iter_vector
from sklearn import datasets
from sklearn.model_selection import train_test_split
from util import generate_data

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def activation (t, sigma=5):
    return((1-tf.exp(-sigma*t))/(1+tf.exp(-sigma*t)))

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def create_dir_images(names):
    import os

    if not os.path.exists(names['train']):
        os.makedirs(names['train'])

    if not os.path.exists(names['test']):
        os.makedirs(names['test'])


def main():
    train_X, test_X, train_y, test_y= generation_train_test(size=300, p=0.7)
    print(test_X.shape, train_y.shape)
    # Layer's sizes
    INPUT_UNITS = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
    HIDDEN_UNITS = 10  # Number of hidden nodes
    OUTPUT_UNITS = train_y.shape[0]  # Number of outcomes (3 iris flowers)
    LEARNING_RATE = 0.009
    GENERATE_IMAGES = False

    names = {
        'train': 'img\\train{}_{}'.format(str(LEARNING_RATE).replace('.', '-'), HIDDEN_UNITS),
        'test': 'img\\test{}_{}'.format(str(LEARNING_RATE).replace('.', '-'), HIDDEN_UNITS)
    }

    create_dir_images(names)

    # Symbols
    X = tf.placeholder("float32")
    y = tf.placeholder("float32")

    omega = {
        'v': tf.Variable(tf.truncated_normal(shape=[HIDDEN_UNITS, 1], dtype=tf.float32), name="v"),
        'w': tf.Variable(tf.truncated_normal(shape=[HIDDEN_UNITS, 2], dtype=tf.float32), name="w"),
        'b': tf.Variable(tf.truncated_normal(shape=[HIDDEN_UNITS, 1], dtype=tf.float32), name="b")
    }

    pi = {
        'rho': tf.constant(0.0001)
    }

    con = tf.concat([omega['v'], omega['w'], omega['b']], 1)
    regularization = pi['rho'] * tf.square(tf.norm(con))

    # tanh mat w tras x -b
    # mul v hidden
    mm = activation(tf.add(tf.matmul(omega['w'], X, transpose_b=True), omega['b']))
    f = tf.matmul(omega['v'], mm, transpose_a=True)

    mse = tf.reduce_sum(tf.square(tf.norm(tf.subtract(f, y)))) / 2.0

    loss = mse + regularization
    # reduce_mean

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    losses = {
        'train': [],
        'test': []
    }

    loss_old = 0

    for epoch in range(10000):
        # Train with each example
        _, loss_training = sess.run([optimizer, loss], feed_dict={X: train_X, y: train_y})

        if(abs(loss_old-loss_training)/loss_training < 1e-5):
            break
        loss_old = loss_training
        loss_test = sess.run([loss], feed_dict={X: test_X, y:test_y})
        losses['train'].append(loss_training)
        losses['test'].append(loss_test)
        #if(abs(loss_past-loss_value)/loss_value < 1e-6):

        if(epoch % 50 == 0):
            print("epoch: {} loss_training: {} loss_test: {}".format(epoch, loss_training, loss_test))
            y_pred = sess.run([f], feed_dict={X: test_X, y: test_y})
            y_pred = np.array(y_pred).reshape(test_y.shape)
            y_pred_train = sess.run([f], feed_dict={X: train_X, y: train_y})
            y_pred_train = np.array(y_pred_train).reshape(train_y.shape)

            if GENERATE_IMAGES:
                plot_3d_both_functions(test_X, test_y, y_pred,
                                       name="test{}_{}\\{}".format(str(LEARNING_RATE).replace('.','-'), HIDDEN_UNITS, epoch))
                plot_3d_both_functions(train_X, train_y, y_pred_train,
                                       name="train{}_{}\\{}".format(str(LEARNING_RATE).replace('.','-'), HIDDEN_UNITS, epoch))

    #plot_iter_vector(losses['train'])
    #plot_iter_vector(losses['test'])
    plot_pred_real(test_y, y_pred)
    plot_pred_real(train_y, y_pred_train)

    sess.close()


if __name__ == '__main__':
    main()