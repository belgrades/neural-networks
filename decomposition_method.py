import tensorflow as tf
import numpy as np
from util import generate_data_bias, plot_pred_real, plot_3d_data
from sklearn import datasets
from sklearn.model_selection import train_test_split
from util import generate_data

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

from util import generation_train_test, create_dir_images, plot_3d_both_functions, plot_compare_losses

def init_weights(shape, trainable=True, seed=1743734):
    """ Weight initialization """
    # weights = tf.random_normal(shape, mean=0.0847, stddev=0.4737, seed=seed
    weights = tf.truncated_normal(shape, seed=seed)
    return tf.Variable(weights)


def activation (t, sigma=5.0):
    return((1-tf.exp(-sigma*t))/(1+tf.exp(-sigma*t)))

# TODO: Rename yhat and h

def main():
    # Layer's sizes
    INPUT_UNITS = 2     # Number of input nodes: 4 features and 1 bias
    HIDDEN_UNITS = 100              # Number of hidden nodes
    OUTPUT_UNITS = 1   # Number of outcomes (3 iris flowers)
    LEARNING_RATE = 0.01
    SAMPLE_SIZE = 1000
    TRAIN_TEST_PROPORTION = 0.7
    RHO_CONSTANT = 0.00001
    SIGMA_CONSTANT = 4
    GENERATE_IMAGES = False
    EPOCHS = 20000

    NAMES = {
        'train': 'img\\train{}_{}'.format(str(LEARNING_RATE).replace('.', '-'), HIDDEN_UNITS),
        'test': 'img\\test{}_{}'.format(str(LEARNING_RATE).replace('.', '-'), HIDDEN_UNITS),
        'test_solo': 'img\\test_solo{}_{}'.format(str(LEARNING_RATE).replace('.', '-'), HIDDEN_UNITS)
    }

    create_dir_images(NAMES)

    train_X, test_X, train_y, test_y = generation_train_test(size=SAMPLE_SIZE, p=TRAIN_TEST_PROPORTION)
    print(train_y.shape)
    test_y = np.reshape(test_y, newshape=(int(SAMPLE_SIZE*(1-TRAIN_TEST_PROPORTION)), OUTPUT_UNITS))
    train_y = np.reshape(train_y, newshape=(int(SAMPLE_SIZE*TRAIN_TEST_PROPORTION), OUTPUT_UNITS))
    print(train_y.shape)

    # Symbols
    X = tf.placeholder("float", shape=[None, INPUT_UNITS])
    y = tf.placeholder("float", shape=[None, OUTPUT_UNITS])

    # Weight initializations

    decomposition = ['convex', 'non_convex']

    omega = {
        'w': init_weights([INPUT_UNITS, HIDDEN_UNITS]),
        'v': init_weights([HIDDEN_UNITS, OUTPUT_UNITS]),
        'b': init_weights([HIDDEN_UNITS])
    }

    pi ={
        'rho': tf.constant(RHO_CONSTANT),
        'sigma': SIGMA_CONSTANT
    }

    con = tf.concat([tf.reshape(omega['w'], shape=[1, INPUT_UNITS * HIDDEN_UNITS]),
                     tf.reshape(omega['v'], shape=[1, HIDDEN_UNITS * OUTPUT_UNITS]),
                     tf.reshape(omega['b'], shape=[1, HIDDEN_UNITS])], 1)

    regularization = pi['rho'] * tf.reduce_sum(tf.square(tf.norm(con)))
    # Forward propagation
    h = activation((tf.matmul(X, omega['w']) - omega['b']), sigma=pi['sigma'])
    yhat = tf.matmul(h, omega['v'])

    # Backward propagation
    #cost    =  (1/2.0)*tf.losses.mean_squared_error(yhat, y)
    cost = tf.reduce_sum(tf.square(tf.norm(tf.subtract(yhat, y)))) / 2.0 + regularization
    test_cost = tf.reduce_mean(tf.square(tf.norm(tf.subtract(yhat, y)))) / 2.0

    convex = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost, var_list=[omega['v']])
    non_convex = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost, var_list=[omega['w'], omega['b']])

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    losses = {
        'train': [],
        'test': []
    }

    decomp_stage = 'convex'

    for epoch in range(EPOCHS):
        # Train with each example
        if decomp_stage == 'convex':
            _, loss_training, w_opt_1 = sess.run([convex, cost, omega['w']],
                                               feed_dict={X: train_X, y: train_y})
            decomp_stage = 'non_convex'
        else:
            _, loss_training, w_opt_2 = sess.run([non_convex, cost, omega['w']],
                                               feed_dict={X: train_X, y: train_y})
            decomp_stage = 'convex'
        loss_test = sess.run([test_cost], feed_dict={X: test_X, y:test_y})

        if (epoch % 50 == 1):
            print("epoch: {} loss_training: {} loss_test: {}".format(epoch, loss_training, loss_test))
            print("norm diff", sess.run(tf.norm(tf.subtract(w_opt_1, w_opt_2))))
            y_pred = sess.run([yhat], feed_dict={X: test_X, y: test_y})
            y_pred = np.array(y_pred).reshape(test_y.shape)

            y_pred_train = sess.run([yhat], feed_dict={X: train_X, y: train_y})
            y_pred_train = np.array(y_pred_train).reshape(train_y.shape)

            losses['train'].append(loss_training)
            losses['test'].append(loss_test)

            if GENERATE_IMAGES:
                plot_3d_data(test_X, y_pred, save=True,
                             name="test_solo{}_{}\\{}".format(str(LEARNING_RATE).replace('.', '-'), HIDDEN_UNITS,
                                                                   epoch))
                plot_3d_both_functions(test_X, test_y, y_pred,
                                       name="test{}_{}\\{}".format(str(LEARNING_RATE).replace('.', '-'), HIDDEN_UNITS,
                                                                   epoch))
                plot_3d_both_functions(train_X, train_y, y_pred_train,
                                       name="train{}_{}\\{}".format(str(LEARNING_RATE).replace('.', '-'), HIDDEN_UNITS,
                                                                    epoch))

    y_real = sess.run([yhat], feed_dict={X: train_X, y: train_y})
    y_pred = sess.run([yhat], feed_dict={X: test_X, y: test_y})
    print("w1", w_opt_2)
    print("w2", w_opt_1)
    plot_compare_losses(losses['train'], losses['test'])
    plot_pred_real(train_y, y_real)
    plot_pred_real(test_y, y_pred)

    plot_3d_data(train_X, train_y)
    y_pred = np.array(y_pred).reshape(test_y.shape)
    plot_3d_data(test_X, y_pred)

    print(sess.run(omega['w']))

    sess.close()

if __name__ == '__main__':
    main()
