import tensorflow as tf
import numpy as np
from util import generate_data, split_test_train, plot_pred_real, plot_3d_data, plot_iter_vector


if __name__ == "__main__":
    # Define hyperparameters
    EPOCHS = 1000
    LEARNING_RATE = 0.01
    SEED = 1743734
    SAMPLE_SIZE, p = 1000, 0.3
    INPUT_UNITS = 2
    HIDDEN_UNITS = 10
    OUTPUT_UNITS = 1

    # Define Dataset
    train_size, test_size = int((1.0 - p) * SAMPLE_SIZE), int(SAMPLE_SIZE * p)
    SAMPLE_SIZE = train_size + test_size

    dataset, utopy = generate_data(n=SAMPLE_SIZE)
    test, train = split_test_train(dataset, size=test_size)
    train_x1_x2, train_y = np.split(train, [2], 1)
    test_x1_x2, test_y = np.split(test, [2], 1)

    print("[info] test",type(train_y), train_y.shape)
    print("[info] train", type(train), train.shape)

    # Define placeholders
    x = tf.placeholder(tf.float32, [None, INPUT_UNITS])
    y = tf.placeholder(tf.float32, [None, OUTPUT_UNITS])

    # Creating Training model

    # Defining weights
    pi = {
        'rho': tf.constant(0.00001)
    }
    omega = {
        'w': tf.Variable(tf.random_normal([INPUT_UNITS, HIDDEN_UNITS])),
        'v': tf.Variable(tf.random_normal([HIDDEN_UNITS, OUTPUT_UNITS])),
        'b': tf.Variable(tf.random_normal([HIDDEN_UNITS]) )
    }

    # Creating the model
    con = tf.concat([tf.reshape(omega['w'], shape=[1, INPUT_UNITS*HIDDEN_UNITS]),
                     tf.reshape(omega['v'], shape=[1, HIDDEN_UNITS*OUTPUT_UNITS]),
                     tf.reshape(omega['b'], shape=[1, HIDDEN_UNITS])], 1)

    regularization = pi['rho'] * tf.reduce_sum(tf.square(tf.norm(con)))

    hidden_layer = tf.subtract(tf.matmul(x, omega['w']), omega['b'])
    activation = tf.tanh(hidden_layer/2.0)
    output_layer = tf.matmul(activation, omega['v'])

    # Creating the loss
    #cost = tf.losses.mean_squared_error(output_layer, y)
    cost = (1.0/2.0)*tf.reduce_sum(tf.squared_difference(output_layer, y))+regularization
    #cost = tf.reduce_sum(tf.square(tf.norm(tf.subtract(output_layer, y))))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Initialize variables
    init = tf.initialize_all_variables()

    # Session
    with tf.Session() as sess:
        # Create variables
        sess.run(init)

        # Epochs
        last_opt = 0
        loss_vec = []
        norm_diff = []
        cost_ini = 0
        for epoch in range(EPOCHS):
            timi, cost_opt, output_layerw, hidden_layer2, w, b, v = sess.run([optimizer, cost, output_layer, hidden_layer, omega['w'], omega['v'], omega['b']],
                                            feed_dict={x: train_x1_x2, y: train_y})
            loss_vec.append(cost_opt)
            print(abs(cost_ini-cost_opt))
            if(abs(cost_ini-cost_opt)/cost_opt < 1e-5):
                break
            else:
                cost_ini = cost_opt
            '''
                        for idx, row in enumerate(train_x1_x2):
                #print(row, row.shape, train_y[idx])
                _, cost_opt, w, b, v = sess.run([optimizer, cost, omega['w'], omega['v'], omega['b']], feed_dict={x: row.reshape(1, 2), y: train_y[idx].reshape(1, 1)})
            
            '''

            print(np.sum(np.square(output_layerw-train_y))/len(train_y))
            norm_diff.append(np.sum(np.square(output_layerw-train_y))/len(train_y))
            print(epoch, cost_opt)
        print(hidden_layer2)
        for kk, ll in zip(output_layerw, train_y):
            print(kk, ll)

    print("Solutions")
    print("hl", hidden_layer)
    print(w, w.shape, w.dtype)
    print(b, b.shape)
    print(v, v.shape)

    x_opt = tf.placeholder(tf.float32, [None, INPUT_UNITS])
    y_opt = tf.placeholder(tf.float32, [None, OUTPUT_UNITS])

    # Defining weights
    omega_opt = {
        'w_opt': tf.constant(w, shape=[INPUT_UNITS, HIDDEN_UNITS]),
        'v_opt': tf.constant(v, shape=[HIDDEN_UNITS, OUTPUT_UNITS]),
        'b_opt': tf.constant(b, shape=[HIDDEN_UNITS])
    }

    # Creating the model
    hidden_layer_opt = tf.subtract(tf.matmul(x_opt, omega_opt['w_opt']), omega_opt['b_opt'])
    activation_opt = tf.tanh(hidden_layer_opt/2.0)
    output_layer_opt = tf.matmul(activation_opt, omega_opt['v_opt'])

    # Creating the loss
    #cost = tf.reduce_sum(tf.square(tf.norm(tf.subtract(output_layer, y))))
    #cost_opt = tf.losses.mean_squared_error(output_layer_opt, y)
    cost_opt = (1.0 / 2.0) * tf.reduce_sum(tf.squared_difference(output_layer_opt, y_opt))

    y_pred = []

    with tf.Session() as sess:
        y, cost_opt, hidden_layer_opt  = sess.run([output_layer_opt, cost_opt, hidden_layer_opt],
                                        feed_dict={x_opt: test_x1_x2, y_opt: test_y})
        print(y, test_y, cost_opt)
        #plot_pred_real(test_y.reshape(test_size, ), y_pred[0][0])
        #plot_3d_data(test_x1_x2, y_pred[0].reshape(test_size, 1))4
        plot_iter_vector(norm_diff)
        #plot_pred_real(y=range(len(loss_vec)), y_pred=loss_vec)
        plot_pred_real(test_y, y)
        plot_3d_data(test_x1_x2, y)
        plot_3d_data(test_x1_x2, test_y)
        plot_3d_data(train_x1_x2, train_y)

    print("CHECKING")
    print(hidden_layer_opt)
    print('w', w)
    print('b', b)
    print('v', v)
    print('x0', test_x1_x2[0])
    print('ypred', y[0])



