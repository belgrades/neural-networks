# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
from util import generate_data, split_test_train, plot_3d_data, plot_pred_real
import numpy as np



if __name__ == "__main__":
    sample_size, p = 100, 0.3
    hidden_units = 10

    train_size, test_size = int((1.0-p)*sample_size), int(sample_size*p)
    sample_size = train_size + test_size
    
    dataset, utopy = generate_data(n=sample_size)
    test, train = split_test_train(dataset, size=test_size)

    train_x1_x2, train_y = np.split(train, [2], 1)
    test_x1_x2 , test_y = np.split(test, [2], 1)

    rho = tf.constant(0.01, dtype=tf.float32)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    v = tf.Variable(tf.truncated_normal(shape=[hidden_units,1], dtype=tf.float32), name="v")
    w = tf.Variable(tf.truncated_normal(shape=[hidden_units, 2], dtype=tf.float32), name="w")
    b = tf.Variable(tf.truncated_normal(shape=[hidden_units,1], dtype=tf.float32), name="b")

    con = tf.concat([v, w, b], 1)
    regularization = rho * tf.square(tf.norm(con))

    #tanh mat w tras x -b
    # mul v hidden
    mm = tf.tanh(tf.subtract(tf.matmul(w, x, transpose_b=True), b))
    f = tf.matmul(v, mm, transpose_a=True)

    mse = tf.reduce_mean(tf.squared_difference(f,y)) / 2.0

    loss = mse + regularization
    # reduce_mean
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    train_x1_x2 = np.array(train_x1_x2, dtype=np.dtype('float32'))
    train_y = np.array(train_y, dtype=np.dtype('float32'))

    test_x1_x2 = np.array(test_x1_x2, dtype=np.dtype('float32'))
    test_y = np.array(test_y, dtype=np.dtype('float32'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            a = sess.run([optimizer, loss], feed_dict={x: train_x1_x2, y: train_y})
            print(i, a[1])

        y_pred = sess.run([f], feed_dict={x: test_x1_x2})

## TEST ##
        plot_pred_real(test_y.reshape(test_size,), y_pred[0][0])
        plot_3d_data(test_x1_x2, y_pred[0].reshape(test_size,1))
        plot_3d_data(test_x1_x2, test_y)
        plot_3d_data(train_x1_x2, train_y)

print("hola")
for j, k in zip(test_y, y_pred[0][0]):
    print(j, k)