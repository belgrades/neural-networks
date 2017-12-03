# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:10:47 2017

@author: Fernando Crema
"""
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def create_dir_images(names):
    import os

    if not os.path.exists(names['train']):
        os.makedirs(names['train'])

    if not os.path.exists(names['test']):
        os.makedirs(names['test'])

    if not os.path.exists(names['test_solo']):
        os.makedirs(names['test_solo'])

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
    noise = np.random.uniform(a, b,  size=size)
    y = np.array([franke2d_tensor(x[i])+noise[i] for i in range(0,size)])
    return train_test_split(x, y, train_size=p)

def generate_data(n=100, a=-1e-1, b=1e-1, seed=1743734):
    # Generate the Columns x1 x2 from uniform([0,1]x[0,1])
    data = np.random.uniform(size = [n, 2],
                             low=0,
                             high=1)
    
    # Generate 1 as franke2d(x1, x2)
    y = np.reshape(list(map(franke2d_tensor, data)), (n, 1))
    
    # Generate noise shape (100, 1) from uniform(a, b)
    noise = np.random.uniform(size = [n, 1],
                              low=a,
                              high=b)
    
    # Adding noise to y
    w_noise = y + noise

    final = np.concatenate((data, w_noise), 1)

    return final, w_noise


def generate_data_bias(n=100, a=-1e-1, b=1e-1, RANDOM_SEED=1743734):
    """
    Function to generate the data adding the bias term.
    :param n: 
    :param a: 
    :param b: 
    :param seed: 
    :return: 
    """
    # Generate the Columns x1 x2 from uniform([0,1]x[0,1])


    all_X= np.random.uniform(size=[n, 2],
                             low=0,
                             high=1)

    # Generate 1 as franke2d(x1, x2)
    y = np.reshape(list(map(franke2d_tensor, all_X)), (n, 1))

    # Generate noise shape (100, 1) from uniform(a, b)
    noise = np.random.uniform(size=[n, 1],
                              low=a,
                              high=b)

    # Adding noise to y
    all_Y = y + noise

    # Adding the bias term

    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)


def split_test_train(dataset, size=30):
    """
    Method that randomly splits the dataset using random_shuffle and split.
    """
    return np.split(dataset, [size], 0)


def plot_compare_losses(loss_training, loss_test, y_lim=(0.0, 1.0)):
    grid = list(range(len(loss_test)))
    print(loss_training)
    plt.scatter(grid, loss_training)
    plt.scatter(grid, loss_test)
    plt.title('Loss value in function of epochs')
    plt.xlabel('Epochs mod 50')
    plt.ylabel('E(w, pi)')
    plt.ylim(y_lim)
    plt.show()
    plt.close()


def plot_iter_vector(vec, verbose=True):
    if verbose:
        print("[plot iteration vector {}".format("inertia"), vec)
    plot_pred_real(range(len(vec)), vec)


def plot_pred_real(y, y_pred, x_lim=(-0.2,1.2), y_lim=(-0.2, 1.2)):
    print(y_pred)
    plt.scatter(np.array(y), np.array(y_pred))
    plt.plot([x_lim[0], x_lim[1]], [y_lim[0], y_lim[1]], 'k-', lw=3)
    #plt.scatter('y', 'y_pred', data=data)
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.show()
    plt.close()


def plot_3d_data(train_x1_x2, pred_y, test_y=None, save=False, name=None):
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

    if save:
        plt.savefig('C:\\Users\\Giulia\\fernando\\optimization\\neural-networks\\img\\{}.png'.format(name))
    else:
        plt.show()
    plt.close()


def plot_3d_both_functions(test_x1_x2, test_y, pred_y, name):
    x1, x2 = np.split(test_x1_x2, [1], 1)
    n = x1.shape[0]

    x1 = x1.reshape([n])
    x2 = x2.reshape([n])
    test_y = test_y.reshape([n])
    pred_y = pred_y.reshape([n])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x1,
                    x2,
                    pred_y,
                    linewidth=0.2,
                    antialiased=True,
                    cmap=cm.coolwarm)

    ax.scatter(x1, x2, pred_y)

    ax.plot_trisurf(x1,
                    x2,
                    test_y,
                    linewidth=0.2,
                    antialiased=True)

    ax.scatter(x1, x2, test_y, c='r')

    plt.savefig('C:\\Users\\Giulia\\fernando\\optimization\\neural-networks\\img\\{}.png'.format(name))
    plt.close()


def plot_3d_data_old(train_x1_x2, pred_y, test_y=None):
    train_x1, train_x2 = np.split(train_x1_x2, [1], 1)
    n = train_x1.shape[0]
    train_x1 = train_x1.reshape([n])
    train_x2 = train_x2.reshape([n])
    pred_y = pred_y.reshape([n])

    if test_y is not None:
        test_y = test_y.reshape([n])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(train_x1,
                    train_x2,
                    pred_y,
                    linewidth=0.2,
                    antialiased=True,
                    cmap=cm.coolwarm)
    ax.scatter(train_x1, train_x2, pred_y)

    plt.show()
    plt.close()
