from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from util import plot_iter_vector


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


if __name__ == '__main__':
    VERBOSE = True
    train_X, test_X, train_y, test_y = generation_train_test(size=100, p=0.7)

    if VERBOSE:
        print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

    inertias = []

    for k in range(2, train_X.shape[0]):
        clustering = KMeans(n_clusters=k,
                            init='k-means++',
                            tol=1e-4,
                            random_state=1743734).fit(train_X)
        print(clustering.cluster_centers_.shape)
        print(clustering.cluster_centers_)
        inertias.append(clustering.inertia_)
        print(clustering.inertia_)

    plot_iter_vector(inertias)
