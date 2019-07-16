import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def mixedMI(x, y, k=5, algorithm='auto', leaf_size=30):
    """
        KSG Mutual Information Estimator for continuous/discrete mixtures.
        
        Based on: https://arxiv.org/abs/1709.06212
    """

    # dimensions & parameters
    xdim = x.shape[1] if x.ndim > 1 else 1
    ydim = y.shape[1] if y.ndim > 1 else 1
    params = {'algorithm': algorithm, 'leaf_size':leaf_size, 'metric':'chebyshev', 'n_jobs':1}

    # stack data for tree
    xy = np.concatenate((x.reshape(-1,1) if x.ndim==1 else x, 
                         y.reshape(-1,1) if y.ndim==1 else y), axis=1)

    # compute search radii and k values
    xyNN = NearestNeighbors(**params)
    xyNN.fit(xy)

    e = xyNN.kneighbors(n_neighbors=k, return_distance=True)[0][:,k-1]

    k_list = k*np.ones(e.shape, dtype='i')
    where_zero = np.array(e == 0.0, dtype='?')
    if np.sum(where_zero) > 0:
        matches = xyNN.radius_neighbors(xy[where_zero], radius=0.0, return_distance=False)
        k_list[where_zero] = np.fromiter(map(lambda x: len(x), matches), dtype='i')
    del (xyNN)

    # count samples
    xNN = NearestNeighbors(**params)
    xNN.fit(x)
    nx = np.fromiter(map(lambda x: len(x), xNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del (xNN)

    yNN = NearestNeighbors(**params)
    yNN.fit(y)
    ny = np.fromiter(map(lambda x: len(x), yNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del (yNN)

    return np.log(xy.shape[0]) + np.mean(digamma(k_list) - digamma(nx+1) - digamma(ny+1))


def ksgMI(x, y, k=5, algorithm='auto', leaf_size=30, eps=1e-12):
    """
        KSG Mutual Information Estimator, 
        with low amplitude noise added to break ties.

        Based on: https://arxiv.org/abs/cond-mat/0305641
    """
    x += np.random.normal(0, 1, size=x.shape) * eps
    y += np.random.normal(0, 1, size=y.shape) * eps

    # dimensions & parameters
    xdim = x.shape[1] if x.ndim > 1 else 1
    ydim = y.shape[1] if y.ndim > 1 else 1
    params = {'algorithm': algorithm, 'leaf_size':leaf_size, 'metric':'chebyshev', 'n_jobs':1}

    # stack data for tree
    xy = np.concatenate((x.reshape(-1,1) if x.ndim==1 else x, 
                         y.reshape(-1,1) if y.ndim==1 else y), axis=1)

    # compute search radii and k values
    xyNN = NearestNeighbors(**params)
    xyNN.fit(xy)

    e = xyNN.kneighbors(n_neighbors=k, return_distance=True)[0][:,k-1]
    del (xyNN)

    # count samples
    xNN = NearestNeighbors(**params)
    xNN.fit(x)
    nx = np.fromiter(map(lambda x: len(x), xNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del (xNN)

    yNN = NearestNeighbors(**params)
    yNN.fit(y)
    ny = np.fromiter(map(lambda x: len(x), yNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del (yNN)

    return np.log(xy.shape[0]) + np.mean(digamma(k) - digamma(nx+1) - digamma(ny+1))
