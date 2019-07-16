import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def mixedCMI(x, y, z, k=5, algorithm='auto', leaf_size=30):
    """
        KSG Conditional Mutual Information Estimator for continuous/discrete mixtures.
        
        See e.g. http://proceedings.mlr.press/v84/runge18a.html
        as well as: https://arxiv.org/abs/1709.06212
    """

    # dimensions & parameters
    xdim = x.shape[1] if x.ndim > 1 else 1
    ydim = x.shape[1] if y.ndim > 1 else 1
    zdim = x.shape[1] if z.ndim > 1 else 1
    params = {'algorithm': algorithm, 'leaf_size':leaf_size, 'metric':'chebyshev', 'n_jobs':1}

    # stack data for tree
    xzy = np.concatenate((x.reshape(-1,1) if x.ndim==1 else x, 
                          z.reshape(-1,1) if z.ndim==1 else z, 
                          y.reshape(-1,1) if y.ndim==1 else y), axis=1)

    # compute search radii and k values
    xzyNN = NearestNeighbors(**params)
    xzyNN.fit(xzy)

    e = xzyNN.kneighbors(n_neighbors=k, return_distance=True)[0][:,k-1]

    k_list = k*np.ones(e.shape, dtype='i')
    where_zero = np.array(e == 0.0, dtype='?')
    if np.sum(where_zero) > 0:
        matches = xzyNN.radius_neighbors(xzy[where_zero], radius=0.0, return_distance=False)
        k_list[where_zero] = np.fromiter(map(lambda x: len(x), matches), dtype='i')
    del (xzyNN)

    # count samples
    xzNN = NearestNeighbors(**params)
    xzNN.fit(xzy[:,:xdim+zdim])
    nxz = np.fromiter(map(lambda x: len(x), xzNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del(xzNN)

    zyNN = NearestNeighbors(**params)
    zyNN.fit(xzy[:,xdim:])
    nyz = np.fromiter(map(lambda x: len(x), zyNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del(zyNN)

    zNN = NearestNeighbors(**params)
    zNN.fit(xzy[:,xdim:xdim+zdim])
    nz = np.fromiter(map(lambda x: len(x), zNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del(zNN)

    return np.mean(digamma(k_list) - digamma(nxz+1) - digamma(nyz+1) + digamma(nz+1))


def ksgCMI(x, y, z, k=5, algorithm='auto', leaf_size=30, eps=1e-12):
    """
        KSG Conditional Mutual Information Estimator with added low amplitude noise to break ties.
        
        See e.g. http://proceedings.mlr.press/v84/runge18a.html

    """
    x += np.random.normal(0, 1, size=x.shape) * eps
    y += np.random.normal(0, 1, size=y.shape) * eps
    z += np.random.normal(0, 1, size=z.shape) * eps

    # dimensions & parameters
    xdim = x.shape[1] if x.ndim > 1 else 1
    ydim = x.shape[1] if y.ndim > 1 else 1
    zdim = x.shape[1] if z.ndim > 1 else 1
    params = {'algorithm': algorithm, 'leaf_size':leaf_size, 'metric':'chebyshev', 'n_jobs':1}

    # stack data for tree
    xzy = np.concatenate((x.reshape(-1,1) if x.ndim==1 else x, 
                          z.reshape(-1,1) if z.ndim==1 else z, 
                          y.reshape(-1,1) if y.ndim==1 else y), axis=1)

    # compute search radii and k values
    xzyNN = NearestNeighbors(**params)
    xzyNN.fit(xzy)

    e = xzyNN.kneighbors(n_neighbors=k, return_distance=True)[0][:,k-1]
    del (xzyNN)

    # count samples
    xzNN = NearestNeighbors(**params)
    xzNN.fit(xzy[:,:xdim+zdim])
    nxz = np.fromiter(map(lambda x: len(x), xzNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del(xzNN)

    zyNN = NearestNeighbors(**params)
    zyNN.fit(xzy[:,xdim:])
    nyz = np.fromiter(map(lambda x: len(x), zyNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del(zyNN)

    zNN = NearestNeighbors(**params)
    zNN.fit(xzy[:,xdim:xdim+zdim])
    nz = np.fromiter(map(lambda x: len(x), zNN.radius_neighbors(radius=e, return_distance=False)), dtype='i')
    del(zNN)

    return np.mean(digamma(k) - digamma(nxz+1) - digamma(nyz+1) + digamma(nz+1))
