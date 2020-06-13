""" k-NN mutual information estimator for mixed continuous-discrete data """
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def mixed_mi(x_data, y_data, k=5):
    """
        KSG Mutual Information Estimator for continuous/discrete mixtures.
        Based on: https://arxiv.org/abs/1709.06212

        x: data with shape (num_samples, x_dim) or (num_samples,)
        y: data with shape (num_samples, y_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing 
    """
    assert x_data.shape[0] == y_data.shape[0]
    num_samples = x_data.shape[0]

    x_y = np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                          y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1)

    lookup = NearestNeighbors(metric='chebyshev')
    lookup.fit(x_y)

    # compute k-NN distances
    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    # modification for discrete-continuous
    k_list = k*np.ones(radius.shape, dtype='i')
    where_zero = np.array(radius == 0.0, dtype='?')
    if np.any(where_zero > 0):
        matches = lookup.radius_neighbors(x_y[where_zero], radius=0.0, return_distance=False)
        k_list[where_zero] = np.array([i.size for i in matches])

    # estimate entropies
    lookup.fit(x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data)
    n_x = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data)
    n_y = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    return digamma(num_samples) + np.mean(digamma(k_list) - digamma(n_x+1.) - digamma(n_y+1.))
