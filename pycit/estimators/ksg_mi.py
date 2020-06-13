""" k-NN mutual information estimator for continuous data """
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def ksg_mi(x_data, y_data, k=5):
    """
        KSG Mutual Information Estimator
        Based on: https://arxiv.org/abs/cond-mat/0305641

        x_data: data with shape (num_samples, x_dim) or (num_samples,)
        y_data: data with shape (num_samples, y_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """
    assert x_data.shape[0] == y_data.shape[0]
    num_samples = x_data.shape[0]

    lookup = NearestNeighbors(metric='chebyshev')
    lookup.fit(np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                               y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1))

    # estimate entropies
    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    lookup.fit(x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data)
    n_x = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data)
    n_y = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    return digamma(num_samples) + digamma(k) - np.mean(digamma(n_x+1.) + digamma(n_y+1.))
