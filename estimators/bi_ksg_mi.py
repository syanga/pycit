""" Bias improved k-NN mutual information estimator for continuous data """
import numpy as np
from scipy.special import digamma,gamma
from sklearn.neighbors import NearestNeighbors


def bi_ksg_mi(x_data, y_data, k=5):
    """
        Bias improved KSG Mutual Information Estimator
        Based on: https://arxiv.org/abs/1604.03006

        x_data: data with shape (num_samples, x_dim) or (num_samples,)
        y_data: data with shape (num_samples, y_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """
    assert x_data.shape[0] == y_data.shape[0]
    num_samples = x_data.shape[0]

    # compute search radii and k values
    lookup = NearestNeighbors(metric='euclidean', algorithm='auto')
    lookup.fit(np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                               y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1))

    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    # compute entropies
    lookup.fit(x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data)
    n_x = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data)
    n_y = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    # correction term for using 2-norm
    x_dim = x_data.shape[1] if x_data.ndim > 1 else 1
    y_dim = y_data.shape[1] if y_data.ndim > 1 else 1
    correction = log_vol(x_dim) + log_vol(y_dim) - log_vol(x_dim+y_dim)

    return correction + np.log(num_samples) + digamma(k) - np.mean(np.log(n_x) + np.log(n_y))


def log_vol(dim, p_norm=2.):
    """ log-volume of d-dimensional hypersphere, with respect to the p-norm (p<infty) """
    return dim*np.log(2*gamma(1.+1./p_norm)) - np.log(gamma(1.+dim*1./p_norm))
