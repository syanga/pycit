""" k-NN conditional mutual information estimator for discrete-continuous mixtures """
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def mixed_cmi(x_data, y_data, z_data, k=5):
    """
        KSG Conditional Mutual Information Estimator for continuous/discrete mixtures.

        See e.g. http://proceedings.mlr.press/v84/runge18a.html
        as well as: https://arxiv.org/abs/1709.06212

        x_data: data with shape (num_samples, x_dim) or (num_samples,)
        y_data: data with shape (num_samples, y_dim) or (num_samples,)
        z_data: conditioning data with shape (num_samples, z_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """
    xzy_data = np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                               z_data.reshape(-1, 1) if z_data.ndim == 1 else z_data,
                               y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1)

    lookup = NearestNeighbors(metric='chebyshev')
    lookup.fit(xzy_data)

    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    # modification for discrete-continuous
    k_list = k*np.ones(radius.shape, dtype='i')
    where_zero = np.array(radius == 0.0, dtype='?')
    if np.any(where_zero > 0):
        matches = lookup.radius_neighbors(xzy_data[where_zero], radius=0.0, return_distance=False)
        k_list[where_zero] = np.array([i.size for i in matches])

    x_dim = x_data.shape[1] if x_data.ndim > 1 else 1
    z_dim = z_data.shape[1] if z_data.ndim > 1 else 1

    # compute entropies
    lookup.fit(xzy_data[:, :x_dim+z_dim])
    n_xz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy_data[:, x_dim:])
    n_yz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy_data[:, x_dim:x_dim+z_dim])
    n_z = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    return np.mean(digamma(k_list) + digamma(n_z+1.) - digamma(n_xz+1.) - digamma(n_yz+1.))
