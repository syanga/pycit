""" Bias corrected k-NN conditional mutual information estimator for continuous data """
import numpy as np
from scipy.special import digamma,gamma
from sklearn.neighbors import NearestNeighbors


def bi_ksg_cmi(x_data, y_data, z_data, k=5):
    """
        Bias corrected KSG Conditional Mutual Information Estimator: I(X;Y|Z)
        See https://arxiv.org/abs/1604.03006, http://proceedings.mlr.press/v84/runge18a.html

        x_data: data with shape (num_samples, x_dim) or (num_samples,)
        y_data: data with shape (num_samples, y_dim) or (num_samples,)
        z_data: conditioning data with shape (num_samples, z_dim) or (num_samples,)
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """
    xzy_data = np.concatenate((x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data,
                               z_data.reshape(-1, 1) if z_data.ndim == 1 else z_data,
                               y_data.reshape(-1, 1) if y_data.ndim == 1 else y_data), axis=1)

    lookup = NearestNeighbors(metric='chebyshev', algorithm='auto')
    lookup.fit(xzy_data)

    radius = lookup.kneighbors(n_neighbors=k, return_distance=True)[0]
    radius = np.nextafter(radius[:, -1], 0)

    x_dim = x_data.shape[1] if x_data.ndim > 1 else 1
    y_dim = y_data.shape[1] if y_data.ndim > 1 else 1
    z_dim = z_data.shape[1] if z_data.ndim > 1 else 1

    # compute entropies
    lookup.fit(xzy_data[:, :x_dim+z_dim])
    n_xz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy_data[:, x_dim:])
    n_yz = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    lookup.fit(xzy_data[:, x_dim:x_dim+z_dim])
    n_z = np.array([i.size for i in lookup.radius_neighbors(radius=radius, return_distance=False)])

    # correction term for using 2-norm
    correction = log_vol(x_dim+z_dim)+log_vol(y_dim+z_dim)-log_vol(x_dim+y_dim+z_dim)-log_vol(z_dim)

    return correction + digamma(k) + np.mean(np.log(n_z) - np.log(n_xz) - np.log(n_yz))


def log_vol(dim, p_norm=2.):
    """ log-volume of d-dimensional hypersphere, with respect to the p-norm (p<infty) """
    return dim*np.log(2*gamma(1.+1./p_norm)) - np.log(gamma(1.+dim*1./p_norm))
