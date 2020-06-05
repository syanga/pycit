""" k-NN differential entropy estimator for continuous data """
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors


def kl_entropy(x_data, k=5):
    """
        KL Differential Entropy Estimator
        See e.g. https://arxiv.org/abs/1603.08578

        x_data: samples of x, shape (num_samples, dim)
        k: number of nearest neighbors for estimation
    """
    x_data = x_data.reshape(-1, 1) if x_data.ndim == 1 else x_data
    num_samples, dim = x_data.shape

    # compute knn distances
    lookup = NearestNeighbors(metric='chebyshev', algorithm='auto')
    lookup.fit(x_data)

    # want diameter: twice radius
    diameters = 2*lookup.kneighbors(n_neighbors=k, return_distance=True)[0][:, k-1]

    return digamma(num_samples) - digamma(k) + dim*np.mean(np.log(diameters))
