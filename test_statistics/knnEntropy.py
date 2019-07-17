import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors



def KLEntropy(x, k=5, algorithm='auto', leaf_size=30, eps=1e-12):
    """
        KL Differential Entropy Estimator, 
        with low amplitude noise added to break ties.

        See e.g. https://arxiv.org/abs/1603.08578
    """
    x = x.reshape(-1,1) if x.ndim==1 else x
    x += np.random.normal(0, 1, size=x.shape) * eps
    n,d = x.shape

    # compute knn distances
    params = {'algorithm': algorithm, 'leaf_size':leaf_size, 'metric':'chebyshev', 'n_jobs':1}
    lookup = NearestNeighbors(**params)
    lookup.fit(x)

    # want diameter: twice radius
    diameters = 2*lookup.kneighbors(n_neighbors=k, return_distance=True)[0][:,k-1]

    return digamma(n) - digamma(k) + d*np.mean(np.log(diameters))
