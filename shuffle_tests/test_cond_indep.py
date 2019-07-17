import numpy as np
from sklearn.neighbors import NearestNeighbors
from .test_base import HypothesisTest


class CondIndepTest(HypothesisTest):
    """
        Shuffle based conditional independence test
        shuffles y by defualt.
    """
    def __init__(self, x, y, z, test, test_args={}, k_perm=5):
        super().__init__(test=test, test_args=test_args)

        assert x.shape[0] == y.shape[0] == z.shape[0]
        self.x = x if x.ndim > 1 else x.reshape(-1, 1)
        self.y = y if y.ndim > 1 else y.reshape(-1, 1)
        self.z = z if z.ndim > 1 else z.reshape(-1, 1)
        self.batch_nominal = self.perform_test(self.x, self.y, self.z)

        # batch mode: compute lists of nn for each sample z
        self.k_perm = k_perm
        self.zNN = NearestNeighbors(metric='chebyshev')
        self.zNN.fit(self.z)
        self.neighbor_lists = self.zNN.kneighbors(n_neighbors=self.k_perm, return_distance=False)

    
    def batch_shuffle_test(self):
        # generate y_perm, a z-local permutation of y
        used_idx = []
        y_perm = np.zeros(self.y.shape)
        perm = np.random.permutation(self.y.shape[0])
        for i in perm:
            j = self.neighbor_lists[i][0]
            m = 0
            while j in used_idx and m < self.k_perm - 1:
                m += 1
                j = self.neighbor_lists[i][m]
            y_perm[i] = self.y[j]
            used_idx.append(j)

        test_result = self.perform_test(self.x, y_perm, self.z)
        return test_result


    def resample_shuffle_test(self, resample_size):
        # resample data
        I = np.random.choice(self.y.shape[0], resample_size)

        # create neighbor list for this run
        zNN = NearestNeighbors(metric='chebyshev')
        zNN.fit(self.z[I])
        neighbor_lists = zNN.kneighbors(n_neighbors=self.k_perm, return_distance=False)

        # generate y_perm
        used_idx = []
        y_perm = np.zeros(self.y[I].shape)
        perm = np.random.permutation(self.y[I].shape[0])
        for i in perm:
            j = neighbor_lists[i][0]
            m = 0
            while j in used_idx and m < self.k_perm - 1:
                m += 1
                j = neighbor_lists[i][m]
            y_perm[i] = self.y[I][j]
            used_idx.append(j)

        test_result = self.perform_test(self.x[I], y_perm, self.z[I])
        nominal = self.perform_test(self.x[I], self.y[I], self.z[I])
        return (test_result, nominal)
