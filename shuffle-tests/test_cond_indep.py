import numpy as np
from sklearn.neighbors import NearestNeighbors


class CondIndepTest(HypothesisTest):
    """
        Shuffle based conditional independence test
        shuffles y by defualt.
    """
    def __init__(self, x, y, z, test, test_args={}, k_perm=5):
        super(CondIndepTest, self).__init__(test=test, test_args=test_args)

        assert x.shape[0] == y.shape[0] == z.shape[0]
        self.x = x
        self.y = y
        self.z = z
        self.batch_nominal = self.perform_test(x, y, z)

        # batch mode: compute lists of nn for each sample z
        self.k_perm = k_perm
        self.zNN = NearestNeighbors(metric='chebyshev')
        self.zNN.fit(z)
        self.neighbor_lists = self.zNN.kneighbors(n_neighbors=k_perm, return_distance=False)

    
    def batch_shuffle_test(self):
        # generate y_perm, a z-local permutation of y
        used_idx = []
        y_perm = np.zeros(self.y.shape)
        perm = np.random.permutation(self.y.shape[0])
        for i in perm:
            j = self.neighbor_lists[i][0]
            m = 0
            while j in used_idx and m < k_perm - 1:
                m += 1
                j = self.neighbor_lists[i][m]
            y_perm[i] = self.y[j]
            used_idx.append(j)

        test_result = self.perform_test(self.x, y_perm, self.z)
        self.log_test_result(test_result, self.batch_nominal, log_nominal=False)


    def resample_shuffle_test(self, resample_size):
        # resample data
        I = np.random.choice(self.y.shape[0], resample_size, replace=True)

        # create neighbor list for this run
        zNN = NearestNeighbors(metric='chebyshev')
        zNN.fit(z[I])
        neighbor_lists = zNN.kneighbors(n_neighbors=k_perm, return_distance=False)

        # generate y_perm
        used_idx = []
        y_perm = np.zeros(self.y[I].shape)
        perm = np.random.permutation(self.y[I].shape[0])
        for i in perm:
            j = neighbor_lists[i][0]
            m = 0
            while j in used_idx and m < k_perm - 1:
                m += 1
                j = neighbor_lists[i][m]
            y_perm[i] = self.y[I][j]
            used_idx.append(j)

        test_result = self.perform_test(self.x[I], y_perm, self.z[I])
        self.log_test_result(test_result, self.batch_nominal, log_nominal=True)
