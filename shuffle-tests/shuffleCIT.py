import numpy as np
from sklearn.neighbors import NearestNeighbors


class HypothesisTest:
    """
    Base class for hypothesis testing.
    Tracks test statistics, computes p-values.
    """
    def __init__(self, batch_nominal=None):
        self.pval_counter = 0
        self.n_runs = 0

        self.log_tests = []
        self.log_nominal = []               # resampling mode
        self.batch_nominal = batch_nominal  # for batch mode


    def batch_shuffle_test(self):
        """
        Batch mode shuffle test
        """
        raise NotImplementedError


    def resample_shuffle_test(self, resample_size):
        """
        Resample data, then perform shuffle test
        """
        raise NotImplementedError


    def log_test_result(self, test_result, nominal_value=None):        
        if nominal_value is not None:
            # resampling mode
            self.log_nominal.append(nominal_value)
            if test_result >= nominal_value:
                self.pval_counter += 1
        else:
            # batch mode
            if test_result >= self.batch_nominal:
                self.pval_counter += 1

        self.n_bootstraps += 1
        self.log_tests.append(test_result)


    def get_pvalue(self):
        """
        Compute and return p value, based on runs so far
        """
        assert self.n_bootstraps > 0
        return self.pval_counter / self.n_bootstraps

    # super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)




class IndepTest:
    """
        Shuffle-based independence test. 
        Shuffles y by default.
    """
    def __init__(self, x, y, test, test_args={}):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        self.test = test
        self.test_args = test_args

        self.pval_counter = 0
        self.n_bootstraps = 0

        # log computed values
        self.nominal = test(x, y, **test_args)
        self.log_tests = []
        self.log_nominal = []


    def batch_shuffle_test(self):
        perm = np.random.permutation(self.y.shape[0])
        test_result = self.test(self.x, self.y[perm], **self.test_args)
        
        if test_result >= self.nominal:
            self.pval_counter += 1
        
        self.n_bootstraps += 1
        self.log_tests.append(test_result)


    def resample_shuffle_test(self, resample_size):
        # resample data, with replacement
        I = np.random.choice(self.y.shape[0], resample_size, replace=True)
        perm = np.random.permutation(self.y[I].shape[0])

        nominal = self.test(self.x[I], self.y[I], **self.test_args)
        test_result = self.test(self.x[I], self.y[I][perm], **self.test_args)

        if test_result >= nominal:
            self.pval_counter += 1
        
        self.n_bootstraps += 1
        self.log_tests.append(test_result)
        self.log_nominal.append(nominal)


    def get_pvalue(self):
        assert self.n_bootstraps > 0
        return self.pval_counter / self.n_bootstraps



class CondIndepTest:
    """
        Shuffle based conditional independence test
        shuffles y by defualt.
    """
    def __init__(self, x, y, z, test, test_args={}, k_perm=5):
        assert x.shape[0] == y.shape[0] == z.shape[0]
        self.x = x
        self.y = y
        self.z = z
        self.test = test
        self.test_args = test_args

        self.pval_counter = 0
        self.n_bootstraps = 0

        # log computed values
        self.nominal = test(x, y, z, **test_args)
        self.log_tests = []
        self.log_nominal = []

        # batch mode: compute lists of nn for each sample z
        self.zNN = NearestNeighbors(metric='chebyshev')
        self.zNN.fit(z)
        self.neighbor_lists = self.zNN.kneighbors(n_neighbors=k_perm, return_distance=False)
        self.k_perm = k_perm


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

        test_result = self.test(self.x, y_perm, self.z, **self.test_args)
        if test_result >= self.nominal:
            self.p_val += 1


    def get_pvalue(self):
        assert self.n_bootstraps > 0
        return self.pval_counter / self.n_bootstraps




def run_bootstrap(test_object):
    return test_object.batch_shuffle_test()










def cindep_test(x, y, z, k_cmi, k_perm, n_bootstrap):
    cmi_nominal = noiseCMI(x, y, z, k=k_cmi)

    # compute lists of nn for each sample z
    zNN = NearestNeighbors(metric='chebyshev')
    zNN.fit(z)
    neighbor_lists = zNN.kneighbors(n_neighbors=k_perm, return_distance=False)

    p_val = 0
    for b in range(n_bootstrap):
        used_idx = []
        y_perm = np.zeros(y.shape)
        perm = np.random.permutation(y.shape[0])
        for i in perm:
            j = neighbor_lists[i][0]
            m = 0
            while j in used_idx and m < k_perm - 1:
                m += 1
                j = neighbor_lists[i][m]
            y_perm[i] = y[j]
            used_idx.append(j)
        nw = noiseCMI(x, y_perm, z, k=k_cmi)
        if nw >= cmi_nominal:
            p_val += 1
        # print (nw, cmi_nominal)
    return p_val/n_bootstrap
