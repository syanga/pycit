import numpy as np
from .test_base import HypothesisTest


class IndepTest(HypothesisTest):
    """
        Shuffle-based independence test. 
        Shuffles y by default.
    """
    def __init__(self, x, y, test, test_args={}):
        super().__init__(test=test, test_args=test_args)

        assert x.shape[0] == y.shape[0]
        self.x = x if x.ndim > 1 else x.reshape(-1, 1)
        self.y = y if y.ndim > 1 else y.reshape(-1, 1)
        self.batch_nominal = self.perform_test(self.x, self.y)


    def batch_shuffle_test(self):
        perm = np.random.permutation(self.y.shape[0])
        test_result = self.perform_test(self.x, self.y[perm])
        return test_result


    def resample_shuffle_test(self, resample_size):
        # resample data
        I = np.random.choice(self.y.shape[0], resample_size)
        
        # perform test
        perm = np.random.permutation(self.y[I].shape[0])
        test_result = self.perform_test(self.x[I], self.y[I][perm])
        nominal = self.perform_test(self.x[I], self.y[I])
        return (test_result, nominal)
