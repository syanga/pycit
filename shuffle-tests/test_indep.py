import numpy as np
from .test_base import HypothesisTest


class IndepTest(HypothesisTest):
    """
        Shuffle-based independence test. 
        Shuffles y by default.
    """
    def __init__(self, x, y, test, test_args={}):
        super(IndepTest, self).__init__(test=test, test_args=test_args)

        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        self.batch_nominal = self.perform_test(x, y)


    def batch_shuffle_test(self):
        perm = np.random.permutation(self.y.shape[0])
        test_result = self.perform_test(self.x, self.y[perm])
        self.log_test_result(test_result, self.batch_nominal, log_nominal=False)


    def resample_shuffle_test(self, resample_size):
        # resample data
        I = np.random.choice(self.y.shape[0], resample_size, replace=True)
        
        # perform test
        perm = np.random.permutation(self.y[I].shape[0])
        nominal = self.perform_test(self.x[I], self.y[I])
        test_result = self.perform_test(self.x[I], self.y[I][perm])
        self.log_test_result(test_result, nominal, log_nominal=True)
