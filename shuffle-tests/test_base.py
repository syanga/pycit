import numpy as np


class HypothesisTest:
    """
    Base class for hypothesis testing.
    Tracks test statistics, computes p-values.
    Supports resampling data at each run.
    """
    def __init__(self, test, test_args={}):
        self.test = test
        self.test_args = test_args

        self.pval_counter = 0
        self.n_runs = 0

        self.batch_nominal = None           # overwrite in child
        self.log_tests = []                 # save test results
        self.log_nominal = []               # for resampling mode


    def perform_test(*data):
        """
        Call the test statistic provided, with variable arguments
        """
        return self.test(*data, **self.test_args)


    def log_test_result(self, test_result, nominal_value, log_nominal=False):        
        """
        Log a test result, as well as a nominal test value
        if applicable.
        """
        if log_nominal:
            # resampling mode
            self.log_nominal.append(nominal_value)

        if test_result >= nominal_value:
            self.pval_counter += 1

        self.n_bootstraps += 1
        self.log_tests.append(test_result)


    def get_pvalue(self):
        """
        Compute and return p value, based on runs so far
        """
        assert self.n_bootstraps > 0
        return self.pval_counter / self.n_bootstraps


    def get_logs(self):
        """
        Get logged values
        """
        return (self.log_tests, self.batch_nominal, self.log_nominal)


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
