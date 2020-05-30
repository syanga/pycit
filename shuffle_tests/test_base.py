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
        

    def perform_test(self, *data):
        """
        Call the test statistic provided, with variable arguments
        """
        return self.test(*data, **self.test_args)


    def batch_pvalue(self, test_results, nominal_value):
        """
        Compute p value for a batch run
        """
        assert len(test_results) > 0
        pval = 0
        for i,result in enumerate(test_results):
            if result >= nominal_value:
                pval += 1
        return pval / len(test_results)


    def resampled_pvalue(self, test_results):
        """
        Compute p value for a batch run
        """
        assert len(test_results) > 0
        pval = 0
        for i,result in enumerate(test_results):
            if result[0] >= result[1]:
                pval += 1
        return pval / len(test_results)


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
