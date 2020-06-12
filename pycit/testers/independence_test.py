""" Independence Testing """
import numpy as np
from .base_hypothesis_test import HypothesisTest


class IndependenceTest(HypothesisTest):
    """
        Shuffle-based independence test.

        H0: x and y are independent
        H1: x and y are dependent.

        * x_data: data with shape (num_samples, x_dim) or (num_samples,)
        * y_data: data with shape (num_samples, y_dim) or (num_samples,)
        * statistic: function that computes scalar test statistic.
            Is a function of x_data, y_data
        * statistic_args: dictionary of additional args for statistic()

        Shuffles samples of y when estimating distribution
        of test statistic under H0.
    """
    def __init__(self, x_data, y_data, statistic, statistic_args=None):
        super().__init__(statistic, statistic_args=statistic_args)
        assert x_data.shape[0] == y_data.shape[0]
        self.total_samples = x_data.shape[0]

        # flatten extra dimensions of x and y
        self.x_data = x_data.reshape(x_data.shape[0], -1)
        self.y_data = y_data.reshape(y_data.shape[0], -1)

    def subsample_instance(self, subsample_size=None, shuffle=True):
        """ Compute test statistic using shuffled data
            returns tuple: (shuffled_statistic, nominal_statistic)
                * shuffled_statistic: computed using shuffled subsampleped dataset
                * nominal_statistic: computed using unshuffled subsampleped dataset

            * subsample_size: if None, return None for nominal_statistic.
                Otherwise, statistics using subsample resample of dataset
            * shuffle: if True, perform shuffling for p-value estimation.
                Set to False to compute batch nominal statistic.
                In this case, the returned shuffle_stat will be None
        """
        if subsample_size is not None:
            # use subsample subsample of dataset
            idx1 = np.random.choice(self.total_samples, subsample_size, replace=False)

            if shuffle:
                # shuffle data for p-value estimation
                idx2 = np.random.choice(idx1, subsample_size, replace=False)
                nominal_stat = self.compute_statistic(self.x_data[idx1], self.y_data[idx1])
                shuffle_stat = self.compute_statistic(self.x_data[idx1], self.y_data[idx2])
            else:
                # no shuffling, subsample subsample
                nominal_stat = self.compute_statistic(self.x_data[idx1], self.y_data[idx1])
                shuffle_stat = None

        elif shuffle:
            # compute shuffled statistic using full dataset
            idx = np.random.permutation(self.total_samples)
            nominal_stat = None
            shuffle_stat = self.compute_statistic(self.x_data, self.y_data[idx])

        else:
            # No subsampleping, no shuffling. Only compute nominal test statistic
            nominal_stat = self.compute_statistic(self.x_data, self.y_data)
            shuffle_stat = None

        return (shuffle_stat, nominal_stat)
