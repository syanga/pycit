""" Conditional Independence Testing """
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base_hypothesis_test import HypothesisTest


class ConditionalIndependenceTest(HypothesisTest):
    """
        Shuffle-based independence test.

        H0: x and y are conditionally independent given z
        H1: x and y are conditionally dependent given z

        * x_data: data with shape (num_samples, x_dim) or (num_samples,)
        * y_data: data with shape (num_samples, y_dim) or (num_samples,)
        * z_data: data with shape (num_samples, z_dim) or (num_samples,)
        * statistic: function that computes scalar test statistic.
            Is a function of x_data, y_data, z_data
        * statistic_args: dictionary of additional args for statistic()
        * k_perm: number of nearest neighbors for nearest-neighbor permutation

        Shuffles samples of y based on nearest-neighbor z values to estimate
        test statistic under H0,
        Based on: http://proceedings.mlr.press/v84/runge18a.html
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, x_data, y_data, z_data, statistic, statistic_args=None, k_perm=5):
        # pylint: disable=too-many-arguments
        super().__init__(statistic, statistic_args=statistic_args)
        assert x_data.shape[0] == y_data.shape[0] == z_data.shape[0]
        self.total_samples = x_data.shape[0]

        # flatten extra dimensions of x and y
        self.x_data = x_data.reshape(x_data.shape[0], -1)
        self.y_data = y_data.reshape(y_data.shape[0], -1)
        self.z_data = z_data.reshape(z_data.shape[0], -1)

        # setup nearest-neighbor search for z (max-norm) initially for batch mode
        self.k_perm = k_perm
        self.lookup_z = NearestNeighbors(metric='chebyshev')

        # store nearest neighbor lists for efficient batch mode
        self.nn_lists = None
        self.batch_initialized = False

    def _initialize_batch(self):
        """
            Compute 2d array of nearest indices for each point in dataset
            Run this once before batch runs
        """
        self.lookup_z.fit(self.z_data)
        self.nn_lists = self.lookup_z.kneighbors(n_neighbors=self.k_perm, return_distance=False)
        self.batch_initialized = True

    def _batch_permute(self):
        """
            Generate batch permutation of y_data by locally shuffling
            with k_perm nearest-neighbors of z-samples. Uses precomputed nearest neighbors
        """
        if not self.batch_initialized:
            # initialize nearest neighbor lists
            self._initialize_batch()

        used_idx = []
        perm = np.random.permutation(self.total_samples)
        idx = np.zeros(self.total_samples, dtype='i')
        for i in perm:
            temp = i
            for j in range(self.k_perm):
                temp = self.nn_lists[i][j]
                if temp not in used_idx:
                    break
            idx[i] = temp
            used_idx.append(temp)

        return idx

    def _bootstrap_permute(self, bootstrap_size):
        """
            Generate bootstrap permutation of y_data by locally shuffling
            with k_perm nearest-neighbors of z-samples.
        """
        # bootstrap subsample of dataset
        idx1 = np.random.choice(self.total_samples, bootstrap_size, replace=True)

        # create nearest neighbor lists for this instance
        self.lookup_z.fit(self.z_data[idx1])
        nn_lists = self.lookup_z.kneighbors(n_neighbors=self.k_perm, return_distance=False)

        # generate y permutation
        used_idx = []
        perm = np.random.permutation(bootstrap_size)
        idx2 = np.zeros(bootstrap_size, dtype='i')
        for i in perm:
            temp = i
            for j in range(self.k_perm):
                temp = nn_lists[i][j]
                if temp not in used_idx:
                    break

            # index into idx1
            temp = idx1[temp]
            idx2[i] = temp
            used_idx.append(temp)

        # return bootstrap indices, shuffled bootstrap indices
        return idx1, idx2

    def bootstrap_instance(self, bootstrap_size=None, shuffle=True):
        """ Compute test statistic using shuffled data
            returns tuple: (shuffled_statistic, nominal_statistic)
                * shuffled_statistic: computed using shuffled bootstrapped dataset
                * nominal_statistic: computed using unshuffled bootstrapped dataset

            * bootstrap_size: if None, return None for nominal_statistic.
                Otherwise, statistics using bootstrap resample of dataset
            * shuffle: if True, perform shuffling for p-value estimation.
                Set to False to compute batch nominal statistic.
                In this case, the returned shuffle_stat will be None
        """
        if bootstrap_size is not None:
            # use bootstrap subsample of dataset
            idx1, idx2 = self._bootstrap_permute(bootstrap_size)

            if shuffle:
                # shuffle data for p-value estimation
                nominal_stat = self.compute_statistic(self.x_data[idx1],
                                                      self.y_data[idx1],
                                                      self.z_data[idx1])
                shuffle_stat = self.compute_statistic(self.x_data[idx1],
                                                      self.y_data[idx2],
                                                      self.z_data[idx1])
            else:
                # no shuffling, bootstrap subsample
                nominal_stat = self.compute_statistic(self.x_data[idx1],
                                                      self.y_data[idx1],
                                                      self.z_data[idx1])
                shuffle_stat = None

        elif shuffle:
            # batch mode: compute shuffled statistic using full dataset
            idx = self._batch_permute()
            nominal_stat = None
            shuffle_stat = self.compute_statistic(self.x_data, self.y_data[idx], self.z_data)

        else:
            # No bootstrapping, no shuffling. Only compute nominal test statistic
            nominal_stat = self.compute_statistic(self.x_data, self.y_data, self.z_data)
            shuffle_stat = None

        return (shuffle_stat, nominal_stat)
