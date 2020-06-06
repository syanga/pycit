""" Markov Blanket feature selection """
from itertools import combinations
import numpy as np


class MarkovBlanket:
    """
        Base class for Markov Blanket feature selection
        See e.g. https://arxiv.org/abs/1911.04628

        y_data: target variable. shape (num_samples, y_dim)
        x_data: predictor variables. shape (num_samples, x_dim, num_features)
        x_labels: names of each feature, length num_features.
                  If None, defaults to 'X_{1}',...,'X_{m}'
        cit_funcs: dictionary requiring the following keys:
        {
            'it': independence test function that takes data and returns a pvalue,
            'it_args': dictionary of additional arguments for it()
            'cit': conditional independence test function that takes data and returns a pvalue,
            'cit_args': dictionary of additional arguments for cit()
        }
    """
    def __init__(self, x_data, y_data, cit_funcs, x_labels=None):
        assert x_data.shape[0] == y_data.shape[0]
        assert x_data.ndim == 3
        self.num_samples = x_data.shape[0]
        self.num_features = x_data.shape[2]

        if x_labels is not None:
            assert len(x_labels) == self.num_features
            self.x_labels = x_labels
        else:
            self.x_labels = ['X_{%d}'%i for i in range(self.num_features)]

        self.x_data = x_data
        self.y_data = y_data

        assert 'it' in cit_funcs
        assert 'it_args' in cit_funcs
        assert 'cit' in cit_funcs
        assert 'cit_args' in cit_funcs
        self.cit_funcs = cit_funcs

    def test_feature(self, feature, conditioning_set):
        """
            Test if Y is CI of feature given conditioning_set. returns test p-value

            * feature: index of feature in x_data
            * conditioning_set: sorted list of feature indices
        """
        if len(conditioning_set) == 0:
            # independence test
            pval = self.cit_funcs['it'](self.x_data[:, :, feature],
                                        self.y_data,
                                        **self.cit_funcs['it_args'])
        else:
            # conditional independence test
            pval = self.cit_funcs['cit'](self.x_data[:, :, feature],
                                         self.y_data,
                                         self.x_data[:, :, conditioning_set].reshape(-1, 1),
                                         **self.cit_funcs['cit_args'])

        return pval

    def find_adjacents(self, confidence=0.95, max_conditioning=None, verbose=False):
        """
            Find parents and children of Y
        """
        if max_conditioning is None:
            # default to largest possible conditioning set size
            max_conditioning = self.num_features-1

        # iteratively rule out adjacent features
        adjacents = np.random.permutation(self.num_features).tolist()

        # increase conditioning set size from 0
        for conditioning_size in range(max_conditioning+1):
            # loop through each not-yet eliminated feature
            adj_idx = 0
            while adj_idx < len(adjacents):
                # identify feature being tested and possible conditioning features
                curr_feature = adjacents[adj_idx]
                conditioning_candidates = [j for j in adjacents if j != adj_idx]

                if verbose:
                    print("==========Testing %s=========="%self.x_labels[curr_feature])

                # try all possible conditioning sets of size conditioning_size
                for conditioning_set in combinations(conditioning_candidates, conditioning_size):
                    if verbose:
                        print("\tCond. set:%s"%str([self.x_labels[k] for k in conditioning_set]))

                    # conditioning_set is a tuple, sorted(conditioning_set) is a list
                    pval = self.test_feature(curr_feature, sorted(conditioning_set))

                    if verbose:
                        print("\t\t Is CI:%r, pval:%0.3f"%(pval >= 1.- confidence, pval))

                    if pval >= 1.- confidence:
                        # remove feature if it is CI of Y
                        adjacents.remove(curr_feature)
                        break

                if curr_feature in adjacents:
                    # increment if feature adjacents[adj_idx] was not eliminated
                    adj_idx += 1

        return adjacents

    def find_coparents(self, adjacents, confidence=0.95, verbose=False):
        """
            Find co-parents of Y, given adjacents

            If feature i is not CI given adjacents, add to list of coparents
            The Markov blanket is the union of adjacents and coparents
        """
        markov_blanket = adjacents.copy()
        coparents = []
        for i in range(self.num_features):
            if i in adjacents:
                continue

            markov_blanket = sorted(markov_blanket)
            pval = self.test_feature(i, markov_blanket)
            is_dependent = bool(pval < 1.-confidence)

            if verbose:
                result = '!CI' if is_dependent else 'CI'
                cond_set_str = str([self.x_labels[k] for k in markov_blanket])
                print("Y%s%s|%s,p-val:%0.2f"%(self.x_labels[i], result, cond_set_str, pval))

            if is_dependent:
                markov_blanket.append(i)
                coparents.append(i)

        return sorted(coparents)
