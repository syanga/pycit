""" Markov Blanket feature selection """
from itertools import combinations
import numpy as np
from .ci_test import itest, citest


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
            'it': independence test function that takes data and returns a pvalue
                # defaults to itest()
            'it_args': dictionary of additional arguments for it()
            'cit': conditional independence test function that takes data and returns a pvalue,
                # defaults to citest()
            'cit_args': dictionary of additional arguments for cit()
        }
    """
    def __init__(self, x_data, y_data, cit_funcs=None, x_labels=None):
        assert x_data.shape[0] == y_data.shape[0]
        assert x_data.ndim == 3
        self.num_samples = x_data.shape[0]
        self.num_features = x_data.shape[2]

        if x_labels is not None:
            assert len(x_labels) == self.num_features
            self.x_labels = x_labels
        else:
            self.x_labels = ['X_{%d}'%(i+1) for i in range(self.num_features)]

        self.x_data = x_data
        self.y_data = y_data

        default_cit_funcs = {
            'it': itest,
            'it_args': {},
            'cit': citest,
            'cit_args': {}
        }

        if cit_funcs is None:
            self.cit_funcs = default_cit_funcs

        else:
            # populate missing fields with defaults
            self.cit_funcs = cit_funcs
            for key in default_cit_funcs:
                if key not in self.cit_funcs:
                    self.cit_funcs[key] = default_cit_funcs[key]

    def find_markov_blanket(self, min_conditioning=0, max_conditioning=None,
                            confidence=0.95, verbose=False):
        """
            Finds the adjacents, then adds coparents
            * max_conditioning: maximum conditioning set size
            * confidence level for CI testing
            * verbose: print results of CI tests
        """
        if verbose:
            print("==========Finding Adjacents...==========")

        adjacents = self.find_adjacents(min_conditioning, max_conditioning, confidence, verbose)

        if verbose:
            print("Adjacents found: %s"%str([self.x_labels[k] for k in adjacents]))
            print("==========Finding Coparents...==========")

        coparents = self.find_coparents(adjacents, confidence, verbose)
        markov_blanket = sorted(adjacents+coparents)

        if verbose:
            print("Discovered Markov blanket: %s"%str([self.x_labels[k] for k in markov_blanket]))

        return markov_blanket

    def test_feature(self, feature, conditioning_set):
        """
            Test if Y is CI of feature given conditioning_set. returns test p-value

            * feature: index of feature in x_data
            * conditioning_set: sorted list of feature indices
        """
        if len(conditioning_set) == 0:
            # independence test
            pval = self.cit_funcs['it'](
                self.x_data[:, :, feature],
                self.y_data,
                **self.cit_funcs['it_args'])
        else:
            # conditional independence test
            pval = self.cit_funcs['cit'](
                self.x_data[:, :, feature],
                self.y_data,
                self.x_data[:, :, conditioning_set].reshape(self.num_samples, -1),
                **self.cit_funcs['cit_args'])

        return pval

    def find_adjacents(self, min_conditioning=0, max_conditioning=None,
                       confidence=0.95, verbose=False):
        """
            Find parents and children of target variable Y
            * max_conditioning: maximum conditioning set size
            * confidence level for CI testing
            * verbose: print results of CI tests
        """
        if max_conditioning is None:
            # default to largest possible conditioning set size
            max_conditioning = self.num_features-1

        # iteratively rule out adjacent features
        adjacents = np.random.permutation(self.num_features).tolist()

        # increase conditioning set size from 0
        for conditioning_size in range(min_conditioning, max_conditioning+1):
            # loop through each not-yet eliminated feature
            adj_idx = 0
            while adj_idx < len(adjacents):
                # identify feature being tested and possible conditioning features
                curr_feature = adjacents[adj_idx]
                conditioning_candidates = [j for j in adjacents if j != curr_feature]

                if verbose:
                    print("Testing %s"%self.x_labels[curr_feature])

                # try all possible conditioning sets of size conditioning_size
                sets = [sorted(c) for c in combinations(conditioning_candidates, conditioning_size)]
                np.random.shuffle(sets)
                for conditioning_set in sets:
                    if verbose:
                        print("  Cond. set: %s"%str([self.x_labels[k] for k in conditioning_set]))

                    # conditioning_set is a tuple, sorted(conditioning_set) is a list
                    pval = self.test_feature(curr_feature, conditioning_set)

                    if verbose:
                        print("    Is CI: %r, pval: %0.3f"%(pval >= 1.- confidence, pval))

                    if pval >= 1.- confidence:
                        # remove feature if it is CI of Y
                        adjacents.remove(curr_feature)
                        break

                if curr_feature in adjacents:
                    # increment if feature adjacents[adj_idx] was not eliminated
                    adj_idx += 1

        return sorted(adjacents)

    def find_coparents(self, adjacents, confidence=0.95, verbose=False):
        """
            Find co-parents of Y, given adjacents. If feature i is not
            CI given adjacents, add to list of coparents
            * adjacents: list of adjacent features
            * confidence level for CI testing
            * verbose: print results of CI tests
        """
        markov_blanket = adjacents.copy()
        coparents = []
        non_adjacents = [j for j in range(self.num_features) if j not in adjacents]
        np.random.shuffle(non_adjacents)
        for i in non_adjacents:
            markov_blanket = sorted(markov_blanket)

            if verbose:
                print("Testing %s"%self.x_labels[i])

            pval = self.test_feature(i, markov_blanket)
            is_dependent = bool(pval < 1.-confidence)

            if verbose:
                print("    Is dependent: %r, pval: %0.3f"%(is_dependent, pval))

            if is_dependent:
                markov_blanket.append(i)
                coparents.append(i)

        return sorted(coparents)
