""" Preprocess data before CI testing """
import numpy as np


def normalize(data):
    """
        Normalize data for each feature to the range of [0,1]
        data has shape: (n,), (n,d), or (n,d,m). If shape is (n,d,m),
        normalize separately for each of the m last dimensions
    """
    mini = np.min(data, axis=0)
    maxi = np.max(data, axis=0)
    den = maxi-mini
    if data.ndim > 1:
        den[den == 0] = 1.
    elif den == 0:
        den = 1.

    return (data-mini)/den


def standardize(data):
    """
        Standardize data for each feature to zero mean unit variance
        data has shape: (n,), (n,d), or (n,d,m). If shape is (n,d,m),
        standardize separately for each of the m last dimensions
    """
    mean = np.mean(data, axis=0)
    stdv = np.std(data, axis=0)
    if data.ndim > 1:
        stdv[stdv == 0] = 1.
    elif stdv == 0:
        stdv = 1.

    return (data-mean)/stdv


def low_amplitude_noise(data, eps=1e-10):
    """
        Add low amplitude noise in order to break ties
        Necessary if have discrete data but using an
        estimator for continuous data
    """
    return data + np.random.normal(0, eps, size=data.shape)
