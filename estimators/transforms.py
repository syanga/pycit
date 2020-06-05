import numpy as np

def standardize(mat):
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    return (mat - mean)/std


def low_amplitude_noise():
    # xy += 1e-10 * np.maximum(1.0, np.mean(np.abs(xy))) * np.random.normal(0,1, xy.shape)
    pass
