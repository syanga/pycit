""" Find Markov blanket for a particular Gaussian DAG """
import os
import numpy as np


def gaussian_network(num_samples, dim, std=1.0):
    """ Generate gaussian data from a particular DAG """
    x_data = np.zeros((num_samples, dim, 6))
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    gen_noise = lambda num_samples, dim, std: np.random.normal(0.0, std, size=(num_samples, dim))

    # x1
    x_data[:, :, 0] = gen_noise(num_samples, dim, 1.0)

    # x1->x2
    x_data[:, :, 1] = x_data[:, :, 0] + gen_noise(num_samples, dim, std)

    # x1->x3
    x_data[:, :, 2] = x_data[:, :, 0] + gen_noise(num_samples, dim, std)

    # x3->x4
    x_data[:, :, 3] = x_data[:, :, 2] + gen_noise(num_samples, dim, std)

    # x2->x5<-x4
    x_data[:, :, 4] = 0.5*(x_data[:, :, 1] + x_data[:, :, 3]) + gen_noise(num_samples, dim, std)

    # x3->y<-x5
    y_data = 0.5*(x_data[:, :, 2] + x_data[:, :, 4]) + gen_noise(num_samples, dim, std)

    # y->x6<-x4
    x_data[:, :, 5] = 0.5*(x_data[:, :, 3] + y_data) + gen_noise(num_samples, dim, std)

    return x_data, y_data


if __name__ == "__main__":
    # Find markov blanket of Y.
    #     * Ground truth: [X_3, X_4, X_5, X_6]
    #     * Adjacents: [X_3, X_5, X_6]
    #     * Coparents: [X_4]
    import sys
    sys.path.append("..")
    # pylint: disable=no-name-in-module
    # run this example from examples directory
    from pycit import MarkovBlanket, standardize

    # set settings
    N_SAMPLES = 8000
    CONFIDENCE_LEVEL = 0.95
    K_KNN = 5
    K_PERM = 10
    SUBSAMPLE_SIZE = None
    N_TRIALS = 200
    N_JOBS = 10

    cit_funcs = {
        'it_args': {
            'statistic_args': {
                'k': K_KNN
            },
            'test_args': {
                'statistic': 'mixed_mi',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        },
        'cit_args': {
            'statistic_args': {
                'k': K_KNN
            },
            'test_args': {
                'statistic': 'mixed_cmi',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        }
    }

    # generate data, then standardize to zero mean unit variance
    x, y = gaussian_network(N_SAMPLES, 1, std=1.0)
    x = standardize(x)
    y = standardize(y)

    # find Markov blanket
    mb = MarkovBlanket(x, y, cit_funcs)
    selected = mb.find_markov_blanket(confidence=CONFIDENCE_LEVEL, verbose=True)
