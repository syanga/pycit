""" Generate gaussian data from a particular DAG """
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
