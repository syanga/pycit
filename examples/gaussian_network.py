""" Generate gaussian data from a particular DAG """
import numpy as np


def gaussian_network(num_samples, dim, std=1.0):
    """ Generate gaussian data from a particular DAG """
    x_data = np.zeros((num_samples, dim, 6))

    # x1
    x_data[:, :, 0] = np.random.normal(0, 1.0, size=(num_samples, dim))

    # x1->x2
    x_data[:, :, 1] = x_data[:, :, 0] + np.random.normal(0, std, size=(num_samples, dim))

    # x1->x3
    x_data[:, :, 2] = x_data[:, :, 0] + np.random.normal(0, std, size=(num_samples, dim))

    # x3->x4
    x_data[:, :, 3] = x_data[:, :, 2] + np.random.normal(0, std, size=(num_samples, dim))

    # x2->x5<-x4
    x_data[:, :, 4] = 0.5*x_data[:, :, 1] \
        + 0.5*x_data[:, :, 3] + np.random.normal(0, std, size=(num_samples, dim))

    # x3->y<-x5
    y_data = 0.5*(x_data[:, :, 2] + x_data[:, :, 4]) + np.random.normal(0, std, size=(num_samples, dim))

    # y->x6<-x4
    x_data[:, :, 5] = 0.5*(x_data[:, :, 3] + y_data) + np.random.normal(0, std, size=(num_samples, dim))

    return x_data, y_data
