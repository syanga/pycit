""" Generate bullseye data from a particular DAG """
import numpy as np


def bullseye_network(num_samples, dim, eps=0.075):
    """ Generate bullseye data from a particular DAG """
    assert eps in [0.025, 0.05, 0.075, 0.1, 0.125]

    # generate radius values for bullseye
    def sample_radius(num_samples):
        inner_idx = np.random.binomial(1, 0.5, size=(num_samples,))
        outer_idx = 1-inner_idx
        inner_samples = inner_idx*np.random.uniform(0.25, 0.5, size=(num_samples,))
        outer_samples = outer_idx*np.random.uniform(0.75, 1.0, size=(num_samples,))
        return inner_samples + outer_samples

    def sample_noise(num_samples, eps):
        return np.random.uniform(-eps, eps, size=(num_samples,))

    r_data = np.zeros((num_samples, 6))

    # r1
    r_data[:, 0] = sample_radius(num_samples)

    # r1->r2
    r_data[:, 1] = r_data[:, 0] + sample_noise(num_samples, eps)

    # r1->r3
    r_data[:, 2] = r_data[:, 0] + sample_noise(num_samples, eps)

    # r3->r4
    r_data[:, 3] = r_data[:, 2] + sample_noise(num_samples, eps)

    # r4->r5<-r2
    r_data[:, 4] = 0.5*r_data[:, 1] + 0.5*r_data[:, 3] + sample_noise(num_samples, eps)

    # r3->y<-r5
    y_data = 0.5*r_data[:, 2] + 0.5*r_data[:, 4] + sample_noise(num_samples, eps)

    # y->r6<-r4
    r_data[:, 5] = 0.5*r_data[:, 3] + 0.5*y_data + sample_noise(num_samples, 0.025)

    # generate X values from R values
    x_data = np.random.normal(0, 1, size=(num_samples, dim, 6))
    for i in range(6):
        radius = np.linalg.norm(x_data[:, :, i], axis=1)
        x_data[:, :, i] = (r_data[:, i]*x_data[:, :, i].T/radius).T

    return r_data.reshape((num_samples, 1, 6)), x_data, y_data.reshape(-1, 1)
