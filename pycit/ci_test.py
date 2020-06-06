""" Wrapper function for performing CI tests """
from . import estimators
from . testers import IndependenceTest, ConditionalIndependenceTest


def citest(x_data, y_data, z_data, statistic="mixed_cmi", statistic_args=None, test_args=None):
    """
        Test if x and y are conditionally independent given z using built-in methods

        x_data: shape (num_samples, x_dim)
        y_data: shape (num_samples, y_dim)
        z_data: shape (num_samples, z_dim)
        statistic: name of test statistic function: [ksg_cmi, mixed_cmi, bi_ksg_cmi]
        statistic_args: dictionary of additional args for statistic
            default:
            {
                'k': num_samples/20     # k for knn method
            }
        test_args: dictionary of additional args for cit
            default:
            {
                'k_perm': 5,            # permutation nearest neigbhors
                'n_trials': 1000,       # number of trials for estimating p-value
                'bootstrap_size': None, # not bootstrapped dataset
                'n_jobs': 1             # number of parallel processes for ci testing
            }
    """
    assert x_data.shape[0] == y_data.shape[0] == z_data.shape[0]
    num_samples = x_data.shape[0]

    if statistic_args is None:
        statistic_args = {'k': max(1, int(num_samples/20))}

    if test_args is None:
        test_args = {
            'k_perm': 5,
            'n_trials': 1000,
            'bootstrap_size': None,
            'n_jobs': 1
        }

    else:
        if 'k_perm' not in test_args:
            test_args['k_perm'] = 5
        if 'n_trials' not in test_args:
            test_args['n_trials'] = 1000
        if 'bootstrap_size' not in test_args:
            test_args['bootstrap_size'] = None
        if 'n_jobs' not in test_args:
            test_args['n_jobs'] = 1

    tester = ConditionalIndependenceTest(x_data, y_data, z_data, \
        getattr(estimators, statistic), statistic_args=statistic_args, k_perm=test_args['k_perm'])

    pval = tester.test(test_args['n_trials'], \
        bootstrap_size=test_args['bootstrap_size'], n_jobs=test_args['n_jobs'])

    return pval


def itest(x_data, y_data, statistic="mixed_mi", statistic_args=None, test_args=None):
    """
        Test if x and y are conditionally independent given z using built-in methods

        x_data: shape (num_samples, x_dim)
        y_data: shape (num_samples, y_dim)
        statistic: name of test statistic function: [ksg_cmi, mixed_cmi, bi_ksg_cmi]
        statistic_args: dictionary of additional args for statistic
            default:
            {
                'k': num_samples/20     # k for knn method
            }
        test_args: dictionary of additional args for cit
            default:
            {
                'n_trials': 1000,       # number of trials for estimating p-value
                'bootstrap_size': None, # not bootstrapped dataset
                'n_jobs': 1             # number of parallel processes for ci testing
            }
    """
    assert x_data.shape[0] == y_data.shape[0]
    num_samples = x_data.shape[0]

    if statistic_args is None:
        statistic_args = {'k': max(1, int(num_samples/20))}

    if test_args is None:
        test_args = {
            'n_trials': 1000,
            'bootstrap_size': None,
            'n_jobs': 1
        }

    else:
        if 'n_trials' not in test_args:
            test_args['n_trials'] = 1000
        if 'bootstrap_size' not in test_args:
            test_args['bootstrap_size'] = None
        if 'n_jobs' not in test_args:
            test_args['n_jobs'] = 1

    tester = IndependenceTest(x_data, y_data, \
        getattr(estimators, statistic), statistic_args=statistic_args)

    pval = tester.test(test_args['n_trials'], \
        bootstrap_size=test_args['bootstrap_size'], n_jobs=test_args['n_jobs'])

    return pval
