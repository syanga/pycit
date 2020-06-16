""" Wrapper function for performing CI tests """
from . import estimators
from . testers import IndependenceTest, ConditionalIndependenceTest


def citest(x_data, y_data, z_data, statistic="mixed_cmi", statistic_args=None, test_args=None):
    """
        Test if x and y are conditionally independent given z using built-in methods
        * H0: x and y are conditionally independent given z
        * H1: x and y are conditionally dependent given z

        x_data: shape (num_samples, x_dim)
        y_data: shape (num_samples, y_dim)
        z_data: shape (num_samples, z_dim)
        statistic: name of test statistic function: [ksg_cmi, mixed_cmi, bi_ksg_cmi]
        statistic_args: dictionary of additional args for statistic
            default:
            {
                'k': 5                  # k for knn method
            }
        test_args: dictionary of additional args for cit
            default:
            {
                'k_perm': 10,           # permutation nearest neigbhors
                'n_trials': 1000,       # number of trials for estimating p-value
                'subsample_size': None, # not subsampleped dataset
                'n_jobs': 1             # number of parallel processes for ci testing
            }
    """
    # pylint: disable=too-many-arguments
    assert x_data.shape[0] == y_data.shape[0] == z_data.shape[0]
    # num_samples = x_data.shape[0]

    if statistic_args is None:
        statistic_args = {'k': 5}

    default_test_args = {
        'k_perm': 10,
        'n_trials': 1000,
        'subsample_size': None,
        'n_jobs': 1
    }

    if test_args is None:
        test_args = default_test_args

    else:
        # populate missing fields with defaults
        for key in default_test_args:
            if key not in test_args:
                test_args[key] = default_test_args[key]

    tester = ConditionalIndependenceTest(x_data, y_data, z_data, \
        getattr(estimators, statistic), statistic_args=statistic_args, k_perm=test_args['k_perm'])

    # initialize z nearest neighbor search preemptively to avoid multiprocessing issues
    if default_test_args['subsample_size'] is None:
        tester.initialize_batch()

    pval = tester.test(test_args['n_trials'], \
        subsample_size=test_args['subsample_size'], n_jobs=test_args['n_jobs'])

    del tester
    return pval


def itest(x_data, y_data, statistic="mixed_mi", statistic_args=None, test_args=None):
    """
        Test if x and y are conditionally independent given z using built-in methods
        * H0: x and y are independent
        * H1: x and y are dependent

        x_data: shape (num_samples, x_dim)
        y_data: shape (num_samples, y_dim)
        statistic: name of test statistic function: [ksg_cmi, mixed_cmi, bi_ksg_cmi]
        statistic_args: dictionary of additional args for statistic
            default:
            {
                'k': 5                  # k for knn method
            }
        test_args: dictionary of additional args for cit
            default:
            {
                'n_trials': 1000,       # number of trials for estimating p-value
                'subsample_size': None, # not subsampleped dataset
                'n_jobs': 1             # number of parallel processes for ci testing
            }
    """
    assert x_data.shape[0] == y_data.shape[0]
    # num_samples = x_data.shape[0]

    if statistic_args is None:
        statistic_args = {'k': 5}

    default_test_args = {
        'n_trials': 1000,
        'subsample_size': None,
        'n_jobs': 1
    }

    if test_args is None:
        test_args = default_test_args

    else:
        # populate missing fields with defaults
        for key in default_test_args:
            if key not in test_args:
                test_args[key] = default_test_args[key]

    tester = IndependenceTest(x_data, y_data, \
        getattr(estimators, statistic), statistic_args=statistic_args)

    pval = tester.test(test_args['n_trials'], \
        subsample_size=test_args['subsample_size'], n_jobs=test_args['n_jobs'])

    del tester
    return pval
