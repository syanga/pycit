""" Perform CI testing for data generated from different Gaussian channel structures """
import numpy as np

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    # pylint: disable=no-name-in-module
    # run this example from examples directory
    from pycit import citest, itest

    # Two Independent Variables
    x = np.random.normal(0, 1, size=1000)
    y = np.random.normal(0, 1, size=1000)

    print("Example: Independent variables")
    pval = itest(x, y, test_args={'n_jobs': 2})
    print("  Declared independent: %r, p-value: %0.3f" % ((pval > 0.05), pval))

    # Two Dependent Variables
    x = np.random.normal(0, 1, size=1000)
    y = x + np.random.normal(0, 1, size=1000)

    print("Example: Dependent variables")
    pval = itest(x, y, test_args={'n_jobs': 2})
    print("  Declared independent: %r, p-value: %0.3f" % ((pval > 0.05), pval))

    # Two Conditionally Independent Variables
    # x -> z -> y
    x = np.random.normal(0, 1, size=1000)
    z = x + np.random.normal(0, 1, size=1000)
    y = z + np.random.normal(0, 1, size=1000)

    print("Example: Conditionally independent variables")
    pval = citest(x, y, z, test_args={'n_jobs': 2})
    print("  Declared independent: %r, p-value: %0.3f" % ((pval > 0.05), pval))

    # Two Conditionally Dependent Variables
    x = np.random.normal(0, 1, size=1000)
    z = x + np.random.normal(0, 1, size=1000)
    y = 0.5*z +0.5*x + np.random.normal(0, 1, size=1000)

    print("Example: Conditionally dependent variables")
    pval = citest(x, y, z, test_args={'n_jobs': 2})
    print("  Declared independent: %r, p-value: %0.3f" % ((pval > 0.05), pval))
