import numpy as np
from multiprocessing import Pool
from functools import partial


def batch_run(_, hyp_test):
    """
    Queue a batch run on hyp_test object
    """
    return hyp_test.batch_shuffle_test()


def resample_run(_, hyp_test, resample_size):
    """
    Queue a resampled run on hyp_test object
    """
    return hyp_test.resample_shuffle_test(resample_size)


def batch_job(hyp_test, n_runs, n_jobs=1):
    """
    Create a multiprocessing pool for batch runs
    """
    pool = Pool(processes=n_jobs)
    test_results = list(pool.map_async(partial(batch_run, hyp_test=hyp_test), range(n_runs)).get())
    pool.close()
    pool.join()
    return hyp_test.batch_pvalue(test_results, hyp_test.batch_nominal), test_results


def resample_job(hyp_test, n_runs, n_resample, n_jobs=1):
    """
    Create a multiprocessing pool for resampled runs
    """
    pool = Pool(processes=n_jobs)
    tuple_results = list(pool.map_async(partial(resample_run, hyp_test=hyp_test, resample_size=n_resample), range(n_runs)).get())
    pool.close()
    pool.join()
    return hyp_test.resampled_pvalue(tuple_results), tuple_results
