import numpy as np
import pickle
from tqdm import tqdm
from itertools import combinations

from test_statistics.knnMI import mixedMI
from test_statistics.knnCMI import mixedCMI
from shuffle_tests import CondIndepTest, IndepTest
from shuffle_tests.test_parallel import *

from sdcit.utils import rbf_kernel_median, K2D
from sdcit.hsic import HSIC, c_HSIC
from sdcit.kcipt import c_KCIPT

from CCIT import CCIT


class MarkovBlanketBase:
    """
        Y: (N,1) or (N,) shape
        X: (N,dim,features)
        max_cond: maximum conditioning set size
    """
    def __init__(self, X, Y, names=None, confidence=0.95):
        self.X = X
        self.Y = Y

        self.n_vars = X.shape[2]
        self.names = names
        if names is not None:
            assert len(names) == self.n_vars

        self.confidence = confidence
        self.selected = list(range(self.n_vars))


    def ci_test(self, i, cond_set, verbose=False):
        raise NotImplementedError


    def test_var(self, i, size, verbose=False):
        candidates = [j for j in self.selected]
        candidates.remove(i)
        combs = combinations(candidates, size)
        for comb in combs:
            pval = self.ci_test(i, comb, verbose)

            if pval >= 1-self.confidence:
                self.selected.remove(i)
                break
        

    def find_cop(self, verbose=False):
        for i in range(self.n_vars):
            if i in self.selected:
                continue

            self.selected = list(np.sort(self.selected))
            pval = self.ci_test(i, self.selected, verbose)

            if pval < 1-self.confidence:
                self.selected.append(i)

        self.selected = list(np.sort(self.selected))


    def find_mb(self, max_cond, verbose=False):
        # random init
        np.random.shuffle(self.selected)

        # increase conditioning set size from 0
        for size in range(max_cond+1):
            # iterate through variables
            idx = 0
            while 1:
                if idx >= len(self.selected):
                    break

                test_i = self.selected[idx]
                self.test_var(test_i, size, verbose)

                if test_i in self.selected:
                    idx += 1

        # identify coparents
        if verbose:
            print ("Looking for co-parents...")

        self.find_cop(verbose)
        return self.selected


class MarkovBlanketKNN(MarkovBlanketBase):
    def __init__(self, X, Y, names=None, confidence=0.95, null_size=1000, n_resample=None, k_cmi=5, k_perm=5, transform='none', n_jobs=1):
        super().__init__(X, Y, names, confidence)
        self.k_cmi = k_cmi
        self.k_perm = k_perm
        self.n_resample = n_resample
        self.null_size = null_size
        self.transform = transform
        self.n_jobs = n_jobs


    def ci_test(self, i, cond_set, verbose=False):
        if len(cond_set) == 0:
            tester = IndepTest(self.X[:,:,i], self.Y, mixedMI, test_args={"k":self.k_cmi, 'transform':self.transform})
            if self.n_resample is None:
                pval,_ = batch_job(tester, self.null_size, n_jobs=self.n_jobs)
            else:
                pval,_ = resample_job(tester, self.null_size, self.n_resample, n_jobs=self.n_jobs)
        else:
            tester = CondIndepTest(self.X[:,:,i], self.Y, self.X[:,:,cond_set].reshape((self.X.shape[0],-1)), mixedCMI, test_args={"k":self.k_cmi, 'transform':self.transform}, k_perm=self.k_perm)
            if self.n_resample is None:
                pval,_ = batch_job(tester, self.null_size, n_jobs=self.n_jobs)
            else:
                pval,_ = resample_job(tester, self.null_size, self.n_resample, n_jobs=self.n_jobs)
        
        if verbose:
            print ("Test", i+1, "Cond. set:", [j+1 for j in cond_set], pval)

        return pval


class MarkovBlanketKernel(MarkovBlanketBase):
    def __init__(self, X, Y, names=None, confidence=0.95, num_boot=1000, B=25, b=1000, M=1000, n_jobs=1):
        super().__init__(X, Y, names, confidence)
        self.num_boot = num_boot
        self.B = B
        self.b = b
        self.M = M
        self.n_jobs = n_jobs

        self.X += np.random.normal(0, 1e-10, size=self.X.shape)
        self.Y += np.random.normal(0, 1e-10, size=self.Y.shape)


    def ci_test(self, i, cond_set, verbose):
        if len(cond_set) == 0:
            Kx, Ky = rbf_kernel_median(self.X[:,:,i],  self.Y)
            # pval = HSIC(Kx, Ky, p_val_method='bootstrap', num_boot=self.num_boot)
            _,pval = c_HSIC(Kx, Ky, size_of_null_sample=self.num_boot, with_null=False, n_jobs=self.n_jobs)
        else:
            Kx, Ky, Kz = rbf_kernel_median(self.X[:,:,i], self.Y, self.X[:,:,cond_set].reshape((self.X.shape[0],-1)))
            Dz = K2D(Kz)
            pval, _,_,_ = c_KCIPT(Kx, Ky, Kz, Dz=Dz, B=self.B, b=self.b, M=self.M, n_jobs=self.n_jobs)
        
        if verbose:
            print ("Test", i+1, "Cond. set:", [j+1 for j in cond_set], pval)

        return pval



class MarkovBlanketCCIT(MarkovBlanketBase):
    def __init__(self, X, Y, names=None, confidence=0.95, n_jobs=1):
        super().__init__(X, Y, names, confidence)
        self.n_jobs = n_jobs


    def ci_test(self, i, cond_set, verbose):
        def standardize(mat):
            mean = np.mean(mat, axis=0)
            std = np.std(mat, axis=0)
            return (mat - mean)/std

        if len(cond_set) == 0:
            pval = CCIT.CCIT(standardize(self.X[:,:,i]), 
                             standardize(self.Y), 
                             None, 
                             max_depths = [ 20, 40, 60,  80],
                             n_estimators=[50, 200, 400, 500],
                             colsample_bytrees=[0.3, 0.5, 0.8],
                             feature_selection=1,
                             num_iter=20, bootstrap=True, nthread=self.n_jobs)
        else:
            pval = CCIT.CCIT(standardize(self.X[:,:,i]), 
                             standardize(self.Y), 
                             standardize(self.X[:,:,cond_set].reshape((self.X.shape[0], -1))),
                             max_depths = [ 20, 40, 60,  80],
                             n_estimators=[50, 200, 400, 500],
                             colsample_bytrees=[0.3, 0.5, 0.8],
                             feature_selection=1,
                             num_iter=20, bootstrap=True, nthread=self.n_jobs)

        if verbose:
            print ("Test", i+1, "Cond. set:", [j+1 for j in cond_set], pval)

        return pval


class MarkovBlanketIFest(MarkovBlanketBase):
    def __init__(self, X, Y, names=None, null_size=1000, confidence=0.95, n_jobs=1):
        super().__init__(X, Y, names, confidence)
        self.n_jobs = n_jobs


    def ci_test(self, i, cond_set, verbose):
        pass

        # if verbose:
        #     print ("Test", i+1, "Cond. set:", [j+1 for j in cond_set], pval)

