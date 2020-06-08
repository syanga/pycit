import numpy as np
import h5py
from scipy.special import comb

class BullseyeSum:
    def __init__(self, n, eps, n_vars):
        self.n = n
        self.eps = eps
        self.n_vars = n_vars

        self.X = np.zeros((n, 2, n_vars))
        self.R = np.zeros((n, 1, n_vars))
        self.T = np.zeros((n, 1, n_vars))
        for i in range(n_vars):
            inner_idx = np.random.binomial(1, 0.5, size=n).astype('?')
            self.R[inner_idx,0,i] = np.random.uniform(1, 2, size=int(np.sum(inner_idx)))
            self.R[~inner_idx,0,i] = np.random.uniform(3, 4, size=int(np.sum(~inner_idx)))
            
            self.T[:,0,i] = np.random.uniform(0, 2*np.pi, size=(n,))

            self.X[:,0,i] = self.R[:,0,i]*np.cos(self.T[:,0,i])
            self.X[:,1,i] = self.R[:,0,i]*np.sin(self.T[:,0,i])

        self.statistic = np.mean(self.R[:,0,:], axis=1).reshape(-1,1)
        self.Y = self.statistic + np.random.uniform(-eps, eps, size=self.statistic.shape)


    def make_X_data(self, dest):
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=self.X)
            f.create_dataset("Y", data=self.Y)


    def make_R_data(self, dest):
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=self.R)
            f.create_dataset("Y", data=self.Y)


class BullseyeData:
    def __init__(self, n, eps, copies=1, scale_4=False):
        eps_list = [0.025, 0.05, 0.075, 0.1, 0.125, -1, -2]
        assert eps in eps_list
        a,b = 0.25, 0.5
        c,d = 0.75, 1.0
        self.n = n
        self.copies = copies
        
        self.eps = eps
        if eps == 0.025:
            self.ground_truth = 2.4026 * copies
        elif eps == 0.05:
            self.ground_truth = 1.8095 * copies
        elif eps == 0.075:
            self.ground_truth = 1.5040 * copies
        elif eps == 0.1:
            self.ground_truth = 1.3163 * copies
        elif eps == 0.125:
            self.ground_truth = 1.1931 * copies
        elif eps == -1:
            self.ground_truth = (2.4026 + 1.8095 + 1.5040 + 1.3163 + 1.1931)*(1/5) * copies
        elif eps == -2:
            # W is 0-mean gaussian, eps is 0.025 if W<0, 0.125 if W>=0.
            # ground truth is I(X;Y|W)
            self.ground_truth = (2.4026 + 1.1931)*0.5 * copies

        # scale by 4
        if scale_4:
            a = 4*a
            b = 4*b
            c = 4*c
            d = 4*d
            eps = 4*eps
            
        # init
        idx = np.random.permutation(n)
        self.R = np.zeros((n, copies))
        self.T = np.zeros((n, copies))
        self.Y = np.zeros((n, copies))
        self.X = np.zeros((n, 2*copies)) 

        for i in range(copies):
            self.R[:,i] = np.hstack((np.random.uniform(a,b,size=int(n/2)), np.random.uniform(c,d,size=n-int(n/2))))[idx]

        if eps == -1:
            self.e = np.random.choice(3, n, replace=True).reshape(-1,1)
            N = np.zeros(self.R.shape)
            for i in range(3):
                N[self.e==i] = np.random.uniform(-eps_list[i], eps_list[i], size=N[self.e==i].shape)
            self.Y = self.R + N
        if eps == -2:
            self.W = np.random.normal(0, 1, size=n).reshape(-1,1)
            N = np.zeros(self.R.shape)
            N[self.W < 0] = np.random.uniform(-0.025, 0.025, size=N[self.W < 0].shape)
            N[self.W >= 0] = np.random.uniform(-0.125, 0.125, size=N[self.W >= 0].shape)
            self.Y = self.R + N
            self.e = self.W >= 0
        else:
            N = np.random.uniform(-eps, eps, size=self.R.shape)
            self.Y = self.R + N

        for i in range(copies):
            self.T[:,i] = np.random.uniform(0, 2*np.pi, size=n)
            self.X[:,i] = self.R[:,i]*np.cos(self.T[:,i])
            self.X[:,i+copies] = self.R[:,i]*np.sin(self.T[:,i])


    def make_X_data(self, dest, include_polar=False):
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=np.expand_dims(self.X, 2))
            f.create_dataset("Y", data=self.Y)


    def make_R_data(self, dest):
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=np.expand_dims(self.R, 2))
            f.create_dataset("Y", data=self.Y)


    def make_XR_data(self, dest, include_polar=False):
        R = np.concatenate((self.R, np.zeros(self.R.shape)), axis=1)
        Z = np.stack((self.X, R), axis=2)
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=Z)
            f.create_dataset("Y", data=self.Y)


    def make_XRT_data(self, dest, include_polar=False):
        R = np.concatenate((self.R, np.zeros(self.R.shape)), axis=1)
        T = np.concatenate((self.T, np.zeros(self.T.shape)), axis=1)
        Z = np.stack((self.X, R, T), axis=2)
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=Z)
            f.create_dataset("Y", data=self.Y)


    def make_XRTe_data(self, dest, include_polar=False):
        R = np.concatenate((self.R, np.zeros(self.R.shape)), axis=1)
        T = np.concatenate((self.T, np.zeros(self.T.shape)), axis=1)
        e = np.concatenate((self.e, np.zeros(self.e.shape)), axis=1)
        W = np.concatenate((self.W, np.zeros(self.e.shape)), axis=1)
        Z = np.stack((self.X, R, T, e, W), axis=2)
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=Z)
            f.create_dataset("Y", data=self.Y)



class BullseyeData2:
    def __init__(self, n, copies=1):
        a,b = 1/7,3/7
        c,d = 5/7,1
        eps = 1/7
        self.n = n
        self.copies = copies
        self.ground_truth = 1.1931 * copies

        # init
        idx = np.random.permutation(n)
        self.R = np.zeros((n, copies))
        self.T = np.zeros((n, copies))
        self.Y = np.zeros((n, copies))
        self.X = np.zeros((n, 2*copies))      

        for i in range(copies):
            self.R[:,i] = np.hstack((np.random.uniform(a,b,size=int(n/2)), np.random.uniform(c,d,size=n-int(n/2))))[idx]
            self.T[:,i] = np.random.uniform(0, 2*np.pi, size=n)
            self.X[:,i] = self.R[:,i]*np.cos(self.T[:,i])
            self.X[:,i+copies] = self.R[:,i]*np.sin(self.T[:,i])

        # self.X = np.vstack((self.R*np.cos(self.T), self.R*np.sin(self.T))).T
        N = np.random.uniform(-eps, eps, size=self.R.shape)
        self.Y = self.R + N


    def make_X_data(self, dest, include_polar=False):
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=np.expand_dims(self.X, 2))
            f.create_dataset("Y", data=self.Y)


    def make_R_data(self, dest):
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=np.expand_dims(self.R, 2))
            f.create_dataset("Y", data=self.Y)


    def make_XRT_data(self, dest, include_polar=False):
        Z = np.vstack((self.X.T, self.R.T, self.T.T)).T
        with h5py.File(dest, "w") as f:
            f.create_dataset("X", data=np.expand_dims(Z, 2))
            f.create_dataset("Y", data=self.Y)
