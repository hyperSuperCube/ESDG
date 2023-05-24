import numpy as np
import DG2d as dg
import numpy.linalg as la
from matplotlib import pyplot as plt
class operators():
    def __init__(self, N, K):
        self.K = K
        self.N = N
        # Construct interpolatory space
        self.r, self.s = dg.Nodes(N)
        self.VDP = dg.Vandermonde2D(N,self.r,self.s)
        # Construct volume qudrature space
        self.rq,self.sq,self.wq = dg.quadNode(N,"Xiao_Gimbutas")
        self.Vq = dg.Vandermonde2D(N,self.rq,self.sq)@la.inv(self.VDP)
        self.M = np.diag(self.wq)
        self.Mhat = self.Vq.T@self.M@self.Vq
        self.Pq = la.inv(self.Mhat)@(self.Vq.T)@self.M
        """FIXME the first test should be implemented for testing the Pq"""
        # Constrct surface qudrature space
        self.r1d,self.w1d = dg.JacobiGQ(0,0,N)
        self.Nfp = len(self.r1d)
        self.e = np.ones(self.Nfp)
        self.z = np.zeros(self.Nfp)
        rf = np.concatenate((self.r1d, -self.r1d, -self.e),axis=0).T
        





        self.M = np.diag(self.w)
        self.invM = la.inv(self.h/2*self.M)
        self.D_hat = la.inv(self.Vq)@DG.Grad_Vandermonde1D(N,self.r)
        self.M_hat = (self.Vq.T)@self.M@self.Vq
        self.P = la.inv(self.M_hat)@(self.Vq.T)@self.M
        self.R = self.Vf@self.P
        self.nx = np.diag([[-1,-1],[1,1]])
        self.nx_ = np.ones((2,N+1))
        self.nx_[0] = -1
        self.Bf = np.diag([1,1])
        self.E = self.R.T@(self.Bf*self.nx)@self.R
        self.Vqf = np.concatenate((self.Vq,self.Vf),axis=0)
        self.D = 0.5*la.inv(self.M)@((self.R+self.Vf@self.P).T@(self.Bf*self.nx)@(self.R-self.Vf@self.P)) + self.Vq@self.D_hat@self.P
        self.Dm = np.block([[self.D-0.5*la.inv(self.M)@self.E, 0.5*(la.inv(self.M)@self.R.T@(self.Bf*self.nx))],
               [-0.5 * self.R * self.nx_,          0.5 * self.nx * np.eye(2)]])
        self.Mm = np.block([[self.M, np.zeros((N+1, 2))],
              [np.zeros((2,N+1)), self.Bf]])
        self.Qm = self.Mm@self.Dm            
        self.Q = self.M@self.D

