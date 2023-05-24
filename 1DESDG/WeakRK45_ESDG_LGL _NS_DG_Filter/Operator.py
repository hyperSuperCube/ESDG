import DG as DG
import numpy as np
import numpy.linalg as la

''' Here the normal nx in Shu's context is the direct Hadamard operation, one the correspounding 
faces, it has the corrspounding normal direction'''
CFL = 0.1
deltp = 0
NP = 3
NQ = NP + deltp
Ke = 300
class preComp():
    def __init__(self, N, K, RL = 0, RR = 0, UL = 0, PL = 0, PR = 0,f = 0):
        self.f = f
        self.RL = RL
        self.RR = RR
        self.UL = UL
        self.PL = PL
        self.PR = PR

        self.K = K
        self.N = N
        self.xRef = np.linspace(0,1,self.K+1)
        self.h = abs(self.xRef[1] - self.xRef[0])
        self.r, self.w = DG.JacobiGL(0,0,N)
        self.xP = DG.GLP(0,0,N,self.xRef)
        self.Vq = DG.Vandermonde1D(N,self.r)
        self.Vf = DG.Vandermonde1D(N,[-1,1])

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
    def Init(self):
        rho = np.zeros((self.N+1,self.K))
        u   = np.zeros((self.N+1,self.K))
        p   = np.zeros((self.N+1,self.K))
        for k in range(self.K):
            for j in range(self.N+1):
                if self.xP[j,k] < 0.125:
                    rho[j,k] = self.RL
                    u[j,k] = self.UL
                    p[j,k] = self.PL
                else:
                    rho[j,k] = self.f(self.xP[j,k])
                    p[j,k] = self.PR
        return rho, u, p

class oInt(preComp):
    def __init__(self,N,K,RL,RR,UL,PL,PR,f):
        super().__init__(N,K,RL,RR,UL,PL,PR,f)
        self.q = N + deltp
        self.rq,self.wq = DG.JacobiGL(0,0,self.q)
        self.Mq = np.diag(self.wq)
        self.Mqhat = DG.Vandermonde1D(N,self.rq).T@self.Mq@DG.Vandermonde1D(N,self.rq)
        self.It = DG.Vandermonde1D(N,self.rq)@la.inv(self.Vq)
        # Project the high order polynomial to the nodal of pN order space
        self.Pq = self.Vq@la.inv(self.Mqhat)@DG.Vandermonde1D(N,self.rq).T@self.Mq
RL = 3.857143
RR = 0
UL = 2.629369
PL = 10.33333
PR = 1

# RL = 4
# RR = 1.211
# UL = 0
# PL = 5
# PR = .5

# RL = 1
# RR = 0.1
# UL = 0
# UR = 0
# PL = 1
# PR = 0.1
gamma = 1.4
f = lambda x: 1+0.2*np.sin(8*2*np.pi*x)
nPspace = preComp(NP,Ke,RL,RR,UL,PL,PR,f)
nQspace = preComp(NP,Ke,RL,RR,UL,PL,PR,f)
overInt =    oInt(NP,Ke,RL,RR,UL,PL,PR,f)
rho, u, p = nPspace.Init()


