import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import Operator as pcD
import Entropy as SS
gamma = 1.4
mu = 0
cp = gamma/(gamma-1)
kappa = mu*cp/0.75
cv = 1/(gamma-1)
Rc = cp-cv
N = pcD.NP
Kelms = pcD.Ke
Q = pcD.nPspace.Q
R = pcD.nPspace.R
Bf = pcD.nPspace.Bf
One = np.ones(pcD.NQ+3)
nx = np.ones((2,pcD.Ke))
nx[0] = -1
h = pcD.nPspace.h
invM = pcD.nPspace.invM
def centralFlx(qR,qL):
    return (qR+qL)/2
def Neuman(uR,uL):
    ubL  = uL[0]
    ubR  = uR[-1]
    return ubR,ubL

def SAT(Q, flux,BC):
    QL , QR  = SS.LRface(Q)
    QbR, QbL = BC(QR, QL)
    qs = np.zeros((2,pcD.Ke))
    for k in range(1,pcD.Ke-1):
        qs[0,k] = flux(QL[k],QR[k-1])
        qs[1,k] = flux(QR[k],QL[k+1])
    qs[0,0] = flux(QL[0],QbL)
    qs[1,0] = flux(QR[0],QL[1])

    qs[1,-1] = flux(QR[-1],QbR)
    qs[0,-1] = flux(QL[-1],QR[-2])
    return invM@(R.T@Bf@(nx*qs))

def diff(q,flux,BC):
    sat = SAT(q,flux,BC)
    return sat - invM@((Q.T)@q)

def rhsV(rho,rhou,rhoE):
    # rho = pcD.overInt.It@rho
    # rhou = pcD.overInt.It@rhou
    # rhoE = pcD.overInt.It@rhoE

    rho,u,p = SS.conservativeToPrimitive(rho,rhou,rhoE)
    T = p/(Rc*rho)

    gradu = diff(u,centralFlx,Neuman)
    gradT = diff(T,centralFlx,Neuman)
    
    tau = 4/3*mu*gradu
    qx = kappa*gradT
    maV = 0
    moV = diff(tau,centralFlx,Neuman)
    enV = diff(u*tau,centralFlx,Neuman) + diff(qx,centralFlx,Neuman)

    return maV, moV, enV

