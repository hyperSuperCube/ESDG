#----------------------------------------------------------
#   Broad way ES preparation functions for DG
#   Constructing operators and flux needed for rhs
#   Chandrashekar flux 
#   Solid/Periodic boundary condition
#   Author: Zirui Wang
#   Date: Nov.17 2022
#----------------------------------------------------------
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import Operator as pcD
# Prepare for hardamada product
gamma = 1.4
nx = np.ones((2,pcD.Ke))
nx[0,:] = -1

nx_ = np.zeros((pcD.NP+1,pcD.Ke))
nx_[0] = -1
nx_[-1] = 1

def roeAvg(rhoL,rhoR,rhouL,rhouR,rhoEL,rhoER):
    r""" Evaluate the roe average of the x and sound velocity
    """
    rhoLg, rhoRg, rhouLg, rhouRg, rhoELg, rhoERg = BC(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL)
    rhoL,uL,pL = conservativeToPrimitive(rhoL,rhouL,rhoEL)
    rhoR,uR,pR = conservativeToPrimitive(rhoR,rhouR,rhoER)

    rhoLg,uLg,pLg = conservativeToPrimitive(rhoLg,rhouLg,rhoELg)
    rhoRg,uRg,pRg = conservativeToPrimitive(rhoRg,rhouRg,rhoERg)

    enthL = (rhoEL+pL)/rhoL
    enthR = (rhoER+pR)/rhoR
    roeU, roeH, roec = np.zeros(pcD.Ke+1),np.zeros(pcD.Ke+1),np.zeros(pcD.Ke+1)
    roeU[1:-1] = (np.sqrt(rhoL[1:])*uL[1:] + np.sqrt(rhoR[:-1])*uR[:-1])/(np.sqrt(rhoL[1:]) + np.sqrt(rhoR[:-1]))
    roeH[1:-1] = (np.sqrt(rhoL[1:])*enthL[1:] + np.sqrt(rhoR[:-1])*enthR[:-1])/(np.sqrt(rhoL[1:]) + np.sqrt(rhoR[:-1]))
    
    roeU[0] = (np.sqrt(rhoL[0])*uL[0] + np.sqrt(rhoLg)*uLg)/(np.sqrt(rhoL[0]) + np.sqrt(rhoLg))
    roeU[0] = (np.sqrt(rhoL[0])*uL[0] + np.sqrt(rhoLg)*uLg)/(np.sqrt(rhoL[0]) + np.sqrt(rhoLg))

    roec = (gamma-1)*(roeH-0.5*roeU)
    return roeU,roec
    

def conservativeToPrimitive(rho,rhou,E):
    u = rhou/rho
    p = (E/rho-u**2/2)*rho*(gamma-1)
    return rho,u,p

def primitiveToConservative(rho, u, p):
    e = p/(rho*(gamma-1))
    rhoE = (e+u**2/2)*rho
    return rho, rho*u, rhoE

def conservativeToEntropy(rho, rhou, rhoE):
    rho, u, p = conservativeToPrimitive(rho, rhou, rhoE)
    s = np.log(p) - gamma*np.log(rho)
    v1 = (gamma+1-s)-(gamma-1)*rhoE/p
    v2 = rhou*(gamma-1)/p
    v3 = -rho*(gamma-1)/p
    return v1,v2,v3

def entropyToConservative(v1,v2,v3):
    s = gamma-v1+0.5*v2**2/v3
    rhoe = ((gamma-1)/((-v3)**gamma))**(1/(gamma-1)) * np.exp(-s/(gamma-1))
    rho = rhoe*(-v3)
    rhou = rhoe*v2
    rhoE = rhoe*(1-0.5*v2**2/v3)
    return rho, rhou, rhoE


def LRface(Q):
    QL = Q[0,:]
    QR = Q[-1,:]
    return QL,QR

'''   These are scalar functions for D and pannelty term, 
but the final penalty function use these scalar functions
                              to apply boundary condition''' 
    
def scalarAVG(qR,qL):
    return (qR+qL)/2


def scalarLOGM(aR,aL):
    da = aR - aL
    aAvg = (aR + aL) / 2
    f = da / aAvg
    v = f ** 2
    if abs(f) < 1e-4:
        toReturn = aAvg * (1 + v * (-.2 - v * (.0512 - v * 0.026038857142857)))
    else:
        toReturn = da / (np.log(aR) - np.log(aL))
    return toReturn


# @return mass/momentum/energy flux value at interface/(Fs)ij
def chandFlux(rhoR,rhoL,uR,uL,pR,pL):
    betaL = rhoL/(2*pL)
    betaR = rhoR/(2*pR)
    rholog = scalarLOGM(rhoL,rhoR)
    betalog = scalarLOGM(betaL,betaR)

    rhoavg = .5*(rhoL+rhoR)
    uavg   = .5*(uL+uR)

    unorm = uL*uR
    pa    = rhoavg/(betaL+betaR)
    f4aux = rholog/(2*(gamma-1)*betalog) + pa + .5*rholog*unorm

    FxS1 = rholog*uavg
    FxS2 = FxS1*uavg + pa
    FxS3 = f4aux*uavg
    return FxS1, FxS2, FxS3

# The Lax coeff.
def LaxC(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL):
    K = len(rhoR)
    rhoL, uL, pL = conservativeToPrimitive(rhoL, rhouL, rhoEL)
    rhoR, uR, pR = conservativeToPrimitive(rhoR, rhouR, rhoER)
    lmL = abs(uL) + np.sqrt(gamma*pL/rhoL)
    lmR = abs(uR) + np.sqrt(gamma*pR/rhoR)
    LFc = np.zeros((2,K))
    LFc[0,1:] = np.max([lmR[:-1], lmL[1:]])
    LFc[1,:-1] = np.max([lmR[:-1], lmL[1:]])

    LFc[0,0] = lmL[0]
    LFc[-1,-1] = lmR[-1]

    return LFc

'''
    @ type Periodic/Mirror B.C.
    @ return the specified EOS at boundaries
    @ shape 6' of [q1bL q1bR q2bL q2bR q3bL q3bR]
'''
def BC(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL,TYPE="shu"):
    # rhoR,rhoL = faceIndexing(rho)
    # rhouR,rhouL= faceIndexing(rhou)
    # rhoER,rhoEL = faceIndexing(rhoE)
    if TYPE == "sod":
        rhobL = pcD.RL
        rhobR = pcD.RR
        rhoubL = 0
        rhoubR = 0
        rhoEbL = pcD.PL/(gamma-1)
        rhoEbR = pcD.PR/(gamma-1)
    if TYPE == "shu":
        rhobL  = pcD.RL
        rhobR  = pcD.f(1)
        rhoubL = pcD.RL*pcD.UL
        rhoubR = 0
        rhoEbL = pcD.PL/(gamma-1) + pcD.RL*pcD.UL**2/2
        rhoEbR = pcD.PR/(gamma-1) 
    if TYPE == 'Per':
        rhobL  = rhoR[-1]
        rhobR  = rhoL[0]
        rhoubL = rhouR[-1]
        rhoubR = rhouL[0]
        rhoEbL = rhoER[-1]
        rhoEbR = rhoEL[0]
    return rhobL, rhobR, rhoubL, rhoubR, rhoEbL, rhoEbR

''' 
    The jump is computed as the interior quantity suntract the exterior quantity, 
    thus the jump at two overlapping faces should be equal and opposite sign [q] 
    and we assume there is no jump at boundary. For vector jump, we need to time 
    n-, the outward normal vector.
'''
def Jump(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL):
    K = len(rhoR)
    rhoJ  = np.zeros((2,K))
    rhouJ = np.zeros((2,K))
    rhoEJ = np.zeros((2,K))
    rhoJ[0,1:] = rhoL[1:] - rhoR[:-1]
    rhoJ[1,:-1] = rhoR[:-1] - rhoL[1:]

    rhouJ[0,1:] = rhouL[1:] - rhouR[:-1]
    rhouJ[1,:-1] = rhouR[:-1] - rhouL[1:]

    rhoEJ[0,1:] = rhoEL[1:] - rhoER[:-1]
    rhoEJ[1,:-1] = rhoER[:-1] - rhoEL[1:]
    rhoLg, rhoRg, rhouLg, rhouRg, rhoELg, rhoERg = BC(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL)
    rhoJ[0,0]  =  rhoL[0] - rhoLg
    rhouJ[0,0] = rhouL[0] - rhouLg
    rhoEJ[0,0] = rhoEL[0] - rhoELg

    rhoJ[-1,-1]  =  rhoR[-1] - rhoRg
    rhouJ[-1,-1] = rhouR[-1] - rhouRg
    rhoEJ[-1,-1] = rhoER[-1] - rhoERg

    return rhoJ, rhouJ, rhoEJ

def Jump2(q1,q2,q3):
    # for Neuman boundary condition, we set zero gradiant at the boundary
    q1J = np.zeros((2,pcD.Ke))
    q2J = np.zeros((2,pcD.Ke))
    q3J = np.zeros((2,pcD.Ke))
    q1J[0,1:] = q1[0,1:] - q1[-1,:-1]
    q2J[0,1:] = q2[0,1:] - q2[-1,:-1]
    q3J[0,1:] = q3[0,1:] - q3[-1,:-1]

    q1J[1,:-1] = -(q1[0,1:] - q1[-1,:-1])
    q2J[1,:-1] = -(q2[0,1:] - q2[-1,:-1])
    q3J[1,:-1] = -(q3[0,1:] - q3[-1,:-1])
    return q1J,q2J,q3J



# The Lax dissipation function
def LaxD(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL):
    LFc = LaxC(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL)
    rhoJ, rhouJ, rhoEJ = Jump(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL)
    return nx*LFc/2*rhoJ, nx*LFc/2*rhouJ, nx*LFc/2*rhoEJ

'''
    @ arg1 = state variable
    @ arg2 = one of euler flux
    @ input is full shape of the EOS
    @ return the flux array of interelement matrix array 
    @ shape ((N+3,N+3,K,3))
    @ Test should be symmetric; diagnal should be consistant with physical flux
'''

def Fs(rho,rhou,rhoE,fluxFunc=chandFlux):
    Vq = pcD.nQspace.Vq
    Vf = pcD.nQspace.Vf
    P = pcD.nQspace.P
    VPq = Vq@P
    VPf = Vf@P
    rhop = np.concatenate((VPq@rho,VPf@rho), axis=0)
    rhoup = np.concatenate((VPq@rhou,VPf@rhou), axis=0)
    rhoEp = np.concatenate((VPq@rhoE,VPf@rhoE), axis=0)

    rho,u,p = conservativeToPrimitive(rhop, rhoup, rhoEp)
    row, K = rho.shape
    Fs = np.zeros((row,row,K,3))
    for k in range(K):
        for j in range(row):
            for i in range(row):
                Fs[j,i,k,:] = fluxFunc(rho[j,k],rho[i,k],u[j,k],u[i,k],p[j,k],p[i,k])[:]
    return Fs
def filtering(Q):
    Qmode = pcD.nPspace.P@Q # finding the mode of the conservative variable
    #higest mode is the highest bais mode
    Qmode[-1,:] *= 0
    return (pcD.nPspace.Vq)@Qmode

sig = lambda qm,qp,dqm,dqp: 0.5*(qm**2+qp**2) + pcD.nPspace.h**2/4*(
    dqm**2 + dqp**2
    )**0.5

def damperHelp(rho,rhou,rhoE):
    rhoL, rhoR = LRface(rho)
    rhouL, rhouR = LRface(rhou)
    rhoEL, rhoER = LRface(rhoE)
    rho,u,p = conservativeToPrimitive(rho,rhou,rhoE)
    c = np.sqrt(gamma*p/rho)
    # Using high dimensional array to store different eigRinv!!!! Bug! Here the u should be taken a
    # as the element wise average
    # eigRinv is the inverse of the characteristic decomposition.
    """arg: :
                eigmatcol, eigmatrow, #elem, L or R faces (3,3,K,2)
    """
    eigRinv= np.zeros((3,3,pcD.Ke,2))
    for k in range(pcD.Ke):
        # for the left face do roe avg
        eigRinv[:,:,k,0] = (gamma-1)/c*np.array([[0.5*u*c + 0.25*(gamma-1)*u**2, -0.5*(gamma-1)*u-0.5*c, 0.5*(gamma-1)],
                                    [c**2-0.5*(gamma-1)*u**2,       (gamma-1)*u,                  1-gamma],
                                    [-0.5*u*c + 0.25*(gamma-1)*u**2,-0.5*(gamma-1)*u+0.5*c, 0.5*(gamma-1)]])
    
    rhoJ,rhouJ,rhoEJ = Jump(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL)
    drho,du,dp = pcD.nPspace.D@(rho,u,p)
    print(1)
    dLf = eigRinv@np.concatenate((drho[0, :],du[0, :],dp[0, :]),axis=0)
    dRf = eigRinv@np.concatenate((drho[-1,:],du[-1,:],dp[-1,:]),axis=0)

    drho = np.concatenate((dLf[0,:],dRf[0,:]),axis = 0)
    drhou= np.concatenate((dLf[1,:],dRf[1,:]),axis = 0)
    drhoE= np.concatenate((dLf[2,:],dRf[2,:]),axis = 0)

    drhoJ,drhouJ,drhoEJ = Jump2(drho,drhou,drhoE)

    sig1 = sig(rhoJ[0,:], rhoJ[-1,:], drhoJ[0,:], drhoJ[-1, :])
    sig2 = sig(rhouJ[0,:],rhouJ[-1,:],drhouJ[0,:],drhouJ[-1,:])
    sig3 = sig(rhoEJ[0,:],rhoEJ[-1,:],drhoEJ[0,:],drhoEJ[-1,:])

    sigma = np.maximum(sig1**0.5,sig2**0.5,sig3**0.5)
    return sigma


def damper(rho,rhou,rhoE):
    pass




