import numpy as np
import math as m
import numpy.linalg as la
'''Generating the functional space for solution and test function'''
def JacobiP(x,alpha,beta,N):
    x = np.array(x)
    PL = np.zeros((N+1,len(x)))
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*m.gamma(alpha+1)*m.gamma(beta+1)/m.gamma(alpha+beta+1)
    PL[0,:] = 1/np.sqrt(gamma0)
    if N == 0:
        P = PL[-1,:]
        return P
    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    PL[1,:] = ((alpha+beta+2)*x/2+(alpha-beta)/2)/np.sqrt(gamma1)
    if N == 1:
        P = PL[-1,:]
        return P
    a_old = 2/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
    for i in range(1,N):
        h1 = 2*i+alpha+beta
        a_new = 2/(h1+2)*np.sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3))
        b_new = -(alpha**2-beta**2)/h1/(h1+2)
        PL[i+1,:] = 1/a_new*(-a_old*PL[i-1,:]+(x-b_new)*PL[i,:])
        a_old = a_new
    P = PL[-1,:]
    return P
def JacobiGQ(alpha,beta,N):
    if N == 0:
        x = np.array([-(alpha-beta)/(alpha+beta+2)])
        w = np.array([2])
        return x,w
    h1 = 2*np.arange(N+1)+alpha+beta
    J = np.diag(-0.5*(alpha**2-beta**2+1e-25)/(h1+2)/(h1+1e-25),0)+np.diag(2/(h1[:-1]+2)*np.sqrt(np.arange(1,N+1)*(alpha+beta+np.arange(1,N+1))
                                                           *((np.arange(1,N+1)+alpha)*(np.arange(1,N+1)+beta)/(h1[:-1]+1)/(h1[:-1]+3))),1)
    if alpha+beta < 1e-10:
        J[0][0] = 0
    J = J+J.T
    X,V = la.eigh(J)
    w = V[0,:].T**2*2**(alpha+beta+1)/(alpha+beta+1)*m.gamma(alpha+1)*m.gamma(beta+1)/m.gamma(alpha+beta+1)
    return X,w
def JacobiGL(alpha,beta,N):
    x = np.zeros(N+1)
    if N == 1:
        w = np.array([1,1])
        x[0] = -1
        x[1] = 1
        return x,w
    x,w = JacobiGQ(alpha+1,beta+1,N-2)
    X = [-1]
    for i in x:
        X.append(i)
    X.append(1)
    X = np.array(X)
    V = Vandermonde1D(N,X)
    w = np.sum((la.inv(V@V.T)),axis=1)
    return X,w
'''This function generate all of the physical GL points, storing the points of each element in col'''
def GLP(alpha,beta,N,x):
    '''x is the physical coordinate of grids'''
    K = len(x)-1
    xF = np.zeros((2,K))
    xF[0] = x[0:-1] # the left physical coordinate
    xF[1] = x[1: ] # the right physical coordinate
    r,w = JacobiGL(alpha,beta,N)
    x = np.zeros((N+1, K))
    for i in range(K):
        x[:,i] = xF[0,i]+(1+r)*np.abs(xF[1,i]-xF[0,i])/2
    return x
def pNodes(alpha,beta,x,r):
    '''x is the physical coordinate of grids'''
    K = len(x)-1
    xF = np.zeros((2,K))
    xF[0] = x[0:-1] # the left physical coordinate
    xF[1] = x[1: ] # the right physical coordinate
    x = np.zeros((len(r), K))
    for i in range(K):
        x[:,i] = xF[0,i]+(1+r)*np.abs(xF[1,i]-xF[0,i])/2
    return x
def Vandermonde1D(N,r):
    V1D = np.zeros((len(r),N+1))
    for i in range(N+1):
        V1D[:,i] = JacobiP(r,0,0,i)
    return V1D
def JacobiGrad(r,alpha,beta,N):
    if N == 0:
        dP = np.zeros(len(r))
    else:
        dP = np.sqrt(N*(N+alpha+beta+1))*JacobiP(r,alpha+1,beta+1,N-1)
    return dP
def Grad_Vandermonde1D(N,r):
    DVr = np.zeros((len(r),N+1))
    for i in range(N+1):
        DVr[:,i] = JacobiGrad(r,0,0,i)
    return DVr






