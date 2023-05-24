import numpy as np
import math as m
import numpy.linalg as la
from matplotlib import pyplot as plt
"""
    NOTE: :The routines for setting up 2D DG 
    
"""
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

def simplex2D(a,b,i,j):
    r""" Evaluate 2D orthonormal polynomial on simplex 
         at (a,b) of order (i,j)
         
         ..math:: 
            \phi_{i,j}(r,s) = \sqrt{2}P^{(0,0)}_{i}(a)P^{(2i+1,0)}_{j}(b)(1-b)^i

         Parameters
         ----------
         a: :ndarray a = 2\frac{1+r}{1-s} - 1
         b: :ndarray b = s

         return the value of the function phi at (a,b)
         -----------
         NOTE: :There are (N+1)(N+2)/2 Dofs
    """
    h1, h2 = JacobiP(a,0,0,i), JacobiP(b,2*i+1,0,j)
    P = np.sqrt(2)*h1*h2*(1-b)**i
    return P

def rstoab(r,s):
    r""" Converting from the standard coordinate to the basis function parameter
    
         ..math::
            a: :2\frac{1+r}{1-s} - 1
            b: :s
        Parameter
        ---------
        s (-1,1)
        ^
        |
        |     I
        |
        |___________> r (1,-1)
        (-1,-1)
        return (a,b).
        ---------
    """
    Np = len(r)
    a = np.zeros(Np)
    for n in range(Np):
        if s[n] != 1:
            a[n] = 2*(1+r[n])/(1-s[n]) - 1
        else:
            a[n] = -1
    return a, s

def Warpfactor(N, rout):
    r"""This is the helper function to use Blend and recursive approach to 
    generate some interpolatory points on the reference triangle
    
    Parameter
    ---------
    More details please refer to the book 
    "Nodal discontinuous Galerkin Method: Algorithem, Analysis, and Application"
    ---------
    """
    LGLr,w = JacobiGL(0,0,N)
    req = np.linspace(-1,1,N+1)
    Veq = Vandermonde1D(N,req)
    Pmat = np.zeros((N+1,len(rout)))
    for i in range(N+1):
        Pmat[i,:] = JacobiP(rout, 0, 0, i)
    Lmat = la.inv(Veq.T)@Pmat
    print((LGLr - req).shape)
    warp = Lmat.T@(LGLr - req)

    zerof = abs(rout) < 1.0-1e-10
    sf = 1 - (zerof*rout)**2
    warp = warp/sf + warp*(zerof-1)
    return warp

def Nodes(N):
    r"""Genrating nodes on the equilateral triangles"""
    alpopt = [0.0000,0.0000,1.4152,0.1001,0.2751,0.9800,1.0999,1.2832,1.3648,1.4773,1.4959,1.5743,1.5770,1.6223,1.6258]
    if N < 16:
        alpha = alpopt[N]
    else:
        alpha = 5/3
    Np = int((N+1)*(N+2)/2)
    L1 = np.zeros(Np)
    L2 = np.zeros(Np)
    L3 = np.zeros(Np)
    sk = 0
    for n in range(N+1):
        for m in range(N-n+1):
            L1[sk] = n/N
            L3[sk] = m/N
            sk += 1
    L2 = 1-L1-L3
    x = -L2+L3
    y = (-L2-L3+2*L1)/np.sqrt(3)
    blend1 = 4*L2*L3
    blend2 = 4*L1*L3
    blend3 = 4*L1*L2
    warpf1 = Warpfactor(N,L3-L2)
    warpf2 = Warpfactor(N,L1-L3)
    warpf3 = Warpfactor(N,L2-L1)

    warp1 = blend1*warpf1*(1 + (alpha*L1)**2)
    warp2 = blend2*warpf2*(1 + (alpha*L2)**2)
    warp3 = blend3*warpf3*(1 + (alpha*L3)**2)

    x = x + 1*warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4*np.pi/3)*warp3
    y = y + 0*warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4*np.pi/3)*warp3

    return xytors(x,y)
def xytors(x,y):
    r""" Converting from the equilateral triangle to the reference simplex
    
         ..math::
            r: :-L2 + L3 - L1
            s: :-L2 - L3 + L1
            L_{n}: :Blending factor
        Parameter
        ---------
        (x,y): :Coordinate on equilateral triangle
        return (r,s)
        ---------
    """
    L1 = (np.sqrt(3.0)*y+1.0)/3.0
    L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    L3 = ( 3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1
    return r,s

def gradSimplex2D(a,b,id,jd):
    r"""Evalute the partial derivative w.r.t. (r,s) of 2D "Legendre" polynomial with
    index, or order, (id,jd) at (a,b)
    """
    fa = JacobiP(a,0,0,id)
    gb = JacobiP(b, 2*id+1, 0, jd)
    dfa = JacobiGrad(a, 0, 0, id)
    dgb = JacobiGrad(b, 2*id+1, 0, jd)
    dmodedr = dfa*gb
    if id > 0:
        dmodedr *= (0.5*(1-b))**(id-1)
    dmodeds = dfa*(gb*(0.5*(1+a)))
    if id > 0:
        dmodeds *= (0.5*(1-b))**(id-1)
    tmp = dgb*((0.5*(1-b))**id)
    if id > 0:
        tmp -= 0.5*id*gb*((0.5*(1 - b))**(id-1))
    dmodeds = dmodeds+fa*tmp

    dmodedr = 2**(id+0.5)*dmodedr
    dmodeds = 2**(id+0.5)*dmodeds
    return dmodedr, dmodeds

def Vandermonde2D(N,r,s):
    r"""Initialize the 2D Vandermonde matrix of order N "Legendre" polynomials at
        nodes (r,s), which is the point in qudrature space
        NOTE
        ------
        The V2D should be a square matrix, but it can alos be a non-square matrix 
        when qudrature space does not fall into the interoplatory space. 
        
        V2D.col: :The # Dofs in interpolatory space (Represent P^N polynomial)
        V2D.row: :The # Dofs in the qudrature space

        ..math: :V_{ij} = phi_j(r_i, s_i)
        ------
    """
    # FIXME
    Np = int((N+1)*(N+2)/2)
    V2D = np.zeros((len(r),Np))
    a,b = rstoab(r,s)
    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2D[:,sk] = simplex2D(a,b,i,j)
            sk += 1
    return V2D

def gradVandermonde2D(N,r,s):

    Np = int((N+1)*(N+2)/2)
    V2Dr = np.zeros((len(r),Np))
    V2Ds = np.zeros((len(r),Np))
    a, b = rstoab(r,s)
    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            V2Dr[:,sk], V2Ds[:,sk] = gradSimplex2D(a,b,i,j)
            sk += 1
    return V2Dr, V2Ds

def quadNode(N,name):
    r"""Calculate the required Nth order qudrature space, that the corespounding qudrature rule
    can integrate the Nth order polunoial exactly

    Parameter
    ---------
    N: :The required qudrature order

    name: :The qudrature used for calculation

        Propety: :Xiao_Gimbutas; vioreanu_rokhlin
        
    NOTE: :Maximum polynomial integration accuracy supported is N=28
    ---------
    """
    if name == "Xiao_Gimbutas":
        data = np.loadtxt("QuadratureData/quad_nodes_tri_N{}.txt".format(N))
        r = data[:,0]
        s = data[:,1]
        w = data[:,2]
        return r,s,w


r,s = Nodes(4)
rq,sq,wq = quadNode(8,'Xiao_Gimbutas')

print(Vandermonde2D(4,r,s).shape)
plt.scatter(r,s,label="Intepolatory P = 4")
plt.scatter(rq,sq,label="qudrature Q = 2P")
plt.legend()
plt.show()













