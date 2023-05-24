import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la
import DG as dg
from sympy import fourier_series, pi
from sympy import *

I,J,x = symbols('I,J,x')
N = 10
str = 20
bot = 1
def Init(xP):
    u = []
    for i in range(len(xP)):
        if xP[i] < 0:
            u.append(I+J)
        else:
            u.append(I)
    return u
pPoints = np.linspace(-1,1,1000)
xref = np.linspace(-1,1,2)
r,w = dg.JacobiGL(0,0,N)
V1d = dg.Vandermonde1D(N,r)
Vp = dg.Vandermonde1D(N,pPoints)

p = dg.GLP(0,0,N,xref).flatten("F")
u = Init(p)
umold = la.inv(V1d)@u
# umold[1]*=0.99
for i in range(len(umold)):
    print(umold[i])
f = lambda x: umold[0] + umold[1]*x + umold[2]*x**2 + umold[3]*x**3
s = fourier_series(f(x), (x, -1, 1))
print(s)
