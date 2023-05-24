import Operator as pcD
import DG as DG
import Entropy as SS
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
import viscos as vis
import scipy
from scipy.integrate import simps
import plot as dgp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("runTime",help="Final time",type=float)
args = parser.parse_args()
tf = args.runTime



gamma = 1.4
Qm = pcD.nQspace.Qm
nx = np.ones((2,pcD.Ke))
nx[0] = -1
invM = pcD.nQspace.invM
Pq = pcD.nQspace.P
Vqf = pcD.nQspace.Vqf
R = pcD.nQspace.R
Bf = pcD.nQspace.Bf
Vf = pcD.nQspace.Vf
Vfpq = Vf@Pq
rk4a = np.array([ 0.0 ,
-567301805773.0/1357537059087.0 ,
-2404267990393.0/2016746695238.0 ,
-3550918686646.0/2091501179385.0 ,
-1275806237668.0/842570457699.0])

rk4b = [ 1432997174477.0/9575080441755.0,
5161836677717.0/13612068292357.0,
1720146321549.0/2090206949498.0,
3134564353537.0/4481467310338.0,
2277821191437.0/14882151754819.0]

rk4c = [ 0.0,
1432997174477.0/9575080441755.0,
2526269341429.0/6820363962896.0,
2006345519317.0/3224310063776.0,
2802321613138.0/2924317926251.0]

# @ return the flux differencing term
def fluxDiff(rho, rhou, rhoE):
    FS = SS.Fs(rho, rhou, rhoE)
    rhoDfs  = np.zeros((pcD.NQ+1, pcD.Ke))
    rhouDfs = np.zeros((pcD.NQ+1, pcD.Ke))
    rhoEDfs = np.zeros((pcD.NQ+1, pcD.Ke))
    for k in range(pcD.Ke):
        rhoDfs [:,k] = (Pq.T)@(Vqf.T)@(((Qm-Qm.T)*FS[:,:,k,0]).sum(axis=1))
        rhouDfs[:,k] = (Pq.T)@(Vqf.T)@(((Qm-Qm.T)*FS[:,:,k,1]).sum(axis=1))
        rhoEDfs[:,k] = (Pq.T)@(Vqf.T)@(((Qm-Qm.T)*FS[:,:,k,2]).sum(axis=1))

    return rhoDfs,rhouDfs,rhoEDfs

def SAT(rho, rhou, rhoE, EC = SS.chandFlux, Disp = SS.LaxD):
    rhoL, rhoR = SS.LRface(rho)
    rhouL, rhouR = SS.LRface(rhou)
    rhoEL, rhoER = SS.LRface(rhoE)
    rhoLg, rhoRg, rhouLg, rhouRg, rhoELg, rhoERg = SS.BC(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL)
    rhoLg, uLg, pLg = SS.conservativeToPrimitive(rhoLg, rhouLg, rhoELg)
    rhoRg, uRg, pRg = SS.conservativeToPrimitive(rhoRg, rhouRg, rhoERg)
    rhoL, uL, pL = SS.conservativeToPrimitive(rhoL, rhouL, rhoEL)
    rhoR, uR, pR = SS.conservativeToPrimitive(rhoR, rhouR, rhoER)
    
    fs1 = np.zeros((2,pcD.Ke))
    fs2 = np.zeros((2,pcD.Ke))
    fs3 = np.zeros((2,pcD.Ke))

    for k in range(1,pcD.Ke-1):
        fs1[0,k], fs2[0,k], fs3[0,k] = EC(rhoR[k-1],rhoL[k],uR[k-1],uL[k],pR[k-1],pL[k])
        fs1[1,k], fs2[1,k], fs3[1,k] = EC(rhoL[k+1],rhoR[k],uL[k+1],uR[k],pL[k+1],pR[k])
    fs1[0,0], fs2[0,0], fs3[0,0] = EC(rhoL[0], rhoLg, uL[0], uLg, pL[0], pLg)
    fs1[1,0], fs2[1,0], fs3[1,0] = EC(rhoR[0], rhoL[1], uR[0], uL[1], pR[0], pL[1])

    fs1[-1,-1], fs2[-1,-1], fs3[-1,-1] = EC(rhoRg,rhoR[-1],uRg,uR[-1],pRg,pR[-1])
    fs1[0,-1], fs2[0,-1], fs3[0,-1] = EC(rhoL[-1], rhoR[-2],uL[-1],uR[-2],pL[-1],pR[-2])

    lax1,lax2,lax3 = Disp(rhoR,rhoL,rhouR,rhouL,rhoER,rhoEL)


    fs1 = fs1 + lax1
    fs2 = fs2 + lax2
    fs3 = fs3 + lax3

    SAT1 = (R.T@Bf@(nx*(fs1)))
    SAT2 = (R.T@Bf@(nx*(fs2)))
    SAT3 = (R.T@Bf@(nx*(fs3)))


    return SAT1, SAT2, SAT3

def rhs(rho, rhou, rhoE, EC = SS.chandFlux, Disp = SS.LaxD):

    rhoDfs, rhouDfs, rhoEDfs = fluxDiff(rho, rhou, rhoE)
    sat1, sat2, sat3 = SAT(rho, rhou, rhoE, EC, Disp)
    rhs1 = -invM@(rhoDfs  + sat1)
    rhs2 = -invM@(rhouDfs + sat2)
    rhs3 = -invM@(rhoEDfs + sat3)

    return pcD.overInt.Pq@rhs1,pcD.overInt.Pq@rhs2,pcD.overInt.Pq@rhs3
ener = []
pRef = np.linspace(-5,5,pcD.Ke)
def solver(rho,u,p,Tf):
    t = 0
    rho,rhou,rhoE = SS.primitiveToConservative(rho,u,p)
    resrho, resrhou, resrhoE = np.zeros(rho.shape),np.zeros(rho.shape),np.zeros(rho.shape)
    while t < Tf:
        print(t)
        # print(SS.damperHelp(rho,rhou,rhoE))
        rho, u, p = SS.conservativeToPrimitive(rho, rhou, rhoE)
        cvel = np.sqrt(pcD.gamma*p/rho)
        dt = pcD.CFL*np.min(pcD.nQspace.h/(np.abs(u)+cvel) + vis.mu/(pcD.nPspace.h))
        if t+dt > Tf:
            dt = Tf-t 
        for iter in range(5):
            
            # rho = SS.filtering(rho)
            # rhou = SS.filtering(rhou)
            # rhoE = SS.filtering(rhoE)
            # v1,v2,v3 = SS.conservativeToPrimitive(rho,rhou,rhoE)
            # v1 = SS.filtering(v1)
            # v2 = SS.filtering(v2)
            # v3 = SS.filtering(v3)
            # rho,rhou,rhoE = SS.primitiveToConservative(v1,v2,v3)
            rhsrho, rhsrhou, rhsrhoE = rhs(rho,rhou,rhoE)    
            resrho = rk4a[iter]*resrho+dt*rhsrho
            rho += rk4b[iter]*resrho

            resrhou = rk4a[iter]*resrhou+dt*rhsrhou
            rhou += rk4b[iter]*resrhou

            resrhoE = rk4a[iter]*resrhoE+dt*rhsrhoE
            rhoE += rk4b[iter]*resrhoE

        t = t+dt
    return rho, rhou, rhoE
rho, rhou, rhoE = solver(pcD.rho, pcD.u, pcD.p, tf)
rho,u,p = SS.conservativeToPrimitive(rho,rhou,rhoE)
# v1,v2,v3 = SS.conservativeToEntropy(rho,rhou,rhoE)
# rhoavg = np.sum(rho,axis=0)/(pcD.NP+1)
# uavg = np.sum(np.abs(u),axis=0)/(pcD.NP+1)
# pavg = np.sum(p,axis=0)/(pcD.NP+1)
# pRef = np.linspace(-5,5,pcD.Ke)
path = '/home/zirui/Desktop/Main/hybSBP_ESDG/ESDG_testing/figs/ESDGK150N3(F|NF)/'

import pandas as pd
pd.DataFrame(rho.flatten("F")).to_csv('rhoEX.csv')
pd.DataFrame(pcD.nPspace.xP.flatten("F")).to_csv('XpES.csv')
# ener = np.array(ener)
# pd.DataFrame(rho).to_csv(path+'A1.csv')
# pd.DataFrame(rhou).to_csv(path+'A2.csv')
# pd.DataFrame(rhoE).to_csv(path+'A3.csv')
# plt.figure(dpi=300)
# plt.plot(pRef, rhoavg,label="rho")
# plt.plot(pRef, uavg,label = "u")
# plt.plot(pRef, pavg,label="p")
# pd.DataFrame(ener).to_csv(path+'ESDG.csv')
# tt = np.linspace(0,1.8,len(ener))
# plt.plot(tt,ener)
# plt.legend()
# plt.show()
# # s = np.log(p*rho**(-gamma))
# plt.figure(dpi=300)
plt.plot(pcD.nPspace.xP.flatten("F"), rho.flatten("F"),label="rho")
# plt.plot(pcD.nPspace.xP.flatten("F"), np.abs(u).flatten("F"),label = "u")
# plt.plot(pcD.nPspace.xP.flatten("F"), p.flatten("F"), label="p")
# rho = SS.filtering(rho)
# rhou = SS.filtering(rhou)
# rhoE = SS.filtering(rhoE)
# rho,u,p = SS.conservativeToPrimitive(rho,rhou,rhoE)
# Node = (pcD.NP+1)*pcD.Ke
# plottingGrid = DG.pNodes(0,0,pcD.NP,pcD.nPspace.xRef)
# Vpl = pcD.nPspace.Vpl
# plotPres = Vpl@pcD.nPspace.P@p
# pdata = plotPres.flatten("F")
# import scipy.fft as fft
# freq = np.linspace(0.0,45,int(Node/2))
# freqdata = fft.fft(pdata)
# y = 2/Node * np.abs (freqdata [0:int(Node/2)])
# plt.figure(dpi=200)
# markerline, stemlines, baseline = plt.stem(freq[0:-1:5],y[0:-1:5])
# plt.setp(markerline, linewidth=0.5)
# # plt.scatter((pcD.D@rhou).flatten("F"),(M@rhoDfs).flatten("F"))
# plt.xlabel("Frequency Hz")
# plt.ylabel("Magnitude")
# plt.legend()
# plt.show()
# path = '/home/zirui/Desktop/Main/hybSBP_ESDG/ESDG_testing/figs/ploySpec/'
# plots = dgp.plot(path,rho,rhou,rhoE,tf)
# plots.primitive()
# plots.polySpec()
plt.legend()
plt.show()
print("done")



    






