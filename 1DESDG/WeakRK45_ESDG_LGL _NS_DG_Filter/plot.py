import DG as dg
import numpy as np
from matplotlib import pyplot as plt
import Operator as pcD
import Entropy as SS
class plot(pcD.preComp):
    def __init__(self,path,rho,rhou,rhoE,Tf,EPS = 20,N = pcD.NP,K = pcD.Ke,timeData = []):
        super().__init__(N,K)
        self.path = path
        self.N = N
        self.K = K
        self.Tf = Tf
        # Change nodal to modal
        self.rho = rho
        self.rhou = rhou
        self.rhoE = rhoE
        self.rho, self.u, self.p = self.P @ SS.conservativeToPrimitive(rho,rhou,rhoE)
        self.r = np.linspace(-1,1,EPS)
        self.Vpl = dg.Vandermonde1D(N,self.r)
        self.Xpl = dg.pNodes(0,0,self.xRef,self.r)
    def primitive(self):
        plt.figure(dpi=300)
        # rho = self.Vpl@self.rho
        u   = self.Vpl@self.u
        p   = self.Vpl@self.p
        pRef = np.linspace(-5,5,pcD.Ke)
        import pandas as pd
        pd.DataFrame(self.rho.flatten("F")).to_csv('rhoEX.csv')
        pd.DataFrame(self.Xpl.flatten("F")).to_csv('XpEX.csv')
        rhoavg = np.sum(self.rho,axis=0)/(pcD.NP+1)
        # plt.plot(self.Xpl.flatten("F"), rho.flatten("F"),label="rho")
        # plt.plot(self.Xpl.flatten("F"), np.abs(u).flatten("F"),label = "u")
        plt.plot(pRef, rhoavg.flatten("F"), label="rho ESDG")
        plt.legend()
        plt.show()
        # plt.savefig(self.path + "Shu_w=5_F_0.995h_N5K100.png")
    def enrgy(self):
        pass
    def polySpec(self):
        plt.figure(dpi = 300)
        ax = plt.axes(projection='3d')
        x = np.linspace(-5,5,self.K)
        rho = self.rho
        for i in range(self.N+1):
            ax.plot3D(x,[i for j in range(self.K)],rho[i,:])
        ax.set_xlabel('Element Number')
        ax.set_ylabel('The order x[n]')
        ax.set_zlabel('Magnitude')
        ax.set_title("The Density spectra of Smooth problem")
        plt.savefig(self.path + "DensitySPec.png")


        plt.figure(dpi = 300)
        ax = plt.axes(projection='3d')
        x = np.linspace(-5,5,self.K)
        p = self.p
        for i in range(self.N+1):
            ax.plot3D(x,[i for j in range(self.K)],p[i,:])
        ax.set_xlabel('Element Number')
        ax.set_ylabel('The order x[n]')
        ax.set_zlabel('Magnitude')
        ax.set_title("The Pressure spectra of Smooth problem")
        plt.savefig(self.path + "PressureSPec.png")


        plt.figure(dpi = 300)
        ax = plt.axes(projection='3d')
        x = np.linspace(-5,5,self.K)
        u = self.u
        for i in range(self.N+1):
            ax.plot3D(x,[i for j in range(self.K)],u[i,:])
        ax.set_xlabel('Element Number')
        ax.set_ylabel('The order x[n]')
        ax.set_zlabel('Magnitude')
        ax.set_title("The Velocity spectra of Smooth problem")
        plt.savefig(self.path + "USPec.png")
    
        


