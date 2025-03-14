import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import rcParams

rcParams['font.weight'] = 'bold'

nmtoau=18.8972598858 #1nm in Bohr
autoev= 27.2114079527 # 1 hartree in ev
autoj = 4.3597447222071e-18 # 1 Hartree in J
jtoev=6.2415091e18 #1j in eV
evtoj=1.602176565e-19 #1eV in J
hbar= 1.054571817e-34 #J⋅s
kB=1.380649e-23 #J/K
T= 300 #K
beta= 1/kB/T/jtoev #1/ev

E_tot=np.load('E_tot_sPTRE.npy') #eV
t=np.load('t.npy')*1e15 #fs
Ee=np.load('Ee.npy')*jtoev #eV
Ndim=len(Ee)
filename = 'exciton.dat'
ExcEn= np.array([line.strip().split()[1] for line in open(filename, 'r')]).astype(float)*autoev
ExcEn=ExcEn[:Ndim] #eV

def func(x,k,q,a,b):
    return a*np.exp(-k*x)+b
rho_eq=np.exp(-beta*Ee)/np.sum(np.exp(-beta*Ee))
E_eq=np.sum(rho_eq*ExcEn) #eV
print('Ndim',Ndim)
# print('Ee',Ee)
# print('rho_eq',rho_eq)
print('E_eq',E_eq)
para, pcov = curve_fit(func,t ,E_tot-E_eq)
print('parameter',para[0],para[1],para[2],para[3])
fig, ax = plt.subplots(figsize=(10,8))
plt.plot(t,E_tot-E_eq)
plt.plot(t,func(t,para[0],para[1],para[2],para[3]),linestyle='--')
#plt.title('Energy loss for In35Ga65P',fontsize=24)
plt.xlabel('Time (fs)',fontsize=24, fontweight="bold")
plt.ylabel('Energy above eq (eV)',fontsize=24, fontweight="bold")
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.savefig('E_loss.png',dpi=300,bbox_inches='tight')
plt.show()
