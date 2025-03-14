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

pop=np.load('pop.npy') #eV
pop_d=np.diagonal(pop,axis1=1,axis2=2)
t=np.load('t.npy')*1e15 #fs


Ee=np.load('Ee.npy')*jtoev #eV
Ndim=len(Ee)

filename = 'exciton.dat'
ExcEn= np.array([line.strip().split()[1] for line in open(filename, 'r')]).astype(float)*autoev
ExcEn=ExcEn[:Ndim] #eV

filename = 'OS.dat'
mu= np.array([line.strip().split()[1] for line in open(filename, 'r')]).astype(float)
mu=mu[:Ndim] #eV

def delta(x,a,width):
    return 1/np.sqrt(2*np.pi*width**2)*np.exp(-(x-a)**2/2/width**2)


###########################plot pop###################
tidx=np.arange(0,1010,10)
tidx[-1]=999
pop_d1=pop_d[tidx,:]
t1=t[tidx]
x_grid=np.arange(2.4,2.9,0.001)
width=0.008
y=np.zeros((len(t1),len(x_grid)))
fig, ax = plt.subplots(figsize=(10,8))
for i in range (len(t1)):
    for j in range (Ndim):
        y[i]=y[i]+pop_d1[i,j]*delta(x_grid,ExcEn[j],width)
    plt.plot(x_grid,y[i]*10+t1[i],label='%i fs'%(t1[i]),color=plt.cm.viridis(i/len(t1)))

plt.xlabel('Energy(eV)',fontsize=24,fontweight="bold")
plt.ylabel('Time (fs)',fontsize=24,fontweight="bold")
plt.xlim(2.4,2.85)
plt.yticks(np.arange(0,1100,100))
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.savefig('pop_t.png',dpi=300,bbox_inches='tight')
plt.show()

###########################plot abs###################
tidx=np.arange(0,220,20)
pop_d2=pop_d[tidx,:]
t2=t[tidx]
x_grid=np.arange(2.4,2.9,0.001)
width=0.008
abs=np.zeros((len(t2),len(x_grid)))
fig, ax = plt.subplots(figsize=(10,8))
for i in range (len(t2)):
    for j in range (Ndim):
        abs[i]=abs[i]+pop_d2[i,j]*delta(x_grid,ExcEn[j],width)*mu[j]**2*x_grid
    plt.plot(x_grid,abs[i]/max(abs[0])*100,label='%i fs'%(t2[i]),color=plt.cm.viridis(i/len(t2)))

plt.xlabel('Energy (eV)', fontsize=20)
plt.ylabel('Change in absorption', fontsize=20)
plt.legend(fontsize=16)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.xlim(2.4,2.85)
plt.savefig('abs_t_200fs.png',dpi=300,bbox_inches='tight')
plt.show()

tidx=np.arange(0,1000,100)
pop_d2=pop_d[tidx,:]
t2=t[tidx]
x_grid=np.arange(2.4,2.9,0.001)
width=0.008
abs=np.zeros((len(t2),len(x_grid)))
fig, ax = plt.subplots(figsize=(10,8))
for i in range (len(t2)):
    for j in range (Ndim):
        abs[i]=abs[i]+pop_d2[i,j]*delta(x_grid,ExcEn[j],width)*mu[j]**2*x_grid
    plt.plot(x_grid,abs[i]/max(abs[0])*100,label='%i fs'%(t2[i]),color=plt.cm.viridis(i/len(t2)))

plt.xlabel('Energy (eV)', fontsize=20)
plt.ylabel('Change in absorption', fontsize=20)
plt.legend(fontsize=16)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.xlim(2.4,2.85)
plt.savefig('abs_t_1000fs.png',dpi=300,bbox_inches='tight')
plt.show()
