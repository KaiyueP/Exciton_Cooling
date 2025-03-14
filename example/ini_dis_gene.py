import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
from scipy import integrate
import sys
from scipy.special import erf
import multiprocessing as mp
from functools import partial
import os
print('start calculation')
print(datetime.datetime.now())

####################parameter to generate##############
Ndim=350
shift=-0.55 #eV
ini_E=2.10 #eV
devi=0.003 #eV
#######################################################

def gene_dis(x,fx,ini_E,devia):
    return fx*np.exp(-(x-ini_E)**2/2/devia**2)


nmtoau=18.8972598858 #1nm in Bohr
autoev= 27.2114079527 # 1 hartree in ev
autoj = 4.3597447222071e-18 # 1 Hartree in J
jtoev=6.2415091e18 #1j in eV
evtoj=1.602176565e-19 #1eV in J
hbar= 1.054571817e-34 #J⋅s
kB=1.380649e-23 #J/K
C_light = 3e8  # m/s
C_au = C_light/2.18769126364e6  # AU
autos = 2.4188843265857e-17
Ndim2=Ndim**2

#hartree to eV
filename = 'exciton.dat'
ExcEn= np.array([line.strip().split()[1] for line in open(filename, 'r')]).astype(float)*autoev
ExcEn=ExcEn[:Ndim]+shift

#oscillator strength
filename = 'OS.dat'
osci_stren= np.array([line.strip().split()[3] for line in open(filename, 'r')]).astype(float)
osci_stren=osci_stren[:Ndim]

sigma_0=np.zeros(Ndim)
for i in range (len(sigma_0)):
    sigma_0[i]=gene_dis(ExcEn[i],osci_stren[i],ini_E,devi)
sigma_0=sigma_0/np.sum(sigma_0)

plt.figure()
plt.scatter(ExcEn, sigma_0)
plt.title('%.2f eV 65Ga'%ini_E)
plt.savefig('pop_ini.png')
plt.show()

with open ('initial_dis.npy', 'wb') as f:
    np.save (f,np.diag(sigma_0))

