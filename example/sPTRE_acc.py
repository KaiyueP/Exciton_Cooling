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
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"

def buildParser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--startidx', type=int, default=1,
                        help='start idx for total system')
    parser.add_argument('--ave_inter', type=int, default=0,
                        help='1: have term, 0: without term')
    parser.add_argument('--Ndim', type=int, default=1,
                        help='donor level (no ground)')
    parser.add_argument('--temp', type=float, default=300,
                        help='temperature (K)')
    parser.add_argument('--savefig', type=int, default=1,
                        help='0:quit,   1:save')   
    parser.add_argument('--timeLength', type=float, default=1000,
                        help='propogate time length (fs)') 
    parser.add_argument('--timeStep', type=float, default=1,
                        help='time gap (fs)') 
    return parser

parser = buildParser()
params = parser.parse_args()
shift=-0.55 #eV
ex=0
startidx=params.startidx
ave_inter=params.ave_inter
savefig=params.savefig
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

T= params.temp #K
beta= 1/kB/T
Ndim=params.Ndim ###continuous in first few states
Ndim2=Ndim**2
h=params.timeStep #time step length fs
length=params.timeLength #total evolution time fs
t_corr=np.arange(0,length,h)*1e-15 #s
Nstep=length/h #number of time steps 
h=h*1e-15#s
length=length*1e-15#s

# 1 / s
filename = '/home/kaiyuepeng/hot_carrier_cooling/GaP/In35Ga65P/w.dat'
freq = np.array([line.strip().split()[1] for line in open(filename, 'r')]).astype(float)*1e12*2.*np.pi
Natom=len(freq)
freq=freq[6:]



# J / sqrt(kg)*m
Vklq=np.load("/home/kaiyuepeng/hot_carrier_cooling/GaP/In35Ga65P/Vklq.npy")
Vklq=Vklq[6:,:(Ndim),:(Ndim)]
Vklq_diag=np.diagonal(Vklq,axis1=1,axis2=2)

#hartree to J
filename = '/home/kaiyuepeng/hot_carrier_cooling/GaP/In35Ga65P/exciton.dat'
ExcEn= np.array([line.strip().split()[1] for line in open(filename, 'r')]).astype(float)*autoev
ExcEn=ExcEn+shift
ExcEn=ExcEn*evtoj

#######end parameters#######

#reorgnization without cav
reorg0=0.5*((Vklq_diag**2).T)@(1/freq**2) #j

#coupling strength with cav
H_s = np.diag(ExcEn[:(Ndim)]) #j
V=Vklq
V_diag=np.diagonal(V,axis1=1,axis2=2)
#reorgnization with cav
reorg=0.5*((V_diag**2).T)@(1/freq**2) #j


Ee=ExcEn[:(Ndim)]-reorg


###build PTRE omega_ab###
Omega_m=np.zeros((Ndim,Ndim)) # present omega_ab
for i in range(Ndim):
    for j in range (Ndim):
        Omega_m[i,j]=(Ee[i]-Ee[j])/hbar #1/s

####average of interaction 
n_ph=1/(np.exp(beta*hbar*freq)-1)
if ave_inter==0:
    H_I_ave=np.zeros((Ndim,Ndim))
if ave_inter==1:
    H_I_ave=np.zeros((Ndim,Ndim))
    for n in range (Ndim):
        for m in range (n):
            d_a=V[:,n,n]/hbar/freq**2*np.sqrt(hbar*freq/2)
            d_b=V[:,m,m]/hbar/freq**2*np.sqrt(hbar*freq/2)
            c_ab=V[:,n,m]*np.sqrt(hbar/2/freq)
            exp_sum=np.sum((d_a-d_b)**2*(n_ph+0.5))
            sum1=np.sum(-c_ab*d_a-c_ab*d_b)
            H_I_ave[n,m]=H_I_ave[m,n]=np.exp(-exp_sum)*sum1
            # print('exp in H_I_ave',np.exp(-exp_sum))
            # print('preexp in H_I_ave in eV',sum1*jtoev)

print('finish setting up system') 
print(datetime.datetime.now()) 

#build C_abcd
###########build time-denpendent R_tensor#############  
R_ten=np.zeros((Ndim,Ndim,len(t_corr)),dtype=complex)
wt=np.einsum('i,j->ij',freq,t_corr)
cos=np.cos(wt)  
sin=np.sin(wt)
freq_sp=np.repeat(freq,len(t_corr)).reshape(len(freq),len(t_corr))
n_ph=1/(np.exp(beta*hbar*freq_sp)-1)


def R_ten_func(idx):
    a=idx[0]
    b=idx[1]
    if a!=b:
        d_ab=(V[:,a,a]-V[:,b,b])/hbar/freq**2*np.sqrt(hbar*freq/2)
        c_ab=V[:,a,b]*np.sqrt(hbar/2/freq)
        sum_ab=np.sum(c_ab*(V[:,a,a]+V[:,b,b])/hbar/freq**2*np.sqrt(hbar*freq/2))
        c_ab_sp=np.repeat(c_ab,len(t_corr)).reshape(-1,len(t_corr))
        d_ab_sp=np.repeat(d_ab,len(t_corr)).reshape(-1,len(t_corr))

        ht_ab=np.sum(d_ab_sp*c_ab_sp*(2j*sin*n_ph-cos+1j*sin),axis=0)-sum_ab
        ht_ba=np.sum(-d_ab_sp*c_ab_sp*(2j*sin*n_ph-cos+1j*sin),axis=0)-sum_ab
        gt=np.sum(c_ab_sp**2*(n_ph*2*cos+cos-1j*sin),axis=0)
        ft=np.exp(np.sum((d_ab_sp**2)*(-2*n_ph-1+2*n_ph*cos+cos-1j*sin),axis=0))
        C_ten_ab=(ht_ab**2+gt)*ft-H_I_ave[a,b]**2
        C_ten_ba=(ht_ba**2+gt)*ft-H_I_ave[a,b]**2
        ft_xy=C_ten_ab/hbar/hbar*np.exp(1j*Omega_m[a,b]*t_corr)
        ft_yx=C_ten_ba/hbar/hbar*np.exp(1j*Omega_m[b,a]*t_corr)
        R_xy=integrate.cumtrapz(ft_xy, t_corr, initial=0)
        R_yx=integrate.cumtrapz(ft_yx, t_corr, initial=0)
        return (idx,R_xy,R_yx)
    else:
        return (idx,0,0)
    
idx_lst=np.array([(x,y) for x in range (Ndim) for y in range (x,Ndim)])
pool=mp.Pool(16)
print('cpu number 16')
Rst=pool.map(R_ten_func,idx_lst)
for kk in range (len(Rst)):
    ((x,y),tmpxy,tmpyx)=Rst[kk]
    R_ten[x,y]=tmpxy
    R_ten[y,x]=tmpyx
pool.close()
pool.join()    


print('finish R_ten') 
print(datetime.datetime.now()) 

R_ten_con=np.conj(R_ten)
F_ten=np.zeros((Ndim,Ndim,len(t_corr)),dtype=complex)
W_ten=np.zeros((Ndim,Ndim,len(t_corr)),dtype=complex)
J_ten=np.zeros((Ndim,Ndim,len(t_corr)),dtype=complex)
W_ten=R_ten+R_ten_con
for tidx in range (len(t_corr)):
    F_ten[:,:,tidx]=np.diag(np.einsum('ij->i',R_ten[:,:,tidx]))
    J_ten[:,:,tidx]=np.diag(np.einsum('ij->i',R_ten_con[:,:,tidx]))

def func(sigma_h,tidx):
    Rtot_rho=np.zeros((Ndim,Ndim),dtype=complex)
    R_S=-1j/hbar*((np.diag(Ee)+H_I_ave)@sigma_h-sigma_h@(np.diag(Ee)+H_I_ave))
    R_B=-F_ten[:,:,tidx]@sigma_h-sigma_h@J_ten[:,:,tidx]+np.diag((W_ten[:,:,tidx].T)@np.diagonal(sigma_h))
    Rtot_rho=R_B+R_S
    return Rtot_rho



###initial basis bare state####)
sigma=np.zeros((int(Nstep),Ndim,Ndim),dtype=complex)
sigma[0,:,:]=np.load('initial_dis.npy')

# if startidx>=0:
#     sigma[0,startidx,startidx]=1 #occupy the specified state


sigma[1]=func(sigma[0],0)*h+sigma[0]
for i in range(int(Nstep)-2):
    k1=func(sigma[i],i)
    k2=func(sigma[i]+h*k1,i+1)
    k3=func(sigma[i]+h*k2,i+1)
    k4=func(sigma[i]+h*2*k3,i+2)
    sigma[i+2]=sigma[i]+1/6*(k1+2*k2+2*k3+k4)*2*h
    if (np.diagonal(sigma[i+1].reshape(Ndim,Ndim))>1).any():
        print('unphysical population')

# ###########approximation ghere#############
# for i in range(int(Nstep)-1):
#     k1=func(sigma[i],i)
#     k2=func(sigma[i]+h*k1/2,i)
#     k3=func(sigma[i]+h*k2/2,i)
#     k4=func(sigma[i]+h*k3,i)
#     sigma[i+1]=sigma[i]+1/6*(k1+2*k2+2*k3+k4)*h
#     if (np.diagonal(sigma[i+1].reshape(Ndim,Ndim))>1).any():
#         print('unphysical population')
#         break

plt.figure()
plt.title("Total energy")
E_tot=np.zeros(len(t_corr))
for ti in range (len(t_corr)):
    E_tot[ti]=np.sum(np.diagonal(sigma,axis1=1,axis2=2)[ti,:]*ExcEn[:Ndim])*jtoev #ev
plt.plot(t_corr*1e15,E_tot)
plt.xlim(0,length*1e15)
plt.xlabel('t (fs)')
plt.ylabel('Total_energy (eV)')
plt.savefig('totE_sPTRE.png',dpi=300)

plt.figure()
#plt.ylim(0,1)
plt.xlim(0,length*1e15)
plt.xlabel('t (fs)') 
plt.ylabel('population')   
plt.title("Reduced density matrix in original basis") 
for i in range (Ndim):
    plt.plot(t_corr*1e15,sigma[:,i,i].real,label='sigma_%i'%(i))  
plt.legend(loc='upper right')
if savefig==1:
    plt.savefig('sigma_orin_FRET.png',dpi=300)
plt.show()



with open ('E_tot_sPTRE.npy','wb') as f:
    np.save(f,E_tot)  #eV

with open ('Ee.npy','wb') as f:
    np.save(f,Ee)
  

with open ('pop.npy','wb') as f:
    np.save(f,sigma)

with open ('t.npy','wb') as f:
    np.save(f,t_corr)
    

with open ('R_ten.npy','wb') as f:
    np.save(f,R_ten)

print('finish calculation')
print(datetime.datetime.now())
print('\n ana calculation\n')
