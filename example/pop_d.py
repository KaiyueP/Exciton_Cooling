import numpy as np
pop=np.load('pop.npy') #eV
pop_d=np.diagonal(pop,axis1=1,axis2=2)
with open ('pop_d.npy','wb') as f:
    np.save(f,pop_d)