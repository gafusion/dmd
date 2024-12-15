import os
import numpy as np
#from gacodefuncs import *
from cgyro.data import cgyrodata
import matplotlib.pyplot as plt
from matplotlib import rc
from pydmd import DMD
import matplotlib.gridspec as gridspec
from dmd_util import *


#---------------------------------------------------------------------------
# DMD INPUTS
mydir='/path/to/cgyro/output/data/'
nmode = 3

tol = 0.2

# time downsample
# depends on PRINT_STEP
# note: optimal DMD time step = (0.25 -- 1) to resolve omega/ gamma
# optimal DMD time step = (0.1 -- 0.25) to separate properly real and imag parts of eigenfunctions
k = 500

# max time
tmax = 0

# theta downsample
l = 1 

# SVD rank to perform DMD 
svd_rank = 0

# observables
# action: add dens
ostr = ['phi','apar','bpar']

#---------------------------------------------------------------------------
# COLLECT DATA
sim = cgyrodata(mydir)
sim.getbigfield()

t = sim.t
theta = sim.theta
thetab = sim.thetab
n_radial  = sim.n_radial  
n_theta   = sim.theta_plot
n_species = sim.n_species

#ovec = {}
#ovec['phi']  = sim.kxky_phi[0,:,:,0,:] +1j*sim.kxky_phi[1,:,:,0,:]
#ovec['apar'] = sim.kxky_apar[0,:,:,0,:]+1j*sim.kxky_apar[1,:,:,0,:]
#ovec['bpar'] = sim.kxky_bpar[0,:,:,0,:]+1j*sim.kxky_bpar[1,:,:,0,:]
# balloonings-space potentials
ovec = {}
y = sim.kxky_phi[0,:,:,0,:]+1j*sim.kxky_phi[1,:,:,0,:]
ovec['phi'],a0 = map1d(y,sim.q)
y = sim.kxky_apar[0,:,:,0,:]+1j*sim.kxky_apar[1,:,:,0,:]
ovec['apar'],a0 = map1d(y,sim.q)
y = sim.kxky_bpar[0,:,:,0,:]+1j*sim.kxky_bpar[1,:,:,0,:]
ovec['bpar'],a0 = map1d(y,sim.q)



# step for DMD5/2
#dt = k*(t[1]-t[0])
delt = t[::k][1]-t[::k][0]

print ("DMD delt = ", delt)

#---------------------------------------------------------------------------
# RUN DMD
dmd = DMD(svd_rank=svd_rank,exact=True)


evec = {}
for x in ['phi','apar','bpar']:
    #dmd.fit(ovec[x][:,::l,::k])
    dmd.fit(ovec[x][:,::k])
    evec[x] = np.log(dmd.eigs)/(-complex(0,1)*delt) #1j*np.log(dmd.eigs)/dt

    print (evec[x])
    print (x)

#---------------------------------------------------------------------------
# determine most unstable modes (zvec)

zvec = np.zeros([3,nmode],dtype=complex)
for i,x in enumerate(ostr):
    z  = evec[x] 
    zi = z.imag
    k = np.flip(np.argsort(zi))
    zvec[i,:] = z[k[:nmode]]

# compute errors
err = np.zeros(3)
for i in range(nmode):
    err[0] = abs(zvec[0,i]-zvec[1,i])
    err[1] = abs(zvec[0,i]-zvec[2,i])
    err[2] = abs(zvec[1,i]-zvec[2,i])
    etot = sum(err)
    eave = np.average(zvec[:,i])
    if etot < tol:
        print("gamma = {:.3f} omega = {:+.3f} | err = {:.3e}".format(eave.imag,eave.real,etot))

#---------------------------------------------------------------------------
# PLOTTING

plt.figure()
plt.plot(thetab, ovec['phi'][:,-1].real, "-o")
plt.plot(thetab, ovec['phi'][:,-1].imag, "-o")

plt.figure()
plt.plot(thetab, ovec['apar'][:,-1].real, "-o")
plt.plot(thetab, ovec['apar'][:,-1].imag, "-o")
#plt.show()


fig = plt.figure()
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])


ax1.axvline(0., linestyle="dashed", color="k", linewidth=0.5)
ax1.axhline(0., linestyle="dashed", color="k", linewidth=0.5)

#for x in ['phi','apar','bpar']:

    #ax1.plot(evec[x].real, evec[x].imag, "o", color="tab:blue")

ax1.plot(evec['phi'].real, evec['phi'].imag, 'o', color="tab:blue")
ax1.plot(evec['apar'].real, evec['apar'].imag, 'o', color="red", mfc="none")
ax1.plot(evec['bpar'].real, evec['bpar'].imag, 's', color="tab:green", mfc="none", markersize=9.5)

ax1.set_xlabel(r"realEigs.real, i.e. $\omega$", fontsize=15)
ax1.set_ylabel(r"realEigs.imag, i.e. $\gamma$", fontsize=15)
ax1.tick_params(labelsize=15)

plt.show()

    
