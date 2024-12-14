import os,sys
import numpy as np
#from gacodefuncs import *
from cgyro.data import cgyrodata
import matplotlib.pyplot as plt
from matplotlib import rc
from pydmd import DMD
import matplotlib.gridspec as gridspec
from dmd_util import *

#---------------------------------------------------------------------------
# INPUTS
mydir = sys.argv[1]+'/'
nmode = 3

tol = 0.2

# time downsample
k = 20

# max time
tmax = 0

# theta downsample
l = 1

# SVD rank to perform DMD 
svd_rank = 0

# observables
ostr = ['phi','apar','bpar']

#---------------------------------------------------------------------------
# COLLECT DATA
sim = cgyrodata(mydir)
sim.getbigfield()

t = sim.t
n_radial  = sim.n_radial  
n_theta   = sim.theta_plot
n_species = sim.n_species
n_time = len(t)

# balloonings-space potentials
ovec = {}
y = sim.kxky_phi[0,:,:,0,:]+1j*sim.kxky_phi[1,:,:,0,:]
ovec['phi'],a0 = map1d(y,sim.q)
y = sim.kxky_apar[0,:,:,0,:]+1j*sim.kxky_apar[1,:,:,0,:]
ovec['apar'],a0 = map1d(y,sim.q)
y = sim.kxky_bpar[0,:,:,0,:]+1j*sim.kxky_bpar[1,:,:,0,:]
ovec['bpar'],a0 = map1d(y,sim.q)

# step for DMD
dt = k*(t[1]-t[0])

if tmax > 0:
    imax = int(tmax/t[-1]*len(t))
else:
    imax = len(t)
    
#---------------------------------------------------------------------------
# RUN DMD
dmd = DMD(svd_rank=svd_rank,exact=True)

evec = {}
for x in ostr:
    print(x)
    dmd.fit(ovec[x][::l,:imax:k])
    evec[x] = 1j*np.log(dmd.eigs)/dt

#---------------------------------------------------------------------------
# PLOTTING

rc('font',size=25)
rc('text',usetex=True)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)

ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\gamma$")
ax.grid(which="both",ls=":")
ax.grid(which="major",ls=":")

# symbols
args = {}
args['phi']  = {'color':'r','marker':'o','alpha':0.4}
args['apar'] = {'color':'b','marker':'s','facecolors':'none'}
args['bpar'] = {'color':'k','marker':'+'}

for x in ostr:
    ax.scatter(evec[x].real,evec[x].imag,s=60,**args[x])

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

ax.set_xlim([-1.2,1.2])
ax.set_ylim([-0.2,0.5])

plt.tight_layout()
plt.show()

