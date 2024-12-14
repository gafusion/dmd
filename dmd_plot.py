import os,sys
import numpy as np
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

ovec = {}
ovec['phi']  = sim.kxky_phi[0,:,:,0,:] +1j*sim.kxky_phi[1,:,:,0,:]
ovec['apar'] = sim.kxky_apar[0,:,:,0,:]+1j*sim.kxky_apar[1,:,:,0,:]
ovec['bpar'] = sim.kxky_bpar[0,:,:,0,:]+1j*sim.kxky_bpar[1,:,:,0,:]
    
#---------------------------------------------------------------------------
# PLOTTING

rc('font',size=25)
rc('text',usetex=True)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)

ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\phi$")
ax.grid(which="both",ls=":")
ax.grid(which="major",ls=":")

# plot phi
y,anorm = map1d(ovec['phi'],sim.q)
ax.plot(np.real(y[:,-1]/anorm[-1]))

# plot apar
y,dummy = map1d(ovec['apar'],sim.q)
ax.plot(np.real(y[:,-1]/anorm[-1]))

plt.tight_layout()
plt.show()
