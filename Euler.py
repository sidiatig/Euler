# Copyright (C) 2017 Simon Legrand
from __future__ import print_function

import sys
import numpy as np
import density
import plots
import inout

param = dict(filename_Rho0 = "Inputs/CasTest01/ECMWF_20080901_060000_Latmin-36_Lonmin30_Latmax-16_Lonmax50.txt",
			 filename_Rho1 = "Inputs/CasTest01/ECMWF_20080901_120000_Latmin-36_Lonmin30_Latmax-16_Lonmax50.txt",
			 epsilon = 2e-3,
			 nFrames = 3,				# Number of interpolated frames (including marginals)
			 lambda0 = np.inf,			# lambda \in [0, +inf[, no mass creation if np.inf
			 lambda1 = np.inf,
			 name = "ECMWF_20080901", 	# For the saving
			 save_interp = True,
			 save_w2 = True,
			 save_moments = True)


lambda0 = param['lambda0']
lambda1 = param['lambda1']

#### Adapt the exported file name to the parameters ####
param['name'] = param['name']+'_eps='+str(param['epsilon'])
if(lambda0 != np.inf):
	param['name'] += "_l0="+str(lambda0)
	if(lambda1 != np.inf):
		param['name'] += "_l1="+str(lambda1)

# For development
#box = [0.,1.5,0.,1.]
#Nx = 150
#Ny = 100
#x,y = np.linspace(box[0], box[1], Nx), np.linspace(box[2], box[3], Ny)
#Rho0 = density.Density([x,y],inout.gaussian(x,y,0.25,0.3,0.1,0.1))
#Rho1 = density.Density([x,y],inout.gaussian(x,y,1.25,0.7,0.1,0.1))


#### Densities creation ####
Rho0 = density.Density.from_file(param['filename_Rho0'], rescale=True)
Rho1 = density.Density.from_file(param['filename_Rho1'], rescale=True)

if(lambda0==np.inf and lambda1==np.inf):
	# For balanced transport, both densities are normalized.
	Rho0.divide_mass(Rho0.mass())
	Rho1.divide_mass(Rho1.mass())
else:
	# For unbalanced transport, mass ratio is conserved.
	m0 = Rho0.mass()
	Rho0.divide_mass(m0)
	Rho1.divide_mass(m0)
	

#### Interpolation creation ####
Interp = density.Interpolant(Rho0, Rho1, param)
Interp.run_frames()
Interp.plot_frames()
Interp.plot_wasserstein_distance()
Interp.run_moments()
Interp.plot_moments()
Interp.save()
