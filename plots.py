# Copyright (C) 2017 Simon Legrand
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


def plot_density(X,Rho,title=None,Ampl=None, filename=None):
	fig,ax = plt.subplots()
	x = X[0]
	y = X[1]
	if Ampl is None:
		im = ax.imshow(Rho,extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),origin='lower')
	else:
		im = ax.imshow(Rho,extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),origin='lower',vmin=Ampl[0],vmax=Ampl[1])
	if title:
		ax.set_title(title)
	plt.colorbar(im)
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename+'.png')
		

def plot_wasserstein_distance(W2):
	t = np.linspace(0,1,len(W2))
	plt.plot(t,W2)
	plt.show()
	

def plot_moments(Rho0, moments):
	[x,y] = Rho0.vertices
	Nx = len(x)
	Ny = len(y)
	
	j_plot,i_plot = np.meshgrid(np.linspace(0,Nx-1,Nx),np.linspace(0,Ny-1,Ny))
	j_plot,i_plot = np.reshape(j_plot,(Nx*Ny,)), np.reshape(i_plot,(Nx*Ny,))
	indices_list = np.vstack([i_plot,j_plot]).T.astype(np.int16)
	# Prendre la norme du plus grand vecteur et la reduire a
	# la taille du pas de la grille.
	max_norm = np.max(np.linalg.norm(moments, axis=1))
	moments /= max_norm
	# Adapt arrow size to min(dx,dy)
	moments *= min((max(x)-min(x))/Nx, (max(y)-min(y))/Ny)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for k in xrange(Nx*Ny):
		i0,j0 = indices_list[k,:]
		ax.arrow(x[j0], y[i0], moments[k,0], moments[k,1], head_width=0.002,width=0.00002, fc='k', ec='k')

	# Add Rho0 in background of the plot
	ax.imshow(Rho0.values,extent=(min(x), max(x), min(y), max(y)), origin='lower')
	ax.set_title('Average Moments')
	plt.show()
