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