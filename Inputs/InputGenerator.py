# Copyright (C) 2017 Simon Legrand
from __future__ import division, print_function, absolute_import

import numpy as np

def gaussian(x,y,x0,y0,sigma_x,sigma_y):
	"""
	Generates a gaussian density on a grid defined
	by x and y.
	"""
	Nx = len(x)
	Ny = len(y)
	x_tiled = np.tile(x,(Ny,1))
	y_tiled = np.tile(y,(Nx,1)).T

	gaus = np.exp(-((x_tiled-x0)**2)/(sigma_x**2) - ((y_tiled-y0)**2)/(sigma_y**2))
	return gaus

	

##### Writing in the file #####
myFile = file('input.txt','w')