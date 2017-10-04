# Copyright (C) 2016 Simon Legrand
from __future__ import division, print_function, absolute_import
"""
This module contains functions to read or write files of different types.
"""
import sys
import numpy as np
import h5py


def read_txt(fn):
	"""
	This function returns points coordinates
	contained in file fn.
	
	Parameters
	----------
	fn : string
		File name
	
	Returns
	-------
	X : 2D array
		Samples coordinates
	z : 1D array
		Weights associated to samples
	"""
	infile = open(fn, "r")
		
	try:	
		x = []
		y = []
		z = []
		for line in infile:
			data = line.rstrip('\n\r').split(" ")
			x.append(float(data[0]))
			y.append(float(data[1]))
			z.append(float(data[2]))

		x = np.unique(np.asarray(x))
		y = np.unique(np.asarray(y))
		z = np.reshape(np.asarray(z), (np.size(y),np.size(x)), order='F')
		#print((np.size(y),np.size(x)))
		return [x,y], z
		
	except Exception as e:
		print(e)
		sys.exit(1)	
	finally:
		infile.close()


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


def uniform_square(x,y,box):
	"""
	Generates a uniform square density on box, included
	in the grid defined by x and y.
	"""
	Nx = len(x)
	Ny = len(y)
	x_tiled = np.tile(x,(Ny,1))
	y_tiled = np.tile(y,(Nx,1)).T
	dens = np.zeros((Ny,Nx))
	I = (x_tiled<box[1]) & (x_tiled>box[0]) & (y_tiled<box[3]) & (y_tiled>box[2])
	dens[I] = 1.
	return dens

	
def export_hdf(param, interp_frames, w2, moments=None):
	"""
	Export data into a hdf5 file.
	"""
	f = h5py.File(param['name']+'.hdf5','w')
	f.create_dataset('Interp',data=interp_frames)
	f.create_dataset('W2',data=w2)
	f.create_dataset('epsilon',data=param['epsilon'])
	if(param['lambda0'] != np.inf):
		f.create_dataset('lambda0',data=param['lambda0'])
	if(param['lambda0'] != np.inf):
		f.create_dataset('lambda1',data=param['lambda1'])
	if(moments is not None):
		f.create_dataset('Moments',data=moments)
	return
		
def save_density():
	print("pouet")
