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
	

def parse_parameter_file(parser):
	"""
	Initialise parameters of the problem according
	to a parameter file.
	
	Parameters
	----------
	parser : parser object
		Contains the parameter file name.
		
	Returns
	-------
	param : dictionnary
		Contains source and target files name, dimensions,
		target base coordinates, solver type and output file name.
	"""
	parser.add_argument('--f', '--file', type=str, default='0', help='parameter file', metavar='f')
	args = parser.parse_args()
	
	param = default_parameters()
	# If no parameter file is given
	if args.f == '0':
		print('****\nYou have to pass a parameter file as argument.\n****')
		sys.exit(1)
		
	# If a parameter file is given
	if args.f != '0':
		try:
			infile = open(args.f, "r")
			
			try:
				i = 1
				for line in infile:
					# \n and # (comments) beginning lines are ignored
					if line[0]=='\n' or line[0]=='#':
						i += 1
					else:
						# Separator is tab
						data = line.rstrip('\n\r').split("\t")
						if data[0] in param:
							param[data[0]] = data[1]
						else:
							print(args.f,':','line', i, data[0],' is not a valid parameter.')
							sys.exit(1)
						i += 1
				
				if param['filename_Rho0'] is None:
					print("Error: You have to enter a file path to the source density.")
				if param['filename_Rho1'] is None:
					print("Error: You have to enter a file path to the target density.")
				param['nFrames'] = int(param['nFrames'])
				param['lambda0'] = float(param['lambda0'])
				param['lambda1'] = float(param['lambda1'])
				if(param['lambda0']<0 or param['lambda1']<0):
					print("Error: Lambdas must be postive")
					sys.exit(1)
				param['epsilon'] = float(param['epsilon'])
				if(param['epsilon']<0):
					print("Error: Regularization parameter must be postive")
					sys.exit(1)
				
				print(param)
			finally:
				infile.close()
				
		except IOError:
			print ("Error: can\'t find file or read data")
			sys.exit(1)
			
		except ValueError:
			print ("Error: wrong data type in the file")
			sys.exit(1)
			
	return param
	

def default_parameters():
	"""
	Set default values and names of parameters. They will be erased by the
	values in the parameter file.
	"""
	param = dict(filename_Rho0 = None,
			 filename_Rho1 = None,
			 epsilon = None,
			 nFrames = None,		# Number of interpolated frames (including marginals)
			 lambda0 = np.inf,		# lambda \in [0, +inf[, no mass creation if np.inf
			 lambda1 = np.inf,
			 name = None, 			# For the saving
			 save_interp = True,
			 save_w2 = True,
			 save_moments = True)
	return param
