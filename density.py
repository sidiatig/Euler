# Copyright (C) 2017 Simon Legrand
from __future__ import print_function

import numpy as np
import inout
import functions as func
import plots

class Density:
	"""
		Describes a discrete 2D density defined on a finite
		set of vertices. The vertices are translated and rescaled to
		obtain a density contained in [0.,1.]x[0.,1.].
	"""
	def __init__(self, vert, val, rescale=False, uniq=False):
		"""
			Parameters
			----------
			vert : array_like
				For gridded support, vertices can be 
				2 raws array describing the density support. vert[0] is x and vert[1] y.
			val : array_like
				
				
		"""
		self.Nx, self.Ny = len(vert[0]), len(vert[1]) 	# Nb columns, raws
		self.vertices = vert
		self.values = val								# 2D Array(Ny,Nx)
		self.t = [0.,0.]								# Translation vector
		self.alpha = 1.									# Scaling factor
		if(rescale):
			self.rescale()
		self.mass_ratio = 1.
		
		
	@classmethod
	def from_file(cls, filename, rescale=False, uniq=False):
		"""
		Parameters
		---------- 
        filename : String
					Path of the file from which the Density object
					will be constructed.
		rescale : Bool
					If True, rescale the support into [0.,1.]x[0.,1.].
		"""
		vert, val = inout.read_txt(filename)
		return cls(vert, val, rescale, uniq)
	
	# Ajouter init ac hdf5
	
	def rescale(self):
		mx, my = min(self.vertices[0]), min(self.vertices[1])
		self.t = [mx,my]
		self.vertices[0], self.vertices[1] = self.vertices[0]-self.t[0], self.vertices[1]-self.t[1]
		self.alpha = np.max(self.vertices)					
		self.vertices /= self.alpha 					# New support into [0.,1.]x[0.,1.]
	
	
	def divide_mass(self, ratio):
		"""
		Divide the total mass of the density by ratio.
		"""
		self.values /= ratio
		self.mass_ratio = ratio
		
		
	def mass(self):
		"""
		Returns the total mass of the density
		"""
		return np.sum(self.values)
		
		
	def get_initial_mass(self):
		"""
		Returns the original mass of the density,
		i.e. before any call to divide_mass.
		"""
		return np.sum(self.values) * self.mass_ratio
	
	
	def get_initial_vertices(self):
		"""
		Returns the original support of the density,
		i.e. before the rescaling
		"""
		x = self.vertices[0]*self.alpha + self.t[0]
		y = self.vertices[1]*self.alpha + self.t[1]
		return [x,y]
	
	
	def plot(self):
		plots.plot_density(self.vertices, self.values)
		
	
	def save(self):
		inout.save_density()
		
class Interpolant:
	def __init__(self, Rho0, Rho1, param):
		"""
		This class describe an interpolation problem based on optimal transport.
		"""
		self.Rho0 = Rho0
		self.Rho1 = Rho1
		self.param = param
		self.has_run = False
		self.Frames = []
		self.W2 = np.zeros((param['nFrames']))
		self.Gamma_x = func.compute_gamma(Rho0.vertices[0], Rho0.vertices[0], param['epsilon'])
		self.Gamma_y = func.compute_gamma(Rho0.vertices[1], Rho0.vertices[1], param['epsilon'])
		self.Rho0_tilde = None
		self.Rho1_tilde= None
		
	def run(self):
		"""
		Run the interpolation process
		"""
		if(self.has_run):
			print("The interpolation has already been calculated")
			return
		
		t = np.linspace(0.,1.,self.param['nFrames'])
		buf = np.empty_like(self.Rho0.values)
		# Balanced transport
		if(self.param['lambda0']==np.inf and self.param['lambda1']==np.inf):
			# Call interpolator
			for i in xrange(self.param['nFrames']):
				self.Frames.append(Density(self.Rho0.vertices,\
				func.interpolator_splitting(self.Gamma_x, self.Gamma_y, self.Rho0.values, self.Rho1.values, t[i], self.param['epsilon'])))
		#else:
			# Find Rho0_tilde...
			# Call interpolator
		self.has_run = True
		
