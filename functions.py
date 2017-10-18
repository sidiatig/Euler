# Copyright (C) 2017 Simon Legrand
from __future__ import print_function

import numpy as np

def compute_gamma(x,y,eps):
	
	Nx = len(x)
	Ny = len(y)
	
	x_tiled = np.tile(x,(Ny,1)).T
	y_tiled = np.tile(y,(Nx,1))

	C = 0.5*(x_tiled - y_tiled)**2
	return np.exp(-C/eps)
	

def div0(x1,y1):
	"""
	Avoid division by zero
	"""
	x = np.nan_to_num(x1)
	y = np.nan_to_num(y1)
	res = np.zeros_like(x)
	I = np.logical_and((x>=0.),(y>0.))
	res[I] = np.divide(x[I],y[I])
	J = np.logical_and(x>=0., y==0.)
	res[J] = 0.
	return res

	
def solve_IPFP_split(Gx, Gy, R0, R1, epsilon):
	
	A0 = np.ones_like(R0)
	error_min = 1e-8
	niter_max = 3000

	for i in xrange(niter_max):
	
		A1 = div0(R1, Gy.T.dot(A0).dot(Gx))
		A0n = div0(R0, Gy.dot(A1).dot(Gx.T))
		
		# TODO
		# Tester un autre critere d'arret
		tmp = A0n/A0
		I = np.isfinite(tmp)
		error = np.amax(np.log(tmp[I]))
		#error = np.sum(np.absolute(A0n-A0))/np.sum(A0)
		if(i%10 == 0):
			print('error at step', i, '=', error)
		A0 = A0n
		if(error < error_min):
			break
	
	return A0,A1


def interpolator_splitting(X, Gx, Gy, R0, R1, t, eps):
	
	A1t = np.ones_like(R0)
	error_min = 1e-4
	niter_max = 3000
	Gx_t, Gy_t = Gx**t, Gy**t
	Gx_1mt, Gy_1mt = Gx**(1-t), Gy**(1-t)
	
	for i in xrange(niter_max):
	
		A0t = div0( R0, Gy_t.dot( Gy_1mt.dot(A1t).dot(Gx_1mt).dot(Gx_t) ) )
		A1nt = div0( R1, Gy_1mt.dot(Gy_t.dot(A0t).dot(Gx_t).dot(Gx_1mt)) )
		
		# Thompson metric stopping criterion
		# error = np.amax(np.abs(eps*np.log(A1nt/A1t)))
		error = np.sum(np.absolute(A1nt-A1t))/np.sum(A1t)
		if(i%10 == 0):
			print('error at step', i, '=', error)
		A1t = A1nt
		if(error < error_min):
			break
		i += 1
	
	interp = np.multiply( Gy_1mt.dot(A0t.dot(Gx_1mt)), (Gy_t.dot(A1t)).dot(Gx_t) )
	A0,A1 = solve_IPFP_split(Gx,Gy,R0,interp,eps)
	W2 = wasserstein_distance(X, Gx, Gy, A0, A1)
	
	return W2, interp
	

def solve_IPFP_split_penalization(Gx, Gy, R0, R1, param):
	""" 
	Lambda_0 and lambda_1 can be either a scalars or matrices of the same
	dimension as Gammas.
	"""
	A0 = np.ones_like(R0)
	error_min = 1e-4
	niter_max = 3000
	epsilon = param['epsilon']
	lambda0 = param['lambda0']
	lambda1 = param['lambda1']
	exp0 = lambda0/(lambda0 + epsilon)
	if(not np.isfinite(exp0)):
		exp0 = 1.
	exp1 = lambda1/(lambda1 + epsilon)
	if(not np.isfinite(exp1)):
		exp1 = 1.
	
	for i in xrange(niter_max):
		A1 = np.power(div0(R1,Gx.dot(Gy.dot(A0).T).T),exp1)
		A0n = np.power(div0(R0,Gy.dot(Gx.dot(A1.T).T)),exp0)
		
		# Thompson metric stopping criterion
		#tmp = np.log(div0(A0n,A0))
		#tmp[np.isinf(tmp)] = 0.
		#error = np.amax(np.abs(epsilon*tmp))
		error = np.sum(np.absolute(A0n-A0))/np.sum(A0)
		if(i%10 == 0):
			print('error at step', i, '=', error)
		A0 = A0n
		if(error < error_min):
			break
	
	return A0,A1


def wasserstein_distance(X,Gamma_x,Gamma_y,A0,A1,t=None):
	"""
	Returns the wasserstein distances between two measures
	which scalings are A0 and A1.
	"""
	Nx = len(X[0])
	Ny = len(X[1])
	
	x_tiled = np.tile(X[0],(Nx,1))
	y_tiled = np.tile(X[1],(Ny,1))

	Cx = 0.5*(x_tiled.T - x_tiled)**2
	Cy = 0.5*(y_tiled.T - y_tiled)**2
	
	if t is None:
		CGx = np.multiply(Gamma_x,Cx)
		CGy = np.multiply(Gamma_y,Cy)
		
		W2 = np.multiply(A0.dot(CGx), Gamma_y.dot(A1))
		W2 += np.multiply(A0.dot(Gamma_x), CGy.dot(A1))
		
	else:
		# Not working! Certainement un pb avec la formulation
		CGx = (Gamma_x**t).dot(np.multiply(Gamma_x**(1-t),Cx))
		CGy = (Gamma_y**t).dot(np.multiply(Gamma_y**(1-t),Cy))

		W2 = np.multiply(A0.dot(CGx), (Gamma_y**t).dot(Gamma_y**(1-t)).dot(A1))
		W2 += np.multiply(A0.dot(Gamma_x**t).dot(Gamma_x**(1-t)), CGy.dot(A1))
	
	W2 = np.sum(W2)

	return np.sqrt(W2)
	
	
def average_moments(A0, A1, Gamma_x, Gamma_y, Rho0, ampl=None, filename=None):
	"""
	Compute the average displacement of every grid point,
	from t=0 to t=1
	
	Parameters
	----------
	
		Rho0 : Density object
			Initial density
			
		A0, A1 : Scalings of the two densities Rho0 and Rho1
	"""
	x,y = Rho0.get_initial_vertices()
	Nx = len(x) #Number of columns
	Ny = len(y) #Number of rows
	x_grid,y_grid = np.meshgrid(x,y)

	j_plot,i_plot = np.meshgrid(np.linspace(0,Nx-1,Nx),np.linspace(0,Ny-1,Ny))
	j_plot,i_plot = np.reshape(j_plot,(Nx*Ny,)), np.reshape(i_plot,(Nx*Ny,))
	indices_list = np.vstack([i_plot,j_plot]).T.astype(np.int16)
	
	# Calculer les tranches du transport_plan correspondantes
	transport_plan_list = np.empty((Nx*Ny,Ny,Nx))
	for k in xrange(Nx*Ny):
		i0,j0 = indices_list[k,:]
		transport_plan_list[k,:,:] = (A0[i0,j0]*(Gamma_x[j0,:][np.newaxis,:]).T*Gamma_y[i0,:]).T*A1[:,:]
		transport_plan_list[k,:,:] /= np.sum(transport_plan_list[k,:,:])

	# Displacement of each pixel from t=0 to t=1
	vec_list = np.zeros((Nx*Ny,2))
	for k in xrange(Nx*Ny):
		i0,j0 = indices_list[k,:]
		# We define a (2,Ny,Nx) matrice containing displacement of
		# (i0,j0) to every grid point.
		t = np.stack((x_grid - x[j0], y_grid - y[i0]),axis=0)
		vec_list[k,:] = np.sum(np.sum(np.multiply(transport_plan_list[k,:,:],t),axis=1),axis=1)
	
	# Moments calculation
	#Rho = np.tile(np.reshape(Rho0,(Nx*Ny,)),(2,1)).T
	Rho = np.ones_like(vec_list)
	moments = np.multiply(vec_list,Rho)

	return moments
