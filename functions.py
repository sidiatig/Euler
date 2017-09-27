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
	
	
def interp_frames_calculation_splitting(X,Rho0, Rho1, Gamma_x, Gamma_y, epsilon, nb_frames=5, w2=None):
	t = np.linspace(0.,1.,nb_frames)
	Nx,Ny = Rho0.shape[1],Rho0.shape[0]
	interp_frames = np.zeros((nb_frames,Ny,Nx))
	W2_frames = np.zeros((nb_frames,))
	W2_prime_frames = np.zeros((nb_frames,))
	if w2:
		for i in xrange(nb_frames):
			interp_frames[i,:,:],W2_frames[i],W2_prime_frames[i] = mccan_interp_splitting(t[i], X, Rho0, Rho1, Gamma_x, Gamma_y, epsilon, w2)
	else:
		for i in xrange(nb_frames):
			interp_frames[i,:,:] = mccan_interp_splitting(t[i],X, Rho0, Rho1, Gamma_x, Gamma_y, epsilon)[0]

	return interp_frames if w2 is None else interp_frames,W2_frames,W2_prime_frames
	
	
def mccan_interp_splitting(t, X, Rho0, Rho1, Gamma_x, Gamma_y, epsilon, w2=None):
	A1t = np.ones_like(Rho0)
	error_min = 3e-6
	count = 1
	niter_max = 20000

	for i in xrange(niter_max):

		A0t = np.divide( Rho0, (Gamma_y**t).dot( (Gamma_y**(1-t)).dot(A1t).dot(Gamma_x**(1-t)).dot(Gamma_x**t) ) )

		A1nt = np.divide( Rho1, (Gamma_y**(1-t)).dot((Gamma_y**t).dot(A0t).dot(Gamma_x**t).dot(Gamma_x**(1-t))) )

		# Thompson metric stopping criterion
		error = np.amax(np.abs(epsilon*np.log(A1nt/A1t)))
		
		print('error at step', count, '=', error)
		A1t = A1nt
		print((error < error_min))
		if(error < error_min):
			break
		count += 1
	
	interp = np.multiply( (Gamma_y**(1-t)).dot(A0t.dot(Gamma_x**(1-t))), ((Gamma_y**t).dot(A1t)).dot(Gamma_x**t) )
	
	 
	# Wasserstein distance calculation
	W=None
	W_prime=None
	if w2:
		A0,A1 = solve_IPFP_split(Gamma_x,Gamma_y,Rho0,interp,epsilon)
		W = wasserstein_distance(X,Gamma_x,Gamma_y,A0,A1)
		W_prime = wasserstein_distance(X,Gamma_x,Gamma_y,A0t,A1t,t)
		
	print('Total mass of interpolant at t =', t, ':', np.sum(interp))
	return interp if w2 is None else interp,W,W_prime


def interpolator_splitting(Gx, Gy, R0, R1, t, eps):
	
	A1t = np.ones_like(R0)
	error_min = 3e-6
	count = 1
	niter_max = 20000
	Gx_t, Gy_t = Gx**t, Gy**t
	Gx_1mt, Gy_1mt = Gx**(1-t), Gy**(1-t)
	
	for i in xrange(niter_max):
	
		A0t = np.divide( R0, Gy_t.dot( Gy_1mt.dot(A1t).dot(Gx_1mt).dot(Gx_t) ) )

		A1nt = np.divide( R1, Gy_1mt.dot(Gy_t.dot(A0t).dot(Gx_t).dot(Gx_1mt)) )

		# Thompson metric stopping criterion
		error = np.amax(np.abs(eps*np.log(A1nt/A1t)))
		
		print('error at step', count, '=', error)
		A1t = A1nt
		if(error < error_min):
			break
		count += 1
	
	return np.multiply( Gy_1mt.dot(A0t.dot(Gx_1mt)), (Gy_t.dot(A1t)).dot(Gx_t) )


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
		CGx = (Gamma_x**t).dot(np.multiply(Gamma_x**(1-t),Cx))
		CGy = (Gamma_y**t).dot(np.multiply(Gamma_y**(1-t),Cy))

		W2 = np.multiply(A0.dot(CGx), (Gamma_y**t).dot(Gamma_y**(1-t)).dot(A1))
		W2 += np.multiply(A0.dot(Gamma_x**t).dot(Gamma_x**(1-t)), CGy.dot(A1))
	
	W2 = np.sum(W2)

	return np.sqrt(W2)