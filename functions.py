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
	error_min = 1e-4
	niter_max = 3000

	for i in xrange(niter_max):
	
		A1 = div0(R1, Gy.T.dot(A0).dot(Gx))
		#A1[np.isinf(A1)] = 0
		A0n = div0(R0, Gy.dot(A1).dot(Gx.T))
		#A0n[np.isinf(A0n)] = 0
		
		error = np.sum(np.absolute(A0n-A0))/np.sum(A0)
		if(i%10 == 0):
			print('error at step', i, '=', error)
		A0 = A0n
		if(error < error_min):
			break
	
	return A0,A1
#def interp_frames_calculation_splitting(X,Rho0, Rho1, Gamma_x, Gamma_y, epsilon, nb_frames=5, w2=None):
#	t = np.linspace(0.,1.,nb_frames)
#	Nx,Ny = Rho0.shape[1],Rho0.shape[0]
#	interp_frames = np.zeros((nb_frames,Ny,Nx))
#	W2_frames = np.zeros((nb_frames,))
#	W2_prime_frames = np.zeros((nb_frames,))
#	if w2:
#		for i in xrange(nb_frames):
#			interp_frames[i,:,:],W2_frames[i],W2_prime_frames[i] = mccan_interp_splitting(t[i], X, Rho0, Rho1, Gamma_x, Gamma_y, epsilon, w2)
#	else:
#		for i in xrange(nb_frames):
#			interp_frames[i,:,:] = mccan_interp_splitting(t[i],X, Rho0, Rho1, Gamma_x, Gamma_y, epsilon)[0]

#	return interp_frames if w2 is None else interp_frames,W2_frames,W2_prime_frames
#	
#	
#def mccan_interp_splitting(t, X, Rho0, Rho1, Gamma_x, Gamma_y, epsilon, w2=None):
#	A1t = np.ones_like(Rho0)
#	error_min = 3e-6
#	count = 1
#	niter_max = 20000

#	for i in xrange(niter_max):

#		A0t = np.divide( Rho0, (Gamma_y**t).dot( (Gamma_y**(1-t)).dot(A1t).dot(Gamma_x**(1-t)).dot(Gamma_x**t) ) )

#		A1nt = np.divide( Rho1, (Gamma_y**(1-t)).dot((Gamma_y**t).dot(A0t).dot(Gamma_x**t).dot(Gamma_x**(1-t))) )

#		# Thompson metric stopping criterion
#		error = np.amax(np.abs(epsilon*np.log(A1nt/A1t)))
#		
#		print('error at step', count, '=', error)
#		A1t = A1nt
#		print((error < error_min))
#		if(error < error_min):
#			break
#		count += 1
#	
#	interp = np.multiply( (Gamma_y**(1-t)).dot(A0t.dot(Gamma_x**(1-t))), ((Gamma_y**t).dot(A1t)).dot(Gamma_x**t) )
#	
#	 
#	# Wasserstein distance calculation
#	W=None
#	W_prime=None
#	if w2:
#		A0,A1 = solve_IPFP_split(Gamma_x,Gamma_y,Rho0,interp,epsilon)
#		W = wasserstein_distance(X,Gamma_x,Gamma_y,A0,A1)
#		W_prime = wasserstein_distance(X,Gamma_x,Gamma_y,A0t,A1t,t)
#		
#	print('Total mass of interpolant at t =', t, ':', np.sum(interp))
#	return interp if w2 is None else interp,W,W_prime


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
		#A1[np.isnan(A1)] = 0.
		A0n = np.power(div0(R0,Gy.dot(Gx.dot(A1.T).T)),exp0)
		#A0n[np.isnan(A0n)] = 0.
		
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
		# Not working!
		CGx = (Gamma_x**t).dot(np.multiply(Gamma_x**(1-t),Cx))
		CGy = (Gamma_y**t).dot(np.multiply(Gamma_y**(1-t),Cy))

		W2 = np.multiply(A0.dot(CGx), (Gamma_y**t).dot(Gamma_y**(1-t)).dot(A1))
		W2 += np.multiply(A0.dot(Gamma_x**t).dot(Gamma_x**(1-t)), CGy.dot(A1))
	
	W2 = np.sum(W2)

	return np.sqrt(W2)
