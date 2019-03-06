#!/usr/bin/env python

## Graptor - Graz Application for Tomographic Reconstruction
##
## Copyright (C) 2019 Richard Huber, Martin Holler, Kristian Bredies
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.
##
##
## find_bad_ele.py:
## Determine unsuitable projection data.
##
## -------------------------
## Richard Huber (richard.huber@uni-graz.at)
## Martin Holler (martin.holler@uni-graz.at)
## Kristian Bredies (kristian.bredies@uni-graz.at)
## 
## 21.02.2019
## -------------------------
## If you consider this code to be useful, please cite:
## 
## [1] R. Huber, G. Haberfehlner, M. Holler, G. Kothleitner,
##     K. Bredies. Total Generalized Variation regularization for
##     multi-modal electron tomography. *Nanoscale*, 2019. 
##     DOI: [10.1039/C8NR09058K](https://doi.org/10.1039/C8NR09058K).
##
## [2] M. Holler, R. Huber, F. Knoll. Coupled regularization with
##     multiple data discrepancies. Inverse Problems, Special
##     issue on joint reconstruction and multi-modality/multi-spectral
##     imaging, 34(8):084003, 2018.


from __future__ import print_function

import numpy as np
import h5utilities as h5
from scipy import misc
import sys
from numpy import *
import matplotlib
#matplotlib.use('gtkagg')
from matplotlib.pyplot import *
import mrcfile

def compare_shift_twosided(U,V,steps):
	norm1,E1=compare_shift(U,V,steps)
	norm2,E2=compare_shift(V,U,steps)
	Nx,Ny=E1.shape
	E_new=np.zeros([2,Nx,Ny])
	E_new[0]=E1
	E_new[1]=E2
	E=np.max(E_new,axis=0)
	E**(0.5)
	E=E.reshape(E.size)
	
	return np.linalg.norm(E,1),E.reshape([Nx,Ny])
def reduce_noise(U,steps):
	Nx,Ny=U.shape
	U_new=U[steps:Nx-steps,steps:Ny-steps]
	E=np.zeros([Nx-2*steps,Ny-2*steps])
	for k in range(-steps,steps+1):
		for l in range(-steps,steps+1):
			E+=U[steps+k:Nx-steps+k,steps+l:Ny-steps+l]
	return E/(4*steps**2)
def compare_shift(U,V,steps):
	[Nx,Ny]=U.shape
	#print steps

	[Nx,Ny]=U.shape
	#print Nx,Ny
	Nxold=Nx;Nyold=Ny

	U=U[steps:Nx-steps][:,steps:Ny-steps]
	
	[Nx,Ny]=U.shape
	E=np.zeros([(2*steps+1)**2,Nx,Ny])

	counter=0
	for k in range(-steps,steps+1):
		for l in range(-steps,steps+1):
			
			V_=V[steps+k:Nxold-steps+k][:,steps+l:Nyold-steps+l]
	#		print V_.shape, U.shape,E.shape
			E[counter]=(abs(U-V_))
			counter+=1
	
	E=np.min(E,axis=0)


	
	return np.linalg.norm(E),E
			
	
def weighted_gradient(u):
				
	[Nx,Ny]=u.shape
	gradval=[]
	neighborval=[]
	for i in range(0,Nx):
		for j in range(0,Ny):
			gradval.append([])
			neighborval.append([])

	E=np.zeros([8,Nx-2,Ny-2])
	Neighborvalues=np.zeros([8,Nx-2,Ny-2])
	counter=0
	for  i in [-1,0,1]:
		for j in [-1,1]:
			E[counter]=abs(u[1:Nx-1,1:Ny-1]-u[1+i:Nx-1+i,1+j:Ny-1+j])*1/((i**2+j**2)**0.5)
			Neighborvalues[counter]=abs(u[1+i:Nx-1+i,1+j:Ny-1+j])
			counter+=1
	for i in [-1,1]:
		j=0
		E[counter]=abs(u[1:Nx-1,1:Ny-1]-u[1+i:Nx-1+i,1+j:Ny-1+j])*1/((i**2+j**2)**0.5)
		Neighborvalues[counter]=abs(u[1+i:Nx-1+i,1+j:Ny-1+j])
		counter+=1
	E=np.max(E,axis=0)
	Neighborvalues=np.min(Neighborvalues,axis=0)
	#index=np.ones([Nx-2,Ny-2])
	#index[np.where(Neighborvalues< np.mean(u))]=0
	value=E*Neighborvalues#*index*Neighborvalues


	
	return np.linalg.norm(value) ,value


def Find_Bad_projections(name,searchradius,prozent_to_reduce):
	print('###### Determining Bad Projections ######')
	data,angles,a=h5.Brightness_correction(name,'',0.0,[])
	[NA,Nz,Nx]=data.shape

	if  prozent_to_reduce==-1:
		prozent_to_reduce=0.1
	smoothing_step=3
	
	Nx_new=250;Nz_new=int(Nx_new*Nz/float(Nx))
	if Nx>Nx_new:
		data2=data[:,range(0,data.shape[1],Nz/Nz_new)][:,:,range(0,data.shape[2],Nx/Nx_new)]
		data=data2
		
	data_new=np.zeros([data.shape[0],data.shape[1]-2*smoothing_step,data.shape[2]-2*smoothing_step],dtype=np.float32)
	for i in range(0,data.shape[0]):
		data_new[i]=reduce_noise(data[i],smoothing_step)
	data=data_new




	[NA,Nz,Nx]=data.shape
	if searchradius==None:
		searchradius=min(Nx,Nz)/10

	searchradius=int(searchradius)
	B=[]
	
	Ny=Nz


	WGrad=np.zeros([NA,Nz-2,Nx-2],dtype=np.float32)

	for i in range(0,NA):
		sys.stdout.write('Step 1/3: at {:3.0%}'.format(float(i)/NA)*100)
		sys.stdout.flush()
		[norm, wgrad]=weighted_gradient(data[i])

		WGrad[i]=wgrad
	error1=[]
	error2=[]


	for i in range(0,NA-1):	
		#print 'wgradshape',WGrad.shape
		sys.stdout.write('Step 2/3: at {:3.0%}'.format(float(i)/NA)*100)
		sys.stdout.flush()
		norm,E=compare_shift_twosided(WGrad[i],WGrad[i+1],searchradius)
		error1.append(norm)
		

	for i in range(0,NA-2):
		sys.stdout.write('Step 3/3: at {:3.0%}'.format(float(i)/NA)*100)

		sys.stdout.flush()
		norm,E=compare_shift_twosided(WGrad[i],WGrad[i+2],int(searchradius*1.5))
		error2.append(norm)


	N=len(error1)+1
	Error_combined=[]
	for i in range(0,N):
		j=np.clip(i,0,N-2)
		jj=np.clip(i-1,0,N-2)
		jjj=np.clip(i,0,N-3)
		jjjj=np.clip(i-2,0,N-3)
		a=error1[j]+0.5*error2[jjj]
		b=error1[jj]+0.5*error2[jjjj]
		if i==1:
			jjjj=1
		if i== N-2:
			jjjj=N-4
		Error_combined.append(min([a,b]))
	#Error_combined.append(error1[j]+error1[jj]+0.5*(error2[jjj]+error2[jjjj]))
	error_save=list(Error_combined)
	error=Error_combined
	bad_proj=[]
	if prozent_to_reduce>=1:
		prozent_to_reduce=float(prozent_to_reduce)/N
	
	
	for j in range(0,len(error)):
		maxx=max(error)
		maximiser=error.index(maxx)
		error[maximiser]=-1
		bad_proj.append(maximiser+1)
	bad_proj=bad_proj[0: int(round(N*prozent_to_reduce))]
	sys.stdout.write('\r ')	
	sys.stdout.flush()
	print('badprojections found',bad_proj)
	return bad_proj


#name='HAADF_reduceSeries_aligned'
#name='HAADF_hrData_aligned'
#name='FEI_HAADF_aligned'
#searchradius=10
#prozent_to_reduce=15
#image_rescaling=4

#smoothing_step=3
#print (Find_Bad_projections(name,searchradius,prozent_to_reduce,image_rescaling,smoothing_step))



