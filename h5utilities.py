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
## h5utilities.py:
## Functions concerning reading and preprocessing of data.
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
import h5py
import sys
import mrcfile 
import os

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def Preprocessing_steps_with_Brightness(data,angles,badname,cutedges,Badprojections):
	text=[]
	if badname!='':
		text=open(badname,'r')
		text=text.read()
		text=text.split()
		text=map(int,text)
	text=text+Badprojections
	

	index=[]
	#Remove Bad elements
	for i in range(0,data.shape[0]):
		if i+1 in text:
			a=1
		else:
			index.append(i)
	data=data[index]
	angles=angles[index]

	Nx=data.shape[2]
	cut=np.ceil(Nx*cutedges)
	cut=int(cut)
	

	#Cut of Edges
	if cut/2!=cut and cut/2>0:
		average_background=np.mean(data[:,:,(cut//2):cut])+np.mean(data[:,:,Nx-(cut//2):Nx] )
		average_background=average_background*0.5
	else:
		average_background=0

	data=data[:,:,cut:Nx-cut]
	data=data-average_background

	#Compute Brightness
	C=[]
	for i in range(0,data.shape[0]):
		B=data[i]
		C.append( np.mean(B))
	
	
	#Rescale Brightness
	W=np.mean(C)
	for i in range(0,data.shape[0]):
		data[i]=W/C[i]*data[i]
	#data=data/np.max(data)
	return data, angles, 0
	
def Preprocessing_steps_without_Brightness(data,angles,badname,cutedges,Badprojections):
	text=[]
	if badname!='':
		text=open(badname,'r')
		text=text.read()
		text=text.split()
		text=map(int,text)
	text=text+Badprojections
	

	index=[]
	#Remove Bad elements
	for i in range(0,data.shape[0]):
		if not i+1 in text:
			index.append(i)
	data=data[index]
	angles=angles[index]

	Nx=data.shape[2]
	cut=np.ceil(Nx*cutedges)
	cut=int(cut)
	

	#Cut of Edges
	if cut/2!=cut and cut/2>0:
		average_background=np.mean(data[:,:,(cut//2):cut])+np.mean(data[:,:,Nx-(cut//2):Nx] )
		average_background=average_background*0.5
	else:
		average_background=0

	data=data[:,:,cut:Nx-cut]
	data=data-average_background

	return data, angles, 0



##Reades data from name, takes int of Badnames and Badprojections and removes the corresponding projections. Removes cutedge part*100% of data on the left and right . 
#Uses Brightness-correction in order to make sure Brighnes is same in all projections
def Brightness_correction(name,badname,cutedges,Badprojections):
	data,angles=readh5(name)
	text=[]

	#Gather bad projections 
	if badname!='':
		text=open(badname,'r')
		text=text.read()
		text=text.split()
		text=map(int,text)
	text=text+Badprojections
	

	index=[]
	#Remove Bad elements
	for i in range(0,data.shape[0]):
		if i+1 in text:
			a=1
		else:
			index.append(i)
	data=data[index]
	angles=angles[index]

	Nx=data.shape[2]
	cut=np.ceil(Nx*cutedges)
	cut=int(cut)
	

	#Cut of Edges
	if cut/2!=cut and cut/2>0:
		average_background=np.mean(data[:,:,(cut//2):cut])+np.mean(data[:,:,Nx-(cut//2):Nx] )
		average_background=average_background*0.5
	else:
		average_background=0

	data=data[:,:,cut:Nx-cut]
	data=data-average_background

	#Compute Brightness
	C=[]
	for i in range(0,data.shape[0]):
		B=data[i]
		C.append( np.mean(B))
	
	
	#Rescale Brightness
	W=np.mean(C)
	for i in range(0,data.shape[0]):
		data[i]=W/C[i]*data[i]
	#data=data/np.max(data)
	


	return data, angles, 0

##Reades data from name, takes int of Badnames and Badprojections and removes the corresponding projections. Removes cutedge part*100% of data on the left and right . 
#Uses Brightness-correction in order to make sure Brighnes is same in all projections
def Brightness_correction_alternativ(name,badname,cutedges,Badprojections):
	data,angles,Pixelsize=readh5alternative(name)
	text=[]
	
	#Gather bad projections 
	if badname!='':
		text=open(badname,'r')
		text=text.read()
		text=text.split()
		text=map(int,text)
	

	text=text+Badprojections

	#Remove Bad elements
	index=[]
	for i in range(0,data.shape[0]):
		if i+1 in text:
			a=1
		else:
			index.append(i)
	data=data[index]
	angles=angles[index]

	Nx=data.shape[2]
	
	#Cut Data
	cut=np.ceil(Nx*cutedges)
	cut=int(cut)
	if cut/2!=cut and cut/2>0:
		average_background=np.mean(data[:,:,(cut//2):cut])+np.mean(data[:,:,Nx-(cut//2):Nx] )
		average_background=average_background*0.5
	else:
		average_background=0	
	data=data[:,:,cut:Nx-cut]
	data=data-average_background

	#Compute Brightness
	C=[]
	for i in range(0,data.shape[0]):
		B=data[i]
		C.append( np.mean(B))

	
	#Rescale Brightness
	W=np.mean(C)
	for i in range(0,data.shape[0]):
		data[i]=W/C[i]*data[i]
	#data=data/np.max(data)
	


	return data, angles, 0,Pixelsize


def Brightness_correction2(sino):
	
	Nx=sino.shape[1]
	C=[]
	for i in range(0,sino.shape[0]):
		B=sino[i]
		C.append( np.mean(B))

	W=np.mean(C)
	for i in range(0,sino.shape[0]):
		sino[i]=W/C[i]*sino[i]
	#sino=sino/np.max(sino)

	return sino
	

##Reades data from name, takes int of Badnames and Badprojections and removes the corresponding projections. Removes cutedge part*100% of data on the left and right . 
def global_rescaling(name,badname,cutedges,Badprojections):
	data,angles=readh5(name)
	text=[]
	#Gather bad projections 
	if badname!='':
		text=open(badname,'r')
		text=text.read()
		text=text.split()
		text=map(int,text)
	
	text=text+Badprojections
		
	data=data-np.min(data)
	#data=data/np.max(data)
	
	#Remove bad  projections
	index=[]
	for i in range(0,data.shape[0]):
		if i+1 in text:
			a=1
		else:
			index.append(i)
	data=data[index]
	angles=angles[index]

	data=data-np.min(data)

	#data=data/np.max(data)
	Nx=data.shape[2]
	
	#cut Edges
	cut=np.ceil(Nx*cutedges)
	cut=int(cut)
	if cut/2!=cut and cut/2>0:
		average_background=np.mean(data[:,:,(cut//2):cut])+np.mean(data[:,:,Nx-(cut//2):Nx] )
		average_background=average_background*0.5
	else:
		average_background=0  
	data=data[:,:,cut:Nx-cut]
	return data, angles, average_background

##Reades data from name, takes int of Badnames and Badprojections and removes the corresponding projections. Removes cutedge part*100% of data on the left and right . 
def global_rescaling_alternativ(name,badname,cutedges,Badprojections):
	data,angles,Pixelsize=readh5alternative(name)
	text=[]
	#Gather Bad Projections
	if badname!='':
		text=open(badname,'r')
		text=text.read()
		text=text.split()
		text=map(int,text)
	
	text=text+Badprojections
	
	data=data-np.min(data)
	#data=data/np.max(data)
	
	#Remove Bad projections
	index=[]
	for i in range(0,data.shape[0]):
		if i+1 in text:
			a=1
		else:
			index.append(i)
	data=data[index]
	angles=angles[index]

	data=data-np.min(data)

	#data=data/np.max(data)
	Nx=data.shape[2]
	
	#Cut edges
	cut=np.ceil(Nx*cutedges)
	cut=int(cut)
	if cut/2!=cut and cut/2>0:
		average_background=np.mean(data[:,:,(cut//2):cut])+np.mean(data[:,:,Nx-(cut//2):Nx] )
		average_background=average_background*0.5
	else:
		average_background=0  
	data=data[:,:,cut:Nx-cut]
	return data, angles, average_background, Pixelsize


##Reads H5 files and returns Data and angles
def readh5(name):
	if '.h5' in name:
		with h5py.File(name,'r') as hf:
		
			data = hf.get('data')
			info=hf.get('Info')

			np_data=np.array(data)
			Dimensions=info.get('Dimensions');iii=Dimensions.get('3');	angles=np.array(iii)
			
	elif '.mrc' in name:
		mrc= mrcfile.open(name,'r',permissive=True)
		A=mrc.data
		Na,Nx,Ny=A.shape
		np_data=np.zeros([Na,Nx,Ny])
		np_data[:]=A[:]
		#np_data=np.zeros([Na,Ny,Nx])
		#for i in range(Nx):
		#	np_data[:,:,i]=data[:,i,:]
		
		angles=[]
		if os.path.isfile(name.replace('.mrc','.rawtlt'))==True:
			File = open(name.replace('.mrc','.rawtlt'),'r') 
			File=File.read()
			File=File.split()

			try:		
				for entry in File:
					angles.append(float(entry)/180*np.pi)
				angles=np.array(angles)
			except ValueError:
				raise ValueError('Numbers "'+entry[0]+'" in the angle file '+ name.replace('.mrc','.rawtlt')+ ' could not be interpret correctly')
				
		elif os.path.isfile(name.replace('.mrc','.csv'))==True:
			import csv
			File=open(name.replace('.mrc','.csv'),"r")
			File=csv.reader(File,delimiter=',')
			try:		
				for entry in File:
					angles.append(float(entry[0])/180*np.pi)
				angles=np.array(angles)
			except ValueError as err:
				template = "An exception of type {0} occurred. Arguments:\n{1!r}"
				message = template.format(type(err).__name__, err.args)
				eprint(message)
				raise ValueError('Number "'+entry[0]+'" in the angle file '+ name.replace('.mrc','.csv')+ ' could not be interpret correctly')
						
		else:
			
			raise IOError('Could not find angle files ' +name.replace('.mrc','.csv')+' or '+name.replace('.mrc','.rawtlt'))
			
		if os.path.isfile(name.replace('.mrc','.rawtlt'))==True and os.path.isfile(name.replace('.mrc','.csv'))==True:
			eprint('Warning, two candidates for angle files '+name.replace('.mrc','.rawtlt')+' and '+name.replace('.mrc','.csv')+'. By default rawtlt information is used')
			
	else:
		raise IOError('Could not interpret datatype of '+name+', aborting process')
		
	
	
	
	return np_data, angles
	
##Reads H5 files and returns Data and angles and Pixelsize
def readh5alternative(name):
	if '.h5' in name:
		with h5py.File(name,'r') as hf:
		
			data = hf.get('data')
			info=hf.get('Info')

			np_data=np.array(data)
			Dimensions=info.get('Dimensions');iii=Dimensions.get('3');	angles=np.array(iii)
			ii=Dimensions.get('2');Pixelsize=np.array(ii);
			
	elif '.mrc' in name:
		mrc= mrcfile.open(name,'r',permissive=True)
		A=mrc.data
		Na,Nx,Ny=A.shape
		np_data=np.zeros([Na,Nx,Ny])
		np_data[:]=A[:]
		#np_data=np.zeros([Na,Ny,Nx])
		#for i in range(Nx):
		#	np_data[:,:,i]=data[:,i,:]
		angles=[]
		if os.path.isfile(name.replace('.mrc','.rawtlt'))==True:
			File = open(name.replace('.mrc','.rawtlt'),'r') 
			File=File.read()
			File=File.split()

			try:		
				for entry in File:
					angles.append(float(entry)/180*np.pi)
				angles=np.array(angles)
			except ValueError:
				raise ValueError('Numbers "'+entry[0]+'" in the angle file '+ name.replace('.mrc','.rawtlt')+ ' could not be interpret correctly')
				
		elif os.path.isfile(name.replace('.mrc','.csv'))==True:
			import csv
			File=open(name.replace('.mrc','.csv'),"r")
			File=csv.reader(File,delimiter=',')
			try:		
				for entry in File:
					angles.append(float(entry[0])/180*np.pi)
				angles=np.array(angles)
			except ValueError:
				raise ValueError('Number "'+entry[0]+'" in the angle file '+ name.replace('.mrc','.csv')+ ' could not be interpret correctly')
						
		else:
			raise IOError('Could not find angle files ' +name.replace('.mrc','.csv')+' or '+name.replace('.mrc','.rawtlt'))
			
		if os.path.isfile(name.replace('.mrc','.rawtlt'))==True and os.path.isfile(name.replace('.mrc','.csv'))==True:
			eprint('Warning, two candidates for angle files '+name.replace('.mrc','.rawtlt')+' and '+name.replace('.mrc','.csv')+'. By default rawtlt information is used')
			
		Pixelsize=[-1,-1]
	else:
		raise IOError('Could not interpret datatype of '+name+', aborting process')
		
	return np_data, angles, Pixelsize


def get_slice(i,data):
	return data[i][:,:]
	
def get_sinogram(i,data,j=None):
	sino=np.array(data[:,i,:])
	return sino
	
