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
## Reconstruction_coupled.py:
## Joint variational reconstruction approach.
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

from numpy import *
import os
import numpy as np
import scipy

import sys
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import math
import h5utilities as hdf5
import pyopencl as cl
import argparse
import mrcfile
import find_bad_ele as find_bad_projections
import copy


def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def get_gpu_context(GPU_choice):
	platforms = cl.get_platforms()
	my_gpu_devices = []
	try:
		for platform in platforms:
			my_gpu_devices.extend(platform.get_devices())
		gpu_devices = [device for device in my_gpu_devices if device.type == cl.device_type.GPU]
		non_gpu_devices = [device for device in my_gpu_devices if device not in gpu_devices]
		my_gpu_devices = gpu_devices + non_gpu_devices
	except:
		pass
	if my_gpu_devices == []:
		raise cl.Error('No device found, make sure PyOpenCL was installed correctly.')
	while GPU_choice not in range(0,len(my_gpu_devices)):
		for i in range(0,len(my_gpu_devices)):
			print( '(',i,')' ,my_gpu_devices[i])
		GPU_choice=input('Choose device to use by entering the number preceeding it: ' )
		try:
			GPU_choice = int(GPU_choice)
		except ValueError:
			print('Please enter an integer value')

	if GPU_choice in range(0,len(my_gpu_devices)):
		gerat=[my_gpu_devices[GPU_choice]]
		ctx = cl.Context(devices=gerat)
	else:
		ctx = cl.create_some_context()
	return ctx, my_gpu_devices, GPU_choice
	
	
def Allows_number_slices(imageshape,sinogram_shape,number_channels,device):
	#4 byte pro pixel mit 23 Images * number_channels +2 Images (for norms) + 3 sinograms
	bytes_per_slice=4*(imageshape[0]*imageshape[1]*23+sinogram_shape[0]*sinogram_shape[1]*3)*number_channels+4*imageshape[0]*imageshape[1]*2
	maximal_allocation_per_slice= 4*max(8*imageshape[0]*imageshape[1],sinogram_shape[0]*sinogram_shape[1])*number_channels
	global_memory=device.global_mem_size
	max_individual_allocation=device.max_mem_alloc_size
	
	Limit1=max_individual_allocation//maximal_allocation_per_slice
	Limit2=global_memory//bytes_per_slice
	Limit=min(Limit1,Limit2)
	return Limit

	
def Coupled_Reconstruction2d(names,result, mu,alpha,maxiter,plott,method,Slice_levels,datahandling,GPU_choice,Channelnames,reductionparameter,Regularisation,Discrepancy,Scalepars):
	"""
	Reconstruction method employed for 2D reconstruction. Mainly deals with pre and post processing of data and calls reconstruction algorithm
	
	Arguments:
		names ...A list of names corresponding to the data that shall be read
		result ...A string representing the path where results should be saved
		mu ... A list of parameter for the weighting
		alpha A list of TGV parameters
		maxiter An integer representing the number of iterations done by the algorithm
		plott ... A list with 2 entries, the first 1 if a plot shall be constructed, 0 otherwise, the second in procent, i.e. 0.1 for 10% where this gives the amount of replotting of the live results
		method ... The reconstruction method being used, like frobenius or uncorrelated
		Slice_levels ... A list with 3 entries, being used in a range argument (first beginning, second end, last stepsize)
		datahandling ... List of objects, first a string for brightness, than a string for thresholding options,  then in a number smaller 1 how much of the data should be removed on the sides,then the name of a file (potentially '') containing the bad projections, finally a thresholding value if required
		GPU_choice ... An integer depicting which GPU to use
		Channelnames ... A list of names for the respective channels, which which the results are saved
		reductionparameter ... A list with two entries, the procent of projections to remove(e.g. 10% with 0.1) or the absolute number as an integer, and a Search radius (Pixelwidth for smoothing)
		Regularisation ... TV or TGV as String to determine the regularisation functional used
		Discrepancies ... KL or L2 as String, to denote the data fidelity term used
		"""
	print('######################## Pre Processing #########################')
	sys.stdout.flush()
	[prozent_to_reduce,searchradius]=reductionparameter
	name=names[0]
	
	[brightness,baseline,cutedges,Badelefile,threshold,Badprojections]=datahandling

	cutedges=float(cutedges);threshold=float(threshold)
	if prozent_to_reduce!=0. and prozent_to_reduce!=0:
		Badprojections.append(find_bad_projections.Find_Bad_projections(name,searchradius,prozent_to_reduce))
	np_data_coll=[]
	ratio=[]
	Averagenoise=[]
	
	#Loading Data
	Number_Channels=len(names)
	considered=range(Slice_levels[0],Slice_levels[1]+1,Slice_levels[2])
	for name in names:
		averagenoise=0
		if brightness.upper()=='bright'.upper():
			np_data,angles,averagenoise=hdf5.Brightness_correction(name,Badelefile,cutedges,Badprojections)
		elif brightness.upper()=='basic'.upper():
			np_data,angles, averagenoise=hdf5.global_rescaling(name,Badelefile,cutedges,Badprojections)
		else :
			np_data,angles=hdf5.readh5(name)

		[slicenumber,Ny,Nx]=np_data.shape
		if min(Slice_levels[0],Slice_levels[1])<0 or max(Slice_levels[0],Slice_levels[1])>=Ny:
			raise IndexError('Warning!!!, Slice levels are not in a suitable scope. Must lie between 0 and Ny=', str(Ny-1),'. Aborting operation!')
			
		np_data-=np.percentile(np_data,Scalepars[0])
		ratio.append(np.sum(np_data[:,considered])/len(angles))

		np_data=np_data/np.percentile(np_data,Scalepars[1]) 
		np_data_coll.append(np_data)
		Averagenoise.append(averagenoise)		

		Nxoriginal=Nx


	Nz=len(considered)
	Solution=np.zeros([Number_Channels,Nz,Nx,Nx],dtype=np.float32)
	U=np.zeros([Number_Channels,Nz,Nx,Nx],dtype=np.float32)
	New_sinogram=np.zeros([Number_Channels,slicenumber,Nz,Nx],dtype=np.float32)
	resulting_sino=np.zeros([Number_Channels,slicenumber,Nz,Nx],dtype=np.float32)
	print('Computing the slice levels: ',considered)
	sys.stdout.flush()
	

	j=0
	##Iteratively working on Sinogramlevels
	print('########## Starting	'+str(Discrepancy)+' '+str(Regularisation)+' '+str(method)+' ############')
	sys.stdout.flush()
	
	#Decide on suitable GPU device and context
	ctx, my_gpu_devices, GPU_choice = get_gpu_context(GPU_choice)

	import time
	mytime=time.time()	
	for sinogram_level in considered:
		
		sino_coll=[]
		for i in range(0,Number_Channels):
			sino=hdf5.get_sinogram(sinogram_level,np_data_coll[i])

			#Additional Preprocessing (Changing baseline/thresholding)
			if baseline.upper()=='mean'.upper():
				sino=sino-Averagenoise[i]
			if baseline.upper()=='meanthres'.upper():
				mean= np.mean(sino[np.where(sino<threshold)])
				sino=sino-mean
			if baseline.upper()=='thresholding'.upper():
	
				sino[np.where(sino<threshold)]=0

			if baseline.upper()=='mean_threshold_bright'.upper():
				np_data,angles, averagenoise=hdf5.global_rescaling(name,Badelefile,cutedges,Badprojections)
				np_data=np_data-averagenoise
				sino[np.where(sino<threshold)]=0
				sino=hdf5.Brightness_correction2(sino)
			
			sino_coll.append(sino)
		
		

		#Set Data
		U0=[]
		for i in range(0,len(names)):
			U0.append(sino_coll[i].reshape(Nx*slicenumber).T)


		#Execution of TGV code
		TGV_Parameter=(alpha[0],alpha[1])
				
			

		considered2=list(considered)
		considered2.append(sinogram_level)
		start=considered.index(sinogram_level)/float(len(considered))
		currentwidth=1/float(len(considered))
		current=sinogram_level
		Info=[start,currentwidth,current]
		try:
			import radon_tgv_primal_dual_2d as Radon_GPU
			u,Sinonew=Radon_GPU.Reconstructionapproach2d(sino_coll,angles+pi/2,TGV_Parameter,mu,maxiter,ctx,plott,Discrepancy,method if Number_Channels > 1 else 'Uncorrelated 2D Reconstruction',Regularisation,Info)
		except cl.MemoryError as err:
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(err).__name__, err.args)
			eprint(message)	
			raise cl.MemoryError('The data size for a single slice exceeds the available memory, operation aborted.')

			

		#Save result of individual slice 
		U[:,j,:,:]=u
		resulting_sino[:,:,j,:]=Sinonew
		Solution=copy.deepcopy(U)
		New_sinogram=copy.deepcopy(resulting_sino)
		j+=1
		if method=='Uncorrelated 2D Reconstruction':
			meth='UNCORR2D'
		if method=='Frobenius 2D Reconstruction':
			meth='FROB2D'
		if method=='Nuclear 2D Reconstruction':
			meth='NUCL2D'	
			
		
			
		#name=result+Discrepancy+Regularisation+meth+'_'+str(mu)+str(alpha)+'_'+str(maxiter)+'['+str(Slice_levels[0])+','+str(Slice_levels[1])+','+str(Slice_levels[2])+']';	name=name.replace(' ','')
		name=result
		
		#Rescale data
		for i in range(Number_Channels):
			Faktor=np.sum(New_sinogram[i])/ratio[i]
			Solution[i]/=Faktor/len(angles)
			New_sinogram[i]/=Faktor/len(angles)
	
		#Save as Mrc Files
		for i in range(0,Number_Channels):
			mrc=mrcfile.new(name+'_'+Channelnames[i]+'.mrc',overwrite=True)
			mrcsino=mrcfile.new(name+'_'+Channelnames[i]+'_sinogram.mrc',overwrite=True);
			mrc.set_data(Solution[i]);	
			mrcsino.set_data(New_sinogram[i]);

		#Save Collected_data
		if Number_Channels>1:
			
			mrcColl=mrcfile.new(name+'_all_channels.mrc',overwrite=True);
			Collection=np.zeros([Nz,Nx,Number_Channels*Nx],dtype=np.float32)
			for i in range(0,Nz):
				for l in range(0,Number_Channels):
					Collection[i,:,l*Nx:(l+1)*Nx]=Solution[l,i,:,:]
			mrcColl.set_data(Collection);
				
			mrcColl=mrcfile.new(name+'_all_channels_normalized.mrc',overwrite=True);
			Collection=np.zeros([Nz,Nx,Number_Channels*Nx],dtype=np.float32)
			for i in range(0,Nz):
				for l in range(0,Number_Channels):
					Collection[i,:,l*Nx:(l+1)*Nx]=Solution[l,i,:,:]/np.max(Solution[l,:,:,:])
			mrcColl.set_data(Collection);	
			
	print('\nTime required for reconstruction:', int(10.*(time.time()-mytime))/10.,' seconds.\n')
	sys.stdout.flush()
	return Solution

def Coupled_Reconstruction3d(names,result, mu,alpha,maxiter,plott,method,sinogram_level,datahandling,GPU_choice,Channelnames,reductionparameter,Regularisation,Discrepancy,overlay,Chunksize,Scalepars):
	"""
	Reconstruction method employed for 3D reconstruction. Mainly deals with pre and post processing of data and calls reconstruction algorithm
	
	Arguments:
		names ...A list of names corresponding to the data that shall be read
		result ...A string representing the path where results should be saved
		mu ... A list of parameter for the weighting
		alpha A list of TGV parameters
		maxiter An integer representing the number of iterations done by the algorithm
		plott ... A list with 2 entries, the first 1 if a plot shall be constructed, 0 otherwise, the second in procent, i.e. 0.1 for 10% where this gives the amount of replotting of the live results
		method ... The reconstruction method being used, like frobenius or uncorrelated
		Slice_levels ... A list with 3 entries, being used in a range argument (first beginning, second end, last stepsize)
		datahandling ... List of objects, first a string for brightness, than a string for thresholding options,  then in a number smaller 1 how much of the data should be removed on the sides,then the name of a file (potentially '') containing the bad projections, finally a thresholding value if required
		GPU_choice ... An integer depicting which GPU to use
		Channelnames ... A list of names for the respective channels, which which the results are saved
		reductionparameter ... A list with two entries, the procent of projections to remove(e.g. 10% with 0.1) or the absolute number as an integer, and a Search radius (Pixelwidth for smoothing)
		Regularisation ... TV or TGV as String to determine the regularisation functional used
		Discrepancies ... KL or L2 as String, to denote the data fidelity term used
		overlay ... Integer value denoting the degree of overlapping if the problem must be split in subproblems due to memory constraints
		Chunksize ... Allows to limit the maximal size of a split
		"""
	
	
	print('######################## Reduction of Bad Projections#####################')
	sys.stdout.flush()
	[prozent_to_reduce,searchradius]=reductionparameter
	name=names[0]
		#Which Sinogramlevels will be considered
	section=range(sinogram_level[0],sinogram_level[1]+1,sinogram_level[2])
	[brightness,baseline,cutedges,Badelefile,threshold,Badprojections]=datahandling
	cutedges=float(cutedges);threshold=float(threshold)

	if prozent_to_reduce!=0. and prozent_to_reduce!=0:
		Badprojections.append(find_bad_projections.Find_Bad_projections(name,searchradius,prozent_to_reduce))

	np_data_coll=[]
	ratio=[]
	for name in names:
		#Load data and general Preprocessing
		if brightness.upper()=='bright'.upper():
			np_data,angles,averagenoise=hdf5.Brightness_correction(name,Badelefile,cutedges,Badprojections)
		elif brightness.upper()=='basic'.upper():
			np_data,angles, averagenoise=hdf5.global_rescaling(name,Badelefile,cutedges,Badprojections)
		else :
			np_data,angles=hdf5.readh5(name)

		[slicenumber,Ny,Nx]=np_data.shape
		
		if min(sinogram_level[0],sinogram_level[1])<0 or max(sinogram_level[0],sinogram_level[1])>=Ny:
			raise IndexError('Slice levels levels are not in a suitable scope. Must lie between 0 and Ny='+ str(Ny-1)+'. Aborting operation!')

		
		#Data scalation
		ratio.append(np.sum(np_data[:,section])/len(angles))
		np_data-=np.percentile(np_data,Scalepars[0])
		np_data=np_data/np.percentile(np_data,Scalepars[1]) 
		np_data_coll.append(np_data)
	
		Nxoriginal=Nx

	Number_Channels=len(names)

	#Decide on suitable GPU device and context
	ctx, my_gpu_devices, GPU_choice = get_gpu_context(GPU_choice)

	#Get Relevant Sinogram/ thresholding/ baseline
	sino_coll=[]
	for i in range(0,Number_Channels):
		sino=np_data_coll[i]
		if baseline.upper()=='mean1'.upper():
			sino=sino-averagenoise
			
		if baseline.upper()=='mean2'.upper():
			mean= np.mean(sino[np.where(sino<threshold)])
			sino=sino-mean
			
		if baseline.upper()=='thresholding'.upper():
			sino[np.where(sino<threshold)]=0
			
		if baseline.upper()=='mean_threshold_bright'.upper():
			np_data,angles, averagenoise=hdf5.global_rescaling(name,Badelefile,cutedges,Badprojections)
			np_data=np_data-averagenoise
			sino[np.where(sino<threshold)]=0
			sino=hdf5.Brightness_correction2(sino)
		
		sino_coll.append(sino)

	print('Computing the slice levels: ',section)
	sys.stdout.flush()

	for i in range(0,Number_Channels):
		sino_coll[i]=sino_coll[i][:,section,:]
	
	
	
	overlay_s=overlay
	overlay_e=overlay
	#Execute Primal Dual Algorithm
	TGV_Parameter=(alpha[0],alpha[1])
	import radon_tgv_primal_dual_3d as Radon_GPU
	considered=section[:]
	remaining=considered[:]
	relevants=[]
	
	

	ndetectors=sino_coll[0].shape[2]
	nangles=sino_coll[0].shape[0]
	possible=len(remaining)
	Memory_limit=int(Allows_number_slices([ndetectors,ndetectors],[ndetectors,nangles],len(sino_coll),my_gpu_devices[GPU_choice]))#70% of technical limitations
	if Memory_limit>=1:
		possible=min(possible,Memory_limit)
	if Chunksize>=1:
		possible=min(possible,Chunksize)

	U=np.zeros([Number_Channels,Nx,Nx,len(section)])
	New_sinogram=np.zeros([Number_Channels,sino_coll[0].shape[0],sino_coll[0].shape[1],sino_coll[0].shape[2]])
	import time
	mytime=time.time()
	while len(remaining)>0:
		if overlay>possible/4.:
			overlay=int(possible/4)
			eprint('Overlap is too large for possible split size, reducing overlap to '+str(overlay)+' which is a fourth of the possible splits.')
			sys.stdout.flush()
		current=remaining[0:min(possible,len(remaining))]
		startposition=considered.index(current[0])
		endposition=considered.index(current[len(current)-1])
		sinocurrent=[]
		for i in range(Number_Channels):
			sinocurrent.append(sino_coll[i][:,range(startposition,endposition+1),:])

		if possible<len(remaining):
			localending=possible-2*overlay
		else:
			localending=len(remaining)
			
		if len(remaining)==len(considered):
			overlay_s=0
		else:
			overlay_s=overlay
		if possible>=len(remaining):
			overlay_e=len(current)-len(remaining)
		else:
			overlay_e=overlay
			
		relevants=current[overlay_s:len(current)-overlay_e]
		print('\ncurrent',current,'\n')
		sys.stdout.flush()
		start=considered.index(current[0])/float(len(considered))
		currentwidth=(len(current)-overlay_e)/float(len(considered))
		Info=[start,currentwidth,current]		

		try:			
			U_local,New_sinogram_local=Radon_GPU.Reconstructionapproach3d(sinocurrent,angles+pi/2,TGV_Parameter,mu,maxiter,ctx,plott,sinogram_level[2],Discrepancy,method,Regularisation,Info)
			remaining=remaining[localending:len(remaining)]
			
			#Save results from current subproblem
			U[:,:,:,range(startposition+overlay_s,endposition+1-overlay_e)]=U_local[:,:,:,overlay_s:len(current)-overlay_e]
			New_sinogram[:,:,range(startposition+overlay_s,endposition+1-overlay_e),:]=New_sinogram_local[:,:,overlay_s:len(current)-overlay_e,:]
			
			
			
			if method=='Nuclear 3D Reconstruction':
				meth='NUCL3D'
				
			if method=='Uncorrelated 3D Reconstruction':
				meth='UNCORR3D'
			if method=='Frobenius 3D Reconstruction':
				meth='FROB3D'
				
				
			################Saving	Data###################	
			[A,Nx,Ny,Nz]=U.shape
			#name=result+Discrepancy+Regularisation+meth+'_'+str(mu)+str(alpha)+'_'+str(maxiter)+'['+str(sinogram_level[0])+','+str(sinogram_level[1])+','+str(sinogram_level[2])+']';	name=name.replace(' ','')
			name=result
			Solution=np.zeros([A,Nz,Nx,Ny],dtype=float32)
			
			Newsino2=np.zeros([A,New_sinogram.shape[1],New_sinogram.shape[2],New_sinogram.shape[3]],dtype=float32)
			for i in range(0,len(section)):
				Solution[:,i,:,:]=U[:,:,:,i]
				
			for i in range(0,New_sinogram.shape[0]):
				Newsino2[i,:,:,:]=New_sinogram[i,:,:,:]
			
			
			for i in range(Number_Channels):
				Faktor=np.sum(New_sinogram[i])/ratio[i]
				Solution[i]/=Faktor/len(angles)
				Newsino2[i]/=Faktor/len(angles)

				
			#Save as Mrc Files
			for i in range(0,Number_Channels):
				mrc=mrcfile.new(name+'_'+Channelnames[i]+'.mrc',overwrite=True)
				mrcsino=mrcfile.new(name+'_'+Channelnames[i]+'_sinogram.mrc',overwrite=True);
				mrc.set_data(Solution[i]);	
				mrcsino.set_data(Newsino2[i]);	
				
			#Save Collected_data
			if Number_Channels>1:
				mrcColl=mrcfile.new(name+'_all_channels.mrc',overwrite=True);
				Collection=np.zeros([Nz,Nx,Number_Channels*Ny],dtype=np.float32)
				for i in range(0,Nz):
					for j in range(0,Number_Channels):
						Collection[i,:,j*Ny:(j+1)*Ny]=Solution[j,i,:,:]
				mrcColl.set_data(Collection);	
				
				mrcColl=mrcfile.new(name+'_all_channels_normalized.mrc',overwrite=True);
				Collection=np.zeros([Nz,Nx,Number_Channels*Nx],dtype=np.float32)
				for i in range(0,Nz):
					for l in range(0,Number_Channels):
						Collection[i,:,l*Nx:(l+1)*Nx]=Solution[l,i,:,:]/np.max(Solution[l,:,:,:])
				mrcColl.set_data(Collection);	
		
		#Handling cases in which an Error occurs due to lack of memory
		except cl.MemoryError as err:
			possible=int(possible*0.7)
			eprint(	 'Warning: Computation requires too much memory, the problem is split, trying to compute '+str(possible)+' slices at once.')
			sys.stderr.flush()
			#del(ctx)
			#gerat=[my_gpu_devices[GPU_choice]]
			#ctx = cl.Context(devices=gerat)
			
			if possible<1:
				template = "An exception of type {0} occurred. Arguments:\n{1!r}"
				message = template.format(type(err).__name__, err.args)
				eprint(message)
				raise ('Even a single slice requires too much memory.')
				
		except cl.Error as err:
			possible=int(possible*0.7)
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(err).__name__, err.args)
			eprint(message)	

			eprint('Warning: unexpected PyOpenCL error occurred, trying to restart context.')
			sys.stderr.flush()
			#del(ctx)
			#gerat=[my_gpu_devices[GPU_choice]]
			#ctx = cl.Context(devices=gerat)
			if possible<1:
				raise ('Even a single slice creates an unknown error.')
							
	print( '\nTime required for reconstruction:', int(10.*(time.time()-mytime))/10.,' seconds.\n')
	sys.stdout.flush()
	return U

def Coupled_Reconstruction3dbigresu(names,smallerimagestart,result, mu,alpha,maxiter,plott,method,sinogram_level,datahandling,GPU_choice,Channelnames,reductionparameter,Regularisation,Discrepancy,PixelSizeinput,overlay,Chunksize,Scalepars):
	print( '######################## Preprocessing: #####################')
	sys.stdout.flush()
	[prozent_to_reduce,searchradius]=reductionparameter
	name=names[0]	
	
	section=range(sinogram_level[0],sinogram_level[1]+1,sinogram_level[2])
	[brightness,baseline,cutedges,Badelefile,threshold,Badprojections]=datahandling
	cutedges=map(float,cutedges);threshold=map(float,threshold)
	if prozent_to_reduce!=0. and prozent_to_reduce!=0:
		Badprojections.append(find_bad_projections.Find_Bad_projections(name,searchradius,prozent_to_reduce))
	Angles=[]
	Pixelsize=[]
	np_data_coll=[]
	ratio=[]
	Averagenoise=[]
	Averagenoise=[]
	for name in names:
		corresp=name in names[smallerimagestart:len(names)]
	
		averagenoise=0
		#Load data and general Preprocessing
		if brightness[corresp].upper()=='BRIGHT':	
			np_data,angles,averagenoise,Psize=hdf5.Brightness_correction_alternativ(name,Badelefile[corresp],cutedges[corresp],Badprojections[corresp])
		elif brightness[corresp].upper=='BASIC':
			np_data,angles, averagenoise,Psize=hdf5.global_rescaling_alternativ(name,Badelefile[corresp],cutedges[corresp],Badprojections[corresp])
		else :
			np_data,angles,Psize=hdf5.readh5alternative(name)
		Averagenoise.append(averagenoise)

		np_data_coll.append(np_data)
		angles+=np.pi/2
		Angles.append(angles)
		Pixelsize.append(Psize[1])

		
		
	[slicenumber0,Ny0,Nx0]=np_data_coll[0].shape
	[slicenumber1,Ny1,Nx1]=np_data_coll[smallerimagestart].shape
	Number_Channels=len(names)
	Number_Channels0=smallerimagestart
	Number_Channels1=Number_Channels-Number_Channels0

	#Decide on suitable GPU device and context
	ctx, my_gpu_devices, GPU_choice = get_gpu_context(GPU_choice)
		
	#for i in range(0,Number_Channels):
	#	sino_coll[i]=sino_coll[i]/np.mean(sino_coll[i])
	j=Pixelsize.index(max(Pixelsize[0:Number_Channels0]))
	for i in range(Number_Channels0):
		if Pixelsize[i]!=Pixelsize[j] and Pixelsize[i]>0 and Pixelsize[j]>0:
			raise ValueError ('Warning: no uniform Pixelsize in data'+str(name[i])+' and '+str(name[j])+' with sizes '+str(Pixelsize[i])+' and '+str(Pixelsize[i])+'. Aborting computations.')
			
	jj=j		
	j=Pixelsize.index(max(Pixelsize[Number_Channels0:Number_Channels]))
	for i in range(Number_Channels0,Number_Channels):
		if Pixelsize[j]!=Pixelsize[j] and Pixelsize[i]>0 and Pixelsize[j]>0:
			raise ValueError('Warning: no uniform Pixelsize in data'+str(name[i])+' and '+str(name[j])+' with sizes '+str(Pixelsize[i])+' and '+str(Pixelsize[i])+'. Aborting computations.')
			
	
	if Pixelsize[j]>0 and Pixelsize[jj]>0:
		p=Pixelsize[j]/Pixelsize[jj]
		if abs(p-PixelSizeinput)>0.001 and PixelSizeinput>0:
			raise ValueError('Warning: the Pixelratio given as input ('+str(PixelSizeinput)+') does not coincide with the pixel ratio obtained in the data('+str(p)+'). Aborting computations.')
			
	elif PixelSizeinput>0:
		p=PixelSizeinput
	else:
		raise ValueError ('Warning: no suitable information concerning Pixelsize was given in input or in data. Aborting computations.')
		
		
	
	
	#Middlepoints of the pictures are respective reference points
	Refnew=[(Nx0-1)/2.,(Ny0-1)/2.]
	Refold=[(Nx1-1)/2.,(Ny1-1)/2.]
	

	#Which Sinogramlevels will be considered
	section0=range(sinogram_level[0],sinogram_level[1]+1,sinogram_level[2])
	z_=int(np.clip(math.floor(1/p*(min(section0)-Refnew[1])+Refold[1]),0,Ny1-1))
	zu=int(np.clip(math.ceil(1/p*(max(section0)-Refnew[1])+Refold[1]),0,Ny1-1))
	section1=range(z_,zu+1,1)
	Sectionset=[section0,section1]
	print('Computing the Sinogram_levels:',section0)
	print('Corresponds in the Lowresolution Sinograms to ',section1)
	sys.stdout.flush()
	if zu<1 or z_>Ny1:
		eprint ('!!!!!!!!!!! Warning: Section in lowresolution can not cover corresponding Section in High-resolution data!!!!!!')
		sys.stdout.flush()
	
	#import pdb; 
	#pdb.set_trace()	
	
	
	#Data scalation
	for i in range(Number_Channels):
		corresp=i >=smallerimagestart
		np_data=np_data_coll[i]
		ratio.append(np.sum(np_data[:,Sectionset[corresp]]/len(Angles[i])))
		np_data-=np.percentile(np_data,Scalepars[0])
		np_data=np_data/np.percentile(np_data,Scalepars[1]) 
		np_data_coll[i]=np_data
		
	#Get Relevant Sinogram	
	sino_coll=[]
	for i in range(0,Number_Channels):
		corresp=i>=smallerimagestart
		sino=np_data_coll[i]
		#Further Preprocessing
		if baseline[corresp].upper()=='MEAN1':
			sino=sino-Averagenoise[i]
		if baseline[corresp].upper()=='MEAN2':
			mean= np.mean(sino[np.where(sino<threshold[corresp])])
			sino=sino-mean
		if baseline[corresp].upper()=='THRESHOLDING':
			sino[np.where(sino<threshold[corresp])]=0
		if baseline[corresp].upper()=='mean_threshold_bright'.upper():
			np_data,angles, averagenoise=hdf5.global_rescaling(name,Badelefile,cutedges,Badprojections)
			np_data=np_data-averagenoise
			sino[np.where(sino<threshold[corresp])]=0
			sino=hdf5.Brightness_correction2(sino)
		
		sino_coll.append(sino)
		
		
	for i in range(0,Number_Channels0):
		sino_coll[i]=sino_coll[i][:,section0,:]
	for i in range(Number_Channels0,Number_Channels):
		sino_coll[i]=sino_coll[i][:,section1,:]
	Refold_local=Refold[:]
	Refnew_local=Refnew[:]
	
	Refold_local[1]=Refold[1]-z_
	Refnew_local[1]=Refnew[1]-min(section0)

	Topology=[smallerimagestart,p,Refold_local,Refnew_local]
	
	
	overlay_s=overlay
	overlay_e=overlay
	
	#Execute Primal Dual Algorithm
	TGV_Parameter=(alpha[0],alpha[1])
	considered=section0[:]
	considered_small=section1[:]
	remaining=considered[:]
	
	ndetectors=sino_coll[0].shape[2]
	nangles=sino_coll[0].shape[0]
	possible=len(remaining)
	Memory_limit=Allows_number_slices([ndetectors,ndetectors],[ndetectors,nangles],len(sino_coll),my_gpu_devices[GPU_choice])#technical limitations
	if Memory_limit>=1:
		possible=min(possible,Memory_limit)
	if Chunksize>=1:
		possible=min(possible,Chunksize)
	
	U=np.zeros([Number_Channels,Nx0,Nx0,len(section0)])
	New_sino=np.zeros([Number_Channels0,sino_coll[0].shape[0],len(considered),sino_coll[0].shape[2]],dtype=float32)
	New_sino_small=np.zeros([Number_Channels1,sino_coll[Number_Channels0].shape[0],len(considered_small),sino_coll[Number_Channels0].shape[2]],dtype=float32)
	New_sino_high=np.zeros([Number_Channels1,sino_coll[Number_Channels0].shape[0],len(considered),sino_coll[0].shape[2]],dtype=float32)
	import radon_tgv_primal_dual_3dsubsample as Radon_GPU
	relevants=[]
	
	import time
	zeit=time.time()
	while len(remaining)>0:
		if overlay>possible/4.:	
			overlay=int(possible/4)
			eprint('Overlap is too large for possible split size, reducing overlap to '+str(overlay)+' which is a fourth of the possible splits.')
			sys.stdout.flush()
		current=remaining[0:min(possible,len(remaining))]
		startposition=considered.index(current[0])
		endposition=considered.index(current[len(current)-1])
		section0=current[:]
		z_=int(np.clip(round(1/p*(min(section0)-Refnew[1])+Refold[1]),0,Ny1-1))
		zu=int(np.clip(round(1/p*(max(section0)-Refnew[1])+Refold[1]),0,Ny1-1))
		section1=range(z_,zu+1,1)	
		
		if possible<len(remaining):
			a=possible-2*overlay
		else:
			a=len(remaining)
		if len(remaining)==len(considered):
			overlay_s=0
		else:
			overlay_s=overlay
		if possible>=len(remaining):
			overlay_e=len(current)-len(remaining)
		else:
			overlay_e=overlay
			
		relevants=current[overlay_s:len(current)-overlay_e]
		print('\ncurrent',current,'\n', 'relevants:', relevants,'\n', 'corresponding to small',section1,'\n')
		sys.stdout.flush()
		
		
		startposition_small=considered_small.index(section1[0])
		endposition_small=considered_small.index(section1[len(section1)-1])
		sinocurrent=[]
		for i in range(Number_Channels0):
			sinocurrent.append(sino_coll[i][:,range(startposition,endposition+1),:])
		for i in range(Number_Channels0,Number_Channels):
			sinocurrent.append(sino_coll[i][:,range(startposition_small,endposition_small+1),:])
		
		Refold_local[1]=Refold[1]-z_
		Refnew_local[1]=Refnew[1]-min(section0)

		Topology=[smallerimagestart,p,Refold_local,Refnew_local]
				
		start=considered.index(current[0])/float(len(section))
		currentwidth=len(current)/float(len(section))
		Info=[start,currentwidth,current]
		try:
			#import pdb; pdb.set_trace()
			U_local,New_sinogram_local=Radon_GPU.Reconstructionapproach3dsubsmapling(sinocurrent,Angles,Topology,TGV_Parameter,mu,maxiter,ctx,plott,sinogram_level[2],Discrepancy,method,Regularisation,Info)
			remaining=remaining[a:len(remaining)]
			U[:,:,:,range(startposition+overlay_s,endposition+1-overlay_e)]=U_local[:,:,:,overlay_s:len(current)-overlay_e]
			U[:,:,:,range(startposition+overlay_s,endposition+1-overlay_e)]=U_local[:,:,:,overlay_s:len(current)-overlay_e]
			
			New_sino[:,:,range(startposition+overlay_s,endposition+1-overlay_e),:]=New_sinogram_local[0][:,:,overlay_s:len(current)-overlay_e,:]
			
			New_sino_high[:,:,range(startposition+overlay_s,endposition+1-overlay_e),:]=New_sinogram_local[1][:,:,overlay_s:len(current)-overlay_e,:]
			New_sino_small[:,:,range(startposition_small,endposition_small+1),:]=New_sinogram_local[2]
			#maxx=np.max(U)
				
			
				
			################Saving	Data###################	
			if method=='Uncorrelated 3D Subsampling Reconstruction':
				meth='SubsUncor3D'
			if method=='Frobenius 3D Subsampling Reconstruction':
				meth='SubsFROB3D'
			[A,Nx,Ny,Nz]=U.shape
			#name=result+Discrepancy+Regularisation+meth+'_'+str(mu)+str(alpha)+'_'+str(maxiter)+'['+str(sinogram_level[0])+','+str(sinogram_level[1])+','+str(sinogram_level[2])+']';	name=name.replace(' ','')
			#name=name.replace(' ','')
			name=result
			Solution=np.zeros([A,Nz,Nx,Ny],dtype=float32)
			New_sinogram= copy.deepcopy(New_sino)
			New_sinogram_high= copy.deepcopy(New_sino_high)
			New_sinogram_small= copy.deepcopy(New_sino_small)
			
			
			
			#Newsino2=np.zeros([A,New_sinogram.shape[1],New_sinogram.shape[2],New_sinogram.shape[3]],dtype=float32)
			for i in range(0,len(considered)):
				Solution[:,i,:,:]=U[:,:,:,i]
				
			#Solution[:,:,0,0]=maxx#Damit skala passt

			
			for i in range(Number_Channels0):
				Faktor=np.sum(New_sinogram[i])/ratio[i]
				Solution[i]/=Faktor/len(Angles[i])
				New_sinogram[i]/=Faktor/len(Angles[i])
				
				
			for i in range(Number_Channels1):
				Faktor=np.sum(New_sinogram_small[i])/ratio[i+Number_Channels0]
				Solution[i+Number_Channels0]/=Faktor/len(Angles[i+Number_Channels0])
				New_sinogram_small[i]/=Faktor/len(Angles[i+Number_Channels0])
				New_sinogram_high/=Faktor/len(Angles[i+Number_Channels0])
				

			
			#Save as Mrc Files
			for i in range(0,Number_Channels):
				mrc=mrcfile.new(name+'_'+Channelnames[i]+'.mrc',overwrite=True)
				mrc.set_data(Solution[i]);	
					
			for i in range(Number_Channels1):
				mrcsino=mrcfile.new(name+'_'+Channelnames[i+Number_Channels0]+'_sinogram_smallresu.mrc',overwrite=True);
				mrcsino.set_data(New_sinogram_small[i]);
				mrcsino=mrcfile.new(name+'_'+Channelnames[i+Number_Channels0]+'_sinogram.mrc',overwrite=True);
				mrcsino.set_data(New_sinogram_high[i]);
			for i in range(Number_Channels0):
				mrcsino=mrcfile.new(name+'_'+Channelnames[i]+'_sinogram.mrc',overwrite=True);
				mrcsino.set_data(New_sinogram[i]);
			
			#Save Collected_data
			mrcColl=mrcfile.new(name+'_all_channels.mrc',overwrite=True);
			Collection=np.zeros([Nz,Nx,Number_Channels*Ny],dtype=np.float32)
			for i in range(0,Nz):
				for j in range(0,Number_Channels):
					Collection[i,:,j*Ny:(j+1)*Ny]=U[j,:,:,i]
			mrcColl.set_data(Collection);	
			
			mrcColl=mrcfile.new(name+'_all_channels_normalized.mrc',overwrite=True);
			Collection=np.zeros([Nz,Nx,Number_Channels*Nx],dtype=np.float32)
			for i in range(0,Nz):
				for l in range(0,Number_Channels):
					Collection[i,:,l*Nx:(l+1)*Nx]=Solution[l,i,:,:]/np.max(Solution[l,:,:,:])
			mrcColl.set_data(Collection);	
			
		except cl.MemoryError as err:
			possible=int(possible*0.7)
			eprint(	 'Warning: Computation requires too much memory, the problem is split, trying to compute '+str(possible)+' slices at once.')
			sys.stderr.flush()
			#del(ctx)
			#gerat=[my_gpu_devices[GPU_choice]]
			#ctx = cl.Context(devices=gerat)
			
			if possible<1:
				template = "An exception of type {0} occurred. Arguments:\n{1!r}"
				message = template.format(type(err).__name__, err.args)
				eprint(message)
				raise ('Even a single slice requires too much memory.')
				
		except cl.Error as err:
			possible=int(possible*0.7)
			template = "An exception of type {0} occurred. Arguments:\n{1!r}"
			message = template.format(type(err).__name__, err.args)
			eprint(message)	

			eprint('Warning: unexpected PyOpenCL error occurred, trying to restart context.')
			sys.stderr.flush()
			#del(ctx)
			#gerat=[my_gpu_devices[GPU_choice]]
			#ctx = cl.Context(devices=gerat)
			if possible<1:
				raise ('Even a single slice creates an unknown error.')

				
				
	print('\nTime required for reconstruction:', int(10.*(time.time()-zeit))/10.,' seconds.\n')
	sys.stdout.flush()
	return U

			
			


#To	 start from console, provide parameters
def main():
	#Parse command line arguments-----------------------------------------------
	parser = argparse.ArgumentParser(description='Radon transform reconstruction tool based on coupled multi-data Tikhonov regularization for multi-channel projection data. A variety of preprocessing options are available to tackle possible inconsistencies or problems in the data. Offers several multi-channel TV and TGV regularization methods, in particular uncorrelated, Frobenius norm and nuclear norm approaches in 2D and 3D, as well as L2 and Kullback-Leibler divergence discrepancies.\n This means the program solves, for N=\'Number of channels\' and f_1, ..., f_N given projection data, the problem argmin_{u_1, ..., u_N} {\sum_{i=1}^N \mu_i D(T u_i,f_i)+R(u)} with respect to the reconstructions u_1, ..., u_N, where T the Radon Transform, D either the squared L2-discrepancy or Kullback-Leibler divergence, and R a multi-channel TV or TGV functional. This optimization problem is solved via a primal-dual proximal algorithm (splitting method). The program can be started for example via "python Reconstruction_coupled.py example/Al_EDXsino.mrc example/HAADF_lrsino. --SliceLevels 50 55 1 --Coupling=frob3d --Datahandling bright thresholding 0.0 "" 0.05 --Outfile Result/results --Discrepancy KL --Maxiter 2000 --Plot 1 0.05 --GPU_Choice=0 --alpha 4 1 --Channelnames Alum HAADF --Find_Bad_Projections 0 --mu 0.008 0.5  --Badprojections 0 20 21 22 23 24 --Regularisation=TGV". Copyright 2019 by Richard Huber, Martin Holler, Kristian Bredies, Institute of Mathematics and Scientific Computing, University of Graz, Austria.')
	parser.add_argument('Infiles',type=str,nargs='+',
						help='Names of input files (.mrc files, see manual for detailed description) containing projection data. All sinograms need to have the same dimensions (3D array of size #projections x #detectors x #slices). Additionally, a text file with the same name but suffix .rawtlt containing the corresponding angle values in degree (one per line), or a csv file (same name with suffix .csv) where the angles are delimited by ",". Several names may be entered which are then processed jointly.')

	parser.add_argument('--Infiles_Smallresolution',default=[],type=str,nargs='+',
						help='Names of input h5 or mrc files of a second projection data set (with the same requirements as for --Infiles) in a different (lower) resolution. This is only relevant when using subsampling.')
   
	parser.add_argument('-o','--Outfile',default="reconstruction",
						help='Prefix of the output files, e.g. Results/example1/data. Several MRC files are created when the algorithm terminates: The individual reconstructions <prefix><channelname>.mrc (see option --Channelnames), the sinograms of the reconstructions <prefix><channelname>sinogram.mrc, and two collection files containing all reconstructions in a single file (<prefix>all_channels.mrc in correct ratio, <prefix>all_channels_normalized.mrc with normalized values).') 
	
	parser.add_argument('--SliceLevels',
						type=int, default=[0] ,nargs='+',
						help='Slices that shall be reconstructed. This option takes up to 3 numbers representing an interval in the slice dimension: The first number represents the lower bound, the second the upper bound, and third the step size, e.g. 30 60 5 represents the slices [30,35,40,45,50,55,60]. The considered slice numbers will be printed. When entering less than 3 numbers, the algorithm adds the missing such that the step size is 1 if two numbers are given, and only one slice is reconstructed if one number is entered.')

	parser.add_argument('-m', '--Maxiter',
						type=int, default=2000,
						help='Number of iterations the primal-dual algorithm will run (default: 2000). The number of iterations obviously directly impacts the run time, on the other hand the required number of iterations may vary depending on the problem and parameters.')
							
	parser.add_argument('--Regularisation',default='TGV',type=str ,help='The regularization functional to be used in the Tikhonov approach: Options are \'Total Generalized Variation\' via TGV (default) and \'Total Variation\' via TV.')

	parser.add_argument('--Coupling',type=str, default='', help='Specifies the dimension and norm used in the regularization functional. This provides a coupling effect between different slices and channels. Possible choices are: Frob2d, Nucl2d, Uncorr2d, Uncorr3d, Frob3d (default), FrobSubs, Uncorrsubs. "Frob2d" gives a 2D slice-wise TV/TGV with Frobenius norm coupling across channels. "Nucl2d" uses 2D TV/TGV with nuclear norm coupling. "Uncorr2d" uses 2D TV/TGV witout multi-channel coupling. "Frob3d" provides 3D TV/TGV with Frobenius norm coupling across slices and channels. "Frobsubs" activates reconstruction from two data sets of different resolution (see --Infiles_Smallresolution) and uses 3D TV/TGV with Frobenius norm coupling. "Uncorrsubs" does the same but without channel-wise coupling (not adviced). Most standard problems are best solved using "Frob3d" as it couples the channels and computes 3D reconstructions that take similarities along different slices into account.')
   
	parser.add_argument('--Discrepancy',default='KL',type=str,help='Specifies which discrepancy function (data fidelity function) is used in the regularized problem. Options are the Kullback-Leibler divergence via "KL" (default) and the squared L2-norm, i.e., the sum of squares of the difference, via "L2". The choice depends on the noise in data. For Poisson noise, KL is more reasonable, while for Gaussian noise, L2 is the better choice. Warning: the weighting parameters mu strongly depend on the choice of discrepancy!')
	
	parser.add_argument('--mu', type=float, default=[],nargs='+',
						help='Parameter which weighs data terms in the optimization problem. These are associated to the data sets in the order corresponding to the order of data given via --Infiles, and thereafter the order of --Infiles_Smallresolution in case of subsampling. If not enough parameters are entered, the remaining parameters mu_i will be set to 1, i.e., the case of 3 channels and --mu 4 0.5 will be handled as [4,0.5,1]. The following basic rule for finding mu applies: The more noise there is, the smaller the corresponding weight should be!')
   
	parser.add_argument('--alpha',type=float,default=[4,1.],nargs='+', help='Regularization parameter. For TGV^2 two positive values are required, for TV only one positive value has to be given. The default is (4,1); excessive parameters will be ignored.')
	
	parser.add_argument('--Plot',nargs='+', default=[0], help='Determines whether and how a live plot during reconstruction is shown by up to two parameters. If the first parameter is 1, preliminary results will be plotted during the reconstruction, otherwise not (Default: 0). The second value specifies the frequency of updates of the plot, either by a float in [0,1] representing the fraction of the total iterations (e.g., 0.1 for every 10 percent), or by an integer representing the number of iterations (e.g., 100 for every 100 iterations).')	
  
	parser.add_argument('--Pixelratio',type=float,nargs='+', default=[-1], help='Allows for manual input of the pixel ratio when using subsampling (does not have any impact otherwise). The ratio is defined as the pixel size of the low resolution data divided by the pixel size of the high resolution data (i.e., typically larger than 1). Alternatively, one can enter the two pixel sizes and the corresponding ratio will be computed automatically. The default value -1 corresponds to no input. If the pixel ratio is given in h5 data and via manual input, a warning is issued if these do not coincide and the operation will be aborted.')	 
	
	parser.add_argument('--Datahandling',nargs='+',default=['bright','thresholding','0.0','','0.05'], help='Configures the preprocessing steps applied to the data before reconstruction. The first parameter specifies the brightness correction: "bright" (default and adviced) if brightness correction should be used (rescales all projections so that they have a common mean in order to counter intensity fluctuations), and "basic" if no brightness correction, but further preprocessing shall be performed. The second parameter specifies how noise in the data outside the object is removed: "thresholding" (default) with a value given in the 5th entry, "mean" computes the mean outside the object and subtracts (where outside corresponds with the width parameter which is entered as third parameter), "meanthres" subtracts the mean of values below a threshold. The third parameter (default: 0) determines the fraction the sinogram data should be cut off from the outside (for "mean" it is required to be not zero). This can in particular be useful to reduce workload and memory requirements in case the object could be encapsuled in a smaller circle than currently present. The fourth parameter is the name of a text file containing the numbers of bad projections that should be removed before reconstruction (i.e., a file in which each line contains a number of a projection). This might be useful if some projections contain very poor data, e.g., due to defraction or alignment issues. The fifth parameter is the thresholding value (only relevant for "thresholding" or "meanthres"). In particular, if this value is 0, no thresholding will be applied. (Default parameters: bright, thresholding, 0.0000001, \'\', 0.05). Missing entries will be filled with the default.')
	  
	parser.add_argument('--GPU_Choice',type=int,default=-1,help='Specifies the OpenCL device used for the computations. If not used, the program will ask which device to use. The numbers 0 1 2 ... can be specified to determine which device to use (these correspond to the ordering of the devices of the OpenCL plugin with GPUs coming first.)')
	
	parser.add_argument('--Channelnames',default=[],type=str,nargs='+',help='Name of the channels for the reconstructed data. Solutions will be saved as <prefix><channelname[i]>.mrc (default: channel0, channel1, ...). If too few names are given, they will be filled, i.e., --Channelnames A, B and 4 channels will result in A, B, channel2, channel3.')
  
	parser.add_argument('--Badprojections',default=[],type=int,nargs='+',help='Specifies the projections that should not be used for the computation in a direct way. These are combined with the information from the bad projection file (see --Datahandling, fourth parameter). By default, no projections are removed.')
	
	parser.add_argument('--Datahandling2',nargs='+',default=['bright','thresholding','0.0','','0.05'],help='Analogous to --Datahandling, works on the data sets of lower resolution (if omitted, will be defaulted to --Datahandling for the high resolution data).')
	
	parser.add_argument('--Badprojections2',default=[],type=int,nargs='+',help='Analogously to --Badprojections, but for the low resolution data. Defaults to no --Badprojections.')
   
	parser.add_argument('--Find_Bad_Projections',nargs='+',default=[],type=float,help='Options for bad projection determination. Deactivated when 0, otherwise, two parameters are required: The first parameter specifies the fraction of the projections to be discarded (e.g., 0.1 for 10 percent) or the absolute number (e.g., 10 to discard the 10 worst projections). The second parameter specifies the allowed shift the projections can make from one angle to the next as a fraction of the total width. Default values: 0.1, 1/30.')
 
	parser.add_argument('--Overlapping',default=1,type=int,help='Specifies overlap in case of split computation. If the OpenCL device does not possess a sufficient amount of memory, the program will try to split the problem (splitting the slices being reconstructed). This might lead to issues at the slices where the split occurs. Slightly overlapping the subproblems could lead to more consistency. This parameter specifies the number of slices the resulting subproblems overlap. E.g., if the problem is [1,2,...,10] and we can compute 6 slices in parallel, an overlap of 1 would lead to slices [1...6] being processed but only [1...5] of the results being saved, and computing [5...10] but saving only [6...10]. So this overlap leads to larger subproblems and hence, to redundancy and slightly increased computational effort. The default value is 1, i.e., one overlapping slice is used. A value of 0 deactivates the feature. Depending on the size of the slices, greater overlaps might also be reasonable.')

	parser.add_argument('--Chunksize',default=-1,type=int,help='Allows to manually limit the number of slices of the data to be used per computation. If the chunksize parameter set lower than the total number of slices, the program will try to split the problem into (slightly overlapping) subproblems where in each subproblem only as many slices as given with this parameter will be used. The default value is -1, which sets no limit on the number of used slices. This option allows to manually limit the total memory that will be required by the software. While generally the software will try to split the problem automatically in case of insufficient memory, this option is intended in particular for cases where the automatic splitting delivers unsatisfactory results.')
	
	parser.add_argument('--Scalepars',default=[1,98],type=int,nargs='+',help='List of two parameters that allow to adapt the scaling of the data that is carrried out as preprocessing step. If Scalepars=[mn,mx], the data will be scaled such that the mn-th percentile is around 0 and the mx-th percentile is around 1')
	
	args = parser.parse_args()
	# Check if directory to outfile exits
	outfilepath = os.path.split(args.Outfile)[0]#'/'.join( args.Outfile.split('/')[:-1] )
	print('Output will be written to ' + outfilepath)
	sys.stdout.flush()
	if not os.path.exists( outfilepath ) and outfilepath!='':
		os.makedirs(outfilepath)
	find_bad_projectionsoptions=args.Find_Bad_Projections

	if len (find_bad_projectionsoptions)==0:
		find_bad_projectionsoptions.append(0)
	for i in range(len(find_bad_projectionsoptions),2):
		find_bad_projectionsoptions.append(None)
	find_bad_projectionsoptions=find_bad_projectionsoptions[0:4]

	Regularisation=args.Regularisation
	Regularisation=Regularisation.upper()
	if Regularisation not in ['TGV','TV']:
		eprint('Regularisation '+str(Regularisation)+' not recognised, changed to TGV')
		sys.stdout.flush()
		Regularisation='TGV'
	

	#Adjusting Input	
	TGV=args.alpha
	if len(TGV)==1 :
		TGV.append(1.)
		TGV=TGV[0:2]
		eprint('Warning: not sufficiently many regularization parameters "alpha" are given, setting second entry to one resulting in '+str(TGV)+'.')
		sys.stdout.flush()
	else:
		TGV=TGV[0:2]
		
	names=args.Infiles
	methode=args.Coupling
  
	
	#Interpreting method
	methode=methode.upper()
	setting=-1
		
	if methode	in ['FROB2D','FROBENIUS2D','FROBENIUS','FROB']:
		methode='Frobenius 2D Reconstruction'
		setting = 0
	if methode	in ['NUCL2D','NUCLEAR2D','NUCLEAR','NUC2D']:
		methode='Nuclear 2D Reconstruction'
		setting = 0
	if methode	in ['UNCORRELATED','UNCORRELATED2D','UNCORR','UNCORR2D','UNCOR','UNCOR2D']:
		methode='Uncorrelated 2D Reconstruction'
		setting = 0
	if methode in ['NUCL3D','NUCLEAR3D','NUC3D','NUCLE3D']:
		 methode= 'Nuclear 3D Reconstruction'
		 setting=1
	if methode	in ['FROBENIUS3D','3D','FROB3D']:
	   methode= 'Frobenius 3D Reconstruction'
	   setting = 1
	if methode in ['UNCORRELATED3D','UNCOR3D','UNCORR3D']:
		methode='Uncorrelated 3D Reconstruction'
		setting=1
	if methode	in ['FROBENIUS_SUBSAMPLING','SUBSAMPLING_FROBENIUS','FROSUBS','SUBS','SUB','FROBSUBS','FROBSUB','FROSUB']:	
		methode= 'Frobenius 3D Subsampling Reconstruction' 
		setting=2
	if methode	in ['UNCORRELATED_SUBSAMPLING','SUBSAMPLING_UNCORRELATED','UNCORSUBS','UNCORRSUBS','UNCORSUB','UNSUB','UNSUBS']:  
		methode= 'Uncorrelated 3D Subsampling Reconstruction' 
		setting=2


	if setting ==-1:
		eprint('!!!!!!!!!!!! Warning: method '+str(methode)+ 'is not known, setting method to \'Frobenius 2D Reconstruction\'.')
		sys.stdout.flush()
		setting=0;methode='Frobenius 2D Reconstruction'

	if args.Infiles_Smallresolution==[] and	 setting ==2:
		setting=1
		eprint('Warning !!!! Only one type of data is given, changing method to \'Frobenius 3D Reconstruction\'.')
		sys.stdout.flush()
		methode= 'Frobenius 3D Reconstruction'
	if setting == 2:
		imagestart=len(names)
		names=names+args.Infiles_Smallresolution
		
	for j in range(0,len(names)):
		#if names[j][len(names[j])-3:len(names[j])]=='.h5':
		#	names[j]=names[j][0:len(names[j])-3]
		nameold=names
		if '.h5' not in names[j] and '.mrc' not in names[j]:
			if names[j][len(names[j])-1]=='.':
				names[j]=names[j][0:len(names[j])-1]
			if os.path.isfile(names[j]+'.h5')==True:
				names[j]=names[j]+'.h5'			
			elif os.path.isfile(names[j]+'.mrc')==True:
				names[j]=names[j]+'.mrc'
			else:
				raise IOError('The file with name',names[j],  'has not been found. Operation aborted.')
				
				
		if names[j]!=nameold[j]:
			eprint('Warning!!!!! Name \'',namesold[j],'\' could not be interpreted correctly, changed to\'', names[j],'\'.')
			sys.stdout.flush()
			
			
		eprint('Names',names)
		sys.stdout.flush()
		if os.path.isfile(names[j])==False:
			raise IOError ('The file with name',names[j],  'has not been found. Operation aborted.')
			
			
	Discrepancy=(args.Discrepancy).upper()
	if Discrepancy not in ['KL','L2']:
		eprint('Warning!!!!!! Discrepancy input not understood, set to KL (Kullback-Leibler). Possible choices are KL or L2.')
		sys.stdout.flush()
		Discrepancy='KL'
  

	methode=methode
	#Completing Datahandling Options
	datahandling=args.Datahandling
	datahandling2=args.Datahandling2
	if len(datahandling)==0:
		datahandling.append('bright')
	if len(datahandling)==1:
		datahandling.append('thresholding')
	if len(datahandling)==2:
		datahandling.append('0.000001')
	if len(datahandling)==3:
		datahandling.append('')
	if len(datahandling)==4:
		datahandling.append('0.05')
	datahandling=datahandling[0:5]

	j=len(datahandling2)
	if j <=5:
		datahandling2=datahandling2+datahandling[j:5]
   
	datahandling2=datahandling2[0:5]
	datahandling.append(args.Badprojections)
	datahandling2.append(args.Badprojections2)
	Datahandling=[]
	for i in range(0,len(datahandling)):
		Datahandling.append([datahandling[i],datahandling2[i]])
	

	
	
	
   #Default mu values are 1
	Mu=args.mu
	mu=list(np.ones([len(names)],dtype=float))
	for i in range(0,min(len(Mu),len(names))):
		mu[i]=Mu[i]
  #	 for i in range(0,len(mu)):
	#	if Discrepancy=='KL':
	#		mu[i]*=0.0001
   
	#Default Channelnames are Channel1,Channel2...
	Channels=args.Channelnames
	Channelnames=[]
	for i in range(0,len(names)):
		Channelnames.append('Channel'+str(i))
	for i in range(0,min(len(Channels),len(names))):
		Channelnames[i]=Channels[i]
		
	#Default only one slice, and default stepsize 1
	
	SinogramLevels=args.SliceLevels
	if len(SinogramLevels)==1:
		SinogramLevels.append(SinogramLevels[0]);
	if len(SinogramLevels)==2:
		SinogramLevels.append(1)
	if (SinogramLevels[1]+1-SinogramLevels[0])*SinogramLevels[2]<=0:
		raise ValueError('Slice instructions are not usable, as entries are non consistent (start needs to be smaller or equal than end and step size needs to be larger or equal to 1). Aborting operation!')
		
   
	
	print(' ')
	sys.stdout.flush()
	#Diverse Parameter
	result=args.Outfile
	GPU_choice=args.GPU_Choice

	Pixelsize=args.Pixelratio
	if len(Pixelsize)==1:
		Pixelsize=Pixelsize[0]
	elif len(Pixelsize)==2:
		Pixelsize=Pixelsize[0]/Pixelsize[1]
	
	N=args.Maxiter
	plott =args.Plot
	plott[0]=int(plott[0])	
	if len(plott)==1:
		plott.append(0.1)
	if len(plott)>2:
		plott=plott[0:2]
	if len(plott)==2:
		plott[1]=float(plott[1])
	if plott[1]>1:
		plott[1]=plott[1]/N

	if plott[0]==1 and plott[1]<=0:
		eprint('Warning!!! Plot Instructions make no sense, as plotting is set True but frequency parameter is not positive. Setting plot frequency to every 10%.')
		sys.stdout.flush()
		plott[1]=0.1

	#Save Configfile 
	Str=result+methode+'_'+str(mu)+str(TGV)+'_'+str(N)+'Layer'+str(SinogramLevels);Str.replace(' ', '')
	os.system('echo " '+ str(args)+'" > "'+Str+'.config"')
	
	overlay=args.Overlapping
	
	
	#execute Reconstruction code
	if setting==2:
		u_reconst=Coupled_Reconstruction3dbigresu(names,imagestart,result,mu,TGV,N,plott,methode,SinogramLevels,Datahandling,GPU_choice,Channelnames,find_bad_projectionsoptions,Regularisation,Discrepancy,Pixelsize,overlay,args.Chunksize,args.Scalepars)
	elif setting==1:
		 u_reconst=Coupled_Reconstruction3d(names,result,mu,TGV,N,plott,methode,SinogramLevels,datahandling,GPU_choice,Channelnames,find_bad_projectionsoptions,Regularisation,Discrepancy,overlay,args.Chunksize,args.Scalepars)
	else:
		 u_reconst=Coupled_Reconstruction2d(names,result,mu,TGV,N,plott,methode,SinogramLevels,datahandling,GPU_choice,Channelnames,find_bad_projectionsoptions,Regularisation,Discrepancy,args.Scalepars)

	print('\nAlgorithm completed, results are written in '+outfilepath+'.')
	sys.stdout.flush()
	
	return


if __name__ == "__main__":
	main()
	
