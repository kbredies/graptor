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
## radon_tgv_primal_dual_2d.py:
## Code for 2D variational reconstruction of tomographic data.
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

from numpy import *
import matplotlib
#matplotlib.use('gtkagg')
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy

import sys

""" Reconstruction of tomographic data for a single slice (i.e. 2D) using Primal Dual optimization
Input:	
		sino		... A list(length is Number_Channels) containing the sinogram data (Array with shape [Number_of_angles,Nx])
		angles		... A list or array of angles corresponding to Radon transform
		Parameters	... Regularization parameters corresponding to TGV or TV (List, array or scalar)
		mu			... Weights of the subproblems (must be same length as Number_Channels) (array or list)
		maxiter		... Number of iterations used in the primal dual algorithm (scalar)
		ctx			... A pyopencly context corresponding to the device used for GPU implementation
		plott		... A list with two entries, first is 0 or 1 for no live plott or live plott, second for frequence of updates (between 0 and 1 transformed in percent of reconstruction, for greater 1 corresponds to every so much iterations)
		discrepancy	... A string stating whether L2 or KL should be used as discrepancy functions. Options 'L2' or 'KL'
		regularisationmethod	...	 A string stating what regularisation method is used for coupled regularization, i.e. uncorrelated, Frobenius or Nuclear coupling. Options are 'Uncorrelated 2D Reconstruction', 'Frobenius 2D Reconstruction','Nuclear 2D Reconstruction'
		regularization ...String: state what regularization functional to consider, options 'TV' or 'TGV'
		Info		... A list of additional plotting information containing information on the overall progress of the reconstruction, where first entry corresponds to the initial slice, the second on how many slices are to be reconstructed at all and the last on at which slice we currently are.

Output:
		Solution   ... A numpy array containing the reconstuction
		Sinogram   ... A numpy array containing the sinograms corresponding to the solution
"""
def Reconstructionapproach2d(sino,angles,Parameter,mu,maxiter,ctx,plott,discrepancy,regularisationmethod,regularisation,Info):
	start=Info[0]
	sectionwidth=Info[1]
	current=Info[2]
	
	#This is a hack that avoids a bug that seems to occur when zero initializing arrays of size >2GB with clarray.zeros
	def zeros_hack(*args, **kwargs):
		res = clarray.empty(*args, **kwargs)
		res[:] = 0
		return res
	clarray.zeros = zeros_hack


	#Create Py Opencl Program
	class Program(object):
		def __init__(self, ctx, code):
			self._cl_prg = cl.Program(ctx, code)
			self._cl_prg.build()
			self._cl_kernels = self._cl_prg.all_kernels()
			for kernel in self._cl_kernels:
					self.__dict__[kernel.function_name] = kernel
	
	#Choose Context, GPUdevice			
	queue = cl.CommandQueue(ctx)

######Kernel Code (Actual code for computation on the Graphical Processing Unit)
	prg = Program(ctx, r"""
	
	//Projection of V wrt Frobeniusnorm
	__kernel void update_NormV_frob(__global float2 *V,__global float *normV,const float alphainv,const float NumberOfChannels){
	  size_t Nx = get_global_size(0), Ny = get_global_size(1);
	size_t x = get_global_id(0), y = get_global_id(1);
	size_t i = Nx*y + x;

	
	//Computing Norm
	float norm=0;
	for(int j=0;j<NumberOfChannels;j++){
	size_t k=i+Nx*Ny*j;
	norm= hypot(hypot(V[k].s0,V[k].s1),norm);
	}
	norm=norm*alphainv;
	normV[i] =norm;
	
	//Projection
	if (norm > 1.0f) 
	{for(int j=0;j<NumberOfChannels;j++)
	{size_t	 k=i+Nx*Ny*j;
	V[k]=V[k]/norm;
	}}
	}
	
	
	
		//Projection Nuclear Norm
	__kernel void update_NormV_Nucl(__global float2 *V,__global float *normV,const float alphainv,const float NumberOfChannels){
	  size_t Nx = get_global_size(0), Ny = get_global_size(1);
	size_t x = get_global_id(0), y = get_global_id(1);
	size_t i = Nx*y + x;

    float eps=0.00001f;
    float S1, S2; 
	float a=0,b=0,c=0;
	float3 VSV;
	float sigma1,sigma2;
	float norm1,norm2;
	norm1=0;
	norm2=0;
	
	//Computing A*A (A adjoint times A)
	for(int j=0;j<NumberOfChannels;j++)
	{size_t k=i+Nx*Ny*j;
	a+=V[k].s0*V[k].s0;
	b+=V[k].s0*V[k].s1;
	c+=V[k].s1*V[k].s1;
	}
	
	//Computing	 U Unitary Matrix of A*A
	float o=min(a,c);
	if((b*b)/(o*o)>eps && o*o>0){//Case diagonals are not dominant
	float d=a+c;
	float e=a-c;
	
	float f=4*b*b+e*e;
	if (f<0){f=0;}else{f=pow(f,0.5f);}

	if (d-f>0){
	sigma2=sqrt((d-f)*0.5f);}else{sigma2=0;
	}
	if (d+f>0){
	sigma1=sqrt((d+f)*0.5f);}else{sigma1=0;} 
	
	
	float g=(e+f)*0.5f;
	float h=(e-f)*0.5f;
	float4 v;

	v.s0=g*a+b*b;
	v.s1=g*b+c*b;//v1

	v.s2=h*a+b*b;
	v.s3=h*b+b*c; //v2

	norm1=(hypot(v.s0,v.s1));
	norm2=(hypot(v.s2,v.s3));
	if (norm1>0)
	v.s01=v.s01/norm1;
	if(norm2>0)
	v.s23=v.s23/norm2;
	
	 
	if ((sigma1*alphainv)>1) S1=1/(sigma1*alphainv);  else S1 = 1.0f;
	if ((sigma2*alphainv)>1) S2=1/(sigma2*alphainv);  else S2 = 1.0f;


	VSV.s0=v.s0*v.s0*S1+v.s2*v.s2*S2;
	VSV.s2=v.s1*v.s1*S1+v.s3*v.s3*S2;
	VSV.s1=S1*v.s0*v.s1+S2*v.s2*v.s3;
	}else{//case diagonals are Dominant, or Matirx constant 0
		sigma1=pow(a,0.5f);
		sigma2=pow(c,0.5f);
		if ((alphainv*sigma1)>1) S1=1/(alphainv*sigma1);  else S1 = 1.0f;
		if ((alphainv*sigma2)>1) S2=1/(alphainv*sigma2);  else S2 = 1.0f;
		VSV.s0=S1;
		VSV.s2=S2;
		VSV.s1=0.0f;
	}
	
	//Compute A*V*S*V
	for(int j=0;j<NumberOfChannels;j++)
	{size_t k=i+Nx*Ny*j;
	V[k]=V[k].s0*VSV.s01+V[k].s1*VSV.s12;
	}

	normV[i]=sigma1+sigma2;
	}
	
	
		//Projection  wrt Frobeniusnorm
	__kernel void update_NormV_unchor(__global float2 *V,__global float *normV,const float alphainv,const float NumberOfChannels) {
  
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1),z=get_global_id(2);
  
  size_t i = Nx*y + x+Nx*Ny*z;

	//Computing Norm
	float norm=0;	
	norm= hypot(V[i].s0,V[i].s1)*alphainv;
	if (norm > 1.0f) {
	V[i]/=norm;}
	}
	
		//Projection wrt Tensor-Frobeniusnorm
  __kernel void update_NormW_frob(__global float4 *W,__global float *normW,const float alphainv,const float NumberOfChannels){
	size_t Nx = get_global_size(0), Ny = get_global_size(1);
	size_t x = get_global_id(0), y = get_global_id(1);
	size_t i = Nx*y + x;

	//Computing Norm
	float norm=0;
	for(int j=0;j<NumberOfChannels;j++)
	{size_t k=i+Nx*Ny*j;
	norm=hypot(hypot(W[k].s0,W[k].s1),hypot(2.f*W[k].s2,norm));
	}
	norm =alphainv*norm;
	normW[i]=norm;
	
	//Projection
	if (norm > 1.0f) 
	{for(int j=0;j<NumberOfChannels;j++)
	{size_t k=i+Nx*Ny*j;
	W[k]=W[k]/norm;
	}}
	}
	
	__kernel void update_NormW_unchor(__global float4 *W,__global float *normW,const float alphainv, const float NumberOfChannels){
	size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
	size_t x = get_global_id(0), y = get_global_id(1),z=get_global_id(2);
	size_t i = Nx*y + x+Nx*Ny*z;
	
	//Computing Norm
	float norm=0;
	
	
	norm=hypot(hypot(W[i].s0,W[i].s1),1.4142f*W[i].s2)*alphainv;
	if (norm > 1.0f) {
	W[i]=W[i]/norm;}
	
	}
	
		__kernel void update_NormW_empty(){
	
	}
	
	
	__kernel void update_v(__global float2 *v, __global float *u,
					   __global float2 *p,
					   const float sigma, const float alphainv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1),channel=get_global_id(2);
  size_t i = Nx*y + x +Nx*Ny*channel;

  // gradient 
  float2 val = -u[i];
  if (x < Nx-1) val.s0 += u[i+1];  else val.s0 = 0.0f;
  if (y < Ny-1) val.s1 += u[i+Nx]; else val.s1 = 0.0f;

  // step
  val = v[i] + sigma*(val - p[i]);

  // reproject
  v[i]=val;
  //float fac = hypot(val.s0, val.s1)*alphainv;
  //if (fac > 1.0f) v[i] = val/fac; else v[i] = val;
}

__kernel void update_w(__global float3 *w, __global float2 *p,
					   const float sigma, const float alphainv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1),channel=get_global_id(2);
  size_t i = Nx*y + x +Nx*Ny*channel;

  // symmetrized gradient 
  float4 val = (float4)(p[i], p[i]);
  if (x > 0) val.s01 -= p[i-1];	 else val.s01 = (float2)(0.0f, 0.0f);
  if (y > 0) val.s23 -= p[i-Nx]; else val.s23 = (float2)(0.0f, 0.0f);
  float3 val2 = (float3)(val.s0, val.s3, 0.5f*(val.s1 + val.s2));

  // step
  val2 = w[i] + sigma*val2;
	w[i]=val2;
  // reproject
  //float fac = hypot(hypot(val2.s0, val2.s1), 2.0f*val2.s2)*alphainv;
  //if (fac > 1.0f) w[i] = val2/fac; else w[i] = val2;
}


__kernel void update_w_empty(__global float3 *w, __global float2 *p,
					   const float sigma, const float alphainv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1),channel=get_global_id(2);
  size_t i = Nx*y + x +Nx*Ny*channel;

w[i]=0;
}

__kernel void update_lambda_KL(__global float *lambda, __global float *Ku,
							__global float *f, const float sigma, 
						   __global float *mu) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1),channel=get_global_id(2);
  size_t i = Nx*y + x +Nx*Ny*channel;


  float s= lambda[i]+sigma *Ku[i];
  float z=s-mu[channel];
  float d=z*z+4*sigma*f[i]*mu[channel];
  lambda[i]=s- 0.5f*(z+sqrt(d));
}

__kernel void update_lambda_L2(__global float *lambda, __global float *Ku,
							__global float *f, const float sigma,
						   __global float *sigmap1inv) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1),channel=get_global_id(2);
  size_t i = Nx*y + x +Nx*Ny*channel;

  lambda[i] = (lambda[i] + sigma*(Ku[i] - f[i]))*sigmap1inv[channel];
}

__kernel void update_u(__global float *u, __global float *u_,
					   __global float2 *v, __global float *Kstarlambda,
					   const float tau, const float norming) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1),channel=get_global_id(2);
  size_t i = Nx*y + x +Nx*Ny*channel;

  // divergence
  float2 val = v[i];
  if (x == Nx-1) val.s0 = 0.0f;
  if (x > 0) val.s0 -= v[i-1].s0;
  if (y == Ny-1) val.s1 = 0.0f;
  if (y > 0) val.s1 -= v[i-Nx].s1;

  // linear step
  u[i] = u_[i] + tau*(val.s0 + val.s1 - norming*Kstarlambda[i]);
  if(u[i]<0){u[i]=0;}
}

__kernel void update_p(__global float2 *p, __global float2 *p_,
					   __global float2 *v, __global float3 *w,
					   const float tau) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1),channel=get_global_id(2);
  size_t i = Nx*y + x +Nx*Ny*channel;

  // divergence
  float3 val0 = -w[i];
  float4 val = (float4)(val0.s0, val0.s2, val0.s2, val0.s1);
  if (x == 0)	val.s01 = 0.0f;
  if (x < Nx-1) val.s01 += (float2)(w[i+1].s0, w[i+1].s2);
  if (y == 0)	val.s23 = 0.0f;
  if (y < Ny-1) val.s23 += (float2)(w[i+Nx].s2, w[i+Nx].s1);

  // linear step
  p[i] = p_[i] + tau*(v[i] + val.s01 + val.s23);
}

__kernel void update_p_empty(__global float2 *p, __global float2 *p_,
					   __global float2 *v, __global float3 *w,
					   const float tau) {
  size_t Nx = get_global_size(0), Ny = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1),channel=get_global_id(2);
  size_t i = Nx*y + x +Nx*Ny*channel;

  
  // linear step
  p[i] = 0;
}


__kernel void radon(__global float *sino, __global float *img,
					__constant float4 *ofs, const int X,
					const int Y)
{
  size_t I = get_global_size(0);
   size_t J = get_global_size(1); 

  size_t i = get_global_id(0);
  size_t j = get_global_id(1);
  size_t thirddim = get_global_id(2);
  int ii=i;
  float4 o = ofs[j];
  float acc = 0.0f;
  img+=X*Y*thirddim;
  for(int y = 0; y < Y; y++) {
	int x_low, x_high;
	float d = y*o.y + o.z;

	// compute bounds
	if (o.x == 0) {
	  if ((d > ii-1) && (d < ii+1)) {
		x_low = 0; x_high = X-1;
	  } else {
		img += X; continue;
	  }
	} else if (o.x > 0) {
	  x_low = (int)((ii-1 - d)*o.w);
	  x_high = (int)((ii+1 - d)*o.w);
	} else {
	  x_low = (int)((ii+1 - d)*o.w);
	  x_high = (int)((ii-1 - d)*o.w);
	}
	x_low = max(x_low, 0);
	x_high = min(x_high, X-1);

	// integrate
	for(int x = x_low; x <= x_high; x++) {
	  float weight = 1.0 - fabs(x*o.x + d - ii);
	  if (weight > 0.0f) acc += weight*img[x];
	}
	img += X;
  }
  sino[j*I + i+I*J*thirddim] = acc;
}

__kernel void radon_ad(__global float *img, __global float *sino,
					   __constant float4 *ofs, const int I,
					   const int J)
{
  size_t X = get_global_size(0);
	size_t Y = get_global_size(1);
  
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
   size_t thirddim = get_global_id(2);

  float4 c = (float4)(x,y,1,0);
  float acc = 0.0f;
  sino += I*J*thirddim;
  
  for (int j=0; j < J; j++) {
	float i = dot(c, ofs[j]);
	if ((i > -1) && (i < I)) {
	  float i_floor;
	  float p = fract(i, &i_floor);
	  if (i_floor >= 0)	  acc += (1.0f - p)*sino[(int)i_floor];
	  if (i_floor <= I-2) acc += p*sino[(int)(i_floor+1)];
	}
	sino += I;
  }
  img[y*X + x+X*Y*thirddim] = acc;
}
""")


##############################################

	""" Is used to ensure that no data is present where it makes no sense for data to be by setting sino to zero in such positions
		Input		
				sino		...	Np.array with sinogram data in question
				r_struct	... Defining the geometry of an object for the radon transform
				imageshape	... Imagedimensions of the image corresponding to the image (could probably be removed since information is also in r_struct)
		Ouput
				sinonew		... A new sinogram where pixels such that R1 is zero, where R is the radon transform and 1 is the constant 1 image. (This corresponds to all pixels	 we cann not obtain any mass in due to the geometry of the radontransform)
	"""
	def Make_function_feasible(sino,r_struct,imageshape):
		img=np.ones(imageshape)
		img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
		sino_gpu = clarray.zeros(queue, r_struct[2], dtype=float32, order='F')
		radon(sino_gpu, img_gpu, r_struct).wait()

		sino_gpu=sino_gpu.get()
		sino2=sino.copy()
		sino2[np.where(sino_gpu<=0)]=0

		return sino2
#############
## Radon code

	"""Creates the structure of radon geometry required for radontransform and its adjoint
	Input:
			queue ... a queue object corresponding to a context in pyopencl
			shape ... the shape of the object (image) in pixels
			angles ... a list of angles considered
			n_detectors ... Number of detectors, i.e. resolution of the sinogram
			detector_with ... Width of one detector relatively to a pixel in the image (default 1.0)
			detector_shift ... global shift of ofsets (default 0)
	Output:
			ofs_buf ... a buffer object with 4 x number of angles entries corresponding to the cos and sin divided by the detectorwidth, also offset depending on the angle and the inverse of the cos values
			shape ... The same shape as in the input.
			sinogram_shape ... The sinogram_shape is a list with first the number of detectors, then number of angles.
	"""
	def radon_struct(queue, shape, angles, n_detectors=None,
				 detector_width=1.0, detector_shift=0.0):
		if isscalar(angles):
			angles = linspace(0,pi,angles+1)[:-1]
		if n_detectors is None:
			nd = int(ceil(hypot(shape[0],shape[1])))
		else:
			nd = n_detectors
		midpoint_domain = array([shape[0]-1, shape[1]-1])/2.0
		midpoint_detectors = (nd-1.0)/2.0

		X = cos(angles)/detector_width
		Y = sin(angles)/detector_width
		Xinv = 1.0/X

		# set near vertical lines to vertical
		mask = abs(Xinv) > 10*nd
		X[mask] = 0
		Y[mask] = sin(angles[mask]).round()/detector_width
		Xinv[mask] = 0
	
		offset = midpoint_detectors - X*midpoint_domain[0] \
				- Y*midpoint_domain[1] + detector_shift/detector_width

		ofs = zeros((4, len(angles)), dtype=float32, order='F')
		ofs[0,:] = X; ofs[1,:] = Y; ofs[2,:] = offset; ofs[3,:] = Xinv

		ofs_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, ofs.nbytes)
		cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()
	
		sinogram_shape = (nd, len(angles))
	
		return (ofs_buf, shape, sinogram_shape)
		
		
	"""Starts the GPU Radon transform code 
	input	sino ... A pyopencl.array in which result will be saved.
			img ...	 A pyopencl.array in which the image for the radontransform is contained
			r_struct. ..	The r_struct corresponding the given topology (geometry), see radon_struct
	output	An event for the queue to compute the radon transform of image saved into img w.r.t. r_struct geometry
	"""
	def radon(sino, img, r_struct, wait_for=None):
		(ofs_buf, shape, sinogram_shape) = r_struct
	
		return prg.radon(sino.queue, sino.shape, None,
					 sino.data, img.data, ofs_buf,
					 int32(shape[0]), int32(shape[1]),
					 wait_for=wait_for)
					 
					 
	"""Starts the GPU backprojection code 
	input	sino ... A pyopencl.array in which the sinogram for transformation will be saved.
			img ...	 A pyopencl.array in which the result for the adjoint radontransform is contained
			r_struct. ..	The r_struct corresponding the given topology, see radon_struct
	output	An event for the queue to compute the adjoint radon transform of image saved into img w.r.t. r_struct geometry
	"""
	def radon_ad(img, sino, r_struct, wait_for=None):
		(ofs_buf, shape, sinogram_shape) = r_struct

		return prg.radon_ad(img.queue, img.shape, None,
						img.data, sino.data, ofs_buf,
						int32(sinogram_shape[0]),
						int32(sinogram_shape[1]), wait_for=wait_for)


	"""Estimation of the norm of the radontransform with geometry r_struct is computed
	input
			queue ... queue object of some context in pyopencl
			the r_struct corresponding the given topology of a radon transform, see radon_struct
	output
			norm ... an estimate of the norm of the radon transform (square of largest singular value)
	An Power iteration method is applied onto R^T R (Adjoint operator times operator of radontransform)
	"""
	def radon_normest(queue, r_struct):
		img = clarray.to_device(queue, require(random.randn(*r_struct[1]), float32, 'F'))
		sino = clarray.zeros(queue, r_struct[2], dtype=float32, order='F')
 
		V=(radon(sino, img, r_struct, wait_for=img.events))
	
		for i in range(10):
			normsqr = float(clarray.sum(img).get())
			img /= normsqr
			sino.add_event(radon(sino, img, r_struct, wait_for=img.events))
			img.add_event(radon_ad(img, sino, r_struct, wait_for=sino.events))
	
		return sqrt(normsqr)
		

############
## TGV code

	def update_v(v, u, p, sigma, alpha, wait_for=None):
		return prg.update_v(v.queue, u.shape, None, v.data, u.data,
						p.data, float32(sigma), float32(1.0/alpha), wait_for=wait_for)

	#Update_w and update_p is only relevant for TGV, otherwise update_w and update_p is empty
	if regularisation=='TGV':
		def update_w(w, p, sigma, alpha, wait_for=None):
			return prg.update_w(w.queue, w.shape[1:], None, w.data, p.data,
						float32(sigma), float32(1.0/alpha), wait_for=wait_for)
		def update_p(p, p_, v, w, tau, wait_for=None):
			return prg.update_p(p.queue, p.shape[1:], None, p.data, p_.data,
						v.data, w.data, float32(tau), wait_for=wait_for)
	elif regularisation=='TV':
		def update_w(w, p, sigma, alpha, wait_for=None):
			return prg.update_w_empty(w.queue, w.shape[1:], None, w.data, p.data, 
					   float32(sigma), float32(1.0/alpha), wait_for=wait_for)
		def update_p(p, p_, v, w, tau, wait_for=None):
			return prg.update_p_empty(p.queue, p.shape[1:], None, p.data, p_.data,
						v.data, w.data, float32(tau), wait_for=wait_for)
						
	#update_lambda with respect to KL functional
	if discrepancy=='KL':
		def update_lambda(lamb, Ku, f, sigma,mu, normest, wait_for=None):
			return prg.update_lambda_KL(lamb.queue, lamb.shape, None,
							 lamb.data, Ku.data, f.data,
							 float32(sigma/normest),
							 mu.data, wait_for=wait_for)
							 
	 #update_lambda with respect to L2 functional						 
	if discrepancy=='L2':
		def update_lambda(lamb, Ku, f, sigma, mu, normest, wait_for=None):
			return prg.update_lambda_L2(lamb.queue, lamb.shape, None,
							 lamb.data, Ku.data, f.data,
							 float32(sigma/normest),
							 mu.data, wait_for=wait_for)

	def update_u(u, u_, v, Kstarlambda, tau, normest, wait_for=None):
		return prg.update_u(u.queue, u.shape, None, u.data, u_.data,
						v.data, Kstarlambda.data, float32(tau),
						float32(1.0/normest), wait_for=wait_for)


	#update of extragradient
	update_extra = cl.elementwise.ElementwiseKernel(ctx, 'float *u_, float *u',
												'u[i] = 2.0f*u_[i] - u[i]')
												
	if regularisation.upper()=='TGV':
		if regularisationmethod=='Frobenius 2D Reconstruction':
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_frob(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
						 
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_frob(normW.queue, normW.shape, None, W.data,normW.data, float32(1.0/alpha),float32(Number_Channels),
						 wait_for=wait_for)
						 
		if regularisationmethod=='Uncorrelated 2D Reconstruction':
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_unchor(V.queue, V.shape[1:], None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
						 
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_unchor(W.queue, W.shape[1:], None, W.data,normW.data, float32(1.0/alpha),float32(Number_Channels),
						 wait_for=wait_for)
						 
		if regularisationmethod=='Nuclear 2D Reconstruction':
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_Nucl(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
						 
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_frob(normW.queue, normW.shape, None, W.data,normW.data, float32(1.0/alpha),float32(Number_Channels),
						 wait_for=wait_for)
						 
						 
	if regularisation.upper()=='TV':
		if regularisationmethod=='Frobenius 2D Reconstruction':
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_frob(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
						 
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_empty(normW.queue, normW.shape, None,wait_for=wait_for)
				
		if regularisationmethod=='Uncorrelated 2D Reconstruction':
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_unchor(V.queue, V.shape[1:], None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)						  
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_empty(normW.queue, normW.shape, None,wait_for=wait_for)
				
		if regularisationmethod=='Nuclear 2D Reconstruction':
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_Nucl(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_empty(normW.queue, normW.shape, None,wait_for=wait_for)

											   


	""" Primal Dual algorithm for Inverse Radon Transform
	input	
			F0			... np.array [Number_Channels,Nx,Number_of_angles] corresponding to the sinogram data to be reconstructed from
			img_shape	... Array or list in the form [Nx,Ny] (number of pixels in the reconstruction)
			angles		... Angles corresponding to the sinogram data in the F0 data
			alpha		... Regularization parameter (TV or TGV parameter) as list or float
			mu			... Weights of the channels, in an list or array, must be same as number of channels
			maxiter		... Number of iterations performed in the primal dual algorithm
			plott		... list containing to entries. First 0 or 1 for plotting live progress of or on, second for intervalls in which is updated (numbers 0 - 1 for percent), numbers greater 1 for every so many iterations.
	Output	
			U			... Solution to the reconstruction problem
			Sinogram	... The sinogram corresponding to the reconstruction
			
	Uses a primal dual algorithm (see Chambole Pock ) to solve a Radon inversion problem, using TV or TGV for regularization and L2 or Kullback Leibler discrepancies
	"""
	def tgv_radon(F0, img_shape, angles, alpha,mu, maxiter,plott):
		#Computing Structure of Radon Transform
		if angles is None:
			angles = F0[0].shape[1]
		r_struct = radon_struct(queue, img_shape, angles,
							n_detectors=F0[0].shape[0])
							
		#######Preparations							   
		Number_angles=len(angles)  
		Nz=1
		fig_data=np.zeros([Nx,Nx*Number_Channels])
		
		#Rearange Data dimensions				 
		Fnew=np.zeros([Nx,Number_angles,Number_Channels])		
		for j in range(0,Number_Channels):
			Fnew[:,:,j]=F0[j,:,:]
		Fnew=Make_function_feasible(Fnew,r_struct,img_shape)
		
		#####Initialising Variables/ acquiring memory
		F= clarray.to_device(queue, require(Fnew, float32, 'F'))		
		U=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		P=clarray.zeros(queue, (2,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		U_=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		P_=clarray.zeros(queue, (2,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		V=clarray.zeros(queue, (2,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		
		if regularisation.upper()=='TGV':	
			W=clarray.zeros(queue, (4,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
			normW=clarray.zeros_like(U)
			
		if regularisation.upper()=='TV':
			W=clarray.zeros(queue, (1), dtype=float32, order='F')
			normW=clarray.zeros(queue, (1), dtype=float32, order='F')
		
		Lamb=clarray.zeros(queue,(F0.shape[1],F0.shape[2],Nz*Number_Channels),dtype=float32, order='F')
		KU=clarray.zeros(queue,(F0.shape[1],F0.shape[2],Nz*Number_Channels),dtype=float32, order='F')
		KSTARlambda=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		
		normV=clarray.zeros_like(U)
		normW=clarray.zeros_like(U)
		

		#Computing estimates for Parameterchoice, choose sigma* tau <Lsqr approx 12
		normest = radon_normest(queue, r_struct)
		Lsqr = 0.5*(18.0 + sqrt(33))
		sigma = 1.0/sqrt(Lsqr)
		tau = 1.0/sqrt(Lsqr)
		alpha = array(alpha)/(normest**2)

		if discrepancy == 'KL':
			mu=clarray.to_device(queue, require(mu, float32, 'F'))
		if discrepancy == 'L2':
			mu=clarray.to_device(queue, require(mu/(sigma+mu), float32, 'F'))

		fig = None
		
		###Primal Dual Iterations
		for i in range(maxiter):
			###Dual Update
			#Algebraic update of v and w
			V.add_event(update_v(V, U_, P_, sigma, alpha[1],
							 wait_for=U_.events + P_.events))									   
			W.add_event(update_w(W, P_, sigma, alpha[0], wait_for=P_.events))

			#projection of norms in v and w
			normV.add_event(update_NormV(V,normV,alpha[1],Number_Channels,wait_for=V.events ))
			normW.add_event(update_NormW(W,normW,alpha[0],Number_Channels,wait_for=W.events))		
			
			#Updating dual fidelity variable lambda
			KU.add_event(radon(KU, U_, r_struct, wait_for=U_.events))		
			Lamb.add_event(update_lambda(Lamb, KU, F, sigma, mu, normest,
									 wait_for=KU.events + F.events))

			KSTARlambda.add_event(radon_ad(KSTARlambda, Lamb, r_struct,
										wait_for=Lamb.events))

			###Primal Update
			#Algebraic update on U_ and P_
			U_.add_event(update_u(U_, U, V, KSTARlambda, tau, normest,
							 wait_for=U.events + V.events +	 KSTARlambda.events))
		
			P_.add_event(update_p(P_, P, V, W, tau,
							 wait_for= P.events + V.events + W.events))
			
			#Extragradient update
			U.add_event(update_extra(U_, U, wait_for=U.events + U_.events))	
			P.add_event(update_extra(P_, P, wait_for=P.events + P_.events))
			
			#Swapping of extragradient witgh variables.
			(U, U_, P,	P_) = (U_, U,  P_,	P)
			

			###Plott Current Iteration
			if (i % math.ceil(maxiter/100.) == 0):
				for x in [U, U_, P, P_]:
					x.finish()
				
				Progress=start+sectionwidth*float(i)/maxiter
				#sys.stdout.write('\rOverall progress at {:3.0%}'.format(Progress)+'. Progress of current computation {:3.0%}'.format(float(i)/maxiter)+ ' for Slice '+str(current) )
				print ('Overall progress at {:3.0%}'.format(Progress)+'. Progress of current computation {:3.0%}'.format(float(i)/maxiter)+ ' for Slice '+str(current) ) 
				sys.stdout.flush()
					
			if plott[0]==1:
				if i% math.ceil(maxiter*plott[1])==0:
					for j in range(0,Number_Channels):
						fig_data[:,j*Nx:(j+1)*Nx]=U.get()[:,:,Nz//2+j*Nz]
					
					text='iteration %d' %i+' of slice '+str(current)+'. Overall_progress at {:3.0%}'.format(Progress)	
					#Look whether graphik is still open
					if fig == None or not fignum_exists(fig.number):
						fig=figure()						
						disp_im = imshow(fig_data, cmap=cm.gray)
						title(text)
						draw()
						pause(1e-10)
					else:					
						disp_im.set_data(fig_data)
						disp_im.set_clim([fig_data.min(), fig_data.max()])
						title(text)
						draw()
						pause(1e-10)
						
		if plott[0]==1:		
			close(fig)
			

							
		#Postprocessing, Rearangning data				
		Solution=np.zeros([Number_Channels,img_shape[0],img_shape[1]])
			
		Sinograms0=np.zeros([Number_Channels,Number_angles,Nx],dtype=float32)	

		for j in range(0,Number_Channels):
			Solution[j,:,:]=U.get()[:,:,Nz*j]
			Sinograms0[j,:,:]=KU.get()[:,:,Nz*j].T

		Sino2=[]
		for j in range(0,Number_Channels):
			Sino2.append(Sinograms0[j])		
		
		
		return Solution,Sino2


#################Prelimineare ############

	Number_Channels=len(sino)
	[Number_of_angles,Nx]=sino[0].shape

	sino_new=np.zeros([Number_Channels,Number_of_angles,Nx])

	for j in range(0,Number_Channels):
		for i in range(0,Number_of_angles):
			sino_new[j,i,:]=sino[j][i,:]	
			
			
	sino=np.zeros([Number_Channels,Nx,Number_of_angles])
	
	for j in range(0,Number_Channels):
		for i in range(0,Number_of_angles):
			sino[j,:,i]=sino_new[j,i,:]


	
	U,Sinograms = tgv_radon(sino, (sino[0].shape[0],sino[0].shape[0]), angles, Parameter,mu, maxiter,plott)
	
	return U,Sinograms


