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
## radon_tgv_primal_dual_3dsubsample.py:
## Code for 3D variational subsampled reconstruction of tomographic
## data.
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
##     multi-modal electron tomography. RSC Nanoscale, accepted
##     January 2019.
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

def Reconstructionapproach3dsubsmapling(sino,angles,Topology,Parameter,mu,maxiter,ctx,plott,Stepsize,discrepancy,regularisationmethod,regularisation,Info):
	
	start=Info[0]
	sectionwidth=Info[1]
	current=Info[2]
	
	#Create Py Opencl Program
	class Program(object):
		def __init__(self, ctx, code):
			self._cl_prg = cl.Program(ctx, code)
			self._cl_prg.build()
			self._cl_kernels = self._cl_prg.all_kernels()
			for kernel in self._cl_kernels:
					self.__dict__[kernel.function_name] = kernel
	
	

			
	queue = cl.CommandQueue(ctx)

######Kernel Code
	prg = Program(ctx, r"""
	
	
	
	//Projection  wrt Frobeniusnorm
	__kernel void update_NormV_frob(__global float3 *V,__global float *normV,const float alphainv,const float NumberOfChannels) {
  
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1),z=get_global_id(2);
  
  size_t i = Nx*y + x+Nx*Ny*z;

	//Computing Norm
	float norm=0;
	for(int j=0;j<NumberOfChannels;j++){
	size_t k=i+Nx*Ny*Nz*j;
	norm= hypot(hypot(hypot(V[k].s0,V[k].s1),V[k].s2),norm);
	}
	norm=norm*alphainv;
	normV[i] =norm;
	
	//Projection
	if (norm > 1.0f) 
	{	for(int j=0;j<NumberOfChannels;j++)
	{size_t	 k=i+Nx*Ny*Nz*j;
	V[k]=V[k]/norm;
	}}
	}
		//Projection  wrt Frobeniusnorm
	__kernel void update_NormV_unchor(__global float3 *V,__global float *normV,const float alphainv,const float NumberOfChannels) {
  
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1),z=get_global_id(2);
  
  size_t i = Nx*y + x+Nx*Ny*z;

	//Computing Norm
	float norm=0;	
	norm= hypot(hypot(V[i].s0,V[i].s1),V[i].s2)*alphainv;
	if (norm > 1.0f) {
	V[i]/=norm;}
	}
	
	
  __kernel void update_NormW_frob(__global float8 *W,__global float *normW,const float alphainv, const float NumberOfChannels){
	size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
	size_t x = get_global_id(0), y = get_global_id(1),z=get_global_id(2);
	size_t i = Nx*y + x+Nx*Ny*z;
	
	//Computing Norm
	float norm=0;
	for(int j=0;j<NumberOfChannels;j++)
	{size_t k=i+Nx*Ny*Nz*j;
	norm=hypot(hypot(hypot(W[k].s0,W[k].s1),hypot(W[k].s2,1.4142f*W[k].s3)),hypot(1.4142f*hypot(W[k].s4,W[k].s5),norm));
	}
	norm =alphainv*norm;
	normW[i]=norm;
	
	//Projection
	if (norm > 1.0f) 
	{for(int j=0;j<NumberOfChannels;j++)
	{size_t k=i+Nx*Ny*Nz*j;
	W[k]=W[k]/norm;
	}}
	}
	
	
	
	__kernel void update_NormW_unchor(__global float8 *W,__global float *normW,const float alphainv, const float NumberOfChannels){
	size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
	size_t x = get_global_id(0), y = get_global_id(1),z=get_global_id(2);
	size_t i = Nx*y + x+Nx*Ny*z;
	
	//Computing Norm
	float norm=0;
	
	
	norm=hypot(hypot(hypot(W[i].s0,W[i].s1),hypot(W[i].s2,1.4142f*W[i].s3)),hypot(1.4142f*W[i].s4,W[i].s5))*alphainv;
	if (norm > 1.0f) {
	W[i]=W[i]/norm;}
	
	}
	__kernel void update_NormW_empty(){
	
	}
	
  __kernel void update_v(__global float3 *v, __global float *u, __global float3 *p,
					   const float sigma, const float alphainv, const float Stepsize, const float Number_channels) {
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;


  // gradient 
  float3 val = -u[i];
  if (x < Nx-1) val.s0 += u[i+1];  else val.s0 = 0.0f;
  if (y < Ny-1) val.s1 += u[i+Nx]; else val.s1 = 0.0f;
  if (z < Nz-1)	  val.s2 += u[i+Nx*Ny];	  else val.s2=0.0f;
  val.s2/=Stepsize; //adjust to further Jump
  // step
  v[i] = v[i] + sigma*(val - p[i]);
  }
  

__kernel void update_w(__global float8 *w, __global float3 *p,
					   const float sigma, const float alphainv,const float Stepsize,const float Number_channels) {
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;


  // symmetrized gradient 
  float8 val = (float8)(p[i], p[i],0,0);
  float3 val1 = (float3)(p[i]);
  if (x > 0) val.s012 -= p[i-1];  else val.s012 = (float3)(0.0f,0.0f, 0.0f);
  if (y > 0) val.s345 -= p[i-Nx]; else val.s345 = (float3)(0.0f,0.0f, 0.0f);
  if (z > 0) val1.s012-= p[i-Nx*Ny];   else val1.s012= (float3)(0.0f,0.0f, 0.0f);
  val1.s012/=Stepsize;//adjust to further Jump
  float8 val2 = (float8)(val.s0, val.s4,val1.s2, 0.5f*(val.s1 + val.s3),0.5f*(val.s2+val1.s0),0.5f*(val.s5+val1.s1),0,0);

  // step
  w[i] = w[i] + sigma*val2;

}

__kernel void update_w_empty(__global float8 *w, __global float3 *p,
					   const float sigma, const float alphainv,const float Stepsize,const float Number_channels) {
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;


w[i]=0;

}



__kernel void update_lambda_L2(__global float *lambda, __global float *Ku,__global float *f, const float sigma,
							__global float *sigmap1inv, const float Number_channels) {
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;

  lambda[i] = (lambda[i] + sigma*(Ku[i] - f[i]))*sigmap1inv[channel];
}

__kernel void update_lambda_KL(__global float *lambda, __global float *Ku,__global float *f, const float sigma,
							__global float *mu, const float Number_channels) {
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;

  float s= lambda[i]+sigma *Ku[i];
  float w=s-mu[channel];
  float d=w*w+4*sigma*f[i]*mu[channel];
  lambda[i]=s- 0.5f*(w+sqrt(d));
}

__kernel void update_u(__global float *u, __global float *u_,
					   __global float3 *v, __global float *Kstarlambda,
					   const float tau, const float norming, const float Stepsize, const float Number_channels) {
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;

  // divergence
  float3 val = v[i];
  if (x == Nx-1) val.s0 = 0.0f;
  if (x > 0) val.s0 -= v[i-1].s0;
  if (y == Ny-1) val.s1 = 0.0f;
  if (y > 0) val.s1 -= v[i-Nx].s1;
  if (z == Nz-1) val.s2 = 0.0f;
  if (z > 0)	val.s2-=v[i-Nx*Ny].s2;
  val.s2/=Stepsize; //adjust fur further step
  // linear step
  u[i] = u_[i] + tau*(val.s0 + val.s1 + val.s2 - norming*Kstarlambda[i]);
  if(u[i]<0){u[i]=0;}
}




__kernel void update_p(__global float3 *p, __global float3 *p_, __global float3 *v, __global float8 *w,
					   const float tau, const float Stepsize,const float Number_channels) {
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;

  // divergence
  float8 val0 = -w[i];
  float8 val = (float8)(val0.s0, val0.s3,val0.s4, val0.s3, val0.s1,val0.s5,0,0);
  float3 val1= (float3)(val0.s4,val0.s5,val0.s2);
  if (x == 0)	val.s012 = 0.0f;
  if (x < Nx-1) val.s012 += (float3)(w[i+1].s0, w[i+1].s3,w[i+1].s4);
  if (y == 0)	val.s345 = 0.0f;
  if (y < Ny-1) val.s345 += (float3)(w[i+Nx].s3, w[i+Nx].s1,w[i+Nx].s5);
  if (z == 0)	val1.s012= 0.0f;
  if (z < Nz-1)	   val1.s012 += (float3)(w[i+Nx*Ny].s4,w[i+Nx*Ny].s5,w[i+Nx*Ny].s2);
  val1.s012/=Stepsize;
  // linear step
  p[i] = p_[i] + tau*(v[i] + val.s012 + val.s345+val1.s012);
}


__kernel void update_p_empty(__global float3 *p, __global float3 *p_, __global float3 *v, __global float8 *w,
					   const float tau, const float Stepsize,const float Number_channels) {
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;

p[i]=0;
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



								 
__kernel void  Sparsemultiplication(__global float* Result, __global float* Data, __global float* Matrix,__global float* Index, 
											__global float* Rowstart, __global int* Dimensions,const int Number_Channels)
{size_t Nx =get_global_size(0),Number_angle=get_global_size(1),N3=get_global_size(2);
  size_t x=get_global_id(0),angle=get_global_id(1),third_dimension=get_global_id(2);
  size_t Nz =N3/Number_Channels;
  size_t channel=third_dimension/Nz;
  size_t z=third_dimension-channel*Nz;

  size_t i=x+Nx*angle+third_dimension*Nx*Number_angle;
  size_t Nx_big=Dimensions[0];
  size_t Na_big=Dimensions[1];
  size_t Nz_big=Dimensions[2];
  float sum=0;
  size_t k1,x_,z_,k2;
  size_t reference_point=Nx_big*angle+Nx_big*Na_big*Nz_big*channel;
  size_t position=x+Nx*z;
  for (size_t j=Rowstart[position];j<Rowstart[position+1];j++)
  {
	k1=Index[j];
	z_=k1/Nx_big;
	x_=k1-z_*Nx_big;
	k2=x_+Nx_big*Na_big*z_+reference_point;
	sum +=Matrix[j]*Data[k2];
  }
  Result[i]=sum;
}

__kernel void Extract_Channels(__global float *Extract,__global float * Data, __global int * indices,const int Number_Channels)
{size_t Nx =get_global_size(0),Ny=get_global_size(1),Nz=get_global_size(2);
  size_t x=get_global_id(0),y=get_global_id(1),z=get_global_id(2);
  Nz=Nz/Number_Channels;
  int channel=z/Nz;
  int i=x+Nx*y+Nx*Ny*z;
  z=z-channel*Nz;
  
  
  int Channelshift=indices[channel]-channel;
  int I=i+Nx*Ny*Nz*Channelshift;
  Extract[i]=Data[I];

}

__kernel void Combine_Channels(__global float * Combination,__global float * Component1,__global float * Component2,const int Number_Channels1,const int Number_Channels2)
{
size_t Nx =get_global_size(0),Ny=get_global_size(1),Nz=get_global_size(2);
size_t x=get_global_id(0),y=get_global_id(1),z=get_global_id(2);
size_t Number_Channels=Number_Channels1+Number_Channels2;

  Nz=Nz/Number_Channels;
  int channel=z/Nz;
  int i=x+Nx*y+Nx*Ny*z;
  z=z-channel*Nz;
  if(channel<Number_Channels1)
  {
   Combination[i]=Component1[i];
  }
  else
  {
  Combination[i]=Component2[i-Nx*Ny*Nz*Number_Channels1];
  }
}

""")
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

	def radon(sino, img, r_struct, wait_for=None):
		(ofs_buf, shape, sinogram_shape) = r_struct
	
		return prg.radon(sino.queue, sino.shape, None,
					 sino.data, img.data, ofs_buf,
					 int32(shape[0]), int32(shape[1]),
					 wait_for=wait_for)

	def radon_ad(img, sino, r_struct, wait_for=None):
		(ofs_buf, shape, sinogram_shape) = r_struct
		
		return prg.radon_ad(img.queue, img.shape, None,
						img.data, sino.data, ofs_buf,
						int32(sinogram_shape[0]),
						int32(sinogram_shape[1]), wait_for=wait_for)



	def radon_normest_bigresu(queue, r_struct0,r_struct1,Matrix,Matrix_adjoint,Index,Index_adjoint,Rowstarts,Rowstarts_adjoint,Info):
		[img_shape,Nx0,Nx1,Ny0,Ny1,Number_angles0,Number_angles1,Nz0,Nz1,	Number_Channels0,Number_Channels1,indices0,indices1,Dimensions,Dimensions_adjoint]=Info
		Number_Channels=Number_Channels0+Number_Channels1;Nz=Nz0
		
		##Declaration of Variables
		U_=clarray.to_device(queue, require(random.randn(img_shape[0],img_shape[1],Nz*Number_Channels), float32, 'F'))	
		KU0		=clarray.zeros(queue, (Nx0,Number_angles0,Nz0*Number_Channels0), dtype=float32, order='F')
		KU1		=clarray.zeros(queue, (Nx0,Number_angles1,Nz0*Number_Channels1), dtype=float32, order='F')
		KU_=clarray.zeros(queue, (Nx1,Number_angles1,Nz1*Number_Channels1), dtype=float32, order='F')
	
		U_0=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels0), dtype=float32, order='F')
		U_1=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels1), dtype=float32, order='F')
		KSTARlambda0=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels0), dtype=float32, order='F')
		KSTARlambda1=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels1), dtype=float32, order='F')
		Lamb1adtrans=clarray.zeros(queue,(Nx0,Number_angles1,Nz0*Number_Channels1),dtype=float32, order='F')
		

		##Poweriteration
		for i in range(25):
			normsqr = float(clarray.sum(U_**2).get())**0.5
			U_ /= normsqr
			U_0.add_event(Extract_Channels(U_0,U_, indices0,wait_for=U_.events))
			U_1.add_event(Extract_Channels(U_1,U_, indices1,wait_for=U_.events))
			
			KU0.add_event(radon(KU0, U_0, r_struct0, wait_for=U_0.events))		
			KU1.add_event(radon(KU1, U_1, r_struct1, wait_for=U_1.events))
			
			KU_.add_event(Matrixmultiplication(KU_,KU1,Matrix,Index,Rowstarts,
										Dimensions,Number_Channels1,wait_for=KU1.events))	

			Lamb1adtrans.add_event(Matrixmultiplication(Lamb1adtrans,KU_,Matrix_adjoint,Index_adjoint,Rowstarts_adjoint,
										Dimensions_adjoint,Number_Channels1,wait_for=KU_.events))
										
			KSTARlambda0.add_event(radon_ad(KSTARlambda0, KU0, r_struct0,	wait_for=KU0.events))
			KSTARlambda1.add_event(radon_ad(KSTARlambda1, Lamb1adtrans, r_struct1,	wait_for=Lamb1adtrans.events))
			

			U_.add_event(Combine_Channels( U_,KSTARlambda0,KSTARlambda1,int32( Number_Channels0),
											int32(Number_Channels1),wait_for=KSTARlambda0.events+KSTARlambda1.events))

	
		return sqrt(normsqr)		

		
#################
##Subsampling Code
	def compute_intermediate(Refold,Refnew,Nx,Nz,ratio):
		Points=[]
		for j in range(0,Nz):
			for i in range(0,Nx):
				newcoordx=ratio[0]*(i-Refold[0])+Refnew[0]
				newcoordy=ratio[1]*(j-Refold[1])+Refnew[1]
				Points.append([newcoordx,newcoordy])
		return Points
	
	def Create_Matrix_Line(a,x,y,Nx,Ny):
		r=[[],[]]
		r[0]=a[0]/2.
		r[1]=a[1]/2.
		
		xlower=x-r[0];ylower=y-r[1];
		xupper=x+r[0];yupper=y+r[1];
		xlowerint=int(np.ceil(x-r[0]-0.499999)); ylowerint=int(np.ceil (y-r[1]-0.499999));
		xupperint=int(np.floor(x+r[0]+0.499999));yupperint=int(np.floor(y+r[1]+0.499999));

		A=[];I=[];
		for j in range(ylowerint,yupperint+1):
			for i in range(xlowerint,xupperint+1):
				a1=max(min(xupper-i,0.5),-0.5);	a2=max(min(i-xlower,0.5),-0.5)
				b1=max(min(yupper-j,0.5),-0.5);	b2=max(min(j-ylower,0.5),-0.5)
				a=a1+a2;b=b1+b2
			
				value=a*b/(4.*r[0]*r[1])

				ii=np.clip(i,0,Nx-1)
				jj=np.clip(j,0,Ny-1)
				A.append(value)
				I.append(ii+Nx*jj)
		return A,I

	def Create_sparse_matrix_adjoint(Matrix,Index,Rowstarts,Connections,Nx,Ny):
		Adjoint=[]; Index2=[];Rowstarts2=[];k=0
		Index_new=copy(Index)
		Index=list(copy(Index))
		for i in range(0,Nx*Ny):
			Rowstarts2.append(k)
		
			for j in Connections[i]:
				Lineconnections=Index[Rowstarts[j]:Rowstarts[j+1]]				
				Lineelements=Matrix[Rowstarts[j]:Rowstarts[j+1]]
				position=Lineconnections.index(i)
				Index[Rowstarts[j]+position]=-1
				Adjoint.append(Lineelements[position])
				Index2.append(j)
				k+=1
				
		Rowstarts2.append(k)
		return Adjoint,Index2,Rowstarts2

	def Create_sparse_matrix(P,r,Nx,Ny):
		Matrix=[];Rowstarts=[];Index=[];k=0
		Connections=[]
		for i in range(0,Nx*Ny):
			Connections.append([])
		
		for index in range(0,len(P)):
			Rowstarts.append(k)
			x=P[index][0];y=P[index][1];
			A,I=Create_Matrix_Line(r,x,y,Nx,Ny)
			Matrix+=(A)
			Index+=(I)
			k+=len(A)
			for i in I:			
				Connections[i].append(index)
	
		Rowstarts.append(k)
		Adjoint,Index_adjoint,Rowstarts_adjoints=Create_sparse_matrix_adjoint(Matrix,Index,Rowstarts,Connections,Nx,Ny)
						
		return Matrix,Index,Rowstarts,Connections,Adjoint,Index_adjoint,Rowstarts_adjoints
		

	def sparse_mul(Matrix,Index,Rowstarts,Vector):
		Nx=len(Rowstarts)-1
		Result=np.zeros([Nx])
		for i in range (0,Nx):
			for j in range(Rowstarts[i],Rowstarts[i+1]):
				Result[i]+=Matrix[j]*Vector[Index[j]]
		return Result
		
##############
## TGV code
			 
	def update_v(v, u, p, sigma, alpha,Stepsize,Number_Channels, wait_for=None):
		return prg.update_v(v.queue, u.shape, None, v.data, u.data,
						p.data, float32(sigma), float32(1.0/alpha),float32(Stepsize),float32(Number_Channels), wait_for=wait_for)

	if regularisation=='TGV':
		def update_w(w, p, sigma, alpha,Stepsize, Number_Channels,wait_for=None):
			return prg.update_w(w.queue, w.shape[1:], None, w.data, p.data,
						float32(sigma), float32(1.0/alpha),float32(Stepsize),float32(Number_Channels), wait_for=wait_for)
		def update_p(p, p_, v, w, tau,Stepsize,Number_Channels, wait_for=None):
			return prg.update_p(p.queue, p.shape[1:], None, p.data, p_.data,
						v.data, w.data, float32(tau), float32(Stepsize),float32(Number_Channels),wait_for=wait_for)
	elif regularisation=='TV':
		def update_w(w, p, sigma, alpha,Stepsize, Number_Channels,wait_for=None):
			return prg.update_w_empty(w.queue, w.shape[1:], None, w.data, p.data,
						float32(sigma), float32(1.0/alpha),float32(Stepsize),float32(Number_Channels), wait_for=wait_for)
		def update_p(p, p_, v, w, tau,Stepsize,Number_Channels, wait_for=None):
			return prg.update_p_empty(p.queue, p.shape[1:], None, p.data, p_.data,
						v.data, w.data, float32(tau), float32(Stepsize),float32(Number_Channels),wait_for=wait_for)
	if discrepancy=='L2':
		def update_lambda(lamb, Ku, f, sigma,mu, normest,Number_channels, wait_for=None):

			return prg.update_lambda_L2(lamb.queue, lamb.shape, None,
							 lamb.data, Ku.data, f.data,
							 float32(sigma/normest),
							 mu.data, float32(Number_channels), wait_for=wait_for)
	if discrepancy=='KL':
		def update_lambda(lamb, Ku, f, sigma,mu, normest,Number_channels, wait_for=None):

			return prg.update_lambda_KL(lamb.queue, lamb.shape, None,
							 lamb.data, Ku.data, f.data,
							 float32(sigma/normest),
							 mu.data,float32(Number_channels), wait_for=wait_for)

	def update_u(u, u_, v, Kstarlambda, tau, normest,Stepsize,Number_Channels, wait_for=None):
		return prg.update_u(u.queue, u.shape, None, u.data, u_.data,v.data, Kstarlambda.data, float32(tau),
						float32(1.0/normest),float32(Stepsize),float32(Number_Channels), wait_for=wait_for)



	update_extra = cl.elementwise.ElementwiseKernel(ctx, 'float *u_, float *u',
												'u[i] = 2.0f*u_[i] - u[i]')
												
	if regularisationmethod=='Frobenius 3D Subsampling Reconstruction':
		if regularisation =='TGV':									
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_frob(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
						 
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_frob(normW.queue, normW.shape, None, W.data,normW.data, float32(1.0/alpha),float32(Number_Channels),
						 wait_for=wait_for)
						 
		if regularisation=='TV':
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_frob(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
						
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_empty(normW.queue, normW.shape, None,wait_for=wait_for)
	
	if regularisationmethod=='Uncorrelated 3D Subsampling Reconstruction':
		
		if regularisation=='TGV':											
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_unchor(V.queue, V.shape[1:], None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
						 
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_unchor(W.queue, W.shape[1:], None, W.data,normW.data, float32(1.0/alpha),float32(Number_Channels),
						 wait_for=wait_for)
						 
		if regularisation=='TV':										   
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_unchor(V.queue, V.shape[1:], None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
						 
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_empty(normW.queue, normW.shape, None,wait_for=wait_for)	 
						 

											   
#####################
#Subsampling Parallel
	def Extract_Channels(Extract,Data,indices,wait_for=None):
		return prg.Extract_Channels(Extract.queue,Extract.shape,None,Extract.data,Data.data, indices.data,int32(indices.size),
								wait_for=wait_for)
	
	def Combine_Channels(Combination,Component0,Component1,Number_Channels0,Number_Channels1,wait_for=None):
		return prg.Combine_Channels( Combination.queue,Combination.shape,None,Combination.data,Component0.data,
						Component1.data,int32( Number_Channels0),int32(Number_Channels1),wait_for=wait_for)
			
	def Matrixmultiplication(Result,Data,Matrix,Index,Rowstarts,Dimensions,Number_Channels,wait_for=None):
		return prg.Sparsemultiplication(Result.queue,Result.shape,None,Result.data,Data.data,Matrix.data,Index.data,Rowstarts.data,
							Dimensions.data,int32(Number_Channels),wait_for=wait_for)


	def Gap_TGV_KL(U,Sino0,Sino1,P,U0,U1,mu,alpha):
		global E; global gradient; global NN
		Dataprim=0;Regprim1=0;Regprim2=0;
		Dataprim2=0; infeasible_points=0
		for i in range(0,Sino0.shape[0]):
			u=U[i];sino=Sino0[i];p=P[i][0:NN*3];u0=U0[i]
			Dataprim+=mu[i]*sum(( -u0[np.where(u0>0)]*np.log(sino[np.where(u0>0)])))+mu[i]*sum(sino)
			sino_alt=sino[np.where(sino>0)]
			u0_alt=u0[np.where(sino>0)]
			sino_alt=sino_alt[np.where(u0_alt>0)]
			u0_alt=u0_alt[np.where(u0_alt>0)]
			
			u0_alt2=u0[np.where(sino<=0)]
			u0_alt2=u0_alt2[np.where(u0_alt2>0)]
			
			infeasible_points+=u0_alt2.size

			Dataprim2+=mu[i]*sum((- u0_alt*np.log(sino_alt)))+mu[i]*sum(sino)
		
			A=(gradient*u-p).reshape(3,NN)
			A=np.linalg.norm(A,axis=0)
			
			B=(E*p).reshape(9,NN)
			B=np.linalg.norm(B,axis=0)
			
			Regprim1+=A**2
			Regprim2+=B**2
		for i in range(Sino0.shape[0],U.shape[0]):
			u=U[i];sino=Sino1[i-Sino0.shape[0]];p=P[i][0:NN*3];u0=U1[i-Sino0.shape[0]]
			Dataprim+=mu[i]*sum(( -u0[np.where(u0>0)]*np.log(sino[np.where(u0>0)])))+mu[i]*sum(sino)
			sino_alt=sino[np.where(sino>0)]
			u0_alt=u0[np.where(sino>0)]
			sino_alt=sino_alt[np.where(u0_alt>0)]
			u0_alt=u0_alt[np.where(u0_alt>0)]
			
			u0_alt2=u0[np.where(sino<=0)]
			u0_alt2=u0_alt2[np.where(u0_alt2>0)]
			
			infeasible_points+=u0_alt2.size

			Dataprim2+=mu[i]*sum((- u0_alt*np.log(sino_alt)))+mu[i]*sum(sino)
		
			A=(gradient*u-p).reshape(3,NN)
			A=np.linalg.norm(A,axis=0)
			
			B=(E*p).reshape(9,NN)
			B=np.linalg.norm(B,axis=0)
			
			Regprim1+=A**2
			Regprim2+=B**2
		Regprim1=np.linalg.norm(Regprim1**0.5,1)
		Regprim2=np.linalg.norm(Regprim2**0.5,1)
		Regprim=alpha[1]*Regprim1+alpha[0]*Regprim2

		#Datadual

		#for i in range(0,len(u0)):
		#	u0[i]=u0[i]*mu
		#mu=0.05
		#Datadual=0
		#Datadual=mu*sum((u0[np.where(u0!=0)]*(np.log(u0*mu/(mu-r))-1)[np.where(u0!=0)]))
		
		#F1dual=np.linalg.norm(-v+E.T*w)+np.linalg.norm(gradient.T*v+Kstarr)
		
		G=Dataprim+Regprim#+Datadual+F1dual
		#print G,Dataprim2+Regprim, infeasible_points,'\n'#F1dual,Datadual
		#print 'fertig'
		return G,Dataprim2+Regprim, infeasible_points
											   
	def create_symetrised_jacobi_3d(Nx,Ny,Nz,Stepsize):

		ones1=np.ones(Nx*Ny*Nz)
		ones2=np.ones(Nx*Ny*Nz)
		ones1[np.array(range(1,Ny*Nz+1))*Nx-1]=0

		ones2[np.array(range(0,Ny*Nz))*Nx]=0

		a=list(-ones1);	b=list(ones2)
		diagonals1=np.array([a,b])
		dx=spdiags(diagonals1,[0,1],Nx*Ny*Nz,Nx*Ny*Nz)

	
		ones1=np.ones(Nx*Ny*Nz)
		ones2=np.ones(Nx*Ny*Nz)
		for i in range(0,Nz):
			ones1[np.array(range(Nx*(Ny-1)+i*Nx*Ny,Nx*Ny+i*Nx*Ny))]=0
			ones2[np.array(range(Nx*(Ny-1)-Nx,Nx*Ny-Nx))]=0

		a=list(-ones1);	b=list(ones2)
		diagonals1=np.array([a,b])
		dy=spdiags(diagonals1,[0,Nx],Nx*Ny*Nz,Nx*Ny*Nz)
	
		
		ones1=np.ones(Nx*Ny*Nz)
		ones2=np.ones(Nx*Ny*Nz)
		ones1[np.array(range(Nx*Ny*(Nz-1),Nx*Ny*Nz))]=0
	
		a=list(-ones1);	b=list(ones2)
		diagonals1=np.array([a,b])
		dz=spdiags(diagonals1,[0,Nx*Ny],Nx*Ny*Nz,Nx*Ny*Nz)/Stepsize

		z=np.zeros(Nx*Ny*Nz)
		Zeross=spdiags(z,[0],Nx*Ny*Nz,Nx*Ny*Nz)
		jacobi1=sparse.hstack((-dx.T,Zeross,Zeross))

		jacobi2=sparse.hstack((Zeross,-dy.T,Zeross))
		jacobi3=sparse.hstack((Zeross,Zeross,-dz.T))
		jacobi4=1/2.*sparse.hstack((-dy.T,-dx.T,Zeross))
		jacobi5=1/2.*sparse.hstack((-dz.T,Zeross,-dx.T))
		jacobi6=1/2.*sparse.hstack((Zeross,-dz.T,-dy.T))
		jacobi=sparse.vstack((jacobi1,jacobi2,jacobi3))
		jacobi=sparse.vstack((jacobi,jacobi4))
		jacobi=sparse.vstack((jacobi,jacobi4))
		jacobi=sparse.vstack((jacobi,jacobi5))
		jacobi=sparse.vstack((jacobi,jacobi5))
		jacobi=sparse.vstack((jacobi,jacobi6))
		jacobi=sparse.vstack((jacobi,jacobi6))
		return jacobi
		
	def create_gradient3d(Nx,Ny,Nz,Stepsize):
		ones1=np.ones(Nx*Ny*Nz)
		ones2=np.ones(Nx*Ny*Nz)
		ones1[np.array(range(1,Ny*Nz+1))*Nx-1]=0

		ones2[np.array(range(0,Ny*Nz))*Nx]=0
	
		a=list(-ones1);	b=list(ones2)
		diagonals1=np.array([a,b])
		grad1=spdiags(diagonals1,[0,1],Nx*Ny*Nz,Nx*Ny*Nz)

		ones1=np.ones(Nx*Ny*Nz)
		ones2=np.ones(Nx*Ny*Nz)
		for i in range(0,Nz):
			ones1[np.array(range(Nx*(Ny-1)+i*Nx*Ny,Nx*Ny+i*Nx*Ny))]=0
			ones2[np.array(range(Nx*(Ny-1)-Nx,Nx*Ny-Nx))]=0

		a=list(-ones1);	b=list(ones2)
		diagonals1=np.array([a,b])
		grad2=spdiags(diagonals1,[0,Nx],Nx*Ny*Nz,Nx*Ny*Nz)



		ones1=np.ones(Nx*Ny*Nz)
		ones2=np.ones(Nx*Ny*Nz)
		ones1[np.array(range(Nx*Ny*(Nz-1),Nx*Ny*Nz))]=0

		a=list(-ones1);	b=list(ones2)
		diagonals1=np.array([a,b])
		grad3=spdiags(diagonals1,[0,Nx*Ny],Nx*Ny*Nz,Nx*Ny*Nz)/Stepsize
		
		
		grad=sparse.vstack((grad1,grad2))
		grad=sparse.vstack((grad,grad3))

		return grad
#################
## main iteration

	def tgv_radon(F0, img_shape, Angles, alpha,mu, maxiter,plott,Stepsize):
		print('########## Starting	'+str(discrepancy)+' '+str(regularisation)+' '+str(regularisationmethod)+' ############')
		f0=F0[0][0,:,:,0];	

		angles0=Angles[0];angles1=Angles[1]
		
		#Computing Structure of Radon Transform
		if angles0 is None:
			angles0 = f0.shape[1]
		r_struct0 = radon_struct(queue, img_shape, angles0,
							n_detectors=f0.shape[0])
		r_struct1 = radon_struct(queue, img_shape, angles1,
							n_detectors=f0.shape[0])					

	
		#######Preparations
		[Number_Channels0,Nx0,Number_angles0,Nz0]=F0[0].shape;
		[Number_Channels1,Nx1,Number_angles1,Nz1]=F0[1].shape;
		Number_Channels=Number_Channels0+Number_Channels1
		Nz=Nz0
		indices0=range(0,Number_Channels0);indices1=range(Number_Channels0,Number_Channels1+Number_Channels0);
		indices0=clarray.to_device(queue, require(indices0, int32, 'F'));indices1=clarray.to_device(queue, require(indices1, int32, 'F'));
		
		########Create interpolation Matrices 
		
		#Point-Correspondance
		Refnew[1]/=float(Stepsize)
		Points=compute_intermediate(Refold,Refnew,Nx1,Nz1,[p,p/float(Stepsize)])

		#Create_matrix
		Matrix0,Index0,Rowstarts0,Connections0,Matrix_adjoint0,Index_adjoint0,Rowstarts_adjoint0=Create_sparse_matrix(Points,[p,p/float(Stepsize)],Nx0,Nz0)


		#Test wether Matrices are Adjoint(sequentially) (not required for actual code)
##		Testcases=100
##		Testresults=[]
##		epsilon=0.001
##		for j in range(0,Testcases):
##			A=random.randn(len(Rowstarts_adjoint0)-1)
##			B=random.randn(len(Rowstarts0)-1)
##			C=sparse_mul(Matrix0,Index0,Rowstarts0,A)
##			D=sparse_mul(Matrix_adjoint0,Index_adjoint0,Rowstarts_adjoint0,B)
##			aa=np.dot(A,D);
##			bb=np.dot(B,C)
			#print 'TEST Sequential',aa, ' =? ', bb
##			if (aa-bb)< epsilon*min(abs(aa),abs(bb)):
##				Testresults.append(1)
##			else:
##				Testresults.append(0)
##		print 'Test-Sequential: ',Testresults.count(1),'/',Testcases

		#Change to appropriate Format
		Index=clarray.to_device(queue,require(np.array(Index0),float32,'F'));
		Rowstarts=clarray.to_device(queue,require(np.array(Rowstarts0),float32,'F'));
		Matrix=clarray.to_device(queue,require(np.array(Matrix0),float32,'F'));
		Index_adjoint=clarray.to_device(queue,require(np.array(Index_adjoint0),float32,'F'));
		Rowstarts_adjoint=clarray.to_device(queue,require(np.array(Rowstarts_adjoint0),float32,'F'));
		Matrix_adjoint=clarray.to_device(queue,require(np.array(Matrix_adjoint0),float32,'F'));
		Dimensions=np.array([Nx0,Number_angles1,Nz0]);		Dimensions=clarray.to_device(queue,require(Dimensions,int32,'F'))
		Dimensions_adjoint=np.array([Nx1,Number_angles1,Nz1]);	
		Dimensions_adjoint=clarray.to_device(queue,require(Dimensions_adjoint,int32,'F'))

		
		
		#Rearange Data dimensions
		Fnew0=np.zeros([Nx0,Number_angles0,Nz0*Number_Channels0])
		Fnew1=np.zeros([Nx1,Number_angles1,Nz1*Number_Channels1])
		for j in range(0,Number_Channels0):
			for i in range(0,Nz0):
				Fnew0[:,:,i+Nz*j]=F0[0][j,:,:,i]
		Fnew0=Make_function_feasible(Fnew0,r_struct0,[img_shape[0],img_shape[1],Nz*Number_Channels])
		for j in range(0,Number_Channels1):	
			for i in range(0,Nz1):
				Fnew1[:,:,i+Nz1*(j)]=F0[1][j,:,:,i]
		Fnew1=Make_function_feasible(Fnew1,r_struct1,[img_shape[0],img_shape[1],Nz*Number_Channels])
			
		
		fig_data=np.zeros([img_shape[0],img_shape[1]*Number_Channels])
		
		
		#####Initialising Variables
		F0= clarray.to_device(queue, require(Fnew0, float32, 'F'))
		F1= clarray.to_device(queue, require(Fnew1, float32, 'F'))			
		U=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		P=clarray.zeros(queue, (4,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		U_=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		P_=clarray.zeros(queue, (4,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		V=clarray.zeros(queue, (4,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		W=clarray.zeros(queue, (8,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		
		Lamb0=clarray.zeros(queue,(Nx0,Number_angles0,Nz0*Number_Channels0),dtype=float32, order='F')
		Lamb1=clarray.zeros(queue,(Nx1,Number_angles1,Nz1*Number_Channels1),dtype=float32, order='F')
		
		
		KU0	=clarray.zeros(queue, (Nx0,Number_angles0,Nz0*Number_Channels0), dtype=float32, order='F')
		KU1	=clarray.zeros(queue, (Nx0,Number_angles1,Nz0*Number_Channels1), dtype=float32, order='F')
		KU_=clarray.zeros(queue, (Nx1,Number_angles1,Nz1*Number_Channels1), dtype=float32, order='F')
		
		Lamb1adtrans=clarray.zeros(queue,(Nx0,Number_angles1,Nz0*Number_Channels1),dtype=float32, order='F')
		KSTARlambda0=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels0), dtype=float32, order='F')
		KSTARlambda1=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels1), dtype=float32, order='F')
		
		KSTARlambda=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		
		W=clarray.zeros(queue, (8,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		U_=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		
		normV=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz), dtype=float32, order='F')
		normW=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz), dtype=float32, order='F')
		U_0=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels0), dtype=float32, order='F')
		U_1=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels1), dtype=float32, order='F')
	
		
		#Computing estimates for Parameterchoice
		
		Info= [img_shape,Nx0,Nx1,Ny0,Ny1,Number_angles0,Number_angles1,Nz0,Nz1,	Number_Channels0,Number_Channels1,indices0,indices1,Dimensions,Dimensions_adjoint]
		normest=radon_normest_bigresu(queue, r_struct0,r_struct1,Matrix,Matrix_adjoint,Index,Index_adjoint,Rowstarts,Rowstarts_adjoint,Info)
		
		Lsqr = 36
		sigma = 1.0/sqrt(Lsqr)
		tau = 1.0/sqrt(Lsqr)
		alpha = array(alpha)/(normest**2)

		mu0=mu[0:Number_Channels0]
		mu1=mu[Number_Channels0:Number_Channels]
		if discrepancy == 'KL':
			mu0=clarray.to_device(queue, require(mu0, float32, 'F'))
			mu1=clarray.to_device(queue, require(mu1, float32, 'F'))
		if discrepancy == 'L2':
			mu0=clarray.to_device(queue, require(mu0/(sigma+mu0), float32, 'F'))
			mu1=clarray.to_device(queue, require(mu1/(sigma+mu1), float32, 'F'))
			
		fig = None

		#Testing whether Matrix Multiplication is adjoint (not required for actual code)
##		Testcases=100
##		Testresults=[]
##		epsilon=0.001
##		for j in range(0,Testcases):
##			A= clarray.to_device(queue, require(random.randn(Nx0,Number_angles1,Nz0*Number_Channels1), float32, 'F'))
##			B=clarray.to_device (queue,require(random.rand(Nx1,Number_angles1,Nz1*Number_Channels1),float32,'F'))
##			C=clarray.zeros(queue,(Nx1,Number_angles1,Nz1*Number_Channels1),dtype=float32,order='F')
##			D=clarray.zeros(queue,(Nx0,Number_angles1,Nz0*Number_Channels1),dtype=float32,order='F')
##
##			Matrixmultiplication(D,B,Matrix_adjoint,Index_adjoint,Rowstarts_adjoint,
##										Dimensions_adjoint,Number_Channels1,wait_for=D.events+B.events)
##			Matrixmultiplication(C,A,Matrix,Index,Rowstarts,
##										Dimensions,Number_Channels1,wait_for=C.events+A.events)

			
##			aa= np.dot(A.get().reshape(Nx0*Number_angles1*Nz0*Number_Channels1),D.get().reshape(Nx0*Number_angles1*Nz0*Number_Channels1))
##			bb= np.dot(B.get().reshape(Nx1*Number_angles1*Nz1*Number_Channels1),C.get().reshape(Nx1*Number_angles1*Nz1*Number_Channels1))
##			#print aa ,'=?',bb
##			if abs(aa-bb)<epsilon*abs(aa):
##				Testresults.append(1)
##			else:
##				Testresults.append(0)
##		print 'Testresults-Parallelisiert'
##		print Testresults.count(1),' / ',Testcases
		
	
	
		
		#Primal Dual Iterations
		for i in range(maxiter):
			
			#Seperate Images according to Datasize
			U_0.add_event(Extract_Channels(U_0,U_, indices0,wait_for=U_.events))
			U_1.add_event(Extract_Channels(U_1,U_, indices1,wait_for=U_.events))
			

			#Dual Update
			V.add_event(update_v(V, U_, P_, sigma, alpha[1],Stepsize,Number_Channels,
							 wait_for=U_.events + P_.events))
			W.add_event(update_w(W, P_, sigma, alpha[0],Stepsize,Number_Channels, wait_for=P_.events))				
			normV.add_event(update_NormV(V,normV,alpha[1],Number_Channels,wait_for=V.events ))
			normW.add_event(update_NormW(W,normW,alpha[0],Number_Channels,wait_for=W.events))	
			
			
		
			KU0.add_event(radon(KU0, U_0, r_struct0, wait_for=U_0.events))		
			KU1.add_event(radon(KU1, U_1, r_struct1, wait_for=U_1.events))	
			##Reduce KU1 to lower Resolution
			KU_.add_event(Matrixmultiplication(KU_,KU1,Matrix,Index,Rowstarts,
										Dimensions,Number_Channels1,wait_for=Lamb1.events))


	
			Lamb0.add_event(update_lambda(Lamb0, KU0, F0, sigma,mu0, normest,Number_Channels0,
									 wait_for=KU0.events + F0.events))		
			Lamb1.add_event(update_lambda(Lamb1, KU_, F1, sigma,mu1, normest,Number_Channels1,
									 wait_for=KU_.events + F1.events))
			##Return Lamb1 to higher Resolution
			Lamb1adtrans.add_event(Matrixmultiplication(Lamb1adtrans,Lamb1,Matrix_adjoint,Index_adjoint,Rowstarts_adjoint,
										Dimensions_adjoint,Number_Channels1,wait_for=Lamb1.events))

									
			KSTARlambda0.add_event(radon_ad(KSTARlambda0, Lamb0, r_struct0,	wait_for=Lamb0.events))
			KSTARlambda1.add_event(radon_ad(KSTARlambda1, Lamb1adtrans, r_struct1,	wait_for=Lamb1adtrans.events))
			##Combine Kstar back together
			KSTARlambda.add_event(Combine_Channels( KSTARlambda,KSTARlambda0,KSTARlambda1,int32( Number_Channels0),
											int32(Number_Channels1),wait_for=KSTARlambda0.events+KSTARlambda1.events))
			
			#Primal Update
			U_.add_event(update_u(U_, U, V, KSTARlambda, tau, normest,Stepsize,Number_Channels,
									wait_for=U.events + normV.events +	KSTARlambda.events))
				
			P_.add_event(update_p(P_, P, V, W, tau,Stepsize,Number_Channels,
							 wait_for= P.events + normV.events + normW.events))
										
			U.add_event(update_extra(U_, U, wait_for=U.events + U_.events))	
				
			P.add_event(update_extra(P_, P, wait_for=P.events + P_.events))

			(U, U_, P,	P_) = (U_, U,  P_,	P)


			
			#Plot Current Iteration
			if (i % int(math.ceil(maxiter/100.)) == 0):
				for x in [U, U_, P, P_]:
					x.finish()
				
				if i% int(math.ceil(maxiter*plott[1]))==0:
					
					if len(current)>3:
						a=str(current[0])+', '+str(current[1])+'...'+str(current[len(current)-1])
					else:
						a=str(current)
				Progress=start+sectionwidth*float(i)/maxiter
				sys.stdout.write('Overall progress at {:3.0%}'.format(Progress)+'. Progress of current computation {:3.0%}'.format(float(i)/maxiter)+ ' for Section '+str(a) )
				sys.stdout.flush()
	
			if plott[0]==1:
				if i% math.ceil(maxiter*plott[1])==0:
					for j in range(0,Number_Channels):
						fig_data[:,j*Nx0:(j+1)*Nx0]=U.get()[:,:,Nz//2+j*Nz]
					text='iteration %d' %i+' of section '+a+'. Overall progress at {:3.0%}'.format(Progress)	
					if fig == None or not fignum_exists(fig.number):
						fig = figure()						
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
		print('iterations done')
		
		#Postprocessing, Rearangning data
		Solution=np.zeros([Number_Channels,img_shape[0],img_shape[1],Nz])
			
		Sinograms0=np.zeros([Number_Channels0,Number_angles0,Nz,Nx0],dtype=float32)	
		Sinograms1_highresu=np.zeros([Number_Channels1,Number_angles1,Nz,Nx0],dtype=float32)
		Sinograms1_lowresu=np.zeros([Number_Channels1,Number_angles1,Nz1,Nx1],dtype=float32)
		for j in range(0,Number_Channels):
			for i in range(0,Nz):
				Solution[j,:,:,i]=U.get()[:,:,i+Nz*j]
		for j in range(0,Number_Channels0):
			for i in range(0,Nz):
				Sinograms0[j,:,i,:]=KU0.get()[:,:,i+Nz*j].T
		for j in range(0,Number_Channels1):
			for i in range(0,Nz):
				Sinograms1_highresu[j,:,i,:]=KU1.get()[:,:,i+Nz*j].T
			for i in range(0,Nz1):
				Sinograms1_lowresu[j,:,i,:]=KU_.get()[:,:,i+Nz1*j].T
		

		Sino2=[]
		Sino2.append(Sinograms0)
		Sino2.append(Sinograms1_highresu)
		Sino2.append(Sinograms1_lowresu)
		

		return Solution,Sino2
		



	#################Prelimineare Reconstructing############
	[smallerimagestart,p,Refold,Refnew]=Topology
	Number_Channels0=smallerimagestart
	Number_Channels=len(sino)
	Number_Channels1=Number_Channels-Number_Channels0
	[Number_of_angles0,Ny0,Nx0]=sino[0].shape
	[Number_of_angles1,Ny1,Nx1]=sino[smallerimagestart].shape
	sino_new0=np.zeros([Number_Channels0,Number_of_angles0,Nx0,Ny0])
	sino_new1=np.zeros([Number_Channels1,Number_of_angles1,Nx1,Ny1])
	####Reordering Dimensions

	for j in range(0,Number_Channels0):
		for i in range(0,Number_of_angles0):
			sino_new0[j,i,:,:]=sino[j][i,:,:].T
			

	for j in range(Number_Channels0,Number_Channels):
		for i in range(0,Number_of_angles1):
			sino_new1[j-Number_Channels0,i,:,:]=sino[j][i,:,:].T	
			
	sino0=np.zeros([Number_Channels0,Nx0,Number_of_angles0,Ny0])
	sino1=np.zeros([Number_Channels1,Nx1,Number_of_angles1,Ny1])
	for j in range(0,Number_Channels0):
		for i in range(0,Number_of_angles0):
			sino0[j,:,i,:]=sino_new0[j,i,:,:]

	for j in range(0,Number_Channels1):
		for i in range(0,Number_of_angles1):
			sino1[j,:,i,:]=sino_new1[j,i,:,:]	


	#Executing Primal Dual Algorithm
	Sino=[];Sino.append(sino0);Sino.append(sino1)
	Angles=[angles[0],angles[smallerimagestart]]

	NN=Nx0**2*Ny0
	u,sinogram = tgv_radon(Sino, (Sino[0].shape[1],Sino[0].shape[1]), Angles, Parameter,mu, maxiter,plott,Stepsize)
	
	return u,sinogram

