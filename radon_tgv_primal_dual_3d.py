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
## radon_tgv_primal_dual_3d.py:
## Code for 3D variational reconstruction of tomographic data.
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

""" Reconstruction of tomographic data for multiple using Primal Dual optimization
Input:	
		sino		... A list(length is Number_Channels) containing the sinogram data (Array with shape [Number_of_channels,Number_of_angles,Nx])
		angles		... A list or array of angles corresponding to Radon transform
		Parameters	... Regularization parameters corresponding to TGV or TV (List, array or scalar)
		mu			... Weights of the subproblems (must be same length as Number_Channels) (array or list)
		maxiter		... Number of iterations used in the primal dual algorithm (scalar)
		ctx			... A pyopencl context corresponding to the device used for GPU implementation
		plott		... A list with two entries, first is 0 or 1 for no live plott or live plott, second for frequence of updates (between 0 and 1 transformed in percent of reconstruction, for greater 1 corresponds to every so much iterations)
		Stepsize	... An integer in case of skipping slices (e.g. 2 if every second slice is reconstructed)
		discrepancy	... A string stating whether L2 or KL should be used as discrepancy functions. Options 'L2' or 'KL'
		regularisationmethod	...	 A string stating what regularisation method is used for coupled regularization, i.e. uncorrelated, Frobenius or Nuclear coupling. Options are 'Uncorrelated 2D Reconstruction', 'Frobenius 2D Reconstruction','Nuclear 2D Reconstruction'
		regularization ...String: state what regularization functional to consider, options 'TV' or 'TGV'
		Info		... A list of additional plotting information containing information on the overall progress of the reconstruction, where first entry corresponds to percent already complete, the second the width (in percent) of the subproblem currently computed,	last on at which slices we currently compute .

Output:
		Solution   ... A numpy array containing the reconstuction
		Sinogram   ... A numpy array containing the sinograms corresponding to the solution
"""
def Reconstructionapproach3d(sino,angles,Parameter,mu,maxiter,ctx,plott,Stepsize,discrepancy,regularisationmethod,regularisation,Info):
	
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
	
		
		__kernel void gradient(__global float3 *grad, __global float *u, int Number_channels,int Stepsize) 
{
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
  if (z < Nz-1) val.s2 += u[i+Nx*Ny]; else val.s2 = 0.0f;
  val.s2=val.s2/Stepsize;
  grad[i]=val;
}

__kernel void divergence(__global float *div, __global float3 *p,int Number_channels, int Stepsize) 
{
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;
  
 // divergence
  float3 val = p[i];
  if (x == Nx-1) val.s0 = 0.0f;
  if (x > 0) val.s0 -= p[i-1].s0;
  if (y == Ny-1) val.s1 = 0.0f;
  if (y > 0) val.s1 -= p[i-Nx].s1;
   if (z == Nz-1) val.s2 = 0.0f;
  if (z > 0) val.s2 -= p[i-Nx*Ny].s2;
  val.s2=val.s2/Stepsize;
  div[i]=val.s0+val.s1+val.s2;
}


__kernel void Jacobian(__global float8 *jac, __global float3 *p,int Number_channels,int Stepsize) 
{
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
  jac[i]=val2;
}


__kernel void div_jac(__global float3 *div, __global float8 *w,int Number_channels,int Stepsize) 
{
  int Number_Channels = Number_channels;		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2)/Number_Channels;
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t channel=z/Nz;
  z=z-channel*Nz;
  size_t i = Nx*y + x+Nx*Ny*z+Nx*Ny*Nz*channel;
  
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
  div[i]=  val.s012 + val.s345+val1.s012;
  
}
//#############################################################################################################################################################
//Nuclearcode

float* eigenvalues3d(float a,float b,float c,float d,float e,float f){
	float c2,c1,c0,p,q,phi,x1,x2,x3,lambda1,lambda2,lambda3;
	float pi=3.14159265359;
	c2=-(a+b+c);
	c1=a*b+a*c+b*c-d*d-e*e-f*f;
	c0=a*f*f+b*e*e+c*d*d-a*b*c-2*d*e*f;
	
	p=c2*c2-3.*c1;
	q=-27.f/2.f*c0-pow(c2,3)+9.f/2.f*c2*c1;
	
	
	
	
	phi=1.f/3.f*atan(sqrt(27.f*(1.f/4.f*c1*c1*(p-c1)+c0*(q+27.f/4.f*c0)))/q);
	if (q<0)
		{phi+=pi/3.f;}
	
	
	x1=2*cos(phi);
	x2=2*cos(phi+2*pi/3.f);
	x3=2*cos(phi-2*pi/3.f);
	p=sqrt(p)/3.f;
	c2=c2/3.f;
	lambda1=p*x1-c2;
	lambda2=p*x2-c2;
	lambda3=p*x3-c2;


	
	float eigenvalues[3] = {lambda1,lambda2,lambda3};
	return eigenvalues;
	}
	
	float3 crossproduct(float a1,float a2,float a3,float b1,float b2,float b3){
	float u1,u2,u3;
	u1=a2*b3-a3*b2;
	u2=b1*a3-a1*b3;
	u3=a1*b2-a2*b1;
	float3 cross = (float3)(u1,u2,u3);
	
	return cross;
	}
	
	
	 float* diagonalization2d(float a, float b, float c	 )
	{
	float eps=0.001f;
	float S1, S2; 
	float sigma1,sigma2;
	float norm1;
	norm1=0;
	float4 v;
	
	float o=min(a,c);
	if((b*b)/(o*o)>eps && o*o>0){//Case diagonals are not dominant
	//printf("Non-Diagonal %f,%f ", b*b/(o*o),eps);
	float d=a+c;
	float e=a-c;
	
	float f=4*b*b+e*e;
	if (f<0){f=0;}else{f=sqrt(f);}

	if (d-f>0){
	sigma2=(d-f)*0.5f;}else{sigma2=0;
	}
	if (d+f>0){
	sigma1=(d+f)*0.5f;}else{sigma1=0;} 
	
	
	float g=(e+f)*0.5f;
	float h=(e-f)*0.5f;

	v.s0=1;
	v.s1=-h/b;//v1

	norm1=(hypot(v.s0,v.s1));
	
	v.s01=v.s01/norm1;
	
	v.s2=v.s1	;
	v.s3=-v.s0;
	}
	else{//case diagonals are Dominant, or Matirx constant 0
	//printf("Diagonal");
		sigma1=a;
		sigma2=c;
		v.s0=1;
		v.s1=0;
		v.s2=0;
		v.s3=-1;
		}
		
	 float decomposition[6] = {sigma1,sigma2,v.s0,v.s1,v.s2,v.s3};
	 return decomposition;
	}

	
	
	
	//Projection Nuclear Norm
	__kernel void update_NormV_Nucl(__global float3 *V,__global float *normV,const float alphainv,const float NumberOfChannels)
	{	
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1),z=get_global_id(2);
  size_t i = Nx*y + x+Nx*Ny*z;
	
	
	float S0, S1, S2; 
	float a=0,b=0,c=0,r=0,t=0,d=0;
	float8 VSV;
	float sigma0,sigma1,sigma2;
	float norm1,norm2,norm3;
	
	norm1=0;
	norm2=0;
	norm3=0;
	float* X;
	float3 v0,v1,v2;
	float u1,u2;
	
	//Computing A*A (A adjoint times A)
	for(int j=0;j<NumberOfChannels;j++)
	{	size_t k=i+Nx*Ny*Nz*j;
		
		a+=V[k].s0*V[k].s0;
		b+=V[k].s1*V[k].s1;
		c+=V[k].s2*V[k].s2;
		r+=V[k].s0*V[k].s1;
		t+=V[k].s0*V[k].s2;
		d+=V[k].s1*V[k].s2;
	}
		
		//A= [ a r t
		//	  r b d
		//	  t d c]
	
	 //printf( "A:\n%f,%f,%f \n%f,%f, %f\n%f,%f, %f \n", a,r,t, r,b,d, t,d,c);
	//Decomposition of Matrix
	//printf("");
	float epsdt=0.0001f;
	float epsr =0.0001f;
	float epsoo=0.0001f/alphainv;
	float ndt=d*d+t*t;
	float oo=min(a,b);
	if (ndt<epsdt*oo*oo && oo>epsoo){//printf("ndt small");
	
		//A= [ a r 0
		//	  r b 0
		//	  0 0 c]
		
		
		 X=diagonalization2d(a ,r, b);
		
		//printf(" %f %f, %f %f %f %f \n",X[0],X[1],X[2],X[3],X[4],X[5]);
		sigma0=X[0];
		sigma1=X[1];
		if (c*c<ndt)
		{sigma2=1/epsdt	;
		}else{
		sigma2=c;
		}
		v0=(float3)(X[2],X[3],0.f);
		v1=(float3)(X[4],X[5],0.f);
		v2=(float3)(0.f,0.f,1.f);
		//printf("v: %f %f %f,\n %f %f %f \n %f %f %f\n",v0.s0,v0.s1,v0.s2,v1.s0,v1.s1,v1.s2,v2.s0,v2.s1,v2.s2);
		//printf("sigma:, %f,%f,%f",sigma0,sigma1,sigma2);
		u1=1.f;
		u2=0.f;
	}
	else
	{
	if (oo<epsoo)
	{//printf("Case Zero matrix");
	
	if (a>=b)
	{
	sigma0= b;
	v0=(float3) (0.f,1.f,0.f);
	X=diagonalization2d(a ,t, c);
	sigma1=X[0] ;
	sigma2=X[1];
	v1=(float3)(X[2],0.f,X[3]);
	v2=(float3)(X[4],0.f,X[5]);
	}
	else{
	sigma0=a;
	v0=(float3)(1.f,0.f,0.f);
	X=diagonalization2d(b ,d, c);
	sigma1=X[0] ;
	sigma2=X[1];
	v1=(float3)(0.f,X[2],X[3]);
	v2=(float3)(0.f,X[4],X[5]);
	}

	u1=1.f;
	u2=0.f;
	
	
	}else{
	//printf("case2 %f\n",ndt);
	//#A= [ a r t
	//	#	  r b d
	//	#	  t d c]
	//	#with t^2+d^2!=0
	//#Precomputation
	float t1 = b*t - d*r;
	float t2 = a*d - t*r;
	
	u1=d/sqrt(ndt);
	u2=t/sqrt(ndt);
	
	//#Update elements
	b = ( d*(b*d + r*t) + t*(a*t + d*r) )/ndt;
	a = ( t*t1 + d*t2 )/ndt;
	r = ( -d*t1	 +t*t2 )/ndt;
	d=sqrt(ndt);
	//printf("Transformed Matrix:\n %f,%f,%f \n%f,%f, %f\n%f,%f, %f \n", a,r,0., r,b,d, 0.,d,c);
	
	//#A= [ a r 0
	//#	  r b d
	//#	  0 d c]
	//#with d!=0
	
	oo=min(c,b);
	if (fabs(r)<epsr*oo){	
		//printf("case2 r small\n");
		//#A= [ a 0 0
		//#	  0 b d
		//#	  0 d c]
		

		X=diagonalization2d(b,d,c);
		
		//printf(" %f %f, %f %f %f %f \n",X[0],X[1],X[2],X[3],X[4],X[5]);
		sigma0=a;
		sigma1=X[0];
		sigma2=X[1];	
			
		v0=(float3)(1.f,0.f,0.f);
		v1=(float3)(0.f,X[2],X[3]);
		v2=(float3)(0.f,X[4],X[5]);
		}
	else{//printf("case2 r big\n");
			//#A= [ a r 0
		//#	  r b d
		//#	  0 d c]
		//# with r and d not zero
		
	
		X=eigenvalues3d(a,b,c,r,0,d);
		sigma0 =X[0];
		sigma1 = X[1];
		sigma2 = X[2];
		//printf(" Eigenwerte durch 3d Formel %f %f %f \n",sigma0,sigma1,sigma2);	
		v0=crossproduct(a-sigma0,r,0,r,b-sigma0,d);
		norm1=sqrt(v0.s0*v0.s0+v0.s1*v0.s1+v0.s2*v0.s2);
		v0/=norm1;
			
		v1=crossproduct(v0.s0,v0.s1,v0.s2,0,d,c-sigma1);
		norm2=sqrt(v1.s0*v1.s0+v1.s1*v1.s1+v1.s2*v1.s2);
		v1/=norm2;
			
		v2=crossproduct(v0.s0,v0.s1,v0.s2,v1.s0,v1.s1,v1.s2);
		norm3=sqrt(v2.s0*v2.s0+v2.s1*v2.s1+v2.s2*v2.s2);
		v2/=norm3;
		//printf("\n\nInterior Eigenvectors:\n %f %f %f \n %f %f %f\n %f %f %f\n",v0.s0,v0.s1,v0.s2,v1.s0,v1.s1,v1.s2,v2.s0,v2.s1,v2.s2);
		}
		}	
	
	}
	
float3 w0, w1, w2;


//#Matrix U^TV berechnen
w0=(float3)(u1*v0.s0+u2*v0.s1,-u2*v0.s0+u1*v0.s1,v0.s2);
w1=(float3)(u1*v1.s0+u2*v1.s1,-u2*v1.s0+u1*v1.s1,v1.s2);
w2=(float3)(u1*v2.s0+u2*v2.s1,-u2*v2.s0+u1*v2.s1,v2.s2);
	
//printf("\n\nExterior Eigenvectors:\n %f %f %f \n %f %f %f\n %f %f %f\n",w0.s0,w0.s1,w0.s2,w1.s0,w1.s1,w1.s2,w2.s0,w2.s1,w2.s2);	
//Projection
	if (sigma0>0){sigma0=sqrt(sigma0);}else{ sigma0=0;}
	if (sigma1>0){sigma1=sqrt(sigma1);}else{ sigma1=0;}
	if (sigma2>0){sigma2=sqrt(sigma2);}else{ sigma2=0;}
	
//printf("eigenwerte: %f, %f, %f", sigma0,sigma1,sigma2);

S0=1;
S1=1;
S2=1;
if (sigma0*alphainv>1){
	S0=1.f/(alphainv*sigma0);}
if (sigma1*alphainv>1)
	{	S1=1.f/(alphainv*sigma1);}
if (sigma2*alphainv>1){
	S2=1.f/(alphainv*sigma2);}

//printf("\n S values %f,%f,%f",S0,S1,S2);



float8 M;	 //0, 3 ,4;
			// 3  1	 5
			// 4  5	 2


M.s0=S0*w0.s0*w0.s0+S1*w1.s0*w1.s0+S2*w2.s0*w2.s0;
M.s1=S0*w0.s1*w0.s1+S1*w1.s1*w1.s1+S2*w2.s1*w2.s1;	
M.s2=S0*w0.s2*w0.s2+S1*w1.s2*w1.s2+S2*w2.s2*w2.s2;
M.s3=S0*w0.s0*w0.s1+S1*w1.s0*w1.s1+S2*w2.s0*w2.s1;
M.s4=S0*w0.s0*w0.s2+S1*w1.s0*w1.s2+S2*w2.s0*w2.s2;
M.s5=S0*w0.s1*w0.s2+S1*w1.s1*w1.s2+S2*w2.s1*w2.s2;

//printf("\n\nProjectionmatrix:\n %f %f %f \n %f %f %f\n %f %f %f\n",M.s0,M.s3,M.s4,M.s3,M.s1,M.s5,M.s4,M.s5,M.s2);

for(int j=0;j<NumberOfChannels;j++)
{	size_t k=i+Nx*Ny*Nz*j;
	float A=V[k].s0;
	float B=V[k].s1;
	float C=V[k].S2;


	a = A*M.s0+B*M.s3+C*M.s4;
	b = A*M.s3+B*M.s1+C*M.s5;
	c = A*M.s4+B*M.s5+C*M.s2;
	V[k]=(float3)(a,b,c);
	
	
	}
normV[i]=sigma0+sigma1+sigma2;
}

//#############################################################################################################################################################
	
	
	
	
	
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
		//Projection  wrt uncorrelated
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

  w[i] =0;

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



""")

###########################

	""" Is used to ensure that no data is present where it makes no sense for data to be by setting sino to zero in such positions
		Input		
				sino		...	Np.array with sinogram data in question
				r_struct	... Defining the geometry of an object for the radon transform
				imageshape	... Imagedimensions of the image corresponding to the image (could probably be removed since information is also in r_struct)
		Ouput
				sinonew		... A new sinogram where pixels such that R1 is zero, where R is the radon transform and 1 is the constant 1 image. (This corresponds to all pixels	 we cann not obtain any mass in due to the geometry of the radontransform)
	"""
	def Make_function_feasible(sino,r_struct,imageshape):
		img=np.ones([imageshape[0],imageshape[0],sino.shape[2]])
		img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
		sino_gpu = clarray.zeros(queue, r_struct[2], dtype=float32, order='F')
		radon(sino_gpu, img_gpu, r_struct).wait()

		sino_gpu=sino_gpu.get()
		sino2=sino.copy()
		sino2[np.where(sino_gpu<=0)]=0

		return sino2

	def gradientt(grad,image,Number_Channels,stepsize,wait_for=None):
		return prg.gradient(queue,image.shape,None,grad.data,image.data, int32(Number_Channels),int32(stepsize), wait_for=wait_for)
	def diver(div,grad,Number_Channels,stepsize,wait_for=None):
		return prg.divergence(queue,div.shape,None,div.data,grad.data,int32(Number_Channels),int32(stepsize),wait_for=wait_for)
	def Symjacobi(jacobi,p,Number_Channels,stepsize,wait_for=None):
		return prg.Jacobian(queue,jacobi.shape[1:],None,jacobi.data,p.data,int32(Number_Channels),int32(stepsize),wait_for=wait_for)
	def jac_div(p,jacobi,Number_Channels,stepsize,wait_for=None):
		return prg.div_jac(queue,p.shape[1:],None,p.data,jacobi.data,int32(Number_Channels),int32(stepsize),wait_for=wait_for)


	def Gap_TGV_KL(U,Sino,P,V,W,R,KStarr,U0,mu,alpha,Grad,Jacobi,Diver,Jac_diver):
		NN=U[:,0,:].size
		Dataprim=0;Regprim=0
		Dataprim2=0; infeasible_points=0
		grad_sum=0;jacobi_sum=0
		Datadual=0;F1dual=0
		Dualcond1=0;Dualcond2=0
		for i in range(0,U.shape[1]):
			
			sino=Sino[:,i,:];p=P[:,:,i,:];v=V[:,:,i,:];w=W[:,:,i,:];r=R[:,i,:];Kstarr=KStarr[:,i,:];u0=U0[:,i,:]
			
			grad=Grad[:,:,i,:];jacobi=Jacobi[:,:,i,:];diver=Diver[:,i,:];jac_diver=Jac_diver[:,:,i,:]
			
			Dataprim+=mu[i]/2*np.linalg.norm((u0-sino).reshape(u0.size))**2
			
			grad=np.linalg.norm(grad.reshape(4,NN)-p.reshape(4,NN),axis=0)
			
			grad_sum+=grad**2
		
			jacobi_sum+=np.linalg.norm(jacobi,axis=0)**2
			jacobi_sum+=jacobi[3]**2+jacobi[4]**2+jacobi[5]**2

			Datadual+=1./(mu[i]*2.)*np.linalg.norm(r.reshape(r.size),2)**2+np.dot(r.reshape(r.size),u0.reshape(r.size))

			Dualcond1+=-v.reshape(v.size)-jac_diver.reshape(v.size)

			Dualcond2+=-diver.reshape(diver.size)+Kstarr.reshape(diver.size)
			

		grad_sum=np.linalg.norm(grad_sum.reshape(NN)**0.5,1)
	
		jacobi_sum=np.linalg.norm(jacobi_sum.reshape(NN)**0.5,1);
	
		Regprim+=alpha[1]*grad_sum+alpha[0]*jacobi_sum	
		
		Dualcond1=np.linalg.norm(Dualcond1)
	
		Dualcond2=np.linalg.norm(Dualcond2[np.where(Dualcond2<=0)])
		
		F1dual=Dualcond1+Dualcond2


		
		G=Dataprim+Regprim
		print('\n',G,F1dual,Datadual,'\n')
		return G
											   

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
		

########### 
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
							 mu.data,float32(Number_channels), wait_for=wait_for)
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
												
											 
	if regularisationmethod=='Frobenius 3D Reconstruction':		
		
		if regularisation =='TGV' and regularisationmethod=='Frobenius 3D Reconstruction':										   
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_frob(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
					   wait_for=wait_for)
						 
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_frob(normW.queue, normW.shape, None, W.data,normW.data, float32(1.0/alpha),float32(Number_Channels),
						 wait_for=wait_for)
						 
		if regularisation=='TV' and regularisationmethod=='Frobenius 3D Reconstruction':
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_frob(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
					   wait_for=wait_for)
								
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_empty(normW.queue, normW.shape, None,wait_for=wait_for)
	
	if regularisationmethod=='Uncorrelated 3D Reconstruction':
		 
		if regularisation=='TGV' and regularisationmethod=='Uncorrelated 3D Reconstruction':													  
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_unchor(V.queue, V.shape[1:], None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_unchor(W.queue, W.shape[1:], None, W.data,normW.data, float32(1.0/alpha),float32(Number_Channels),
						 wait_for=wait_for)		  
								   
		if regularisation=='TV' and regularisationmethod=='Uncorrelated 3D Reconstruction':								   
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_unchor(V.queue, V.shape[1:], None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
					   wait_for=wait_for)
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_empty(normW.queue, normW.shape, None,wait_for=wait_for)										 
	
	
	if regularisationmethod=='Nuclear 3D Reconstruction':
		if regularisation=='TGV' and regularisationmethod=='Nuclear 3D Reconstruction':												  
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_Nucl(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_frob(normW.queue, normW.shape, None, W.data,normW.data, float32(1.0/alpha),float32(Number_Channels),
						 wait_for=wait_for)			 
						 
		if regularisation=='TV' and regularisationmethod=='Nuclear 3D Reconstruction':									
			def update_NormV(V,normV,alpha,Number_Channels,wait_for=None):
				return prg.update_NormV_Nucl(normV.queue, normV.shape, None, V.data,normV.data, float32(1.0/alpha),float32(Number_Channels),
						wait_for=wait_for)
			def update_NormW(W,normW,alpha,Number_Channels,wait_for=None):
				return prg.update_NormW_empty(normW.queue, normW.shape, None,wait_for=wait_for)	  

#################
## main iteration

	def tgv_radon(F0, img_shape, angles, alpha,mu, maxiter,plott,Stepsize):
		print('########## Starting	'+str(discrepancy)+' '+str(regularisation)+' '+str(regularisationmethod)+' ############')
		f0=F0[0,:,:,0];	
		
		#Computing Structure of Radon Transform
		if angles is None:
			angles = f0.shape[1]
		r_struct = radon_struct(queue, img_shape, angles,
							n_detectors=f0.shape[0])
		#######Preparations
		[Number_Channels,Nx,Number_angles,Nz]=F0.shape
		fig_data=np.zeros([Nx,Nx*Number_Channels])		
		

		##Rearanging Values in F
		Fnew=np.zeros([Nx,Number_angles,Nz*Number_Channels])
		for i in range(0,Nz):
			for j in range(0,Number_Channels):
				Fnew[:,:,i+Nz*j]=F0[j,:,:,i]
		
		
		Fnew=Make_function_feasible(Fnew,r_struct,img_shape)
		
		
		###Initialising Variables
		F= clarray.to_device(queue, require(Fnew, float32, 'F'))		
		U=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		P=clarray.zeros(queue, (4,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		U_=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		P_=clarray.zeros(queue, (4,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		V=clarray.zeros(queue, (4,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		if regularisation.upper() =='TGV':
			W=clarray.zeros(queue, (8,img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
			normW=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz), dtype=float32, order='F')
		if regularisation.upper() =='TV':
			W=clarray.zeros(queue, (1), dtype=float32, order='F')
			normW=clarray.zeros(queue, (1), dtype=float32, order='F')
		Lamb=clarray.zeros(queue,(F0.shape[1],F0.shape[2],Nz*Number_Channels),dtype=float32, order='F')
		KU=clarray.zeros(queue, (F0.shape[1],F0.shape[2],Nz*Number_Channels), dtype=float32, order='F')
		KSTARlambda=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz*Number_Channels), dtype=float32, order='F')
		
		normV=clarray.zeros(queue, (img_shape[0],img_shape[1],Nz), dtype=float32, order='F')

	
		#Computing estimates for Parameter
		normest = radon_normest(queue, r_struct)
		Lsqr = 17
		sigma = 1.0/sqrt(Lsqr)
		tau = 1.0/sqrt(Lsqr)
		alpha = array(alpha)/(normest**2)

		if discrepancy == 'KL':
			mu=clarray.to_device(queue, require(mu, float32, 'F'))
		if discrepancy == 'L2':
			mu=clarray.to_device(queue, require(mu/(sigma+mu), float32, 'F'))
	
		fig = None
		
		#Primal Dual Iterations
		for i in range(maxiter):		
			
			#Dual Update
			V.add_event(update_v(V, U_, P_, sigma, alpha[1],Stepsize,Number_Channels,
							 wait_for=U_.events + P_.events))
			W.add_event(update_w(W, P_, sigma, alpha[0],Stepsize,Number_Channels, wait_for=P_.events))
				
			normV.add_event(update_NormV(V,normV,alpha[1],Number_Channels,wait_for=V.events))	
			normW.add_event(update_NormW(W,normW,alpha[0],Number_Channels,wait_for=W.events))	
				
			KU.add_event(radon(KU, U_, r_struct, wait_for=U_.events))		 
			Lamb.add_event(update_lambda(Lamb, KU, F, sigma,mu, normest,Number_Channels,
									 wait_for=KU.events + F.events))			   
			KSTARlambda.add_event(radon_ad(KSTARlambda, Lamb, r_struct,
										wait_for=Lamb.events))
			
			#Primal Update
			U_.add_event(update_u(U_, U, V, KSTARlambda, tau, normest,Stepsize,Number_Channels,
							 wait_for=U.events + normV.events +	 KSTARlambda.events))
		
			P_.add_event(update_p(P_, P, V, W, tau,Stepsize,Number_Channels,
							 wait_for= P.events + normV.events + normW.events))
										
			U.add_event(update_extra(U_, U, wait_for=U.events + U_.events))	
				
			P.add_event(update_extra(P_, P, wait_for=P.events + P_.events))

			(U, U_, P,	P_) = (U_, U,  P_,	P)

			#import ipdb as pdb; pdb.set_trace()

			#Plot Current Iteration
			if (i % math.ceil(maxiter/100.) == 0):
				for x in [U, U_, P, P_]:
					x.finish()

				if i% math.ceil(maxiter*plott[1])==0:
					if len(current)>3:
						a=str(current[0])+', '+str(current[1])+'...'+str(current[len(current)-1])
					else:
						a=str(current)
				Progress=start+sectionwidth*float(i)/maxiter
				#.stdout.write('\rOverall progress at {:3.0%}'.format(Progress)+'. Progress of current computation {:3.0%}'.format(float(i)/maxiter)+ ' for Section '+str(a) )
				print  ('Overall progress at {:3.0%}'.format(Progress)+'. Progress of current computation {:3.0%}'.format(float(i)/maxiter)+ ' for Section '+str(a) )
				sys.stdout.flush()
				text='Iteration %d' %i+' of section '+a+'. Overall_progress at {:3.0%}'.format(Progress)	
			
			if plott[0]==1:
				if i% math.ceil(maxiter*plott[1])==0:
					for j in range(0,Number_Channels):
						fig_data[:,j*Nx:(j+1)*Nx]=U.get()[:,:,Nz//2+j*Nz]
					if len(current)>3:
						a=str(current[0])+', '+str(current[1])+'...'+str(current[len(current)-1])
					else:
						a=str(current)
							
					text='iteration %d' %i+' of section '+a+'. Overall_progress at {:3.0%}'.format(Progress)	
					
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
	
		#sys.stdout.write('\rProgress at {:3.0%}'.format(float(maxiter)/maxiter))
		print ('Progress at {:3.0%}'.format(float(maxiter)/maxiter))
		sys.stdout.flush()
						
		print('Iterations done for section '+a)
		
		
		#Rearanging Data
		[Nx,Number_angles]=KU.get()[:,:,0].shape
		Sinograms=np.zeros([Number_Channels,Nx,Nz,Number_angles])		
		Solution=np.zeros([Number_Channels,img_shape[0],img_shape[1],Nz])
		
		for j in range(0,Number_Channels):
			for i in range(0,Nz):
				Sinograms[j,:,i,:]=KU.get()[:,:,i+Nz*j]
				Solution[j,:,:,i]=U.get()[:,:,i+Nz*j]

		Sinograms2=np.zeros([Number_Channels,Number_angles,Nz,Nx])		
		for k in range(0,Number_angles):
			for j in range(0,Number_Channels):
				Sinograms2[j,k,:,:]=Sinograms[j,:,:,k].T
			
	
		return Solution,Sinograms2
		



	#################Prelimineare ############
	
	Number_Channels=len(sino)
	[Number_of_angles,Ny,Nx]=sino[0].shape
	
	sino_new=np.zeros([Number_Channels,Number_of_angles,Nx,Ny])

	####Reordering Dimensions
	for j in range(0,Number_Channels):
		for i in range(0,Number_of_angles):
			sino_new[j,i,:,:]=sino[j][i,:,:].T	
	sino=np.zeros([Number_Channels,Nx,Number_of_angles,Ny])
	for j in range(0,Number_Channels):
		for i in range(0,Number_of_angles):
			sino[j,:,i,:]=sino_new[j,i,:,:]
	
	
	
	#Executing Primal Dual Algorithm
	u,sinogram = tgv_radon(sino, (sino.shape[1],sino.shape[1]), angles, Parameter,mu, maxiter,plott,Stepsize)
	
	return u,sinogram

