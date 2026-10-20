#include "precision.cuh"
#pragma once
#include <math.h>
namespace osc_geometry_cuda {
constexpr real_t eps=REAL_C(8.881784197001252e-16);
constexpr real_t pi=REAL_C(3.1415926535897932384626433832795);
__device__ inline void axis_matrix(const real_t*axis,real_t*m){
 real_t a=sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);
 real_t q[4];if(a==0){q[0]=q[1]=q[2]=0;q[3]=1;}
 else{real_t s=sin(a*REAL_C(.5))/a;for(int i=0;i<3;i++)q[i]=axis[i]*s;q[3]=cos(a*REAL_C(.5));}
 // Same explicit FP32 quaternion boundary as the original source.
 float f[4]={(float)q[3],(float)q[0],(float)q[1],(float)q[2]};
 real_t acc=0;for(int i=0;i<4;i++)acc+=(real_t)(f[i]*f[i]);float n=(float)acc;
 real_t scale=sqrt(REAL_C(2.)/n);for(int i=0;i<4;i++)f[i]=(float)(f[i]*scale);
 m[0]=REAL_C(1.)-(real_t)(f[2]*f[2])-(real_t)(f[3]*f[3]);m[4]=REAL_C(1.)-(real_t)(f[1]*f[1])-(real_t)(f[3]*f[3]);m[8]=REAL_C(1.)-(real_t)(f[1]*f[1])-(real_t)(f[2]*f[2]);
 m[1]=(real_t)(f[1]*f[2]-f[3]*f[0]);m[2]=(real_t)(f[1]*f[3]+f[2]*f[0]);m[3]=(real_t)(f[1]*f[2]+f[3]*f[0]);
 m[5]=(real_t)(f[2]*f[3]-f[1]*f[0]);m[6]=(real_t)(f[1]*f[3]-f[2]*f[0]);m[7]=(real_t)(f[2]*f[3]+f[1]*f[0]);
}
__device__ inline void euler2mat(const real_t*e,real_t*m){
 real_t ai=-e[2],aj=-e[1],ak=-e[0];real_t si=sin(ai),sj=sin(aj),sk=sin(ak),ci=cos(ai),cj=cos(aj),ck=cos(ak);
 real_t cc=ci*ck,cs=ci*sk,sc=si*ck,ss=si*sk;
 m[8]=cj*ck;m[7]=sj*sc-cs;m[6]=sj*cc+ss;m[5]=cj*sk;m[4]=sj*ss+cc;m[3]=sj*cs-sc;m[2]=-sj;m[1]=cj*si;m[0]=cj*ci;
}
__device__ inline void mat2euler(const real_t*input,real_t*e){
 float m[9];for(int i=0;i<9;i++)m[i]=(float)input[i];
 float s=m[0]*m[0]+m[3]*m[3];real_t cy=sqrt((real_t)s);
 real_t x,y,z;if(cy>eps){x=atan2((real_t)m[7],(real_t)m[8]);y=atan2(-(real_t)m[6],cy);z=atan2((real_t)m[3],(real_t)m[0]);}
 else{x=atan2(-(real_t)m[5],(real_t)m[4]);y=atan2(-(real_t)m[6],cy);z=0;}
 e[0]=(float)x;e[1]=(float)y;e[2]=(float)z;
}
// Rotation geometry replaces the original 4x4 leading-eigenvector extraction.
// All input entries are rounded to the same FP32 source boundary. The largest
// component is >=1/2 on SO(3), avoiding the small-angle divisor of trace-only
// formulas. Local tests include the original K eigenvector residual/angle.
__device__ inline void mat2quat_geometry(const real_t*input,real_t*q){
 real_t m[9];for(int i=0;i<9;i++)m[i]=(float)input[i];
 real_t t=m[0]+m[4]+m[8],s;
 if(t>0){s=2*sqrt(t+1);q[3]=REAL_C(.25)*s;q[0]=(m[7]-m[5])/s;q[1]=(m[2]-m[6])/s;q[2]=(m[3]-m[1])/s;}
 else if(m[0]>m[4]&&m[0]>m[8]){s=2*sqrt(1+m[0]-m[4]-m[8]);q[3]=(m[7]-m[5])/s;q[0]=REAL_C(.25)*s;q[1]=(m[1]+m[3])/s;q[2]=(m[2]+m[6])/s;}
 else if(m[4]>m[8]){s=2*sqrt(1+m[4]-m[0]-m[8]);q[3]=(m[2]-m[6])/s;q[0]=(m[1]+m[3])/s;q[1]=REAL_C(.25)*s;q[2]=(m[5]+m[7])/s;}
 else{s=2*sqrt(1+m[8]-m[0]-m[4]);q[3]=(m[3]-m[1])/s;q[0]=(m[2]+m[6])/s;q[1]=(m[5]+m[7])/s;q[2]=REAL_C(.25)*s;}
 real_t n=0;for(int i=0;i<4;i++)n+=q[i]*q[i];n=sqrt(n);real_t sign=q[3]<0?-REAL_C(1.):REAL_C(1.);
 for(int i=0;i<4;i++)q[i]=(float)(sign*q[i]/n);
}
__device__ inline void quat2mat(const real_t*input,real_t*m){
 float q[4]={(float)input[3],(float)input[0],(float)input[1],(float)input[2]};
 float n=0;for(int i=0;i<4;i++)n+=q[i]*q[i];
 if(n<eps){for(int i=0;i<9;i++)m[i]=i%4==0?1:0;return;}
 real_t scale=sqrt(REAL_C(2.)/n);for(int i=0;i<4;i++)q[i]=(float)(q[i]*scale);
 float a[16];for(int i=0;i<4;i++)for(int j=0;j<4;j++)a[i*4+j]=q[i]*q[j];
 m[0]=REAL_C(1.)-a[10]-a[15];m[1]=a[6]-a[12];m[2]=a[7]+a[8];
 m[3]=a[6]+a[12];m[4]=REAL_C(1.)-a[5]-a[15];m[5]=a[11]-a[4];
 m[6]=a[7]-a[8];m[7]=a[11]+a[4];m[8]=REAL_C(1.)-a[5]-a[10];
}
__device__ inline void slerp(const real_t*input0,const real_t*input1,real_t fraction,real_t*out){
 real_t a[4],b[4],an=0,bn=0;for(int i=0;i<4;i++){an+=input0[i]*input0[i];bn+=input1[i]*input1[i];}
 an=sqrt(an);bn=sqrt(bn);for(int i=0;i<4;i++){a[i]=input0[i]/an;b[i]=input1[i]/bn;}
 real_t dot=0;for(int i=0;i<4;i++)dot+=a[i]*b[i];
 if(fraction==0){for(int i=0;i<4;i++)out[i]=a[i];return;}
 if(fraction==1){for(int i=0;i<4;i++)out[i]=b[i];return;}
 if(fabs(fabs(dot)-1)<eps){for(int i=0;i<4;i++)out[i]=a[i];return;}
 if(dot<0){dot=-dot;for(int i=0;i<4;i++)b[i]=-b[i];}
 real_t angle=acos(fmin(fmax(dot,-REAL_C(1.)),REAL_C(1.)));
 if(fabs(angle)<eps){for(int i=0;i<4;i++)out[i]=a[i];return;}
 real_t inv=REAL_C(1.)/sin(angle),wa=sin((1-fraction)*angle)*inv,wb=sin(fraction*angle)*inv;
 for(int i=0;i<4;i++)out[i]=a[i]*wa+b[i]*wb;
}
__device__ inline void interpolate_euler(const real_t*start,const real_t*goal,real_t fraction,real_t*out){
 real_t ma[9],mb[9],qa[4],qb[4],q[4],m[9];euler2mat(start,ma);euler2mat(goal,mb);
 mat2quat_geometry(ma,qa);mat2quat_geometry(mb,qb);slerp(qa,qb,fraction,q);quat2mat(q,m);mat2euler(m,out);
}
__device__ inline void limit_matrix(real_t*m,const real_t*limit){
 bool present=false;for(int i=0;i<6;i++)present=present||limit[i]!=0;if(!present)return;
 real_t e[3];mat2euler(m,e);bool limited=false;
 for(int i=0;i<3;i++){
  real_t lo=limit[i],hi=limit[3+i];
  if(lo<hi ? lo<e[i]&&e[i]<hi : lo<e[i]||e[i]<hi)continue;
  limited=true;real_t lower=e[i]-lo;if(lower>pi)lower-=2*pi;else if(lower<-pi)lower+=2*pi;
  real_t higher=e[i]-hi;
  // Preserve the original source's lower-distance-dependent high-side wrap.
  if(lower>pi)higher-=2*pi;else if(lower<-pi)higher+=2*pi;
  e[i]=lower<higher?lo:hi;
 }if(limited)euler2mat(e,m);
}
}
