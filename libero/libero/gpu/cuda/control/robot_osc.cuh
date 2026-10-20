#include "precision.cuh"
#pragma once
#include <math.h>
#include <stdint.h>
namespace osc_robot_cuda {
__device__ inline void project(int w,int*active,int*error,real_t*arm,real_t*action,
 real_t*gripper,real_t*ctrl,int*ids,real_t*range,int nu,int nj,int ng,int gdof,int mode,real_t speed){
 if(active[w]!=1||error[w])return;
 for(int i=0;i<nj;i++){
  real_t x=arm[w*7+i];if(!isfinite(x)){error[w]=2;return;}
  ctrl[w*nu+ids[i]]=fmin(fmax(x,range[2*i]),range[2*i+1]);
 }
 for(int g=0;g<ng;g++){
  real_t x=gdof?action[w*(gdof?gdof:1)+(gdof==1?0:g)]:0;
  x=fmin(fmax(x,-REAL_C(1.)),REAL_C(1.));real_t sign=(x>0)-(x<0),value;
  if(mode==1||mode==2){real_t direction=(g==0?-REAL_C(1.):REAL_C(1.))*(mode==1?REAL_C(1.):-REAL_C(1.));value=fmin(fmax(gripper[w*(ng?ng:1)+g]+direction*speed*sign,-REAL_C(1.)),REAL_C(1.));}
  else if(mode==3||mode==4)value=fmin(fmax(gripper[w*(ng?ng:1)+g]+(mode==3?REAL_C(1.):-REAL_C(1.))*speed*sign,-REAL_C(1.)),REAL_C(1.));
  else if(mode==5)value=g==3?-REAL_C(1.):fmin(fmax(gripper[w*(ng?ng:1)+g]+speed*x,-REAL_C(1.)),REAL_C(1.));
  else value=x;
  if(!isfinite(value)){error[w]=2;return;}gripper[w*(ng?ng:1)+g]=value;
  int k=nj+g;ctrl[w*nu+ids[k]]=REAL_C(.5)*(range[2*k]+range[2*k+1])+REAL_C(.5)*(range[2*k+1]-range[2*k])*value;
 }
}
__device__ inline void reset(int w,int*mask,real_t*gripper,int ng,int clear){
 if(mask[w]!=1||!clear)return;for(int g=0;g<ng;g++)gripper[w*(ng?ng:1)+g]=0;
}
__device__ inline void reset_public(int w,int*mask,int*first,real_t*source,real_t*ctrl,
 int*valid,int64_t*written,int64_t*event,int nu){
 if(mask[w]!=1||first[w])return;
 for(int i=0;i<nu;i++)if(!isfinite(source[w*nu+i])){first[w]=202;valid[w]=0;return;}
 for(int i=0;i<nu;i++)ctrl[w*nu+i]=source[w*nu+i];event[w]=written[w]=0;valid[w]=1;
}
}
