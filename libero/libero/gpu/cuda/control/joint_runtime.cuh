#include "precision.cuh"
#pragma once
#include <math.h>
#include <stdint.h>
namespace joint_cuda {
__device__ void matvec(int,int*,int*,real_t*,real_t*,real_t*);
__device__ inline real_t clip(real_t x,real_t a,real_t b){return fmin(fmax(x,a),b);}
__device__ inline void refresh(int w,int*active,int*needs,int*valid,int*error,
 real_t*pm,real_t*pq,real_t*pv,real_t*m,real_t*q,real_t*v,int n){
 if(active[w]!=1||needs[w]!=1)return;
 for(int i=0;i<n*n;i++)if(!isfinite(pm[w*n*n+i])){error[w]=2;return;}
 for(int i=0;i<n;i++)if(!isfinite(pq[w*n+i])||!isfinite(pv[w*n+i])){error[w]=2;return;}
 for(int i=0;i<n*n;i++)m[w*n*n+i]=pm[w*n*n+i];
 for(int i=0;i<n;i++){q[w*n+i]=pq[w*n+i];v[w*n+i]=pv[w*n+i];}valid[w]=1;
}
// config layout (all n-wide): input lo/hi, output lo/hi, kp/kd,
// target lo/hi, kp lo/hi, damping lo/hi. Robot clip bounds separate.
// history counters: derivative ring pointer/size, prior saturation, interpolation step.
__device__ inline void prepare(int w,int*active,int*policy,int*valid,int*error,
 real_t*action,real_t*q,real_t*v,real_t*cfg,real_t*goal,real_t*start,real_t*gains,
 real_t*last,real_t*sum,real_t*ring,real_t*history,real_t*drive,
 int n,int adim,int family,int impedance,int interpolate,int total){
 if(active[w]!=1||error[w])return;if(valid[w]!=1){error[w]=3;return;}
 real_t*h=history+w*4;int ptr=(int)h[0],size=(int)h[1];
 if(policy[w]==1){
  int offset=impedance==2?2*n:impedance==1?n:0;
  for(int i=0;i<n;i++){
   real_t a=action[w*adim+offset+i];if(!isfinite(a)){error[w]=2;return;}
   real_t inlo=cfg[i],inhi=cfg[n+i],outlo=cfg[2*n+i],outhi=cfg[3*n+i];
   real_t scaled=(clip(a,inlo,inhi)-REAL_C(.5)*(inhi+inlo))*fabs(outhi-outlo)/fabs(inhi-inlo)+REAL_C(.5)*(outhi+outlo);
   if(interpolate)start[w*n+i]=goal[w*n+i];
   goal[w*n+i]=clip(family==1?q[w*n+i]+scaled:scaled,cfg[6*n+i],cfg[7*n+i]);
   real_t kp=cfg[4*n+i],kd=cfg[5*n+i];
   if(impedance){kp=clip(action[w*adim+(impedance==2?n:0)+i],cfg[8*n+i],cfg[9*n+i]);
    real_t damping=impedance==2?clip(action[w*adim+i],cfg[10*n+i],cfg[11*n+i]):REAL_C(1.);kd=REAL_C(2.)*sqrt(kp)*damping;}
   gains[w*2*n+i]=kp;gains[w*2*n+n+i]=kd;
  }if(interpolate)h[3]=0;
 }
 if(family==2){ptr=(ptr+1)%5;if(size<5)size++;h[0]=ptr;h[1]=size;}
 for(int i=0;i<n;i++){
  real_t target=goal[w*n+i];
  if(interpolate)target=start[w*n+i]+(target-start[w*n+i])/(total-(int)h[3]);
  real_t value=target;
  if(family==1)value=(target-q[w*n+i])*gains[w*2*n+i]-v[w*n+i]*gains[w*2*n+n+i];
  if(family==2){
   real_t err=target-v[w*n+i],delta=err-last[w*n+i];last[w*n+i]=err;
   ring[w*5*n+ptr*n+i]=delta;if(h[2]==0)sum[w*n+i]+=err;
   real_t avg=0;for(int k=0;k<size;k++)avg+=ring[w*5*n+k*n+i];avg/=size;
   real_t kp=cfg[4*n+i];value=kp*err+(kp*REAL_C(.005))*sum[w*n+i]+(kp*REAL_C(.001))*avg;
  }
  drive[w*n+i]=value;if(!isfinite(value)){error[w]=2;return;}
 }if(interpolate&&h[3]<total-1)h[3]++;
}
__device__ inline void finish(int w,int*active,int*error,int*needs,real_t*action,
 real_t*bias,real_t*drive,real_t*product,real_t*raw,real_t*ctrl,real_t*range,
 int*ids,real_t*gripper,real_t*history,int n,int nu,int adim,int family,int ng,int gdof,
 int gripmode,real_t speed){
 if(active[w]!=1||error[w])return;
 real_t difference=0;
 for(int i=0;i<n;i++){
  real_t t=(family==1?product[w*n+i]:drive[w*n+i])+bias[w*n+i];
  if(!isfinite(t)){error[w]=2;return;}raw[w*n+i]=t;
  real_t clipped=clip(t,range[2*i],range[2*i+1]);ctrl[w*nu+ids[i]]=clipped;difference+=fabs(clipped-t);
 }if(family==2)history[w*4+2]=difference==0?0:1;
 for(int g=0;g<ng;g++){
  real_t a=gdof?action[w*adim+adim-gdof+(gdof==1?0:g)]:0;
  real_t f=0;real_t sign=(a>0)-(a<0);
  if(gripmode==1||gripmode==2){real_t direction=(g==0?-REAL_C(1.):REAL_C(1.))*(gripmode==1?REAL_C(1.):-REAL_C(1.));
   f=clip(gripper[w*(ng?ng:1)+g]+direction*speed*sign,-1,1);}
  else if(gripmode==3||gripmode==4)f=clip(gripper[w*(ng?ng:1)+g]+(gripmode==3?REAL_C(1.):-REAL_C(1.))*speed*sign,-1,1);
  else if(gripmode==5)f=g==3?-REAL_C(1.):clip(gripper[w*(ng?ng:1)+g]+speed*clip(a,-1,1),-1,1);
  else f=clip(a,-1,1);
  if(!isfinite(f)){error[w]=2;return;}gripper[w*(ng?ng:1)+g]=f;
  int k=n+g;ctrl[w*nu+ids[k]]=REAL_C(.5)*(range[2*k]+range[2*k+1])+REAL_C(.5)*(range[2*k+1]-range[2*k])*f;
 }needs[w]=1;
}
__device__ inline void reset(int w,int*mask,real_t*pq,real_t*cfg,real_t*goal,
 real_t*start,real_t*gains,real_t*last,real_t*sum,real_t*ring,real_t*history,
 real_t*gripper,int*error,int*valid,int*needs,int n,int ng,int family,int clear){
 if(mask[w]!=1)return;
 for(int i=0;i<n;i++){goal[w*n+i]=family==1?pq[w*n+i]:0;start[w*n+i]=0;
  gains[w*2*n+i]=cfg[4*n+i];gains[w*2*n+n+i]=cfg[5*n+i];last[w*n+i]=sum[w*n+i]=0;}
 for(int i=0;i<5*n;i++)ring[w*5*n+i]=0;
 for(int i=0;i<4;i++)history[w*4+i]=0;history[w*4]=-1;
 if(clear)for(int i=0;i<ng;i++)gripper[w*(ng?ng:1)+i]=0;
 error[w]=0;valid[w]=0;needs[w]=1;
}
__device__ inline void reset_public(int w,int*mask,int*first,real_t*source,real_t*ctrl,
 int*valid,int64_t*written,int64_t*event,int nu){
 if(mask[w]!=1||first[w])return;
 for(int i=0;i<nu;i++)if(!isfinite(source[w*nu+i])){first[w]=202;valid[w]=0;return;}
 for(int i=0;i<nu;i++)ctrl[w*nu+i]=source[w*nu+i];event[w]=written[w]=0;valid[w]=1;
}
__device__ inline void gather(int w,int*mask,int*first,int64_t*state,int64_t*source,
 int64_t*epoch,int*valid,int64_t*written,real_t*qp,real_t*qv,real_t*qm,
 int*qi,int*vi,int*sparse,real_t*q,real_t*v,real_t*m,int nq,int nv,int nm,int n){
 if(mask[w]!=1||first[w]||state[w]!=source[w]||valid[w]!=1||written[w]!=epoch[w])return;
 for(int i=0;i<n;i++){
  real_t a=qp[w*nq+qi[i]],b=qv[w*nv+vi[i]];
  if(!isfinite(a)||!isfinite(b)){first[w]=202;return;}q[w*n+i]=a;v[w*n+i]=b;
  for(int j=0;j<n;j++){int s=sparse[i*n+j];real_t t=s<0?0:qm[w*nm+s];
   if(!isfinite(t)){first[w]=202;return;}m[w*n*n+i*n+j]=t;}
 }
}
__device__ inline void live_bias(int w,int*mask,int*first,int64_t*state,int64_t*source,
 int64_t*epoch,int*valid,int64_t*written,real_t*bias,real_t*selected,int*vi,
 int*parent_valid,int64_t*parent_written,int nv,int n){
 parent_valid[w]=0;
 if(mask[w]!=1||first[w]||state[w]!=source[w]||valid[w]!=1||written[w]!=epoch[w])return;
 for(int i=0;i<n;i++){real_t b=bias[w*nv+vi[i]];if(!isfinite(b)){first[w]=202;return;}selected[w*n+i]=b;}
 parent_written[w]=epoch[w];parent_valid[w]=1;
}
}
