#include "precision.cuh"
#pragma once
#include "euler.cuh"
// Explicit E01-only fast ABI; qualification and installation default off.
#if defined(CF_EULER_NV)
#ifndef CF_EULER_LEAN_JOINT_PARALLEL
#define CF_EULER_LEAN_JOINT_PARALLEL 1
#endif
namespace cf_euler_lean {
static_assert(4==1 || 4==4,"Euler RHS slots");
struct Work {
 real_t qm[CF_EULER_NM],ld[CF_EULER_NM];
 real_t rhs[4*CF_EULER_NV];
 real_t solution[4*CF_EULER_NV];
 real_t inv[CF_EULER_NV],si[CF_EULER_NV];
 real_t pos[CF_EULER_NQ],vel[CF_EULER_NV],used[CF_EULER_NV];
 int pivot[CF_EULER_NV];
 int status,mass_status,damped,mass_active,warnings,last;
 real_t next_time;
};
__device__ __forceinline__ real2_t pair_load(const real_t* p) {
 return libero_vec::load_pair(p);
}
__device__ __forceinline__ void pair_store(real_t* p,real_t x,real_t y) {
 libero_vec::store_pair(p,x,y);
}
__device__ __forceinline__ void copy(const real_t* a,real_t* b,int n) {
 libero_vec::copy_block(a,b,n);
}
// Combined exact-coefficient Horner is defined below the core implementation.


template<int Kind>
__device__ bool joint(int j,const int* jt,const int* padr,const int* vadr,
                     real_t* pos,const real_t* vel,real_t dt) {
 const int kind=Kind<0?jt[j]:Kind;
 int pa=padr[j],va=vadr[j];
 if(kind==0) {
  const real2_t p=pair_load(pos+pa),v=pair_load(vel+va);
  pair_store(pos+pa,p.x+dt*v.x,p.y+dt*v.y);
  pos[pa+2]+=dt*vel[va+2];pa+=3;va+=3;
 }else if(kind==2||kind==3)pos[pa]+=dt*vel[va];
 if(kind!=0&&kind!=1)return true;
 const real2_t v=pair_load(vel+va);
 const real_t rawx=v.x,rawy=v.y,rawz=vel[va+2];
 real_t x=rawx,y=rawy,z=rawz;
 const real_t xx=x*x,yy=y*y,zz=z*z,sxy=xx+yy,sum=sxy+zz;
 if(!isfinite(xx)||!isfinite(yy)||!isfinite(zz)||!isfinite(sxy)||!isfinite(sum))return false;
 const real_t length=sqrt(sum);
 if(!isfinite(length))return false;
 const bool axis_fallback=length<REAL_C(1e-15);
 const real_t a=dt*length;
 if(!isfinite(a))return false;
 const real_t h=REAL_C(.5)*a;
 if(!isfinite(h))return false;
 real_t b[4];
 {
  if(axis_fallback){x=1;y=z=0;}else{const real_t inverse=REAL_C(1.0)/length;x*=inverse;y*=inverse;z*=inverse;}
  if(!isfinite(x)||!isfinite(y)||!isfinite(z))return false;
  real_t s=0,c=1;
  if(a!=0) {
   s=fabs(h)<=4?cf_euler::trig(h,true):sin(h);
   c=fabs(h)<=4?cf_euler::trig(h,false):cos(h);
  }
  b[0]=c;b[1]=x*s;b[2]=y*s;b[3]=z*s;
  if(a==0)b[1]=b[2]=b[3]=0;
 }
 for(int k=0;k<4;k++)if(!isfinite(b[k]))return false;
 const real2_t p01=pair_load(pos+pa),p23=pair_load(pos+pa+2);
 const real_t a0=p01.x,a1=p01.y,a2=p23.x,a3=p23.y;
 real_t t[4]={a0*b[0]-a1*b[1]-a2*b[2]-a3*b[3],
              a0*b[1]+a1*b[0]+a2*b[3]-a3*b[2],
              a0*b[2]-a1*b[3]+a2*b[0]+a3*b[1],
              a0*b[3]+a1*b[2]-a2*b[1]+a3*b[0]};
 for(int k=0;k<4;k++)if(!isfinite(t[k]))return false;
 real_t sq[4];for(int k=0;k<4;k++){sq[k]=t[k]*t[k];if(!isfinite(sq[k]))return false;}
 const real_t s01=sq[0]+sq[1],s012=s01+sq[2],ss=s012+sq[3];
 if(!isfinite(s01)||!isfinite(s012)||!isfinite(ss))return false;
 const real_t qlen=sqrt(ss);
 if(!isfinite(qlen))return false;
 if(qlen<REAL_C(1e-15)){t[0]=1;t[1]=t[2]=t[3]=0;}
 else{const real_t inverse=REAL_C(1.0)/qlen;for(int k=0;k<4;k++)t[k]*=inverse;}
 for(int k=0;k<4;k++)if(!isfinite(t[k]))return false;
 pair_store(pos+pa,t[0],t[1]);pair_store(pos+pa+2,t[2],t[3]);
 return true;
}
template<int Kind>
__device__ void bucket(int begin,int end,const int* ids,const int* jt,const int* pa,
                      const int* va,real_t* pos,const real_t* vel,real_t dt,bool& bad) {
 for(int k=begin+threadIdx.x;k<end;k+=blockDim.x)
  bad|=!joint<Kind>(ids[k],jt,pa,va,pos,vel,dt);
}

// E01 retains original state finite checks; only false is instantiated.
template<bool StateCertified>
__device__ void prepare(int w,int active,const real_t* qp,const real_t* qv,
 const real_t* time,const real_t* qm,const real_t* smooth,const real_t* constraint,
 const real_t* acc,const real_t* damping,const int* madr,real_t dt,int nq,int nv,
 int nm,Work& s) {
 if(threadIdx.x==0){s.status=0;s.mass_status=0;s.damped=0;s.mass_active=0;s.warnings=0;s.last=-1;}
 libero_vec::fill_block(s.qm,nm,REAL_C(0.0));
 libero_vec::fill_block(s.rhs,4*nv,REAL_C(0.0));
 libero_vec::fill_block(s.ld,nm,REAL_C(0.0));libero_vec::fill_block(s.inv,nv,REAL_C(0.0));
 libero_vec::fill_block(s.si,nv,REAL_C(0.0));libero_vec::fill_block(s.solution,4*nv,REAL_C(0.0));
 libero_vec::fill_block(s.used,nv,REAL_C(0.0));libero_vec::fill_block(s.pivot,nv,0);
 __syncthreads();
 if(active!=1){if(threadIdx.x==0&&active!=0)s.status=1;__syncthreads();return;}
 if(threadIdx.x<32) {
  const int lane=threadIdx.x;
  bool bad=!isfinite(time[w]);
  if(!StateCertified)for(int i=lane;i<nq;i+=32)bad|=!isfinite(qp[w*nq+i]);
  bad=libero_warp::any(bad);
  if(bad){if(lane==0)s.status=2;}
  else {
   int first=libero_warp::no_index;
   for(int i=lane;i<nv;i+=32) {
    bool invalid=!isfinite(smooth[w*nv+i])||!isfinite(constraint[w*nv+i]);
    if(!StateCertified)invalid|=!isfinite(qv[w*nv+i])||!isfinite(acc[w*nv+i]);
    if(invalid)first=min(first,i);
   }
   first=libero_warp::min(first);
   bool positive=false,shadow_ok=true;
   for(int i=lane;i<nv;i+=32) {
    if(!0 || first!=libero_warp::no_index)
     if(i<first)positive|=damping[i]>0;
   }
   positive=libero_warp::any(positive);
   shadow_ok=!libero_warp::any(!shadow_ok);
   if(lane==0){s.damped=positive;if(first!=libero_warp::no_index)s.status=2;}
   if(first==libero_warp::no_index&&positive) {
    int first_bad=libero_warp::no_index;
    for(int i=lane;i<nm;i+=32)if(!isfinite(qm[w*nm+i]))first_bad=min(first_bad,i);
    first_bad=libero_warp::min(first_bad);
    for(int i=lane;i<nm&&i<first_bad;i+=32)s.qm[i]=qm[w*nm+i];
    __syncwarp(libero_warp::mask);
    if(first_bad!=libero_warp::no_index){if(lane==0)s.status=2;}
    else {
     if(lane==0) {
      for(int i=0;i<nv;i++) {
       s.qm[madr[i]]+=dt*damping[i];
      }
      s.mass_active=1;
     }
     for(int i=lane;i<nv;i+=32)s.rhs[i]=smooth[w*nv+i]+constraint[w*nv+i];
    }
   }
  }
 }
 __syncthreads();
}

__device__ bool mass(int w,Work& s,const int* madr,const int* parent,int nv,int nm) {
 if(!s.mass_active)return true;
 bool invalid=false;
 for(int i=threadIdx.x;i<nm;i+=blockDim.x)invalid|=!isfinite(s.qm[i]);
 for(int i=threadIdx.x;i<4*nv;i+=blockDim.x)invalid|=!isfinite(s.rhs[i]);
 if(__syncthreads_or(invalid)){if(threadIdx.x==0)s.mass_status=3;__syncthreads();return false;}
 libero_packed_mass::ForestPlan forest;
 bool factor_ok=true;
 factor_ok=libero_packed_mass::factor_block(s.qm,s.ld,s.inv,s.si,s.pivot,s.warnings,s.last,nv,nm,madr,parent,forest);
 const bool solve_ok=libero_packed_mass::solve_rhs_block(s.rhs,s.solution,s.ld,s.inv,1,nv,madr,parent,forest);
 if(threadIdx.x==0){
  if(!factor_ok||!solve_ok)s.mass_status=4;
  else if(s.warnings)s.mass_status=5;
 }
 __syncthreads();
 return factor_ok&&solve_ok;
}

template<bool StateCertified>
__device__ void evaluate(int w,const int* active,real_t* qp,real_t* qv,real_t* time,
 const real_t* qm,const real_t* smooth,const real_t* constraint,const real_t* acc,
 const real_t* damping,const int* madr,int* count,int* ma,int* damp,int* status,
 real_t dt,int nq,int nv,int nm,const int* parent,int* ms,int* wc,int* wl,
 const int* jt,const int* padr,const int* vadr,real_t* used,int nj,real_t* qh,real_t* qhi,
 const int* joint_quat,const int* offsets,const int* ids) {
 __shared__ Work s;
 __shared__ int vote[32];
 prepare<StateCertified>(w,active[w],qp,qv,time,qm,smooth,constraint,acc,damping,madr,dt,nq,nv,nm,s);
 mass(w,s,madr,parent,nv,nm);
 real_t* vel=s.vel;
 if(active[w]==1&&s.status==0) {
  if(s.damped&&s.mass_status!=0&&s.mass_status!=5){if(threadIdx.x==0)s.status=10+s.mass_status;}
  else {
   bool bad=false;
   for(int i=2*threadIdx.x;i<nv;i+=2*blockDim.x) {
    if(i+1<nv) {
     const real2_t a=pair_load(s.damped?s.solution+i:acc+w*nv+i),v=pair_load(qv+w*nv+i);
     const real_t x=v.x+dt*a.x,y=v.y+dt*a.y;
     pair_store(s.used+i,a.x,a.y);pair_store(vel+i,x,y);bad|=!isfinite(x)||!isfinite(y);
    }else{const real_t a=s.damped?s.solution[i]:acc[w*nv+i];s.used[i]=a;vel[i]=qv[w*nv+i]+dt*a;bad|=!isfinite(vel[i]);}
   }
   if(cf_euler::integration_bad(bad,vote)){if(threadIdx.x==0)s.status=3;}
   else {
    copy(qp+w*nq,s.pos,nq);__syncthreads();bad=false;
    if(CF_EULER_LEAN_JOINT_PARALLEL&&joint_quat&&offsets&&ids) {
     for(int kind=0;kind<4;kind++)bucket<-1>(offsets[kind],offsets[kind+1],ids,jt,padr,vadr,s.pos,vel,dt,bad);
    }else if(threadIdx.x==0)for(int j=0;j<nj;j++){if(!joint<-1>(j,jt,padr,vadr,s.pos,vel,dt)){bad=true;break;}}
    if(cf_euler::integration_bad(bad,vote)){if(threadIdx.x==0)s.status=3;}
    else {
     if(threadIdx.x==0){s.next_time=time[w]+dt;if(!isfinite(s.next_time))s.status=3;}
     __syncthreads();bad=false;
     for(int i=threadIdx.x;i<nq;i+=blockDim.x)bad|=!isfinite(s.pos[i]);
     if(cf_euler::integration_bad(bad,vote)){if(threadIdx.x==0)s.status=3;}
    }
   }
  }
 }
 __syncthreads();
 const bool failed=active[w]!=1||s.status!=0;
 if(failed) {
  libero_vec::fill_block(s.used,nv,REAL_C(0.0));
  if(threadIdx.x==0){s.warnings=0;s.last=-1;}
 }else copy(s.pos,qp+w*nq,nq);
 __syncthreads();
 if(!failed)copy(vel,qv+w*nv,nv);
 __syncthreads();
 if(!failed&&threadIdx.x==0)time[w]=s.next_time;
 __syncthreads();
 if(!failed&&s.damped)copy(s.ld,qh+w*nm,nm);
 __syncthreads();
 if(!failed&&s.damped)copy(s.inv,qhi+w*nv,nv);
 __syncthreads();
 copy(s.used,used+w*nv,nv);
 if(threadIdx.x==0){count[w]=1;ma[w]=s.mass_active;damp[w]=s.damped;status[w]=s.status;ms[w]=s.mass_status;wc[w]=s.warnings;wl[w]=s.last;}
 __syncthreads();
}

__device__ void evaluate_plain(int w,const int* active,real_t* qp,real_t* qv,real_t* time,
 const real_t* qm,const real_t* smooth,const real_t* constraint,const real_t* acc,
 const real_t* damping,const int* madr,int* count,int* ma,int* damp,int* status,
 real_t dt,int nq,int nv,int nm,const int* parent,int* ms,int* wc,int* wl,
 const int* jt,const int* padr,const int* vadr,real_t* used,int nj,real_t* qh,real_t* qhi,
 const int* joint_quat,const int* offsets,const int* ids) {
 evaluate<false>(w,active,qp,qv,time,qm,smooth,constraint,acc,damping,madr,count,ma,damp,status,dt,nq,nv,nm,parent,ms,wc,wl,jt,padr,vadr,used,nj,qh,qhi,joint_quat,offsets,ids);
}
} // namespace cf_euler_lean
#endif
