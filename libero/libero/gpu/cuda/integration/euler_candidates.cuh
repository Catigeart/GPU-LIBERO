#include "precision.cuh"
#pragma once
#include "euler.cuh"

// Retained measured-positive operations; pruned source qualification pending. The existing compatible/default
// implementation is unchanged. No consumer may request diagnostic exports
// from this ABI; status/warnings/qacc_used and public held fields are retained.
#if defined(CF_EULER_NV)
#ifndef CF_EC_RHS_SLOTS
#define CF_EC_RHS_SLOTS 4
#endif
#ifndef CF_EC_REUSE_VEL
#define CF_EC_REUSE_VEL 0
#endif
#ifndef CF_EC_SMALL_ANGLE
#define CF_EC_SMALL_ANGLE 0
#endif
#ifndef CF_EC_SCALAR_IO
#define CF_EC_SCALAR_IO 0
#endif
#ifndef CF_EC_CACHE
#define CF_EC_CACHE 0
#endif
#ifndef CF_EC_JOINT_PARALLEL
#define CF_EC_JOINT_PARALLEL 1
#endif

namespace cf_euler_candidate {
static_assert(CF_EC_RHS_SLOTS==1 || CF_EC_RHS_SLOTS==4,"Euler RHS slots");
struct Work {
 real_t qm[CF_EULER_NM],ld[CF_EULER_NM];
 real_t rhs[CF_EC_RHS_SLOTS*CF_EULER_NV];
 real_t solution[CF_EC_RHS_SLOTS*CF_EULER_NV];
 real_t inv[CF_EULER_NV],si[CF_EULER_NV];
 real_t pos[CF_EULER_NQ],vel[CF_EC_REUSE_VEL?1:CF_EULER_NV],used[CF_EULER_NV];
 int pivot[CF_EULER_NV];
 int status,mass_status,damped,mass_active,warnings,last;
 real_t next_time;
};
__device__ __forceinline__ real2_t pair_load(const real_t* p) {
#if CF_EC_SCALAR_IO
 return make_real2(p[0],p[1]);
#else
 return libero_vec::load_pair(p);
#endif
}
__device__ __forceinline__ void pair_store(real_t* p,real_t x,real_t y) {
#if CF_EC_SCALAR_IO
 p[0]=x;p[1]=y;
#else
 libero_vec::store_pair(p,x,y);
#endif
}
__device__ __forceinline__ void copy(const real_t* a,real_t* b,int n) {
#if CF_EC_SCALAR_IO
 for(int i=threadIdx.x;i<n;i+=blockDim.x)b[i]=a[i];
#else
 libero_vec::copy_block(a,b,n);
#endif
}



__device__ bool joint(int j,const int* jt,const int* padr,const int* vadr,
                     real_t* pos,const real_t* vel,real_t dt) {
 const int kind=jt[j];
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
#if CF_EC_SMALL_ANGLE
 // Keep the original increment near normalize4's 1e-15 fallback and
 // overflow boundaries.  In this domain ||q||2 >= 1e-10 and <= 2e100;
 // the small-angle unit increment cannot approach either boundary under
 // the local binary64 error bound.  The original norm graph remains below.
 const real_t qmax=fmax(fmax(fabs(pos[pa]),fabs(pos[pa+1])),
                        fmax(fabs(pos[pa+2]),fabs(pos[pa+3])));
 // h==0, dt/2 subnormal, and axis-fallback retain the original evaluation.
 // |h|<=2^-8: omitted sinc term <=2^-64/9!, cos term <=2^-64/8!.
 if(qmax>=REAL_C(1e-10) && qmax<=REAL_SAFE_MAGNITUDE && !axis_fallback && a!=0 && h!=0 && fabs(h)<=REAL_C(0.00390625) &&
    fabs(dt)>=REAL_C(4.450147717014403e-308)) {
  const real_t u=h*h;
  const real_t sinc=(((-REAL_C(1.0)/REAL_C(5040.0)*u+REAL_C(1.0)/REAL_C(120.0))*u-REAL_C(1.0)/REAL_C(6.0))*u+REAL_C(1.0));
  const real_t cosine=(((-REAL_C(1.0)/REAL_C(720.0)*u+REAL_C(1.0)/REAL_C(24.0))*u-REAL_C(.5))*u+REAL_C(1.0));
  const real_t scale=REAL_C(.5)*dt;
  b[0]=cosine;b[1]=(rawx*scale)*sinc;
  b[2]=(rawy*scale)*sinc;b[3]=(rawz*scale)*sinc;
 }else
#endif
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
__device__ void bucket(int begin,int end,const int* ids,const int* jt,const int* pa,
                      const int* va,real_t* pos,const real_t* vel,real_t dt,bool& bad) {
 for(int k=begin+threadIdx.x;k<end;k+=blockDim.x)
  bad|=!joint(ids[k],jt,pa,va,pos,vel,dt);
}

// Original finite scans are retained; the measured-negative certified gate is removed.
__device__ void prepare(int w,int active,const real_t* qp,const real_t* qv,
 const real_t* time,const real_t* qm,const real_t* smooth,const real_t* constraint,
 const real_t* acc,const real_t* damping,const int* madr,real_t dt,int nq,int nv,
 int nm,Work& s) {
 if(threadIdx.x==0){s.status=0;s.mass_status=0;s.damped=0;s.mass_active=0;s.warnings=0;s.last=-1;}
 libero_vec::fill_block(s.qm,nm,REAL_C(0.0));
 libero_vec::fill_block(s.rhs,CF_EC_RHS_SLOTS*nv,REAL_C(0.0));
 libero_vec::fill_block(s.ld,nm,REAL_C(0.0));libero_vec::fill_block(s.inv,nv,REAL_C(0.0));
 libero_vec::fill_block(s.si,nv,REAL_C(0.0));libero_vec::fill_block(s.solution,CF_EC_RHS_SLOTS*nv,REAL_C(0.0));
 libero_vec::fill_block(s.used,nv,REAL_C(0.0));libero_vec::fill_block(s.pivot,nv,0);
 __syncthreads();
 if(active!=1){if(threadIdx.x==0&&active!=0)s.status=1;__syncthreads();return;}
 if(threadIdx.x<32) {
  const int lane=threadIdx.x;
  bool bad=!isfinite(time[w]);
  for(int i=lane;i<nq;i+=32)bad|=!isfinite(qp[w*nq+i]);
  bad=libero_warp::any(bad);
  if(bad){if(lane==0)s.status=2;}
  else {
   int first=libero_warp::no_index;
   for(int i=lane;i<nv;i+=32) {
    bool invalid=!isfinite(smooth[w*nv+i])||!isfinite(constraint[w*nv+i]);
    invalid|=!isfinite(qv[w*nv+i])||!isfinite(acc[w*nv+i]);
    if(invalid)first=min(first,i);
   }
   first=libero_warp::min(first);
   bool positive=false;
   for(int i=lane;i<nv;i+=32) {
     if(i<first)positive|=damping[i]>0;
   }
   positive=libero_warp::any(positive);
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

// A cache owns complete constructed MhB bits and topology keys per world.
// Keys include signed zero. Only successful factors are published, and warning
// events are replayed on a hit. Caller privately owns/invalidate all cache arrays.
__device__ bool mass(int w,Work& s,const int* madr,const int* parent,int nv,int nm,
 real_t dt,real_t* key,real_t* cached_ld,real_t* cached_inv,real_t* cached_si,
 int* cached_pivot,int* cached_parent,int* cached_madr,int64_t* cached_dt,
 int* cache_valid,int* cached_warnings,int* cached_last,int* cache_hit) {
 if(threadIdx.x==0)cache_hit[w]=0;
 __syncthreads();
 if(!s.mass_active)return true;
 bool invalid=false;
 for(int i=threadIdx.x;i<nm;i+=blockDim.x)invalid|=!isfinite(s.qm[i]);
 for(int i=threadIdx.x;i<CF_EC_RHS_SLOTS*nv;i+=blockDim.x)invalid|=!isfinite(s.rhs[i]);
 if(__syncthreads_or(invalid)){if(threadIdx.x==0)s.mass_status=3;__syncthreads();return false;}
 bool hit=false;
#if CF_EC_CACHE
 bool mismatch=cache_valid[w]!=1||cached_dt[w]!=real_bits(dt);
 for(int i=threadIdx.x;i<nm;i+=blockDim.x)mismatch|=real_bits(key[w*nm+i])!=real_bits(s.qm[i]);
 for(int i=threadIdx.x;i<nv;i+=blockDim.x)mismatch|=cached_parent[w*nv+i]!=parent[i]||cached_madr[w*nv+i]!=madr[i];
 hit=__syncthreads_or(mismatch)==0;
#endif
 libero_packed_mass::ForestPlan forest;
 bool factor_ok=true;
 if(hit) {
  // inspect still supplies the exact solve mapping; F01 prepared plan is optional.
  forest=libero_packed_mass::inspect_forest_block(nv,parent);
  copy(cached_ld+w*nm,s.ld,nm);copy(cached_inv+w*nv,s.inv,nv);copy(cached_si+w*nv,s.si,nv);
  for(int i=threadIdx.x;i<nv;i+=blockDim.x)s.pivot[i]=cached_pivot[w*nv+i];
  if(threadIdx.x==0){s.warnings=cached_warnings[w];s.last=cached_last[w];cache_hit[w]=1;}
  __syncthreads();
 }else {
  factor_ok=libero_packed_mass::factor_block(s.qm,s.ld,s.inv,s.si,s.pivot,s.warnings,s.last,nv,nm,madr,parent,forest);
 }
 const bool solve_ok=libero_packed_mass::solve_rhs_block(s.rhs,s.solution,s.ld,s.inv,1,nv,madr,parent,forest);
 if(threadIdx.x==0){
  if(!factor_ok||!solve_ok)s.mass_status=4;
  else if(s.warnings)s.mass_status=5;
 }
 __syncthreads();
#if CF_EC_CACHE
 if(!factor_ok||!solve_ok){if(threadIdx.x==0)cache_valid[w]=0;}
 else if(!hit) {
  copy(s.qm,key+w*nm,nm);copy(s.ld,cached_ld+w*nm,nm);
  copy(s.inv,cached_inv+w*nv,nv);copy(s.si,cached_si+w*nv,nv);
  for(int i=threadIdx.x;i<nv;i+=blockDim.x){cached_pivot[w*nv+i]=s.pivot[i];cached_parent[w*nv+i]=parent[i];cached_madr[w*nv+i]=madr[i];}
  __syncthreads();
  if(threadIdx.x==0){cached_dt[w]=real_bits(dt);cached_warnings[w]=s.warnings;cached_last[w]=s.last;cache_valid[w]=1;}
 }
 __syncthreads();
#endif
 return factor_ok&&solve_ok;
}

__device__ void evaluate(int w,const int* active,real_t* qp,real_t* qv,real_t* time,
 const real_t* qm,const real_t* smooth,const real_t* constraint,const real_t* acc,
 const real_t* damping,const int* madr,int* count,int* ma,int* damp,int* status,
 real_t dt,int nq,int nv,int nm,const int* parent,int* ms,int* wc,int* wl,
 const int* jt,const int* padr,const int* vadr,real_t* used,int nj,real_t* qh,real_t* qhi,
 const int* joint_quat,const int* offsets,const int* ids,real_t* key,real_t* cached_ld,
 real_t* cached_inv,real_t* cached_si,int* cached_pivot,int* cached_parent,int* cached_madr,
 int64_t* cached_dt,int* cache_valid,int* cached_warnings,int* cached_last,int* cache_hit) {
 __shared__ Work s;
 __shared__ int vote[32];
 prepare(w,active[w],qp,qv,time,qm,smooth,constraint,acc,damping,madr,dt,nq,nv,nm,s);
 mass(w,s,madr,parent,nv,nm,dt,key,cached_ld,cached_inv,cached_si,cached_pivot,cached_parent,cached_madr,cached_dt,cache_valid,cached_warnings,cached_last,cache_hit);
 real_t* vel=CF_EC_REUSE_VEL?s.solution:s.vel;
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
    if(CF_EC_JOINT_PARALLEL&&joint_quat&&offsets&&ids) {
     for(int kind=0;kind<4;kind++)bucket(offsets[kind],offsets[kind+1],ids,jt,padr,vadr,s.pos,vel,dt,bad);
    }else if(threadIdx.x==0)for(int j=0;j<nj;j++){if(!joint(j,jt,padr,vadr,s.pos,vel,dt)){bad=true;break;}}
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

// BEGIN GENERATED ABI WRAPPERS (no arithmetic translation)
__device__ void evaluate_plain(int w,const int* active,real_t* qp,real_t* qv,real_t* time,
 const real_t* qm,const real_t* smooth,const real_t* constraint,const real_t* acc,
 const real_t* damping,const int* madr,int* count,int* ma,int* damp,int* status,
 real_t dt,int nq,int nv,int nm,const int* parent,int* ms,int* wc,int* wl,
 const int* jt,const int* padr,const int* vadr,real_t* used,int nj,real_t* qh,real_t* qhi,
 const int* joint_quat,const int* offsets,const int* ids,real_t* key,real_t* cached_ld,
 real_t* cached_inv,real_t* cached_si,int* cached_pivot,int* cached_parent,int* cached_madr,
 int64_t* cached_dt,int* cache_valid,int* cached_warnings,int* cached_last,int* cache_hit) {
 evaluate(w,active,qp,qv,time,qm,smooth,constraint,acc,damping,madr,count,ma,damp,status,dt,nq,nv,nm,parent,ms,wc,wl,jt,padr,vadr,used,nj,qh,qhi,joint_quat,offsets,ids,key,cached_ld,cached_inv,cached_si,cached_pivot,cached_parent,cached_madr,cached_dt,cache_valid,cached_warnings,cached_last,cache_hit);
}
} // namespace cf_euler_candidate
#endif
