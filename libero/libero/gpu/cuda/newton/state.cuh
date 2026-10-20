#include "precision.cuh"
#include "io_policy.cuh"
#pragma once
#if LIBERO_NEWTON_SCALAR_IO
#else
// Pair only independent element updates. Shifted aliases retain scalar lane
// ownership; checks and alpha==0 remain before every write.
__device__ inline bool n_state_disjoint(const void* a,size_t an,const void* b,size_t bn){
  const uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
  return !an||!bn||(x<=y?y-x>=an:x-y>=bn);
}
__device__ inline void n_state_axpy(real_t* dst,const real_t* src,int n,real_t a){
  if(dst==src||n_state_disjoint(dst,size_t(n)*8,src,size_t(n)*8)){
    int i=2*int(threadIdx.x);
    for(;i+1<n;i+=2*int(blockDim.x)){
      real2_t x=libero_vec::load_pair(dst+i),y=libero_vec::load_pair(src+i);
      libero_vec::store_pair(dst+i,x.x+a*y.x,x.y+a*y.y);
    }
    if(i<n)dst[i]+=a*src[i];
  }else for(int i=threadIdx.x;i<n;i+=blockDim.x)dst[i]+=a*src[i];
}
#endif // LIBERO_NEWTON_SCALAR_IO
__device__ void warm_select(int w,int* active,int* nefc,real_t* wc,real_t* sc,real_t* warm,real_t* smooth,
 real_t* sf,int* ss,real_t* sq,real_t* qacc,real_t* force,int* state,real_t* qfrc,int* choice,int* status,int nv,int rows){
  if(status[w]||!active[w])return;if(active[w]!=1){status[w]=1;return;}int n=nefc[w],v=w*nv,r=w*rows;
  if(n<=0||n>rows){status[w]=2;return;}if(!isfinite(wc[w])||!isfinite(sc[w])){status[w]=3;return;}
  for(int i=0;i<nv;++i)if(!isfinite(warm[v+i])||!isfinite(smooth[v+i])||!isfinite(sq[v+i])){status[w]=3;return;}
  for(int i=0;i<n;++i)if(!isfinite(sf[r+i])){status[w]=3;return;}
  int select=wc[w]>sc[w];choice[w]=select;
  for(int i=0;i<nv;++i){qacc[v+i]=select?smooth[v+i]:warm[v+i];qfrc[v+i]=sq[v+i];}
  for(int i=0;i<n;++i){force[r+i]=sf[r+i];state[r+i]=ss[r+i];}
}
__device__ void commit_step(int w,int* active,int* nefc,real_t* alpha,real_t* search,real_t* mv,real_t* jv,
 real_t* qacc,real_t* ma,real_t* jar,int* committed,int* status,int nv,int rows){
  if(status[w]||!active[w])return;if(active[w]!=1){status[w]=1;return;}int n=nefc[w],v=w*nv,r=w*rows;
  if(n<=0||n>rows){status[w]=2;return;}real_t a=alpha[w];if(!isfinite(a)){status[w]=3;return;}
  if(a==0){committed[w]=0;return;}
  for(int i=0;i<nv;++i)if(!isfinite(search[v+i])||!isfinite(mv[v+i])||!isfinite(qacc[v+i])||!isfinite(ma[v+i])||
    !isfinite(qacc[v+i]+a*search[v+i])||!isfinite(ma[v+i]+a*mv[v+i])){status[w]=3;return;}
  for(int i=0;i<n;++i)if(!isfinite(jv[r+i])||!isfinite(jar[r+i])||!isfinite(jar[r+i]+a*jv[r+i])){status[w]=3;return;}
  committed[w]=1;for(int i=0;i<nv;++i)qacc[v+i]+=a*search[v+i];
  for(int i=0;i<nv;++i)ma[v+i]+=a*mv[v+i];for(int i=0;i<n;++i)jar[r+i]+=a*jv[r+i];
}
__device__ void stats_stop(int w,int* active,int* nefc,int* stage,int* committed,int* oldstate,int* state,real_t* grad,
 real_t* oldcost,real_t* cost,real_t* inertia,real_t* tolerance,real_t* slope,int* nactive,int* nupdate,int* neval,
 int* iteration,int* base,int* maxiter,real_t* sf,int* si,int* nchange,real_t* improvement,real_t* gradient,
 int* stop,int* reason,int* out_iter,int* status,int nv,int rows,int* precomputed_change=nullptr,real_t* precomputed_norm=nullptr,bool gradient_scanned=false,const int* residual_roundoff_ok=nullptr){
  if(status[w]||!active[w])return;if(active[w]!=1){status[w]=1;return;}int n=nefc[w],it=iteration[w],b=base[w];
  if(n<=0||n>rows||it<0||b<0||b>2147483647-it||it>=maxiter[w]||maxiter[w]<0||maxiter[w]>1000000){status[w]=2;return;}
  if(stage[w]!=1||committed[w]!=1){status[w]=4;return;}
  if(!isfinite(oldcost[w])||!isfinite(cost[w])||!isfinite(inertia[w])||!isfinite(tolerance[w])||!isfinite(slope[w])||inertia[w]<=0||tolerance[w]<0){status[w]=3;return;}
  if(nactive[w]<0||nupdate[w]<0||neval[w]<0){status[w]=2;return;}
  int change=precomputed_change?precomputed_change[0]:0;
  if(!precomputed_change)for(int j=0;j<n;++j)if(oldstate[w*rows+j]!=state[w*rows+j])++change;
// Internal norm production visits every gradient element. A finite sum of
  // squared components implies finite operands; avoid a second shared/global
  // load pass only for that call path. Scalar callers retain their guard.
  if(!gradient_scanned)for(int j=0;j<nv;++j)if(!isfinite(grad[w*nv+j])){status[w]=3;return;}
  real_t d;
  if(precomputed_norm)d=precomputed_norm[0];
  else{
  // Stats starts each lane at +0 (unlike mju_dot's lane priming).
  real_t s0=0,s1=0,s2=0,s3=0;int j=0;real_t* g=grad+w*nv;
  for(;j<=nv-4;j+=4){s0+=g[j]*g[j];s1+=g[j+1]*g[j+1];s2+=g[j+2]*g[j+2];s3+=g[j+3]*g[j+3];}
  d=(s0+s2)+(s1+s3);
  if(nv-j==3)d+=g[j]*g[j]+g[j+1]*g[j+1]+g[j+2]*g[j+2];else if(nv-j==2)d+=g[j]*g[j]+g[j+1]*g[j+1];else if(nv-j==1)d+=g[j]*g[j];
  }
  if(!isfinite(d)){status[w]=3;return;}real_t scale=inertia[w]*real_t(nv),imp=(oldcost[w]-cost[w])/scale,gn=sqrt(d)/scale;
  if(!isfinite(imp)||!isfinite(gn)){status[w]=3;return;}int slot=b+it;if(slot<0){status[w]=2;return;}
  if(slot<1000){int f=(w*1000+slot)*3,i=(w*1000+slot)*4;
    sf[f]=imp;sf[f+1]=gn;sf[f+2]=slope[w];si[i]=nactive[w];si[i+1]=change;si[i+2]=neval[w];si[i+3]=nupdate[w];}
  // Total-cost quantization is stagnation, not evidence of convergence.
  // Keep the original improvement criterion when its decrease is resolved.
  // In its rounding envelope require a componentwise backward residual check.
  const real_t magnitude=fmax(fabs(oldcost[w]),fabs(cost[w]));
  const real_t resolution=REAL_C(8.0)*real_t(REAL_EPSILON)*magnitude;
  const real_t delta=oldcost[w]-cost[w];
  const bool ambiguous=fabs(delta)<=resolution;
  const bool resolved_small=delta>=REAL_C(0.0)&&!ambiguous&&imp<tolerance[w];
  const bool roundoff_small=ambiguous&&residual_roundoff_ok&&residual_roundoff_ok[0];
  int next=it+1,st=0,re=0;if(resolved_small||roundoff_small||gn<tolerance[w]){st=1;re=1;}else if(next>=maxiter[w]){st=1;re=2;}
  nchange[w]=change;improvement[w]=imp;gradient[w]=gn;stop[w]=st;reason[w]=re;out_iter[w]=next;
}

// The actual solver calls this with all 32/64/128 block threads. Validation and
// stopping remain in stats_stop, so legal stop tests and write priority stay put.
__device__ void stats_stop_block(int w,int* active,int* nefc,int* stage,int* committed,int* oldstate,int* state,real_t* grad,
 real_t* oldcost,real_t* cost,real_t* inertia,real_t* tolerance,real_t* slope,int* nactive,int* nupdate,int* neval,
 int* iteration,int* base,int* maxiter,real_t* sf,int* si,int* nchange,real_t* improvement,real_t* gradient,
 int* stop,int* reason,int* out_iter,int* status,int nv,int rows,int* precomputed_change=nullptr, const real_t* residual_Ma=nullptr, const real_t* residual_smooth=nullptr, const real_t* residual_J=nullptr, const real_t* residual_force=nullptr){
  __shared__ real_t norm_warp[4],norm;
  __shared__ int changes_warp[4],changes;
  __shared__ int residual_bad_warp[4],residual_ok;
  if(status[w]||!active[w])return;
  const int lane=libero_warp::lane(),wid=libero_warp::warp();
#ifdef NEWTON_BLOCK_THREADS
  constexpr int warps=NEWTON_BLOCK_THREADS/32;
#else
  const int warps=blockDim.x/32;
#endif
  // Counts may be malformed: don't dereference any rows until bounds are safe.
  const int n=nefc[w];real_t value=REAL_C(0.);int change=0;
  // A standard gamma_k bound covers the product/accumulation in J^T f,
  // the packed Mq product and the two residual subtractions. This is a
  // componentwise backward-error contract, not a claim of small forward error.
  const real_t mag=fmax(fabs(oldcost[w]),fabs(cost[w]));
  const bool ambiguous=fabs(oldcost[w]-cost[w])<=REAL_C(8.0)*real_t(REAL_EPSILON)*mag;
  int bad=0;
  if(ambiguous&&residual_Ma&&residual_smooth&&residual_J&&residual_force&&n>0&&n<=rows){
    const real_t eps=REAL_C(1.1102230246251565e-16);
    const real_t k=real_t(n+2*nv+8),gamma=(k*eps)/(REAL_C(1.0)-k*eps);
    for(int j=threadIdx.x;j<nv;j+=blockDim.x){
      real_t sum=fabs(residual_Ma[w*nv+j])+fabs(residual_smooth[w*nv+j]);
      for(int r=0;r<n;++r)sum+=fabs(residual_J[(w*rows+r)*nv+j]*residual_force[w*rows+r]);
      // Bound the positive summation's own rounding before using its scale.
      const real_t limit=tolerance[w]*inertia[w]*real_t(nv)+gamma*sum/(REAL_C(1.0)-gamma);
      if(!isfinite(sum)||!isfinite(limit)||!isfinite(grad[w*nv+j])||fabs(grad[w*nv+j])>limit)bad=1;
    }
  }else bad=1;
  bad=int(libero_warp::any(bad!=0));
  if(lane==0)residual_bad_warp[wid]=bad;
  __syncthreads();
  if(wid==0){bad=int(libero_warp::any(lane<warps&&residual_bad_warp[lane]!=0));if(lane==0)residual_ok=!bad;}
  __syncthreads();

  // The fused caller already has a block barrier after this stage. Keep small
  // sums in registers, with no shared partials or additional block barriers.
  // NV is the compiled capacity; using it also removes this entire candidate
  // path from larger kernels, where paired timings showed no improvement.
  if(NV<=32&&nv<=32&&(precomputed_change||n<=64)){
    if(wid!=0)return;
    if(active[w]==1&&n>0&&n<=rows){
      for(int j=lane;j<nv;j+=32){real_t g=grad[w*nv+j];value+=g*g;}
      if(!precomputed_change)for(int j=lane;j<n;j+=32)
        change+=oldstate[w*rows+j]!=state[w*rows+j];
    }
    value=libero_warp::sum(value);
    if(precomputed_change)change=precomputed_change[0];
    else change=libero_warp::sum(change);
    if(lane==0)stats_stop(w,active,nefc,stage,committed,oldstate,state,grad,
      oldcost,cost,inertia,tolerance,slope,nactive,nupdate,neval,iteration,base,maxiter,
      sf,si,nchange,improvement,gradient,stop,reason,out_iter,status,nv,rows,&change,&value,true,&residual_ok);
    return;
  }
  if(active[w]==1&&n>0&&n<=rows){
    for(int j=threadIdx.x;j<nv;j+=blockDim.x){real_t g=grad[w*nv+j];value+=g*g;}
    if(!precomputed_change)for(int j=threadIdx.x;j<n;j+=blockDim.x)
      change+=oldstate[w*rows+j]!=state[w*rows+j];
  }
  value=libero_warp::sum(value);change=libero_warp::sum(change);
  if(lane==0){norm_warp[wid]=value;changes_warp[wid]=change;}
  __syncthreads();
  if(wid==0){value=libero_warp::sum(lane<warps?norm_warp[lane]:REAL_C(0.));
    change=libero_warp::sum(lane<warps?changes_warp[lane]:0);
    if(lane==0){norm=value;changes=precomputed_change?precomputed_change[0]:change;}
  }
  __syncthreads();
  if(threadIdx.x==0)stats_stop(w,active,nefc,stage,committed,oldstate,state,grad,
    oldcost,cost,inertia,tolerance,slope,nactive,nupdate,neval,iteration,base,maxiter,
    sf,si,nchange,improvement,gradient,stop,reason,out_iter,status,nv,rows,&changes,&norm,true,&residual_ok);
}

__device__ void finalize(int w,int* active,int* exit,int* iteration,real_t* qacc,real_t* warm,int* iter,int* nnz,int* status,int nv){
  if(status[w]||!active[w])return;if(active[w]!=1){status[w]=1;return;}
  if(exit[w]!=0&&exit[w]!=1){status[w]=4;return;}if(!exit[w])return;int it=iteration[w];
  if(it<0||iter[w]<0||iter[w]>2147483647-it){status[w]=2;return;}
  for(int j=0;j<nv;++j)if(!isfinite(qacc[w*nv+j])){status[w]=3;return;}
  iter[w]+=it;nnz[w]=nv*nv;for(int j=0;j<nv;++j)warm[w*nv+j]=qacc[w*nv+j];
}


// All-thread transactional entry points. Scalar entry points above remain the
// reference for standalone callers. The caller publishes each stage with its
// existing block barrier; no destination is touched before collective checks.
__device__ bool n5_any_bad(bool bad){
  __shared__ int partial[4],result;
  const int lane=libero_warp::lane(),wid=libero_warp::warp();
  bad=libero_warp::any(bad);
  if(lane==0)partial[wid]=bad;
  __syncthreads();
  if(wid==0){bool value=lane<blockDim.x/32&&partial[lane];
    value=libero_warp::any(value);if(lane==0)result=value;}
  __syncthreads();return result!=0;
}
__device__ void warm_select_block(int w,int* active,int* nefc,real_t* wc,real_t* sc,real_t* warm,real_t* smooth,
 real_t* sf,int* ss,real_t* sq,real_t* qacc,real_t* force,int* state,real_t* qfrc,int* choice,int* status,int nv,int rows){
  __shared__ int run;
  if(threadIdx.x==0){run=!status[w]&&active[w];}
  __syncthreads();if(!run)return;
  if(threadIdx.x==0){
    if(active[w]!=1)status[w]=1;
    else if(nefc[w]<=0||nefc[w]>rows)status[w]=2;
    else if(!isfinite(wc[w])||!isfinite(sc[w]))status[w]=3;
  }
  __syncthreads();if(status[w])return;
  const int n=nefc[w],v=w*nv,r=w*rows;bool bad=false;
  for(int i=threadIdx.x;i<nv;i+=blockDim.x)
    bad|=!isfinite(warm[v+i])||!isfinite(smooth[v+i])||!isfinite(sq[v+i]);
  for(int i=threadIdx.x;i<n;i+=blockDim.x)bad|=!isfinite(sf[r+i]);
  if(n5_any_bad(bad)){if(threadIdx.x==0)status[w]=3;return;}
  const int select=wc[w]>sc[w];if(threadIdx.x==0)choice[w]=select;
#if LIBERO_NEWTON_SCALAR_IO
  for(int i=threadIdx.x;i<nv;i+=blockDim.x){qacc[v+i]=select?smooth[v+i]:warm[v+i];qfrc[v+i]=sq[v+i];}
  for(int i=threadIdx.x;i<n;i+=blockDim.x){force[r+i]=sf[r+i];state[r+i]=ss[r+i];}
#else
  const real_t* selected=select?smooth+v:warm+v;
  const bool independent=n_state_disjoint(qacc+v,size_t(nv)*8,qfrc+v,size_t(nv)*8)&&
    n_state_disjoint(qacc+v,size_t(nv)*8,sq+v,size_t(nv)*8)&&
    n_state_disjoint(qfrc+v,size_t(nv)*8,selected,size_t(nv)*8)&&
    (qacc+v==selected||n_state_disjoint(qacc+v,size_t(nv)*8,selected,size_t(nv)*8))&&
    (qfrc+v==sq+v||n_state_disjoint(qfrc+v,size_t(nv)*8,sq+v,size_t(nv)*8));
  if(independent){libero_newton_io::copy_block(selected,qacc+v,nv);libero_newton_io::copy_block(sq+v,qfrc+v,nv);}
  else for(int i=threadIdx.x;i<nv;i+=blockDim.x){qacc[v+i]=selected[i];qfrc[v+i]=sq[v+i];}
  const bool row_independent=n_state_disjoint(force+r,size_t(n)*8,state+r,size_t(n)*4)&&
    n_state_disjoint(force+r,size_t(n)*8,ss+r,size_t(n)*4)&&
    n_state_disjoint(state+r,size_t(n)*4,sf+r,size_t(n)*8)&&
    (force+r==sf+r||n_state_disjoint(force+r,size_t(n)*8,sf+r,size_t(n)*8))&&
    (state+r==ss+r||n_state_disjoint(state+r,size_t(n)*4,ss+r,size_t(n)*4));
  if(row_independent){libero_newton_io::copy_block(sf+r,force+r,n);libero_newton_io::copy_block(ss+r,state+r,n);}
  else for(int i=threadIdx.x;i<n;i+=blockDim.x){force[r+i]=sf[r+i];state[r+i]=ss[r+i];}
#endif // LIBERO_NEWTON_SCALAR_IO
}
__device__ void commit_step_block(int w,int* active,int* nefc,real_t* alpha,real_t* search,real_t* mv,real_t* jv,
 real_t* qacc,real_t* ma,real_t* jar,int* committed,int* status,int nv,int rows){
  __shared__ int run;
  if(threadIdx.x==0){run=!status[w]&&active[w];}
  __syncthreads();if(!run)return;
  if(threadIdx.x==0){if(active[w]!=1)status[w]=1;
    else if(nefc[w]<=0||nefc[w]>rows)status[w]=2;
    else if(!isfinite(alpha[w]))status[w]=3;}
  __syncthreads();if(status[w])return;
  const int n=nefc[w],v=w*nv,r=w*rows;const real_t a=alpha[w];
  if(a==0){if(threadIdx.x==0)committed[w]=0;return;}
  bool bad=false;
  for(int i=threadIdx.x;i<nv;i+=blockDim.x)
    bad|=!isfinite(search[v+i])||!isfinite(mv[v+i])||!isfinite(qacc[v+i])||!isfinite(ma[v+i])||
      !isfinite(qacc[v+i]+a*search[v+i])||!isfinite(ma[v+i]+a*mv[v+i]);
  for(int i=threadIdx.x;i<n;i+=blockDim.x)
    bad|=!isfinite(jv[r+i])||!isfinite(jar[r+i])||!isfinite(jar[r+i]+a*jv[r+i]);
  if(n5_any_bad(bad)){if(threadIdx.x==0)status[w]=3;return;}
  if(threadIdx.x==0)committed[w]=1;
#if LIBERO_NEWTON_SCALAR_IO
  for(int i=threadIdx.x;i<nv;i+=blockDim.x)qacc[v+i]+=a*search[v+i];
  for(int i=threadIdx.x;i<nv;i+=blockDim.x)ma[v+i]+=a*mv[v+i];
  for(int i=threadIdx.x;i<n;i+=blockDim.x)jar[r+i]+=a*jv[r+i];
#else
  const real_t* spans[6]={qacc+v,search+v,ma+v,mv+v,jar+r,jv+r};
  bool independent=true;
  for(int x=0;x<6;++x)for(int y=x+1;y<6;++y)
    independent&=n_state_disjoint(spans[x],size_t(x<4?nv:n)*8,spans[y],size_t(y<4?nv:n)*8);
  if(independent){
    n_state_axpy(qacc+v,search+v,nv,a);
    n_state_axpy(ma+v,mv+v,nv,a);
    n_state_axpy(jar+r,jv+r,n,a);
  }else{
    for(int i=threadIdx.x;i<nv;i+=blockDim.x)qacc[v+i]+=a*search[v+i];
    for(int i=threadIdx.x;i<nv;i+=blockDim.x)ma[v+i]+=a*mv[v+i];
    for(int i=threadIdx.x;i<n;i+=blockDim.x)jar[r+i]+=a*jv[r+i];
  }
#endif // LIBERO_NEWTON_SCALAR_IO
}
__device__ void finalize_block(int w,int* active,int* exit,int* iteration,real_t* qacc,real_t* warm,int* iter,int* nnz,int* status,int nv){
  __shared__ int run;
  if(threadIdx.x==0){run=!status[w]&&active[w];}
  __syncthreads();if(!run)return;
  if(threadIdx.x==0){if(active[w]!=1)status[w]=1;
    else if(exit[w]!=0&&exit[w]!=1)status[w]=4;
    else if(exit[w]&&(iteration[w]<0||iter[w]<0||iter[w]>2147483647-iteration[w]))status[w]=2;}
  __syncthreads();if(status[w]||!exit[w])return;
  bool bad=false;
  for(int j=threadIdx.x;j<nv;j+=blockDim.x)bad|=!isfinite(qacc[w*nv+j]);
  if(n5_any_bad(bad)){if(threadIdx.x==0)status[w]=3;return;}
  if(threadIdx.x==0){iter[w]+=iteration[w];nnz[w]=nv*nv;}
#if LIBERO_NEWTON_SCALAR_IO
  for(int j=threadIdx.x;j<nv;j+=blockDim.x)warm[w*nv+j]=qacc[w*nv+j];
#else
  if(warm==qacc||n_state_disjoint(warm+w*nv,size_t(nv)*8,qacc+w*nv,size_t(nv)*8))
    libero_newton_io::copy_block(qacc+w*nv,warm+w*nv,nv);
  else for(int j=threadIdx.x;j<nv;j+=blockDim.x)warm[w*nv+j]=qacc[w*nv+j];
#endif // LIBERO_NEWTON_SCALAR_IO
}
