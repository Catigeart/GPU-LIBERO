#include "precision.cuh"
#include "io_policy.cuh"
#pragma once
#ifndef DATAFLOW_FAST_PROFILE
#define DATAFLOW_FAST_PROFILE 0
#endif
__device__ inline real_t cone_norm(const real_t* jar,const real_t* f,int d){
  real_t u1=0,u2=0,u3=0,u4=0,u5=0,s=0;
  if(d>1){u1=jar[1]*f[0];u2=jar[2]*f[1];}
  if(d>3)u3=jar[3]*f[2];
  if(d==6){u4=jar[4]*f[3];u5=jar[5]*f[4];s=((u1*u1+u3*u3)+(u2*u2+u4*u4))+u5*u5;}
  else if(d==4)s=s+(u1*u1+u2*u2+u3*u3);
  else if(d==3)s=s+(u1*u1+u2*u2);
  return sqrt(s);
}
__device__ void constraint_update(int w,int* active,int* count,int* ne,int* nf,int* ncon,int* flag,
 real_t* jar,real_t* D,real_t* R,real_t* loss,int* typ,int* cid,real_t* J,real_t* mu,real_t* fr,
 int* dim,real_t* force,int* state,real_t* H,real_t* cost,int* status,int rows,int contacts,int nv,
 int* invariant_valid=nullptr,int* active_count=nullptr,int* cone_count=nullptr,
 int* old_state=nullptr,int* changed_count=nullptr,
 bool validate_only=false,int* group_head=nullptr,int* hessian_owner=nullptr,
 bool topology_only=false,bool rows_checked=false,bool collect_owner=false){
  status[w]=0;if(!active[w])return;if(active[w]!=1){status[w]=1;return;}
  int n=count[w],nc=ncon[w],eq=ne[w],fc=nf[w],rb=w*rows,cb=w*contacts;
  if(n<0||n>rows||nc<0||nc>contacts||(flag[w]!=0&&flag[w]!=1)||eq<0||eq>n||fc<0||fc>n-eq){status[w]=2;return;}
  bool cached=invariant_valid&&invariant_valid[0];
  if(!cached&&!rows_checked)for(int i=eq+fc;i<n;){int t=typ[rb+i];
    if(t==7){int c=cid[rb+i];if(c<0||c>=nc){status[w]=3;return;}int d=dim[cb+c];
      if((d!=1&&d!=3&&d!=4&&d!=6)||i+d>n||mu[cb+c]<=0||!isfinite(mu[cb+c])){status[w]=4;return;}
      for(int j=0;j<d;++j)if(typ[rb+i+j]!=7||cid[rb+i+j]!=c){status[w]=5;return;}
      for(int j=0;j<d-1;++j)if(fr[(cb+c)*5+j]<=0||!isfinite(fr[(cb+c)*5+j])){status[w]=6;return;}
      i+=d;
    }else if(t>=0&&t<=6)++i;else{status[w]=7;return;}
  }
  if(topology_only)return;
  if(!rows_checked)for(int r=0;r<n;++r){int z=rb+r;
    if(!isfinite(jar[z])||(!cached&&(!isfinite(D[z])||!isfinite(R[z])||!isfinite(loss[z])))){status[w]=8;return;}
    if(!cached)for(int j=0;j<nv;++j)if(!isfinite(J[z*nv+j])){status[w]=9;return;}
  }
  if(!rows_checked)for(int r=0;r<n;++r){int z=rb+r;
    if(jar[z]>10000||jar[z]<-10000||(!cached&&(D[z]>REAL_C(2e16)||D[z]<-10000||R[z]>10000||R[z]<-10000||loss[z]>10000||loss[z]<-10000))){status[w]=10;return;}
    if(!cached)for(int j=0;j<nv;++j)if(J[z*nv+j]>10000||J[z*nv+j]<-10000){status[w]=10;return;}
  }
  if(hessian_owner&&!rows_checked)for(int c=0;c<nc;++c)hessian_owner[cb+c]=-1;
  for(int i=eq+fc;i<n;){if(typ[rb+i]!=7){++i;continue;}int c=cb+cid[rb+i],d=dim[c];real_t m=mu[c];
    if(!cached){if(m<REAL_C(.0001)||m>10000){status[w]=10;return;}
      for(int j=0;j<d-1;++j)if(fr[c*5+j]<REAL_C(.0001)||fr[c*5+j]>10000){status[w]=10;return;}}
    real_t T=cone_norm(jar+rb+i,fr+c*5,d),N=jar[rb+i]*m;
    bool top=N>=m*T||(T<=0&&N>=0),bottom=m*N+T<=0||(T<=0&&N<0);
    if(!top&&!bottom&&T<REAL_C(1e-12)){status[w]=10;return;}
    // Repeated contact ids are legal. Keep the scalar last middle-cone writer,
    // including cases where a later top/bottom cone intentionally leaves H held.
    if(hessian_owner&&(!rows_checked||collect_owner)&&!top&&!bottom)hessian_owner[c]=i;i+=d;
  }
  // The emitter resets invariant_valid for every launch. The group map only
  // lives across Newton iterations with immutable row topology in that launch.
  if(group_head&&!cached){
    for(int r=0;r<eq+fc;++r)group_head[rb+r]=r;
    for(int i=eq+fc;i<n;){int d=typ[rb+i]==7?dim[cb+cid[rb+i]]:1;
      group_head[rb+i]=i;for(int j=1;j<d;++j)group_head[rb+i+j]=-1;i+=d;}
  }
  if(invariant_valid)invariant_valid[0]=1;
  if(validate_only)return;
  real_t s=0;int na=eq+fc,nco=0,nchanged=0;
  for(int r=0;r<n;++r)force[rb+r]=(-REAL_C(1.0)*D[rb+r])*jar[rb+r];
  for(int r=0;r<eq;++r){s+=REAL_C(.5)*D[rb+r]*jar[rb+r]*jar[rb+r];
    if(old_state&&old_state[rb+r]!=1)++nchanged;state[rb+r]=1;}
  for(int r=eq;r<eq+fc;++r){int z=rb+r;real_t v=jar[z],rr=R[z],f=loss[z];
    int next;if(v<=(-REAL_C(1.0)*rr)*f){s+=-REAL_C(.5)*rr*f*f-f*v;force[z]=f;next=2;}
    else if(v>=rr*f){s+=-REAL_C(.5)*rr*f*f+f*v;force[z]=-REAL_C(1.0)*f;next=3;}
    else{s+=REAL_C(.5)*D[z]*v*v;next=1;}
    if(old_state&&old_state[z]!=next)++nchanged;state[z]=next;
  }
  for(int i=eq+fc;i<n;){int z=rb+i;
    if(typ[z]!=7){int next;if(jar[z]>=0){force[z]=REAL_C(0.0);next=0;}else{s+=REAL_C(.5)*D[z]*jar[z]*jar[z];next=1;++na;}
      if(old_state&&old_state[z]!=next)++nchanged;state[z]=next;++i;continue;}
    int c=cb+cid[z],d=dim[c],fb=c*5;real_t m=mu[c],N=jar[z]*m,T=cone_norm(jar+z,fr+fb,d);
    if(N>=m*T||(T<=0&&N>=0)){for(int j=0;j<d;++j)force[z+j]=REAL_C(0.0);state[z]=0;}
    else if(m*N+T<=0||(T<=0&&N<0)){for(int j=0;j<d;++j)s+=REAL_C(.5)*D[z+j]*jar[z+j]*jar[z+j];state[z]=1;}
    else{real_t dm=D[z]/(m*m*(1+m*m)),nm=N-m*T;s+=REAL_C(.5)*dm*nm*nm;force[z]=(-REAL_C(1.0)*dm)*nm*m;
      for(int j=1;j<d;++j)force[z+j]=(-REAL_C(1.0)*force[z])/T*(jar[z+j]*fr[fb+j-1])*fr[fb+j-1];state[z]=4;
      if(flag[w]){real_t* h=H+c*36;real_t scale=(-REAL_C(1.0)*m)/T;h[0]=1;
        for(int j=1;j<d;++j)h[j]=scale*(jar[z+j]*fr[fb+j-1]);scale=m*N/(T*T*T);
        for(int k=1;k<d;++k){real_t uk=jar[z+k]*fr[fb+k-1];for(int j=k;j<d;++j)h[k*d+j]=scale*(jar[z+j]*fr[fb+j-1])*uk;}
        scale=m*m-m*N/T;for(int j=1;j<d;++j)h[j*(d+1)]+=scale;
        for(int k=0;k<d;++k){real_t a=dm*(k==0?m:fr[fb+k-1]);for(int j=k;j<d;++j)h[k*d+j]*=a*(j==0?m:fr[fb+j-1]);}
        for(int k=0;k<d;++k)for(int j=k+1;j<d;++j)h[j*d+k]=h[k*d+j];
      }
    }int next=state[z];if(next)na+=d;if(next==4)nco+=d;
    for(int j=0;j<d;++j){if(old_state&&old_state[z+j]!=next)++nchanged;state[z+j]=next;}i+=d;
  }cost[w]=s;
  if(active_count)active_count[w]=na;if(cone_count)cone_count[w]=nco;
  if(changed_count)changed_count[0]=nchanged;
}

// Runtime span qualification preserves the admitted alias domain: overlapping
// outputs use the retained direct read/write equations below, never staging.
__device__ inline bool constraint_spans_overlap(const real_t* a,int an,const real_t* b,int bn){
  const unsigned long long ab=(unsigned long long)a,bb=(unsigned long long)b;
  return an>0&&bn>0&&ab<bb+(unsigned long long)bn*sizeof(real_t)&&bb<ab+(unsigned long long)an*sizeof(real_t);
}

// Fixed dimension kernels keep the middle-cone's at-most-six component work
// local to its owning lane. Typed passes make the dimension uniform per warp;
// no shuffle or barrier occurs inside a numeric zone branch.
template<int d>
__device__ inline real_t constraint_middle_cone(real_t* jar,real_t* fr,real_t D,
 real_t m,real_t N,real_t T,real_t* force,real_t* h,bool write_h){
  // The sole middle-cone owner writes its mathematical d*d block directly.
  // Avoid an address-taken local matrix and a second full copy; held tails,
  // input staging and overlapping-buffer fallback retain their existing rules.
  real_t dm=D/(m*m*(1+m*m)),nm=N-m*T;
  real_t cost=REAL_C(.5)*dm*nm*nm;force[0]=(-REAL_C(1.0)*dm)*nm*m;
  #pragma unroll
  for(int j=1;j<d;++j)force[j]=(-REAL_C(1.0)*force[0])/T*(jar[j]*fr[j-1])*fr[j-1];
  if(write_h){real_t scale=(-REAL_C(1.0)*m)/T;h[0]=1;
    #pragma unroll
    for(int j=1;j<d;++j)h[j]=scale*(jar[j]*fr[j-1]);scale=m*N/(T*T*T);
    #pragma unroll
    for(int k=1;k<d;++k){real_t uk=jar[k]*fr[k-1];
      #pragma unroll
      for(int j=k;j<d;++j)h[k*d+j]=scale*(jar[j]*fr[j-1])*uk;}
    scale=m*m-m*N/T;
    #pragma unroll
    for(int j=1;j<d;++j)h[j*(d+1)]+=scale;
    #pragma unroll
    for(int k=0;k<d;++k){real_t a=dm*(k==0?m:fr[k-1]);
      #pragma unroll
      for(int j=k;j<d;++j)h[k*d+j]*=a*(j==0?m:fr[j-1]);}
    #pragma unroll
    for(int k=0;k<d;++k){
      #pragma unroll
      for(int j=k+1;j<d;++j)h[j*d+k]=h[k*d+j];}
  }
  return cost;
}

__device__ inline int constraint_head_rank(const int* heads,int count,int row){
  int lo=0,hi=count;
  while(lo<hi){int mid=lo+(hi-lo)/2;if(heads[mid]<row)lo=mid+1;else hi=mid;}
  return lo<count&&heads[lo]==row?lo:-1;
}

// Newton/S07 consumes offsets[t:t+2] only for t<ceil(nefc/16).
// Produce that live prefix cooperatively after ordered heads are published.
// The lower bound counts exactly the heads whose final row precedes tile t.
__device__ inline void constraint_tile_head_offsets_block(int n,int eq,int fc,
 int rb,int cb,const int* typ,const int* cid,const int* dim,
 const int* heads,int nh,int* offsets){
  for(int t=threadIdx.x;t<=(n+15)/16;t+=blockDim.x){
    int lo=0,hi=nh;
    while(lo<hi){
      const int mid=lo+(hi-lo)/2,r=heads[mid];
      const int d=r>=eq+fc&&typ[rb+r]==7?dim[cb+cid[rb+r]]:1;
      if((r+d-1)/16<t)lo=mid+1;else hi=mid;
    }
    offsets[t]=lo;
  }
}

// Advisory ordinal operands are selected only after the first ordinary N update
// has validated current rows and built its own head/G lists. The emitter resets
// selected at every invocation and keeps a block barrier after this lane0 gate.
// Metadata mismatch is a fallback, never a new solver/publication error.
__device__ void constraint_head_metadata_gate(const int* active,const int* status,
 const int* invariant_valid,const int* count,const int* nefc,const int* ne,const int* nf,
 const int* ncon,const int* typ,const int* cid,const int* dim,const real_t* mu,const real_t* fr,
 const int* row_dim,const real_t* row_mu,const real_t* row_fr,const int* group_head,
 const int* head_rows,const int* head_count,const int* source_counts,const int* source_ncon,
 const int* compact_heads,const int* compact_count,const int* compact_valid,
 const int* head_type,const int* head_id,const int* head_dim,const real_t* head_mu,
 const real_t* head_friction,const int* head_valid,int* selected,int rows,int contacts,int nv){
  if(threadIdx.x!=0)return;
  selected[0]=0;
  if(active[0]!=1||status[0]||invariant_valid[0]!=1)return;
  const int n=count[0],eq=ne[0],fc=nf[0],nc=ncon[0],nh=head_count[0];
  if(n<=0||n>rows||nefc[0]!=n||eq<0||eq>n||fc<0||fc>n-eq||nc<0||nc>contacts)return;
  if(compact_valid[0]!=1||head_valid[0]!=1||nh<0||nh>n||compact_count[0]!=nh)return;
  if(source_counts[0]!=eq||source_counts[1]!=fc||source_counts[3]!=n||
     source_counts[4]!=n*nv||source_ncon[0]!=nc)return;
  if(source_counts[2]<0||source_counts[2]>n-eq-fc)return;
  int next=0,limits=0;
  for(int k=0;k<nh;++k){
    const int r=head_rows[k];
    if(r!=next||r<0||r>=n||compact_heads[k]!=r||group_head[r]!=r)return;
    const int t=typ[r],c=cid[r];
    if(head_type[k]!=t||head_id[k]!=c)return;
    int d=1;
    if(r>=eq+fc&&t==7){
      if(c<0||c>=nc)return;
      d=dim[c];
      if((d!=1&&d!=3&&d!=4&&d!=6)||d>n-r||head_dim[k]!=d||row_dim[r]!=d)return;
      // Compare both current N views against S's ordinal view. Positive finite
      // operands were already admitted by the original ordered N validation.
      const long long m_bits=real_bits(head_mu[k]);
      if(m_bits!=real_bits(mu[c])||m_bits!=real_bits(row_mu[r]))return;
      for(int j=0;j<d-1;++j){
        const real_t f=head_friction[j*rows+k];
        const long long f_bits=real_bits(f);
        if(f_bits!=real_bits(fr[c*5+j])||f_bits!=real_bits(row_fr[r*5+j]))return;
      }
      for(int j=1;j<d;++j)
        if(group_head[r+j]!=-1||typ[r+j]!=t||cid[r+j]!=c)return;
    }else{
      if(head_dim[k]!=1)return;
      if(r>=eq+fc&&(t==3||t==4))++limits;
    }
    next=r+d;
  }
  if(next!=n||limits!=source_counts[2])return;
  selected[0]=1;
}

// All threads enter this collective. Each scalar row or complete cone group
// has one owner; no thread writes another group's force/state/H cells. Group
// mathematics retains the scalar evaluation order; only cost/count sums change.
__device__ void constraint_update_block(int w,int* active,int* count,int* ne,int* nf,int* ncon,int* flag,
 real_t* jar,real_t* D,real_t* R,real_t* loss,int* typ,int* cid,real_t* J,real_t* mu,real_t* fr,
 int* dim,real_t* force,int* state,real_t* H,real_t* cost,int* status,int rows,int contacts,int nv,
 int* invariant_valid,int* active_count,int* cone_count,int* old_state,int* changed_count,int* group_head,int* hessian_owner,
 int* head_rows=nullptr,int* head_count=nullptr,int* tile_head_offset=nullptr,
 int* typed_head_rows=nullptr,int* typed_head_offsets=nullptr,
 const int* head_dim=nullptr,const real_t* head_mu=nullptr,const real_t* head_friction=nullptr){
  __shared__ real_t cost_warp[4];
  __shared__ int active_warp[4],cone_warp[4],change_warp[4];
  __shared__ int check_warp[4],check_result;
  // Topology parsing stays ordered: variable-size cone groups and original
  // error precedence must be established before any parallel row access.
  if(threadIdx.x==0)constraint_update(w,active,count,ne,nf,ncon,flag,jar,D,R,loss,typ,cid,J,mu,fr,
    dim,force,state,H,cost,status,rows,contacts,nv,invariant_valid,nullptr,nullptr,nullptr,nullptr,true,group_head,hessian_owner,true);
  __syncthreads();
  if(status[w]||!active[w])return; // uniform after the validation barrier
  int n=count[w],eq=ne[w],fc=nf[w],rb=w*rows,cb=w*contacts;
  const int check_lane=libero_warp::lane(),check_wid=libero_warp::warp(),check_warps=blockDim.x/32;
  const bool cached=invariant_valid&&invariant_valid[0];
  int bad=libero_warp::no_index;
  for(int r=threadIdx.x;r<n;r+=blockDim.x){int z=rb+r;
    if(!isfinite(jar[z])||(!cached&&(!isfinite(D[z])||!isfinite(R[z])||!isfinite(loss[z]))))
      bad=min(bad,2*r);
    if(jar[z]>10000||jar[z]<-10000||(!cached&&(D[z]>REAL_C(2e16)||D[z]<-10000||R[z]>10000||R[z]<-10000||loss[z]>10000||loss[z]<-10000)))
      bad=min(bad,2*n);
  }
  if(!cached)for(int k=threadIdx.x;k<n*nv;k+=blockDim.x){real_t v=J[rb*nv+k];
    if(!isfinite(v))bad=min(bad,2*(k/nv)+1);
    if(v>10000||v<-10000)bad=min(bad,2*n);
  }
  // Range verdict sorts after every finite row/J verdict; a single reduction
  // therefore preserves both scalar phases without a second block barrier.
  bad=libero_warp::min(bad);
  if(check_lane==0)check_warp[check_wid]=bad;
  __syncthreads();
  if(check_wid==0){int first=libero_warp::min(check_lane<check_warps?check_warp[check_lane]:libero_warp::no_index);
    if(check_lane==0){check_result=first;if(first!=libero_warp::no_index)status[w]=first==2*n?10:((first&1)?9:8);}}
  __syncthreads();
  if(check_result!=libero_warp::no_index)return;
  // Reuse the ordered numeric admission's cone classification to choose the
  // last middle-cone writer. The map is private; no numeric output is touched
  // before all original checks pass. Compatible callers retain their path.
#if DATAFLOW_FAST_PROFILE
  if(hessian_owner)
    for(int c=threadIdx.x;c<ncon[w];c+=blockDim.x)hessian_owner[cb+c]=-1;
  __syncthreads();
  constexpr bool collect_owner=true;
#else
  constexpr bool collect_owner=false;
#endif
  // Complete numeric cone checks in their original order, then publish the
  // topology map. The scalar implementation remains available as baseline.
  if(threadIdx.x==0)constraint_update(w,active,count,ne,nf,ncon,flag,jar,D,R,loss,typ,cid,J,mu,fr,
    dim,force,state,H,cost,status,rows,contacts,nv,invariant_valid,nullptr,nullptr,nullptr,nullptr,true,group_head,hessian_owner,false,true,collect_owner);
  __syncthreads();
  if(status[w])return;
  // World-local scratch, rebuilt after admission on the first iteration of
  // each launch. Keep ascending original row IDs for line search; a separate
  // stable typed permutation makes each work pass uniform in type/dimension.
  // No numeric-zone sorting: zones vary with jar at every Newton iteration.
  if(threadIdx.x==0&&!cached&&head_rows&&head_count){
    int sizes[7]={0,0,0,0,0,0,0},nh=0;
    for(int r=0;r<n;){
      int d=r>=eq+fc&&typ[rb+r]==7?dim[cb+cid[rb+r]]:1;
      int bucket=r<eq?0:(r<eq+fc?1:(typ[rb+r]!=7?2:(d==1?3:(d==3?4:(d==4?5:6)))));
      head_rows[nh++]=r;++sizes[bucket];r+=d;
    }
    head_count[0]=nh;
    if(typed_head_rows&&typed_head_offsets){
      int cursor[7];typed_head_offsets[0]=0;
      for(int b=0;b<7;++b){cursor[b]=typed_head_offsets[b];typed_head_offsets[b+1]=cursor[b]+sizes[b];}
      for(int k=0;k<nh;++k){int r=head_rows[k];
        int d=r>=eq+fc&&typ[rb+r]==7?dim[cb+cid[rb+r]]:1;
        int bucket=r<eq?0:(r<eq+fc?1:(typ[rb+r]!=7?2:(d==1?3:(d==3?4:(d==4?5:6)))));
        typed_head_rows[cursor[bucket]++]=r;
      }
    }
#if !DATAFLOW_FAST_PROFILE
    if(tile_head_offset){
      int k=0;
      for(int t=0;t<=(rows+15)/16;++t){
        while(k<nh){int r=head_rows[k];int d=r>=eq+fc&&typ[rb+r]==7?dim[cb+cid[rb+r]]:1;
          if((r+d-1)/16>=t)break;++k;}
        tile_head_offset[t]=k;
      }
    }
#endif
  }
  __syncthreads();
#if DATAFLOW_FAST_PROFILE
  if(!cached&&head_rows&&head_count&&tile_head_offset)
    constraint_tile_head_offsets_block(n,eq,fc,rb,cb,typ,cid,dim,
                                      head_rows,head_count[0],tile_head_offset);
  __syncthreads();
#endif
#if !DATAFLOW_FAST_PROFILE
  if(hessian_owner){
    for(int c=threadIdx.x;c<ncon[w];c+=blockDim.x)hessian_owner[cb+c]=-1;
    __syncthreads();
    const bool compact=head_rows&&head_count;
    for(int k=threadIdx.x;k<(compact?head_count[0]:n);k+=blockDim.x){
      int r=compact?head_rows[k]:k;
      if(r<eq+fc)continue;
      if(group_head[rb+r]!=r||typ[rb+r]!=7)continue;
      int c=cb+cid[rb+r],d=dim[c];real_t m=mu[c],T=cone_norm(jar+rb+r,fr+c*5,d),N=jar[rb+r]*m;
      bool top=N>=m*T||(T<=0&&N>=0),bottom=m*N+T<=0||(T<=0&&N<0);
      // The scalar contract is the last middle-cone writer, not the last
      // group with this contact id. Atomic max only reduces integer ownership.
      if(!top&&!bottom)atomicMax(hessian_owner+c,r);
    }
    __syncthreads();
  }
#endif
  // Use one warp only for small compiled models with at most one warp of rows.
  // The capacity condition removes this candidate from larger kernels entirely.
  const bool small=NV<=32&&n<=32;
  if(small&&threadIdx.x>=32)return;
  real_t sum=REAL_C(0.);int na=0,nco=0,nchanged=0;
  const bool compact=head_rows&&head_count;
  const bool typed=compact&&typed_head_rows&&typed_head_offsets;
  // Inspect actual input/output spans, including shifted aliases. This is a
  // qualification for staging, not a new public nonalias admission rule.
  const bool stage_inputs=
    !constraint_spans_overlap(force+rb,n,jar+rb,n)&&
    !constraint_spans_overlap(force+rb,n,D+rb,n)&&
    !constraint_spans_overlap(force+rb,n,fr+cb*5,ncon[w]*5)&&
    !constraint_spans_overlap(H+cb*36,ncon[w]*36,jar+rb,n)&&
    !constraint_spans_overlap(H+cb*36,ncon[w]*36,D+rb,n)&&
    !constraint_spans_overlap(H+cb*36,ncon[w]*36,fr+cb*5,ncon[w]*5)&&
    !constraint_spans_overlap(H+cb*36,ncon[w]*36,force+rb,n)&&
    !constraint_spans_overlap(force+rb,n,mu+cb,ncon[w])&&
    !constraint_spans_overlap(H+cb*36,ncon[w]*36,mu+cb,ncon[w])&&
    !constraint_spans_overlap(force+rb,n,(const real_t*)(dim+cb),(ncon[w]+1)/2)&&
    !constraint_spans_overlap(H+cb*36,ncon[w]*36,(const real_t*)(dim+cb),(ncon[w]+1)/2);

  for(int bucket=0;bucket<(typed?7:1);++bucket){
  int begin=typed?typed_head_offsets[bucket]:0;
  int end=typed?typed_head_offsets[bucket+1]:(compact?head_count[0]:n);
  for(int k=begin+threadIdx.x;k<end;k+=blockDim.x){
    int r=typed?typed_head_rows[k]:(compact?head_rows[k]:k);
    if(group_head[rb+r]!=r)continue;
    int z=rb+r,next=1;
    if(r<eq){force[z]=(-REAL_C(1.0)*D[z])*jar[z];sum+=REAL_C(.5)*D[z]*jar[z]*jar[z];++na;}
    else if(r<eq+fc){real_t v=jar[z],rr=R[z],f=loss[z];force[z]=(-REAL_C(1.0)*D[z])*v;
      if(v<=(-REAL_C(1.0)*rr)*f){sum+=-REAL_C(.5)*rr*f*f-f*v;force[z]=f;next=2;}
      else if(v>=rr*f){sum+=-REAL_C(.5)*rr*f*f+f*v;force[z]=-REAL_C(1.0)*f;next=3;}
      else sum+=REAL_C(.5)*D[z]*v*v;++na;
    }else if(typ[z]!=7){force[z]=(-REAL_C(1.0)*D[z])*jar[z];
      if(jar[z]>=0){force[z]=REAL_C(0.0);next=0;}else{sum+=REAL_C(.5)*D[z]*jar[z]*jar[z];++na;}
    }else{
      const int rank=compact?constraint_head_rank(head_rows,head_count[0],r):-1;
      int c=cb+cid[z],d=stage_inputs&&head_dim&&rank>=0?head_dim[rank]:dim[c],fb=c*5;
      real_t m=stage_inputs&&head_mu&&rank>=0?head_mu[rank]:mu[c];
      // At most six components remain in registers through force, norm, cost
      // and the middle-cone Hessian producer. Scatter retains original row IDs.
      real_t local_jar[6],local_D[6],local_fr[5];
      real_t* input_jar=jar+z;real_t* input_D=D+z;real_t* input_fr=fr+fb;
      if(stage_inputs){
        for(int j=0;j<d;++j){local_jar[j]=jar[z+j];local_D[j]=D[z+j];}
        for(int j=0;j<d-1;++j)local_fr[j]=head_friction&&rank>=0?head_friction[j*rows+rank]:fr[fb+j];
        input_jar=local_jar;input_D=local_D;input_fr=local_fr;
      }
      real_t N=input_jar[0]*m,T=cone_norm(input_jar,input_fr,d);
#if LIBERO_NEWTON_SCALAR_IO
      for(int j=0;j<d;++j)force[z+j]=(-REAL_C(1.0)*input_D[j])*input_jar[j];
      if(N>=m*T||(T<=0&&N>=0)){for(int j=0;j<d;++j)force[z+j]=REAL_C(0.0);next=0;}
#else
      // The owning lane's pair domain is this cone only. Staged inputs are
      // disjoint from every output; aliases keep the direct scalar equation.
      if(stage_inputs){int j=0;
        for(;j+1<d;j+=2){real2_t dv=libero_vec::load_pair(input_D+j),jv=libero_vec::load_pair(input_jar+j);
          libero_vec::store_pair(force+z+j,(-REAL_C(1.0)*dv.x)*jv.x,(-REAL_C(1.0)*dv.y)*jv.y);}
        if(j<d)force[z+j]=(-REAL_C(1.0)*input_D[j])*input_jar[j];
      }else for(int j=0;j<d;++j)force[z+j]=(-REAL_C(1.0)*input_D[j])*input_jar[j];
      if(N>=m*T||(T<=0&&N>=0)){
        if(stage_inputs)libero_newton_io::fill_serial(force+z,d,REAL_C(0.0));
        else for(int j=0;j<d;++j)force[z+j]=REAL_C(0.0);
        next=0;}
#endif // LIBERO_NEWTON_SCALAR_IO
      else if(m*N+T<=0||(T<=0&&N<0)){for(int j=0;j<d;++j)sum+=REAL_C(.5)*input_D[j]*input_jar[j]*input_jar[j];next=1;}
      else{bool write_h=flag[w]&&hessian_owner[c]==r;real_t* h=H+c*36;
        if(d==3)sum+=constraint_middle_cone<3>(input_jar,input_fr,input_D[0],m,N,T,force+z,h,write_h);
        else if(d==4)sum+=constraint_middle_cone<4>(input_jar,input_fr,input_D[0],m,N,T,force+z,h,write_h);
        else if(d==6)sum+=constraint_middle_cone<6>(input_jar,input_fr,input_D[0],m,N,T,force+z,h,write_h);
        else sum+=constraint_middle_cone<1>(input_jar,input_fr,input_D[0],m,N,T,force+z,h,write_h);
        next=4;
      }
      if(next)na+=d;if(next==4)nco+=d;
#if LIBERO_NEWTON_SCALAR_IO
      for(int j=0;j<d;++j){if(old_state&&old_state[z+j]!=next)++nchanged;state[z+j]=next;}
#else
      // Scan before packet stores only when the old state cannot be changed
      // by them. Shifted aliases retain the original scan/write ordering.
      const uintptr_t sa=(uintptr_t)(state+z),oa=(uintptr_t)(old_state?old_state+z:nullptr);
      const bool state_disjoint=!old_state||(sa<=oa?oa-sa>=size_t(d)*sizeof(int):sa-oa>=size_t(d)*sizeof(int));
      if(state_disjoint){
        for(int j=0;j<d;++j)if(old_state&&old_state[z+j]!=next)++nchanged;
        libero_newton_io::fill_serial(state+z,d,next);
      }else for(int j=0;j<d;++j){if(old_state&&old_state[z+j]!=next)++nchanged;state[z+j]=next;}
#endif // LIBERO_NEWTON_SCALAR_IO
      continue;
    }
    if(old_state&&old_state[z]!=next)++nchanged;state[z]=next;
  }
  }
  sum=libero_warp::sum(sum);na=libero_warp::sum(na);
  nco=libero_warp::sum(nco);nchanged=libero_warp::sum(nchanged);
  if(small){
    if(threadIdx.x==0){cost[w]=sum;if(active_count)active_count[w]=na;if(cone_count)cone_count[w]=nco;
      if(changed_count)changed_count[0]=nchanged;}
    return; // caller's stage barrier publishes force/state/H and counts
  }
  const int lane=libero_warp::lane(),wid=libero_warp::warp(),warps=blockDim.x/32;
  if(lane==0){cost_warp[wid]=sum;active_warp[wid]=na;cone_warp[wid]=nco;change_warp[wid]=nchanged;}
  __syncthreads();
  if(wid==0){sum=libero_warp::sum(lane<warps?cost_warp[lane]:REAL_C(0.));
    na=libero_warp::sum(lane<warps?active_warp[lane]:0);
    nco=libero_warp::sum(lane<warps?cone_warp[lane]:0);
    nchanged=libero_warp::sum(lane<warps?change_warp[lane]:0);
    if(lane==0){cost[w]=sum;if(active_count)active_count[w]=na;if(cone_count)cone_count[w]=nco;
      if(changed_count)changed_count[0]=nchanged;}
  }
}

__device__ void constraint_transpose(int tid,int* active,int* count,real_t* J,real_t* force,real_t* qfrc,int* status,int rows,int nv){
  int w=tid/nv,j=tid%nv;if(!active[w]||status[w])return;real_t s=0;
  for(int r=0;r<count[w];++r){real_t f=force[w*rows+r];if(f!=0)s+=J[(w*rows+r)*nv+j]*f;}qfrc[tid]=s;
}
