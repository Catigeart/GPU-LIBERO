#include "precision.cuh"
#pragma once
#ifndef CF_PM_STATIC_NBODY
#define CF_PM_STATIC_NBODY -1
#endif
#ifndef CF_PM_STATIC_NJNT
#define CF_PM_STATIC_NJNT -1
#endif
#ifndef CF_PM_STATIC_NV
#define CF_PM_STATIC_NV -1
#endif
#ifndef CF_PM_STATIC_NM
#define CF_PM_STATIC_NM -1
#endif
#ifdef LIBERO_POSITION_DATAFLOW
#define LIBERO_POSITION_MASS_INLINE __forceinline__
#else
#define LIBERO_POSITION_MASS_INLINE inline
#endif
#include <stdint.h>
#include <math.h>

// Handwritten MuJoCo spatial-inertia and tree-sparse mass stages. These are
// tree recurrences retain their order; compos/crb are whole-block entries.
namespace cf_pm {
// Private component planes replace shared AoS storage, never add a mirror.
__device__ __forceinline__ int slot(int item,int component,int count,int width,bool planes) {
    return planes?component*count+item:item*width+component;
}
__device__ __forceinline__ void gather(const real_t* source,real_t* value,int item,int count,int width,bool planes) {
    if(!planes || count==1){libero_vec::copy_serial(source+(planes?item:item*width),value,width);return;}
    for(int c=0;c<width;++c)value[c]=source[slot(item,c,count,width,planes)];
}
__device__ __forceinline__ void scatter(real_t* destination,const real_t* value,int item,int count,int width,bool planes) {
    if(!planes || count==1){libero_vec::copy_serial(value,destination+(planes?item:item*width),width);return;}
    for(int c=0;c<width;++c)destination[slot(item,c,count,width,planes)]=value[c];
}
// Publication is a permutation of raw uint64 bits, including signed zero/NaN.
__device__ __forceinline__ void publish_planes(real_t* destination,const real_t* source,int count,int width) {
    real_bits_t* out=reinterpret_cast<real_bits_t*>(destination);
    const real_bits_t* in=reinterpret_cast<const real_bits_t*>(source);
    // Own pairs of consecutive AoS output cells. The input permutation may
    // cross planes, so read each source cell separately; guard the actual
    // destination alignment and retain a scalar odd tail.
    const int cells=count*width;
    for(int pair=threadIdx.x;pair<cells/2;pair+=blockDim.x){
        const int k=2*pair;
        const real_bits_t x=in[(k%width)*count+k/width];
        const real_bits_t y=in[((k+1)%width)*count+(k+1)/width];
        if(libero_vec::aligned16(destination+k))
            libero_vec::store_pair(destination+k,real_from_bits((long long)x),real_from_bits((long long)y));
        else {out[k]=x;out[k+1]=y;}
    }
    if((cells&1)&&threadIdx.x==0){const int k=cells-1;out[k]=in[(k%width)*count+k/width];}
}
// Called by all threads in a whole-warp block. Register predicates are combined
// within each warp before the cross-warp shared-memory synchronization.
__device__ __forceinline__ bool finite_lane(const real_t* a,int n) {
    bool ok=true;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)ok&=isfinite(a[i]);
    return ok;
}
// Explicit positive vector route for cold specialization; default has no guard.
__device__ __forceinline__ bool finite_lane_vector(const real_t* a,int n) {
    return !libero_vec::any_nonfinite_block_lane(a,n);
}
// Cold model specialization: absent model macros compile to the old scalar loop.
template<int Count> __device__ __forceinline__ bool finite_lane_known(const real_t* a,int n) {
    if constexpr(Count==342)return finite_lane_vector(a,n);
    else return finite_lane(a,n);
}
__device__ __forceinline__ bool invalid_block(bool bad) {
    return __syncthreads_or(libero_warp::any(bad))!=0;
}
__device__ __forceinline__ void zero_block(real_t* a,int n) {
    libero_vec::fill_block(a,n,REAL_C(0.0));
}
__device__ LIBERO_POSITION_MASS_INLINE void zero(real_t* a,int n) { libero_vec::fill_serial(a,n,REAL_C(0.0)); }
__device__ LIBERO_POSITION_MASS_INLINE bool finite(const real_t* a,int n) {
    return !libero_vec::any_nonfinite_lane(a,n,0,1);
}

template<int Kind>
__device__ __forceinline__ void compos_joint(int j,const real_t* com,const real_t* anchor,const real_t* xmat,const real_t* axis,const int* root,const int* body,const int* dof,real_t* cv,int nb,int nv,bool planes){
 const int b=body[j],base=dof[j]*6;
 real_t off[3];for(int c=0;c<3;++c)off[c]=com[slot(root[b],c,nb,3,planes)]-anchor[3*j+c];
 if constexpr(Kind==0){cv[slot(dof[j],3,nv,6,planes)]=REAL_C(1.);cv[slot(dof[j]+1,4,nv,6,planes)]=REAL_C(1.);cv[slot(dof[j]+2,5,nv,6,planes)]=REAL_C(1.);}
 if constexpr(Kind==2){for(int c=0;c<3;++c)cv[slot(dof[j],3+c,nv,6,planes)]=axis[3*j+c];}
 else {
  constexpr int n=(Kind==0||Kind==1)?3:1,skip=Kind==0?18:0;
  for(int k=0;k<n;++k){
   real_t a[3];for(int c=0;c<3;++c)a[c]=(Kind==0||Kind==1)?xmat[9*b+3*c+k]:axis[3*j+c];
   real_t values[6];real_t* v=values;for(int c=0;c<3;++c)v[c]=a[c];
   v[3]=a[1]*off[2]-a[2]*off[1];v[4]=a[2]*off[0]-a[0]*off[2];v[5]=a[0]*off[1]-a[1]*off[0];
   scatter(cv,v,dof[j]+skip/6+k,nv,6,planes);
  }
 }
}
// Native reverse-body traversal adds siblings in descending original body ID.
// Keep that arithmetic order within each parent while independent parents run.
__device__ __forceinline__ void compos_tree(const real_t* xipos,const real_t* mass,const real_t* subtree_mass,int nb,real_t* com,const int* offsets,const int* ids,int levels,const int* children_offsets,const int* children,bool planes){
 for(int level=levels-1;level>=0;--level){
  for(int at=offsets[level]+threadIdx.x;at<offsets[level+1];at+=blockDim.x){
   const int b=ids[at];
   for(int c=0;c<3;++c){real_t sum=REAL_C(0.);for(int k=children_offsets[b];k<children_offsets[b+1];++k)sum+=com[slot(children[k],c,nb,3,planes)];sum+=mass[b]*xipos[3*b+c];com[slot(b,c,nb,3,planes)]=sum;}
  }
  __syncthreads();
 }
 if(threadIdx.x==0)for(int c=0;c<3;++c){real_t sum=REAL_C(0.);for(int k=children_offsets[0];k<children_offsets[1];++k)sum+=com[slot(children[k],c,nb,3,planes)];sum+=mass[0]*xipos[c];com[slot(0,c,nb,3,planes)]=sum;}
 __syncthreads();
 for(int b=threadIdx.x;b<nb;b+=blockDim.x)for(int c=0;c<3;++c)com[slot(b,c,nb,3,planes)]=subtree_mass[b]<REAL_C(1e-15)?xipos[3*b+c]:com[slot(b,c,nb,3,planes)]/subtree_mass[b];
 __syncthreads();
}

__device__ LIBERO_POSITION_MASS_INLINE void compos(int w,const int* active,const real_t* xipos,
    const real_t* ximat,const real_t* anchor,const real_t* xmat,const real_t* axis,
    const real_t* mass,const real_t* subtree_mass,const real_t* inertia,
    const int* parent,const int* root,const int* type,const int* body,const int* dof,
    int nb,int nj,int nv,real_t* com,real_t* ci,real_t* cv,int* status,
    const int* joint_offsets=nullptr,const int* joint_indices=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int level_count=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,bool component_planes=false) {
    xipos+=w*nb*3; ximat+=w*nb*9; anchor+=w*nj*3;
    xmat+=w*nb*9; axis+=w*nj*3;
    com+=w*nb*3;ci+=w*nb*10;cv+=w*nv*6;
    if(threadIdx.x==0)status[w]=0;
    zero_block(com,nb*3);zero_block(ci,nb*10);zero_block(cv,nv*6);
    __syncthreads();
    if(active[w]==0)return;
    if(active[w]!=1){if(threadIdx.x==0)status[w]=1;__syncthreads();return;}
    const bool bad_input=!finite_lane_known<CF_PM_STATIC_NBODY*3>(xipos,nb*3)|!finite_lane_known<CF_PM_STATIC_NBODY*9>(ximat,nb*9)|
        !finite_lane_known<CF_PM_STATIC_NBODY*9>(xmat,nb*9)|!finite_lane_known<CF_PM_STATIC_NJNT*3>(anchor,nj*3)|!finite_lane_known<CF_PM_STATIC_NJNT*3>(axis,nj*3);
    if(invalid_block(bad_input)){if(threadIdx.x==0)status[w]=2;__syncthreads();return;}
    if(level_offsets)compos_tree(xipos,mass,subtree_mass,nb,com,level_offsets,level_bodies,level_count,children_offsets,children_indices,component_planes);
    else if(threadIdx.x==0)for(int i=nb-1;i>=0;--i){
        for(int c=0;c<3;++c)com[slot(i,c,nb,3,component_planes)]+=mass[i]*xipos[3*i+c];
        if(i)for(int c=0;c<3;++c)com[slot(parent[i],c,nb,3,component_planes)]+=com[slot(i,c,nb,3,component_planes)];
        for(int c=0;c<3;++c)com[slot(i,c,nb,3,component_planes)]=subtree_mass[i]<REAL_C(1e-15)?
            xipos[3*i+c]:com[slot(i,c,nb,3,component_planes)]/subtree_mass[i];
    }
    __syncthreads();
    const int row[6]={0,1,2,0,0,1},col[6]={0,1,2,1,2,2};
    for(int i=1+threadIdx.x;i<nb;i+=blockDim.x){
        const real_t* R=ximat+9*i;real_t values[10];real_t* I=values;
        real_t d[3];for(int c=0;c<3;++c)d[c]=xipos[3*i+c]-com[slot(root[i],c,nb,3,component_planes)];
        for(int c=0;c<6;++c){
            int r=row[c],s=col[c];
            I[c]=R[3*r]*inertia[3*i]*R[3*s]+
                 R[3*r+1]*inertia[3*i+1]*R[3*s+1]+
                 R[3*r+2]*inertia[3*i+2]*R[3*s+2];
        }
        real_t m=mass[i];
        I[0]+=m*(d[1]*d[1]+d[2]*d[2]);
        I[1]+=m*(d[0]*d[0]+d[2]*d[2]);
        I[2]+=m*(d[0]*d[0]+d[1]*d[1]);
        I[3]-=m*d[0]*d[1];I[4]-=m*d[0]*d[2];I[5]-=m*d[1]*d[2];
        for(int c=0;c<3;++c)I[6+c]=m*d[c];I[9]=m;scatter(ci,I,i,nb,10,component_planes);
    }
    if(joint_offsets){
     for(int at=joint_offsets[0]+threadIdx.x;at<joint_offsets[1];at+=blockDim.x)compos_joint<0>(joint_indices[at],com,anchor,xmat,axis,root,body,dof,cv,nb,nv,component_planes);
     for(int at=joint_offsets[1]+threadIdx.x;at<joint_offsets[2];at+=blockDim.x)compos_joint<1>(joint_indices[at],com,anchor,xmat,axis,root,body,dof,cv,nb,nv,component_planes);
     for(int at=joint_offsets[2]+threadIdx.x;at<joint_offsets[3];at+=blockDim.x)compos_joint<2>(joint_indices[at],com,anchor,xmat,axis,root,body,dof,cv,nb,nv,component_planes);
     for(int at=joint_offsets[3]+threadIdx.x;at<joint_offsets[4];at+=blockDim.x)compos_joint<3>(joint_indices[at],com,anchor,xmat,axis,root,body,dof,cv,nb,nv,component_planes);
    }else {
    for(int j=threadIdx.x;j<nj;j+=blockDim.x){
        int b=body[j],t=type[j],base=dof[j]*6,skip=0;
        real_t off[3];for(int c=0;c<3;++c)off[c]=com[slot(root[b],c,nb,3,component_planes)]-anchor[3*j+c];
        if(t==0){cv[slot(dof[j],3,nv,6,component_planes)]=REAL_C(1.);cv[slot(dof[j]+1,4,nv,6,component_planes)]=REAL_C(1.);cv[slot(dof[j]+2,5,nv,6,component_planes)]=REAL_C(1.);skip=18;}
        if(t==2){for(int c=0;c<3;++c)cv[slot(dof[j],3+c,nv,6,component_planes)]=axis[3*j+c];continue;}
        int n=(t==0||t==1)?3:1;
        for(int k=0;k<n;++k){
            real_t a[3];for(int c=0;c<3;++c)a[c]=(t==0||t==1)?xmat[9*b+3*c+k]:axis[3*j+c];
            real_t values[6];real_t* v=values;
            for(int c=0;c<3;++c)v[c]=a[c];
            v[3]=a[1]*off[2]-a[2]*off[1];v[4]=a[2]*off[0]-a[0]*off[2];v[5]=a[0]*off[1]-a[1]*off[0];scatter(cv,v,dof[j]+skip/6+k,nv,6,component_planes);
        }
    }
    }
    __syncthreads();
    const bool bad_output=!finite_lane_known<CF_PM_STATIC_NBODY*3>(com,nb*3)|!finite_lane_known<CF_PM_STATIC_NBODY*10>(ci,nb*10)|!finite_lane_known<CF_PM_STATIC_NV*6>(cv,nv*6);
    if(invalid_block(bad_output)){
        if(threadIdx.x==0)status[w]=3;
        zero_block(com,nb*3);zero_block(ci,nb*10);zero_block(cv,nv*6);
    }
    __syncthreads();
}
__device__ LIBERO_POSITION_MASS_INLINE void crb(int w,const int* active,const real_t* ci,const real_t* cv,
    const int* bp,const int* db,const int* dp,const int* ma,const int* simple,
    const real_t* m0,const real_t* arm,int nb,int nv,int nm,real_t* cr,real_t* qm,int* status,
    const int* chain_offsets=nullptr,const int* chain_indices=nullptr,const int* chain_packed=nullptr,const int* simple_offsets=nullptr,const int* simple_indices=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int level_count=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,bool component_planes=false){
    ci+=w*nb*10;cv+=w*nv*6;cr+=w*nb*10;qm+=w*nm;
    if(threadIdx.x==0)status[w]=0;
    zero_block(cr,nb*10);zero_block(qm,nm);__syncthreads();
    if(active[w]==0)return;
    if(active[w]!=1){if(threadIdx.x==0)status[w]=1;__syncthreads();return;}
    if(invalid_block(!finite_lane_known<CF_PM_STATIC_NBODY*10>(ci,nb*10)|!finite_lane_known<CF_PM_STATIC_NV*6>(cv,nv*6))){
        if(threadIdx.x==0)status[w]=2;__syncthreads();return;
    }
    libero_vec::copy_block(ci,cr,nb*10);
    __syncthreads();
    if(level_offsets){
     for(int level=level_count-1;level>=0;--level){
      for(int at=level_offsets[level]+threadIdx.x;at<level_offsets[level+1];at+=blockDim.x){const int b=level_bodies[at];
       for(int child=children_offsets[b];child<children_offsets[b+1];++child)for(int k=0;k<10;++k)cr[slot(b,k,nb,10,component_planes)]+=cr[slot(children_indices[child],k,nb,10,component_planes)];
      }
      __syncthreads();
     }
    }else {
    // CRB tree order is unchanged. Packed mass rows have disjoint addresses.
    if(threadIdx.x==0)for(int b=nb-1;b>0;--b)if(bp[b]>0)
        for(int k=0;k<10;++k)cr[slot(bp[b],k,nb,10,component_planes)]+=cr[slot(b,k,nb,10,component_planes)];
    }
    __syncthreads();
    if(simple_offsets)for(int at=simple_offsets[1]+threadIdx.x;at<simple_offsets[2];at+=blockDim.x){const int i=simple_indices[at];qm[ma[i]]=m0[i];}
    const int row_count=simple_offsets?simple_offsets[1]-simple_offsets[0]:nv;
    for(int at=threadIdx.x;at<row_count;at+=blockDim.x){
        const int i=simple_offsets?simple_indices[simple_offsets[0]+at]:nv-1-at;
        int adr=ma[i];if(!simple_offsets&&simple[i]){qm[adr]=m0[i];continue;}
        qm[adr]=arm[i];real_t inertia_value[10],dof_value[6];gather(cr,inertia_value,db[i],nb,10,component_planes);gather(cv,dof_value,i,nv,6,component_planes);const real_t* I=inertia_value;const real_t* v=dof_value;
        real_t f[6]={I[0]*v[0]+I[3]*v[1]+I[4]*v[2]-I[8]*v[4]+I[7]*v[5],
          I[3]*v[0]+I[1]*v[1]+I[5]*v[2]+I[8]*v[3]-I[6]*v[5],
          I[4]*v[0]+I[5]*v[1]+I[2]*v[2]-I[7]*v[3]+I[6]*v[4],
          I[8]*v[1]-I[7]*v[2]+I[9]*v[3],I[6]*v[2]-I[8]*v[0]+I[9]*v[4],
          I[7]*v[0]-I[6]*v[1]+I[9]*v[5]};
        if(chain_offsets){
         for(int at=chain_offsets[i];at<chain_offsets[i+1];++at){real_t value[6];gather(cv,value,chain_indices[at],nv,6,component_planes);const real_t* u=value;real_t sum=REAL_C(0.);for(int c=0;c<6;++c)sum+=u[c]*f[c];qm[chain_packed[at]]+=sum;}
        }else {
        for(int j=i;j>=0;j=dp[j],++adr){
            real_t value[6];gather(cv,value,j,nv,6,component_planes);const real_t* u=value;real_t sum=REAL_C(0.);for(int c=0;c<6;++c)sum+=u[c]*f[c];qm[adr]+=sum;
        }
        }
    }
    __syncthreads();
    if(invalid_block(!finite_lane_known<CF_PM_STATIC_NBODY*10>(cr,nb*10)|!finite_lane_known<CF_PM_STATIC_NM>(qm,nm))){
        if(threadIdx.x==0)status[w]=3;zero_block(cr,nb*10);zero_block(qm,nm);
    }
    __syncthreads();
}
__device__ LIBERO_POSITION_MASS_INLINE void gate1(int w,const int* active,const int* status,int* next,int* stage,int* code){
    next[w]=0;stage[w]=0;code[w]=0;if(!active[w])return;
    if(status[w]){stage[w]=1;code[w]=status[w];}else next[w]=1;
}
__device__ LIBERO_POSITION_MASS_INLINE void gate2(int w,const int* active,const int* status,int* next,int* stage,int* code){
    next[w]=0;if(!active[w])return;if(status[w]){stage[w]=2;code[w]=status[w];}else next[w]=1;
}
__device__ LIBERO_POSITION_MASS_INLINE void clear_factor(int nv,int nm,real_t* ld,real_t* inv,real_t* sinv,real_t* sol){
    zero(ld,nm);zero(inv,nv);zero(sinv,nv);zero(sol,4*nv);
}
// This is sparse tree LDL^T, not a dense LAPACK POTRF operation.
__device__ LIBERO_POSITION_MASS_INLINE void factor_solve(int w,const real_t* qm,const real_t* rhs,const int* count,
    const int* active,const int* ma,const int* parent,real_t* ld,real_t* inv,real_t* sinv,
    real_t* sol,int* pivot,int* status,int* warnings,int* last,int nv,int nm){
    qm+=w*nm;rhs+=w*4*nv;ld+=w*nm;inv+=w*nv;sinv+=w*nv;sol+=w*4*nv;pivot+=w*nv;
    status[w]=0;warnings[w]=0;last[w]=-1;clear_factor(nv,nm,ld,inv,sinv,sol);
    for(int i=0;i<nv;++i)pivot[i]=0;
    if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}
    int n=count[w];if(n<1||n>4){status[w]=2;return;}
    if(!finite(qm,nm)||!finite(rhs,4*nv)){status[w]=3;return;}
    libero_vec::copy_serial(qm,ld,nm);bool bad=false;
    for(int k=nv-1;k>=0;--k){
        int kk=ma[k];if(ld[kk]<REAL_C(1e-15)){pivot[k]=1;++warnings[w];last[w]=k;ld[kk]=REAL_C(1e-15);}
        int ki=kk+1;
        for(int i=parent[k];i>=0;i=parent[i],++ki){
            real_t t=ld[ki]/ld[kk];bad|=!isfinite(t);
            int len=(i+1<nv?ma[i+1]:nm)-ma[i];
            for(int a=0;a<len;++a){ld[ma[i]+a]-=ld[ki+a]*t;bad|=!isfinite(ld[ma[i]+a]);}
            ld[ki]=t;
        }
    }
    for(int i=0;i<nv;++i){inv[i]=REAL_C(1.)/ld[ma[i]];sinv[i]=REAL_C(1.)/sqrt(ld[ma[i]]);bad|=!isfinite(inv[i])||!isfinite(sinv[i]);}
    for(int r=0;r<n;++r){
        real_t* x=sol+r*nv;libero_vec::copy_serial(rhs+r*nv,x,nv);
        for(int i=nv-1;i>=0;--i){if(x[i]==REAL_C(0.))continue;int a=ma[i]+1;
            for(int j=parent[i];j>=0;j=parent[j],++a){x[j]-=ld[a]*x[i];bad|=!isfinite(x[j]);}}
        for(int i=0;i<nv;++i){x[i]*=inv[i];bad|=!isfinite(x[i]);}
        for(int i=0;i<nv;++i){int a=ma[i]+1;for(int j=parent[i];j>=0;j=parent[j],++a){x[i]-=ld[a]*x[j];bad|=!isfinite(x[i]);}}
    }
    if(bad){clear_factor(nv,nm,ld,inv,sinv,sol);status[w]=4;}else if(warnings[w])status[w]=5;
}
__device__ LIBERO_POSITION_MASS_INLINE void finish(int w,const int* active,const int* next,const int* status,int* stage,int* code,int* kind){
    kind[w]=0;if(!active[w])return;if(!next[w]){kind[w]=3;return;}
    int c=status[w];if(c==0)kind[w]=1;else if(c==5)kind[w]=2;else{stage[w]=3;code[w]=c;kind[w]=3;}
}
} // namespace cf_pm


#ifdef LIBERO_POSITION_DATAFLOW
// Same public wrapper as the legacy packed solver. Scratch qM/LD/solution
// belongs to this block; the caller publishes it once after this phase.
namespace cf_pm {
__device__ __forceinline__ void factor_solve_shared(int w,const real_t* qm,
    const real_t* rhs,const int* count,const int* active,const int* ma,
    const int* parent,real_t* ld,real_t* inv,real_t* sinv,real_t* sol,
    int* pivot,int* status,int* warnings,int* last,int nv,int nm) {
    qm+=w*nm;rhs+=w*4*nv;ld+=w*nm;inv+=w*nv;sinv+=w*nv;
    sol+=w*4*nv;pivot+=w*nv;
    if(threadIdx.x==0){status[w]=0;warnings[w]=0;last[w]=-1;}
    zero_block(ld,nm);zero_block(inv,nv);zero_block(sinv,nv);zero_block(sol,4*nv);
    for(int i=threadIdx.x;i<nv;i+=blockDim.x)pivot[i]=0;
    __syncthreads();
    if(active[w]==0)return;
    if(active[w]!=1){if(threadIdx.x==0)status[w]=1;__syncthreads();return;}
    const int n=count[w];
    if(n<1||n>4){if(threadIdx.x==0)status[w]=2;__syncthreads();return;}
    // Validate every RHS slot, including inactive RHS rows, as the old gate did.
    if(invalid_block(!finite_lane_known<CF_PM_STATIC_NM>(qm,nm)|!finite_lane_known<CF_PM_STATIC_NV*4>(rhs,4*nv))){
        if(threadIdx.x==0)status[w]=3;__syncthreads();return;
    }
    libero_packed_mass::ForestPlan forest;
    const bool factor_ok=libero_packed_mass::factor_block(qm,ld,inv,sinv,pivot,
        warnings[w],last[w],nv,nm,ma,parent,forest);
    // Independent roots and RHS share the completed factor. Even a failed
    // factor attempts every RHS before the original final clear gate.
    const bool bad=!libero_packed_mass::solve_rhs_block(rhs,sol,ld,inv,n,nv,ma,parent,forest);
    if(threadIdx.x==0){
     if(!factor_ok||bad){clear_factor(nv,nm,ld,inv,sinv,sol);status[w]=4;}
     else if(warnings[w])status[w]=5;
    }
    __syncthreads();
}
}
#endif
