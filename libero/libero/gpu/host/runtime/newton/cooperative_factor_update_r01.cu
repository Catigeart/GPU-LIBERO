#include "precision.cuh"
// Independent diagnostic candidate: complete S05 factor_update body.
// validate/assemble remain original. No admission/integration granted here.
// Exactly one 64-thread block/world, nv<=57. Columns, contact rows, rank
// updates and fallback branches remain strictly ordered; only disjoint cells
// inside a column or an original cellwise copy/assembly are distributed.
#include <cuda_runtime.h>

__device__ real_t n_dot(const real_t* a,int p,int q,int n) {
  real_t s0=REAL_C(0.0),s1=REAL_C(0.0),s2=REAL_C(0.0),s3=REAL_C(0.0);int k=0;
  if(n>=4){s0=real_mul_rn(a[p],a[q]);s1=real_mul_rn(a[p+1],a[q+1]);
    s2=real_mul_rn(a[p+2],a[q+2]);s3=real_mul_rn(a[p+3],a[q+3]);k=4;}
  while(k+4<=n){s0=real_add_rn(s0,real_mul_rn(a[p+k],a[q+k]));
    s1=real_add_rn(s1,real_mul_rn(a[p+k+1],a[q+k+1]));
    s2=real_add_rn(s2,real_mul_rn(a[p+k+2],a[q+k+2]));
    s3=real_add_rn(s3,real_mul_rn(a[p+k+3],a[q+k+3]));k+=4;}
  real_t out=real_add_rn(real_add_rn(s0,s2),real_add_rn(s1,s3));
  if(n-k==3)out=real_add_rn(out,real_add_rn(real_add_rn(real_mul_rn(a[p+k],a[q+k]),real_mul_rn(a[p+k+1],a[q+k+1])),real_mul_rn(a[p+k+2],a[q+k+2])));
  else if(n-k==2)out=real_add_rn(out,real_add_rn(real_mul_rn(a[p+k],a[q+k]),real_mul_rn(a[p+k+1],a[q+k+1])));
  else if(n-k==1)out=real_add_rn(out,real_mul_rn(a[p+k],a[q+k]));
  return out;
}
__device__ int n_finite(real_t x){union{real_t d;unsigned long long u;}b;b.d=x;return (b.u&0x7ff0000000000000ULL)!=0x7ff0000000000000ULL;}
struct NShared {real_t inv,c,cinv,s;int rank,nonzero,lost,fail;};
__device__ int n_factor(real_t* a,int base,int n,int stride,NShared* sh) {
  int lane=threadIdx.x;if(lane==0)sh->rank=n;__syncthreads();
  for(int j=0;j<n;++j){
    if(lane==0){real_t diag=a[base+j*stride+j];
      if(j>0)diag=real_sub_rn(diag,n_dot(a,base+j*stride,base+j*stride,j));
      if(diag<REAL_C(1.e-15)){diag=REAL_C(1.e-15);--sh->rank;}
      a[base+j*stride+j]=real_sqrt_rn(diag);sh->inv=real_div_rn(REAL_C(1.0),a[base+j*stride+j]);}
    __syncthreads();
    if(lane>j&&lane<n)a[base+lane*stride+j]=real_mul_rn(real_sub_rn(a[base+lane*stride+j],n_dot(a,base+lane*stride,base+j*stride,j)),sh->inv);
    __syncthreads();
  }return sh->rank;
}
__device__ int n_rank(real_t* a,int base,real_t* vec,int vb,int plus,int nv,NShared* sh){
  int lane=threadIdx.x;if(lane==0)sh->rank=nv;__syncthreads();
  for(int k=0;k<nv;++k){
    if(lane==0){real_t x=vec[vb+k];sh->nonzero=(x!=REAL_C(0.0));
      if(sh->nonzero){real_t Lkk=a[base+k*nv+k];real_t tmp=real_mul_rn(Lkk,Lkk);real_t xx=real_mul_rn(x,x);
        tmp=plus!=0?real_add_rn(tmp,xx):real_sub_rn(tmp,xx);
        if(tmp<REAL_C(1.e-15)){tmp=REAL_C(1.e-15);--sh->rank;}
        real_t r=real_sqrt_rn(tmp);sh->c=real_div_rn(r,Lkk);sh->cinv=real_div_rn(REAL_C(1.0),sh->c);sh->s=real_div_rn(x,Lkk);a[base+k*nv+k]=r;}}
    __syncthreads();
    if(sh->nonzero&&lane>k&&lane<nv){int adr=base+lane*nv+k;
      real_t sv=real_mul_rn(sh->s,vec[vb+lane]);real_t value=plus!=0?real_add_rn(a[adr],sv):real_sub_rn(a[adr],sv);
      a[adr]=real_mul_rn(value,sh->cinv);
      vec[vb+lane]=real_sub_rn(real_mul_rn(sh->c,vec[vb+lane]),real_mul_rn(sh->s,a[adr]));}
    __syncthreads();
  }return sh->rank;
}
__device__ real_t n_cell(int w,int i,int j,int rows,int nv,int nm,const int* count,const real_t* J,const real_t* D,const int* state,const real_t* qm,const int* madr,const int* parent,const int* simple){
  real_t result=REAL_C(0.0);
  for(int r=0;r<count[w];++r)if(state[w*rows+r]==1){real_t diag=D[w*rows+r];
    if(diag!=REAL_C(0.0)){real_t tmp=J[(w*rows+r)*nv+i];if(tmp!=REAL_C(0.0))result=real_add_rn(result,real_mul_rn(J[(w*rows+r)*nv+j],real_mul_rn(tmp,diag)));}}
  int adr=madr[i],k=i,found=-1;
  while(k>=0){if(k==j){found=adr;break;}if(simple[i]!=0)break;k=parent[k];++adr;}
  if(found>=0)result=real_add_rn(result,qm[w*nm+found]);return result;
}
__device__ void n_cone(int w,int rows,int contacts,int nv,const int* count,const int* state,const int* id,const int* dim,const real_t* contactH,const real_t* J,const real_t* H,real_t* Hcone,real_t* local,real_t* ltj,real_t* vec,int* nupdate,NShared* sh){
  int lane=threadIdx.x,hb=w*nv*nv,lb=w*36,tb=w*6*nv,vb=w*nv;
  for(int z=lane;z<nv*nv;z+=64)Hcone[hb+z]=H[hb+z];__syncthreads();
  int i=0;
  while(i<count[w]){
    if(state[w*rows+i]==4){int con=id[w*rows+i],d=dim[w*contacts+con];
      for(int z=lane;z<d*d;z+=64)local[lb+z]=contactH[(w*contacts+con)*36+z];__syncthreads();
      n_factor(local,lb,d,d,sh);
      for(int z=lane;z<d*nv;z+=64)ltj[tb+z]=REAL_C(0.0);__syncthreads();
      // A (c,k) cell receives exactly original r=c..dim-1 additions.
      for(int z=lane;z<d*nv;z+=64){int c=z/nv,k=z%nv;
        for(int r=c;r<d;++r)ltj[tb+z]=real_add_rn(ltj[tb+z],real_mul_rn(J[(w*rows+i+r)*nv+k],local[lb+r*d+c]));}
      __syncthreads();
      for(int r=0;r<d;++r){if(lane<nv)vec[vb+lane]=ltj[tb+r*nv+lane];__syncthreads();n_rank(Hcone,hb,vec,vb,1,nv,sh);}
      if(lane==0)nupdate[w]=nupdate[w]+d;__syncthreads();i+=d;
    }else ++i;
  }
}
extern "C" __global__ void n_cooperative_factor_update_r01(
 const int* count,const int* ncon,const int* active,const int* kind,const real_t* J,const real_t* D,const int* state,const int* oldstate,const real_t* qm,const int* madr,const int* parent,const int* simple,const int* id,const int* dim,const real_t* contactH,const int* ncone,real_t* H,real_t* Hcone,int* nnz,int* nupdate,real_t* local,real_t* ltj,real_t* vec,int* status,int* route,int rows,int contacts,int nv,int nm,int nworld){
  int w=blockIdx.x,lane=threadIdx.x;
  if(w>=nworld||active[w]==0||status[w]!=0)return;
  __shared__ NShared sh;
  int hb=w*nv*nv,vb=w*nv;
  if(kind[w]==1){
    n_factor(H,hb,nv,nv,&sh);if(lane==0)nnz[w]=nv*nv;__syncthreads();
    if(ncone[w]!=0){if(lane==0)nupdate[w]=0;__syncthreads();n_cone(w,rows,contacts,nv,count,state,id,dim,contactH,J,H,Hcone,local,ltj,vec,nupdate,&sh);if(lane==0)route[w]=2;}
    else if(lane==0)route[w]=1;
    if(lane==0)nupdate[w]=count[w];__syncthreads();
  }else if(kind[w]==2){
    n_cone(w,rows,contacts,nv,count,state,id,dim,contactH,J,H,Hcone,local,ltj,vec,nupdate,&sh);if(lane==0)route[w]=6;__syncthreads();
  }else{
    if(lane==0){nupdate[w]=0;sh.lost=0;}__syncthreads();
    for(int r=0;r<count[w];++r){int before=oldstate[w*rows+r]==1,after=state[w*rows+r]==1;
      if(before!=after){real_t scale=real_sqrt_rn(D[w*rows+r]);if(lane<nv)vec[vb+lane]=real_mul_rn(J[(w*rows+r)*nv+lane],scale);__syncthreads();
        int rank=n_rank(H,hb,vec,vb,after?1:0,nv,&sh);
        if(lane==0){++nupdate[w];if(rank<nv)sh.lost=1;}__syncthreads();if(sh.lost)break;}}
    if(sh.lost){
      // Preserve original fallback first error, before any assembly write.
      if(lane==0){sh.fail=0;
        for(int z=0;z<nm;++z)if(!n_finite(qm[w*nm+z])){sh.fail=1;break;}
        if(!sh.fail)for(int r=0;r<count[w];++r)if(state[w*rows+r]==1){
          if(D[w*rows+r]<REAL_C(0.0)||!n_finite(D[w*rows+r])){sh.fail=1;break;}
          if(D[w*rows+r]!=REAL_C(0.0)){for(int k=0;k<nv;++k)if(!n_finite(J[(w*rows+r)*nv+k])){sh.fail=1;break;}if(sh.fail)break;}}
        if(sh.fail)status[w]=4;}
      __syncthreads();if(sh.fail)return;
      for(int z=lane;z<nv*nv;z+=64){int i=z/nv,j=z%nv;if(j<=i){real_t value=n_cell(w,i,j,rows,nv,nm,count,J,D,state,qm,madr,parent,simple);H[hb+z]=value;if(i!=j)H[hb+j*nv+i]=value;}}
      __syncthreads();n_factor(H,hb,nv,nv,&sh);if(lane==0)nnz[w]=nv*nv;__syncthreads();
      if(ncone[w]!=0){if(lane==0)nupdate[w]=0;__syncthreads();n_cone(w,rows,contacts,nv,count,state,id,dim,contactH,J,H,Hcone,local,ltj,vec,nupdate,&sh);}
      if(lane==0){nupdate[w]=count[w];route[w]=5;}__syncthreads();
    }else{
      if(ncone[w]!=0){n_cone(w,rows,contacts,nv,count,state,id,dim,contactH,J,H,Hcone,local,ltj,vec,nupdate,&sh);if(lane==0)route[w]=4;}
      else if(lane==0)route[w]=3;__syncthreads();
    }
  }
  // Keep original scalar z first-error traversal, no global status reduction.
  if(lane==0)for(int z=0;z<nv*nv;++z){int checkH=kind[w]==1||route[w]==5;
    if(kind[w]==3&&route[w]!=5&&z/nv>=z%nv)checkH=1;
    if(checkH&&!n_finite(H[hb+z])){status[w]=5;return;}
    if(ncone[w]!=0&&!n_finite(Hcone[hb+z])){status[w]=5;return;}}
}
