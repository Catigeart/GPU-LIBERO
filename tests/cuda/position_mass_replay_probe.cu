// Regional production CoM/CRB replay; no independent reference arithmetic here.
#include "precision.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../common/packed_mass.cuh"
#include "../position/position_mass.cuh"
struct PositionMassReplay {
  const int *active;
  const real_t *xipos, *ximat, *anchor, *xmat, *axis;
  const real_t *mass, *submass, *inertia;
  const int *parent, *root, *jt, *jb, *jd;
  const int *db, *dp, *ma, *simple;
  const real_t *m0, *arm;
  const int *levels, *bodies, *children_offsets, *children, *subtrees;
  const int *joint_offsets, *joint_indices;
  real_t *com, *ci, *cv, *cr, *qm;
  int *com_status, *cr_status, *cr_active;
  int nb, nj, nv, nm, nlevels;
};
namespace {
template<bool Subtrees>
__global__ void replay(PositionMassReplay a) {
  const int w=blockIdx.y;
  extern __shared__ real_t scratch[];
  real_t *com=scratch, *ci=com+3*a.nb, *cv=ci+10*a.nb;
  real_t *cr=cv+6*a.nv, *qm=cr+10*a.nb;
  cf_pm::compos(0,a.active+w,a.xipos+w*3*a.nb,a.ximat+w*9*a.nb,
    a.anchor+w*3*a.nj,a.xmat+w*9*a.nb,a.axis+w*3*a.nj,
    a.mass,a.submass,a.inertia,a.parent,a.root,a.jt,a.jb,a.jd,a.nb,a.nj,a.nv,
    com,ci,cv,a.com_status+w,a.joint_offsets,a.joint_indices,a.levels,a.bodies,
    a.nlevels,a.children_offsets,a.children,true,Subtrees?a.subtrees:nullptr);
  if(threadIdx.x==0) a.cr_active[w]=a.active[w]==1 && a.com_status[w]==0;
  __syncthreads();
  cf_pm::crb(0,a.cr_active+w,ci,cv,a.parent,a.db,a.dp,a.ma,a.simple,a.m0,a.arm,
    a.nb,a.nv,a.nm,cr,qm,a.cr_status+w,nullptr,nullptr,nullptr,nullptr,nullptr,
    a.levels,a.bodies,a.nlevels,a.children_offsets,a.children,true,
    Subtrees?a.subtrees:nullptr);
  libero_vec::copy_block(com,a.com+w*3*a.nb,3*a.nb);
  libero_vec::copy_block(ci,a.ci+w*10*a.nb,10*a.nb);
  libero_vec::copy_block(cv,a.cv+w*6*a.nv,6*a.nv);
  libero_vec::copy_block(cr,a.cr+w*10*a.nb,10*a.nb);
  libero_vec::copy_block(qm,a.qm+w*a.nm,a.nm);
}
}
extern "C" size_t flash_position_mass_replay_arguments_bytes(){return sizeof(PositionMassReplay);}
extern "C" int flash_position_mass_replay_configure(int candidate,int *resources){
  cudaFuncAttributes a;
  auto e=cudaFuncGetAttributes(&a,candidate?replay<true>:replay<false>);
  if(e!=cudaSuccess)return int(e);
  resources[0]=a.numRegs;resources[1]=a.sharedSizeBytes;resources[2]=a.localSizeBytes;return 0;
}
extern "C" int flash_position_mass_replay_launch(const PositionMassReplay *a,int worlds,
    int candidate,int threads,cudaStream_t stream){
  if(!a||worlds<1||(threads!=64&&threads!=128&&threads!=256))return int(cudaErrorInvalidValue);
  const size_t bytes=sizeof(real_t)*(23*a->nb+6*a->nv+a->nm);
  if(candidate)replay<true><<<dim3(1,worlds),threads,bytes,stream>>>(*a);
  else replay<false><<<dim3(1,worlds),threads,bytes,stream>>>(*a);
  return int(cudaGetLastError());
}
