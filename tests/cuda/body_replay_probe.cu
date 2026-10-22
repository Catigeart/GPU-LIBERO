// Actual body_block scheduler and body_one arithmetic, with all ABI owners.
#include "precision.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#define LIBERO_POSITION_BODY_DIAGNOSTICS 0
#include "../position/position.cuh"
using namespace libero_position;
struct BodyReplay {
  BodyInputs in; BodyOutputs out;
  real_t *qpos,*mocap_quat,*world_xipos,*world_ximat,*norm_before,*mocap_before;
  int *active,*status,*norm_branch,*mocap_branch;
  const int *levels,*bodies;
  int nq,nb,nj,nm,nlevels;
};
namespace {
template<bool Private> __global__ void replay(BodyReplay a) {
 const int w=blockIdx.y;
 __shared__ real_t trig[128];
 const auto &i=a.in;const auto &o=a.out;
 body_block(w,i.qpos0,i.parent,i.jnum,i.jadr,i.mocapid,i.bpos,i.bquat,
   i.jtype,i.qadr,i.jaxis,i.jpos,a.nq,a.nb,a.nj,a.nm,
   a.active,a.qpos,i.mocap_pos,a.mocap_quat,i.qtmp,i.mqtmp,
   o.xpos,o.xquat,o.xmat,o.xanchor,o.xaxis,a.world_xipos,a.world_ximat,a.status,
   a.norm_before,a.mocap_before,o.pre_pos,o.pre_quat,o.joint_pos,o.joint_quat,
   o.body_norm_before,o.body_route,o.joint_route,o.body_mat_identity,
   o.joint_rot_identity,o.hinge_zero,o.offcenter_rot_identity,o.trig_arg,o.trig_sin,
   o.trig_cos,a.norm_branch,a.mocap_branch,o.body_norm_branch,
   a.levels,a.bodies,a.nlevels,Private,Private?trig:nullptr);
}
}
extern "C" size_t flash_body_replay_size(){return sizeof(BodyReplay);}
extern "C" int flash_body_replay_launch(const BodyReplay *a,int worlds,int candidate,
 int threads,cudaStream_t stream) {
 if(!a||worlds<1||a->nb>64||(threads!=64&&threads!=128&&threads!=256))
  return int(cudaErrorInvalidValue);
 if(candidate)replay<true><<<dim3(1,worlds),threads,0,stream>>>(*a);
 else replay<false><<<dim3(1,worlds),threads,0,stream>>>(*a);
 return int(cudaGetLastError());
}
