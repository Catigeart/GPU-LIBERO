// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// CUDA adaptation of pinned MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf:
// forward.py::_qfrc_smooth(False), support.py::_apply_ft(flg_add=True).
#include "mjwarp_smooth_force.cuh"
#ifndef FLASH_MJW_SMOOTH_FORCE_SOURCE_ID
#error "Build through flash_mjwarp_smooth_force_fp32"
#endif
namespace flash::mjwarp {
namespace {
__device__ float external_projection(const SmoothForceInputs& a, int w, int dof) {
  const float* cdof = a.cdof + (w * a.nv + dof) * 6;
  // cdof is (angular, linear); applied wrench is (force, torque), in world axes.
  float jac[6] = {cdof[3], cdof[4], cdof[5], cdof[0], cdof[1], cdof[2]};
  int body = a.dof_body[dof];
  const float* com = a.subtree_com + (w * a.bodies + a.body_root[body]) * 3;
  float sum = 0.0f;
  // Model's depth-first order makes each subtree a contiguous body interval.
  for (int b = body; b < body + a.subtree_size[body]; ++b) {
    const float* ft = a.external + (w * a.bodies + b) * 6;
    if (ft[0] == 0 && ft[1] == 0 && ft[2] == 0 && ft[3] == 0 && ft[4] == 0 && ft[5] == 0)
      continue;
    const float* point = a.body_com + (w * a.bodies + b) * 3;
    float dx = point[0] - com[0], dy = point[1] - com[1], dz = point[2] - com[2];
    float cx = cdof[1] * dz - cdof[2] * dy;
    float cy = cdof[2] * dx - cdof[0] * dz;
    float cz = cdof[0] * dy - cdof[1] * dx;
    float dot6 = 0.0f;
    for (int k = 0; k < 6; ++k) dot6 += jac[k] * ft[k];
    float dot3 = cx * ft[0] + cy * ft[1] + cz * ft[2];
    sum += dot6 + dot3;
  }
  return sum;
}

template<int Mode>
__global__ void smooth_force(SmoothForceInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.nv) return;
  int w = t / a.nv, dof = t % a.nv, at = w * a.vector_stride + dof;
  float value;
  if constexpr (Mode == 1) value = a.smooth[at];
  else value = a.passive[at] - a.bias[at] + a.actuator[at] + a.applied[at];
  if constexpr (Mode != 0) value += external_projection(a, w, dof);
  a.smooth[at] = value;
}
}
}
extern "C" {
const char* flash_mjwarp_smooth_force_source_id() { return FLASH_MJW_SMOOTH_FORCE_SOURCE_ID; }
std::size_t flash_mjwarp_smooth_force_arguments_bytes() { return sizeof(flash::mjwarp::SmoothForceInputs); }
int flash_mjwarp_smooth_force_arch() { return FLASH_MJW_SMOOTH_FORCE_ARCH; }
int flash_mjwarp_smooth_force_launch(const flash::mjwarp::SmoothForceInputs* a, int mode, cudaStream_t stream) {
  using namespace flash::mjwarp;
  if (!a || a->worlds < 1 || a->nv < 1 || a->bodies < 1) return int(cudaErrorInvalidValue);
  int blocks = (a->worlds * a->nv + 63) / 64;
  if (mode == 0) smooth_force<0><<<blocks, 64, 0, stream>>>(*a);
  else if (mode == 1) smooth_force<1><<<blocks, 64, 0, stream>>>(*a);
  else if (mode == 2) smooth_force<2><<<blocks, 64, 0, stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
