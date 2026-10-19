// Controller scheduling is independent of scene NV specializations.
// One CTA owns a world. Warp zero gathers; remaining warps seed disjoint
// private channels. Only protocol metadata has a single writer.
#include "controller_pipeline.cuh"
#include "osc_solver.cuh"
#include "osc_profile.cuh"
#include "osc_runtime.cuh"
#include "../step/substep_types.cuh"
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include <cmath>
#include <cstdint>

namespace physics_controller::protocol {
static __device__ void gate(int w, int *active, int *pv, int64_t *written, int64_t *expected, int *first,
                            int *error, int *policy, int step);
}

namespace physics_controller::protocol {
static __device__ void commit_mask(int w, int *active, int *error, int *first, int *commit, int64_t *event,
                                   int *valid);
}

namespace physics_controller::protocol {
static __device__ void publish(int w, int *mask, int64_t *event, int64_t *parent, int64_t *written,
                               int64_t *consumed, int *valid);
}

namespace physics_controller_bias::cf_osc_gather {
static __device__ void live_bias(int w, const int32_t *mask, const int32_t *first, const int64_t *state,
                                 const int64_t *source, const int64_t *epoch, const int32_t *valid,
                                 const int64_t *written, const double *bias, double *selected,
                                 const int32_t *indices, int32_t *parent_valid, int64_t *parent_written,
                                 int nv);
}

namespace physics_controller_gather::cf_osc_gather {
static __device__ void gate(int w, const int32_t *mask, int32_t *error, const int64_t *state,
                            const int64_t *source, const int64_t *epoch, const int32_t *valid,
                            const int64_t *written, int32_t *active);
}

namespace physics_controller_gather::cf_osc_gather {
static __device__ void site(int w, const double *pos, const double *mat, const int32_t *active, double *outp,
                            double *outm, int ns, int id);
}

namespace physics_controller_gather::cf_osc_gather {
static __device__ void publish_velocity(int w, const int32_t *active, const int32_t *status, const double *vp,
                                        const double *vo, double *target, int32_t *first);
}

namespace physics_controller_gather::cf_osc_gather {
static __device__ void local_selected(int w, const double *cdof, const double *subtree, const double *pos,
                                      const int32_t *active, const int32_t *ancestor, int root, int body,
                                      int nv, int nb, double *jp, double *jr, int32_t *jac_status,
                                      int32_t *adapter_active, const double *qpos, const double *qvel,
                                      const double *rot, const double *qm, const double *qbias,
                                      const int32_t *qi, const int32_t *vi, const int32_t *sparse, int nq,
                                      int nm, double *ep, double *er, double *vp, double *vo, double *q,
                                      double *qd, double *J, double *M, double *bias, int32_t *adapter_status,
                                      int32_t *out_status, const int32_t *body_dof_offsets = nullptr,
                                      const int32_t *body_dof_indices = nullptr, bool trusted = false,
                                      bool parallel_velocity = false);
}

// Count-parametric controller gather, shared by the fast and ordered paths.

// Shared FP64 controller_gather implementation.

namespace physics_controller_gather {

// Count-parametric site Jacobian and seven-joint controller projection.
// jacobian, selected and local_selected require the first complete warp.
// The ordered path uses leader-only metadata helpers; the fast path uses warp-local handoffs.
namespace cf_osc_gather {

static __device__ void jacobian(int w, const double *cdof, const double *subtree, const double *pos,
                                const int32_t *active, const int32_t *ancestor, int root, int body, int nv,
                                int nb, double *jp, double *jr, int32_t *status,
                                const int32_t *body_dof_offsets = nullptr,
                                const int32_t *body_dof_indices = nullptr) {
  if (threadIdx.x >= 32)
    return;
  const int lane = libero_warp::lane();
  double *p = jp + w * 3 * nv;
  double *r = jr + w * 3 * nv;
  if (lane == 0)
    status[w] = 0;
  for (int k = lane; k < 3 * nv; k += 32) {
    p[k] = 0;
    r[k] = 0;
  }
  __syncwarp();
  if (active[w] == 0)
    return;
  if (active[w] != 1) {
    if (lane == 0)
      status[w] = 4;
    return;
  }
  // One subtraction per offset, broadcast without changing arithmetic.
  double ox = 0, oy = 0, oz = 0;
  if (lane == 0) {
    ox = pos[w * 3] - subtree[(w * nb + root) * 3];
    oy = pos[w * 3 + 1] - subtree[(w * nb + root) * 3 + 1];
    oz = pos[w * 3 + 2] - subtree[(w * nb + root) * 3 + 2];
  }
  ox = __shfl_sync(libero_warp::mask, ox, 0);
  oy = __shfl_sync(libero_warp::mask, oy, 0);
  oz = __shfl_sync(libero_warp::mask, oz, 0);
  if (!isfinite(ox) || !isfinite(oy) || !isfinite(oz)) {
    if (lane == 0)
      status[w] = 5;
    return;
  }
  bool bad = false;
  // Optional immutable model CSR removes inactive-column branches. The
  // dense diagnostics still receive their required zero values above.
  const bool indexed = body_dof_offsets && body_dof_indices;
  const int begin = indexed ? body_dof_offsets[body] : 0;
  const int end = indexed ? body_dof_offsets[body + 1] : nv;
  for (int z = begin + lane; z < end; z += 32) {
    const int k = indexed ? body_dof_indices[z] : z;
    if (!indexed && ancestor[body * nv + k] != 1)
      continue;
    const double *d = cdof + (w * nv + k) * 6;
    const double vx = d[3] + (d[1] * oz - d[2] * oy);
    const double vy = d[4] + (d[2] * ox - d[0] * oz);
    const double vz = d[5] + (d[0] * oy - d[1] * ox);
    bool ok = isfinite(vx) && isfinite(vy) && isfinite(vz);
    for (int j = 0; j < 6; j++)
      ok = ok && isfinite(d[j]);
    bad = bad || !ok;
    if (ok) {
      for (int j = 0; j < 3; j++)
        r[j * nv + k] = d[j];
      p[k] = vx;
      p[nv + k] = vy;
      p[2 * nv + k] = vz;
    }
  }
  // Failure clears the entire diagnostic, including columns completed by
  // other lanes. Register voting alone does not order global writes.
  const bool failed = libero_warp::any(bad);
  __syncwarp();
  if (failed) {
    if (lane == 0)
      status[w] = 5;
    for (int k = lane; k < 3 * nv; k += 32) {
      p[k] = 0;
      r[k] = 0;
    }
  }
  __syncwarp();
}
static __device__ void route(int w, const int32_t *active, const int32_t *status, int32_t *out) {
  out[w] = (active[w] == 1 && status[w] == 0) ? 1 : 0;
}
static __device__ void selected(int w, const double *qpos, const double *qvel, const double *pos,
                                const double *rot, const double *jp, const double *jr, const double *qm,
                                const double *qbias, const int32_t *active, const int32_t *qi,
                                const int32_t *vi, const int32_t *sparse, int nq, int nv, int nm,
                                double *ee_pos, double *ee_mat, double *vp, double *vo, double *q, double *qd,
                                double *J, double *M, double *bias, int32_t *status, bool trusted = false,
                                bool parallel_velocity = false) {
  if (threadIdx.x >= 32)
    return;
  const int lane = libero_warp::lane();
  if (lane == 0)
    status[w] = 0;
  libero_vec::fill_lane(ee_pos + w * 3, 3, 0.0, lane, 32);
  libero_vec::fill_lane(vp + w * 3, 3, 0.0, lane, 32);
  libero_vec::fill_lane(vo + w * 3, 3, 0.0, lane, 32);
  libero_vec::fill_lane(ee_mat + w * 9, 9, 0.0, lane, 32);
  libero_vec::fill_lane(q + w * 7, 7, 0.0, lane, 32);
  libero_vec::fill_lane(qd + w * 7, 7, 0.0, lane, 32);
  libero_vec::fill_lane(bias + w * 7, 7, 0.0, lane, 32);
  libero_vec::fill_lane(J + w * 42, 42, 0.0, lane, 32);
  libero_vec::fill_lane(M + w * 49, 49, 0.0, lane, 32);
  __syncwarp();
  if (active[w] == 0)
    return;
  if (active[w] != 1) {
    if (lane == 0)
      status[w] = 1;
    return;
  }
  if (!trusted) {
    bool bad = false;
    bad = libero_vec::any_nonfinite_lane(qpos + w * nq, nq, lane, 32);
    bad = libero_vec::any_nonfinite_lane(qvel + w * nv, nv, lane, 32) || bad;
    bad = libero_vec::any_nonfinite_lane(qbias + w * nv, nv, lane, 32) || bad;
    for (int r = 0; r < 3; r++) {
      bad = libero_vec::any_nonfinite_lane(jp + (w * 3 + r) * nv, nv, lane, 32) || bad;
      bad = libero_vec::any_nonfinite_lane(jr + (w * 3 + r) * nv, nv, lane, 32) || bad;
    }
    bad = libero_vec::any_nonfinite_lane(qm + w * nm, nm, lane, 32) || bad;
    bad = libero_vec::any_nonfinite_lane(pos + w * 3, 3, lane, 32) || bad;
    bad = libero_vec::any_nonfinite_lane(rot + w * 9, 9, lane, 32) || bad;
    if (libero_warp::any(bad)) {
      if (lane == 0)
        status[w] = 2;
      return;
    }
  }
  libero_vec::copy_lane(pos + w * 3, ee_pos + w * 3, 3, lane, 32);
  libero_vec::copy_lane(rot + w * 9, ee_mat + w * 9, 9, lane, 32);
  // Six independent serial dots: preserve full-NV order within each output.
  if (parallel_velocity) {
    if (lane < 6) {
      const int r = lane % 3;
      const double *jac = lane < 3 ? jp : jr;
      double value = 0;
      for (int k = 0; k < nv; ++k)
        value += jac[(w * 3 + r) * nv + k] * qvel[w * nv + k];
      (lane < 3 ? vp : vo)[w * 3 + r] = value;
    }
  } else {
    if (lane == 0)
      for (int r = 0; r < 3; r++) {
        double a = 0, b = 0;
        for (int k = 0; k < nv; k++) {
          a += jp[(w * 3 + r) * nv + k] * qvel[w * nv + k];
          b += jr[(w * 3 + r) * nv + k] * qvel[w * nv + k];
        }
        vp[w * 3 + r] = a;
        vo[w * 3 + r] = b;
      }
  }
  // Read live descriptors for every call; there is no stale cold map.
  // Pair destinations only inside each seven-column row. Indirect sources
  // use wide reads solely when both live selectors are consecutive.
  for (int pair = lane; pair < 4; pair += 32) {
    const int i = pair * 2, v0 = vi[i], q0 = qi[i];
    if (i + 1 < 7) {
      const int v1 = vi[i + 1], q1 = qi[i + 1];
      double2 x = q1 == q0 + 1 ? libero_vec::load_pair(qpos + w * nq + q0)
                               : make_double2(qpos[w * nq + q0], qpos[w * nq + q1]);
      libero_vec::store_pair(q + w * 7 + i, x.x, x.y);
      x = v1 == v0 + 1 ? libero_vec::load_pair(qvel + w * nv + v0)
                       : make_double2(qvel[w * nv + v0], qvel[w * nv + v1]);
      libero_vec::store_pair(qd + w * 7 + i, x.x, x.y);
      x = v1 == v0 + 1 ? libero_vec::load_pair(qbias + w * nv + v0)
                       : make_double2(qbias[w * nv + v0], qbias[w * nv + v1]);
      libero_vec::store_pair(bias + w * 7 + i, x.x, x.y);
      for (int r = 0; r < 3; r++) {
        x = v1 == v0 + 1 ? libero_vec::load_pair(jp + (w * 3 + r) * nv + v0)
                         : make_double2(jp[(w * 3 + r) * nv + v0], jp[(w * 3 + r) * nv + v1]);
        libero_vec::store_pair(J + w * 42 + r * 7 + i, x.x, x.y);
        x = v1 == v0 + 1 ? libero_vec::load_pair(jr + (w * 3 + r) * nv + v0)
                         : make_double2(jr[(w * 3 + r) * nv + v0], jr[(w * 3 + r) * nv + v1]);
        libero_vec::store_pair(J + w * 42 + (r + 3) * 7 + i, x.x, x.y);
      }
    } else {
      q[w * 7 + i] = qpos[w * nq + q0];
      qd[w * 7 + i] = qvel[w * nv + v0];
      bias[w * 7 + i] = qbias[w * nv + v0];
      for (int r = 0; r < 3; r++) {
        J[w * 42 + r * 7 + i] = jp[(w * 3 + r) * nv + v0];
        J[w * 42 + (r + 3) * 7 + i] = jr[(w * 3 + r) * nv + v0];
      }
    }
  }
  // The flattened pair may cross a row boundary; both sparse addresses
  // are checked independently. Missing cells retain the zero initialization.
  for (int pair = lane; pair < 25; pair += 32) {
    const int z = pair * 2, row = vi[z / 7], col = vi[z % 7];
    const int a0 = sparse[row * nv + col];
    if (z + 1 < 49) {
      const int a1 = sparse[vi[(z + 1) / 7] * nv + vi[(z + 1) % 7]];
      if (a0 >= 0 && a0 < nm - 1 && a1 == a0 + 1) {
        const double2 x = libero_vec::load_pair(qm + w * nm + a0);
        libero_vec::store_pair(M + w * 49 + z, x.x, x.y);
      } else {
        if (a0 >= 0)
          M[w * 49 + z] = qm[w * nm + a0];
        if (a1 >= 0)
          M[w * 49 + z + 1] = qm[w * nm + a1];
      }
    } else if (a0 >= 0)
      M[w * 49 + z] = qm[w * nm + a0];
  }
  __syncwarp();
}
static __device__ void status(int w, const int32_t *producer, const int32_t *adapter, int32_t *out) {
  out[w] = producer[w] ? 10 + producer[w] : adapter[w];
}

// Cooperative local adapter: columns are written once and validated before
// selected gathers. The accepted sequential velocity accumulation is retained.
static __device__ __forceinline__ void
local_selected(int w, const double *cdof, const double *subtree, const double *pos, const int32_t *active,
               const int32_t *ancestor, int root, int body, int nv, int nb, double *jp, double *jr,
               int32_t *jac_status, int32_t *adapter_active, const double *qpos, const double *qvel,
               const double *rot, const double *qm, const double *qbias, const int32_t *qi, const int32_t *vi,
               const int32_t *sparse, int nq, int nm, double *ep, double *er, double *vp, double *vo,
               double *q, double *qd, double *J, double *M, double *bias, int32_t *adapter_status,
               int32_t *out_status, const int32_t *body_dof_offsets, const int32_t *body_dof_indices,
               bool trusted, bool parallel_velocity) {
  if (threadIdx.x >= 32)
    return;
  jacobian(w, cdof, subtree, pos, active, ancestor, root, body, nv, nb, jp, jr, jac_status, body_dof_offsets,
           body_dof_indices);
  __syncwarp();
  if (libero_warp::lane() == 0)
    route(w, active, jac_status, adapter_active);
  __syncwarp();
  selected(w, qpos, qvel, pos, rot, jp, jr, qm, qbias, adapter_active, qi, vi, sparse, nq, nv, nm, ep, er, vp,
           vo, q, qd, J, M, bias, adapter_status, trusted, parallel_velocity);
  __syncwarp();
  if (libero_warp::lane() == 0)
    status(w, jac_status, adapter_status, out_status);
}
} // namespace cf_osc_gather

} // namespace physics_controller_gather

namespace physics_controller_gather::cf_osc_gather {
static __device__ __forceinline__ void gate(int w, const int32_t *mask, int32_t *error, const int64_t *state,
                                            const int64_t *source, const int64_t *epoch, const int32_t *valid,
                                            const int64_t *written, int32_t *active) {
  const int requested = mask[w];
  active[w] = 0;
  if (requested != 1 || error[w] != 0)
    return;
  if (state[w] != source[w]) {
    error[w] = 203;
    return;
  }
  if (valid[w] == 1 && written[w] == epoch[w])
    active[w] = 1;
}

static __device__ __forceinline__ void site(int w, const double *pos, const double *mat,
                                            const int32_t *active, double *outp, double *outm, int ns,
                                            int id) {
  if (active[w] != 1)
    return;
  for (int k = 0; k < 3; k++)
    outp[3 * w + k] = pos[(w * ns + id) * 3 + k];
  for (int k = 0; k < 9; k++)
    outm[9 * w + k] = mat[(w * ns + id) * 9 + k];
}

static __device__ __forceinline__ void publish_velocity(int w, const int32_t *active, const int32_t *status,
                                                        const double *vp, const double *vo, double *target,
                                                        int32_t *first) {
  if (active[w] != 1)
    return;
  if (status[w] != 0) {
    if (first[w] == 0)
      first[w] = 202;
    return;
  }
  for (int k = 0; k < 3; k++) {
    target[w * 6 + k] = vp[w * 3 + k];
    target[w * 6 + k + 3] = vo[w * 3 + k];
  }
}

} // namespace physics_controller_gather::cf_osc_gather

namespace physics_controller_bias::cf_osc_gather {
static __device__ __forceinline__ void
live_bias(int w, const int32_t *mask, const int32_t *first, const int64_t *state, const int64_t *source,
          const int64_t *epoch, const int32_t *valid, const int64_t *written, const double *bias,
          double *selected, const int32_t *indices, int32_t *parent_valid, int64_t *parent_written, int nv) {
  parent_valid[w] = 0;
  if (mask[w] != 1 || first[w] != 0 || state[w] != source[w])
    return;
  if (valid[w] != 1 || written[w] != epoch[w])
    return;
  const uintptr_t src = (uintptr_t)(bias + w * nv), dst = (uintptr_t)(selected + w * 7);
  const bool overlap =
      src <= dst ? dst - src < (uintptr_t)nv * sizeof(double) : src - dst < 7 * sizeof(double);
  if (overlap) {
    for (int k = 0; k < 7; k++)
      selected[w * 7 + k] = bias[w * nv + indices[k]];
  } else {
    int k = 0;
    for (; k + 1 < 7; k += 2) {
      const int i = indices[k], j = indices[k + 1];
      const double2 x = j == i + 1 ? libero_vec::load_pair(bias + w * nv + i)
                                   : make_double2(bias[w * nv + i], bias[w * nv + j]);
      libero_vec::store_pair(selected + w * 7 + k, x.x, x.y);
    }
    selected[w * 7 + k] = bias[w * nv + indices[k]];
  }
  parent_written[w] = epoch[w];
  parent_valid[w] = 1;
}

} // namespace physics_controller_bias::cf_osc_gather

namespace physics_controller::protocol {
static __device__ __forceinline__ void gate(int w, int *active, int *pv, int64_t *written, int64_t *expected,
                                            int *first, int *error, int *policy, int step) {
  error[w] = 0;
  policy[w] = step;
  if (pv[w] != 1 || written[w] != expected[w] || first[w])
    active[w] = 0;
}

static __device__ __forceinline__ void commit_mask(int w, int *active, int *error, int *first, int *commit,
                                                   int64_t *event, int *valid) {
  commit[w] = valid[w] = 0;
  if (active[w] != 1 || first[w])
    return;
  if (error[w]) {
    first[w] = 202;
    return;
  }
  if (event[w] == INT64_MAX) {
    first[w] = 9;
    return;
  }
  event[w]++;
  commit[w] = 1;
}

static __device__ __forceinline__ void publish(int w, int *mask, int64_t *event, int64_t *parent,
                                               int64_t *written, int64_t *consumed, int *valid) {
  if (mask[w] == 1) {
    consumed[w] = parent[w];
    written[w] = event[w];
    valid[w] = 1;
  }
}

} // namespace physics_controller::protocol

namespace physics_controller {
namespace {
using namespace flash_libero;
// No barriers inside: independent fields may overlap execution, while all
// aliased fields run in original order on lane 0. The caller publishes once.
__device__ void copy_controller_channels(const ControllerChannels &channels, int world, bool to_working) {
  for (int field = 0; field < channels.floating_count; ++field) {
    const auto &view = channels.floating[field];
    if (to_working && !(view.parallel_copy & 2))
      continue;
    const double *source = (to_working ? view.published : view.working) + world * view.width;
    double *target = (to_working ? view.working : view.published) + world * view.width;
    if (view.parallel_copy & 1)
      libero_vec::copy_block(source, target, size_t(view.width));
    else if (threadIdx.x == 0)
      ::osc_cuda::serial_copy_double(source, target, view.width);
  }
}

// All lanes enter. Aliased publication fields retain their lane-0 order.
static __device__ __noinline__ void run_ordered(const SubstepArguments &args, int world,
                                              bool policy_step, unsigned char *shared) {
  CF_OSC_PROFILE_STAGES(Gather);
  if (threadIdx.x == 0) {
    args.gather_gate.mask[world] = args.controller_plan.refresh[world];
    args.controller_bias.mask[world] = args.controller_plan.call[world];
    const auto &in = args.gather_gate;
    physics_controller_gather::cf_osc_gather::gate(world, in.mask, in.error, in.state, in.source, in.epoch,
                                                   in.valid, in.written, in.active);
    const auto &pose = args.end_effector_pose;
    physics_controller_gather::cf_osc_gather::site(world, pose.pos, pose.mat, pose.active, pose.outp,
                                                   pose.outm, pose.ns, pose.id);
  }
  __syncthreads();
  {
    const auto &in = args.controller_inputs;
    physics_controller_gather::cf_osc_gather::local_selected(
        world, in.cdof, in.subtree, in.pos, in.active, in.ancestor, in.root, in.body, in.nv, in.nb, in.jp,
        in.jr, in.jac_status, in.adapter_active, in.qpos, in.qvel, in.rot, in.qm, in.qbias, in.qi, in.vi,
        in.sparse, in.nq, in.nm, in.ep, in.er, in.vp, in.vo, in.q, in.qd, in.J, in.M, in.bias,
        in.adapter_status, in.out_status, in.body_dof_offsets, in.body_dof_indices, true);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const auto &in = args.end_effector_velocity;
    physics_controller_gather::cf_osc_gather::publish_velocity(world, in.active, in.status, in.vp, in.vo,
                                                               in.target, in.first);
    const auto &bias = args.controller_bias;
    physics_controller_bias::cf_osc_gather::live_bias(
        world, bias.mask, bias.first, bias.state, bias.source, bias.epoch, bias.valid, bias.written,
        bias.bias, bias.selected, bias.indices, bias.parent_valid, bias.parent_written, bias.nv);
  }
  __syncthreads();

  CF_OSC_PROFILE_NEXT(Seed);
  const auto &channels = args.controller_channels;
  copy_controller_channels(channels, world, true);
  if (threadIdx.x == 0) {
    for (int field = 0; field < channels.integer_count; ++field) {
      const auto &view = channels.integer[field];
      view.working[world] = view.published[world];
    }
    args.controller_gate.active[world] = args.controller_plan.call[world];
    args.controller_solve.needs[world] = args.controller_plan.refresh[world];
    const auto &in = args.controller_gate;
    physics_controller::protocol::gate(world, in.active, in.pv, in.written, in.expected, in.first, in.error,
                                       in.policy, int(policy_step));
  }
  __syncthreads();
  CF_OSC_PROFILE_NEXT(SolveEnvelope);
  {
    CF_OSC_PROFILE_SCOPE(Solve);
    const auto &in = args.controller_solve;
    physics_controller::solve(in, world, shared);
  }
  __syncthreads();
  CF_OSC_PROFILE_NEXT(Commit);
  if (threadIdx.x == 0) {
    const auto &in = args.controller_commit;
    physics_controller::protocol::commit_mask(world, in.active, in.error, in.first, in.commit, in.event,
                                              in.valid);
  }
  __syncthreads();

  if (args.controller_commit.commit[world] == 1) {
    copy_controller_channels(channels, world, false);
    if (threadIdx.x == 0) {
      for (int field = 0; field < channels.integer_count; ++field) {
        const auto &view = channels.integer[field];
        view.published[world] = view.working[world];
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const auto &in = args.controller_publish;
    physics_controller::protocol::publish(world, in.mask, in.event, in.parent, in.written, in.consumed,
                                          in.valid);
  }
  __syncthreads();
}

// Every descriptor keeps the original field's world stride. Packets never
// cross field tails, and binding proves all public/working owners disjoint.
__device__ void copy_packets(const ControllerChannels &channels, int w, bool seed, int lane, int stride) {
  const int offset = seed ? 0 : channels.seed_packets;
  const int count = seed ? channels.seed_packets : channels.commit_packets;
  for (int z = lane; z < count; z += stride) {
    const auto &packet = channels.copy_packets[offset + z];
    const double *source = (seed ? packet.published : packet.working) + w * packet.width;
    double *target = (seed ? packet.working : packet.published) + w * packet.width;
    if (packet.words == 2) {
      const double2 value = libero_vec::load_pair(source);
      libero_vec::store_pair(target, value.x, value.y);
    } else
      *target = *source;
  }
}

// A complete first warp enters. Gate decisions stay in registers after one
// leader read; no lane rereads an error word another lane may have updated.
__device__ void prepare_inputs(const SubstepArguments &args, int w, bool policy) {
  const int lane = threadIdx.x;
  const auto &g = args.gather_gate;
  const auto &b = args.controller_bias;
  const int refresh = args.controller_plan.refresh[w];
  const int call = args.controller_plan.call[w];
  int error = lane == 0 ? g.error[w] : 0;
  error = __shfl_sync(0xffffffffu, error, 0);
  const bool same_state = g.state[w] == g.source[w];
  const bool current_bias = g.valid[w] == 1 && g.written[w] == g.epoch[w];
  const bool requested = refresh == 1 && error == 0;
  const bool gather = requested && same_state && current_bias;
  if (requested && !same_state)
    error = 203;
  if (lane == 0) {
    g.mask[w] = refresh;
    b.mask[w] = call;
    g.active[w] = int(gather);
  }
  const auto &pose = args.end_effector_pose;
  if (gather) {
    if (lane < 3)
      pose.outp[w * 3 + lane] = pose.pos[(w * pose.ns + pose.id) * 3 + lane];
    if (lane < 9)
      pose.outm[w * 9 + lane] = pose.mat[(w * pose.ns + pose.id) * 9 + lane];
  }
  __syncwarp(); // site and active are consumed by Jacobian/selected
  const auto &in = args.controller_inputs;
  physics_controller_gather::cf_osc_gather::local_selected(
      w, in.cdof, in.subtree, in.pos, in.active, in.ancestor, in.root, in.body, in.nv, in.nb, in.jp,
      in.jr, in.jac_status, in.adapter_active, in.qpos, in.qvel, in.rot, in.qm, in.qbias, in.qi, in.vi,
      in.sparse, in.nq, in.nm, in.ep, in.er, in.vp, in.vo, in.q, in.qd, in.J, in.M, in.bias,
      in.adapter_status, in.out_status, in.body_dof_offsets, in.body_dof_indices, true, true);
  __syncwarp(); // last gather leader publishes out_status
  const auto &velocity = args.end_effector_velocity;
  const bool velocity_ok = gather && velocity.status[w] == 0;
  if (gather && !velocity_ok && error == 0)
    error = 202;
  if (velocity_ok && lane < 6)
    velocity.target[w * 6 + lane] = lane < 3 ? velocity.vp[w * 3 + lane] : velocity.vo[w * 3 + lane - 3];
  const bool bias_ok = call == 1 && error == 0 && same_state && current_bias;
  if (bias_ok && lane < 7)
    b.selected[w * 7 + lane] = b.bias[w * b.nv + b.indices[lane]];
  __syncwarp();
  if (lane == 0) {
    g.error[w] = error;
    b.parent_valid[w] = int(bias_ok);
    if (bias_ok)
      b.parent_written[w] = b.epoch[w];
    const auto &channels = args.controller_channels;
    for (int z = 0; z < channels.integer_count; ++z) {
      const auto &field = channels.integer[z];
      field.working[w] = field.published[w];
    }
    args.controller_gate.active[w] = call;
    args.controller_solve.needs[w] = refresh;
    const auto &c = args.controller_gate;
    protocol::gate(w, c.active, c.pv, c.written, c.expected, c.first, c.error, c.policy, int(policy));
  }
}

__device__ void run_parallel(const SubstepArguments &args, int w, bool policy, unsigned char *shared) {
  CF_OSC_PROFILE_STAGES(Gather);
  const auto &channels = args.controller_channels;
  const auto &commit = args.controller_commit;
  // Immutable until after B1 and the solver's internal rendezvous. Binding
  // proves this word does not alias any payload or preparation owner.
  const int64_t event = commit.event[w];
  if (threadIdx.x < 32)
    prepare_inputs(args, w, policy);
  else
    copy_packets(channels, w, true, threadIdx.x - 32, blockDim.x - 32);
  __syncthreads(); // B1: gather and concurrent seed hand off to the solver
  CF_OSC_PROFILE_NEXT(SolveEnvelope);
  // Each lane closes its first-error read before solve's first rendezvous;
  // publication uses only register predicates after the solver returns.
  const bool eligible = commit.active[w] == 1 && commit.first[w] == 0;
  bool solved;
  {
    CF_OSC_PROFILE_SCOPE(Solve);
    solved = solve(args.controller_solve, w, shared);
  }
  CF_OSC_PROFILE_NEXT(Commit);
  const bool publish = eligible && solved && event != INT64_MAX;
  if (threadIdx.x == 0) {
    commit.commit[w] = commit.valid[w] = 0;
    if (eligible) {
      if (!solved)
        commit.first[w] = 202;
      else if (event == INT64_MAX)
        commit.first[w] = 9;
      else {
        commit.event[w] = event + 1;
        commit.commit[w] = 1;
      }
    }
  }
  if (publish) {
    copy_packets(channels, w, false, threadIdx.x, blockDim.x);
    if (threadIdx.x == 0)
      for (int z = 0; z < channels.integer_count; ++z) {
        const auto &field = channels.integer[z];
        field.published[w] = field.working[w];
      }
  }
  __syncthreads(); // B2: close payload writes before valid/epoch publication
  if (threadIdx.x == 0) {
    const auto &p = args.controller_publish;
    protocol::publish(w, p.mask, p.event, p.parent, p.written, p.consumed, p.valid);
  }
  // The outer recovery admission's existing block rendezvous closes metadata
  // publication and scratch lifetime. No third standalone barrier is needed.
}
} // namespace

__device__ __noinline__ void run_controller(const flash_libero::SubstepArguments &args, int world,
                                           bool policy, unsigned char *shared) {
  CF_OSC_PROFILE_BEGIN();
  if (args.controller_channels.copy_packets && blockDim.x > 32)
    run_parallel(args, world, policy, shared);
  else
    run_ordered(args, world, policy, shared);
}
} // namespace physics_controller
