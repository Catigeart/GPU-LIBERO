#include "precision.cuh"
// Maintained single-physical-substep CUDA implementation (catalogue dimensions).
// The ordinary kernel retains the CPU application's three complete forwards.
// A separate FP32 candidate follows step1 -> OSC -> step2 with one solve.
// Newton numerical bodies remain separately compiled in newton_solver.cu.
// CMake compiles this file directly. There is no build/runtime source generation.

// One block owns one world's physical substep; no loop over control substeps.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cublasdx.hpp>
#include "substep_kernel.cuh"
#include "substep_shared_layout.cuh"
#include "newton_solver.cuh"
#include "newton_shared_layout.cuh"
#include "../postprocess/post_solver.cuh"
#include "substep_shared.cuh"
#include "substep_profile.cuh"
#include "../common/math.cuh"
#include "../common/vector.cuh"
#include "../common/warp.cuh"
#include "../common/packed_mass.cuh"
#include "forward_mass.cuh"
#include "../control/osc_runtime.cuh"
#include "../control/controller_pipeline.cuh"
#include "../runtime/forward.cuh"
#include "../runtime/position_reuse.cuh"
#include "../runtime/recovery.cuh"

using flash_libero::SubstepShape;


namespace physics_bias {
static __device__ void d_gate(int w, int *active, int *error, int64_t *state, int64_t *source, int64_t *epoch,
                              int *valid, int64_t *written, int *gate, int require);
}

namespace physics_bias {
static __device__ void d_reference_gate(int w, int *done, int *err, int64_t *written, int64_t *expected,
                                        int64_t *epoch, int *gate, int *first);
}

namespace physics_bias {
static __device__ void d_commit_bias_block(int w, int *active, int *gate, int *status, int64_t *epoch,
                                           int *error, real_t *work, real_t *pub, int *valid,
                                           int64_t *written, int width);
}

namespace physics_bias {
static __device__ void
d_rne_bias_cached(int w, int *active, real_t *cdot, real_t *qvel, real_t *cinert, real_t *cvel, real_t *cdof,
                  int *bp, int *ba, int *bn, int *db, real_t *gravity, int disabled, int nb, int nv,
                  real_t *cacc, real_t *frc, real_t *result, int *status, const real_t *cache_acc,
                  const real_t *cache_force, const int *cache_status, const int *cache_valid,
                  const int64_t *cache_state, const int64_t *cache_forward, const int64_t *state_epoch,
                  const int64_t *forward_epoch, const int *level_offsets = nullptr,
                  const int *level_bodies = nullptr, int nlevels = 0, const int *children_offsets = nullptr,
                  const int *children_indices = nullptr);
}

// Maintained world-local implementation, compiled as part of fused_substep.cu.

// Shared FP64 bias implementation.

namespace physics_bias {

// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.

// One block owns one world. These are private working values, never public
// aliases; their lifetime extends across the force-chain device operations.
// Handwritten sparse rigid-body arithmetic. These routines receive whole
// allocations; only the selected world row is touched. The caller executes
// each collective tree routine on all threads. Its ordered tree arithmetic
// remains on lane zero; array checks and independent projections cooperate.

// Collective helpers: every thread in the world block must enter. Warp votes
// exchange predicates only; the barriers publish and protect inter-warp flags.
static __device__ inline bool d1_finite_block(const real_t *x, int n, const real_t *y = nullptr, int ny = 0,
                                              const real_t *z = nullptr, int nz = 0,
                                              const real_t *u = nullptr, int nu = 0,
                                              const real_t *v = nullptr, int nv = 0) {
  __shared__ int warp_bad[32];
  bool bad = false;
  bad |= libero_vec::any_nonfinite_block_lane(x, n);
  bad |= libero_vec::any_nonfinite_block_lane(y, ny);
  bad |= libero_vec::any_nonfinite_block_lane(z, nz);
  bad |= libero_vec::any_nonfinite_block_lane(u, nu);
  bad |= libero_vec::any_nonfinite_block_lane(v, nv);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  // Membership comes from block geometry, not the instantaneous active
  // lanes after unequal per-thread strides. Include every live warp lane.
  const int remaining = (int)blockDim.x - (warp << 5);
  const unsigned mask = remaining >= 32 ? 0xffffffffu : ((1u << remaining) - 1u);
  const int vote = __any_sync(mask, bad);
  if (lane == 0)
    warp_bad[warp] = vote;
  __syncthreads();
  bool any = false;
  for (int k = 0; k < (blockDim.x + 31) / 32; ++k)
    any |= warp_bad[k] != 0;
  __syncthreads();
  return !any;
}
static __device__ inline void d1_zero_block(real_t *x, int n) { libero_vec::fill_block(x, n, REAL_C(0.0)); }

// Collective publication: callers enter with the complete world block. Gate
// decisions and scalar flags retain lane-zero order; payload stores complete
// before the epoch becomes visible. Shifted overlaps retain serial copy order.
static __device__ inline bool d_copy_overlap(const real_t *a, const real_t *b, int width) {
  const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  const uintptr_t bytes = uintptr_t(width) * sizeof(real_t);
  return x < y + bytes && y < x + bytes;
}

static __device__ __forceinline__ void d_commit_bias_block(int w, int *active, int *gate, int *status,
                                                           int64_t *epoch, int *error, real_t *work,
                                                           real_t *pub, int *valid, int64_t *written,
                                                           int width) {
  __shared__ int admitted, serial;
  real_t *s = work + (long long)w * width, *t = pub + (long long)w * width;
  if (threadIdx.x == 0) {
    valid[w] = 0;
    admitted = active[w] == 1 && gate[w] == 1 && !error[w];
    if (admitted && status[w]) {
      error[w] = 211;
      admitted = 0;
    }
    serial = d_copy_overlap(s, t, width);
  }
  __syncthreads();
  if (admitted) {
    if (serial) {
      if (threadIdx.x == 0)
        for (int i = 0; i < width; ++i)
          t[i] = s[i];
    } else
      libero_vec::copy_block(s, t, width);
  }
  __syncthreads();
  if (threadIdx.x == 0 && admitted) {
    written[w] = epoch[w];
    valid[w] = 1;
  }
  __syncthreads();
}

// Qualified topology only: levels omit world body 0 and parent IDs precede
// children. A level barrier publishes every parent before its children read.

// Compressed spatial inertia multiplication, not a dense BLAS operand.
static __device__ inline void d_inertia(const real_t *i, const real_t *v, real_t *f) {
  f[0] = i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5];
  f[1] = i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5];
  f[2] = i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4];
  f[3] = i[8] * v[1] - i[7] * v[2] + i[9] * v[3];
  f[4] = i[6] * v[2] - i[8] * v[0] + i[9] * v[4];
  f[5] = i[7] * v[0] - i[6] * v[1] + i[9] * v[5];
}
static __device__ inline void d_cross_force(const real_t *v, const real_t *t, real_t *x) {
  x[0] = -v[2] * t[1] + v[1] * t[2];
  x[1] = v[2] * t[0] - v[0] * t[2];
  x[2] = -v[1] * t[0] + v[0] * t[1];
  x[3] = -v[2] * t[4] + v[1] * t[5];
  x[4] = v[2] * t[3] - v[0] * t[5];
  x[5] = -v[1] * t[3] + v[0] * t[4];
  x[0] += -v[5] * t[4] + v[4] * t[5];
  x[1] += v[5] * t[3] - v[3] * t[5];
  x[2] += -v[4] * t[3] + v[3] * t[4];
}
static __device__ inline void d1_rne_body(int b, const real_t *d, const real_t *q, const real_t *i,
                                          const real_t *v, const int *bp, const int *ba, const int *bn,
                                          real_t *acc, real_t *f, int nb = 0, bool component_planes = false) {
  real_t t[6] = {};
  const int adr = ba[b], n = bn[b];
  if (n == 1)
    for (int k = 0; k < 6; ++k)
      t[k] = d[6 * adr + k] * q[adr];
  else
    for (int j = 0; j < n; ++j)
      if (q[adr + j] != 0)
        for (int k = 0; k < 6; ++k)
          t[k] += d[6 * (adr + j) + k] * q[adr + j];
  real_t local_acc[6];
  for (int k = 0; k < 6; ++k) {
    const int out = component_planes ? k * nb + b : 6 * b + k;
    const int parent = component_planes ? k * nb + bp[b] : 6 * bp[b] + k;
    local_acc[k] = acc[parent] + t[k];
    acc[out] = local_acc[k];
  }
  real_t fi[6], corr[6], cross[6];
  d_inertia(i + 10 * b, local_acc, fi);
  d_inertia(i + 10 * b, v + 6 * b, corr);
  d_cross_force(v + 6 * b, corr, cross);
  for (int k = 0; k < 6; ++k)
    f[component_planes ? k * nb + b : 6 * b + k] = fi[k] + cross[k];
}
static __device__ inline void d1_rne_forward(const real_t *d, const real_t *q, const real_t *i,
                                             const real_t *v, const int *bp, const int *ba, const int *bn,
                                             const real_t *gravity, int disabled, int nb, real_t *acc,
                                             real_t *f, const int *lo, const int *lb, int nl,
                                             bool component_planes = false) {
  if (threadIdx.x == 0 && !disabled)
    for (int k = 0; k < 3; ++k)
      acc[component_planes ? (3 + k) * nb : 3 + k] = -gravity[k];
  __syncthreads();
  if (lo && lb && nl > 0) {
    for (int l = 0; l < nl; ++l) {
      for (int k = lo[l] + threadIdx.x; k < lo[l + 1]; k += blockDim.x)
        d1_rne_body(lb[k], d, q, i, v, bp, ba, bn, acc, f, nb, component_planes);
      __syncthreads();
    }
  } else if (threadIdx.x == 0) {
    for (int b = 1; b < nb; ++b)
      d1_rne_body(b, d, q, i, v, bp, ba, bn, acc, f, nb, component_planes);
  }
  __syncthreads();
}
static __device__ inline void d1_rne_reverse(const int *bp, int nb, real_t *f, const int *lo, const int *lb,
                                             int nl, const int *co, const int *ci) {
  if (lo && lb && co && ci && nl > 0) {
    // One writer per parent; descending child IDs exactly reproduce the
    // original reverse-body addition order, including cancellation.
    for (int l = nl - 1; l >= 0; --l) {
      for (int k = lo[l] + threadIdx.x; k < lo[l + 1]; k += blockDim.x) {
        const int b = lb[k];
        for (int a = co[b]; a < co[b + 1]; ++a)
          for (int j = 0; j < 6; ++j)
            f[6 * b + j] += f[6 * ci[a] + j];
      }
      __syncthreads();
    }
  } else if (threadIdx.x == 0) {
    for (int b = nb - 1; b > 0; --b)
      if (bp[b] > 0)
        for (int k = 0; k < 6; ++k)
          f[6 * bp[b] + k] += f[6 * b + k];
  }
  __syncthreads();
}
static __device__ inline void
d_rne_bias(int w, int *active, real_t *cdot, real_t *qvel, real_t *cinert, real_t *cvel, real_t *cdof,
           int *bp, int *ba, int *bn, int *db, real_t *gravity, int disabled, int nb, int nv, real_t *cacc,
           real_t *frc, real_t *result, int *status, const int *level_offsets = nullptr,
           const int *level_bodies = nullptr, int nlevels = 0, const int *children_offsets = nullptr,
           const int *children_indices = nullptr) {
  const real_t *d = cdot + w * nv * 6, *q = qvel + w * nv, *i = cinert + w * nb * 10, *v = cvel + w * nb * 6,
               *a = cdof + w * nv * 6;
  real_t *acc = cacc + w * nb * 6, *f = frc + w * nb * 6, *r = result + w * nv;
  if (threadIdx.x == 0)
    status[w] = 0;
  d1_zero_block(acc, nb * 6);
  d1_zero_block(f, nb * 6);
  d1_zero_block(r, nv);
  __syncthreads();
  if (!active[w])
    return;
  if (active[w] != 1) {
    if (threadIdx.x == 0)
      status[w] = 1;
    return;
  }
  if (!d1_finite_block(d, nv * 6) || !d1_finite_block(a, nv * 6, q, nv) || !d1_finite_block(i, nb * 10) ||
      !d1_finite_block(v, nb * 6)) {
    if (threadIdx.x == 0)
      status[w] = 2;
    return;
  }
  d1_rne_forward(d, q, i, v, bp, ba, bn, gravity, disabled, nb, acc, f, level_offsets, level_bodies, nlevels);
  d1_rne_reverse(bp, nb, f, level_offsets, level_bodies, nlevels, children_offsets, children_indices);
  for (int j = threadIdx.x; j < nv; j += blockDim.x) {
    const real_t *aa = a + 6 * j, *ff = f + 6 * db[j];
    r[j] =
        ((aa[0] * ff[0] + aa[2] * ff[2]) + (aa[1] * ff[1] + aa[3] * ff[3])) + (aa[4] * ff[4] + aa[5] * ff[5]);
  }
  __syncthreads();
  if (!d1_finite_block(acc, nb * 6, f, nb * 6, r, nv)) {
    if (threadIdx.x == 0)
      status[w] = 3;
    d1_zero_block(acc, nb * 6);
    d1_zero_block(f, nb * 6);
    d1_zero_block(r, nv);
  }
  __syncthreads();
}

// Compute only the forward part of bias RNE while the velocity input is live.
// Its private status does not decide velocity publication or the later S gate.
// The reverse force tree and public bias commit remain after S admission.

static __device__ __forceinline__ void
d_rne_bias_cached(int w, int *active, real_t *cdot, real_t *qvel, real_t *cinert, real_t *cvel, real_t *cdof,
                  int *bp, int *ba, int *bn, int *db, real_t *gravity, int disabled, int nb, int nv,
                  real_t *cacc, real_t *frc, real_t *result, int *status, const real_t *cache_acc,
                  const real_t *cache_force, const int *cache_status, const int *cache_valid,
                  const int64_t *cache_state, const int64_t *cache_forward, const int64_t *state_epoch,
                  const int64_t *forward_epoch, const int *level_offsets, const int *level_bodies,
                  int nlevels, const int *children_offsets, const int *children_indices) {
  if (active[w] != 1 || cache_valid[w] != 1 || cache_status[w] != 0 || cache_state[w] != state_epoch[w] ||
      cache_forward[w] != forward_epoch[w]) {
    d_rne_bias(w, active, cdot, qvel, cinert, cvel, cdof, bp, ba, bn, db, gravity, disabled, nb, nv, cacc,
               frc, result, status, level_offsets, level_bodies, nlevels, children_offsets, children_indices);
    return;
  }
  real_t *acc = cacc + (long long)w * nb * 6, *f = frc + (long long)w * nb * 6,
         *r = result + (long long)w * nv;
  const real_t *a = cdof + (long long)w * nv * 6;
  const real_t *ca = cache_acc + (long long)w * nb * 6, *cf = cache_force + (long long)w * nb * 6;
  if (threadIdx.x == 0)
    status[w] = 0;
  // Cache producer writes [component][body] directly. Each linear cache
  // tile is read contiguously; public RNE diagnostics keep [body][component].
  for (int k = threadIdx.x; k < nb * 6; k += blockDim.x) {
    const int body = k % nb, component = k / nb, dst = body * 6 + component;
    acc[dst] = ca[k];
    f[dst] = cf[k];
  }
  d1_zero_block(r, nv);
  __syncthreads();
  d1_rne_reverse(bp, nb, f, level_offsets, level_bodies, nlevels, children_offsets, children_indices);
  for (int j = threadIdx.x; j < nv; j += blockDim.x) {
    const real_t *aa = a + 6 * j, *ff = f + 6 * db[j];
    r[j] =
        ((aa[0] * ff[0] + aa[2] * ff[2]) + (aa[1] * ff[1] + aa[3] * ff[3])) + (aa[4] * ff[4] + aa[5] * ff[5]);
  }
  __syncthreads();
  if (!d1_finite_block(acc, nb * 6, f, nb * 6, r, nv)) {
    if (threadIdx.x == 0)
      status[w] = 3;
    d1_zero_block(acc, nb * 6);
    d1_zero_block(f, nb * 6);
    d1_zero_block(r, nv);
  }
  __syncthreads();
}

// Full-world-block helpers; blockDim.x must be a positive multiple of 32.
// Warp collectives only exchange registers; shared publication uses block barriers.
namespace d_passive_detail {}

namespace d_passive_detail {
struct Vec {
  real_t v[3];
  __device__ real_t &operator[](int i) { return v[i]; }
  __device__ real_t operator[](int i) const { return v[i]; }
};

} // namespace d_passive_detail

namespace d_passive_detail {
// The body producer retains scalar arithmetic and diagnostic-prefix semantics.

} // namespace d_passive_detail

// Large-NU preparation; the ordinary small actuator family retains scalar work.

// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.

// Smooth combine and external-force projection use one full warp per world.
// Their callers must enter with all block threads and rendezvous between stages.
// Other helpers in this header retain the owning-lane contract.
// Topology is immutable, qualified by the host before launch. World arrays use
// their original compact strides. Compilation permits FMA contraction.

// Packed tree LDL^T, not dense POTRF. No dense expansion or external LAPACK
// replacement: each packed row contains its diagonal followed by ancestors.

// The original externally visible work arrays are written once at the tail.
// All stage-to-stage operands, the single active RHS and candidate acceleration
// remain in shared memory. Failure still isolates durable public outputs.

} // namespace physics_bias

// Keep the shape-specific policy local to this implementation.

namespace physics_constraint_projection::cf_s {
static __device__ void n_mask(int w, const int *active, const int *first, const int *smooth,
                              const int *reference, const int *sstatus, int *admitted, int *error);
}

namespace physics_constraint_projection::cf_s {
static __device__ void project_counts(int w, const int *admitted, const int *source_counts,
                                      const int *source_ncon, int *nefc, int *ne, int *nf, int *ncon,
                                      int *count, int *empty, int *newton, int *active, int *status,
                                      int *item, int rows, int contacts, int nv);
}

namespace physics_constraint_projection::cf_s {
static __device__ void
project_cone_heads_verified(int w, const int *admitted, const int *nefc, const int *ne, const int *nf,
                            const int *ncon, const int *typ, const int *id, const int *dim, const real_t *mu,
                            const real_t *friction, int *out_dim, real_t *out_mu, real_t *out_friction,
                            int *status, int *item, int rows, int contacts, const int *compact_heads,
                            const int *compact_count, const int *compact_valid, const int *head_dim,
                            real_t *head_mu, real_t *head_friction, int *head_cone_valid);
}

namespace physics_constraint_projection::cf_s {
static __device__ void increment_plan(int w, const int *mask, const int *first, int64_t *epoch);
}

// Maintained world-local implementation, compiled as part of fused_substep.cu.

// Shared FP64 constraint_projection implementation.

namespace physics_constraint_projection {

// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.

// Typed S -> N metadata projection. Numeric rows already share their owners.

namespace cf_s {

static __device__ void project_cone_heads(int w, const int *admitted, const int *nefc, const int *ne,
                                          const int *nf, const int *ncon, const int *typ, const int *id,
                                          const int *dim, const real_t *mu, const real_t *friction,
                                          int *out_dim, real_t *out_mu, real_t *out_friction, int *status,
                                          int *item, int rows, int contacts) {
  __shared__ int head[32], contact[32], width[32], kind[32], parse_code[32];
  __shared__ int cursor, used, finished;
  if (threadIdx.x == 0) {
    finished = 0;
    cursor = ne[w] + nf[w];
    if (admitted[w] != 1 || status[w])
      finished = 1;
    else if (nefc[w] < 0 || nefc[w] > rows || ne[w] < 0 || nf[w] < 0 || cursor < 0 || cursor > nefc[w]) {
      status[w] = 9001;
      item[w] = 3;
      finished = 1;
    } else if (ncon[w] < 0 || ncon[w] > contacts) {
      status[w] = 9001;
      item[w] = 5;
      finished = 1;
    }
  }
  __syncthreads();
  const int rejected = finished;
  __syncthreads();
  if (rejected)
    return;
  const int n = nefc[w], start = ne[w] + nf[w], rb = w * rows, cb = w * contacts;
  // Two complete passes: no history write precedes global validation success.
  for (int pass = 0; pass < 2; pass++) {
    if (threadIdx.x == 0) {
      cursor = start;
      finished = cursor == n;
    }
    __syncthreads();
    while (!finished) {
      if (threadIdx.x == 0) {
        used = 0;
        while (used < 32 && cursor < n) {
          const int k = used++, i = cursor, t = typ[rb + i];
          int c = -1, d = 1, code = 0;
          if (t == 7 || t == 5) {
            c = id[rb + i];
            if (c < 0 || c >= ncon[w])
              code = 9002;
            else {
              d = dim[cb + c];
              if (t == 7) {
                if (d != 1 && d != 3 && d != 4 && d != 6)
                  code = 9003;
                else if (d > n - i)
                  code = 9004;
              } else if (d != 1)
                code = 9003;
            }
          } else if (t < 0 || t > 6)
            code = 9005;
          head[k] = i;
          contact[k] = c;
          width[k] = d;
          kind[k] = t;
          parse_code[k] = code;
          if (code)
            break;
          cursor += t == 7 ? d : 1;
        }
      }
      __syncthreads();
      if (threadIdx.x < 32) {
        const int k = threadIdx.x;
        int code = k < used ? parse_code[k] : 0, bad = k < used ? head[k] : -1;
        if (k < used && !code && kind[k] == 7) {
          const int i = head[k], c = contact[k], d = width[k];
          if (!pass) {
            for (int j = 0; j < d; j++)
              if (typ[rb + i + j] != 7 || id[rb + i + j] != c) {
                code = 9004;
                bad = i + j;
                break;
              }
            if (!code && (!isfinite(mu[cb + c]) || mu[cb + c] <= REAL_C(0.)))
              code = 9006;
            if (!code)
              for (int j = 0; j < d - 1; j++) {
                const real_t f = friction[(cb + c) * 5 + j];
                if (!isfinite(f) || f <= REAL_C(0.)) {
                  code = 9007;
                  break;
                }
              }
          } else {
            out_dim[rb + i] = d;
            out_mu[rb + i] = mu[cb + c];
            for (int j = 0; j < d - 1; j++)
              out_friction[(rb + i) * 5 + j] = friction[(cb + c) * 5 + j];
          }
        }
        const int first = libero_warp::min(code ? k : libero_warp::no_index);
        const int winner = first == libero_warp::no_index ? 0 : first;
        const int rejection = __shfl_sync(libero_warp::mask, code, winner);
        const int rejection_item = __shfl_sync(libero_warp::mask, bad, winner);
        if (k == 0) {
          if (first != libero_warp::no_index) {
            status[w] = rejection;
            item[w] = rejection_item;
            finished = 1;
          } else if (cursor == n)
            finished = 1;
        }
      }
      __syncthreads();
    }
    __syncthreads();
    if (status[w])
      return;
  }
}
// This fast entry is bound only to the same S publication whose successful
// impedance invocation rebuilt this ascending list. Generic external callers
// retain project_cone_heads. Projection-specific numeric admission is repeated.
static __device__ __forceinline__ void
project_cone_heads_verified(int w, const int *admitted, const int *nefc, const int *ne, const int *nf,
                            const int *ncon, const int *typ, const int *id, const int *dim, const real_t *mu,
                            const real_t *friction, int *out_dim, real_t *out_mu, real_t *out_friction,
                            int *status, int *item, int rows, int contacts, const int *compact_heads,
                            const int *compact_count, const int *compact_valid, const int *head_dim,
                            real_t *head_mu, real_t *head_friction, int *head_cone_valid) {
  if (threadIdx.x == 0)
    head_cone_valid[w] = 0;
  __syncthreads();
  if (compact_valid[w] != 1 || compact_count[w] < 0 || compact_count[w] > rows) {
    project_cone_heads(w, admitted, nefc, ne, nf, ncon, typ, id, dim, mu, friction, out_dim, out_mu,
                       out_friction, status, item, rows, contacts);
    return;
  }
  __shared__ int begin, compatible;
  if (threadIdx.x == 0) {
    begin = 0;
    compatible = 1;
    if (admitted[w] == 1 && !status[w]) {
      const int n = nefc[w], start = ne[w] + nf[w];
      if (n < 0 || n > rows || ne[w] < 0 || nf[w] < 0 || start < 0 || start > n) {
        status[w] = 9001;
        item[w] = 3;
      } else if (ncon[w] < 0 || ncon[w] > contacts) {
        status[w] = 9001;
        item[w] = 5;
      } else {
        int lo = 0, hi = compact_count[w];
        while (lo < hi) {
          const int mid = (lo + hi) / 2;
          if (compact_heads[w * rows + mid] < start)
            lo = mid + 1;
          else
            hi = mid;
        }
        begin = lo;
        // Raw arrays remain externally addressable: prove that this invocation's
        // complete current row partition equals the stored list before reuse.
        // A changed follower, count, dim or start falls back to the generic
        // parser, which retains its original exact error/partial-write rules.
        int cursor = 0;
        bool boundary = start == 0;
        for (int ordinal = 0; ordinal < compact_count[w] && compatible; ordinal++) {
          const int h = compact_heads[w * rows + ordinal];
          if (h != cursor || h < 0 || h >= n) {
            compatible = 0;
            break;
          }
          if (h == start)
            boundary = true;
          int d = 1;
          if (typ[w * rows + h] == 7) {
            const int c = id[w * rows + h];
            if (c < 0 || c >= ncon[w]) {
              compatible = 0;
              break;
            }
            d = dim[w * contacts + c];
            if (d != head_dim[w * rows + ordinal]) {
              compatible = 0;
              break;
            }
            if ((d != 1 && d != 3 && d != 4 && d != 6) || d > n - h) {
              compatible = 0;
              break;
            }
          }
          cursor += d;
        }
        if (cursor != n || (!boundary && start != n))
          compatible = 0;
      }
    }
  }
  __syncthreads();
  if (admitted[w] != 1 || status[w])
    return;
  if (!compatible) {
    project_cone_heads(w, admitted, nefc, ne, nf, ncon, typ, id, dim, mu, friction, out_dim, out_mu,
                       out_friction, status, item, rows, contacts);
    return;
  }
  const int rb = w * rows, cb = w * contacts, n = nefc[w], total = compact_count[w];
  // All group heads retain source order. Followers and scalar-type groups
  // never execute cone arithmetic. A block barrier protects each next batch.
  for (int base = begin; base < total; base += 32) {
    __syncthreads();
    if (threadIdx.x < 32) {
      const int lane = threadIdx.x, ordinal = base + lane;
      int code = 0, bad = -1;
      if (ordinal < total) {
        const int i = compact_heads[rb + ordinal];
        bad = i;
        if (i < ne[w] + nf[w] || i >= n) {
          code = 9004;
        } else {
          const int t = typ[rb + i];
          if (t == 7) {
            const int c = id[rb + i];
            if (c < 0 || c >= ncon[w])
              code = 9002;
            else {
              const int d = dim[cb + c];
              if (d != 1 && d != 3 && d != 4 && d != 6)
                code = 9003;
              else if (d > n - i)
                code = 9004;
              else {
                for (int j = 0; j < d; j++)
                  if (typ[rb + i + j] != 7 || id[rb + i + j] != c) {
                    code = 9004;
                    bad = i + j;
                    break;
                  }
                if (!code && (!isfinite(mu[cb + c]) || mu[cb + c] <= REAL_C(0.)))
                  code = 9006;
                if (!code)
                  for (int j = 0; j < d - 1; j++) {
                    const real_t f = friction[(cb + c) * 5 + j];
                    if (!isfinite(f) || f <= REAL_C(0.)) {
                      code = 9007;
                      break;
                    }
                  }
              }
            }
          } else if (t == 5) {
            const int c = id[rb + i];
            if (c < 0 || c >= ncon[w])
              code = 9002;
            else if (dim[cb + c] != 1)
              code = 9003;
          } else if (t < 0 || t > 6)
            code = 9005;
        }
      }
      const int first = libero_warp::min(code ? lane : libero_warp::no_index);
      const int winner = first == libero_warp::no_index ? 0 : first;
      const int rejection = __shfl_sync(libero_warp::mask, code, winner);
      const int rejection_item = __shfl_sync(libero_warp::mask, bad, winner);
      if (lane == 0 && first != libero_warp::no_index) {
        status[w] = rejection;
        item[w] = rejection_item;
      }
    }
    __syncthreads();
    if (status[w])
      return;
  }
  __syncthreads();
  for (int ordinal = begin + threadIdx.x; ordinal < total; ordinal += blockDim.x) {
    const int i = compact_heads[rb + ordinal];
    if (typ[rb + i] != 7)
      continue;
    const int c = id[rb + i], d = dim[cb + c];
    out_dim[rb + i] = d;
    out_mu[rb + i] = mu[cb + c];
    head_mu[rb + ordinal] = mu[cb + c];
    // Wide contiguous records stay within the validated cone prefix. The
    // transposed private head metadata retains its existing scalar layout.
    int j = 0;
    for (; j + 1 < d - 1; j += 2) {
      const real2_t f = libero_vec::load_pair(friction + (cb + c) * 5 + j);
      libero_vec::store_pair(out_friction + (rb + i) * 5 + j, f.x, f.y);
      head_friction[(w * 5 + j) * rows + ordinal] = f.x;
      head_friction[(w * 5 + j + 1) * rows + ordinal] = f.y;
    }
    if (j < d - 1) {
      const real_t f = friction[(cb + c) * 5 + j];
      out_friction[(rb + i) * 5 + j] = f;
      head_friction[(w * 5 + j) * rows + ordinal] = f;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    head_cone_valid[w] = 1;
}

} // namespace cf_s

} // namespace physics_constraint_projection

// Keep the shape-specific policy local to this implementation.

namespace physics_constraint_reference::cf_s {
static __device__ void reference_gate(int w, int32_t *active, int32_t *status, int32_t *committed,
                                      int32_t *counts, real_t *qvel, int64_t *epoch, int64_t *expected,
                                      int32_t *ready, int32_t rows, int32_t nv);
}

namespace physics_constraint_reference::cf_s {
static __device__ void commit_reference(int w, int32_t *ready, int32_t *status, int32_t *committed,
                                        int32_t *counts, real_t *sv, real_t *sa, int64_t *epoch, real_t *pv,
                                        real_t *pa, int64_t *ref_epoch, int32_t rows);
}

namespace physics_constraint_reference::cf_s {
static __device__ void reference(int w, int32_t *committed, int32_t *counts, real_t *efc_J, real_t *kbip,
                                 real_t *efc_pos, real_t *efc_margin, real_t *qvel, int64_t *qvel_epoch,
                                 real_t *vel, real_t *aref, int64_t *ref_epoch, int32_t rows, int32_t nv);
}

namespace physics_constraint_reference::cf_s {
constexpr int CF_S_DX_SM = 800;
static __device__ __forceinline__ bool s1_block_any(bool bad) {
  __shared__ int votes[32], result;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  bool v = libero_warp::any(bad);
  if (lane == 0)
    votes[warp] = v;
  __syncthreads();
  if (warp == 0) {
    bool all = libero_warp::any(lane < (blockDim.x >> 5) && votes[lane]);
    if (lane == 0)
      result = all;
  }
  __syncthreads();
  return result != 0;
}

static __device__ __forceinline__ void reference_gate(int w, int32_t *active, int32_t *status,
                                                      int32_t *committed, int32_t *counts, real_t *qvel,
                                                      int64_t *epoch, int64_t *expected, int32_t *ready,
                                                      int32_t rows, int32_t nv) {
  if (threadIdx.x == 0)
    ready[w] = 0;
  __syncthreads();
  if (!active[w])
    return;
  if (status[w]) {
    if (threadIdx.x == 0)
      committed[w] = 0;
    return;
  }
  if (active[w] != 1) {
    if (threadIdx.x == 0) {
      status[w] = 1;
      committed[w] = 0;
    }
    return;
  }
  if (committed[w] != 1) {
    if (threadIdx.x == 0) {
      status[w] = 9;
      committed[w] = 0;
    }
    return;
  }
  int n = counts[w * 5 + 3];
  if (n < 0 || n > rows) {
    if (threadIdx.x == 0) {
      status[w] = 2;
      committed[w] = 0;
    }
    return;
  }
  if (!n)
    return;
  if (epoch[w] != expected[w]) {
    if (threadIdx.x == 0) {
      status[w] = 9;
      committed[w] = 0;
    }
    return;
  }
  bool bad = false;
  for (int j = threadIdx.x; j < nv; j += blockDim.x)
    bad |= !isfinite(qvel[w * nv + j]);
  if (s1_block_any(bad)) {
    if (threadIdx.x == 0) {
      status[w] = 4;
      committed[w] = 0;
    }
    return;
  }
  if (threadIdx.x == 0)
    ready[w] = 1;
}

static __device__ __forceinline__ void commit_reference(int w, int32_t *ready, int32_t *status,
                                                        int32_t *committed, int32_t *counts, real_t *sv,
                                                        real_t *sa, int64_t *epoch, real_t *pv, real_t *pa,
                                                        int64_t *ref_epoch, int32_t rows) {
  if (ready[w] != 1 || status[w])
    return;
  int n = counts[w * 5 + 3];
  __shared__ int reference_nonfinite;
  __shared__ int reference_warp_nonfinite[32];
  bool thread_nonfinite = false;
  thread_nonfinite |= libero_vec::any_nonfinite_block_lane(sv + w * rows, n);
  thread_nonfinite |= libero_vec::any_nonfinite_block_lane(sa + w * rows, n);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  bool warp_nonfinite = libero_warp::any(thread_nonfinite);
  if (lane == 0)
    reference_warp_nonfinite[warp] = warp_nonfinite;
  __syncthreads();
  if (warp == 0) {
    bool block_nonfinite = libero_warp::any(lane < (blockDim.x >> 5) && reference_warp_nonfinite[lane] != 0);
    if (lane == 0)
      reference_nonfinite = block_nonfinite;
  }
  __syncthreads();
  if (reference_nonfinite) {
    if (threadIdx.x == 0) {
      status[w] = 4;
      committed[w] = ready[w] = 0;
    }
    return;
  }
  libero_vec::copy_block(sv + w * rows, pv + w * rows, n);
  libero_vec::copy_block(sa + w * rows, pa + w * rows, n);
  __syncthreads();
  if (threadIdx.x == 0)
    ref_epoch[w] = epoch[w];
}

using ReferenceGemm =
    decltype(cublasdx::Size<32, 1, SubstepShape::nv>() + cublasdx::Precision<real_t>() +
             cublasdx::Type<cublasdx::type::real>() +
             cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::col_major>() +
             cublasdx::Function<cublasdx::function::MM>() + cublasdx::SM<CF_S_DX_SM>() + cublasdx::Block() +
             cublasdx::BlockDim<64>());
static __device__ __forceinline__ void reference(int w, int32_t *committed, int32_t *counts, real_t *efc_J,
                                                 real_t *kbip, real_t *efc_pos, real_t *efc_margin,
                                                 real_t *qvel, int64_t *qvel_epoch, real_t *vel, real_t *aref,
                                                 int64_t *ref_epoch, int32_t rows, int32_t nv) {
  if (committed[w] != 1)
    return;
  constexpr int tile = 32;
  __shared__ __align__(16) char storage[cublasdx::get_shared_storage_size<ReferenceGemm>()];
  auto [pa, pb, pc] = cublasdx::slice_shared_memory<ReferenceGemm>(storage);
  auto A = cublasdx::make_tensor(pa, ReferenceGemm::get_layout_smem_a());
  auto B = cublasdx::make_tensor(pb, ReferenceGemm::get_layout_smem_b());
  auto C = cublasdx::make_tensor(pc, ReferenceGemm::get_layout_smem_c());
  int total = counts[w * 5 + 3];
  // B is invariant for all row tiles. Keep it resident in the same shared
  // allocation instead of reloading qvel once per tile.
  for (int k = threadIdx.x; k < SubstepShape::nv; k += blockDim.x)
    B(k, 0) = qvel[w * SubstepShape::nv + k];
  __syncthreads();
  for (int base = 0; base < total; base += tile) {
    // Scatter wide global loads through tensor accessors; keep library layout.
    for (int z = 2 * threadIdx.x; z < tile * SubstepShape::nv; z += 2 * blockDim.x) {
      const int r = z / SubstepShape::nv, k = z % SubstepShape::nv, next = z + 1,
                r1 = next / SubstepShape::nv, k1 = next % SubstepShape::nv;
      if (next < tile * SubstepShape::nv && base + r1 < total) {
        const real2_t v = libero_vec::load_pair(efc_J + (w * rows + base) * SubstepShape::nv + z);
        A(r, k) = v.x;
        A(r1, k1) = v.y;
      } else {
        A(r, k) = base + r < total ? efc_J[(w * rows + base) * SubstepShape::nv + z] : REAL_C(0.);
        if (next < tile * SubstepShape::nv)
          A(r1, k1) = base + r1 < total ? efc_J[(w * rows + base) * SubstepShape::nv + next] : REAL_C(0.);
      }
    }
    for (int r = threadIdx.x; r < tile; r += blockDim.x)
      C(r, 0) = REAL_C(0.);
    __syncthreads();
    ReferenceGemm().execute(REAL_C(1.), A, B, REAL_C(0.), C);
    __syncthreads();
    for (int r = threadIdx.x; r < tile && base + r < total; r += blockDim.x) {
      int g = w * rows + base + r;
      real_t acc = C(r, 0);
      vel[g] = acc;
      aref[g] = -kbip[g * 4 + 1] * acc - kbip[g * 4] * kbip[g * 4 + 2] * (efc_pos[g] - efc_margin[g]);
    }
    __syncthreads();
  }
}
} // namespace physics_constraint_reference::cf_s


// Formal single-substep Euler implementation. All bodies are visible to the
// fused kernel; the standalone checked Euler remains in cuda/integration/.
namespace physics_euler {
// Included inside physics_euler. Tree-packed FP64 LDL and solves.
namespace libero_packed_mass {
// Every packed row belongs to the parent=-1 root reached by its ancestors.
// A root and its descendants form an independent matrix block. MuJoCo orders
// these blocks contiguously; descriptors that do not satisfy that property
// keep the original serial implementation below. No additional model ABI or
// capacity-sized shared metadata is required.
struct ForestPlan {
  int roots;
  bool contiguous;
  const int *bounds = nullptr;
  const int *row_lengths = nullptr;
};
static __device__ __forceinline__ ForestPlan inspect_forest_block(int nv, const int *parent) {
  __shared__ int root_count, valid_layout;
  if (threadIdx.x == 0) {
    int first = 0, roots = 0;
    bool valid = nv > 0;
    for (int i = 0; i < nv; ++i) {
      if (parent[i] == -1) {
        first = i;
        ++roots;
      } else if (parent[i] < first || parent[i] >= i)
        valid = false;
    }
    root_count = roots;
    valid_layout = valid && roots > 0;
  }
  __syncthreads();
  return {root_count, valid_layout != 0};
}
// Prepared, owner-private int32 descriptor layout is documented by the pure
// CPU build_mass_descriptor(). Shadow comparison is exact on every call;
// a changed source always takes the original inspection and range traversal.
static __device__ __forceinline__ ForestPlan prepared_forest_block(int nv, int nm, const int *madr,
                                                                   const int *parent, const int *descriptor,
                                                                   int descriptor_size) {
  // v2 is issued only by the exact measured cold topology selector. Legacy
  // v1/generic descriptors preserve the original inspect/range traversal.
  if (!descriptor || descriptor_size < 5 || descriptor[0] != 2 || descriptor[1] != nv || descriptor[2] != nm)
    return inspect_forest_block(nv, parent);
  const int roots = descriptor[3];
  if (roots <= 1 || roots > nv || descriptor_size != 6 + roots + 3 * nv || !descriptor[4])
    return inspect_forest_block(nv, parent);
  const int *bounds = descriptor + 5;
  const int *source_parent = bounds + roots + 1;
  const int *source_madr = source_parent + nv;
  const int *lengths = source_madr + nv;
  bool mismatch = bounds[0] != 0 || bounds[roots] != nv;
  for (int i = threadIdx.x; i < nv; i += blockDim.x)
    mismatch |= parent[i] != source_parent[i] || madr[i] != source_madr[i];
  if (__syncthreads_or(mismatch))
    return inspect_forest_block(nv, parent);
  return {roots, true, bounds, lengths};
}
static __device__ __forceinline__ void forest_range(int root, int nv, const int *parent, int &begin,
                                                    int &end) {
  int ordinal = -1;
  begin = end = nv;
  for (int i = 0; i < nv; ++i)
    if (parent[i] == -1) {
      ++ordinal;
      if (ordinal == root)
        begin = i;
      else if (ordinal == root + 1) {
        end = i;
        break;
      }
    }
}
static __device__ __forceinline__ void forest_range(ForestPlan plan, int root, int nv, const int *parent,
                                                    int &begin, int &end) {
  if (plan.bounds) {
    begin = plan.bounds[root];
    end = plan.bounds[root + 1];
  } else
    forest_range(root, nv, parent, begin, end);
}
template <bool Checked = true>
static __device__ __forceinline__ bool
factor(const real_t *qm, real_t *ld, real_t *inv, real_t *sqrt_inv, int *pivot, int &warnings, int &last,
       int nv, int nm, const int *madr, const int *parent, bool reciprocal_pivot = false) {
  reciprocal_pivot =
      reciprocal_pivot && ((nv == 21 && nm == 57) || (nv == 43 && nm == 115) || (nv == 57 && nm == 153));
  bool bad = false;
  libero_vec::copy_serial(qm, ld, nm);
  for (int k = nv - 1; k >= 0; --k) {
    const int kk = madr[k];
    if (ld[kk] < REAL_C(1.0e-15)) {
      pivot[k] = 1;
      ++warnings;
      last = k;
      ld[kk] = REAL_C(1.0e-15);
    }
    const real_t reciprocal = reciprocal_pivot ? REAL_C(1.0) / ld[kk] : REAL_C(0.0);
    int ki = kk + 1;
    for (int i = parent[k]; i >= 0; i = parent[i], ++ki) {
      const real_t multiplier = reciprocal_pivot ? ld[ki] * reciprocal : ld[ki] / ld[kk];
      if constexpr (Checked)
        bad |= !isfinite(multiplier);
      const int len = (i < nv - 1 ? madr[i + 1] : nm) - madr[i];
      for (int a = 0; a < len; ++a) {
        const int dest = madr[i] + a;
        ld[dest] += ld[ki + a] * (-multiplier);
        if constexpr (Checked)
          bad |= !isfinite(ld[dest]);
      }
      ld[ki] = multiplier;
    }
  }
  for (int i = 0; i < nv; ++i) {
    const real_t diagonal = ld[madr[i]];
    inv[i] = REAL_C(1.0) / diagonal;
    if (sqrt_inv)
      sqrt_inv[i] = REAL_C(1.0) / sqrt(diagonal);
    if constexpr (Checked)
      bad |= !isfinite(inv[i]) || (sqrt_inv && !isfinite(sqrt_inv[i]));
  }
  return !bad;
}
template <bool Checked = true>
static __device__ __forceinline__ bool solve_inplace(real_t *x, const real_t *ld, const real_t *inv, int nv,
                                                     const int *madr, const int *parent) {
  bool bad = false;
  // A zero descendant is deliberately skipped: ignored nonfinite products
  // and signed-zero behavior remain those of the original sparse solve.
  for (int i = nv - 1; i >= 0; --i) {
    if (x[i] == REAL_C(0.0))
      continue;
    int a = madr[i] + 1;
    for (int j = parent[i]; j >= 0; j = parent[j], ++a) {
      x[j] -= ld[a] * x[i];
      if constexpr (Checked)
        bad |= !isfinite(x[j]);
    }
  }
  for (int i = 0; i < nv; ++i) {
    x[i] *= inv[i];
    if constexpr (Checked)
      bad |= !isfinite(x[i]);
  }
  for (int i = 0; i < nv; ++i) {
    int a = madr[i] + 1;
    for (int j = parent[i]; j >= 0; j = parent[j], ++a) {
      x[i] -= ld[a] * x[j];
      if constexpr (Checked)
        bad |= !isfinite(x[i]);
    }
  }
  return !bad;
}

// All block threads enter. Each component has exactly one owner. Within a
// component the pivot, ancestor, and packed-row operation order is identical
// to factor(); different components never read or write one another's rows.
template <bool Checked = true>
static __device__ __forceinline__ bool
factor_block(const real_t *qm, real_t *ld, real_t *inv, real_t *sqrt_inv, int *pivot, int &warnings,
             int &last, int nv, int nm, const int *madr, const int *parent, ForestPlan &plan,
             const int *descriptor = nullptr, int descriptor_size = 0, bool reciprocal_pivot = false) {
  reciprocal_pivot =
      reciprocal_pivot && ((nv == 21 && nm == 57) || (nv == 43 && nm == 115) || (nv == 57 && nm == 153));
  plan = prepared_forest_block(nv, nm, madr, parent, descriptor, descriptor_size);
  __shared__ int failed, new_warnings, last_warning;
  if (threadIdx.x == 0) {
    failed = 0;
    new_warnings = 0;
    last_warning = nv;
  }
  __syncthreads();
  if (!plan.contiguous || plan.roots <= 1) {
    if (threadIdx.x == 0)
      failed = !factor<Checked>(qm, ld, inv, sqrt_inv, pivot, warnings, last, nv, nm, madr, parent,
                                reciprocal_pivot);
    __syncthreads();
    return failed == 0;
  }
  libero_vec::copy_block(qm, ld, nm);
  __syncthreads();
  bool bad = false;
  int count = 0, smallest = nv;
  for (int root = threadIdx.x; root < plan.roots; root += blockDim.x) {
    int begin, end;
    forest_range(plan, root, nv, parent, begin, end);
    for (int k = end - 1; k >= begin; --k) {
      const int kk = madr[k];
      if (ld[kk] < REAL_C(1.0e-15)) {
        pivot[k] = 1;
        ++count;
        smallest = k < smallest ? k : smallest;
        ld[kk] = REAL_C(1.0e-15);
      }
      const real_t reciprocal = reciprocal_pivot ? REAL_C(1.0) / ld[kk] : REAL_C(0.0);
      int ki = kk + 1;
      for (int i = parent[k]; i >= 0; i = parent[i], ++ki) {
        const real_t multiplier = reciprocal_pivot ? ld[ki] * reciprocal : ld[ki] / ld[kk];
        if constexpr (Checked)
          bad |= !isfinite(multiplier);
        const int len = plan.row_lengths ? plan.row_lengths[i] : (i < nv - 1 ? madr[i + 1] : nm) - madr[i];
        for (int a = 0; a < len; ++a) {
          const int dest = madr[i] + a;
          ld[dest] += ld[ki + a] * (-multiplier);
          if constexpr (Checked)
            bad |= !isfinite(ld[dest]);
        }
        ld[ki] = multiplier;
      }
    }
    for (int i = begin; i < end; ++i) {
      const real_t diagonal = ld[madr[i]];
      inv[i] = REAL_C(1.0) / diagonal;
      if (sqrt_inv)
        sqrt_inv[i] = REAL_C(1.0) / sqrt(diagonal);
      if constexpr (Checked)
        bad |= !isfinite(inv[i]) || (sqrt_inv && !isfinite(sqrt_inv[i]));
    }
  }
  if (bad)
    atomicOr(&failed, 1);
  if (count) {
    atomicAdd(&new_warnings, count);
    atomicMin(&last_warning, smallest);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    warnings += new_warnings;
    if (new_warnings)
      last = last_warning;
  }
  __syncthreads();
  return failed == 0;
}

template <bool Checked = true>
static __device__ __forceinline__ bool solve_component(real_t *x, const real_t *ld, const real_t *inv,
                                                       int begin, int end, const int *madr,
                                                       const int *parent) {
  bool bad = false;
  for (int i = end - 1; i >= begin; --i) {
    if (x[i] == REAL_C(0.0))
      continue;
    int a = madr[i] + 1;
    for (int j = parent[i]; j >= 0; j = parent[j], ++a) {
      x[j] -= ld[a] * x[i];
      if constexpr (Checked)
        bad |= !isfinite(x[j]);
    }
  }
  for (int i = begin; i < end; ++i) {
    x[i] *= inv[i];
    if constexpr (Checked)
      bad |= !isfinite(x[i]);
  }
  for (int i = begin; i < end; ++i) {
    int a = madr[i] + 1;
    for (int j = parent[i]; j >= 0; j = parent[j], ++a) {
      x[i] -= ld[a] * x[j];
      if constexpr (Checked)
        bad |= !isfinite(x[i]);
    }
  }
  return !bad;
}

// Cross product of independent RHS and independent mass components. No new
// floating-point reduction is introduced. Return value is block-uniform and
// includes the failure of every component; callers retain their clear gate.
template <bool Checked = true>
static __device__ __forceinline__ bool solve_rhs_block(const real_t *rhs, real_t *solution, const real_t *ld,
                                                       const real_t *inv, int nrhs, int nv, const int *madr,
                                                       const int *parent, ForestPlan plan) {
  libero_vec::copy_block(rhs, solution, size_t(nrhs) * nv);
  __syncthreads();
  bool bad = false;
  if (plan.contiguous && plan.roots > 1) {
    for (int task = threadIdx.x; task < nrhs * plan.roots; task += blockDim.x) {
      const int r = task / plan.roots, root = task % plan.roots;
      int begin, end;
      forest_range(plan, root, nv, parent, begin, end);
      bad |= !solve_component<Checked>(solution + r * nv, ld, inv, begin, end, madr, parent);
    }
  } else {
    for (int r = threadIdx.x; r < nrhs; r += blockDim.x)
      bad |= !solve_inplace<Checked>(solution + r * nv, ld, inv, nv, madr, parent);
  }
  if constexpr (Checked)
    return __syncthreads_or(bad) == 0;
  else {
    __syncthreads();
    return true;
  }
}

// Multiple independent RHS share one completed factorization. The caller
// enters on all block threads after its factor barrier and combines the
// returned lane-local failure before the existing transaction clear/gate.
// No outer pivot/ancestor ordering, zero-descendant skip or arithmetic changes.
// Factor, input RHS and output solution must not overlap.
static __device__ __forceinline__ bool solve_rhs_lane(const real_t *rhs, real_t *solution, const real_t *ld,
                                                      const real_t *inv, int nrhs, int nv, const int *madr,
                                                      const int *parent) {
  // Stage only active RHS cells with consecutive block lanes. No inactive
  // tail is touched; the owning caller already zeroes those cells.
  libero_vec::copy_block(rhs, solution, size_t(nrhs) * nv);
  __syncthreads();
  const int r = int(threadIdx.x);
  if (r >= nrhs)
    return true;
  return solve_inplace(solution + r * nv, ld, inv, nv, madr, parent);
}
} // namespace libero_packed_mass

namespace cf_euler {
// Included inside physics_euler::cf_euler after the packed-mass helpers.
// CPU reference: MuJoCo 2.3.7 mj_EulerSkip, mj_advance, mju_quatIntegrate.
// The fused entry has already performed checkPos/checkVel/checkAcc and owns
// disjoint state/scratch views. Only one RHS is needed by mj_EulerSkip.
static __device__ real_t trig(real_t z, bool sine) {
  const real_t sc[23] = {REAL_C(1.),
                         -REAL_C(.16666666666666666),
                         REAL_C(.008333333333333333),
                         -REAL_C(.0001984126984126984),
                         REAL_C(2.7557319223985893e-6),
                         -REAL_C(2.505210838544172e-8),
                         REAL_C(1.6059043836821613e-10),
                         -REAL_C(7.647163731819816e-13),
                         REAL_C(2.8114572543455206e-15),
                         -REAL_C(8.22063524662433e-18),
                         REAL_C(1.9572941063391263e-20),
                         -REAL_C(3.868170170630684e-23),
                         REAL_C(6.446950284384474e-26),
                         -REAL_C(9.183689863795546e-29),
                         REAL_C(1.1309962886447716e-31),
                         -REAL_C(1.216125041553518e-34),
                         REAL_C(1.151633562077195e-37),
                         -REAL_C(9.67759295863189e-41),
                         REAL_C(7.265460179153071e-44),
                         -REAL_C(4.902469756513544e-47),
                         REAL_C(2.9893108271424046e-50),
                         -REAL_C(1.6552108677421951e-53),
                         REAL_C(8.359650847182804e-57)};
  const real_t cc[23] = {REAL_C(1.),
                         -REAL_C(.5),
                         REAL_C(.041666666666666664),
                         -REAL_C(.001388888888888889),
                         REAL_C(2.48015873015873e-5),
                         -REAL_C(2.755731922398589e-7),
                         REAL_C(2.08767569878681e-9),
                         -REAL_C(1.1470745597729725e-11),
                         REAL_C(4.779477332387385e-14),
                         -REAL_C(1.5619206968586225e-16),
                         REAL_C(4.110317623312165e-19),
                         -REAL_C(8.896791392450574e-22),
                         REAL_C(1.6117375710961184e-24),
                         -REAL_C(2.4795962632247976e-27),
                         REAL_C(3.279889237069838e-30),
                         -REAL_C(3.7699876288159054e-33),
                         REAL_C(3.8003907548547434e-36),
                         -REAL_C(3.387157535521162e-39),
                         REAL_C(2.6882202662866363e-42),
                         -REAL_C(1.911963205040282e-45),
                         REAL_C(1.2256174391283858e-48),
                         -REAL_C(7.117406731291439e-52),
                         REAL_C(3.7618428812322616e-55)};
  const real_t *c = sine ? sc : cc;
  real_t p = c[22], zz = z * z;
  for (int i = 21; i >= 0; i--)
    p = p * zz + c[i];
  return sine ? z * p : p;
}

static __device__ __forceinline__ void gate_core_block(int w, const int *intent, const int64_t *epoch,
                                                       const int *forward_valid,
                                                       const int64_t *forward_written,
                                                       const int *forward_error, int *first, int *active) {
  if (threadIdx.x != 0)
    return;
  active[w] = 0;
  if (intent[w] == 0 || first[w])
    return;
  if (intent[w] != 1) {
    first[w] = 1;
    return;
  }
  if (epoch[w] == INT64_MAX) {
    first[w] = 9;
    return;
  }
  if (forward_error[w]) {
    first[w] = forward_error[w];
    return;
  }
  active[w] = forward_valid[w] == 1 && forward_written[w] == epoch[w];
}

// One joint owns its qpos span. Keep the accepted normalization thresholds,
// quaternion multiplication and small-angle trigonometry; omit trace stores.
static __device__ void integrate_joint_core(int joint, const int *type, const int *qpos_address,
                                            const int *dof_address, real_t *pos, const real_t *vel,
                                            real_t dt) {
  int pa = qpos_address[joint], va = dof_address[joint];
  const int kind = type[joint];
  if (kind == 0) {
    const real2_t p = libero_vec::load_pair(pos + pa);
    const real2_t v = libero_vec::load_pair(vel + va);
    libero_vec::store_pair(pos + pa, p.x + dt * v.x, p.y + dt * v.y);
    pos[pa + 2] += dt * vel[va + 2];
    pa += 3;
    va += 3;
  } else if (kind == 2 || kind == 3) {
    pos[pa] += dt * vel[va];
  }
  if (kind != 0 && kind != 1)
    return;
  const real2_t v = libero_vec::load_pair(vel + va);
  real_t x = v.x, y = v.y, z = vel[va + 2];
  const real_t xx = x * x, yy = y * y, zz = z * z, sxy = xx + yy;
  const real_t length = sqrt(sxy + zz);
  if (length < REAL_C(1e-15)) {
    x = REAL_C(1.0);
    y = z = REAL_C(0.0);
  } else {
    const real_t inv = REAL_C(1.0) / length;
    x *= inv;
    y *= inv;
    z *= inv;
  }
  const real_t angle = dt * length;
  real_t c = REAL_C(1.0), s = REAL_C(0.0);
  if (angle != REAL_C(0.0)) {
    const real_t half = REAL_C(0.5) * angle;
    s = fabs(half) <= 4 ? trig(half, true) : sin(half);
    c = fabs(half) <= 4 ? trig(half, false) : cos(half);
  }
  real_t b[4] = {c, x * s, y * s, z * s};
  if (angle == REAL_C(0.0))
    b[1] = b[2] = b[3] = REAL_C(0.0);
  const real2_t p01 = libero_vec::load_pair(pos + pa);
  const real2_t p23 = libero_vec::load_pair(pos + pa + 2);
  const real_t a0 = p01.x, a1 = p01.y, a2 = p23.x, a3 = p23.y;
  real_t q[4] = {a0 * b[0] - a1 * b[1] - a2 * b[2] - a3 * b[3], a0 * b[1] + a1 * b[0] + a2 * b[3] - a3 * b[2],
                 a0 * b[2] - a1 * b[3] + a2 * b[0] + a3 * b[1],
                 a0 * b[3] + a1 * b[2] - a2 * b[1] + a3 * b[0]};
  const real_t s0 = q[0] * q[0], s1 = q[1] * q[1], s2 = q[2] * q[2], s3 = q[3] * q[3];
  const real_t s01 = s0 + s1, s012 = s01 + s2, qlen = sqrt(s012 + s3);
  if (qlen < REAL_C(1e-15)) {
    q[0] = REAL_C(1.0);
    q[1] = q[2] = q[3] = REAL_C(0.0);
  } else {
    const real_t inv = REAL_C(1.0) / qlen;
    for (int i = 0; i < 4; ++i)
      q[i] *= inv;
  }
  libero_vec::store_pair(pos + pa, q[0], q[1]);
  libero_vec::store_pair(pos + pa + 2, q[2], q[3]);
}

static __device__ __forceinline__ void
evaluate_core(int w, const int *active, real_t *qpos, real_t *qvel, real_t *time, const real_t *mass,
              const real_t *smooth, const real_t *constraint, const real_t *qacc, const real_t *damping,
              const int *mass_address, const int *parent, const int *joint_type, const int *qpos_address,
              const int *dof_address, real_t *qh, real_t *qh_inv, int *count, int *mass_active, int *damped,
              int *status, int *mass_status, int *warnings, int *last_warning, real_t dt, int nq, int nv,
              int nm, int nj, int qacc_stride, const int *joint_quat, const int *type_offsets,
              const int *type_indices) {
  auto *shared = flash_libero::substep_shared_data();
  int &work_warnings = *flash_libero::shared_at<int>(shared, flash_libero::EulerLayout::warnings);
  int &work_last_warning = *flash_libero::shared_at<int>(shared, flash_libero::EulerLayout::last_warning);
  real_t *work_mass = flash_libero::shared_at<real_t>(shared, flash_libero::EulerLayout::mass);
  real_t *work_rhs = flash_libero::shared_at<real_t>(shared, flash_libero::EulerLayout::rhs);
  int *work_pivot = flash_libero::shared_at<int>(shared, flash_libero::EulerLayout::pivot);
  real_t *work_factor = flash_libero::shared_at<real_t>(shared, flash_libero::EulerLayout::factor);
  real_t *work_inv = flash_libero::shared_at<real_t>(shared, flash_libero::EulerLayout::inv);
  real_t *work_solution = flash_libero::shared_at<real_t>(shared, flash_libero::EulerLayout::solution);
  real_t *work_vel = flash_libero::shared_at<real_t>(shared, flash_libero::EulerLayout::vel);
  real_t *work_pos = flash_libero::shared_at<real_t>(shared, flash_libero::EulerLayout::pos);
  if (threadIdx.x == 0) {
    status[w] = mass_status[w] = 0;
    count[w] = 1;
    mass_active[w] = damped[w] = 0;
    warnings[w] = 0;
    last_warning[w] = -1;
    work_warnings = 0;
    work_last_warning = -1;
  }
  __syncthreads();
  if (active[w] != 1)
    return;
  bool positive = false;
  for (int i = threadIdx.x; i < nv; i += blockDim.x)
    positive |= damping[i] > REAL_C(0.0);
  const bool implicit = __syncthreads_or(positive);
  if (threadIdx.x == 0)
    mass_active[w] = damped[w] = implicit;
  if (implicit) {
    libero_vec::copy_block(mass + w * nm, work_mass, nm);
    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
      work_rhs[i] = smooth[w * nv + i] + constraint[w * nv + i];
      work_pivot[i] = 0;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nv; i += blockDim.x)
      work_mass[mass_address[i]] += dt * damping[i];
    __syncthreads();
    libero_packed_mass::ForestPlan forest;
    // Keep the factor's pivot clamp and warning behavior. Full input and
    // intermediate finite scans belong to the checked component entry.
    const bool factor_ok = libero_packed_mass::factor_block<false>(
        work_mass, work_factor, work_inv, nullptr, work_pivot, work_warnings, work_last_warning, nv, nm,
        mass_address, parent, forest);
    const bool solve_ok = libero_packed_mass::solve_rhs_block<false>(
        work_rhs, work_solution, work_factor, work_inv, 1, nv, mass_address, parent, forest);
    if (threadIdx.x == 0) {
      warnings[w] = work_warnings;
      last_warning[w] = work_last_warning;
      mass_status[w] = (!factor_ok || !solve_ok) ? 4 : (work_warnings ? 5 : 0);
      if (!factor_ok || !solve_ok)
        status[w] = 14;
    }
    __syncthreads();
    if (status[w])
      return;
  }
  const real_t *acceleration = implicit ? work_solution : qacc + w * qacc_stride;
  for (int i = threadIdx.x; i < nv; i += blockDim.x)
    work_vel[i] = qvel[w * nv + i] + dt * acceleration[i];
  libero_vec::copy_block(qpos + w * nq, work_pos, nq);
  __syncthreads();
  if (joint_quat) {
    if (type_offsets && type_indices) {
      for (int kind = 0; kind < 4; ++kind)
        for (int i = type_offsets[kind] + threadIdx.x; i < type_offsets[kind + 1]; i += blockDim.x)
          integrate_joint_core(type_indices[i], joint_type, qpos_address, dof_address, work_pos, work_vel,
                               dt);
    } else
      for (int j = threadIdx.x; j < nj; j += blockDim.x)
        integrate_joint_core(j, joint_type, qpos_address, dof_address, work_pos, work_vel, dt);
  } else if (threadIdx.x == 0) {
    for (int j = 0; j < nj; ++j)
      integrate_joint_core(j, joint_type, qpos_address, dof_address, work_pos, work_vel, dt);
  }
  __syncthreads();
  libero_vec::copy_block(work_pos, qpos + w * nq, nq);
  libero_vec::copy_block(work_vel, qvel + w * nv, nv);
  if (implicit) {
    libero_vec::copy_block(work_factor, qh + w * nm, nm);
    libero_vec::copy_block(work_inv, qh_inv + w * nv, nv);
  }
  __syncthreads();
  if (threadIdx.x == 0)
    time[w] += dt;
  __syncthreads();
}

static __device__ __forceinline__ void publish(int w, const int *active, const int *status, int64_t *epoch,
                                               int64_t *written, int *valid, int *first, int *fv) {
  valid[w] = 0;
  if (active[w] != 1)
    return;
  if (status[w]) {
    if (!first[w])
      first[w] = 100 + status[w];
    return;
  }
  epoch[w]++;
  written[w] = epoch[w];
  valid[w] = 1;
  fv[w] = 0;
}

static __device__ __forceinline__ void expected(int w, const int *intent, int64_t *expected) {
  if (intent[w] == 1 && expected[w] < INT64_MAX)
    expected[w]++;
}

static __device__ __forceinline__ void count(int w, const int *valid, int64_t *counter) {
  if (valid[w] == 1)
    counter[w]++;
}
} // namespace cf_euler
} // namespace physics_euler

namespace physics_force {
static __device__ void d_gate(int w, int *active, int *error, int64_t *state, int64_t *source, int64_t *epoch,
                              int *valid, int64_t *written, int *gate, int require);
}

namespace physics_force {
static __device__ void d_force_gate(int w, int *valid, int *gate);
}

namespace physics_force {
static __device__ void d_actuation_finish(int w, int *gate, int *status, int *first);
}

namespace physics_force {
static __device__ void d_copy_admitted_block(int w, int *gate, real_t *source, real_t *target, int width);
}

namespace physics_force {
static __device__ void d_commit_smooth_shared(int w, int *active, int *gate, int *status, int64_t *epoch,
                                              int *error, real_t *wf, real_t *wa, real_t *pf, real_t *pa,
                                              int *valid, int64_t *written, int *repaired, int width);
}

namespace physics_force {
// Collective counterpart. All lanes enter; the caller supplies the exit barrier.
static __device__ void d_commit_smooth_shared_block(int w, int *active, int *gate, int *status,
                                                    int64_t *epoch, int *error, real_t *pf, real_t *pa,
                                                    int *valid, int64_t *written, int *repaired, int width);
} // namespace physics_force

namespace physics_force {
static __device__ void
d_actuation_count(int w, real_t *length, real_t *moment, real_t *velocity, real_t *ctrl, int *active,
                  int *vel_status, real_t *gain, real_t *bias0, real_t *bias1, real_t *bias2, int *bias_type,
                  real_t *ctrl_lo, real_t *ctrl_hi, real_t *force_lo, real_t *force_hi, int *force_limited,
                  int disable_flags, real_t *force, real_t *force_pre, real_t *ctrl_used, int *ctrl_route,
                  int *force_route, real_t *qfrc_actuator, int *status, int *warn_badctrl_count,
                  int *warn_badctrl_lastinfo, int *warning_number, int *warning_lastinfo, int nv, int nu);
}

namespace physics_force {
static __device__ void
d_smooth_chain(int w, const real_t *passive, const real_t *bias, const real_t *applied,
               const real_t *actuator, const int *active, int *entry_status, real_t *smooth, real_t *seed,
               int nv, const real_t *xfrc, const real_t *xipos, const real_t *subtree_com, const real_t *cdof,
               const int *body_ids, const int *body_count, const int *body_rootid, const int *ancestor,
               real_t *scratch, real_t *stages, real_t *qfrc, int *d02_status, int nbody, const real_t *qm,
               const int *count, const int *madr, const int *parent, real_t *ld, real_t *inv,
               real_t *sqrt_inv, real_t *solution, int *pivot, int *solve_status, int *warning_count,
               int *warning_last, int nm, real_t *rhs, real_t *qacc_smooth, int *status,
               const real_t *source_ld, const real_t *source_inv, const real_t *source_sqrt_inv,
               const int *source_pivot, const int *source_status, const int *source_warning_count,
               const int *source_warning_last, const int *source_active, const int *position_valid,
               const int64_t *position_written, const int64_t *forward_epoch, int64_t *factor_reuse_count,
               const int *body_dof_offsets = nullptr, const int *body_dof_indices = nullptr,
               const int *mass_plan = nullptr);
}

// Maintained world-local implementation, compiled as part of fused_substep.cu.

// Shared FP64 force implementation.

namespace physics_force {

// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.

// One block owns one world. These are private working values, never public
// aliases; their lifetime extends across the force-chain device operations.

// Handwritten sparse rigid-body arithmetic. These routines receive whole
// allocations; only the selected world row is touched. The caller executes
// each collective tree routine on all threads. Its ordered tree arithmetic
// remains on lane zero; array checks and independent projections cooperate.

// Collective helpers: every thread in the world block must enter. Warp votes
// exchange predicates only; the barriers publish and protect inter-warp flags.

// Collective publication: callers enter with the complete world block. Gate
// decisions and scalar flags retain lane-zero order; payload stores complete
// before the epoch becomes visible. Shifted overlaps retain serial copy order.
static __device__ inline bool d_copy_overlap(const real_t *a, const real_t *b, int width) {
  const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  const uintptr_t bytes = uintptr_t(width) * sizeof(real_t);
  return x < y + bytes && y < x + bytes;
}
static __device__ __forceinline__ void d_copy_admitted_block(int w, int *gate, real_t *source, real_t *target,
                                                             int width) {
  __shared__ int admitted, serial;
  real_t *s = source + (long long)w * width, *t = target + (long long)w * width;
  if (threadIdx.x == 0) {
    admitted = gate[w] == 1;
    serial = d_copy_overlap(s, t, width);
  }
  __syncthreads();
  if (admitted) {
    if (serial) {
      if (threadIdx.x == 0)
        for (int i = 0; i < width; ++i)
          t[i] = s[i];
    } else
      libero_vec::copy_block(s, t, width);
  }
  __syncthreads();
}

// Qualified topology only: levels omit world body 0 and parent IDs precede
// children. A level barrier publishes every parent before its children read.

// Compressed spatial inertia multiplication, not a dense BLAS operand.

// Compute only the forward part of bias RNE while the velocity input is live.
// Its private status does not decide velocity publication or the later S gate.
// The reverse force tree and public bias commit remain after S admission.

static __device__ __forceinline__ void
d_commit_smooth_shared_block(int w, int *active, int *gate, int *status, int64_t *epoch, int *error,
                             real_t *pf, real_t *pa, int *valid, int64_t *written, int *repaired, int width) {
  __shared__ int admitted, ordered;
  auto *shared = flash_libero::substep_shared_data();
  real_t *work_force = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::force);
  real_t *work_acceleration =
      flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::acceleration);
  pf += size_t(w) * width;
  pa += size_t(w) * width;
  if (threadIdx.x == 0) {
    valid[w] = repaired[w] = 0;
    admitted = active[w] == 1 && gate[w] == 1 && !error[w];
    if (admitted && status[w] && status[w] != 25) {
      error[w] = 213;
      admitted = 0;
    }
    ordered = d_copy_overlap(work_force, pf, width) || d_copy_overlap(work_force, pa, width) ||
              d_copy_overlap(work_acceleration, pf, width) || d_copy_overlap(work_acceleration, pa, width) ||
              d_copy_overlap(pf, pa, width);
  }
  __syncthreads();
  if (admitted) {
    if (ordered) {
      if (threadIdx.x == 0)
        for (int i = 0; i < width; ++i) {
          pf[i] = work_force[i];
          pa[i] = work_acceleration[i];
        }
    } else {
      libero_vec::copy_block(work_force, pf, width);
      libero_vec::copy_block(work_acceleration, pa, width);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && admitted) {
    written[w] = epoch[w];
    repaired[w] = status[w] == 25;
    valid[w] = 1;
  }
}

// Full-world-block helpers; blockDim.x must be a positive multiple of 32.
// Warp collectives only exchange registers; shared publication uses block barriers.
namespace d_passive_detail {
static __device__ inline int first_bad(int index) {
  __shared__ int first[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int value = libero_warp::min(index);
  if (lane == 0)
    first[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < (blockDim.x + 31) / 32 ? first[lane] : libero_warp::no_index;
    value = libero_warp::min(value);
    if (lane == 0)
      first[0] = value;
  }
  __syncthreads();
  const int result = first[0];
  __syncthreads();
  return result;
}
static __device__ inline bool bad(bool value) {
  return first_bad(value ? 0 : libero_warp::no_index) != libero_warp::no_index;
}
} // namespace d_passive_detail

namespace d_passive_detail {
struct Vec {
  real_t v[3];
  __device__ real_t &operator[](int i) { return v[i]; }
  __device__ real_t operator[](int i) const { return v[i]; }
};

} // namespace d_passive_detail

namespace d_passive_detail {
// The body producer retains scalar arithmetic and diagnostic-prefix semantics.

} // namespace d_passive_detail

static __device__ inline void
d_actuation_prepare(int w, real_t *length, real_t *moment, real_t *velocity, real_t *ctrl, int *active,
                    int *vel_status, real_t *gain, real_t *bias0, real_t *bias1, real_t *bias2,
                    int *bias_type, real_t *ctrl_lo, real_t *ctrl_hi, real_t *force_lo, real_t *force_hi,
                    int *force_limited, int disable_flags, real_t *force, real_t *force_pre,
                    real_t *ctrl_used, int *ctrl_route, int *force_route, real_t *qfrc_actuator, int *status,
                    int *warn_badctrl_count, int *warn_badctrl_lastinfo, int *warning_number,
                    int *warning_lastinfo, int nv, int nu) {
  length += (long long)w * nu;
  moment += (long long)w * nu * nv;
  velocity += (long long)w * nu;
  ctrl += (long long)w * nu;
  force += (long long)w * nu;
  force_pre += (long long)w * nu;
  ctrl_used += (long long)w * nu;
  ctrl_route += (long long)w * nu;
  force_route += (long long)w * nu;
  qfrc_actuator += (long long)w * nv;
  status[w] = 0;
  warn_badctrl_count[w] = 0;
  warn_badctrl_lastinfo[w] = -1;
  for (int j = 0; j < nv; ++j)
    qfrc_actuator[j] = REAL_C(0.0);
  for (int i = 0; i < nu; ++i) {
    force[i] = REAL_C(0.0);
    force_pre[i] = REAL_C(0.0);
    ctrl_used[i] = REAL_C(0.0);
    ctrl_route[i] = 0;
    force_route[i] = 0;
  }
  if (active[w] == 0)
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  if (vel_status[w] != 0) {
    status[w] = 2;
    return;
  }
  if ((disable_flags & 1024) != 0)
    return;
  int bad_index = -1;
  for (int i = 0; i < nu; ++i) {
    real_t u = ctrl[i];
    if ((disable_flags & 128) == 0) {
      if (u < ctrl_lo[i]) {
        u = ctrl_lo[i];
        ctrl_route[i] = -1;
      } else if (u > ctrl_hi[i]) {
        u = ctrl_hi[i];
        ctrl_route[i] = 1;
      }
    }
    if (bad_index < 0 && (u != u || u > REAL_C(1e10) || u < -REAL_C(1e10)))
      bad_index = i;
  }
  if (bad_index >= 0) {
    warn_badctrl_count[w] = 1;
    warn_badctrl_lastinfo[w] = bad_index;
    ++warning_number[w];
    warning_lastinfo[w] = bad_index;
  }
  for (int i = 0; i < nu; ++i) {
    real_t u = REAL_C(0.0);
    if (bad_index < 0) {
      u = ctrl[i];
      if (ctrl_route[i] == -1)
        u = ctrl_lo[i];
      else if (ctrl_route[i] == 1)
        u = ctrl_hi[i];
    }
    ctrl_used[i] = u;
    real_t f = gain[i] * u;
    if (bias_type[i] == 1)
      f += bias0[i] + bias1[i] * length[i] + bias2[i] * velocity[i];
    force_pre[i] = f;
    if (force_limited[i] != 0) {
      if (f < force_lo[i]) {
        f = force_lo[i];
        force_route[i] = -1;
      } else if (f > force_hi[i]) {
        f = force_hi[i];
        force_route[i] = 1;
      }
    }
    force[i] = f;
  }
}

// Large-NU preparation; the ordinary small actuator family retains scalar work.
static __device__ inline void d_actuation_prepare_block(
    int w, real_t *length, real_t *moment, real_t *velocity, real_t *ctrl, int *active, int *vel_status,
    real_t *gain, real_t *bias0, real_t *bias1, real_t *bias2, int *bias_type, real_t *ctrl_lo,
    real_t *ctrl_hi, real_t *force_lo, real_t *force_hi, int *force_limited, int disable_flags, real_t *force,
    real_t *force_pre, real_t *ctrl_used, int *ctrl_route, int *force_route, real_t *qfrc_actuator,
    int *status, int *warn_badctrl_count, int *warn_badctrl_lastinfo, int *warning_number,
    int *warning_lastinfo, int nv, int nu) {
  length += (long long)w * nu;
  moment += (long long)w * nu * nv;
  velocity += (long long)w * nu;
  ctrl += (long long)w * nu;
  force += (long long)w * nu;
  force_pre += (long long)w * nu;
  ctrl_used += (long long)w * nu;
  ctrl_route += (long long)w * nu;
  force_route += (long long)w * nu;
  qfrc_actuator += (long long)w * nv;
  if (threadIdx.x == 0) {
    status[w] = 0;
    warn_badctrl_count[w] = 0;
    warn_badctrl_lastinfo[w] = -1;
  }
  libero_vec::fill_block(qfrc_actuator, nv, REAL_C(0.0));
  libero_vec::fill_block(force, nu, REAL_C(0.0));
  libero_vec::fill_block(force_pre, nu, REAL_C(0.0));
  libero_vec::fill_block(ctrl_used, nu, REAL_C(0.0));
  libero_vec::fill_block(ctrl_route, nu, 0);
  libero_vec::fill_block(force_route, nu, 0);
  __syncthreads();
  if (active[w] == 0)
    return;
  if (active[w] != 1) {
    if (threadIdx.x == 0)
      status[w] = 1;
    return;
  }
  if (vel_status[w] != 0) {
    if (threadIdx.x == 0)
      status[w] = 2;
    return;
  }
  if ((disable_flags & 1024) != 0)
    return;
  int bad_index = libero_warp::no_index;
  for (int base = 2 * threadIdx.x; base < nu; base += 2 * blockDim.x) {
    const bool pair = base + 1 < nu;
    const real2_t inputs = pair ? libero_vec::load_pair(ctrl + base) : make_real2(ctrl[base], REAL_C(0.0));
    for (int part = 0; part < (pair ? 2 : 1); ++part) {
      const int i = base + part;
      real_t u = part ? inputs.y : inputs.x;
      if ((disable_flags & 128) == 0) {
        if (u < ctrl_lo[i]) {
          u = ctrl_lo[i];
          ctrl_route[i] = -1;
        } else if (u > ctrl_hi[i]) {
          u = ctrl_hi[i];
          ctrl_route[i] = 1;
        }
      }
      if (u != u || u > REAL_C(1e10) || u < -REAL_C(1e10))
        bad_index = bad_index < i ? bad_index : i;
    }
  }
  bad_index = d_passive_detail::first_bad(bad_index);
  if (threadIdx.x == 0 && bad_index != libero_warp::no_index) {
    warn_badctrl_count[w] = 1;
    warn_badctrl_lastinfo[w] = bad_index;
    ++warning_number[w];
    warning_lastinfo[w] = bad_index;
  }
  for (int base = 2 * threadIdx.x; base < nu; base += 2 * blockDim.x) {
    const bool pair = base + 1 < nu;
    real_t fvals[2], prevals[2], used[2];
    const real2_t inputs = bad_index == libero_warp::no_index
                               ? (pair ? libero_vec::load_pair(ctrl + base) : make_real2(ctrl[base], REAL_C(0.0)))
                               : make_real2(REAL_C(0.0), REAL_C(0.0));
    for (int part = 0; part < (pair ? 2 : 1); ++part) {
      const int i = base + part;
      real_t u = REAL_C(0.0);
      if (bad_index == libero_warp::no_index) {
        u = part ? inputs.y : inputs.x;
        if (ctrl_route[i] == -1)
          u = ctrl_lo[i];
        else if (ctrl_route[i] == 1)
          u = ctrl_hi[i];
      }
      used[part] = u;
      real_t f = gain[i] * u;
      if (bias_type[i] == 1)
        f += bias0[i] + bias1[i] * length[i] + bias2[i] * velocity[i];
      prevals[part] = f;
      if (force_limited[i] != 0) {
        if (f < force_lo[i]) {
          f = force_lo[i];
          force_route[i] = -1;
        } else if (f > force_hi[i]) {
          f = force_hi[i];
          force_route[i] = 1;
        }
      }
      fvals[part] = f;
    }
    if (pair) {
      libero_vec::store_pair(ctrl_used + base, used[0], used[1]);
      libero_vec::store_pair(force_pre + base, prevals[0], prevals[1]);
      libero_vec::store_pair(force + base, fvals[0], fvals[1]);
    } else {
      ctrl_used[base] = used[0];
      force_pre[base] = prevals[0];
      force[base] = fvals[0];
    }
  }
  __syncthreads();
}

// Small-NU production uses a full warp for warning selection, then one lane
// per actuator. Every lane participates in register votes; no CTA flag array.
static __device__ inline void d_actuation_prepare_warp(
    int w, const real_t *length, const real_t *velocity, const real_t *ctrl,
    const int *active, const int *vel_status, const real_t *gain, const real_t *bias0,
    const real_t *bias1, const real_t *bias2, const int *bias_type, const real_t *ctrl_lo,
    const real_t *ctrl_hi, const real_t *force_lo, const real_t *force_hi,
    const int *force_limited, int disable_flags, real_t *work_force, real_t *force_pre,
    real_t *ctrl_used, int *ctrl_route, int *force_route, real_t *qfrc_actuator,
    int *status, int *warn_count, int *warn_last, int *warning_number,
    int *warning_last, int nv, int nu) {
  if (threadIdx.x >= 32)
    return;
  const int lane = threadIdx.x;
  const size_t u = size_t(w) * nu, v = size_t(w) * nv;
  if (lane == 0) {
    status[w] = 0;
    warn_count[w] = 0;
    warn_last[w] = -1;
  }
  for (int j = lane; j < nv; j += 32)
    qfrc_actuator[v + j] = REAL_C(0.0);
  if (lane < nu) {
    work_force[lane] = force_pre[u + lane] = ctrl_used[u + lane] = REAL_C(0.0);
    ctrl_route[u + lane] = force_route[u + lane] = 0;
  }
  if (active[w] != 1 || vel_status[w] || (disable_flags & 1024)) {
    if (lane == 0 && active[w])
      status[w] = active[w] != 1 ? 1 : (vel_status[w] ? 2 : 0);
    return;
  }
  real_t control = lane < nu ? ctrl[u + lane] : REAL_C(0.0);
  int route = 0;
  if (lane < nu && !(disable_flags & 128)) {
    if (control < ctrl_lo[lane]) {
      control = ctrl_lo[lane];
      route = -1;
    } else if (control > ctrl_hi[lane]) {
      control = ctrl_hi[lane];
      route = 1;
    }
  }
  const int bad = libero_warp::first_index(
      lane < nu && (control != control || control > REAL_C(1e10) || control < -REAL_C(1e10)), lane);
  if (lane == 0 && bad != libero_warp::no_index) {
    warn_count[w] = 1;
    warn_last[w] = bad;
    ++warning_number[w];
    warning_last[w] = bad;
  }
  if (lane < nu) {
    ctrl_route[u + lane] = route;
    if (bad != libero_warp::no_index)
      control = REAL_C(0.0);
    ctrl_used[u + lane] = control;
    real_t force = gain[lane] * control;
    if (bias_type[lane] == 1)
      force += bias0[lane] + bias1[lane] * length[u + lane] + bias2[lane] * velocity[u + lane];
    force_pre[u + lane] = force;
    int limited = 0;
    if (force_limited[lane]) {
      if (force < force_lo[lane]) {
        force = force_lo[lane];
        limited = -1;
      } else if (force > force_hi[lane]) {
        force = force_hi[lane];
        limited = 1;
      }
    }
    force_route[u + lane] = limited;
    work_force[lane] = force;
  }
}

static __device__ void d_actuation_gemv_dx(const real_t *, const real_t *, real_t *);

// All world-block lanes enter. Small actuator sets use a warp for scalar
// preparation and independent DOF sums; large sets retain the MathDx route.
static __device__ __forceinline__ void
d_actuation_count(int w, real_t *length, real_t *moment, real_t *velocity, real_t *ctrl, int *active,
                  int *vel_status, real_t *gain, real_t *bias0, real_t *bias1, real_t *bias2, int *bias_type,
                  real_t *ctrl_lo, real_t *ctrl_hi, real_t *force_lo, real_t *force_hi, int *force_limited,
                  int disable_flags, real_t *force, real_t *force_pre, real_t *ctrl_used, int *ctrl_route,
                  int *force_route, real_t *qfrc_actuator, int *status, int *warn_badctrl_count,
                  int *warn_badctrl_lastinfo, int *warning_number, int *warning_lastinfo, int nv, int nu) {
  __shared__ int run;
  auto *shared = flash_libero::substep_shared_data();
  real_t *work_actuator = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::actuator);
  real_t *work_scalar_force =
      flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::scalar_force);
  libero_vec::fill_block(work_actuator, flash_libero::SubstepShape::nv, REAL_C(0.0));
  const long long u = (long long)w * nu, v = (long long)w * nv;
  if (nu <= 32)
    d_actuation_prepare_warp(w, length, velocity, ctrl, active, vel_status, gain, bias0, bias1,
        bias2, bias_type, ctrl_lo, ctrl_hi, force_lo, force_hi, force_limited, disable_flags,
        work_scalar_force, force_pre, ctrl_used, ctrl_route, force_route, qfrc_actuator,
        status, warn_badctrl_count, warn_badctrl_lastinfo, warning_number, warning_lastinfo, nv, nu);
  else
    d_actuation_prepare_block(0, length + u, moment + u * nv, velocity + u, ctrl + u, active + w,
                              vel_status + w, gain, bias0, bias1, bias2, bias_type, ctrl_lo,
                              ctrl_hi, force_lo, force_hi, force_limited, disable_flags, work_scalar_force,
                              force_pre + u, ctrl_used + u, ctrl_route + u, force_route + u,
                              qfrc_actuator + v, status + w, warn_badctrl_count + w,
                              warn_badctrl_lastinfo + w, warning_number + w, warning_lastinfo + w, nv, nu);
  __syncthreads();
  libero_vec::copy_block(work_scalar_force, force + u, nu);
  if (threadIdx.x == 0) {
    run = active[w] == 1 && vel_status[w] == 0 && (disable_flags & 1024) == 0 && status[w] == 0;
    if (nv != flash_libero::SubstepShape::nv || nu != flash_libero::SubstepShape::nu) {
      status[w] = 2;
      run = 0;
    }
  }
  __syncthreads();
  if (run) {
    bool invalid = false;
    if constexpr (flash_libero::SubstepShape::nu <= 32) {
      // Tiny transpose GEMV: each DOF owns its ordered actuator sum. No matrix
      // repacking, padded library tiles or extra shared matrix is required.
      for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        real_t value = REAL_C(0.0);
        for (int a = 0; a < nu; ++a)
          value += moment[u * nv + size_t(a) * nv + j] * work_scalar_force[a];
        work_actuator[j] = qfrc_actuator[v + j] = value;
        invalid |= !isfinite(value);
      }
    } else {
      d_actuation_gemv_dx(moment + u * nv, work_scalar_force, work_actuator);
      for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        const real_t value = work_actuator[j];
        qfrc_actuator[v + j] = value;
        invalid |= !isfinite(value);
      }
    }
    if (__syncthreads_or(invalid) && threadIdx.x == 0)
      status[w] = 3;
  }
  __syncthreads();
}

// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.

// Smooth combine and external-force projection use one full warp per world.
// Their callers must enter with all block threads and rendezvous between stages.
// Other helpers in this header retain the owning-lane contract.
// Topology is immutable, qualified by the host before launch. World arrays use
// their original compact strides. Compilation permits FMA contraction.
static __device__ inline void d_three_rn(int w, const real_t *passive, const real_t *bias,
                                         const real_t *applied, const real_t *actuator, const int *active,
                                         int *entry_status, real_t *smooth, real_t *seed, int nv) {
  // All block threads enter; warp 0 owns ordered DOF tiles.
  if (threadIdx.x >= 32)
    return;
  const int lane = libero_warp::lane();
  const size_t v = size_t(w) * nv;
  if (lane == 0)
    entry_status[w] = 0;
  for (int i = lane; i < nv; i += 32)
    smooth[v + i] = seed[v + i] = REAL_C(0.0);
  __syncwarp();
  if (active[w] == 0)
    return;
  if (active[w] != 1) {
    if (lane == 0)
      entry_status[w] = 1;
    return;
  }
  bool bad = false;
  for (int i = lane; i < nv; i += 32)
    bad |= !isfinite(passive[v + i]) || !isfinite(bias[v + i]) || !isfinite(applied[v + i]) ||
           !isfinite(actuator[v + i]);
  if (libero_warp::any(bad)) {
    if (lane == 0)
      entry_status[w] = 3;
    return;
  }
  for (int base = 0; base < nv; base += 32) {
    const int i = base + lane;
    real_t force = REAL_C(0.0);
    if (i < nv) {
      force = passive[v + i] - bias[v + i];
      force += applied[v + i];
      force += actuator[v + i];
    }
    const int first = libero_warp::first_index(i < nv && !isfinite(force), i);
    if (i < nv && i < first)
      smooth[v + i] = seed[v + i] = force;
    __syncwarp();
    if (first != libero_warp::no_index) {
      if (lane == 0)
        entry_status[w] = 4;
      return;
    }
  }
}

static __device__ inline void d_project(int w, const real_t *seed, const real_t *xfrc, const real_t *xipos,
                                        const real_t *subtree_com, const real_t *cdof, const int *body_ids,
                                        const int *body_count, const int *active, const int *body_rootid,
                                        const int *ancestor, real_t *scratch, real_t *stages, real_t *qfrc,
                                        int *status, int nbody, int nv, int diagnostic,
                                        const int *body_dof_offsets = nullptr,
                                        const int *body_dof_indices = nullptr) {
  // One full warp cooperates. Sequential bodies and ordered DOF tiles retain
  // the scalar accumulation order and precise private failure prefix.
  if (threadIdx.x >= 32)
    return;
  const int lane = libero_warp::lane();
  const size_t v = size_t(w) * nv, b = size_t(w) * nbody;
  if (lane == 0)
    status[w] = 0;
  if (active[w] != 0 && active[w] != 1) {
    if (lane == 0)
      status[w] = 1;
    return;
  }
  if (active[w] == 0)
    return;
  if (body_count[w] != nbody - 1) {
    if (lane == 0)
      status[w] = 2;
    return;
  }
  bool bad = false;
  for (int i = lane; i < nbody - 1; i += 32)
    bad |= body_ids[size_t(w) * (nbody - 1) + i] != i + 1;
  if (libero_warp::any(bad)) {
    if (lane == 0)
      status[w] = 3;
    return;
  }
  bad = false;
  for (int j = lane; j < nv; j += 32)
    bad |= !isfinite(seed[v + j]);
  if (libero_warp::any(bad)) {
    if (lane == 0)
      status[w] = 4;
    return;
  }
  bad = false;
  for (int i = lane; i < (nbody - 1) * 6; i += 32)
    bad |= !isfinite(xfrc[(b + 1) * 6 + i]);
  if (libero_warp::any(bad)) {
    if (lane == 0)
      status[w] = 5;
    return;
  }
  // Only nonzero external-force bodies admit positional/ancestor operands.
  for (int body = 1; body < nbody; ++body) {
    const real_t *f = xfrc + (b + body) * 6;
    int nonzero = 0;
    if (lane == 0)
      for (int k = 0; k < 6; ++k)
        nonzero |= f[k] != REAL_C(0.0);
    nonzero = __shfl_sync(libero_warp::mask, nonzero, 0);
    if (!nonzero)
      continue;
    const int root = body_rootid[body];
    bad = false;
    if (lane < 3)
      bad = !isfinite(xipos[(b + body) * 3 + lane]) || !isfinite(subtree_com[(b + root) * 3 + lane]);
    if (body_dof_offsets && body_dof_indices) {
      for (int k = body_dof_offsets[body] + lane; k < body_dof_offsets[body + 1]; k += 32)
        for (int r = 0; r < 6; ++r)
          bad |= !isfinite(cdof[(v + body_dof_indices[k]) * 6 + r]);
    } else
      for (int j = lane; j < nv; j += 32)
        if (ancestor[size_t(body) * nv + j] != 0)
          for (int k = 0; k < 6; ++k)
            bad |= !isfinite(cdof[(v + j) * 6 + k]);
    if (libero_warp::any(bad)) {
      if (lane == 0)
        status[w] = 6;
      return;
    }
  }
  for (int j = lane; j < nv; j += 32)
    scratch[v + j] = seed[v + j];
  __syncwarp();
  for (int body = 1; body < nbody; ++body) {
    real_t f[6], offset[3];
    int nonzero = 0;
    for (int k = 0; k < 6; ++k) {
      real_t value = lane == 0 ? xfrc[(b + body) * 6 + k] : REAL_C(0.0);
      f[k] = __shfl_sync(libero_warp::mask, value, 0);
      nonzero |= f[k] != REAL_C(0.0);
    }
    if (!nonzero)
      continue;
    const int root = body_rootid[body];
    for (int k = 0; k < 3; ++k) {
      real_t value = lane == 0 ? xipos[(b + body) * 3 + k] - subtree_com[(b + root) * 3 + k] : REAL_C(0.0);
      offset[k] = __shfl_sync(libero_warp::mask, value, 0);
    }
    if (!isfinite(offset[0]) || !isfinite(offset[1]) || !isfinite(offset[2])) {
      if (lane == 0)
        status[w] = 7;
      return;
    }
    // Diagnostic rows retain the dense route. The qualified CSR route
    // schedules ascending active DOFs; inactive columns still perform
    // their original two +0 updates, including signed-zero normalization.
    const bool sparse = !diagnostic && body_dof_offsets && body_dof_indices;
    const int begin = sparse ? body_dof_offsets[body] : 0;
    const int count = sparse ? body_dof_offsets[body + 1] - begin : nv;
    for (int base = 0; base < count; base += 32) {
      const int ordinal = base + lane;
      const int j = ordinal < count ? (sparse ? body_dof_indices[begin + ordinal] : ordinal) : nv;
      real_t jp[3] = {REAL_C(0.0), REAL_C(0.0), REAL_C(0.0)}, jr[3] = {REAL_C(0.0), REAL_C(0.0), REAL_C(0.0)};
      real_t force = REAL_C(0.0), torque = REAL_C(0.0), after_force = REAL_C(0.0), after_torque = REAL_C(0.0);
      int step = 5;
      if (j < nv) {
        if (sparse || ancestor[size_t(body) * nv + j] != 0) {
          const real_t *c = cdof + (v + j) * 6;
          for (int k = 0; k < 3; ++k)
            jr[k] = c[k];
          jp[0] = c[3] + (jr[1] * offset[2] - jr[2] * offset[1]);
          jp[1] = c[4] + (jr[2] * offset[0] - jr[0] * offset[2]);
          jp[2] = c[5] + (jr[0] * offset[1] - jr[1] * offset[0]);
          for (int k = 0; k < 3; ++k)
            if (!isfinite(jp[k]))
              step = 0;
        }
        if (step == 5) {
          for (int k = 0; k < 3; ++k)
            if (f[k] != REAL_C(0.0))
              force += jp[k] * f[k];
          if (!isfinite(force))
            step = 1;
        }
        if (step == 5) {
          after_force = scratch[v + j] + force;
          if (!isfinite(after_force))
            step = 2;
        }
        if (step == 5) {
          for (int k = 0; k < 3; ++k)
            if (f[k + 3] != REAL_C(0.0))
              torque += jr[k] * f[k + 3];
          if (!isfinite(torque))
            step = 3;
        }
        if (step == 5) {
          after_torque = after_force + torque;
          if (!isfinite(after_torque))
            step = 4;
        }
      }
      const int first = libero_warp::first_index(j < nv && step != 5, j);
      if (j < nv && j <= first) {
        real_t *t = diagnostic ? stages + (b + body) * 13 * nv + j : nullptr;
        if (diagnostic && step >= 1)
          for (int k = 0; k < 3; ++k) {
            t[size_t(k) * nv] = offset[k];
            t[size_t(k + 3) * nv] = jp[k];
            t[size_t(k + 6) * nv] = jr[k];
          }
        if (step >= 2)
          scratch[v + j] = after_force;
        if (diagnostic && step >= 3) {
          t[9 * nv] = force;
          t[10 * nv] = after_force;
        }
        if (step >= 4)
          scratch[v + j] = after_torque;
        if (diagnostic && step == 5) {
          t[11 * nv] = torque;
          t[12 * nv] = after_torque;
        }
      }
      __syncwarp();
      if (first != libero_warp::no_index) {
        if (sparse)
          for (int k = lane; k < first; k += 32)
            if (ancestor[size_t(body) * nv + k] == 0)
              scratch[v + k] = (scratch[v + k] + REAL_C(0.0)) + REAL_C(0.0);
        __syncwarp();
        if (lane == 0)
          status[w] = 7;
        return;
      }
    }
    if (sparse)
      for (int j = lane; j < nv; j += 32)
        if (ancestor[size_t(body) * nv + j] == 0)
          scratch[v + j] = (scratch[v + j] + REAL_C(0.0)) + REAL_C(0.0);
    __syncwarp();
  }
  for (int j = lane; j < nv; j += 32)
    qfrc[v + j] = scratch[v + j];
  __syncwarp();
}

// Packed tree LDL^T, not dense POTRF. No dense expansion or external LAPACK
// replacement: each packed row contains its diagonal followed by ancestors.

// The original externally visible work arrays are written once at the tail.
// All stage-to-stage operands, the single active RHS and candidate acceleration
// remain in shared memory. Failure still isolates durable public outputs.
static __device__ __forceinline__ void
d_smooth_chain(int w, const real_t *passive, const real_t *bias, const real_t *applied,
               const real_t *actuator, const int *active, int *entry_status, real_t *smooth, real_t *seed,
               int nv, const real_t *xfrc, const real_t *xipos, const real_t *subtree_com, const real_t *cdof,
               const int *body_ids, const int *body_count, const int *body_rootid, const int *ancestor,
               real_t *scratch, real_t *stages, real_t *qfrc, int *d02_status, int nbody, const real_t *qm,
               const int *count, const int *madr, const int *parent, real_t *ld, real_t *inv,
               real_t *sqrt_inv, real_t *solution, int *pivot, int *solve_status, int *warning_count,
               int *warning_last, int nm, real_t *rhs, real_t *qacc_smooth, int *status,
               const real_t *source_ld, const real_t *source_inv, const real_t *source_sqrt_inv,
               const int *source_pivot, const int *source_status, const int *source_warning_count,
               const int *source_warning_last, const int *source_active, const int *position_valid,
               const int64_t *position_written, const int64_t *forward_epoch, int64_t *factor_reuse_count,
               const int *body_dof_offsets, const int *body_dof_indices, const int *mass_plan) {
  auto *shared = flash_libero::substep_shared_data();
  real_t *work_scratch = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::scratch);
  real_t *work_force = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::force);
  real_t *work_actuator = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::actuator);
  real_t *work_smooth = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::smooth);
  real_t *work_seed = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::seed);
  real_t *work_rhs = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::rhs);
  real_t *work_acceleration =
      flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::acceleration);
  real_t *work_inv = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::inv);
  real_t *work_sqrt_inv = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::sqrt_inv);
  int *work_pivot = flash_libero::shared_at<int>(shared, flash_libero::SmoothLayout::pivot);
  real_t *work_ld = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::ld);
  const size_t v = size_t(w) * nv, b = size_t(w) * nbody, m = size_t(w) * nm, r = size_t(w) * 4 * nv;
  // scratch is a diagnostic with partial-failure semantics; preserve its old
  // value until the original projection's validation has admitted writes.
  libero_vec::copy_block(scratch + v, work_scratch, nv);
  libero_vec::copy_block(qfrc + v, work_force, nv);
  __syncthreads();
  d_three_rn(0, passive + v, bias + v, applied + v, work_actuator, active + w, entry_status + w, work_smooth,
             work_seed, nv);
  __syncthreads();
  d_project(0, work_seed, xfrc + b * 6, xipos + b * 3, subtree_com + b * 3, cdof + v * 6,
            body_ids + size_t(w) * (nbody - 1), body_count + w, active + w, body_rootid, ancestor,
            work_scratch, stages, work_force, d02_status + w, nbody, nv, 0, body_dof_offsets,
            body_dof_indices);
  __syncthreads();
  // The admitted position producer owns and validates these exact immutable-
  // lineage factors. Read them directly: no shared factor copy/clear, repeated
  // matrix/factor scan, or rediscovery of the fixed ancestor topology.
  const bool prepared = mass_plan && active[w] == 1 && count[w] == 1 &&
                        entry_status[w] == 0 && d02_status[w] == 0 &&
                        position_valid[w] == 1 && position_written[w] == forward_epoch[w] &&
                        source_active[w] == 1 && (source_status[w] == 0 || source_status[w] == 5);
  if (prepared) {
    const bool solved = flash_libero::forward_mass::solve(work_force, work_acceleration,
        source_ld + m, source_inv + v, madr, parent, mass_plan);
    if (solved) {
      const int warnings = source_warning_count[w];
      // Preserve existing work-array consumers and failure observability.
      // Only their shared staging has disappeared; canonical publication is
      // still performed once by d_commit_smooth_shared_block below.
      libero_vec::copy_block(source_ld + m, ld + m, nm);
      for (int i = threadIdx.x; i < nv; i += blockDim.x) {
        smooth[v + i] = work_smooth[i];
        seed[v + i] = work_seed[i];
        scratch[v + i] = work_scratch[i];
        qfrc[v + i] = work_force[i];
        qacc_smooth[v + i] = work_acceleration[i];
        rhs[r + i] = work_force[i];
        solution[r + i] = work_acceleration[i];
        inv[v + i] = source_inv[v + i];
        sqrt_inv[v + i] = source_sqrt_inv[v + i];
        pivot[v + i] = source_pivot[v + i];
      }
      libero_vec::fill_block(rhs + r + nv, 3 * nv, REAL_C(0.0));
      libero_vec::fill_block(solution + r + nv, 3 * nv, REAL_C(0.0));
      if (threadIdx.x == 0) {
        solve_status[w] = warnings ? 5 : 0;
        warning_count[w] = warnings;
        warning_last[w] = source_warning_last[w];
        status[w] = warnings ? 25 : 0;
        ++factor_reuse_count[w];
      }
      return; // The caller's publication barrier closes these stores.
    }
    // Reproduce the original routed/zeroed error result with the general
    // implementation, before any durable output or reuse counter was changed.
  }
  __shared__ int solve, warnings, last, reuse;
  if (threadIdx.x == 0) {
    solve = 0;
    warnings = 0;
    last = -1;
    reuse = 0;
  }
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    work_rhs[i] = (active[w] == 1 && entry_status[w] == 0 && d02_status[w] == 0) ? work_force[i] : REAL_C(0.0);
    work_acceleration[i] = REAL_C(0.0);
    work_inv[i] = work_sqrt_inv[i] = REAL_C(0.0);
    work_pivot[i] = 0;
  }
  libero_vec::fill_block(work_ld, nm, REAL_C(0.0));
  __syncthreads();
  // Two bits per warp preserve separate input-failure and factor-reuse
  // decisions. Each lane scans disjoint elements; only the leader routes
  // status/warnings after the scan publication.
  __shared__ int scan_bad[32];
  bool bad_input = false, bad_factor = false;
  const bool check_inputs = active[w] == 1 && count[w] >= 1 && count[w] <= 4;
  const bool check_factor = check_inputs && position_valid[w] == 1 &&
                            position_written[w] == forward_epoch[w] && source_active[w] == 1 &&
                            (source_status[w] == 0 || source_status[w] == 5);
  if (check_inputs) {
    bad_input = libero_vec::any_nonfinite_block_lane(qm + m, nm) ||
                libero_vec::any_nonfinite_block_lane(work_rhs, nv);
    if (check_factor)
      bad_factor = libero_vec::any_nonfinite_block_lane(source_ld + m, nm) ||
                   libero_vec::any_nonfinite_block_lane(source_inv + v, nv) ||
                   libero_vec::any_nonfinite_block_lane(source_sqrt_inv + v, nv);
  }
  const bool input_vote = libero_warp::any(bad_input);
  const bool factor_vote = libero_warp::any(bad_factor);
  if (libero_warp::lane() == 0)
    scan_bad[libero_warp::warp()] = int(input_vote) | (int(factor_vote) << 1);
  __syncthreads();
  if (threadIdx.x == 0 && active[w] != 0) {
    int bad = 0;
    for (int i = 0; i < (blockDim.x + 31) / 32; ++i)
      bad |= scan_bad[i];
    if (active[w] != 1)
      solve = 1;
    else if (count[w] < 1 || count[w] > 4)
      solve = 2;
    else if (bad & 1)
      solve = 3;
    else {
      reuse = check_factor && !(bad & 2);
      if (reuse) {
        warnings = source_warning_count[w];
        last = source_warning_last[w];
        ++factor_reuse_count[w];
      }
    }
  }
  __syncthreads();
  if (active[w] == 1 && solve == 0) {
    libero_packed_mass::ForestPlan forest;
    bool factor_ok = true;
    if (reuse) {
      libero_vec::copy_block(source_ld + m, work_ld, nm);
      libero_vec::copy_block(source_inv + v, work_inv, nv);
      libero_vec::copy_block(source_sqrt_inv + v, work_sqrt_inv, nv);
      libero_vec::copy_block(source_pivot + v, work_pivot, nv);
      __syncthreads();
      forest = libero_packed_mass::inspect_forest_block(nv, parent);
    } else
      factor_ok = libero_packed_mass::factor_block(qm + m, work_ld, work_inv, work_sqrt_inv, work_pivot,
                                                   warnings, last, nv, nm, madr, parent, forest);
    const bool solve_ok = libero_packed_mass::solve_rhs_block(work_rhs, work_acceleration, work_ld, work_inv,
                                                              1, nv, madr, parent, forest);
    if (threadIdx.x == 0) {
      if (!factor_ok || !solve_ok) {
        solve = 4;
      } else if (warnings != 0)
        solve = 5;
    }
  }
  __syncthreads();
  __shared__ int publish;
  if (threadIdx.x == 0) {
    solve_status[w] = solve;
    warning_count[w] = warnings;
    warning_last[w] = last;
    int code = 0;
    if (active[w] != 0) {
      if (entry_status[w])
        code = entry_status[w];
      else if (d02_status[w])
        code = 10 + d02_status[w];
      else if (solve == 5)
        code = 25;
      else if (solve)
        code = 20 + solve;
    }
    status[w] = code;
    publish = active[w] != 0 && (code == 0 || code == 25);
  }
  __syncthreads();
  // Each owner clears and exports its own elements. Successful routing used
  // to copy acceleration onto itself; its data is already in the right place.
  for (int i = threadIdx.x; i < nm; i += blockDim.x) {
    if (solve == 4)
      work_ld[i] = REAL_C(0.0);
    ld[m + i] = work_ld[i];
  }
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    if (solve == 4)
      work_inv[i] = work_sqrt_inv[i] = work_acceleration[i] = REAL_C(0.0);
    if (!publish)
      work_acceleration[i] = work_force[i] = work_smooth[i] = REAL_C(0.0);
    smooth[v + i] = work_smooth[i];
    seed[v + i] = work_seed[i];
    scratch[v + i] = work_scratch[i];
    qfrc[v + i] = work_force[i];
    qacc_smooth[v + i] = work_acceleration[i];
    rhs[r + i] = work_rhs[i];
    solution[r + i] = (solve == 0 || solve == 5) ? work_acceleration[i] : REAL_C(0.0);
    inv[v + i] = work_inv[i];
    sqrt_inv[v + i] = work_sqrt_inv[i];
    pivot[v + i] = work_pivot[i];
  }
  // The API retains four RHS diagnostic columns. Only the first is consumed
  // by smooth acceleration; unused columns keep their original zero content.
  libero_vec::fill_block(rhs + r + nv, 3 * nv, REAL_C(0.0));
  libero_vec::fill_block(solution + r + nv, 3 * nv, REAL_C(0.0));
}

} // namespace physics_force

namespace physics_force {

// Compile the descriptor from cold model dimensions. moment is NU x NV;
// A = moment^T, B = force, C = generalized actuator force.
using DActuationBLAS =
    decltype(cublasdx::Size<flash_libero::SubstepShape::nv, 1, flash_libero::SubstepShape::nu>() +
             cublasdx::Precision<real_t>() + cublasdx::Type<cublasdx::type::real>() +
             cublasdx::Function<cublasdx::function::MM>() +
             cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
             cublasdx::Alignment<sizeof(real_t), sizeof(real_t), sizeof(real_t)>() + cublasdx::Block() +
             cublasdx::BlockDim<flash_libero::SubstepShape::threads>() +
             cublasdx::SM<flash_libero::SubstepShape::sm>());

static __device__ void d_actuation_gemv_dx(const real_t *moment, const real_t *force, real_t *output) {
  using B = DActuationBLAS;
  __shared__ __align__(16) real_t a[B::a_size];
  __shared__ __align__(16) real_t b[B::b_size];
  __shared__ __align__(16) real_t c[B::c_size];
  auto at = cublasdx::make_tensor(a, B::get_layout_smem_a());
  auto bt = cublasdx::make_tensor(b, B::get_layout_smem_b());
  auto ct = cublasdx::make_tensor(c, B::get_layout_smem_c());
  // Admission mask preserves zero-skip semantics: no access to an ignored
  // moment row, including rows containing NaN or infinity.
  // A packet stays within one admitted actuator row, including odd NV.
  constexpr int row_packets = (flash_libero::SubstepShape::nv + 1) / 2;
  for (int z = threadIdx.x; z < row_packets * flash_libero::SubstepShape::nu; z += blockDim.x) {
    const int i = z / row_packets, j = 2 * (z % row_packets);
    real2_t values = make_real2(REAL_C(0.0), REAL_C(0.0));
    if (force[i] != REAL_C(0.0)) {
      if (j + 1 < flash_libero::SubstepShape::nv)
        values = libero_vec::load_pair(moment + i * flash_libero::SubstepShape::nv + j);
      else
        values.x = moment[i * flash_libero::SubstepShape::nv + j];
    }
    // Tensor layout is library-owned; only the compact source is paired.
    at(j, i) = values.x;
    if (j + 1 < flash_libero::SubstepShape::nv)
      at(j + 1, i) = values.y;
  }
  for (int i = 2 * threadIdx.x; i < flash_libero::SubstepShape::nu; i += 2 * blockDim.x) {
    if (i + 1 < flash_libero::SubstepShape::nu) {
      const real2_t values = libero_vec::load_pair(force + i);
      bt(i, 0) = values.x;
      bt(i + 1, 0) = values.y;
    } else
      bt(i, 0) = force[i];
  }
  for (int j = threadIdx.x; j < flash_libero::SubstepShape::nv; j += blockDim.x)
    ct(j, 0) = REAL_C(0.0);
  __syncthreads();
  B().execute(REAL_C(1.0), at, bt, REAL_C(0.0), ct);
  __syncthreads();
  for (int j = 2 * threadIdx.x; j < flash_libero::SubstepShape::nv; j += 2 * blockDim.x) {
    if (j + 1 < flash_libero::SubstepShape::nv)
      libero_vec::store_pair(output + j, ct(j, 0), ct(j + 1, 0));
    else
      output[j] = ct(j, 0);
  }
  __syncthreads();
}

} // namespace physics_force

// Keep the shape-specific policy local to this implementation.

// Newton/Post implementations are compiled independently in newton_solver.cu.

namespace physics_velocity {
static __device__ void d_gate(int w, int *active, int *error, int64_t *state, int64_t *source, int64_t *epoch,
                              int *valid, int64_t *written, int *gate, int require);
}

namespace physics_velocity {
static __device__ void d_finish(int w, int *gate, int *status, int64_t *epoch, int *valid, int64_t *written,
                                int *error, int repair, int code);
}

namespace physics_velocity {
static __device__ void d_velocity_status(int w, int *a, int *b, int *c, int *out);
}

namespace physics_velocity {
static __device__ void d_scalar_transmission_velocity(int w, real_t *moment, real_t *qvel, int *active,
                                                      int *position_status, int nv, int nu, real_t *velocity,
                                                      int *status);
// Collective entry/exit; the successful dot products preserve scalar order.
static __device__ void d_transmission_velocity_block(int w, real_t *moment, real_t *qvel, int *active,
                                                     int *position_status, int nv, int nu, real_t *velocity,
                                                     int *status);
} // namespace physics_velocity

namespace physics_velocity {
// Collective entry: every exit closes flag reads and publishes any output.
static __device__ void d_comvel_bias_cached(int w, int *active, real_t *cdof, real_t *qvel, int *bp, int *ba,
                                            int *bn, int *dj, int *jt, int nb, int nv, real_t *cvel,
                                            real_t *cdot, int *status, real_t *cinert, real_t *gravity,
                                            int disabled, real_t *cache_acc, real_t *cache_force,
                                            int *cache_status, int *cache_valid, int64_t *cache_state,
                                            int64_t *cache_forward, const int64_t *state_epoch,
                                            const int64_t *forward_epoch, const int *level_offsets = nullptr,
                                            const int *level_bodies = nullptr, int nlevels = 0);
} // namespace physics_velocity

namespace physics_velocity {
static __device__ void d_joint_passive_count(int w, real_t *qpos, real_t *qvel, int *active, int *jtype,
                                             int *qadr, int *dadr, real_t *stiffness, real_t *spring,
                                             real_t *damping, real_t *out, int *status, int nv, int nq,
                                             int njnt, int joint_disjoint = 0,
                                             const int *joint_type_offsets = nullptr,
                                             const int *joint_type_indices = nullptr);
}

namespace physics_velocity {
static __device__ void d_fluid_count(int w, real_t *cvel, real_t *subtree_com, real_t *xipos, real_t *ximat,
                                     real_t *cdof, int *active, int *rootid, real_t *mass, real_t *box,
                                     int *ancestor, real_t *wind, real_t *opt, int passive_disabled,
                                     int diagnostic, real_t *stages, real_t *qfrc, int *status, int nbody,
                                     int nv, const int *body_dof_offsets = nullptr,
                                     const int *body_dof_indices = nullptr);
}

// Maintained world-local implementation, compiled as part of fused_substep.cu.

// Shared FP64 velocity implementation.

namespace physics_velocity {

// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.

// One block owns one world. These are private working values, never public
// aliases; their lifetime extends across the force-chain device operations.
// Handwritten sparse rigid-body arithmetic. These routines receive whole
// allocations; only the selected world row is touched. The caller executes
// each collective tree routine on all threads. Its ordered tree arithmetic
// remains on lane zero; array checks and independent projections cooperate.
static __device__ inline bool d_finite(const real_t *x, int n) {
  for (int k = 0; k < n; ++k)
    if (!isfinite(x[k]))
      return false;
  return true;
}
// Collective helpers: every thread in the world block must enter. Warp votes
// exchange predicates only; the barriers publish and protect inter-warp flags.
static __device__ inline bool d1_finite_block(const real_t *x, int n, const real_t *y = nullptr, int ny = 0,
                                              const real_t *z = nullptr, int nz = 0,
                                              const real_t *u = nullptr, int nu = 0,
                                              const real_t *v = nullptr, int nv = 0) {
  __shared__ int warp_bad[32];
  bool bad = false;
  bad |= libero_vec::any_nonfinite_block_lane(x, n);
  bad |= libero_vec::any_nonfinite_block_lane(y, ny);
  bad |= libero_vec::any_nonfinite_block_lane(z, nz);
  bad |= libero_vec::any_nonfinite_block_lane(u, nu);
  bad |= libero_vec::any_nonfinite_block_lane(v, nv);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  // Membership comes from block geometry, not the instantaneous active
  // lanes after unequal per-thread strides. Include every live warp lane.
  const int remaining = (int)blockDim.x - (warp << 5);
  const unsigned mask = remaining >= 32 ? 0xffffffffu : ((1u << remaining) - 1u);
  const int vote = __any_sync(mask, bad);
  if (lane == 0)
    warp_bad[warp] = vote;
  __syncthreads();
  bool any = false;
  for (int k = 0; k < (blockDim.x + 31) / 32; ++k)
    any |= warp_bad[k] != 0;
  __syncthreads();
  return !any;
}
static __device__ inline void d1_zero_block(real_t *x, int n) { libero_vec::fill_block(x, n, REAL_C(0.0)); }
static __device__ inline void d_zero(real_t *x, int n) {
  for (int k = 0; k < n; ++k)
    x[k] = REAL_C(0.0);
}

// Collective publication: callers enter with the complete world block. Gate
// decisions and scalar flags retain lane-zero order; payload stores complete
// before the epoch becomes visible. Shifted overlaps retain serial copy order.

static __device__ __forceinline__ void d_scalar_transmission_velocity(int w, real_t *moment, real_t *qvel,
                                                                      int *active, int *position_status,
                                                                      int nv, int nu, real_t *velocity,
                                                                      int *status) {
  real_t *v = velocity + w * nu;
  const real_t *q = qvel + w * nv;
  const real_t *m = moment + w * nu * nv;
  status[w] = 0;
  d_zero(v, nu);
  if (!active[w])
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  if (position_status[w]) {
    status[w] = 3;
    return;
  }
  if (!d_finite(q, nv)) {
    status[w] = 2;
    return;
  }
  for (int a = 0; a < nu; ++a) {
    real_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int j = 0;
    for (; j + 3 < nv; j += 4) {
      s0 += m[a * nv + j] * q[j];
      s1 += m[a * nv + j + 1] * q[j + 1];
      s2 += m[a * nv + j + 2] * q[j + 2];
      s3 += m[a * nv + j + 3] * q[j + 3];
    }
    real_t x = (s0 + s2) + (s1 + s3);
    if (j < nv) {
      real_t tail = m[a * nv + j] * q[j];
      for (++j; j < nv; ++j)
        tail += m[a * nv + j] * q[j];
      x += tail;
    }
    if (!isfinite(x)) {
      status[w] = 4;
      return;
    }
    v[a] = x;
  }
}
// One lane per actuator; preserve the four-accumulator order within each dot.
// Error fallback reproduces the scalar path's partial writes and first failure.
// Every return publishes outputs to the whole block.
static __device__ __forceinline__ void d_transmission_velocity_block(int w, real_t *moment, real_t *qvel,
                                                                     int *active, int *position_status,
                                                                     int nv, int nu, real_t *velocity,
                                                                     int *status) {
  real_t *v = velocity + (long long)w * nu;
  const real_t *q = qvel + (long long)w * nv, *m = moment + (long long)w * nu * nv;
  if (threadIdx.x == 0) {
    status[w] = 0;
    d_zero(v, nu);
    if (active[w] && active[w] != 1)
      status[w] = 1;
    else if (active[w] == 1) {
      if (position_status[w])
        status[w] = 3;
      else if (!d_finite(q, nv))
        status[w] = 2;
    }
  }
  __syncthreads();
  if (active[w] != 1 || status[w] != 0) {
    __syncthreads();
    return;
  }
  const uintptr_t output = reinterpret_cast<uintptr_t>(v);
  const uintptr_t qsource = reinterpret_cast<uintptr_t>(q), msource = reinterpret_cast<uintptr_t>(m);
  const size_t output_bytes = size_t(nu) * sizeof(real_t);
  const bool qalias =
      output <= qsource ? qsource - output < output_bytes : output - qsource < size_t(nv) * sizeof(real_t);
  const bool malias = output <= msource ? msource - output < output_bytes
                                        : output - msource < size_t(nu) * nv * sizeof(real_t);
  if (qalias || malias) {
    if (threadIdx.x == 0)
      d_scalar_transmission_velocity(w, moment, qvel, active, position_status, nv, nu, velocity, status);
    __syncthreads();
    return;
  }
  bool bad = false;
  for (int a = threadIdx.x; a < nu; a += blockDim.x) {
    real_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int j = 0;
    for (; j + 3 < nv; j += 4) {
      s0 += m[a * nv + j] * q[j];
      s1 += m[a * nv + j + 1] * q[j + 1];
      s2 += m[a * nv + j + 2] * q[j + 2];
      s3 += m[a * nv + j + 3] * q[j + 3];
    }
    real_t x = (s0 + s2) + (s1 + s3);
    if (j < nv) {
      real_t tail = m[a * nv + j] * q[j];
      for (++j; j < nv; ++j)
        tail += m[a * nv + j] * q[j];
      x += tail;
    }
    v[a] = x;
    bad |= !isfinite(x);
  }
  if (__syncthreads_or(bad)) {
    if (threadIdx.x == 0)
      d_scalar_transmission_velocity(w, moment, qvel, active, position_status, nv, nu, velocity, status);
  }
  __syncthreads();
}

static __device__ inline void d_cross_motion(const real_t *v, const real_t *a, real_t *out) {
  out[0] = -v[2] * a[1] + v[1] * a[2];
  out[1] = v[2] * a[0] - v[0] * a[2];
  out[2] = -v[1] * a[0] + v[0] * a[1];
  out[3] = -v[2] * a[4] + v[1] * a[5];
  out[4] = v[2] * a[3] - v[0] * a[5];
  out[5] = -v[1] * a[3] + v[0] * a[4];
  out[3] += -v[5] * a[1] + v[4] * a[2];
  out[4] += v[5] * a[0] - v[3] * a[2];
  out[5] += -v[4] * a[0] + v[3] * a[1];
}
static __device__ inline void d_add_three(real_t *v, const real_t *a, const real_t *q) {
  real_t t[6] = {};
  for (int j = 0; j < 3; ++j)
    if (q[j] != 0)
      for (int k = 0; k < 6; ++k)
        t[k] += a[6 * j + k] * q[j];
  for (int k = 0; k < 6; ++k)
    v[k] += t[k];
}
static __device__ inline void d1_comvel_body(int b, const real_t *a, const real_t *q, const int *bp,
                                             const int *ba, const int *bn, const int *dj, const int *jt,
                                             real_t *v, real_t *d) {
  for (int k = 0; k < 6; ++k)
    v[6 * b + k] = v[6 * bp[b] + k];
  int j = 0;
  while (j < bn[b]) {
    int i = ba[b] + j, type = jt[dj[i]];
    if (type == 0) {
      d_add_three(v + 6 * b, a + 6 * i, q + i);
      j += 3;
      i = ba[b] + j;
    }
    if (type <= 1) {
      for (int k = 0; k < 3; ++k)
        d_cross_motion(v + 6 * b, a + 6 * (i + k), d + 6 * (i + k));
      d_add_three(v + 6 * b, a + 6 * i, q + i);
      j += 3;
    } else {
      d_cross_motion(v + 6 * b, a + 6 * i, d + 6 * i);
      for (int k = 0; k < 6; ++k)
        v[6 * b + k] += a[6 * i + k] * q[i];
      ++j;
    }
  }
}
// Qualified topology only: levels omit world body 0 and parent IDs precede
// children. A level barrier publishes every parent before its children read.
static __device__ inline void d_comvel(int w, int *active, real_t *cdof, real_t *qvel, int *bp, int *ba,
                                       int *bn, int *dj, int *jt, int nb, int nv, real_t *cvel, real_t *cdot,
                                       int *status, const int *level_offsets = nullptr,
                                       const int *level_bodies = nullptr, int nlevels = 0) {
  real_t *v = cvel + w * nb * 6, *d = cdot + w * nv * 6;
  const real_t *a = cdof + w * nv * 6, *q = qvel + w * nv;
  if (threadIdx.x == 0)
    status[w] = 0;
  d1_zero_block(v, nb * 6);
  d1_zero_block(d, nv * 6);
  __syncthreads();
  if (!active[w])
    return;
  if (active[w] != 1) {
    if (threadIdx.x == 0)
      status[w] = 1;
    return;
  }
  if (!d1_finite_block(a, nv * 6, q, nv)) {
    if (threadIdx.x == 0)
      status[w] = 2;
    return;
  }
  if (level_offsets && level_bodies && nlevels > 0) {
    for (int l = 0; l < nlevels; ++l) {
      for (int k = level_offsets[l] + threadIdx.x; k < level_offsets[l + 1]; k += blockDim.x)
        d1_comvel_body(level_bodies[k], a, q, bp, ba, bn, dj, jt, v, d);
      __syncthreads();
    }
  } else if (threadIdx.x == 0) {
    for (int b = 1; b < nb; ++b)
      d1_comvel_body(b, a, q, bp, ba, bn, dj, jt, v, d);
  }
  __syncthreads();
  if (!d1_finite_block(v, nb * 6, d, nv * 6)) {
    if (threadIdx.x == 0)
      status[w] = 3;
    d1_zero_block(v, nb * 6);
    d1_zero_block(d, nv * 6);
  }
  __syncthreads();
}
// Compressed spatial inertia multiplication, not a dense BLAS operand.
static __device__ inline void d_inertia(const real_t *i, const real_t *v, real_t *f) {
  f[0] = i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5];
  f[1] = i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5];
  f[2] = i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4];
  f[3] = i[8] * v[1] - i[7] * v[2] + i[9] * v[3];
  f[4] = i[6] * v[2] - i[8] * v[0] + i[9] * v[4];
  f[5] = i[7] * v[0] - i[6] * v[1] + i[9] * v[5];
}
static __device__ inline void d_cross_force(const real_t *v, const real_t *t, real_t *x) {
  x[0] = -v[2] * t[1] + v[1] * t[2];
  x[1] = v[2] * t[0] - v[0] * t[2];
  x[2] = -v[1] * t[0] + v[0] * t[1];
  x[3] = -v[2] * t[4] + v[1] * t[5];
  x[4] = v[2] * t[3] - v[0] * t[5];
  x[5] = -v[1] * t[3] + v[0] * t[4];
  x[0] += -v[5] * t[4] + v[4] * t[5];
  x[1] += v[5] * t[3] - v[3] * t[5];
  x[2] += -v[4] * t[3] + v[3] * t[4];
}
static __device__ inline void d1_rne_body(int b, const real_t *d, const real_t *q, const real_t *i,
                                          const real_t *v, const int *bp, const int *ba, const int *bn,
                                          real_t *acc, real_t *f, int nb = 0, bool component_planes = false) {
  real_t t[6] = {};
  const int adr = ba[b], n = bn[b];
  if (n == 1)
    for (int k = 0; k < 6; ++k)
      t[k] = d[6 * adr + k] * q[adr];
  else
    for (int j = 0; j < n; ++j)
      if (q[adr + j] != 0)
        for (int k = 0; k < 6; ++k)
          t[k] += d[6 * (adr + j) + k] * q[adr + j];
  real_t local_acc[6];
  for (int k = 0; k < 6; ++k) {
    const int out = component_planes ? k * nb + b : 6 * b + k;
    const int parent = component_planes ? k * nb + bp[b] : 6 * bp[b] + k;
    local_acc[k] = acc[parent] + t[k];
    acc[out] = local_acc[k];
  }
  real_t fi[6], corr[6], cross[6];
  d_inertia(i + 10 * b, local_acc, fi);
  d_inertia(i + 10 * b, v + 6 * b, corr);
  d_cross_force(v + 6 * b, corr, cross);
  for (int k = 0; k < 6; ++k)
    f[component_planes ? k * nb + b : 6 * b + k] = fi[k] + cross[k];
}
static __device__ inline void d1_rne_forward(const real_t *d, const real_t *q, const real_t *i,
                                             const real_t *v, const int *bp, const int *ba, const int *bn,
                                             const real_t *gravity, int disabled, int nb, real_t *acc,
                                             real_t *f, const int *lo, const int *lb, int nl,
                                             bool component_planes = false) {
  if (threadIdx.x == 0 && !disabled)
    for (int k = 0; k < 3; ++k)
      acc[component_planes ? (3 + k) * nb : 3 + k] = -gravity[k];
  __syncthreads();
  if (lo && lb && nl > 0) {
    for (int l = 0; l < nl; ++l) {
      for (int k = lo[l] + threadIdx.x; k < lo[l + 1]; k += blockDim.x)
        d1_rne_body(lb[k], d, q, i, v, bp, ba, bn, acc, f, nb, component_planes);
      __syncthreads();
    }
  } else if (threadIdx.x == 0) {
    for (int b = 1; b < nb; ++b)
      d1_rne_body(b, d, q, i, v, bp, ba, bn, acc, f, nb, component_planes);
  }
  __syncthreads();
}

// Compute only the forward part of bias RNE while the velocity input is live.
// Its private status does not decide velocity publication or the later S gate.
// The reverse force tree and public bias commit remain after S admission.
// Every exit closes flag reads; writing exits publish all outputs to the block.
static __device__ __forceinline__ void
d_comvel_bias_separate(int w, int *active, real_t *cdof, real_t *qvel, int *bp, int *ba, int *bn, int *dj,
                     int *jt, int nb, int nv, real_t *cvel, real_t *cdot, int *status, real_t *cinert,
                     real_t *gravity, int disabled, real_t *cache_acc, real_t *cache_force, int *cache_status,
                     int *cache_valid, int64_t *cache_state, int64_t *cache_forward,
                     const int64_t *state_epoch, const int64_t *forward_epoch, const int *level_offsets,
                     const int *level_bodies, int nlevels) {
  d_comvel(w, active, cdof, qvel, bp, ba, bn, dj, jt, nb, nv, cvel, cdot, status, level_offsets, level_bodies,
           nlevels);
  __syncthreads();
  if (threadIdx.x == 0) {
    cache_valid[w] = 0;
    cache_status[w] = 0;
  }
  real_t *acc = cache_acc + (long long)w * nb * 6, *f = cache_force + (long long)w * nb * 6;
  d1_zero_block(acc, nb * 6);
  d1_zero_block(f, nb * 6);
  __syncthreads();
  if (active[w] != 1 || status[w] != 0) {
    __syncthreads();
    return;
  }
  const real_t *d = cdot + (long long)w * nv * 6, *q = qvel + (long long)w * nv;
  const real_t *i = cinert + (long long)w * nb * 10, *v = cvel + (long long)w * nb * 6;
  // Successful comvel immediately above proved qvel/cdof and its cvel/dot
  // outputs finite. No intervening producer writes them inside this call.
  if (!d1_finite_block(i, nb * 10)) {
    if (threadIdx.x == 0)
      cache_status[w] = 2;
    __syncthreads();
    return;
  }
  d1_rne_forward(d, q, i, v, bp, ba, bn, gravity, disabled, nb, acc, f, level_offsets, level_bodies, nlevels,
                 true);
  if (!d1_finite_block(acc, nb * 6, f, nb * 6)) {
    if (threadIdx.x == 0)
      cache_status[w] = 3;
    d1_zero_block(acc, nb * 6);
    d1_zero_block(f, nb * 6);
    __syncthreads();
    return;
  }
  if (threadIdx.x == 0) {
    cache_state[w] = state_epoch[w];
    cache_forward[w] = forward_epoch[w];
    cache_valid[w] = 1;
  }
  __syncthreads();
}

static __device__ __forceinline__ void
d_comvel_bias_cached(int w, int *active, real_t *cdof, real_t *qvel, int *bp, int *ba, int *bn, int *dj,
                     int *jt, int nb, int nv, real_t *cvel, real_t *cdot, int *status, real_t *cinert,
                     real_t *gravity, int disabled, real_t *cache_acc, real_t *cache_force, int *cache_status,
                     int *cache_valid, int64_t *cache_state, int64_t *cache_forward,
                     const int64_t *state_epoch, const int64_t *forward_epoch, const int *level_offsets,
                     const int *level_bodies, int nlevels) {
  const real_t *a = cdof + size_t(w) * nv * 6, *q = qvel + size_t(w) * nv;
  const real_t *inertia = cinert + size_t(w) * nb * 10;
  // Exceptional inputs keep the original stage-specific status and clear
  // behavior. Normal inputs need one combined admission vote, then one tree.
  if (active[w] != 1 || !d1_finite_block(a, nv * 6, q, nv, inertia, nb * 10)) {
    d_comvel_bias_separate(w, active, cdof, qvel, bp, ba, bn, dj, jt, nb, nv,
        cvel, cdot, status, cinert, gravity, disabled, cache_acc, cache_force,
        cache_status, cache_valid, cache_state, cache_forward, state_epoch,
        forward_epoch, level_offsets, level_bodies, nlevels);
    return;
  }
  real_t *velocity = cvel + size_t(w) * nb * 6, *dot = cdot + size_t(w) * nv * 6;
  real_t *acc = cache_acc + size_t(w) * nb * 6, *force = cache_force + size_t(w) * nb * 6;
  if (threadIdx.x == 0) {
    status[w] = 0;
    cache_valid[w] = 0;
    cache_status[w] = 0;
  }
  d1_zero_block(velocity, nb * 6);
  d1_zero_block(dot, nv * 6);
  d1_zero_block(force, nb * 6);
  for (int k = threadIdx.x; k < nb * 6; k += blockDim.x)
    acc[k] = !disabled && k % nb == 0 && k / nb >= 3 ? -gravity[k / nb - 3] : REAL_C(0.0);
  __syncthreads();
  if (level_offsets && level_bodies && nlevels > 0) {
    for (int level = 0; level < nlevels; ++level) {
      for (int k = level_offsets[level] + threadIdx.x; k < level_offsets[level + 1]; k += blockDim.x) {
        const int body = level_bodies[k];
        // This same thread just produced the body's cvel/cdof_dot. Its parent
        // velocity AND acceleration were published at the preceding level.
        d1_comvel_body(body, a, q, bp, ba, bn, dj, jt, velocity, dot);
        d1_rne_body(body, dot, q, inertia, velocity, bp, ba, bn, acc, force, nb, true);
      }
      __syncthreads();
    }
  } else {
    if (threadIdx.x == 0)
      for (int body = 1; body < nb; ++body) {
        d1_comvel_body(body, a, q, bp, ba, bn, dj, jt, velocity, dot);
        d1_rne_body(body, dot, q, inertia, velocity, bp, ba, bn, acc, force, nb, true);
      }
    __syncthreads();
  }
  if (!d1_finite_block(velocity, nb * 6, dot, nv * 6)) {
    if (threadIdx.x == 0)
      status[w] = 3;
    d1_zero_block(velocity, nb * 6);
    d1_zero_block(dot, nv * 6);
    d1_zero_block(acc, nb * 6);
    d1_zero_block(force, nb * 6);
    __syncthreads();
    return;
  }
  if (!d1_finite_block(acc, nb * 6, force, nb * 6)) {
    if (threadIdx.x == 0)
      cache_status[w] = 3;
    d1_zero_block(acc, nb * 6);
    d1_zero_block(force, nb * 6);
    __syncthreads();
    return;
  }
  if (threadIdx.x == 0) {
    cache_state[w] = state_epoch[w];
    cache_forward[w] = forward_epoch[w];
    cache_valid[w] = 1;
  }
  __syncthreads();
}

// Full-world-block helpers; blockDim.x must be a positive multiple of 32.
// Warp collectives only exchange registers; shared publication uses block barriers.
namespace d_passive_detail {
static __device__ inline int first_bad(int index) {
  __shared__ int first[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int value = libero_warp::min(index);
  if (lane == 0)
    first[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < (blockDim.x + 31) / 32 ? first[lane] : libero_warp::no_index;
    value = libero_warp::min(value);
    if (lane == 0)
      first[0] = value;
  }
  __syncthreads();
  const int result = first[0];
  __syncthreads();
  return result;
}
static __device__ inline bool bad(bool value) {
  return first_bad(value ? 0 : libero_warp::no_index) != libero_warp::no_index;
}
} // namespace d_passive_detail
static __device__ __forceinline__ void
d_joint_passive_count(int w, real_t *qpos, real_t *qvel, int *active, int *jtype, int *qadr, int *dadr,
                      real_t *stiffness, real_t *spring, real_t *damping, real_t *out, int *status, int nv,
                      int nq, int njnt, int joint_disjoint, const int *joint_type_offsets,
                      const int *joint_type_indices) {
  qpos += (long long)w * nq;
  qvel += (long long)w * nv;
  out += (long long)w * nv;
  if (threadIdx.x == 0)
    status[w] = active[w] != 0 && active[w] != 1 ? 1 : 0;
  libero_vec::fill_block(out, nv, REAL_C(0.0));
  __syncthreads();
  if (active[w] != 1)
    return;
  bool invalid = false;
  invalid |= libero_vec::any_nonfinite_block_lane(qpos, nq);
  invalid |= libero_vec::any_nonfinite_block_lane(qvel, nv);
  if (d_passive_detail::bad(invalid)) {
    if (threadIdx.x == 0)
      status[w] = 2;
    __syncthreads();
    return;
  }
  if (joint_disjoint && joint_type_offsets && joint_type_indices) {
    // Qualification proves unique output intervals. Type cohorts keep the
    // free/hinge/slide route uniform while each independent joint writes.
    for (int type = 0; type < 4; ++type) {
      if (type == 1)
        continue;
      for (int k = joint_type_offsets[type] + threadIdx.x; k < joint_type_offsets[type + 1];
           k += blockDim.x) {
        const int i = joint_type_indices[k], q = qadr[i], d = dadr[i];
        const real_t v = stiffness[i];
        if (type == 0) {
          // Two elements belong to this free joint; never cross a joint mask.
          const real2_t qp = libero_vec::load_pair(qpos + q), sp = libero_vec::load_pair(spring + q);
          const real2_t old = libero_vec::load_pair(out + d);
          libero_vec::store_pair(out + d, old.x - v * (qp.x - sp.x), old.y - v * (qp.y - sp.y));
          out[d + 2] -= v * (qpos[q + 2] - spring[q + 2]);
        } else
          out[d] -= v * (qpos[q] - spring[q]);
      }
    }
  } else if (threadIdx.x == 0)
    for (int i = 0; i < njnt; ++i) {
      // Malformed overlapping descriptors retain original ordered updates.
      int t = jtype[i], q = qadr[i], d = dadr[i];
      real_t v = stiffness[i];
      if (t == 0)
        for (int axis = 0; axis < 3; ++axis)
          out[d + axis] -= v * (qpos[q + axis] - spring[q + axis]);
      else if (t == 2 || t == 3)
        out[d] -= v * (qpos[q] - spring[q]);
    }
  __syncthreads();
  for (int k = 2 * threadIdx.x; k + 1 < nv; k += 2 * blockDim.x) {
    const real2_t old = libero_vec::load_pair(out + k), d = libero_vec::load_pair(damping + k),
                  v = libero_vec::load_pair(qvel + k);
    libero_vec::store_pair(out + k, old.x - d.x * v.x, old.y - d.y * v.y);
  }
  if ((nv & 1) && threadIdx.x == 0)
    out[nv - 1] -= damping[nv - 1] * qvel[nv - 1];
  __syncthreads();
}

namespace d_passive_detail {
struct Vec {
  real_t v[3];
  __device__ real_t &operator[](int i) { return v[i]; }
  __device__ real_t operator[](int i) const { return v[i]; }
};
static __device__ inline Vec cross(Vec a, Vec b) {
  return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
}
static __device__ inline Vec rot(const real_t *m, Vec v) {
  return {{m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
           m[6] * v[0] + m[7] * v[1] + m[8] * v[2]}};
}
static __device__ inline Vec rot_t(const real_t *m, Vec v) {
  return {{m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
           m[2] * v[0] + m[5] * v[1] + m[8] * v[2]}};
}
} // namespace d_passive_detail

namespace d_passive_detail {
// The body producer retains scalar arithmetic and diagnostic-prefix semantics.
static __device__ inline bool fluid_body(int b, const real_t *cvel, const real_t *subtree_com,
                                         const real_t *xipos, const real_t *ximat, int *rootid, real_t *box,
                                         real_t *wind, real_t *opt, int diagnostic, real_t *stages,
                                         int stride, real_t *payload) {
  constexpr real_t pi = REAL_C(3.14159265358979323846);
  int root = rootid[b];
  const real_t *m = ximat + 9 * b;
  Vec offset = {{xipos[3 * b] - subtree_com[3 * root], xipos[3 * b + 1] - subtree_com[3 * root + 1],
                 xipos[3 * b + 2] - subtree_com[3 * root + 2]}};
  const real2_t cv01 = libero_vec::load_pair(cvel + 6 * b), cv23 = libero_vec::load_pair(cvel + 6 * b + 2),
                cv45 = libero_vec::load_pair(cvel + 6 * b + 4);
  Vec ang = {{cv01.x, cv01.y, cv23.x}};
  Vec lin = {{cv23.y, cv45.x, cv45.y}};
  Vec local_ang = rot_t(m, ang), ox = cross(offset, ang);
  for (int r = 0; r < 3; ++r)
    lin[r] -= ox[r];
  Vec object_lin = rot_t(m, lin), local_wind = rot_t(m, {{wind[0], wind[1], wind[2]}});
  Vec local_lin;
  for (int r = 0; r < 3; ++r)
    local_lin[r] = object_lin[r] - local_wind[r];
  for (int r = 0; r < 3; ++r)
    if (!isfinite(offset[r]) || !isfinite(local_ang[r]) || !isfinite(object_lin[r]) ||
        !isfinite(local_wind[r]) || !isfinite(local_lin[r])) {
      return false;
    }
  real_t *st = stages + b * stride;
  if (diagnostic != 0)
    for (int r = 0; r < 3; ++r) {
      st[r] = box[3 * b + r];
      st[3 + r] = local_ang[r];
      st[6 + r] = object_lin[r];
      st[9 + r] = REAL_C(0.0);
      st[12 + r] = local_wind[r];
      st[15 + r] = local_ang[r];
      st[18 + r] = local_lin[r];
    }
  const real2_t boxxy = libero_vec::load_pair(box + 3 * b);
  real_t x = boxxy.x, y = boxxy.y, z = box[3 * b + 2], visc = opt[0], density = opt[1];
  Vec torque = {{REAL_C(0.), REAL_C(0.), REAL_C(0.)}}, force = {{REAL_C(0.), REAL_C(0.), REAL_C(0.)}};
  if (visc > REAL_C(0.0)) {
    real_t diam = (x + y + z) / REAL_C(3.0), ang_scl = -pi * diam * diam * diam * visc,
           lin_scl = -REAL_C(3.0) * pi * diam * visc;
    for (int r = 0; r < 3; ++r) {
      torque[r] = local_ang[r] * ang_scl;
      force[r] = local_lin[r] * lin_scl;
    }
  }
  if (density > REAL_C(0.0)) {
    force[0] -= REAL_C(0.5) * density * y * z * fabs(local_lin[0]) * local_lin[0];
    force[1] -= REAL_C(0.5) * density * x * z * fabs(local_lin[1]) * local_lin[1];
    force[2] -= REAL_C(0.5) * density * x * y * fabs(local_lin[2]) * local_lin[2];
    torque[0] -= density * x * (y * y * y * y + z * z * z * z) * fabs(local_ang[0]) * local_ang[0] / REAL_C(64.0);
    torque[1] -= density * y * (x * x * x * x + z * z * z * z) * fabs(local_ang[1]) * local_ang[1] / REAL_C(64.0);
    torque[2] -= density * z * (x * x * x * x + y * y * y * y) * fabs(local_ang[2]) * local_ang[2] / REAL_C(64.0);
  }
  Vec wt = rot(m, torque), wf = rot(m, force);
  for (int r = 0; r < 3; ++r)
    if (!isfinite(torque[r]) || !isfinite(force[r]) || !isfinite(wt[r]) || !isfinite(wf[r])) {
      return false;
    }
  if (diagnostic != 0)
    for (int r = 0; r < 3; ++r) {
      st[21 + r] = torque[r];
      st[24 + r] = force[r];
      st[27 + r] = wt[r];
      st[30 + r] = wf[r];
    }
  for (int r = 0; r < 3; ++r) {
    payload[r] = offset[r];
    payload[3 + r] = wt[r];
    payload[6 + r] = wf[r];
  }
  return true;
}
} // namespace d_passive_detail

static __device__ __forceinline__ void
d_fluid_count(int w, real_t *cvel, real_t *subtree_com, real_t *xipos, real_t *ximat, real_t *cdof,
              int *active, int *rootid, real_t *mass, real_t *box, int *ancestor, real_t *wind, real_t *opt,
              int passive_disabled, int diagnostic, real_t *stages, real_t *qfrc, int *status, int nbody,
              int nv, const int *body_dof_offsets, const int *body_dof_indices) {
  using namespace d_passive_detail;
  cvel += (long long)w * nbody * 6;
  subtree_com += (long long)w * nbody * 3;
  xipos += (long long)w * nbody * 3;
  ximat += (long long)w * nbody * 9;
  cdof += (long long)w * nv * 6;
  qfrc += (long long)w * nv;
  int stride = 33 + 7 * nv;
  stages += (long long)w * nbody * stride;
  if (diagnostic != 0)
    libero_vec::fill_block(stages, (size_t)nbody * stride, REAL_C(0.0));
  __syncthreads();
  if (active[w] == 0 || status[w] != 0)
    return;
  if (passive_disabled != 0 || (opt[0] <= REAL_C(0.0) && opt[1] <= REAL_C(0.0)))
    return;
  bool invalid = false;
  invalid |= libero_vec::any_nonfinite_block_lane(qfrc, nv);
  invalid |= libero_vec::any_nonfinite_block_lane(cvel, nbody * 6);
  invalid |= libero_vec::any_nonfinite_block_lane(subtree_com, nbody * 3);
  invalid |= libero_vec::any_nonfinite_block_lane(xipos, nbody * 3);
  invalid |= libero_vec::any_nonfinite_block_lane(ximat, nbody * 9);
  invalid |= libero_vec::any_nonfinite_block_lane(cdof, nv * 6);
  if (d_passive_detail::bad(invalid)) {
    if (threadIdx.x == 0)
      status[w] = 2;
    __syncthreads();
    return;
  }

  __shared__ real_t payload[9];
  __shared__ int body_ok;
  for (int b = 1; b < nbody; ++b) {
    if (mass[b] < REAL_C(1e-15))
      continue;
    if (threadIdx.x == 0)
      body_ok = fluid_body(b, cvel, subtree_com, xipos, ximat, rootid, box, wind, opt, diagnostic, stages,
                           stride, payload);
    __syncthreads();
    if (!body_ok) {
      if (threadIdx.x == 0)
        status[w] = 2;
      __syncthreads();
      return;
    }
    Vec offset, wt, wf;
    // One shared load per warp; every lane receives the same body payload.
    for (int r = 0; r < 3; ++r) {
      real_t o = REAL_C(0.0), t = REAL_C(0.0), f = REAL_C(0.0);
      if ((threadIdx.x & 31) == 0) {
        o = payload[r];
        t = payload[3 + r];
        f = payload[6 + r];
      }
      offset[r] = __shfl_sync(0xffffffffu, o, 0);
      wt[r] = __shfl_sync(0xffffffffu, t, 0);
      wf[r] = __shfl_sync(0xffffffffu, f, 0);
    }
    real_t *st = stages + b * stride;
    const bool sparse = diagnostic == 0 && body_dof_offsets && body_dof_indices;
    const int begin = sparse ? body_dof_offsets[b] : 0;
    const int count = sparse ? body_dof_offsets[b + 1] - begin : nv;
    for (int base = 0; base < count; base += blockDim.x) {
      const int ordinal = base + threadIdx.x;
      const int j = ordinal < count ? (sparse ? body_dof_indices[begin + ordinal] : ordinal) : nv;
      Vec jp = {{REAL_C(0.), REAL_C(0.), REAL_C(0.)}}, jr = {{REAL_C(0.), REAL_C(0.), REAL_C(0.)}};
      bool projection_bad = false, result_bad = false;
      real_t result = REAL_C(0.0);
      if (j < nv) {
        if (sparse || ancestor[b * nv + j] != 0) {
          jr = {{cdof[6 * j], cdof[6 * j + 1], cdof[6 * j + 2]}};
          Vec cp = cross(jr, offset);
          for (int r = 0; r < 3; ++r)
            jp[r] = cdof[6 * j + 3 + r] + cp[r];
        }
        for (int r = 0; r < 3; ++r)
          projection_bad |= !isfinite(jp[r]) || !isfinite(jr[r]);
        if (!projection_bad) {
          real_t qforce = REAL_C(0.0), qtorque = REAL_C(0.0);
          for (int r = 0; r < 3; ++r)
            if (wf[r] != REAL_C(0.0))
              qforce += jp[r] * wf[r];
          result = qfrc[j] + qforce;
          for (int r = 0; r < 3; ++r)
            if (wt[r] != REAL_C(0.0))
              qtorque += jr[r] * wt[r];
          result += qtorque;
          result_bad = !isfinite(result);
        }
      }
      const int local = projection_bad ? 2 * j : result_bad ? 2 * j + 1 : libero_warp::no_index;
      const int fail = d_passive_detail::first_bad(local);
      // Preserve the scalar prefix: bad projection writes nothing; a bad
      // result writes qfrc but does not write its final diagnostic slot.
      if (j < nv && 2 * j < fail) {
        if (diagnostic != 0)
          for (int r = 0; r < 3; ++r) {
            st[33 + nv * r + j] = jp[r];
            st[33 + 3 * nv + nv * r + j] = jr[r];
          }
        qfrc[j] = result;
        if (diagnostic != 0 && 2 * j + 1 < fail)
          st[33 + 6 * nv + j] = result;
      }
      __syncthreads();
      if (fail != libero_warp::no_index) {
        if (sparse)
          for (int k = threadIdx.x; k < fail / 2; k += blockDim.x)
            if (ancestor[b * nv + k] == 0)
              qfrc[k] = (qfrc[k] + REAL_C(0.0)) + REAL_C(0.0);
        __syncthreads();
        if (threadIdx.x == 0)
          status[w] = 2;
        __syncthreads();
        return;
      }
    }
    if (sparse)
      for (int j = threadIdx.x; j < nv; j += blockDim.x)
        if (ancestor[b * nv + j] == 0)
          qfrc[j] = (qfrc[j] + REAL_C(0.0)) + REAL_C(0.0);
    __syncthreads();
  }
  __syncthreads();
}

// Diagnostic and exceptional inputs retain the original ordered partial writes.
static __device__ __forceinline__ void
d_fluid_original(int w, const flash_libero::FluidPassiveData &in) {
  d_fluid_count(w, in.cvel, in.subtree_com, in.xipos, in.ximat, in.cdof, in.active,
                in.rootid, in.mass, in.box, in.ancestor, in.wind, in.opt,
                in.passive_disabled, in.diagnostic, in.stages, in.qfrc, in.status,
                in.nbody, in.nv, in.body_dof_offsets, in.body_dof_indices);
}

// CPU mj_passive applies inertia-box body forces in ascending body order.
// Body payloads are independent: produce them concurrently, then give each
// DOF one ordered accumulator. No atomic additions or per-body CTA barriers.
// The caller publishes qfrc/status after this function returns.
static __device__ __forceinline__ void
d_fluid_parallel(int w, const flash_libero::FluidPassiveData &in, real_t *workspace) {
  using namespace d_passive_detail;
  if (in.diagnostic || !workspace) {
    d_fluid_original(w, in);
    return;
  }
  if (in.active[w] == 0 || in.status[w] != 0 || in.passive_disabled ||
      (in.opt[0] <= REAL_C(0.0) && in.opt[1] <= REAL_C(0.0)))
    return;

  const int nb = in.nbody, nv = in.nv;
  const real_t *cv = in.cvel + size_t(w) * nb * 6;
  const real_t *com = in.subtree_com + size_t(w) * nb * 3;
  const real_t *pos = in.xipos + size_t(w) * nb * 3;
  const real_t *mat = in.ximat + size_t(w) * nb * 9;
  const real_t *dof = in.cdof + size_t(w) * nv * 6;
  real_t *force = in.qfrc + size_t(w) * nv;
  real_t *payload = workspace + size_t(w) * (9 * nb + nv);
  real_t *candidate = payload + 9 * nb;

  bool bad = libero_vec::any_nonfinite_block_lane(force, nv);
  bad |= libero_vec::any_nonfinite_block_lane(cv, nb * 6);
  bad |= libero_vec::any_nonfinite_block_lane(com, nb * 3);
  bad |= libero_vec::any_nonfinite_block_lane(pos, nb * 3);
  bad |= libero_vec::any_nonfinite_block_lane(mat, nb * 9);
  bad |= libero_vec::any_nonfinite_block_lane(dof, nv * 6);
  for (int b = 1 + threadIdx.x; b < nb; b += blockDim.x)
    if (!(in.mass[b] < REAL_C(1e-15)))
      bad |= !fluid_body(b, cv, com, pos, mat, in.rootid, in.box, in.wind,
                         in.opt, 0, in.stages, 33 + 7 * nv, payload + 9 * b);
  // Publishes every body payload and returns a uniform register decision.
  if (__syncthreads_or(bad)) {
    d_fluid_original(w, in);
    return;
  }

  bad = false;
  for (int j = threadIdx.x; j < nv; j += blockDim.x) {
    real_t result = force[j];
    for (int b = 1; b < nb; ++b) {
      if (in.mass[b] < REAL_C(1e-15))
        continue;
      if (!in.ancestor[b * nv + j]) {
        result = (result + REAL_C(0.0)) + REAL_C(0.0);
        continue;
      }
      const real_t *p = payload + 9 * b;
      const Vec offset = {{p[0], p[1], p[2]}};
      const Vec jr = {{dof[6 * j], dof[6 * j + 1], dof[6 * j + 2]}};
      const Vec cp = cross(jr, offset);
      Vec jp;
      for (int r = 0; r < 3; ++r) {
        jp[r] = dof[6 * j + 3 + r] + cp[r];
        bad |= !isfinite(jp[r]) || !isfinite(jr[r]);
      }
      real_t qforce = REAL_C(0.0), qtorque = REAL_C(0.0);
      for (int r = 0; r < 3; ++r)
        if (p[6 + r] != REAL_C(0.0))
          qforce += jp[r] * p[6 + r];
      result += qforce;
      for (int r = 0; r < 3; ++r)
        if (p[3 + r] != REAL_C(0.0))
          qtorque += jr[r] * p[3 + r];
      result += qtorque;
      bad |= !isfinite(result);
    }
    candidate[j] = result;
  }
  // Until this vote no durable force has changed. On failure, replay the
  // original body/DOF order to preserve its exact error and partial outputs.
  if (__syncthreads_or(bad)) {
    d_fluid_original(w, in);
    return;
  }
  libero_vec::copy_block(candidate, force, nv);
}

} // namespace physics_velocity

// Keep the shape-specific policy local to this implementation.

// World-local admission/publication owned by fused_substep.cu.
// These routines contain no block barriers. Preserve the caller's
// lane-0 scopes; live_bias and project_counts retain their own guards.

namespace physics_velocity {
static __device__ __forceinline__ void d_gate(int w, int *active, int *error, int64_t *state, int64_t *source,
                                              int64_t *epoch, int *valid, int64_t *written, int *gate,
                                              int require) {
  gate[w] = 0;
  if (active[w] != 1 || error[w])
    return;
  if (state[w] != source[w]) {
    error[w] = 203;
    return;
  }
  if (require == 1 && (valid[w] != 1 || written[w] != epoch[w]))
    return;
  gate[w] = 1;
}

static __device__ __forceinline__ void d_finish(int w, int *gate, int *status, int64_t *epoch, int *valid,
                                                int64_t *written, int *error, int repair, int code) {
  valid[w] = 0;
  if (gate[w] != 1 || error[w])
    return;
  if (status[w] != 0 && !(repair == 1 && status[w] == 25)) {
    error[w] = code;
    return;
  }
  written[w] = epoch[w];
  valid[w] = 1;
}

static __device__ __forceinline__ void d_velocity_status(int w, int *a, int *b, int *c, int *out) {
  out[w] = a[w] ? 100 + a[w] : (b[w] ? 200 + b[w] : (c[w] ? 300 + c[w] : 0));
}

} // namespace physics_velocity

namespace physics_bias {
static __device__ __forceinline__ void d_gate(int w, int *active, int *error, int64_t *state, int64_t *source,
                                              int64_t *epoch, int *valid, int64_t *written, int *gate,
                                              int require) {
  gate[w] = 0;
  if (active[w] != 1 || error[w])
    return;
  if (state[w] != source[w]) {
    error[w] = 203;
    return;
  }
  if (require == 1 && (valid[w] != 1 || written[w] != epoch[w]))
    return;
  gate[w] = 1;
}

static __device__ __forceinline__ void d_reference_gate(int w, int *done, int *err, int64_t *written,
                                                        int64_t *expected, int64_t *epoch, int *gate,
                                                        int *first) {
  if (gate[w] == 1 && (done[w] != 1 || err[w] || written[w] != expected[w] || expected[w] != epoch[w])) {
    gate[w] = 0;
    if (!first[w])
      first[w] = 208;
  }
}

} // namespace physics_bias

namespace physics_force {
static __device__ __forceinline__ void d_gate(int w, int *active, int *error, int64_t *state, int64_t *source,
                                              int64_t *epoch, int *valid, int64_t *written, int *gate,
                                              int require) {
  gate[w] = 0;
  if (active[w] != 1 || error[w])
    return;
  if (state[w] != source[w]) {
    error[w] = 203;
    return;
  }
  if (require == 1 && (valid[w] != 1 || written[w] != epoch[w]))
    return;
  gate[w] = 1;
}

static __device__ __forceinline__ void d_force_gate(int w, int *valid, int *gate) {
  if (valid[w] != 1)
    gate[w] = 0;
}

static __device__ __forceinline__ void d_actuation_finish(int w, int *gate, int *status, int *first) {
  if (gate[w] == 1 && status[w]) {
    gate[w] = 0;
    if (!first[w])
      first[w] = 214;
  }
}

} // namespace physics_force

namespace physics_constraint_projection::cf_s {
static __device__ __forceinline__ void n_mask(int w, const int *active, const int *first, const int *smooth,
                                              const int *reference, const int *sstatus, int *admitted,
                                              int *error) {
  admitted[w] = 0;
  error[w] = first[w];
  if (!error[w] && sstatus[w])
    error[w] = sstatus[w];
  if (active[w] == 1 && !error[w] && smooth[w] == 1 && reference[w] == 1)
    admitted[w] = 1;
}

static __device__ __forceinline__ void project_counts(int w, const int *admitted, const int *source_counts,
                                                      const int *source_ncon, int *nefc, int *ne, int *nf,
                                                      int *ncon, int *count, int *empty, int *newton,
                                                      int *active, int *status, int *item, int rows,
                                                      int contacts, int nv) {
  nefc[w] = ne[w] = nf[w] = ncon[w] = count[w] = empty[w] = newton[w] = active[w] = status[w] = 0;
  item[w] = -1;
  if (admitted[w] != 1)
    return;
  ne[w] = source_counts[w * 5];
  nf[w] = source_counts[w * 5 + 1];
  nefc[w] = count[w] = source_counts[w * 5 + 3];
  ncon[w] = source_ncon[w];
  if (!nefc[w])
    empty[w] = 1;
  else if (nefc[w] > 0)
    newton[w] = active[w] = 1;
  int n = nefc[w], eq = ne[w], fr = nf[w], nl = source_counts[w * 5 + 2], nc = ncon[w], bad = -1;
  if (n < 0 || n > rows)
    bad = 3;
  else if (eq < 0 || eq > n)
    bad = 0;
  else if (fr < 0 || fr > n - eq)
    bad = 1;
  else if (nl < 0 || nl > n - eq - fr)
    bad = 2;
  else if (source_counts[w * 5 + 4] != n * nv)
    bad = 4;
  else if (nc < 0 || nc > contacts)
    bad = 5;
  if (bad >= 0) {
    status[w] = 9001;
    item[w] = bad;
  }
}

static __device__ __forceinline__ void increment_plan(int w, const int *mask, const int *first,
                                                      int64_t *epoch) {
  if (mask[w] == 1 && !first[w])
    epoch[w]++;
}

} // namespace physics_constraint_projection::cf_s


namespace flash_libero {
namespace {

// Collective preparation for the current position and velocity. No actuation,
// Newton, warmstart update, or forward-complete publication occurs here.
// The returned flag describes same-state numerical reuse for the FP64 path.
__device__ __forceinline__ bool prepare_dynamics(const SubstepArguments &args, int world) {
  const auto &workspace = args.forward_workspace;
  uint64_t *cache = workspace.metadata + size_t(world) * ForwardCacheWords;
  // The Start entry clears validity. Within this physical substep only the
  // controller (ctrl) and Newton/Post (qacc/history) run between forwards.
  // Model/mocap are immutable in this native owner. A reset changes state;
  // a position rebuild changes generation even when state itself is unchanged.
  const bool reuse = args.state.requested_mask[world] == 1 && args.begin.active[world] == 1 &&
                     !args.state.first_error[world] && !args.position.work[world] &&
                     cache[ForwardCacheValid] == 1 &&
                     cache[ForwardCacheState] == uint64_t(args.state.epoch[world]) &&
                     cache[ForwardCachePosition] == workspace.position_generation[world];
  CF_STEP_PROFILE_STAGES(reuse ? CF_STEP_PROFILE_PHASE(VelocityReuse) : CF_STEP_PROFILE_PHASE(VelocityCold));
  // Velocity and passive forces consume the current, already prepared position.
  if (args.state.requested_mask[world] != 0) {
    if (threadIdx.x == 0) {
      const auto &in = args.velocity_gate;
      physics_velocity::d_gate(world, in.active, in.error, in.state, in.source, in.epoch, in.valid,
                               in.written, in.gate, in.require);
    }
    __syncthreads();
    if (!reuse) {
      {
        const auto &in = args.transmission;
        physics_velocity::d_transmission_velocity_block(
            world, in.moment, in.qvel, in.active, in.position_status, in.nv, in.nu, in.velocity, in.status);
      }
      // The collective transmission helper publishes its outputs on every exit.
      {
        const auto &in = args.body_velocity;
        physics_velocity::d_comvel_bias_cached(
            world, in.active, in.cdof, in.qvel, in.bp, in.ba, in.bn, in.dj, in.jt, in.nb, in.nv, in.cvel,
            in.cdot, in.status, in.cinert, in.gravity, in.disabled, in.cache_acc, in.cache_force,
            in.cache_status, in.cache_valid, in.cache_state, in.cache_forward, in.state_epoch, in.forward_epoch,
            in.level_offsets, in.level_bodies, in.nlevels);
      }
      // d_comvel_bias_cached owns its exit synchronization on every path.
      {
        const auto &in = args.joint_passive;
        physics_velocity::d_joint_passive_count(world, in.qpos, in.qvel, in.active, in.jtype, in.qadr, in.dadr,
                                                in.stiffness, in.spring, in.damping, in.out, in.status, in.nv,
                                                in.nq, in.njnt, in.joint_disjoint, in.joint_type_offsets,
                                                in.joint_type_indices);
      }
      __syncthreads();
      {
        const auto &in = args.fluid_passive;
        physics_velocity::d_fluid_parallel(world, in, workspace.fluid);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const auto &in = args.velocity_status;
      if (!reuse)
        physics_velocity::d_velocity_status(world, in.a, in.b, in.c, in.out);
      else
        // Post still consumes the immutable RNE forward payload. Its numerical
        // contents stay live; only the current-forward publication changes.
        args.body_velocity.cache_forward[world] = args.body_velocity.forward_epoch[world];
      const auto &finish = args.velocity_finish;
      physics_velocity::d_finish(world, finish.gate, finish.status, finish.epoch, finish.valid,
                                 finish.written, finish.error, finish.repair, finish.code);
    }
  }
  // Constraint velocity/reference acceleration.
  CF_STEP_PROFILE_NEXT(reuse ? CF_STEP_PROFILE_PHASE(ReferenceReuse) : CF_STEP_PROFILE_PHASE(ReferenceCold));
  if (threadIdx.x == 0) {
    const auto &in = args.reference_admission;
    forward_cuda::reference_mask(world, in.valid, in.written, in.expected, in.first, in.active);
  }
  __syncthreads();
  if (!reuse) {
    {
      const auto &in = args.reference_gate;
      physics_constraint_reference::cf_s::reference_gate(world, in.active, in.status, in.committed, in.counts,
                                                         in.qvel, in.epoch, in.expected, in.ready, in.rows,
                                                         in.nv);
    }
    __syncthreads();
    {
      const auto &in = args.constraint_reference;
      physics_constraint_reference::cf_s::reference(world, in.committed, in.counts, in.efc_J, in.kbip,
                                                    in.efc_pos, in.efc_margin, in.qvel, in.qvel_epoch, in.vel,
                                                    in.aref, in.ref_epoch, in.rows, in.nv);
    }
    // reference() owns the publication of every numerical tile; no additional
    // barrier is needed before commit_reference consumes those same rows.
    {
      const auto &in = args.reference_commit;
      physics_constraint_reference::cf_s::commit_reference(world, in.ready, in.status, in.committed, in.counts,
                                                           in.sv, in.sa, in.epoch, in.pv, in.pa, in.ref_epoch,
                                                           in.rows);
    }
  } else if (threadIdx.x == 0) {
    // S-position and qvel have not changed; keep vel/aref in their retained
    // global owners and republish the new forward epoch without copying rows.
    // A masked controller forward still clears reference_ready in its gate.
    // Readiness is publication, not numerical validity: restore it on a hit.
    args.reference_commit.ready[world] = 1;
    args.reference_commit.ref_epoch[world] = args.reference_commit.epoch[world];
  }
  __syncthreads();
  // Bias, actuation, and unconstrained acceleration.
  CF_STEP_PROFILE_NEXT(reuse ? CF_STEP_PROFILE_PHASE(BiasReuse) : CF_STEP_PROFILE_PHASE(BiasCold));
  if (args.state.requested_mask[world] != 0) {
    if (threadIdx.x == 0) {
      const auto &in = args.bias_gate;
      physics_bias::d_gate(world, in.active, in.error, in.state, in.source, in.epoch, in.valid, in.written,
                           in.gate, in.require);
      const auto &reference = args.bias_reference;
      physics_bias::d_reference_gate(world, reference.done, reference.err, reference.written,
                                     reference.expected, reference.epoch, reference.gate, reference.first);
    }
    __syncthreads();
    if (!reuse) {
      {
        const auto &in = args.bias_forces;
        physics_bias::d_rne_bias_cached(world, in.active, in.cdot, in.qvel, in.cinert, in.cvel, in.cdof, in.bp,
                                        in.ba, in.bn, in.db, in.gravity, in.disabled, in.nb, in.nv, in.cacc,
                                        in.frc, in.result, in.status, in.cache_acc, in.cache_force,
                                        in.cache_status, in.cache_valid, in.cache_state, in.cache_forward,
                                        in.state_epoch, in.forward_epoch, in.level_offsets, in.level_bodies,
                                        in.nlevels, in.children_offsets, in.children_indices);
      }
      // Normal RNE exits publish their outputs. On an exceptional fallback,
      // status stays leader-owned and commit's entry barrier closes all writes.
      {
        const auto &in = args.bias_commit;
        physics_bias::d_commit_bias_block(world, in.active, in.gate, in.status, in.epoch, in.error, in.work,
                                          in.pub, in.valid, in.written, in.width);
      }
    } else if (threadIdx.x == 0) {
      const auto &in = args.bias_commit;
      physics_velocity::d_finish(world, in.gate, in.status, in.epoch, in.valid, in.written,
                                 in.error, 0, 211);
    }
    if (threadIdx.x == 0) {
      if (reuse)
        ++cache[ForwardCacheReused];
      else {
        ++cache[ForwardCacheComputed];
        if (cache[ForwardCacheValid])
          ++cache[ForwardCacheInvalidated];
        const auto &velocity = args.velocity_finish;
        const auto &bias = args.bias_commit;
        const auto &reference = args.reference_commit;
        cache[ForwardCacheValid] = velocity.valid[world] == 1 && !velocity.error[world] &&
                                  velocity.written[world] == velocity.epoch[world] &&
                                  bias.valid[world] == 1 && !bias.error[world] &&
                                  bias.written[world] == bias.epoch[world] &&
                                  reference.ready[world] == 1 && !reference.status[world] &&
                                  reference.committed[world] == 1 &&
                                  reference.ref_epoch[world] == reference.epoch[world] &&
                                  args.body_velocity.cache_valid[world] == 1 &&
                                  !args.body_velocity.cache_status[world];
        cache[ForwardCacheState] = uint64_t(args.state.epoch[world]);
        cache[ForwardCachePosition] = workspace.position_generation[world];
      }
    }
  }
  return reuse;
}

// Collective step2 suffix. The controller may change ctrl after preparation;
// no position, velocity, bias, or reference computation is repeated here.
__device__ __forceinline__ void solve_dynamics(const SubstepArguments &args, int world,
                                              unsigned char *shared, bool post_outputs,
                                              bool reuse, int smooth_reuse_status = -1) {
  CF_STEP_PROFILE_STAGES(CF_STEP_PROFILE_PHASE(Actuation));
  const auto &workspace = args.forward_workspace;
  if (args.state.requested_mask[world] != 0) {
    if (threadIdx.x == 0) {
      const auto &in = args.actuation_gate;
      physics_force::d_gate(world, in.active, in.error, in.state, in.source, in.epoch, in.valid, in.written,
                            in.gate, in.require);
      const auto &control = args.control_gate;
      physics_force::d_force_gate(world, control.valid, control.gate);
    }
    __syncthreads();
    {
      const auto &in = args.control_copy;
      physics_force::d_copy_admitted_block(world, in.gate, in.source, in.target, in.width);
    }
    __syncthreads();
    {
      const auto &in = args.actuation;
      physics_force::d_actuation_count(
          world, in.length, in.moment, in.velocity, in.ctrl, in.active, in.vel_status, in.gain, in.bias0,
          in.bias1, in.bias2, in.bias_type, in.ctrl_lo, in.ctrl_hi, in.force_lo, in.force_hi,
          in.force_limited, in.disable_flags, in.force, in.force_pre, in.ctrl_used, in.ctrl_route,
          in.force_route, in.qfrc_actuator, in.status, in.warn_badctrl_count, in.warn_badctrl_lastinfo,
          in.warning_number, in.warning_lastinfo, in.nv, in.nu);
    }
    // d_actuation_count ends with a collective publication barrier. Preserve
    // that barrier once; the next operations are owned by the same leader.
    if (threadIdx.x == 0) {
      const auto &in = args.actuation_finish;
      physics_force::d_actuation_finish(world, in.gate, in.status, in.first);
      const auto &smooth = args.smooth_gate;
      physics_force::d_gate(world, smooth.active, smooth.error, smooth.state, smooth.source, smooth.epoch,
                            smooth.valid, smooth.written, smooth.gate, smooth.require);
      const auto &force = args.smooth_force_gate;
      physics_force::d_force_gate(world, force.valid, force.gate);
    }
    __syncthreads();
    {
      CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(Smooth));
      const auto &in = args.smooth_acceleration;
      if (reuse && smooth_reuse_status >= 0 && in.active[world] == 1 &&
          args.smooth_gate.gate[world] == 1) {
        real_t *force = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::force);
        real_t *acc = flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::acceleration);
        for (int i = threadIdx.x; i < in.nv; i += blockDim.x) {
          force[i] = args.smooth_commit.pf[size_t(world) * in.nv + i];
          acc[i] = args.smooth_commit.pa[size_t(world) * in.nv + i];
        }
        if (threadIdx.x == 0) in.status[world] = smooth_reuse_status;
      } else {
        physics_force::d_smooth_chain(
            world, in.passive, in.bias, in.applied, in.actuator, in.active, in.entry_status, in.smooth, in.seed,
            in.nv, in.xfrc, in.xipos, in.subtree_com, in.cdof, in.body_ids, in.body_count, in.body_rootid,
            in.ancestor, in.scratch, in.stages, in.qfrc, in.d02_status, in.nbody, in.qm, in.count, in.madr,
            in.parent, in.ld, in.inv, in.sqrt_inv, in.solution, in.pivot, in.solve_status, in.warning_count,
            in.warning_last, in.nm, in.rhs, in.qacc_smooth, in.status, in.source_ld, in.source_inv,
            in.source_sqrt_inv, in.source_pivot, in.source_status, in.source_warning_count,
            in.source_warning_last, in.source_active, in.position_valid, in.position_written, in.forward_epoch,
            in.factor_reuse_count, in.body_dof_offsets, in.body_dof_indices, workspace.mass_plan);
      }
    }
    __syncthreads();
    {
      const auto &in = args.smooth_commit;
      physics_force::d_commit_smooth_shared_block(world, in.active, in.gate, in.status, in.epoch, in.error,
                                                  in.pf, in.pa, in.valid, in.written, in.repaired, in.width);
    }
    __syncthreads();
  }
  // Assemble the existing S-to-Newton inputs; keep every solve and warmstart write.
  CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(NewtonBridge));
  if (threadIdx.x == 0) {
    const auto &in = args.newton_admission;
    physics_constraint_projection::cf_s::n_mask(world, in.active, in.first, in.smooth, in.reference,
                                                in.sstatus, in.admitted, in.error);
    const auto &counts = args.constraint_counts;
    physics_constraint_projection::cf_s::project_counts(
        world, counts.admitted, counts.source_counts, counts.source_ncon, counts.nefc, counts.ne, counts.nf,
        counts.ncon, counts.count, counts.empty, counts.newton, counts.active, counts.status, counts.item,
        counts.rows, counts.contacts, counts.nv);
  }
  __syncthreads();
  {
    const auto &in = args.cone_parameters;
    physics_constraint_projection::cf_s::project_cone_heads_verified(
        world, in.admitted, in.nefc, in.ne, in.nf, in.ncon, in.typ, in.id, in.dim, in.mu, in.friction,
        in.out_dim, in.out_mu, in.out_friction, in.status, in.item, in.rows, in.contacts, in.compact_heads,
        in.compact_count, in.compact_valid, in.head_dim, in.head_mu, in.head_friction, in.head_cone_valid);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const auto &in = args.newton_epoch;
    physics_constraint_projection::cf_s::increment_plan(world, in.mask, in.first, in.epoch);
    // Three scalar channels per world, qualified by bind(). Their producing
    // epoch increment and consuming copies have the same thread owner.
    for (const auto &copy : args.post_epochs)
      copy.destination[world] = copy.source[world];
  }

  // Post observes current contact addresses and the three input epochs.
  const auto &contacts = args.contact_addresses;
  const int count = contacts.count[world];
  if (contacts.active[world] == 1 && contacts.status[world] == 0 && contacts.committed[world] == 1 &&
      count >= 0 && count <= contacts.capacity) {
    for (int contact = threadIdx.x; contact < count; contact += blockDim.x) {
      const int at = world * contacts.capacity + contact;
      contacts.current[at] = contacts.excluded[at] ? -1 : contacts.address[at];
    }
  }
  // Contact addresses and the leader's scalar snapshots are independent;
  // one barrier publishes both before Newton/Post consume them.
  __syncthreads();
  const auto &newton = args.newton;
  // This invocation owns the force workspace until Newton has finished.
  // Skipped forwards and prefix/reset continuations keep canonical inputs.
  const auto &smooth = args.smooth_commit;
  const bool resident = args.state.requested_mask[world] != 0 && smooth.valid[world] == 1 &&
                        !smooth.error[world] && smooth.written[world] == smooth.epoch[world];
  physics_newton::NewtonOptions options{};
  options.stop_policy = newton.stop_policy;
  options.audit = newton.audit;
  options.first_disagreement = newton.first_disagreement;
  options.smooth_acceleration = resident ?
      flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::acceleration) : nullptr;
  options.smooth_force = resident ?
      flash_libero::shared_at<real_t>(shared, flash_libero::SmoothLayout::force) : nullptr;
  const auto &request = *newton.parameters;
  CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(Newton));
  physics_newton::solve_newton(shared, request.model, request.worlds[world], newton.upstream, newton.enabled,
                               newton.publication, newton.expected_epoch, options);
  CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(Post));
  const bool post_evaluated = physics_post::run(request.post, world,
                    flash_libero::shared_at<real_t>(shared, flash_libero::NewtonLayout::qacc),
                    flash_libero::shared_at<real_t>(shared, flash_libero::NewtonLayout::mathdx), post_outputs);
  __syncthreads();
  CF_STEP_PROFILE_NEXT(CF_STEP_PROFILE_PHASE(ForwardPublish));
  if (threadIdx.x == 0) {
    const auto &in = args.forward_publish;
    // Intermediate completion proves current dynamics, not current Post data.
    // Post.valid stays false until a real force/sensor publication succeeds.
    const int *completed = post_evaluated ? in.nvalid : request.post.valid;
    forward_cuda::publish(world, in.active, completed, in.nerror, in.first, in.projection, in.source,
                          in.state, in.valid, in.written, in.error);
  }
  __syncthreads();
}

// Complete CPU-compatible forward. Keep the FP64 order and every warmstart
// write; only the FP32 substep entry separates these two physical stages.
__device__ __forceinline__ void forward_dynamics(const SubstepArguments &args, int world,
                                                 unsigned char *shared, bool post_outputs,
                                                 int smooth_reuse_status = -1) {
  CF_STEP_PROFILE_SCOPE(Forward);
  const bool reuse = prepare_dynamics(args, world);
  solve_dynamics(args, world, shared, post_outputs, reuse, smooth_reuse_status);
}

// Start another forward without rebuilding position on a valid same-epoch hit.
// A miss is returned to the existing multi-CTA prefix, never computed here.
__device__ void begin_forward(const SubstepArguments &args, int world, const int *mask) {
  CF_STEP_PROFILE_SCOPE(BeginForward);
  // All admission operations belong to the same leader. Only their final
  // results are consumed by other lanes; intermediate barriers add no ordering.
  if (threadIdx.x == 0) {
    args.state.forward_mask[world] = mask[world];
    const auto &in = args.foundation_gate;
    forward_cuda::gate_copy(world, in.admission, in.count, in.epoch, in.expected, in.producer_valid, in.event,
                            in.errors, in.stage_out, in.local_out, in.item_out, in.event_out, in.active,
                            in.valid, in.capacity, in.stage, in.admitted, in.public_error, in.public_stage,
                            in.public_local, in.public_item, in.public_event, in.world);
    args.state.requested_mask[world] = args.begin.mask[world];
    const auto &begin = args.begin;
    forward_cuda::begin(world, begin.mask, begin.error, begin.state, begin.source, begin.epoch, begin.active,
                        begin.position, begin.velocity, begin.bias, begin.smooth, begin.expected);
    const auto &cache = args.position;
    position_reuse::check(world, args.begin.active, args.state.first_error, args.state.epoch, cache.saved,
                          cache.cached, cache.constraint_status, cache.constraint_committed,
                          cache.collision_error, cache.mass_warning, cache.collision_warning, cache.work,
                          cache.hit, args.begin.epoch, args.begin.position, cache.written,
                          cache.collision_mask, cache.transport_active, cache.consumed);
  }
  __syncthreads();
}

// The field inventory and defaults are the existing MuJoCo data-reset owners.
// checkPos and checkVel each have their own reset and epoch update, in that order.
__device__ void check_and_reset(const SubstepArguments &args, int world, const real_t *values, int width,
                                int warning) {
  CF_STEP_PROFILE_SCOPE(Checks);
  const auto &reset = args.recovery;
  cf_recovery::check(world, values, reset.active, reset.reset_mask, reset.reason, reset.bad_index,
                     reset.source_error, width, warning);
  __syncthreads();
  // First publish the check's writes, then finish every lane's mask read.
  // The next check reuses this mask: a fast warp must not overwrite it while
  // another warp still reads the previous decision. The vote returns a stable
  // register predicate and closes that read lifetime before either warp leaves.
  const bool needs_reset = __syncthreads_or(reset.reset_mask[world] == 1);
  if (!needs_reset)
    return;
  for (int field = 0; field < reset.field_count; ++field) {
    const auto &view = reset.fields[field];
    cf_recovery::reset_f64(world, view.values, view.defaults, reset.reset_mask, view.width);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    cf_recovery::warnings(world, reset.warning_number, reset.warning_lastinfo, reset.reset_mask, reset.reason,
                          reset.bad_index);
    cf_recovery::protocol(world, reset.reset_mask, args.state.first_error, args.state.epoch,
                          args.state.expected_epoch, args.state.forward_valid, args.state.step_valid,
                          reset.reset_events);
  }
  __syncthreads();
}

// The pipeline owns gather, seed, solve and publication in an independent TU.
__device__ void run_controller(const SubstepArguments &args, int world, bool policy_step) {
  CF_STEP_PROFILE_SCOPE(Controller);
  physics_controller::run_controller(args, world, policy_step, substep_shared_data());
}

__device__ void integrate(const SubstepArguments &args, int world) {
  CF_STEP_PROFILE_SCOPE(Integrate);
  const auto &state = args.state;
  physics_euler::cf_euler::gate_core_block(world, state.intent, state.epoch, state.forward_valid,
                                           state.forward_written, state.forward_error, state.first_error,
                                           args.euler.active);
  __syncthreads();
  {
    const auto &in = args.euler;
    physics_euler::cf_euler::evaluate_core(
        world, in.active, in.qp, in.qv, in.time, in.qm, in.smooth, in.constraint, in.acc, in.damping, in.madr,
        in.parent, in.jt, in.padr, in.vadr, in.qh, in.qhi, in.count, in.ma, in.damp, in.status, in.ms, in.wc,
        in.wl, in.dt, in.nq, in.nv, in.nm, in.nj, in.qacc_stride, in.joint_quat, in.joint_type_offsets,
        in.joint_type_indices);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    physics_euler::cf_euler::publish(world, args.euler.active, args.euler.status, state.epoch,
                                     state.step_written, state.step_valid, state.first_error,
                                     state.forward_valid);
    physics_euler::cf_euler::expected(world, state.intent, state.expected_epoch);
    physics_euler::cf_euler::count(world, state.step_valid, state.integration_count);
  }
  __syncthreads();
}
__device__ void finish_substep(const SubstepArguments &args, int world, SubstepEntry entry) {
  if (entry != SubstepEntry::AccelerationReset) {
    if (threadIdx.x == 0) {
      cf_recovery::acceleration(world, args.state.intent, args.state.first_error, args.state.forward_valid,
                                args.state.forward_written, args.state.epoch, args.newton.valid,
                                args.newton.written, args.newton.expected_epoch, args.recovery.active);
    }
    __syncthreads();
    check_and_reset(args, world, args.state.qacc, args.state.nv, 6);
    if (args.recovery.reset_mask[world] == 1) {
      if (threadIdx.x == 0)
        args.continuation[world] = int(SubstepEntry::AccelerationReset);
      return;
    }
  }

  // After an acceleration reset the caller has performed the required full
  // reset-only forward. Continue at Euler without repeating the controller.
  integrate(args, world);
  if (threadIdx.x == 0)
    args.continuation[world] = int(SubstepEntry::Complete);
}
} // namespace

// This is the complete world-local substep body. Normal execution enters once.
// Continuations exist only across the original parallel-prefix/reset boundaries.
// A block-wide return always occurs after a barrier and a world-uniform predicate.
__global__ void fused_substep_kernel(const SubstepArguments *__restrict__ arguments, bool policy_step,
                                     SubstepEntry entry, bool publish_post) {
  const auto &args = *arguments;
  auto *shared = substep_shared_data();
  const int world = blockIdx.y;
  if (entry != SubstepEntry::Start && args.continuation[world] != int(entry))
    return;
  CF_STEP_PROFILE_BEGIN(entry, policy_step);
  int smooth_reuse_status = -1;

  if (entry == SubstepEntry::Start) {
    if (threadIdx.x == 0) {
      args.continuation[world] = int(SubstepEntry::Complete);
      args.forward_workspace.metadata[size_t(world) * ForwardCacheWords + ForwardCacheValid] = 0;
    }
    __syncthreads();

    // RobotEnv.step: sim.forward() precedes controller.set_goal/run_controller.
    // Preserve both intermediate force snapshots on policy steps (CPU robot
    // force/torque history); other substeps have no force consumer before the
    // final forward. Newton and warmstart still execute at all three points.
    // Admission and the parallel position prefix were launched by the caller.
    forward_dynamics(args, world, shared, policy_step);
    // This register value dies at a position suspension. The next entry starts
    // with -1; it cannot reuse a payload produced before a prefix refresh.
    const auto &smooth = args.smooth_commit;
    if (args.state.forward_valid[world] == 1 && !args.state.first_error[world] &&
        smooth.valid[world] == 1 && !smooth.error[world] &&
        smooth.written[world] == smooth.epoch[world])
      smooth_reuse_status = smooth.repaired[world] ? 25 : 0;
    if (threadIdx.x == 0) {
      const auto &in = args.controller_plan;
      osc_cuda::plan(world, in.active, in.first, in.needs, in.refresh, in.call, in.force);
    }
    // begin_forward consumes the plan on the same leader, then publishes once.
    begin_forward(args, world, args.controller_plan.refresh);
    if (args.position.work[world]) {
      if (threadIdx.x == 0)
        args.continuation[world] = int(SubstepEntry::ControllerPosition);
      return;
    }
  }

  if (entry == SubstepEntry::Start || entry == SubstepEntry::ControllerPosition) {
    // Controller.update requests its own forward, including its Newton/warmstart.
    forward_dynamics(args, world, shared, policy_step, smooth_reuse_status);
    run_controller(args, world, policy_step);

    // MuJoCo mj_step: checkPos -> checkVel -> forward -> checkAcc -> Euler.
    if (threadIdx.x == 0) {
      cf_recovery::admit(world, args.state.intent, args.state.first_error, args.recovery.active);
    }
    __syncthreads();
    check_and_reset(args, world, args.state.qpos, args.state.nq, 4);
    check_and_reset(args, world, args.state.qvel, args.state.nv, 5);
    begin_forward(args, world, args.state.intent);
    if (args.position.work[world]) {
      if (threadIdx.x == 0)
        args.continuation[world] = int(SubstepEntry::PhysicsPosition);
      return;
    }
  }

  if (entry != SubstepEntry::AccelerationReset) {
    // A qualified control graph may defer force/sensor work in intermediate
    // substeps. The final substep keeps Post here, before Euler, exactly as in
    // mj_forward/mj_step. Every forward still solves Newton and publishes qacc.
    forward_dynamics(args, world, shared, publish_post);
  }
  finish_substep(args, world, entry);
}

#if FLASH_REAL_BITS == 32
// Experimental FP32 schedule. It is a distinct kernel, so the accepted
// three-forward kernel keeps its own register and shared-memory allocation.
__global__ void single_solve_substep_kernel(const SubstepArguments *__restrict__ arguments,
                                           bool policy_step, SubstepEntry entry, bool publish_post) {
  const auto &args = *arguments;
  auto *shared = substep_shared_data();
  const int world = blockIdx.y;
  if (entry != SubstepEntry::Start && entry != SubstepEntry::PositionChecks &&
      args.continuation[world] != int(entry))
    return;
  CF_STEP_PROFILE_BEGIN(entry, policy_step);
  // step1 checks must precede the multi-CTA position prefix. This small entry
  // runs before the host enqueues that prefix; it performs no forward/solve.
  if (entry == SubstepEntry::PositionChecks) {
    if (threadIdx.x == 0) {
      args.continuation[world] = int(SubstepEntry::Complete);
      args.forward_workspace.metadata[size_t(world) * ForwardCacheWords + ForwardCacheValid] = 0;
      cf_recovery::admit(world, args.state.intent, args.state.first_error, args.recovery.active);
    }
    __syncthreads();
    check_and_reset(args, world, args.state.qpos, args.state.nq, 4);
    check_and_reset(args, world, args.state.qvel, args.state.nv, 5);
    return;
  }
  if (entry == SubstepEntry::Start) {
    CF_STEP_PROFILE_SCOPE(Forward);
    prepare_dynamics(args, world);
    if (threadIdx.x == 0) {
      const auto &in = args.controller_plan;
      // MJWarp consumes freshly prepared physical inputs even on the first
      // substep after reset; it never starts from the constructor's cache.
      osc_cuda::plan(world, in.active, in.first, in.needs, in.refresh, in.call, 1);
    }
    __syncthreads();
    run_controller(args, world, policy_step);
    solve_dynamics(args, world, shared, publish_post, false);
  }
  finish_substep(args, world, entry);
}
#endif

// Select exactly the suspended worlds. This small kernel performs no physics.
__global__ void select_substep_continuation(const SubstepArguments *__restrict__ arguments,
                                            SubstepEntry entry, int worlds, int *predicate) {
  if (threadIdx.x != 0)
    return;
  const auto &args = *arguments;
  int any = 0;
  for (int world = 0; world < worlds; ++world) {
    const int selected = args.continuation[world] == int(entry);
    args.position.work[world] = selected;
    args.recovery.reset_mask[world] = selected;
    any |= selected;
  }
  predicate[0] = any;
}
} // namespace flash_libero
