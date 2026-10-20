#include "precision.cuh"
// Fixed-size controller numerical body. CMake compiles this file once for all
// catalogue NVs; the world block and MathDx operators share the configured size.
#include "osc_solver.cuh"
#include "osc_profile.cuh"
#include "../step/substep_types.cuh"
#include "../step/substep_shared.cuh"
#include "../common/vector.cuh"
#include <cmath>
using flash_libero::SubstepShape;

// Fixed-size FP64 OSC numerical implementation, independent of scene NV.

// Shared FP64 controller implementation.

namespace physics_controller {

// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.

extern "C" __device__ void osc_inverse7(real_t *, real_t *, int *, int *);

extern "C" __device__ void osc_mm_6_6_7(real_t *, real_t *, real_t *);

extern "C" __device__ void osc_mm_7_6_7(real_t *, real_t *, real_t *);

extern "C" __device__ void osc_mm_7_6_6(real_t *, real_t *, real_t *);

extern "C" __device__ void osc_mm_7_7_6(real_t *, real_t *, real_t *);
extern "C" __device__ void osc_mm_3_3_3(real_t *, real_t *, real_t *);

namespace osc_cuda {
static __device__ inline real_t clip(real_t x, real_t lo, real_t hi) {
  return isnan(x) ? x : fmin(hi, fmax(lo, x));
}

// One-sided Jacobi: retain scale, ordered rotations, cutoff and orthogonality test.

// Each lane owns a row of both column-major operands. Column-pair rotations
// remain ordered, while their row arithmetic and the final inverse cooperate.
// The three independent OSC problems use one full warp and two half warps.
struct OscPseudoinverseWork {
  real_t columns[36];
  real_t vectors[36];
};

// Only the first N row lanes carry values. A padded 4/8-lane tree sums
// those rows, then broadcasts group zero to every participating lane. Padding
// groups still execute the shuffles; all rotation predicates remain uniform.
template <int N, int Width> static __device__ real_t pinv_row_sum(real_t value, unsigned mask) {
  constexpr int padded = N <= 4 ? 4 : 8;
#pragma unroll
  for (int delta = padded / 2; delta; delta >>= 1)
    value += __shfl_xor_sync(mask, value, delta, Width);
  return __shfl_sync(mask, value, 0, Width);
}

// Jacobi always rotates slots 0/1. Constant-index cyclic moves bring the
// next ordered column into slot 1; both operands follow the same permutation.
// The post-sweep norm/correlation tail alone uses the runtime selector.
template <int N> struct PseudoinverseRow {
  real_t values[N] = {};

  __device__ __forceinline__ real_t get(int column) const {
    real_t result = REAL_C(0.0);
#pragma unroll
    for (int k = 0; k < N; ++k)
      if (column == k)
        result = values[k];
    return result;
  }
  __device__ __forceinline__ void rotate_tail_left() {
    const real_t first = values[1];
#pragma unroll
    for (int k = 1; k < N - 1; ++k)
      values[k] = values[k + 1];
    values[N - 1] = first;
  }
  __device__ __forceinline__ void rotate_left() {
    const real_t first = values[0];
#pragma unroll
    for (int k = 0; k < N - 1; ++k)
      values[k] = values[k + 1];
    values[N - 1] = first;
  }
};

template <int N, int Width>
static __device__ __noinline__ void pinv_group(const real_t *input, real_t *output,
                                               OscPseudoinverseWork &work, int *rank, int *ok,
                                               unsigned mask) {
  const int lane = threadIdx.x & (Width - 1);
  real_t scale = REAL_C(0.0);
  if (lane < N)
    for (int col = 0; col < N; ++col)
      scale = fmax(scale, fabs(input[lane * 6 + col]));
#pragma unroll 1
  for (int row = 0; row < N; ++row)
    scale = fmax(scale, __shfl_sync(mask, scale, row, Width));
  for (int z = lane; z < 36; z += Width)
    output[z] = REAL_C(0.0);
  if (lane == 0) {
    *rank = 0;
    *ok = 1;
  }
  if (!(scale > REAL_C(0.0)))
    return;
  // Each row lane owns N column values and N right-singular-vector values.
  // Sweep updates and permutations use constant indices. Runtime selection
  // is confined to the post-sweep norm/correlation tail.
  PseudoinverseRow<N> columns, vectors;
  if (lane < N) {
#pragma unroll
    for (int col = 0; col < N; ++col) {
      columns.values[col] = input[lane * 6 + col] / scale;
      vectors.values[col] = col == lane ? REAL_C(1.0) : REAL_C(0.0);
    }
  }
  bool converged = false;
  CF_OSC_PROFILE_PINV_COUNTERS();
#pragma unroll 1
  for (int sweep = 0; sweep < 64; ++sweep) {
    CF_OSC_PROFILE_SWEEP();
    int rotations = 0;
#pragma unroll 1
    for (int p = 0; p < N - 1; ++p) {
#pragma unroll 1
      for (int q = p + 1; q < N; ++q) {
        const real_t bp = columns.values[0];
        const real_t bq = columns.values[1];
        const real_t alpha = pinv_row_sum<N, Width>(bp * bp, mask);
        const real_t beta = pinv_row_sum<N, Width>(bq * bq, mask);
        const real_t gamma = pinv_row_sum<N, Width>(bp * bq, mask);
        if (fabs(gamma) > real_t(REAL_EPSILON) * sqrt(alpha) * sqrt(beta)) {
          const real_t zeta = (beta - alpha) / (2 * gamma);
          const real_t t = (zeta < 0 ? -REAL_C(1.0) : REAL_C(1.0)) / (fabs(zeta) + sqrt(1 + zeta * zeta));
          const real_t c = 1 / sqrt(1 + t * t), s = c * t;
          if (lane < N) {
            const real_t vp = vectors.values[0], vq = vectors.values[1];
            columns.values[0] = c * bp - s * bq;
            columns.values[1] = s * bp + c * bq;
            vectors.values[0] = c * vp - s * vq;
            vectors.values[1] = s * vp + c * vq;
          }
          ++rotations;
          CF_OSC_PROFILE_ROTATION();
        }
        columns.rotate_tail_left();
        vectors.rotate_tail_left();
      }
      // This p-stage started at [p,p+1,...,N-1,0,...,p-1].
      // q visited N-p-1 tail columns; p more moves close the N-1 cycle.
#pragma unroll 1
      for (int restore = 0; restore < p; ++restore) {
        columns.rotate_tail_left();
        vectors.rotate_tail_left();
      }
      columns.rotate_left();
      vectors.rotate_left();
    }
    // N-1 outer moves left column N-1 first. Restore canonical order before
    // every exit, including early convergence, and before the next sweep.
    columns.rotate_left();
    vectors.rotate_left();
    if (rotations == 0) {
      converged = true;
      break;
    }
  }
  real_t norms[N] = {}, singular[N] = {}, weights[N] = {}, maxnorm = REAL_C(0.0);
  bool retained[N] = {};
#pragma unroll 1
  for (int p = 0; p < N; ++p) {
    const real_t value = columns.get(p);
    norms[p] = pinv_row_sum<N, Width>(value * value, mask);
    singular[p] = sqrt(norms[p]);
    maxnorm = fmax(maxnorm, singular[p]);
  }
  int retained_count = 0;
#pragma unroll 1
  for (int p = 0; p < N; ++p) {
    retained[p] = singular[p] > REAL_C(1e-15) * maxnorm;
    if (retained[p]) {
      ++retained_count;
      weights[p] = (1 / norms[p]) / scale;
    }
  }
  real_t maxcorr = REAL_C(0.0);
#pragma unroll 1
  for (int p = 0; p < N; ++p) {
#pragma unroll 1
    for (int q = p + 1; q < N; ++q)
      if (retained[p] && retained[q]) {
        const real_t bp = columns.get(p);
        const real_t bq = columns.get(q);
        const real_t gamma = pinv_row_sum<N, Width>(bp * bq, mask);
        maxcorr = fmax(maxcorr, fabs(gamma) / (singular[p] * singular[q]));
      }
  }
  if (maxcorr < REAL_C(2e-15))
    converged = true;
  // Only reconstruction needs another lane's row. Publish the two completed
  // matrices once, then retain the existing cell-owned output distribution.
  if (lane < N) {
#pragma unroll
    for (int col = 0; col < N; ++col) {
      work.columns[col * 6 + lane] = columns.values[col];
      work.vectors[col * 6 + lane] = vectors.values[col];
    }
  }
  __syncwarp(mask);
  for (int z = lane; z < 36; z += Width) {
    const int row = z / 6, col = z % 6;
    real_t value = REAL_C(0.0);
    if (row < N && col < N)
      for (int k = 0; k < N; ++k)
        if (retained[k])
          value += work.vectors[k * 6 + row] * work.columns[k * 6 + col] * weights[k];
    output[z] = value;
  }
  if (lane == 0) {
    CF_OSC_PROFILE_PINV_RESULT();
    *rank = retained_count;
    *ok = converged;
  }
}

// All world-block threads enter. The caller owns the block publication barrier.
static __device__ void pinv_three_block(const real_t *full, const real_t *position, const real_t *orientation,
                                        real_t *full_inverse, real_t *position_inverse,
                                        real_t *orientation_inverse, int *ranks, int *ok,
                                        unsigned char *shared) {
  auto *work = flash_libero::shared_at<OscPseudoinverseWork>(shared, flash_libero::OscLayout::pinv);
  static_assert(sizeof(OscPseudoinverseWork) == 72 * sizeof(real_t), "OSC pinv shared layout");
  if (threadIdx.x < 32)
    pinv_group<6, 32>(full, full_inverse, work[0], ranks, ok, 0xffffffffu);
  else if (threadIdx.x < 64) {
    const int problem = 1 + (threadIdx.x - 32) / 16;
    const unsigned mask = 0xffffu << ((threadIdx.x & 31) >= 16 ? 16 : 0);
    pinv_group<3, 16>(problem == 1 ? position : orientation,
                      problem == 1 ? position_inverse : orientation_inverse, work[problem], ranks + problem,
                      ok + problem, mask);
  }
}

static __device__ inline void rotation_delta(real_t x, real_t y, real_t z, real_t *r, real_t *raw,
                                             float *cast) {
  real_t angle = sqrt(x * x + y * y + z * z), f = angle == 0 ? 0 : sin(angle * REAL_C(.5)) / angle;
  raw[0] = x * f;
  raw[1] = y * f;
  raw[2] = z * f;
  raw[3] = angle == 0 ? 1 : cos(angle * REAL_C(.5));
  float q[4] = {(float)raw[3], (float)raw[0], (float)raw[1], (float)raw[2]};
  real_t normacc = 0;
  for (int i = 0; i < 4; i++) {
    cast[i] = q[i];
    float product = q[i] * q[i];
    normacc += (real_t)product;
  }
  float norm = (float)normacc;
  real_t factor = sqrt(2 / (real_t)norm);
  for (int i = 0; i < 4; i++)
    q[i] = (float)((real_t)q[i] * factor);
  r[0] = 1 - (real_t)(q[2] * q[2]) - (real_t)(q[3] * q[3]);
  r[4] = 1 - (real_t)(q[1] * q[1]) - (real_t)(q[3] * q[3]);
  r[8] = 1 - (real_t)(q[1] * q[1]) - (real_t)(q[2] * q[2]);
  r[1] = (real_t)(q[1] * q[2] - q[3] * q[0]);
  r[2] = (real_t)(q[1] * q[3] + q[2] * q[0]);
  r[3] = (real_t)(q[1] * q[2] + q[3] * q[0]);
  r[5] = (real_t)(q[2] * q[3] - q[1] * q[0]);
  r[6] = (real_t)(q[1] * q[3] - q[2] * q[0]);
  r[7] = (real_t)(q[2] * q[3] + q[1] * q[0]);
}

// The entire block enters. Shared operands make each library operation a
// true device-side collective inside this kernel, including all dense GEMMs.
static __device__ inline bool control(const flash_libero::ControllerSolveData &in, int w,
                                      unsigned char *shared, bool live, const real_t *local_state) {
  // Input selections are views, not copied argument packages. The host has
  // already proved private, nonoverlapping owners for this native entry.
  real_t *mass = live ? in.pm : in.mass, *jacobian = live ? in.pj : in.jacobian;
  real_t *persist_m = live ? in.mass : nullptr, *persist_j = live ? in.jacobian : nullptr;

  real_t *m = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::m);
  real_t *lu = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::lu);
  real_t *mi = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::mi);
  real_t *j = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::j);
  real_t *jt = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::jt);
  real_t *a = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::a);
  real_t *ap = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::ap);
  real_t *ar = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::ar);
  real_t *lf = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::lf);
  real_t *lp = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::lp);
  real_t *lr = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::lr);
  real_t *mjt = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::mjt);
  real_t *jbar = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::jbar);
  real_t *project = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::project);
  real_t *delta = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::delta);
  real_t *current = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::current);
  real_t *gr = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::gr);
  real_t *gp = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::gp);
  real_t *desired = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::desired);
  real_t *lp3 = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::lp3);
  real_t *lr3 = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::lr3);
  real_t *pd = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::pd);
  int &info = *flash_libero::shared_at<int>(shared, flash_libero::OscLayout::info);
  int *piv = flash_libero::shared_at<int>(shared, flash_libero::OscLayout::piv);
  int &admitted = *flash_libero::shared_at<int>(shared, flash_libero::OscLayout::admitted);
  int &reuse = *flash_libero::shared_at<int>(shared, flash_libero::OscLayout::reuse);
  int *pinv_ranks = flash_libero::shared_at<int>(shared, flash_libero::OscLayout::pinv_ranks);
  int *pinv_ok = flash_libero::shared_at<int>(shared, flash_libero::OscLayout::pinv_ok);
  {
    CF_OSC_PROFILE_SCOPE(Load);
    if (threadIdx.x == 0) {
      admitted = in.active[w] != 0 && in.error[w] == 0;
      info = 0;
    }
    __syncthreads();
    if (!admitted)
      return false;
    if (threadIdx.x < 32) {
      const int width = blockDim.x < 32 ? blockDim.x : 32;
      for (int pair = threadIdx.x; pair < 25; pair += width) {
        const int z = pair * 2;
        if (z + 1 < 49) {
          const real2_t v = libero_vec::load_pair(mass + w * 49 + z);
          libero_vec::store_pair(m + z, v.x, v.y);
          libero_vec::store_pair(lu + z, v.x, v.y);
          mi[z] = z / 7 == z % 7 ? 1 : 0;
          mi[z + 1] = (z + 1) / 7 == (z + 1) % 7 ? 1 : 0;
          if (persist_m)
            libero_vec::store_pair(persist_m + w * 49 + z, v.x, v.y);
        } else {
          const real_t v = mass[w * 49 + z];
          m[z] = lu[z] = v;
          mi[z] = z / 7 == z % 7 ? 1 : 0;
          if (persist_m)
            persist_m[w * 49 + z] = v;
        }
      }
      for (int pair = threadIdx.x; pair < 21; pair += width) {
        const int z = pair * 2;
        const real2_t v = libero_vec::load_pair(jacobian + w * 42 + z);
        libero_vec::store_pair(j + z, v.x, v.y);
        jt[(z % 7) * 6 + z / 7] = v.x;
        jt[((z + 1) % 7) * 6 + (z + 1) / 7] = v.y;
        if (persist_j)
          libero_vec::store_pair(persist_j + w * 42 + z, v.x, v.y);
      }
    }
    __syncthreads();
    if (!admitted)
      return false;
  } // input admission/load
  {
    CF_OSC_PROFILE_SCOPE(CacheLookup);
    if (threadIdx.x < 32) {
      bool mismatch = !(in.derived_valid && in.derived_valid[w]);
      const int width = blockDim.x < 32 ? blockDim.x : 32;
      const unsigned participants = width == 32 ? 0xffffffffu : ((1u << width) - 1u);
      if (!mismatch) {
        for (int pair = threadIdx.x; pair < 46; pair += width) {
          const int z = pair * 2;
          const real2_t key = z + 1 < 91 ? libero_vec::load_pair(in.derived_key + w * 91 + z)
                                         : make_real2(in.derived_key[w * 91 + z], 0);
          const real_t x = z < 49 ? m[z] : j[z - 49];
          mismatch = mismatch || (real_bits(x) != real_bits(key.x));
          if (z + 1 < 91) {
            const real_t y = z + 1 < 49 ? m[z + 1] : j[z + 1 - 49];
            mismatch = mismatch || (real_bits(y) != real_bits(key.y));
          }
        }
      }
      const bool any_mismatch = __any_sync(participants, mismatch);
      if (threadIdx.x == 0)
        reuse = !any_mismatch;
    }
    __syncthreads();
  } // cache lookup
  CF_OSC_PROFILE_META(0, reuse);
  {
    CF_OSC_PROFILE_SCOPE(CacheRestore);
    if (reuse) {
      const real_t *d = in.derived_data + w * 206;
      libero_vec::copy_block(d, mi, 49);
      libero_vec::copy_block(d + 157, project, 49);
      libero_vec::copy_block(d + 49, a, 36);
      libero_vec::copy_block(d + 103, lf, 36);
      for (int z = threadIdx.x; z < 36; z += blockDim.x) {
        const int row = z / 6, col = z % 6, k = row * 3 + col;
        const bool used = row < 3 && col < 3;
        ap[z] = used ? d[85 + k] : REAL_C(0.0);
        ar[z] = used ? d[94 + k] : REAL_C(0.0);
        lp[z] = used ? d[139 + k] : REAL_C(0.0);
        lr[z] = used ? d[148 + k] : REAL_C(0.0);
      }
      for (int z = threadIdx.x; z < 9; z += blockDim.x) {
        lp3[z] = d[139 + z];
        lr3[z] = d[148 + z];
      }
      if (threadIdx.x < 3)
        in.ranks[w * 3 + threadIdx.x] = in.derived_ranks[w * 3 + threadIdx.x];
      if (threadIdx.x == 0 && in.cache_hits)
        in.cache_hits[w]++;
    }
    __syncthreads();
  } // cache restoration
  if (!reuse) {
    {
      CF_OSC_PROFILE_SCOPE(MassInverse);
      osc_inverse7(lu, mi, piv, &info);
      if (threadIdx.x == 0 && info != 0) {
        in.error[w] = 2;
        admitted = 0;
      }
      __syncthreads();
      if (!admitted)
        return false;
      for (int z = threadIdx.x; z < 49; z += blockDim.x)
        in.mass_inv[w * 49 + z] = mi[z];
    } // mass inverse
    {
      CF_OSC_PROFILE_SCOPE(OperationalMatrix);
      // X = M^-1 J^T feeds both the task metric and the nullspace map.
      // Reassociation removes J M^-1; no symmetry assumption is required.
      osc_mm_7_6_7(mi, jt, mjt);
      osc_mm_6_6_7(j, mjt, a);
      for (int z = threadIdx.x; z < 36; z += blockDim.x) {
        const int row = z / 6, col = z % 6;
        ap[z] = row < 3 && col < 3 ? a[z] : REAL_C(0.0);
        ar[z] = row < 3 && col < 3 ? a[(row + 3) * 6 + col + 3] : REAL_C(0.0);
      }
      __syncthreads();
    } // operational matrices
    {
      CF_OSC_PROFILE_SCOPE(Pseudoinverse);
      pinv_three_block(a, ap, ar, lf, lp, lr, pinv_ranks, pinv_ok, shared);
      __syncthreads();
      if (threadIdx.x < 3)
        in.ranks[w * 3 + threadIdx.x] = pinv_ranks[threadIdx.x];
      if (threadIdx.x == 0 && (!pinv_ok[0] || !pinv_ok[1] || !pinv_ok[2])) {
        in.error[w] = 4;
        admitted = 0;
      }
      __syncthreads();
      if (!admitted)
        return false;
    } // pseudoinverse
    {
      CF_OSC_PROFILE_SCOPE(Projector);
      for (int z = threadIdx.x; z < 36; z += blockDim.x) {
        in.A_full[w * 36 + z] = a[z];
        in.lambda_full[w * 36 + z] = lf[z];
      }
      for (int z = threadIdx.x; z < 9; z += blockDim.x) {
        const int s = (z / 3) * 6 + z % 3;
        in.A_pos[w * 9 + z] = ap[s];
        in.A_ori[w * 9 + z] = ar[s];
        in.lambda_pos[w * 9 + z] = lp[s];
        in.lambda_ori[w * 9 + z] = lr[s];
        lp3[z] = lp[s];
        lr3[z] = lr[s];
      }
      osc_mm_7_6_6(mjt, lf, jbar);
      osc_mm_7_7_6(jbar, j, project);
      for (int z = threadIdx.x; z < 49; z += blockDim.x)
        project[z] = (z / 7 == z % 7 ? REAL_C(1.0) : REAL_C(0.0)) - project[z];
    } // nullspace projector
    {
      CF_OSC_PROFILE_SCOPE(CacheStore);
      if (in.derived_valid) {
        real_t *d = in.derived_data + w * 206;
        for (int z = threadIdx.x; z < 49; z += blockDim.x) {
          in.derived_key[w * 91 + z] = m[z];
          d[z] = mi[z];
          d[157 + z] = project[z];
        }
        for (int z = threadIdx.x; z < 42; z += blockDim.x)
          in.derived_key[w * 91 + 49 + z] = j[z];
        for (int z = threadIdx.x; z < 36; z += blockDim.x) {
          d[49 + z] = a[z];
          d[103 + z] = lf[z];
        }
        for (int z = threadIdx.x; z < 9; z += blockDim.x) {
          const int t = (z / 3) * 6 + z % 3;
          d[85 + z] = ap[t];
          d[94 + z] = ar[t];
          d[139 + z] = lp[t];
          d[148 + z] = lr[t];
        }
        if (threadIdx.x < 3)
          in.derived_ranks[w * 3 + threadIdx.x] = in.ranks[w * 3 + threadIdx.x];
      }
      __syncthreads();
      if (threadIdx.x == 0 && in.derived_valid)
        in.derived_valid[w] = 1;
    } // cache store
  }   // exact-input derived-matrix cache miss
  {
    CF_OSC_PROFILE_SCOPE(DiagnosticPublish);
    // Misses already materialized these in.diagnostics at their original points.
    // Only hits need replay; avoid extra writes on the compatible alias path.
    if (reuse) {
      for (int z = threadIdx.x; z < 49; z += blockDim.x)
        in.mass_inv[w * 49 + z] = mi[z];
      for (int z = threadIdx.x; z < 36; z += blockDim.x) {
        in.A_full[w * 36 + z] = a[z];
        in.lambda_full[w * 36 + z] = lf[z];
      }
      for (int z = threadIdx.x; z < 9; z += blockDim.x) {
        const int t = (z / 3) * 6 + z % 3;
        in.A_pos[w * 9 + z] = ap[t];
        in.A_ori[w * 9 + z] = ar[t];
        in.lambda_pos[w * 9 + z] = lp[t];
        in.lambda_ori[w * 9 + z] = lr[t];
      }
    }
    if (in.diagnostics)
      for (int z = threadIdx.x; z < 49; z += blockDim.x)
        in.nullspace_matrix[w * 49 + z] = project[z];
  } // cached in.diagnostics / projector transpose
  {
    CF_OSC_PROFILE_SCOPE(Goal);
    if (threadIdx.x == 0) {
      for (int r = 0; r < 3; r++) {
        gp[r] = in.goal_pos[w * 3 + r];
        for (int c = 0; c < 3; c++) {
          gr[r * 3 + c] = in.goal_rot[w * 9 + r * 3 + c];
          current[r * 3 + c] = (local_state ? local_state[17 + r * 3 + c] : in.ee_rot[w * 9 + r * 3 + c]);
        }
      }
      in.quat_valid[w] = 0;
      if (in.policy[w]) {
        for (int r = 0; r < 3; r++)
          gp[r] = (local_state ? local_state[14 + r] : in.ee_pos[w * 3 + r]) +
                  REAL_C(.05) * clip(in.action[w * 7 + r], -1, 1);
        real_t x = REAL_C(.5) * clip(in.action[w * 7 + 3], -1, 1), y = REAL_C(.5) * clip(in.action[w * 7 + 4], -1, 1),
               z = REAL_C(.5) * clip(in.action[w * 7 + 5], -1, 1);
        if (x != 0 || y != 0 || z != 0) {
          rotation_delta(x, y, z, delta, in.quat_input + w * 4, in.quat_cast + w * 4);
          in.quat_valid[w] = 1;
          for (int i = 0; i < 9; i++)
            in.quat_result[w * 9 + i] = delta[i];
        }
      }
    }
    __syncthreads();
    if (in.quat_valid[w])
      osc_mm_3_3_3(delta, current, gr);
    if (threadIdx.x < 3) {
      const int r = threadIdx.x, next = (r + 1) % 3, last = (r + 2) % 3;
      real_t er = REAL_C(0.0);
      for (int c = 0; c < 3; c++)
        er += REAL_C(.5) * (current[next * 3 + c] * gr[last * 3 + c] - current[last * 3 + c] * gr[next * 3 + c]);
      in.position_error[w * 3 + r] = gp[r] - (local_state ? local_state[14 + r] : in.ee_pos[w * 3 + r]);
      in.orientation_error[w * 3 + r] = er;
      desired[r] = in.gains[w * 12 + r] * in.position_error[w * 3 + r] -
                   in.gains[w * 12 + r + 6] * (local_state ? local_state[26 + r] : in.ee_velocity[w * 6 + r]);
      desired[r + 3] =
          in.gains[w * 12 + r + 3] * er -
          in.gains[w * 12 + r + 9] * (local_state ? local_state[26 + r + 3] : in.ee_velocity[w * 6 + r + 3]);
    }
    if (threadIdx.x < 7) {
      const int c = threadIdx.x;
      pd[c] = 10 * (in.q0[w * 7 + c] - (local_state ? local_state[c] : in.q[w * 7 + c])) -
              REAL_C(6.324555320336759) * (local_state ? local_state[7 + c] : in.qvel[w * 7 + c]);
    }
    __syncthreads();
  } // goal and PD inputs
  {
    CF_OSC_PROFILE_SCOPE(Torque);
    // Phase 1: independent task wrench (lanes 0..5) and posture force
    // (lanes 6..12). Phase 2 consumes their register values through warp shuffles.
    // There is no inter-warp consumer until the existing final publication
    // barrier, so these five tiny GEMVs need no block-wide synchronization.
    real_t task_torque = REAL_C(0.0), posture_torque = REAL_C(0.0);
    if (threadIdx.x < 32) {
      const int lane = threadIdx.x;
      real_t operand = REAL_C(0.0);
      if (lane < 6) {
        if constexpr (SubstepShape::osc_uncouple) {
          const int row = lane % 3, offset = lane < 3 ? 0 : 3;
          const real_t *lambda = lane < 3 ? lp3 : lr3;
          for (int k = 0; k < 3; ++k)
            operand += lambda[row * 3 + k] * desired[offset + k];
        } else {
          for (int k = 0; k < 6; ++k)
            operand += lf[lane * 6 + k] * desired[k];
        }
      } else if (lane < 13) {
        const int row = lane - 6;
        for (int k = 0; k < 7; ++k)
          operand += m[row * 7 + k] * pd[k];
      }
      // Every lane in the full warp executes every shuffle, including padding
      // lanes. Only the seven torque lanes multiply the broadcast operands.
      for (int k = 0; k < 6; ++k) {
        const real_t value = __shfl_sync(0xffffffffu, operand, k);
        if (lane < 7)
          task_torque += jt[lane * 6 + k] * value;
      }
      for (int k = 0; k < 7; ++k) {
        const real_t value = __shfl_sync(0xffffffffu, operand, k + 6);
        if (lane < 7)
          posture_torque += project[k * 7 + lane] * value;
      }
    }
    if (threadIdx.x < 3)
      in.goal_pos[w * 3 + threadIdx.x] = gp[threadIdx.x];
    if (threadIdx.x < 9)
      in.goal_rot[w * 9 + threadIdx.x] = gr[threadIdx.x];
    if (threadIdx.x < 7) {
      const int i = threadIdx.x;
      in.nullspace_torque[w * 7 + i] = posture_torque;
      const real_t t = task_torque + in.bias[w * 7 + i] + posture_torque;
      in.raw_tau[w * 7 + i] = t;
      in.clipped[w * 7 + i] = clip(t, in.range[w * 18 + 2 * i], in.range[w * 18 + 2 * i + 1]);
      in.ctrl[w * 9 + i] = in.clipped[w * 7 + i];
    }
    if (threadIdx.x == 0) {
      real_t a6 = in.action[w * 7 + 6], sgn = isnan(a6) ? a6 : (a6 > 0 ? REAL_C(1.) : (a6 < 0 ? -REAL_C(1.) : REAL_C(0.)));
      for (int i = 0; i < 2; i++) {
        real_t value = clip(in.gripper[w * 2 + i] + REAL_C(.01) * (i == 0 ? -REAL_C(1.) : REAL_C(1.)) * sgn, -1, 1);
        in.gripper[w * 2 + i] = value;
        real_t lo = in.range[w * 18 + 2 * (i + 7)], hi = in.range[w * 18 + 2 * (i + 7) + 1];
        in.ctrl[w * 9 + i + 7] = (hi + lo) * REAL_C(.5) + (hi - lo) * REAL_C(.5) * value;
      }
      in.needs[w] = 1;
    }
    __syncthreads();
  } // torque
  return true;
}
static __device__ inline bool control_fast(const flash_libero::ControllerSolveData &in, int w,
                                           unsigned char *shared) {
  int &live = *flash_libero::shared_at<int>(shared, flash_libero::OscLayout::live);
  int &copy_mode = *flash_libero::shared_at<int>(shared, flash_libero::OscLayout::copy_mode);
  real_t *local_state = flash_libero::shared_at<real_t>(shared, flash_libero::OscLayout::local_state);
  {
    CF_OSC_PROFILE_SCOPE(State);
    if (threadIdx.x == 0) {
      live = in.needs[w] != 0;
      copy_mode = 0;
      if (in.active[w] && !in.error[w]) {
        if (live) {
          // M/J materialization is paired with their shared operand load
          // inside control. Each physical matrix element is read once.
          copy_mode = 2;
        } else if (!in.cache_valid[w])
          in.error[w] = 1;
        else
          copy_mode = 3;
      }
    }
    __syncthreads();
    if (copy_mode != 0 && threadIdx.x < 32) {
      const int z = threadIdx.x;
      const int field = z < 7 ? 0 : (z < 14 ? 1 : (z < 17 ? 2 : (z < 26 ? 3 : 4)));
      const int sizes[5] = {7, 7, 3, 9, 6}, offsets[5] = {0, 7, 14, 17, 26};
      const real_t *sources[5] = {in.pq, in.pv, in.pp, in.pr, in.pe};
      real_t *targets[5] = {in.q, in.qvel, in.ee_pos, in.ee_rot, in.ee_velocity};
      const int at = w * sizes[field] + z - offsets[field];
      const real_t value = copy_mode == 2 ? sources[field][at] : targets[field][at];
      local_state[z] = value;
      if (copy_mode == 2)
        targets[field][at] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0 && copy_mode == 2) {
      in.cache_valid[w] = 1;
      in.needs[w] = 0;
    }
  } // live/cached state selection
  return control(in, w, shared, live, local_state);
}
} // namespace osc_cuda

} // namespace physics_controller

namespace physics_controller {
// Collective: every lane enters. The arena overlays completed forward/Newton
// scratch and stays live only until this control invocation returns.
__device__ __noinline__ bool solve(const flash_libero::ControllerSolveData &in, int world,
                                   unsigned char *shared) {
  return osc_cuda::control_fast(in, world, shared);
}
} // namespace physics_controller

#if defined(FLASH_OSC_PROFILE)
namespace physics_controller::profile {
__device__ Configuration configuration{};
}
extern "C" int flash_libero_osc_profile_configure(void *pointer, int capacity) {
  if (pointer && capacity <= 0)
    return int(cudaErrorInvalidValue);
  const physics_controller::profile::Configuration config{static_cast<uint64_t *>(pointer), capacity};
  return int(cudaMemcpyToSymbol(physics_controller::profile::configuration, &config, sizeof(config)));
}
extern "C" int flash_libero_osc_profile_header_cells() { return physics_controller::profile::header_cells; }
extern "C" int flash_libero_osc_profile_record_cells() { return physics_controller::profile::record_cells; }
extern "C" int flash_libero_osc_profile_phase_count() {
  return int(physics_controller::profile::Phase::Count);
}
extern "C" const char *flash_libero_osc_profile_phase_name(int phase) {
  static const char *names[] = {"gather",
                                "seed",
                                "solve_envelope",
                                "solve",
                                "commit",
                                "state",
                                "load",
                                "cache_lookup",
                                "cache_restore",
                                "mass_inverse",
                                "operational_matrix",
                                "pseudoinverse",
                                "projector",
                                "cache_store",
                                "diagnostic_publish",
                                "goal",
                                "torque",
                                "total"};
  static_assert(sizeof(names) / sizeof(names[0]) == int(physics_controller::profile::Phase::Count));
  return phase >= 0 && phase < int(physics_controller::profile::Phase::Count) ? names[phase] : nullptr;
}
#endif
