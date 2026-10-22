#pragma once
// Shared native Euler numerical bodies; included by the world-local entries.
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

// Cooperative CPU-order packed factor for a nine-DOF component.
// All lanes of warp 0 participate. Other warps may factor disjoint components.
// Existing entry copies qm -> ld and publishes it before calling this body.
template<bool Checked>
__device__ __forceinline__ bool factor_nine_warp(real_t *ld, real_t *inv,
    real_t *sqrt_inv, int *pivot, int &count, int &smallest,
    int begin, int end, int nv, int nm, const int *madr, const int *parent) {
  const int lane = threadIdx.x & 31;
  bool bad = false;
  for (int k = end - 1; k >= begin; --k) {
    const int kk = madr[k];
    if (lane == 0 && ld[kk] < REAL_C(1.0e-15)) {
      ld[kk] = REAL_C(1.0e-15);
      pivot[k] = 1;
      ++count;
      smallest = k < smallest ? k : smallest;
    }
    __syncwarp();
    int ancestor = parent[k];
    for (int step = 0; step < lane && ancestor >= begin; ++step)
      ancestor = parent[ancestor];
    real_t multiplier = REAL_C(0.0);
    const int ki = kk + 1 + lane;
    if (ancestor >= begin) {
      multiplier = ld[ki] / ld[kk];
      if constexpr (Checked) bad |= !isfinite(multiplier);
      const int length = (ancestor < nv-1 ? madr[ancestor+1] : nm) - madr[ancestor];
      for (int a = 0; a < length; ++a) {
        const int dest = madr[ancestor] + a;
        ld[dest] += ld[ki+a] * (-multiplier);
        if constexpr (Checked) bad |= !isfinite(ld[dest]);
      }
    }
    // Every ancestor row reads the UNNORMALIZED pivot row. Delay writes to
    // its coefficients until the last such read; this barrier is essential.
    __syncwarp();
    if (ancestor >= begin) ld[ki] = multiplier;
    __syncwarp();
  }
  if (lane < end - begin) {
    const int j = begin + lane;
    const real_t diagonal = ld[madr[j]];
    inv[j] = REAL_C(1.0) / diagonal;
    if (sqrt_inv) sqrt_inv[j] = REAL_C(1.0) / sqrt(diagonal);
    if constexpr (Checked)
      bad |= !isfinite(inv[j]) || (sqrt_inv && !isfinite(sqrt_inv[j]));
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
  // One warp owns the nine-DOF robot block; other threads retain scalar
  // ownership of the small independent components. No task-ID selector.
  const bool cooperative_nine = FLASH_REAL_BITS == 64 && !reciprocal_pivot && blockDim.x >= 64;
  if (cooperative_nine && threadIdx.x < 32) {
    // Walk the root boundaries once. Repeated forest_range(root) scans made
    // this warp do quadratic metadata work on models with many free bodies.
    for (int begin = 0; begin < nv;) {
      int end = begin + 1;
      while (end < nv && parent[end] != -1) ++end;
      if (end - begin == 9)
        bad |= !factor_nine_warp<Checked>(ld, inv, sqrt_inv, pivot, count,
            smallest, begin, end, nv, nm, madr, parent);
      begin = end;
    }
  }
  const int first = cooperative_nine ? int(threadIdx.x)-32 : int(threadIdx.x);
  const int stride = cooperative_nine ? int(blockDim.x)-32 : int(blockDim.x);
  for (int root = first; root >= 0 && root < plan.roots; root += stride) {
    int begin, end;
    forest_range(plan, root, nv, parent, begin, end);
    if (cooperative_nine && end - begin == 9) continue;
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
#if FLASH_REAL_BITS == 64
    // CPU mju_axisAngle2Quat uses library sin/cos; preserve its zero-angle
    // branch above and share range reduction for the two FP64 outputs.
    sincos(half, &s, &c);
#else
    s = fabs(half) <= 4 ? trig(half, true) : sin(half);
    c = fabs(half) <= 4 ? trig(half, false) : cos(half);
#endif
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
