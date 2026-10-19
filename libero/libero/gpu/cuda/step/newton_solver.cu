// FP64 Newton numerical implementation, independently compiled by CMake.
// No source generation, header-body inclusion, or cross-TU forced inlining.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include "newton_solver.cuh"
#include "newton_workspace.cuh"
#include "newton_coefficient_layout.cuh"
#include "newton_profile.cuh"
#include "newton_gemv.cuh"
#include "newton_atomic.cuh"
#include "newton_transpose.cuh"
#include "newton_jacobian_cache.cuh"
#include "newton_reduction_policy.cuh"
#include "substep_shared.cuh"
#include "numerics/detail/newton_mathdx.cuh"
#include "numerics/detail/newton_policy.cuh"
#include "../common/warp.cuh"
#include "../common/vector.cuh"

using flash_libero::SubstepShape;

#ifndef FLASH_NEWTON_ATOMIC_REDUCTIONS
#define FLASH_NEWTON_ATOMIC_REDUCTIONS 1
#endif


namespace physics_newton {
using flash_libero::NewtonLayout;
using flash_libero::shared_at;
static __device__ __forceinline__ NewtonControl &newton_control(unsigned char *shared) {
  return *shared_at<NewtonControl>(shared, NewtonLayout::control);
}
} // namespace physics_newton


// Newton and its forward Post stage share only a completed workspace.
// Their public entries remain separate collective device calls.

namespace physics_newton {

static __device__ inline double *coefficient_plane(unsigned char *base, std::size_t offset) {
  return reinterpret_cast<double *>(base + offset);
}
static __device__ inline double *atomic_workspace(const NewtonWorldData &p) {
  return reinterpret_cast<double *>(p.precomputed + NewtonDeviceWorkspaceLayout::atomic);
}
static __device__ inline jacobian_cache::View jacobian_view(const NewtonWorldData &p) {
  return {p.precomputed + NewtonDeviceWorkspaceLayout::jacobian, SubstepShape::rows, SubstepShape::nv};
}

// D/R/friction/mu are position-prefix outputs (CPU mj_fwdPosition), not
// qacc/alpha outputs. A successful prefix rebuild advances position_generation.
// All lanes capture refresh before the entry barrier, so the leader can update
// the header here without racing a slower warp's refresh decision. The existing
// entry publication barrier publishes both global planes and the header.
static __device__ void prepare_row_coefficients(unsigned char *shared, const NewtonWorldData &p,
                                                bool refresh, uint64_t generation) {
  auto &ctl = newton_control(shared);
  if (!ctl.invariant_valid() || !refresh)
    return;
  CF_NEWTON_PROFILE_SCOPE(Coefficients);
  double *sqrt_d = coefficient_plane(p.precomputed, NewtonCoefficientLayout::sqrt_d);
  double *friction_limit = coefficient_plane(p.precomputed, NewtonCoefficientLayout::friction_limit);
  double *cone_dm = coefficient_plane(p.precomputed, NewtonCoefficientLayout::cone_dm);
  const int n = ctl.count();
  const int ne = ctl.ne(), end_friction = ne + ctl.nf();
  for (int r = threadIdx.x; r < n; r += blockDim.x) {
    sqrt_d[r] = r >= ne ? sqrt(p.D[r]) : 0.0;
    friction_limit[r] = r >= ne && r < end_friction ? p.R[r] * p.loss[r] : 0.0;
    double dm = 0.0;
    if (r >= end_friction && p.constraint_group_heads[r] == r && p.efc_type[r] == 7) {
      const double mu = p.contact_mu[p.efc_id[r]];
      dm = p.D[r] / (mu * mu * (1 + mu * mu));
    }
    cone_dm[r] = dm;
  }
  if (threadIdx.x == 0)
    *reinterpret_cast<uint64_t *>(p.precomputed + NewtonCoefficientLayout::generation) = generation;
}

static __device__ inline double row_friction_limit(const NewtonWorldData &p, int row) {
  return coefficient_plane(p.precomputed, NewtonCoefficientLayout::friction_limit)[row];
}
static __device__ inline double row_sqrt_d(const NewtonWorldData &p, int row) {
  return coefficient_plane(p.precomputed, NewtonCoefficientLayout::sqrt_d)[row];
}

// MuJoCo 2.3.7 source-order Newton line search, fixed dense-J model boundary.
// Source: engine_solver.c:834-1267, engine_support.c:826-873,
// engine_util_blas.c:631-694. Apache 2.0, DeepMind Technologies Limited.

struct P {
  double alpha, cost, d0, d1;
};

// Mirrors engine_util_blas.c mju_dot, including four independent lanes and
// its 1/2/3-element tail grouping. The CMake FP64 policy permits FMA contraction.
static __device__ double dot_strided(const double *a, int astride, const double *b, int bstride, int n) {
  double res = 0.0;
  int i = 0;
  int n4 = n - 4;
  double r0 = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
  if (n4 >= 0) {
    r0 = a[0] * b[0];
    r1 = a[astride] * b[bstride];
    r2 = a[2 * astride] * b[2 * bstride];
    r3 = a[3 * astride] * b[3 * bstride];
    i = 4;
  }
  for (; i <= n4; i += 4) {
    r0 += a[i * astride] * b[i * bstride];
    r1 += a[(i + 1) * astride] * b[(i + 1) * bstride];
    r2 += a[(i + 2) * astride] * b[(i + 2) * bstride];
    r3 += a[(i + 3) * astride] * b[(i + 3) * bstride];
  }
  res = (r0 + r2) + (r1 + r3);
  int tail = n - i;
  if (tail == 3)
    res += a[i * astride] * b[i * bstride] + a[(i + 1) * astride] * b[(i + 1) * bstride] +
           a[(i + 2) * astride] * b[(i + 2) * bstride];
  else if (tail == 2)
    res += a[i * astride] * b[i * bstride] + a[(i + 1) * astride] * b[(i + 1) * bstride];
  else if (tail == 1)
    res += a[i * astride] * b[i * bstride];
  return res;
}

static __device__ void add3(double *total, const double *q) {
  total[0] += q[0];
  total[1] += q[1];
  total[2] += q[2];
}

static __device__ P eval_point(unsigned char *shared, const NewtonModelData &model,
                               const NewtonWorldData &world, const double *smooth_force, int w, int rows,
                               int nefc, int ne, int nf, double alpha, int tag, int *niter) {
  auto &ctl = newton_control(shared);
  const int qbase = w * rows * 3, base = w * rows;
  double total[3] = {(&ctl.quad0())[w], world.quadGauss[w * 3 + 1], world.quadGauss[w * 3 + 2]};
  double cost = 0.0, deriv0 = 0.0, deriv1 = 0.0;
  for (int i = 0; i < ne; i++)
    add3(total, world.quad + qbase + 3 * i);
  for (int i = ne; i < ne + nf; i++) {
    double start = world.jar[base + i], dir = world.Jv[base + i];
    double x = start + alpha * dir;
    double f = world.loss[base + i], Rf = row_friction_limit(world, base + i);
    if (-Rf < x && x < Rf)
      add3(total, world.quad + qbase + 3 * i);
    else if (x <= -Rf) {
      double qf[3] = {f * (-0.5 * Rf - start), -f * dir, 0.0};
      add3(total, qf);
    } else {
      double qf[3] = {f * (-0.5 * Rf + start), f * dir, 0.0};
      add3(total, qf);
    }
  }
  for (int i = ne + nf; i < nefc; i++) {
    if (world.efc_type[base + i] == 7) { // mjCNSTR_CONTACT_ELLIPTIC
      const double *q = world.quad + qbase + 3 * i;
      int dim = world.cone_dim[base + i];
      double mu = world.cone_mu[base + i];
      double U0 = q[3], V0 = q[4], UU = q[5], UV = q[6], VV = q[7], Dm = q[8];
      double N = U0 + alpha * V0;
      double Tsqr = UU + alpha * (2.0 * UV + alpha * VV);
      if (Tsqr <= 0.0) {
        if (N < 0.0)
          add3(total, q);
      } else {
        double T = __dsqrt_rn(Tsqr);
        if (N >= mu * T) {
          // top zone
        } else if (mu * N + T <= 0.0) {
          add3(total, q);
        } else {
          double N1 = V0;
          double T1 = __ddiv_rn(UV + alpha * VV, T);
          double T2 = __ddiv_rn(VV, T) - __ddiv_rn((UV + alpha * VV) * T1, T * T);
          cost += 0.5 * Dm * (N - mu * T) * (N - mu * T);
          deriv0 += Dm * (N - mu * T) * (N1 - mu * T1);
          deriv1 += Dm * ((N1 - mu * T1) * (N1 - mu * T1) + (N - mu * T) * (-mu * T2));
        }
      }
      i += dim - 1;
    } else {
      double x = world.jar[base + i] + alpha * world.Jv[base + i];
      if (x < 0.0)
        add3(total, world.quad + qbase + 3 * i);
    }
  }
  cost += alpha * alpha * total[2] + alpha * total[1] + total[0];
  deriv0 += 2.0 * alpha * total[2] + total[1];
  deriv1 += 2.0 * total[2];
  if (deriv1 <= 0.0)
    deriv1 = 1.0e-15;
  if (!isfinite(alpha) || !isfinite(cost) || !isfinite(deriv0) || !isfinite(deriv1))
    (&ctl.s07_status())[w] = 5;
  P p = {alpha, cost, deriv0, deriv1};
  int idx = *niter;
#if LIBERO_NEWTON_TRACE
  world.eval_trace[(w * 52 + idx) * 4] = alpha;
  world.eval_trace[(w * 52 + idx) * 4 + 1] = cost;
  world.eval_trace[(w * 52 + idx) * 4 + 2] = deriv0;
  world.eval_trace[(w * 52 + idx) * 4 + 3] = deriv1;
  world.eval_tag[w * 52 + idx] = tag;
#endif
  *niter = idx + 1;
  return p;
}

static __device__ int update_bracket(unsigned char *shared, const NewtonModelData &model,
                                     const NewtonWorldData &world, const double *smooth_force, int w,
                                     int rows, int nefc, int ne, int nf, P *p, const P *candidates, P *next,
                                     int *niter, int tag) {
  int flag = 0;
  for (int i = 0; i < 3; i++) {
    if (p->d0 < 0.0 && candidates[i].d0 < 0.0 && p->d0 < candidates[i].d0) {
      *p = candidates[i];
      flag = 1;
    } else if (p->d0 > 0.0 && candidates[i].d0 > 0.0 && p->d0 > candidates[i].d0) {
      *p = candidates[i];
      flag = 2;
    }
  }
  if (flag) {
    double alpha = p->alpha - __ddiv_rn(p->d0, p->d1);
    *next = eval_point(shared, model, world, smooth_force, w, rows, nefc, ne, nf, alpha, tag, niter);
  }
  return flag;
}

static __device__ void finish(unsigned char *shared, const NewtonModelData &model,
                              const NewtonWorldData &world, const double *smooth_force, int w, double alpha,
                              int niter, int result, double slope) {
  auto &ctl = newton_control(shared);
  (&ctl.alpha())[w] = alpha;
  (&ctl.LSiter())[w] = niter;
  world.LSresult[w] = result;
  (&ctl.slope())[w] = slope;
}

static __device__ void search_kernel(int w, unsigned char *shared, const NewtonModelData &model,
                                     const NewtonWorldData &world, const double *smooth_force, int nworld,
                                     int nv, int nM, int rows, int tracecap) {
  auto &ctl = newton_control(shared);
  double *shared_search = shared_at<double>(shared, NewtonLayout::search);
  double *shared_mass_search = shared_at<double>(shared, NewtonLayout::mass_search);
  double *shared_mass_acceleration = shared_at<double>(shared, NewtonLayout::mass_acceleration);

  if (w >= nworld)
    return;
  (&ctl.s07_status())[w] = 0;
  if ((&ctl.active())[w] == 0)
    return;
  if ((&ctl.active())[w] != 1) {
    (&ctl.s07_status())[w] = 1;
    return;
  }
  int nefc = (&ctl.count())[w], ne = (&ctl.ne())[w], nf = (&ctl.nf())[w];
  if (nefc < 0 || nefc > rows || ne < 0 || nf < 0 || ne > nefc || nf > nefc - ne) {
    (&ctl.s07_status())[w] = 2;
    return;
  }
  if (tracecap < 52) {
    (&ctl.s07_status())[w] = 3;
    return;
  }
  int base = w * rows;
  for (int i = 0; i < nefc; i++) {
    if (world.efc_type[base + i] == 7) {
      int dim = world.cone_dim[base + i];
      if (i < ne + nf || (dim != 3 && dim != 4 && dim != 6) || i + dim > nefc ||
          world.cone_mu[base + i] <= 0.0) {
        (&ctl.s07_status())[w] = 4;
        return;
      }
      i += dim - 1;
    }
  }
  const double *v = shared_search + w * nv;
  double snorm = __dsqrt_rn(dot_strided(v, 1, v, 1, nv));
  if (!isfinite(snorm)) {
    (&ctl.s07_status())[w] = 5;
    return;
  }
  // The CPU context carries quadGauss[0] on entry; the device ABI stores that
  // frozen input separately, so materialize its unchanged value in the output.
  world.quadGauss[w * 3] = (&ctl.quad0())[w];
  if (snorm < 1.0e-15) {
    finish(shared, model, world, smooth_force, w, 0.0, 0, 1, 1.0);
    return;
  }
  double mean = model.model_scalars[1];
  double gtol = model.model_scalars[0] * 0.01 * snorm * mean * (nv > 1 ? nv : 1);
  double slopescl = __ddiv_rn(1.0, snorm * mean * (nv > 1 ? nv : 1));
  if (!isfinite(gtol) || !isfinite(slopescl)) {
    (&ctl.s07_status())[w] = 5;
    return;
  }
  double *Mv = shared_mass_search + w * nv;
  const double *M = world.qM + w * nM;
  for (int i = 0; i < nv; i++)
    Mv[i] = 0.0;
  for (int i = 0; i < nv; i++) {
    int adr = model.madr[i];
    if (model.simplenum[i]) {
      Mv[i] = M[adr] * v[i];
    } else {
      Mv[i] += M[adr] * v[i];
      int j = model.parentid[i];
      adr++;
      while (j >= 0) {
        Mv[i] += M[adr] * v[j];
        Mv[j] += M[adr] * v[i];
        j = model.parentid[j];
        adr++;
      }
    }
  }
  for (int i = 0; i < nv; i++)
    if (!isfinite(Mv[i])) {
      (&ctl.s07_status())[w] = 5;
      return;
    }
  double *Jv = world.Jv + base;
  const double *J = world.J + w * rows * nv;
  for (int i = 0; i < nefc; i++)
    Jv[i] = dot_strided(J + i * nv, 1, v, 1, nv);
  for (int i = 0; i < nefc; i++)
    if (!isfinite(Jv[i])) {
      (&ctl.s07_status())[w] = 5;
      return;
    }
  double *qG = world.quadGauss + w * 3;
  qG[1] = dot_strided(v, 1, shared_mass_acceleration + w * nv, 1, nv) -
          dot_strided(v, 1, smooth_force + w * nv, 1, nv);
  qG[2] = 0.5 * dot_strided(v, 1, Mv, 1, nv);
  if (!isfinite(qG[1]) || !isfinite(qG[2])) {
    (&ctl.s07_status())[w] = 5;
    return;
  }
  double *qbase = world.quad + w * rows * 3;
  for (int i = 0; i < nefc; i++) {
    double *q = qbase + 3 * i;
    bool elliptic = world.efc_type[base + i] == 7;
    double DJ0 = world.D[base + i] * world.jar[base + i];
    q[0] = world.jar[base + i] * DJ0;
    q[1] = Jv[i] * DJ0;
    q[2] = Jv[i] * world.D[base + i] * Jv[i];
    if (elliptic) {
      int dim = world.cone_dim[base + i];
      double mu = world.cone_mu[base + i];
      for (int j = 1; j < dim; j++) {
        double DJj = world.D[base + i + j] * world.jar[base + i + j];
        q[0] += world.jar[base + i + j] * DJj;
        q[1] += Jv[i + j] * DJj;
        q[2] += Jv[i + j] * world.D[base + i + j] * Jv[i + j];
      }
      double U0 = world.jar[base + i] * mu, V0 = Jv[i] * mu;
      double UU = 0.0, UV = 0.0, VV = 0.0;
      for (int j = 1; j < dim; j++) {
        double U = world.jar[base + i + j] * world.cone_friction[(base + i) * 5 + j - 1];
        double V = Jv[i + j] * world.cone_friction[(base + i) * 5 + j - 1];
        UU += U * U;
        UV += U * V;
        VV += V * V;
      }
      q[3] = U0;
      q[4] = V0;
      q[5] = UU;
      q[6] = UV;
      q[7] = VV;
      q[8] = __ddiv_rn(world.D[base + i], mu * mu * (1.0 + mu * mu));
      i += dim - 1;
    }
    q[0] *= 0.5;
    q[2] *= 0.5;
    for (int k = 0; k < 3; k++)
      if (!isfinite(q[k])) {
        (&ctl.s07_status())[w] = 5;
        return;
      }
    if (elliptic) {
      for (int k = 3; k < 9; k++)
        if (!isfinite(q[k])) {
          (&ctl.s07_status())[w] = 5;
          return;
        }
    }
  }
  int niter = 0;
  P p0 = eval_point(shared, model, world, smooth_force, w, rows, nefc, ne, nf, 0.0, 1, &niter);
  P p1 = eval_point(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                    p0.alpha - __ddiv_rn(p0.d0, p0.d1), 2, &niter);
  if (p0.cost < p1.cost)
    p1 = p0;
  if (fabs(p1.d0) < gtol) {
    finish(shared, model, world, smooth_force, w, p1.alpha, niter, p1.alpha == 0.0 ? 2 : 0,
           fabs(p1.d0) * slopescl);
    return;
  }
  int dir = p1.d0 < 0.0 ? 1 : -1;
  int p2update = 0;
  P p2 = {};
  while (p1.d0 * dir <= -gtol && niter < 50) {
    p2 = p1;
    p2update = 1;
    p1 = eval_point(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                    p1.alpha - __ddiv_rn(p1.d0, p1.d1), 3, &niter);
    if (fabs(p1.d0) < gtol) {
      finish(shared, model, world, smooth_force, w, p1.alpha, niter, 0, fabs(p1.d0) * slopescl);
      return;
    }
  }
  if (niter >= 50) {
    finish(shared, model, world, smooth_force, w, p1.alpha, niter, 3, fabs(p1.d0) * slopescl);
    return;
  }
  if (!p2update) {
    finish(shared, model, world, smooth_force, w, p1.alpha, niter, 6, fabs(p1.d0) * slopescl);
    return;
  }
  P p2next = p1;
  P p1next = eval_point(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                        p1.alpha - __ddiv_rn(p1.d0, p1.d1), 4, &niter);
  while (niter < 50) {
    P pmid = eval_point(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                        0.5 * (p1.alpha + p2.alpha), 5, &niter);
    P candidates[3] = {p1next, p2next, pmid};
    double bestcost = 0.0;
    int bestind = -1;
    for (int i = 0; i < 3; i++) {
      if (fabs(candidates[i].d0) < gtol && (bestind == -1 || candidates[i].cost < bestcost)) {
        bestcost = candidates[i].cost;
        bestind = i;
      }
    }
    if (bestind >= 0) {
      finish(shared, model, world, smooth_force, w, candidates[bestind].alpha, niter, 0,
             fabs(candidates[bestind].d0) * slopescl);
      return;
    }
    int b1 = update_bracket(shared, model, world, smooth_force, w, rows, nefc, ne, nf, &p1, candidates,
                            &p1next, &niter, 6);
    int b2 = update_bracket(shared, model, world, smooth_force, w, rows, nefc, ne, nf, &p2, candidates,
                            &p2next, &niter, 7);
    if (!b1 && !b2) {
      finish(shared, model, world, smooth_force, w, pmid.alpha, niter, pmid.cost < p0.cost ? 0 : 7,
             fabs(pmid.d0) * slopescl);
      return;
    }
  }
  if (p1.cost <= p2.cost && p1.cost < p0.cost) {
    finish(shared, model, world, smooth_force, w, p1.alpha, niter, 4, fabs(p1.d0) * slopescl);
  } else if (p2.cost <= p1.cost && p2.cost < p0.cost) {
    finish(shared, model, world, smooth_force, w, p2.alpha, niter, 4, fabs(p2.d0) * slopescl);
  } else {
    finish(shared, model, world, smooth_force, w, 0.0, niter, 5, 1.0);
  }
}

// Project-written register collectives. Every lane in a full warp participates.
// Callers pad invalid elements with the operation's identity, never skip lanes.
// These functions do not synchronize shared or global memory.

// Nonstandard tree-packed symmetric product, not a dense BLAS operand.
// The caller admits a valid tree and storage, and owns publication/barriers.
// x/qM/topology must not alias result. Each block owns one world.
namespace libero_packed_mass {

// All block threads call. Each output has one owner, so no atomics or shared
// scratch are needed. Preserve the scalar scatter's contribution order,
// including simple-row assignment rather than addition.
static __device__ __forceinline__ void multiply_gather(const double *x, const double *qM, const int *madr,
                                                       const int *parent, const int *simple, double *result,
                                                       int nv) {
  for (int row = int(threadIdx.x); row < nv; row += int(blockDim.x)) {
    double value = 0.0;
    for (int i = 0; i < nv; ++i) {
      int adr = madr[i];
      if (simple && simple[i]) {
        if (row == i)
          value = qM[adr] * x[i];
        continue;
      }
      if (row == i)
        value += qM[adr] * x[i];
      ++adr;
      for (int j = parent[i]; j >= 0; j = parent[j], ++adr) {
        if (row == i)
          value += qM[adr] * x[j];
        if (row == j)
          value += qM[adr] * x[i];
      }
    }
    result[row] = value;
  }
}

// Experimental broadcast variant: one lane loads each packed contribution
// and all output owners in its warp consume it. Requires blockDim.x % 32 == 0
// and uniform entry. Inactive output lanes still participate in every shuffle.
// No floating-point reduction/reassociation and no implicit memory barrier.
// Keep opt-in until whole-kernel timing justifies the repeated warp traversal.
static __device__ __forceinline__ void multiply_gather_warp(const double *x, const double *qM,
                                                            const int *madr, const int *parent,
                                                            const int *simple, double *result, int nv) {
  const int lane = int(threadIdx.x) & 31;
  const int warp_base = int(threadIdx.x) & ~31;
  for (int base = warp_base; base < nv; base += int(blockDim.x)) {
    const int row = base + lane;
    double value = 0.0;
    for (int i = 0; i < nv; ++i) {
      int adr = 0, row_simple = 0;
      double diagonal = 0.0, xi = 0.0;
      if (lane == 0) {
        adr = madr[i];
        row_simple = simple && simple[i];
        diagonal = qM[adr];
        xi = x[i];
      }
      adr = __shfl_sync(0xffffffffu, adr, 0);
      row_simple = __shfl_sync(0xffffffffu, row_simple, 0);
      diagonal = __shfl_sync(0xffffffffu, diagonal, 0);
      xi = __shfl_sync(0xffffffffu, xi, 0);
      if (row_simple) {
        if (row == i)
          value = diagonal * xi;
        continue;
      }
      if (row == i)
        value += diagonal * xi;
      ++adr;
      for (int j = parent[i]; j >= 0; j = parent[j], ++adr) {
        double coefficient = 0.0, xj = 0.0;
        if (lane == 0) {
          coefficient = qM[adr];
          xj = x[j];
        }
        coefficient = __shfl_sync(0xffffffffu, coefficient, 0);
        xj = __shfl_sync(0xffffffffu, xj, 0);
        if (row == i)
          value += coefficient * xj;
        if (row == j)
          value += coefficient * xi;
      }
    }
    if (row < nv)
      result[row] = value;
  }
}

// Cold ordered CSR: offsets have nv+1 cells; each event carries a packed
// coefficient address, x address, and assignment flag (all int32). The builder
// emits events in the original scalar i/ancestor traversal order. A simple
// diagonal event assigns, including overwriting earlier descendant additions.
// No floating point reduction, no scratch, and no topology traversal at runtime.
static __device__ __forceinline__ void multiply_plan(const double *x, const double *qM, const int *offsets,
                                                     const int *coefficient, const int *source,
                                                     const int *assign, double *result, int nv) {
  for (int row = int(threadIdx.x); row < nv; row += int(blockDim.x)) {
    double value = 0.0;
    for (int e = offsets[row]; e < offsets[row + 1]; ++e) {
      if (assign[e])
        value = qM[coefficient[e]] * x[source[e]];
      else
        value += qM[coefficient[e]] * x[source[e]];
    }
    result[row] = value;
  }
}

// Generic non-simple packed storage. Newton passes its dof_simple array to
// the overload above; dropping that array there would change semantics.
static __device__ __forceinline__ void multiply_gather(const double *x, const double *qM, const int *madr,
                                                       const int *parent, double *result, int nv) {
  multiply_gather(x, qM, madr, parent, nullptr, result, nv);
}
static __device__ __forceinline__ void multiply_gather_warp(const double *x, const double *qM,
                                                            const int *madr, const int *parent,
                                                            double *result, int nv) {
  multiply_gather_warp(x, qM, madr, parent, nullptr, result, nv);
}
} // namespace libero_packed_mass

// Newton-only I/O selection. The production core chooses scalar for the two
// measured regression shapes (NV21/NV43); other entry points default to their
// established packet path. Arithmetic, capacity and commit guards stay local.

namespace libero_newton_io {
template <class T> static __device__ __forceinline__ void copy_block(const T *src, T *dst, size_t n) {
#if LIBERO_NEWTON_SCALAR_IO
  for (size_t i = threadIdx.x; i < n; i += blockDim.x)
    dst[i] = src[i];
#else
  libero_vec::copy_block(src, dst, n);
#endif
}
template <class T> static __device__ __forceinline__ void copy_serial(const T *src, T *dst, size_t n) {
#if LIBERO_NEWTON_SCALAR_IO
  for (size_t i = 0; i < n; ++i)
    dst[i] = src[i];
#else
  libero_vec::copy_serial(src, dst, n);
#endif
}
template <class T> static __device__ __forceinline__ void fill_block(T *dst, size_t n, T value) {
#if LIBERO_NEWTON_SCALAR_IO
  for (size_t i = threadIdx.x; i < n; i += blockDim.x)
    dst[i] = value;
#else
  libero_vec::fill_block(dst, n, value);
#endif
}
template <class T> static __device__ __forceinline__ void fill_serial(T *dst, size_t n, T value) {
#if LIBERO_NEWTON_SCALAR_IO
  for (size_t i = 0; i < n; ++i)
    dst[i] = value;
#else
  libero_vec::fill_serial(dst, n, value);
#endif
}
} // namespace libero_newton_io

// Source-order FP64 helpers. MuJoCo 2.3.7, Apache-2.0.
// NV and NM are cold capacity parameters, never task identifiers.

// Compile-time reference path for rank-update performance comparisons only.
// Both choices retain pivot order, zero skips and rank regularization.
static __device__ inline double dot4(const double *a, const double *b, int n) {
  double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int k = 0;
#if LIBERO_NEWTON_SCALAR_IO
  if (n >= 4) {
    s0 = a[0] * b[0];
    s1 = a[1] * b[1];
    s2 = a[2] * b[2];
    s3 = a[3] * b[3];
    k = 4;
  }
  for (; k + 4 <= n; k += 4) {
    s0 += a[k] * b[k];
    s1 += a[k + 1] * b[k + 1];
    s2 += a[k + 2] * b[k + 2];
    s3 += a[k + 3] * b[k + 3];
  }
#else
  // Pair only the loads: accumulator lanes, initialization, and tails retain
  // the source expression order. Each pair lies inside a full four-wide chunk.
  if (n >= 4) {
    const double2 a01 = libero_vec::load_pair(a), b01 = libero_vec::load_pair(b);
    const double2 a23 = libero_vec::load_pair(a + 2), b23 = libero_vec::load_pair(b + 2);
    s0 = a01.x * b01.x;
    s1 = a01.y * b01.y;
    s2 = a23.x * b23.x;
    s3 = a23.y * b23.y;
    k = 4;
  }
  for (; k + 4 <= n; k += 4) {
    const double2 a01 = libero_vec::load_pair(a + k), b01 = libero_vec::load_pair(b + k);
    const double2 a23 = libero_vec::load_pair(a + k + 2), b23 = libero_vec::load_pair(b + k + 2);
    s0 += a01.x * b01.x;
    s1 += a01.y * b01.y;
    s2 += a23.x * b23.x;
    s3 += a23.y * b23.y;
  }
#endif // LIBERO_NEWTON_SCALAR_IO
  double s = (s0 + s2) + (s1 + s3);
  if (n - k == 3)
    s += a[k] * b[k] + a[k + 1] * b[k + 1] + a[k + 2] * b[k + 2];
  else if (n - k == 2)
    s += a[k] * b[k] + a[k + 1] * b[k + 1];
  else if (n - k == 1)
    s += a[k] * b[k];
  return s;
}
static __device__ inline int mass_index(int i, int j, const int *madr, const int *parent, const int *simple) {
  int adr = madr[i];
  for (int k = i; k >= 0; k = parent[k], ++adr) {
    if (k == j)
      return adr;
    if (simple[i])
      break;
  }
  return -1;
}
static __device__ inline void mass_multiply(const double *v, const double *M, const int *madr,
                                            const int *parent, const int *simple, double *out) {
  for (int i = 0; i < SubstepShape::nv; ++i)
    out[i] = 0.0;
  for (int i = 0; i < SubstepShape::nv; ++i) {
    int adr = madr[i];
    if (simple[i])
      out[i] = M[adr] * v[i];
    else {
      out[i] += M[adr] * v[i];
      ++adr;
      for (int j = parent[i]; j >= 0; j = parent[j], ++adr) {
        out[i] += M[adr] * v[j];
        out[j] += M[adr] * v[i];
      }
    }
  }
}
static __device__ inline int chol_factor(double *H, int n, int stride) {
  int rank = n;
  for (int j = 0; j < n; ++j) {
    double d = H[j * stride + j];
    if (j)
      d -= dot4(H + j * stride, H + j * stride, j);
    if (d < 1e-15) {
      d = 1e-15;
      --rank;
    }
    H[j * stride + j] = sqrt(d);
    double inv = 1.0 / H[j * stride + j];
    for (int i = j + 1; i < n; ++i)
      H[i * stride + j] = (H[i * stride + j] - dot4(H + i * stride, H + j * stride, j)) * inv;
  }
  return rank;
}
static __device__ inline int chol_update(double *H, double *v, int plus) {
  int rank = SubstepShape::nv;
  for (int k = 0; k < SubstepShape::nv; ++k) {
    double x = v[k];
    if (x == 0.0)
      continue;
    double L = H[k * SubstepShape::nv + k], tmp = L * L;
    tmp = plus ? tmp + x * x : tmp - x * x;
    if (tmp < 1e-15) {
      tmp = 1e-15;
      --rank;
    }
    double r = sqrt(tmp), c = r / L, inv = 1.0 / c, s = x / L;
    H[k * SubstepShape::nv + k] = r;
    for (int i = k + 1; i < SubstepShape::nv; ++i)
      H[i * SubstepShape::nv + k] =
          (plus ? H[i * SubstepShape::nv + k] + s * v[i] : H[i * SubstepShape::nv + k] - s * v[i]) * inv;
    for (int i = k + 1; i < SubstepShape::nv; ++i)
      v[i] = c * v[i] - s * H[i * SubstepShape::nv + k];
  }
  return rank;
}
// All block threads enter. Only warp 0 touches the update; the final block
// barrier publishes its complete private factor/vector before another phase.
// Pivot order and each element's arithmetic match chol_update exactly.
static __device__ inline int chol_update_block(double *H, double *v, int plus) {
  CF_NEWTON_PROFILE_SCOPE(RankUpdate);
  __shared__ int result;
  if constexpr (SubstepShape::nv <= 32) {
    if (threadIdx.x == 0)
      result = chol_update(H, v, plus);
  } else if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int rank = SubstepShape::nv;
    for (int k = 0; k < SubstepShape::nv; ++k) {
      double x = lane == 0 ? v[k] : 0.0;
      x = __shfl_sync(0xffffffffu, x, 0);
      if (x == 0.0)
        continue; // uniform warp branch, including signed zero
      double c = 0.0, inv = 0.0, s = 0.0;
      if (lane == 0) {
        double L = H[k * SubstepShape::nv + k], tmp = L * L;
        tmp = plus ? tmp + x * x : tmp - x * x;
        if (tmp < 1e-15) {
          tmp = 1e-15;
          --rank;
        }
        double r = sqrt(tmp);
        c = r / L;
        inv = 1.0 / c;
        s = x / L;
        H[k * SubstepShape::nv + k] = r;
      }
      c = __shfl_sync(0xffffffffu, c, 0);
      inv = __shfl_sync(0xffffffffu, inv, 0);
      s = __shfl_sync(0xffffffffu, s, 0);
      for (int i = k + 1 + lane; i < SubstepShape::nv; i += 32) {
        double updated =
            (plus ? H[i * SubstepShape::nv + k] + s * v[i] : H[i * SubstepShape::nv + k] - s * v[i]) * inv;
        H[i * SubstepShape::nv + k] = updated;
        v[i] = c * v[i] - s * updated;
      }
      __syncwarp(0xffffffffu); // next pivot may read another lane's v[k]
    }
    if (lane == 0)
      result = rank;
  }
  __syncthreads();
  return result;
}

static __device__ inline double cone_norm(const double *jar, const double *f, int d) {
  double u1 = 0, u2 = 0, u3 = 0, u4 = 0, u5 = 0, s = 0;
  if (d > 1) {
    u1 = jar[1] * f[0];
    u2 = jar[2] * f[1];
  }
  if (d > 3)
    u3 = jar[3] * f[2];
  if (d == 6) {
    u4 = jar[4] * f[3];
    u5 = jar[5] * f[4];
    s = ((u1 * u1 + u3 * u3) + (u2 * u2 + u4 * u4)) + u5 * u5;
  } else if (d == 4)
    s = s + (u1 * u1 + u2 * u2 + u3 * u3);
  else if (d == 3)
    s = s + (u1 * u1 + u2 * u2);
  return sqrt(s);
}
static __device__ void
constraint_update(int w, int *active, int *count, int *ne, int *nf, int *ncon, int *flag, double *jar,
                  double *D, double *R, double *loss, int *typ, int *cid, double *J, double *mu, double *fr,
                  int *dim, double *force, int *state, double *H, double *cost, int *status, int rows,
                  int contacts, int nv, int *invariant_valid = nullptr, int *active_count = nullptr,
                  int *cone_count = nullptr, int *old_state = nullptr, int *changed_count = nullptr,
                  bool validate_only = false, int *group_head = nullptr, int *hessian_owner = nullptr,
                  bool topology_only = false, bool rows_checked = false, bool collect_owner = false) {
  status[w] = 0;
  if (!active[w])
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  int n = count[w], nc = ncon[w], eq = ne[w], fc = nf[w], rb = w * rows, cb = w * contacts;
  if (n < 0 || n > rows || nc < 0 || nc > contacts || (flag[w] != 0 && flag[w] != 1) || eq < 0 || eq > n ||
      fc < 0 || fc > n - eq) {
    status[w] = 2;
    return;
  }
  bool cached = invariant_valid && invariant_valid[0];
  if (!cached && !rows_checked)
    for (int i = eq + fc; i < n;) {
      int t = typ[rb + i];
      if (t == 7) {
        int c = cid[rb + i];
        if (c < 0 || c >= nc) {
          status[w] = 3;
          return;
        }
        int d = dim[cb + c];
        if ((d != 1 && d != 3 && d != 4 && d != 6) || i + d > n || mu[cb + c] <= 0 || !isfinite(mu[cb + c])) {
          status[w] = 4;
          return;
        }
        for (int j = 0; j < d; ++j)
          if (typ[rb + i + j] != 7 || cid[rb + i + j] != c) {
            status[w] = 5;
            return;
          }
        for (int j = 0; j < d - 1; ++j)
          if (fr[(cb + c) * 5 + j] <= 0 || !isfinite(fr[(cb + c) * 5 + j])) {
            status[w] = 6;
            return;
          }
        i += d;
      } else if (t >= 0 && t <= 6)
        ++i;
      else {
        status[w] = 7;
        return;
      }
    }
  if (topology_only)
    return;
  if (!rows_checked)
    for (int r = 0; r < n; ++r) {
      int z = rb + r;
      if (!isfinite(jar[z]) || (!cached && (!isfinite(D[z]) || !isfinite(R[z]) || !isfinite(loss[z])))) {
        status[w] = 8;
        return;
      }
      if (!cached)
        for (int j = 0; j < nv; ++j)
          if (!isfinite(J[z * nv + j])) {
            status[w] = 9;
            return;
          }
    }
  if (!rows_checked)
    for (int r = 0; r < n; ++r) {
      int z = rb + r;
      if (jar[z] > 10000 || jar[z] < -10000 ||
          (!cached && (D[z] > 2e16 || D[z] < -10000 || R[z] > 10000 || R[z] < -10000 || loss[z] > 10000 ||
                       loss[z] < -10000))) {
        status[w] = 10;
        return;
      }
      if (!cached)
        for (int j = 0; j < nv; ++j)
          if (J[z * nv + j] > 10000 || J[z * nv + j] < -10000) {
            status[w] = 10;
            return;
          }
    }
  if (hessian_owner && !rows_checked)
    for (int c = 0; c < nc; ++c)
      hessian_owner[cb + c] = -1;
  for (int i = eq + fc; i < n;) {
    if (typ[rb + i] != 7) {
      ++i;
      continue;
    }
    int c = cb + cid[rb + i], d = dim[c];
    double m = mu[c];
    if (!cached) {
      if (m < .0001 || m > 10000) {
        status[w] = 10;
        return;
      }
      for (int j = 0; j < d - 1; ++j)
        if (fr[c * 5 + j] < .0001 || fr[c * 5 + j] > 10000) {
          status[w] = 10;
          return;
        }
    }
    double T = cone_norm(jar + rb + i, fr + c * 5, d), N = jar[rb + i] * m;
    bool top = N >= m * T || (T <= 0 && N >= 0), bottom = m * N + T <= 0 || (T <= 0 && N < 0);
    if (!top && !bottom && T < 1e-12) {
      status[w] = 10;
      return;
    }
    // Repeated contact ids are legal. Keep the scalar last middle-cone writer,
    // including cases where a later top/bottom cone intentionally leaves H held.
    if (hessian_owner && (!rows_checked || collect_owner) && !top && !bottom)
      hessian_owner[c] = i;
    i += d;
  }
  // The emitter resets invariant_valid for every launch. The group map only
  // lives across Newton iterations with immutable row topology in that launch.
  if (group_head && !cached) {
    for (int r = 0; r < eq + fc; ++r)
      group_head[rb + r] = r;
    for (int i = eq + fc; i < n;) {
      int d = typ[rb + i] == 7 ? dim[cb + cid[rb + i]] : 1;
      group_head[rb + i] = i;
      for (int j = 1; j < d; ++j)
        group_head[rb + i + j] = -1;
      i += d;
    }
  }
  if (invariant_valid)
    invariant_valid[0] = 1;
  if (validate_only)
    return;
  double s = 0;
  int na = eq + fc, nco = 0, nchanged = 0;
  for (int r = 0; r < n; ++r)
    force[rb + r] = (-1.0 * D[rb + r]) * jar[rb + r];
  for (int r = 0; r < eq; ++r) {
    s += .5 * D[rb + r] * jar[rb + r] * jar[rb + r];
    if (old_state && old_state[rb + r] != 1)
      ++nchanged;
    state[rb + r] = 1;
  }
  for (int r = eq; r < eq + fc; ++r) {
    int z = rb + r;
    double v = jar[z], rr = R[z], f = loss[z];
    int next;
    if (v <= (-1.0 * rr) * f) {
      s += -.5 * rr * f * f - f * v;
      force[z] = f;
      next = 2;
    } else if (v >= rr * f) {
      s += -.5 * rr * f * f + f * v;
      force[z] = -1.0 * f;
      next = 3;
    } else {
      s += .5 * D[z] * v * v;
      next = 1;
    }
    if (old_state && old_state[z] != next)
      ++nchanged;
    state[z] = next;
  }
  for (int i = eq + fc; i < n;) {
    int z = rb + i;
    if (typ[z] != 7) {
      int next;
      if (jar[z] >= 0) {
        force[z] = 0.0;
        next = 0;
      } else {
        s += .5 * D[z] * jar[z] * jar[z];
        next = 1;
        ++na;
      }
      if (old_state && old_state[z] != next)
        ++nchanged;
      state[z] = next;
      ++i;
      continue;
    }
    int c = cb + cid[z], d = dim[c], fb = c * 5;
    double m = mu[c], N = jar[z] * m, T = cone_norm(jar + z, fr + fb, d);
    if (N >= m * T || (T <= 0 && N >= 0)) {
      for (int j = 0; j < d; ++j)
        force[z + j] = 0.0;
      state[z] = 0;
    } else if (m * N + T <= 0 || (T <= 0 && N < 0)) {
      for (int j = 0; j < d; ++j)
        s += .5 * D[z + j] * jar[z + j] * jar[z + j];
      state[z] = 1;
    } else {
      double dm = D[z] / (m * m * (1 + m * m)), nm = N - m * T;
      s += .5 * dm * nm * nm;
      force[z] = (-1.0 * dm) * nm * m;
      for (int j = 1; j < d; ++j)
        force[z + j] = (-1.0 * force[z]) / T * (jar[z + j] * fr[fb + j - 1]) * fr[fb + j - 1];
      state[z] = 4;
      if (flag[w]) {
        double *h = H + c * 36;
        double scale = (-1.0 * m) / T;
        h[0] = 1;
        for (int j = 1; j < d; ++j)
          h[j] = scale * (jar[z + j] * fr[fb + j - 1]);
        scale = m * N / (T * T * T);
        for (int k = 1; k < d; ++k) {
          double uk = jar[z + k] * fr[fb + k - 1];
          for (int j = k; j < d; ++j)
            h[k * d + j] = scale * (jar[z + j] * fr[fb + j - 1]) * uk;
        }
        scale = m * m - m * N / T;
        for (int j = 1; j < d; ++j)
          h[j * (d + 1)] += scale;
        for (int k = 0; k < d; ++k) {
          double a = dm * (k == 0 ? m : fr[fb + k - 1]);
          for (int j = k; j < d; ++j)
            h[k * d + j] *= a * (j == 0 ? m : fr[fb + j - 1]);
        }
        for (int k = 0; k < d; ++k)
          for (int j = k + 1; j < d; ++j)
            h[j * d + k] = h[k * d + j];
      }
    }
    int next = state[z];
    if (next)
      na += d;
    if (next == 4)
      nco += d;
    for (int j = 0; j < d; ++j) {
      if (old_state && old_state[z + j] != next)
        ++nchanged;
      state[z + j] = next;
    }
    i += d;
  }
  cost[w] = s;
  if (active_count)
    active_count[w] = na;
  if (cone_count)
    cone_count[w] = nco;
  if (changed_count)
    changed_count[0] = nchanged;
}

// Runtime span qualification preserves the admitted alias domain: overlapping
// outputs use the retained direct read/write equations below, never staging.
static __device__ inline bool constraint_spans_overlap(const double *a, int an, const double *b, int bn) {
  const unsigned long long ab = (unsigned long long)a, bb = (unsigned long long)b;
  return an > 0 && bn > 0 && ab < bb + (unsigned long long)bn * sizeof(double) &&
         bb < ab + (unsigned long long)an * sizeof(double);
}

// Fixed dimension kernels keep the middle-cone's at-most-six component work
// local to its owning lane. Typed passes make the dimension uniform per warp;
// no shuffle or barrier occurs inside a numeric zone branch.
template <int d>
static __device__ inline double constraint_middle_cone(double *jar, double *fr, double dm, double m, double N,
                                                       double T, double *force, double *h, bool write_h) {
  // The sole middle-cone owner writes its mathematical d*d block directly.
  // Avoid an address-taken local matrix and a second full copy; held tails,
  // input staging and overlapping-buffer fallback retain their existing rules.
  double nm = N - m * T;
  double cost = .5 * dm * nm * nm;
  force[0] = (-1.0 * dm) * nm * m;
#pragma unroll
  for (int j = 1; j < d; ++j)
    force[j] = (-1.0 * force[0]) / T * (jar[j] * fr[j - 1]) * fr[j - 1];
  if (write_h) {
    double scale = (-1.0 * m) / T;
    h[0] = 1;
#pragma unroll
    for (int j = 1; j < d; ++j)
      h[j] = scale * (jar[j] * fr[j - 1]);
    scale = m * N / (T * T * T);
#pragma unroll
    for (int k = 1; k < d; ++k) {
      double uk = jar[k] * fr[k - 1];
#pragma unroll
      for (int j = k; j < d; ++j)
        h[k * d + j] = scale * (jar[j] * fr[j - 1]) * uk;
    }
    scale = m * m - m * N / T;
#pragma unroll
    for (int j = 1; j < d; ++j)
      h[j * (d + 1)] += scale;
#pragma unroll
    for (int k = 0; k < d; ++k) {
      double a = dm * (k == 0 ? m : fr[k - 1]);
#pragma unroll
      for (int j = k; j < d; ++j)
        h[k * d + j] *= a * (j == 0 ? m : fr[j - 1]);
    }
#pragma unroll
    for (int k = 0; k < d; ++k) {
#pragma unroll
      for (int j = k + 1; j < d; ++j)
        h[j * d + k] = h[k * d + j];
    }
  }
  return cost;
}

static __device__ inline int constraint_head_rank(const int *heads, int count, int row) {
  int lo = 0, hi = count;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (heads[mid] < row)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo < count && heads[lo] == row ? lo : -1;
}

// Newton/S07 consumes offsets[t:t+2] only for t<ceil(nefc/16).
// Produce that live prefix cooperatively after ordered heads are published.
// The lower bound counts exactly the heads whose final row precedes tile t.
static __device__ inline void constraint_tile_head_offsets_block(int n, int eq, int fc, int rb, int cb,
                                                                 const int *typ, const int *cid,
                                                                 const int *dim, const int *heads, int nh,
                                                                 int *offsets) {
  for (int t = threadIdx.x; t <= (n + 15) / 16; t += blockDim.x) {
    int lo = 0, hi = nh;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2, r = heads[mid];
      const int d = r >= eq + fc && typ[rb + r] == 7 ? dim[cb + cid[rb + r]] : 1;
      if ((r + d - 1) / 16 < t)
        lo = mid + 1;
      else
        hi = mid;
    }
    offsets[t] = lo;
  }
}

// Advisory ordinal operands are selected only after the first ordinary N update
// has validated current rows and built its own head/G lists. The emitter resets
// selected at every invocation and keeps a block barrier after this lane0 gate.
// Metadata mismatch is a fallback, never a new solver/publication error.
static __device__ void constraint_head_metadata_gate(
    const int *active, const int *status, const int *invariant_valid, const int *count, const int *nefc,
    const int *ne, const int *nf, const int *ncon, const int *typ, const int *cid, const int *dim,
    const double *mu, const double *fr, const int *row_dim, const double *row_mu, const double *row_fr,
    const int *group_head, const int *head_rows, const int *head_count, const int *source_counts,
    const int *source_ncon, const int *compact_heads, const int *compact_count, const int *compact_valid,
    const int *head_type, const int *head_id, const int *head_dim, const double *head_mu,
    const double *head_friction, const int *head_valid, int *selected, int rows, int contacts, int nv) {
  if (threadIdx.x != 0)
    return;
  selected[0] = 0;
  if (active[0] != 1 || status[0] || invariant_valid[0] != 1)
    return;
  const int n = count[0], eq = ne[0], fc = nf[0], nc = ncon[0], nh = head_count[0];
  if (n <= 0 || n > rows || nefc[0] != n || eq < 0 || eq > n || fc < 0 || fc > n - eq || nc < 0 ||
      nc > contacts)
    return;
  if (compact_valid[0] != 1 || head_valid[0] != 1 || nh < 0 || nh > n || compact_count[0] != nh)
    return;
  if (source_counts[0] != eq || source_counts[1] != fc || source_counts[3] != n ||
      source_counts[4] != n * nv || source_ncon[0] != nc)
    return;
  if (source_counts[2] < 0 || source_counts[2] > n - eq - fc)
    return;
  int next = 0, limits = 0;
  for (int k = 0; k < nh; ++k) {
    const int r = head_rows[k];
    if (r != next || r < 0 || r >= n || compact_heads[k] != r || group_head[r] != r)
      return;
    const int t = typ[r], c = cid[r];
    if (head_type[k] != t || head_id[k] != c)
      return;
    int d = 1;
    if (r >= eq + fc && t == 7) {
      if (c < 0 || c >= nc)
        return;
      d = dim[c];
      if ((d != 1 && d != 3 && d != 4 && d != 6) || d > n - r || head_dim[k] != d || row_dim[r] != d)
        return;
      // Compare both current N views against S's ordinal view. Positive finite
      // operands were already admitted by the original ordered N validation.
      const long long m_bits = __double_as_longlong(head_mu[k]);
      if (m_bits != __double_as_longlong(mu[c]) || m_bits != __double_as_longlong(row_mu[r]))
        return;
      for (int j = 0; j < d - 1; ++j) {
        const double f = head_friction[j * rows + k];
        const long long f_bits = __double_as_longlong(f);
        if (f_bits != __double_as_longlong(fr[c * 5 + j]) ||
            f_bits != __double_as_longlong(row_fr[r * 5 + j]))
          return;
      }
      for (int j = 1; j < d; ++j)
        if (group_head[r + j] != -1 || typ[r + j] != t || cid[r + j] != c)
          return;
    } else {
      if (head_dim[k] != 1)
        return;
      if (r >= eq + fc && (t == 3 || t == 4))
        ++limits;
    }
    next = r + d;
  }
  if (next != n || limits != source_counts[2])
    return;
  selected[0] = 1;
}

// All threads enter this collective. Each scalar row or complete cone group
// has one owner; no thread writes another group's force/state/H cells. Group
// mathematics retains the scalar evaluation order; only cost/count sums change.

// Ordered topology has already been admitted by constraint_update_block.
// Keep its cheap head-map construction ordered, and distribute the expensive
// per-cone numerical admission and last-middle-cone ownership over CTA lanes.
static __device__ inline void
fused_numeric_admission_block(int w, int n, int eq, int fc, int rb, int cb, bool cached, const int *typ,
                              const int *cid, const int *dim, const double *jar, const double *mu,
                              const double *fr, int *group_head, int *hessian_owner, int *invariant_valid,
                              int *status, int *warp_bad) {
  if (threadIdx.x == 0 && !cached) {
    for (int r = 0; r < eq + fc; ++r)
      group_head[rb + r] = r;
    for (int r = eq + fc; r < n;) {
      const int d = typ[rb + r] == 7 ? dim[cb + cid[rb + r]] : 1;
      group_head[rb + r] = r;
      for (int j = 1; j < d; ++j)
        group_head[rb + r + j] = -1;
      r += d;
    }
  }
  __syncthreads();
  int bad = libero_warp::no_index;
  for (int r = eq + fc + threadIdx.x; r < n; r += blockDim.x) {
    if (group_head[rb + r] != r || typ[rb + r] != 7)
      continue;
    const int c = cb + cid[rb + r], d = dim[c];
    const double m = mu[c];
    if (!cached) {
      if (m < .0001 || m > 10000)
        bad = 0;
      for (int j = 0; j < d - 1; ++j)
        if (fr[c * 5 + j] < .0001 || fr[c * 5 + j] > 10000)
          bad = 0;
    }
    const double T = cone_norm(jar + rb + r, fr + c * 5, d), N = jar[rb + r] * m;
    const bool top = N >= m * T || (T <= 0 && N >= 0), bottom = m * N + T <= 0 || (T <= 0 && N < 0);
    if (!top && !bottom) {
      if (T < 1e-12)
        bad = 0;
      // Repeated ids are legal: later top/bottom groups leave the earlier H
      // intact. Only the last middle group is a mathematical Hessian writer.
      atomicMax(hessian_owner + c, r);
    }
  }
  bad = libero_warp::min(bad);
  if (libero_warp::lane() == 0)
    warp_bad[libero_warp::warp()] = bad;
  __syncthreads();
  if (threadIdx.x == 0) {
    int first = libero_warp::no_index;
    for (int k = 0; k < blockDim.x / 32; ++k)
      first = min(first, warp_bad[k]);
    if (first != libero_warp::no_index)
      status[w] = 10;
    else if (invariant_valid)
      invariant_valid[0] = 1;
  }
  __syncthreads();
}

static __device__ void constraint_apply_block(
    int w, int *active, int *count, int *ne, int *nf, int *ncon, bool write_hessian, double *jar, double *D,
    double *R, double *loss, int *typ, int *cid, double *J, double *mu, double *fr, int *dim, double *force,
    int *state, double *H, double *cost, int *status, int rows, int contacts, int nv, int *invariant_valid,
    int *active_count, int *cone_count, int *old_state, int *changed_count, int *group_head,
    int *hessian_owner, int *head_rows, int *head_count, int *tile_head_offset, int *typed_head_rows,
    int *typed_head_offsets, const int *head_dim, const double *head_mu, const double *head_friction,
    const double *cone_norms, bool trusted, unsigned char *precomputed = nullptr) {
  __shared__ double cost_warp[4];
  __shared__ int active_warp[4], cone_warp[4], change_warp[4];
  const int n = count[w], eq = ne[w], fc = nf[w], rb = w * rows, cb = w * contacts;
  // Use one warp only for small compiled models with at most one warp of rows.
  // The capacity condition removes this candidate from larger kernels entirely.
  const bool small = SubstepShape::nv <= 32 && n <= 32;
  if (small && threadIdx.x >= 32)
    return;
  double sum = 0.;
  int na = 0, nco = 0, nchanged = 0;
  const bool compact = head_rows && head_count;
  const bool typed = compact && typed_head_rows && typed_head_offsets;
  // Inspect actual input/output spans, including shifted aliases. This is a
  // qualification for staging, not a new public nonalias admission rule.
  const bool stage_inputs =
      trusted ||
      (!constraint_spans_overlap(force + rb, n, jar + rb, n) &&
       !constraint_spans_overlap(force + rb, n, D + rb, n) &&
       !constraint_spans_overlap(force + rb, n, fr + cb * 5, ncon[w] * 5) &&
       !constraint_spans_overlap(H + cb * 36, ncon[w] * 36, jar + rb, n) &&
       !constraint_spans_overlap(H + cb * 36, ncon[w] * 36, D + rb, n) &&
       !constraint_spans_overlap(H + cb * 36, ncon[w] * 36, fr + cb * 5, ncon[w] * 5) &&
       !constraint_spans_overlap(H + cb * 36, ncon[w] * 36, force + rb, n) &&
       !constraint_spans_overlap(force + rb, n, mu + cb, ncon[w]) &&
       !constraint_spans_overlap(H + cb * 36, ncon[w] * 36, mu + cb, ncon[w]) &&
       !constraint_spans_overlap(force + rb, n, (const double *)(dim + cb), (ncon[w] + 1) / 2) &&
       !constraint_spans_overlap(H + cb * 36, ncon[w] * 36, (const double *)(dim + cb), (ncon[w] + 1) / 2));

  for (int bucket = 0; bucket < (typed ? 7 : 1); ++bucket) {
    int begin = typed ? typed_head_offsets[bucket] : 0;
    int end = typed ? typed_head_offsets[bucket + 1] : (compact ? head_count[0] : n);
    for (int k = begin + threadIdx.x; k < end; k += blockDim.x) {
      int r = typed ? typed_head_rows[k] : (compact ? head_rows[k] : k);
      if (group_head[rb + r] != r)
        continue;
      int z = rb + r, next = 1;
      if (r < eq) {
        force[z] = (-1.0 * D[z]) * jar[z];
        sum += .5 * D[z] * jar[z] * jar[z];
        ++na;
      } else if (r < eq + fc) {
        double v = jar[z], rr = R[z], f = loss[z];
        const double limit = precomputed
            ? coefficient_plane(precomputed, NewtonCoefficientLayout::friction_limit)[r] : rr * f;
        force[z] = (-1.0 * D[z]) * v;
        if (v <= -limit) {
          sum += -.5 * rr * f * f - f * v;
          force[z] = f;
          next = 2;
        } else if (v >= limit) {
          sum += -.5 * rr * f * f + f * v;
          force[z] = -1.0 * f;
          next = 3;
        } else
          sum += .5 * D[z] * v * v;
        ++na;
      } else if (typ[z] != 7) {
        force[z] = (-1.0 * D[z]) * jar[z];
        if (jar[z] >= 0) {
          force[z] = 0.0;
          next = 0;
        } else {
          sum += .5 * D[z] * jar[z] * jar[z];
          ++na;
        }
      } else {
        const int rank = compact && (head_dim || head_mu || head_friction)
                             ? constraint_head_rank(head_rows, head_count[0], r)
                             : -1;
        int c = cb + cid[z], d = stage_inputs && head_dim && rank >= 0 ? head_dim[rank] : dim[c], fb = c * 5;
        double m = stage_inputs && head_mu && rank >= 0 ? head_mu[rank] : mu[c];
        // At most six components remain in registers through force, norm, cost
        // and the middle-cone Hessian producer. Scatter retains original row IDs.
        double local_jar[6], local_D[6], local_fr[5];
        double *input_jar = jar + z;
        double *input_D = D + z;
        double *input_fr = fr + fb;
        if (stage_inputs) {
          for (int j = 0; j < d; ++j) {
            local_jar[j] = jar[z + j];
            local_D[j] = D[z + j];
          }
          for (int j = 0; j < d - 1; ++j)
            local_fr[j] = head_friction && rank >= 0 ? head_friction[j * rows + rank] : fr[fb + j];
          input_jar = local_jar;
          input_D = local_D;
          input_fr = local_fr;
        }
        double N = input_jar[0] * m, T = cone_norms ? cone_norms[rb + r] : cone_norm(input_jar, input_fr, d);
        const int zone = cone_norms ? state[z] : -1;
#if LIBERO_NEWTON_SCALAR_IO
        for (int j = 0; j < d; ++j)
          force[z + j] = (-1.0 * input_D[j]) * input_jar[j];
        if (cone_norms ? zone == 0 : (N >= m * T || (T <= 0 && N >= 0))) {
          for (int j = 0; j < d; ++j)
            force[z + j] = 0.0;
          next = 0;
        }
#else
        // The owning lane's pair domain is this cone only. Staged inputs are
        // disjoint from every output; aliases keep the direct scalar equation.
        if (stage_inputs) {
          int j = 0;
          for (; j + 1 < d; j += 2) {
            double2 dv = libero_vec::load_pair(input_D + j), jv = libero_vec::load_pair(input_jar + j);
            libero_vec::store_pair(force + z + j, (-1.0 * dv.x) * jv.x, (-1.0 * dv.y) * jv.y);
          }
          if (j < d)
            force[z + j] = (-1.0 * input_D[j]) * input_jar[j];
        } else
          for (int j = 0; j < d; ++j)
            force[z + j] = (-1.0 * input_D[j]) * input_jar[j];
        if (cone_norms ? zone == 0 : (N >= m * T || (T <= 0 && N >= 0))) {
          if (stage_inputs)
            libero_newton_io::fill_serial(force + z, d, 0.0);
          else
            for (int j = 0; j < d; ++j)
              force[z + j] = 0.0;
          next = 0;
        }
#endif // LIBERO_NEWTON_SCALAR_IO
        else if (cone_norms ? zone == 1 : (m * N + T <= 0 || (T <= 0 && N < 0))) {
          for (int j = 0; j < d; ++j)
            sum += .5 * input_D[j] * input_jar[j] * input_jar[j];
          next = 1;
        } else {
          bool write_h = write_hessian && hessian_owner[c] == r;
          double *h = H + c * 36;
          const double dm = precomputed
              ? coefficient_plane(precomputed, NewtonCoefficientLayout::cone_dm)[r]
              : input_D[0] / (m * m * (1 + m * m));
          if (d == 3)
            sum += constraint_middle_cone<3>(input_jar, input_fr, dm, m, N, T, force + z, h, write_h);
          else if (d == 4)
            sum += constraint_middle_cone<4>(input_jar, input_fr, dm, m, N, T, force + z, h, write_h);
          else if (d == 6)
            sum += constraint_middle_cone<6>(input_jar, input_fr, dm, m, N, T, force + z, h, write_h);
          else
            sum += constraint_middle_cone<1>(input_jar, input_fr, dm, m, N, T, force + z, h, write_h);
          next = 4;
        }
        if (next)
          na += d;
        if (next == 4)
          nco += d;
#if LIBERO_NEWTON_SCALAR_IO
        for (int j = 0; j < d; ++j) {
          if (old_state && old_state[z + j] != next)
            ++nchanged;
          state[z + j] = next;
        }
#else
        // Scan before packet stores only when the old state cannot be changed
        // by them. Shifted aliases retain the original scan/write ordering.
        const uintptr_t sa = (uintptr_t)(state + z), oa = (uintptr_t)(old_state ? old_state + z : nullptr);
        const bool state_disjoint =
            trusted || !old_state ||
            (sa <= oa ? oa - sa >= size_t(d) * sizeof(int) : sa - oa >= size_t(d) * sizeof(int));
        if (state_disjoint) {
          for (int j = 0; j < d; ++j)
            if (old_state && old_state[z + j] != next)
              ++nchanged;
          libero_newton_io::fill_serial(state + z, d, next);
        } else
          for (int j = 0; j < d; ++j) {
            if (old_state && old_state[z + j] != next)
              ++nchanged;
            state[z + j] = next;
          }
#endif // LIBERO_NEWTON_SCALAR_IO
        continue;
      }
      if (old_state && old_state[z] != next)
        ++nchanged;
      state[z] = next;
    }
  }
  sum = libero_warp::sum(sum);
  na = libero_warp::sum(na);
  nco = libero_warp::sum(nco);
  nchanged = libero_warp::sum(nchanged);
  if (small) {
    if (threadIdx.x == 0) {
      cost[w] = sum;
      if (active_count)
        active_count[w] = na;
      if (cone_count)
        cone_count[w] = nco;
      if (changed_count)
        changed_count[0] = nchanged;
    }
    return; // caller's stage barrier publishes force/state/H and counts
  }
  const int lane = libero_warp::lane(), wid = libero_warp::warp(), warps = blockDim.x / 32;
  if (lane == 0) {
    cost_warp[wid] = sum;
    active_warp[wid] = na;
    cone_warp[wid] = nco;
    change_warp[wid] = nchanged;
  }
  __syncthreads();
  if (wid == 0) {
    sum = libero_warp::sum(lane < warps ? cost_warp[lane] : 0.);
    na = libero_warp::sum(lane < warps ? active_warp[lane] : 0);
    nco = libero_warp::sum(lane < warps ? cone_warp[lane] : 0);
    nchanged = libero_warp::sum(lane < warps ? change_warp[lane] : 0);
    if (lane == 0) {
      cost[w] = sum;
      if (active_count)
        active_count[w] = na;
      if (cone_count)
        cone_count[w] = nco;
      if (changed_count)
        changed_count[0] = nchanged;
    }
  }
}

static __device__ void constraint_update_block(
    int w, int *active, int *count, int *ne, int *nf, int *ncon, int *flag, double *jar, double *D, double *R,
    double *loss, int *typ, int *cid, double *J, double *mu, double *fr, int *dim, double *force, int *state,
    double *H, double *cost, int *status, int rows, int contacts, int nv, int *invariant_valid,
    int *active_count, int *cone_count, int *old_state, int *changed_count, int *group_head,
    int *hessian_owner, int *head_rows = nullptr, int *head_count = nullptr, int *tile_head_offset = nullptr,
    int *typed_head_rows = nullptr, int *typed_head_offsets = nullptr, const int *head_dim = nullptr,
    const double *head_mu = nullptr, const double *head_friction = nullptr) {
  __shared__ int check_warp[4], check_result;
  // Topology parsing stays ordered: variable-size cone groups and original
  // error precedence must be established before any parallel row access.
  if (threadIdx.x == 0)
    constraint_update(w, active, count, ne, nf, ncon, flag, jar, D, R, loss, typ, cid, J, mu, fr, dim, force,
                      state, H, cost, status, rows, contacts, nv, invariant_valid, nullptr, nullptr, nullptr,
                      nullptr, true, group_head, hessian_owner, true);
  __syncthreads();
  if (status[w] || !active[w])
    return; // uniform after the validation barrier
  int n = count[w], eq = ne[w], fc = nf[w], rb = w * rows, cb = w * contacts;
  const int check_lane = libero_warp::lane(), check_wid = libero_warp::warp(), check_warps = blockDim.x / 32;
  const bool cached = invariant_valid && invariant_valid[0];
  int bad = libero_warp::no_index;
  for (int r = threadIdx.x; r < n; r += blockDim.x) {
    int z = rb + r;
    if (!isfinite(jar[z]) || (!cached && (!isfinite(D[z]) || !isfinite(R[z]) || !isfinite(loss[z]))))
      bad = min(bad, 2 * r);
    if (jar[z] > 10000 || jar[z] < -10000 ||
        (!cached && (D[z] > 2e16 || D[z] < -10000 || R[z] > 10000 || R[z] < -10000 || loss[z] > 10000 ||
                     loss[z] < -10000)))
      bad = min(bad, 2 * n);
  }
  if (!cached)
    for (int k = threadIdx.x; k < n * nv; k += blockDim.x) {
      double v = J[rb * nv + k];
      if (!isfinite(v))
        bad = min(bad, 2 * (k / nv) + 1);
      if (v > 10000 || v < -10000)
        bad = min(bad, 2 * n);
    }
  // Range verdict sorts after every finite row/J verdict; a single reduction
  // therefore preserves both scalar phases without a second block barrier.
  bad = libero_warp::min(bad);
  if (check_lane == 0)
    check_warp[check_wid] = bad;
  __syncthreads();
  if (check_wid == 0) {
    int first = libero_warp::min(check_lane < check_warps ? check_warp[check_lane] : libero_warp::no_index);
    if (check_lane == 0) {
      check_result = first;
      if (first != libero_warp::no_index)
        status[w] = first == 2 * n ? 10 : ((first & 1) ? 9 : 8);
    }
  }
  __syncthreads();
  if (check_result != libero_warp::no_index)
    return;
    // Reuse the ordered numeric admission's cone classification to choose the
    // last middle-cone writer. The map is private; no numeric output is touched
    // before all original checks pass. Compatible callers retain their path.
#if DATAFLOW_FAST_PROFILE
  if (hessian_owner)
    for (int c = threadIdx.x; c < ncon[w]; c += blockDim.x)
      hessian_owner[cb + c] = -1;
  __syncthreads();
  constexpr bool collect_owner = true;
#else
  constexpr bool collect_owner = false;
#endif
  // Complete numeric cone checks in their original order, then publish the
  // topology map. The scalar implementation remains available as baseline.
#if DATAFLOW_FAST_PROFILE
  if (group_head && hessian_owner) {
    fused_numeric_admission_block(w, n, eq, fc, rb, cb, cached, typ, cid, dim, jar, mu, fr, group_head,
                                  hessian_owner, invariant_valid, status, check_warp);
  } else
#endif
  {
    if (threadIdx.x == 0)
      constraint_update(w, active, count, ne, nf, ncon, flag, jar, D, R, loss, typ, cid, J, mu, fr, dim,
                        force, state, H, cost, status, rows, contacts, nv, invariant_valid, nullptr, nullptr,
                        nullptr, nullptr, true, group_head, hessian_owner, false, true, collect_owner);
    __syncthreads();
  }
  if (status[w])
    return;
  // World-local scratch, rebuilt after admission on the first iteration of
  // each launch. Keep ascending original row IDs for line search; a separate
  // stable typed permutation makes each work pass uniform in type/dimension.
  // No numeric-zone sorting: zones vary with jar at every Newton iteration.
  if (threadIdx.x == 0 && !cached && head_rows && head_count) {
    int sizes[7] = {0, 0, 0, 0, 0, 0, 0}, nh = 0;
    for (int r = 0; r < n;) {
      int d = r >= eq + fc && typ[rb + r] == 7 ? dim[cb + cid[rb + r]] : 1;
      int bucket =
          r < eq
              ? 0
              : (r < eq + fc ? 1 : (typ[rb + r] != 7 ? 2 : (d == 1 ? 3 : (d == 3 ? 4 : (d == 4 ? 5 : 6)))));
      head_rows[nh++] = r;
      ++sizes[bucket];
      r += d;
    }
    head_count[0] = nh;
    if (typed_head_rows && typed_head_offsets) {
      int cursor[7];
      typed_head_offsets[0] = 0;
      for (int b = 0; b < 7; ++b) {
        cursor[b] = typed_head_offsets[b];
        typed_head_offsets[b + 1] = cursor[b] + sizes[b];
      }
      for (int k = 0; k < nh; ++k) {
        int r = head_rows[k];
        int d = r >= eq + fc && typ[rb + r] == 7 ? dim[cb + cid[rb + r]] : 1;
        int bucket =
            r < eq
                ? 0
                : (r < eq + fc ? 1 : (typ[rb + r] != 7 ? 2 : (d == 1 ? 3 : (d == 3 ? 4 : (d == 4 ? 5 : 6)))));
        typed_head_rows[cursor[bucket]++] = r;
      }
    }
#if !DATAFLOW_FAST_PROFILE
    if (tile_head_offset) {
      int k = 0;
      for (int t = 0; t <= (rows + 15) / 16; ++t) {
        while (k < nh) {
          int r = head_rows[k];
          int d = r >= eq + fc && typ[rb + r] == 7 ? dim[cb + cid[rb + r]] : 1;
          if ((r + d - 1) / 16 >= t)
            break;
          ++k;
        }
        tile_head_offset[t] = k;
      }
    }
#endif
  }
  __syncthreads();
#if DATAFLOW_FAST_PROFILE
  if (!cached && head_rows && head_count && tile_head_offset)
    constraint_tile_head_offsets_block(n, eq, fc, rb, cb, typ, cid, dim, head_rows, head_count[0],
                                       tile_head_offset);
  __syncthreads();
#endif
#if !DATAFLOW_FAST_PROFILE
  if (hessian_owner) {
    for (int c = threadIdx.x; c < ncon[w]; c += blockDim.x)
      hessian_owner[cb + c] = -1;
    __syncthreads();
    const bool compact = head_rows && head_count;
    for (int k = threadIdx.x; k < (compact ? head_count[0] : n); k += blockDim.x) {
      int r = compact ? head_rows[k] : k;
      if (r < eq + fc)
        continue;
      if (group_head[rb + r] != r || typ[rb + r] != 7)
        continue;
      int c = cb + cid[rb + r], d = dim[c];
      double m = mu[c], T = cone_norm(jar + rb + r, fr + c * 5, d), N = jar[rb + r] * m;
      bool top = N >= m * T || (T <= 0 && N >= 0), bottom = m * N + T <= 0 || (T <= 0 && N < 0);
      // The scalar contract is the last middle-cone writer, not the last
      // group with this contact id. Atomic max only reduces integer ownership.
      if (!top && !bottom)
        atomicMax(hessian_owner + c, r);
    }
    __syncthreads();
  }
#endif
  constraint_apply_block(w, active, count, ne, nf, ncon, flag[w], jar, D, R, loss, typ, cid, J, mu, fr, dim,
                         force, state, H, cost, status, rows, contacts, nv, invariant_valid, active_count,
                         cone_count, old_state, changed_count, group_head, hessian_owner, head_rows,
                         head_count, tile_head_offset, typed_head_rows, typed_head_offsets, head_dim, head_mu,
                         head_friction, nullptr, false);
}

// Entry owns the layout; this stage only updates quantities depending on jar.
static __device__ void constraint_update_core_block(unsigned char *shared, const NewtonWorldData &p,
                                                    bool write_hessian, double *jar, bool collect_counts,
                                                    bool track_changes) {
  auto &ctl = newton_control(shared);
  // Only the admitted solve body calls this stage. Entry already established
  // the immutable layout; it cannot fail or change halfway through a solve.
  const int n = ctl.count(), eq = ctl.ne(), fc = ctl.nf(), rb = 0, cb = 0;
  if (write_hessian) {
    for (int c = threadIdx.x; c < ctl.ncon(); c += blockDim.x)
      p.constraint_hessian_owner[cb + c] = -1;
    __syncthreads(); // reset owners before the following atomicMax votes
  }
  for (int k = threadIdx.x; k < p.constraint_head_count[0]; k += blockDim.x) {
    const int r = p.constraint_head_rows[k], z = rb + r;
    if (r < eq + fc || p.efc_type[z] != 7)
      continue;
    const int c = cb + p.efc_id[z], d = p.contact_dim[c];
    const double m = p.contact_mu[c], T = cone_norm(jar + z, p.contact_friction + c * 5, d), N = jar[z] * m;
    const bool top = N >= m * T || (T <= 0 && N >= 0);
    const bool bottom = m * N + T <= 0 || (T <= 0 && N < 0);
    const int zone = top ? 0 : (bottom ? 1 : 4);
    // The Jv direction is dead after commit and is regenerated by the next
    // line search. Reuse its global row storage; no new shared allocation.
    p.Jv[z] = T;
    p.state[z] = zone;
    if (write_hessian && zone == 4)
      atomicMax(p.constraint_hessian_owner + c, r);
  }
  __syncthreads();
  constraint_apply_block(
      0, (&ctl.active()), (&ctl.count()), (&ctl.ne()), (&ctl.nf()), (&ctl.ncon()), write_hessian, jar, p.D,
      p.R, p.loss, p.efc_type, p.efc_id, p.J, p.contact_mu, p.contact_friction, p.contact_dim, p.force,
      p.state, p.contact_H, (&ctl.cost()), (&ctl.status()), SubstepShape::rows, SubstepShape::contacts,
      SubstepShape::nv, (&ctl.invariant_valid()), (collect_counts ? &ctl.nactive() : nullptr),
      (collect_counts ? &ctl.ncone() : nullptr), (track_changes ? p.oldstate : nullptr),
      (track_changes ? &ctl.changed_count() : nullptr), p.constraint_group_heads, p.constraint_hessian_owner,
      p.constraint_head_rows, p.constraint_head_count, p.constraint_tile_head_offset,
      p.constraint_typed_head_rows, p.constraint_typed_head_offsets, nullptr, nullptr, nullptr, p.Jv, true,
      p.precomputed);
}

static __device__ void constraint_transpose(int tid, int *active, int *count, double *J, double *force,
                                            double *qfrc, int *status, int rows, int nv) {
  int w = tid / nv, j = tid % nv;
  if (!active[w] || status[w])
    return;
  double s = 0;
  for (int r = 0; r < count[w]; ++r) {
    double f = force[w * rows + r];
    if (f != 0)
      s += J[(w * rows + r) * nv + j] * f;
  }
  qfrc[tid] = s;
}

static __device__ double assembly_cell(int w, int i, int j, int rows, int *count, double *J, double *D,
                                       int *state, double *qm, int *madr, int *parent, int *simple) {
  double s = 0;
  for (int r = 0; r < count[w]; ++r) {
    int z = w * rows + r;
    if (state[z] == 1 && D[z] != 0) {
      double v = J[z * SubstepShape::nv + i];
      if (v != 0)
        s += J[z * SubstepShape::nv + j] * (v * D[z]);
    }
  }
  int adr = mass_index(i, j, madr, parent, simple);
  if (adr >= 0)
    s += qm[w * SubstepShape::packed_mass + adr];
  return s;
}
static __device__ void hessian_validate(int w, int *count, int *ncon, int *active, int *kind, int *state,
                                        int *oldstate, double *D, double *J, double *qm, double *H, int *id,
                                        int *dim, double *contact_H, int *ncone, int *nnz, int *status,
                                        int *route, int rows, int contacts, int *invariant_valid = nullptr) {
  status[w] = 0;
  route[w] = 0;
  if (!active[w])
    return;
  int k = kind[w], nr = count[w], nc = ncon[w], rb = w * rows;
  if (active[w] != 1 || (k != 1 && k != 2 && k != 3)) {
    status[w] = 1;
    return;
  }
  if (nr < 0 || nr > rows || nc < 0 || nc > contacts) {
    status[w] = 2;
    return;
  }
  if (k != 1 && nnz[w] != SubstepShape::nv * SubstepShape::nv) {
    status[w] = 3;
    return;
  }
  if (k != 1)
    for (int i = 0; i < SubstepShape::nv; ++i)
      for (int j = 0; j < SubstepShape::nv; ++j)
        if ((k == 2 || ncone[w] != 0 || j <= i) &&
            !isfinite(H[w * SubstepShape::nv * SubstepShape::nv + i * SubstepShape::nv + j])) {
          status[w] = 4;
          return;
        }
  int cones = 0;
  for (int r = 0; r < nr; ++r) {
    int s = state[rb + r], old = oldstate[rb + r];
    if (s < 0 || s > 4 || (k == 3 && (old < 0 || old > 4))) {
      status[w] = 3;
      return;
    }
    if (s == 4)
      ++cones;
    bool changed = k == 3 && ((old == 1) != (s == 1));
    if (((s == 1 && k == 1) || changed) &&
        (D[rb + r] < 0 || (!(invariant_valid && invariant_valid[0]) && !isfinite(D[rb + r])))) {
      status[w] = 4;
      return;
    }
    if (!(invariant_valid && invariant_valid[0]) &&
        (s == 4 || (k == 1 && s == 1 && D[rb + r] != 0) || changed))
      for (int j = 0; j < SubstepShape::nv; ++j)
        if (!isfinite(J[(rb + r) * SubstepShape::nv + j])) {
          status[w] = 4;
          return;
        }
  }
  if (cones != ncone[w]) {
    status[w] = 3;
    return;
  }
  for (int r = 0; r < nr;) {
    if (state[rb + r] != 4) {
      ++r;
      continue;
    }
    int c = id[rb + r];
    if (c < 0 || c >= nc) {
      status[w] = 3;
      return;
    }
    int d = dim[w * contacts + c];
    if ((d != 1 && d != 3 && d != 4 && d != 6) || r + d > nr) {
      status[w] = 3;
      return;
    }
    for (int j = 0; j < d; ++j)
      if (state[rb + r + j] != 4 || id[rb + r + j] != c) {
        status[w] = 3;
        return;
      }
    for (int j = 0; j < d * d; ++j)
      if (!isfinite(contact_H[(w * contacts + c) * 36 + j])) {
        status[w] = 4;
        return;
      }
    r += d;
  }
  if (k == 1)
    for (int j = 0; j < SubstepShape::packed_mass; ++j)
      if (!isfinite(qm[w * SubstepShape::packed_mass + j])) {
        status[w] = 4;
        return;
      }
}
// All block threads enter; only the first full warp performs validation.
// The caller must retain its stage barrier before consuming status/route.
static __device__ void hessian_validate_block(int w, int *count, int *ncon, int *active, int *kind,
                                              int *state, int *oldstate, double *D, double *J, double *qm,
                                              double *H, int *id, int *dim, double *contact_H, int *ncone,
                                              int *nnz, int *status, int *route, int rows, int contacts,
                                              int *invariant_valid = nullptr) {
  if (libero_warp::warp() != 0)
    return;
  const int lane = libero_warp::lane();
  int k = kind[w], nr = count[w], nc = ncon[w], rb = w * rows, code = 0;
  if (active[w]) {
    if (active[w] != 1 || (k != 1 && k != 2 && k != 3))
      code = 1;
    else if (nr < 0 || nr > rows || nc < 0 || nc > contacts)
      code = 2;
    else if (k != 1 && nnz[w] != SubstepShape::nv * SubstepShape::nv)
      code = 3;
  }
  if (lane == 0) {
    status[w] = code;
    route[w] = 0;
  }
  if (code || !active[w])
    return;
  bool bad = false;
  if (k != 1)
    for (int z = lane; z < SubstepShape::nv * SubstepShape::nv; z += 32)
      if ((k == 2 || ncone[w] != 0 || z % SubstepShape::nv <= z / SubstepShape::nv) &&
          !isfinite(H[w * SubstepShape::nv * SubstepShape::nv + z]))
        bad = true;
  if (libero_warp::any(bad)) {
    if (lane == 0)
      status[w] = 4;
    return;
  }
  const bool invariant = invariant_valid && invariant_valid[0];
  int first = libero_warp::no_index, first_code = 0, cones = 0;
  for (int r = lane; r < nr; r += 32) {
    int st = state[rb + r], old = oldstate[rb + r], err = 0;
    cones += st == 4;
    if (st < 0 || st > 4 || (k == 3 && (old < 0 || old > 4)))
      err = 3;
    else {
      bool changed = k == 3 && ((old == 1) != (st == 1));
      if (((st == 1 && k == 1) || changed) && (D[rb + r] < 0 || (!invariant && !isfinite(D[rb + r]))))
        err = 4;
      if (!invariant && (st == 4 || (k == 1 && st == 1 && D[rb + r] != 0) || changed))
        for (int j = 0; j < SubstepShape::nv; ++j)
          if (!isfinite(J[(rb + r) * SubstepShape::nv + j]))
            err = 4;
    }
    // Row position dominates error code, exactly as the serial row loop.
    if (err && r < first) {
      first = r;
      first_code = err;
    }
  }
  first = libero_warp::min(first);
  cones = libero_warp::sum(cones);
  if (first != libero_warp::no_index) {
    first_code = __shfl_sync(libero_warp::mask, first_code, first & 31);
    if (lane == 0)
      status[w] = first_code;
    return;
  }
  if (cones != ncone[w]) {
    if (lane == 0)
      status[w] = 3;
    return;
  }
  // Variable-length cone schema is ordered and must precede qM validation.
  code = 0;
  if (lane == 0)
    for (int r = 0; r < nr;) {
      if (state[rb + r] != 4) {
        ++r;
        continue;
      }
      int c = id[rb + r];
      if (c < 0 || c >= nc) {
        code = 3;
        break;
      }
      int d = dim[w * contacts + c];
      if ((d != 1 && d != 3 && d != 4 && d != 6) || r + d > nr) {
        code = 3;
        break;
      }
      for (int j = 0; j < d; ++j)
        if (state[rb + r + j] != 4 || id[rb + r + j] != c) {
          code = 3;
          break;
        }
      if (code)
        break;
      for (int j = 0; j < d * d; ++j)
        if (!isfinite(contact_H[(w * contacts + c) * 36 + j])) {
          code = 4;
          break;
        }
      if (code)
        break;
      r += d;
    }
  code = __shfl_sync(libero_warp::mask, code, 0);
  if (code) {
    if (lane == 0)
      status[w] = code;
    return;
  }
  bad = false;
  if (k == 1)
    for (int j = lane; j < SubstepShape::packed_mass; j += 32)
      if (!isfinite(qm[w * SubstepShape::packed_mass + j]))
        bad = true;
  if (libero_warp::any(bad) && lane == 0)
    status[w] = 4;
}
static __device__ void hessian_assemble(int tid, int *count, int *active, int *kind, double *J, double *D,
                                        int *state, double *qm, int *madr, int *parent, int *simple,
                                        double *H, int *status, int rows) {
  int w = tid / (SubstepShape::nv * SubstepShape::nv), i = (tid / SubstepShape::nv) % SubstepShape::nv,
      j = tid % SubstepShape::nv;
  if (!active[w] || kind[w] != 1 || status[w] || j > i)
    return;
  double v = assembly_cell(w, i, j, rows, count, J, D, state, qm, madr, parent, simple);
  H[w * SubstepShape::nv * SubstepShape::nv + i * SubstepShape::nv + j] = v;
  if (i != j)
    H[w * SubstepShape::nv * SubstepShape::nv + j * SubstepShape::nv + i] = v;
}
static __device__ void hessian_cone(int w, int rows, int contacts, int *count, int *state, int *id, int *dim,
                                    double *contact_H, double *J, double *H, double *HC, double *local,
                                    double *ltj, double *vec, int *nupdate) {
  int rb = w * rows;
  H += w * SubstepShape::nv * SubstepShape::nv;
  HC += w * SubstepShape::nv * SubstepShape::nv;
  local += w * 36;
  ltj += w * 6 * SubstepShape::nv;
  vec += w * SubstepShape::nv;
  for (int z = 0; z < SubstepShape::nv * SubstepShape::nv; ++z)
    HC[z] = H[z];
  for (int i = 0; i < count[w];) {
    if (state[rb + i] != 4) {
      ++i;
      continue;
    }
    int c = w * contacts + id[rb + i], d = dim[c];
    for (int z = 0; z < d * d; ++z)
      local[z] = contact_H[c * 36 + z];
    chol_factor(local, d, d);
    for (int z = 0; z < d * SubstepShape::nv; ++z)
      ltj[z] = 0.0;
    for (int r = 0; r < d; ++r)
      for (int c0 = 0; c0 <= r; ++c0) {
        double s = local[r * d + c0];
        for (int k = 0; k < SubstepShape::nv; ++k)
          ltj[c0 * SubstepShape::nv + k] += J[(rb + i + r) * SubstepShape::nv + k] * s;
      }
    for (int r = 0; r < d; ++r) {
      for (int k = 0; k < SubstepShape::nv; ++k)
        vec[k] = ltj[r * SubstepShape::nv + k];
      chol_update(HC, vec, 1);
    }
    nupdate[w] += d;
    i += d;
  }
}
static __device__ void hessian_factor_update(int w, int *count, int *ncon, int *active, int *kind, double *J,
                                             double *D, int *state, int *oldstate, double *qm, int *madr,
                                             int *parent, int *simple, int *id, int *dim, double *contact_H,
                                             int *ncone, double *H, double *HC, int *nnz, int *nupdate,
                                             double *local, double *ltj, double *vec, int *status, int *route,
                                             int rows, int contacts) {
  if (!active[w] || status[w])
    return;
  int hb = w * SubstepShape::nv * SubstepShape::nv, vb = w * SubstepShape::nv, k = kind[w];
  if (k == 1) {
    chol_factor(H + hb, SubstepShape::nv, SubstepShape::nv);
    nnz[w] = SubstepShape::nv * SubstepShape::nv;
    if (ncone[w]) {
      nupdate[w] = 0;
      hessian_cone(w, rows, contacts, count, state, id, dim, contact_H, J, H, HC, local, ltj, vec, nupdate);
      route[w] = 2;
    } else
      route[w] = 1;
    nupdate[w] = count[w];
  } else if (k == 2) {
    hessian_cone(w, rows, contacts, count, state, id, dim, contact_H, J, H, HC, local, ltj, vec, nupdate);
    route[w] = 6;
  } else {
    nupdate[w] = 0;
    bool lost = false;
    for (int r = 0; r < count[w]; ++r) {
      int z = w * rows + r;
      bool before = oldstate[z] == 1, after = state[z] == 1;
      if (before != after) {
        double s = sqrt(D[z]);
        for (int j = 0; j < SubstepShape::nv; ++j)
          vec[vb + j] = J[z * SubstepShape::nv + j] * s;
        int rank = chol_update(H + hb, vec + vb, after ? 1 : 0);
        ++nupdate[w];
        if (rank < SubstepShape::nv) {
          lost = true;
          break;
        }
      }
    }
    if (lost) {
      for (int z = 0; z < SubstepShape::packed_mass; ++z)
        if (!isfinite(qm[w * SubstepShape::packed_mass + z])) {
          status[w] = 4;
          return;
        }
      for (int r = 0; r < count[w]; ++r) {
        int z = w * rows + r;
        if (state[z] != 1)
          continue;
        if (D[z] < 0 || !isfinite(D[z])) {
          status[w] = 4;
          return;
        }
        if (D[z] != 0)
          for (int j = 0; j < SubstepShape::nv; ++j)
            if (!isfinite(J[z * SubstepShape::nv + j])) {
              status[w] = 4;
              return;
            }
      }
      for (int i = 0; i < SubstepShape::nv; ++i)
        for (int j = 0; j <= i; ++j) {
          double v = assembly_cell(w, i, j, rows, count, J, D, state, qm, madr, parent, simple);
          H[hb + i * SubstepShape::nv + j] = v;
          if (i != j)
            H[hb + j * SubstepShape::nv + i] = v;
        }
      chol_factor(H + hb, SubstepShape::nv, SubstepShape::nv);
      nnz[w] = SubstepShape::nv * SubstepShape::nv;
      if (ncone[w]) {
        nupdate[w] = 0;
        hessian_cone(w, rows, contacts, count, state, id, dim, contact_H, J, H, HC, local, ltj, vec, nupdate);
      }
      nupdate[w] = count[w];
      route[w] = 5;
    } else if (ncone[w]) {
      hessian_cone(w, rows, contacts, count, state, id, dim, contact_H, J, H, HC, local, ltj, vec, nupdate);
      route[w] = 4;
    } else
      route[w] = 3;
  }
  for (int z = 0; z < SubstepShape::nv * SubstepShape::nv; ++z) {
    bool check =
        k == 1 || route[w] == 5 || (k == 3 && route[w] != 5 && z / SubstepShape::nv >= z % SubstepShape::nv);
    if (check && !isfinite(H[hb + z])) {
      status[w] = 5;
      return;
    }
    if (ncone[w] && !isfinite(HC[hb + z])) {
      status[w] = 5;
      return;
    }
  }
}
static __device__ void direction(int w, double *Ma, double *smooth, double *constraint, double *H, double *HC,
                                 int *ncone, int *nefc, int *active, int *upstream, double *grad,
                                 double *Mgrad, double *candidate, int *status, double *scratch_grad,
                                 double *x, double *fwd) {
  if (upstream[w]) {
    status[w] = upstream[w];
    return;
  }
  status[w] = 0;
  if (!active[w])
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  if (nefc[w] < 0 || ncone[w] < 0 || ncone[w] > nefc[w]) {
    status[w] = 2;
    return;
  }
  double *L = (ncone[w] > 0 ? HC : H) + w * SubstepShape::nv * SubstepShape::nv;
  int vb = w * SubstepShape::nv;
  for (int i = 0; i < SubstepShape::nv; ++i) {
    if (!isfinite(Ma[vb + i]) || !isfinite(smooth[vb + i]) || !isfinite(constraint[vb + i])) {
      status[w] = 3;
      return;
    }
    for (int j = 0; j <= i; ++j)
      if (!isfinite(L[i * SubstepShape::nv + j])) {
        status[w] = 3;
        return;
      }
  }
  for (int i = 0; i < SubstepShape::nv; ++i)
    if (L[i * SubstepShape::nv + i] == 0) {
      status[w] = 4;
      return;
    }
  for (int i = 0; i < SubstepShape::nv; ++i) {
    double v = (Ma[vb + i] - smooth[vb + i]) - constraint[vb + i];
    if (!isfinite(v)) {
      status[w] = 5;
      return;
    }
    scratch_grad[vb + i] = v;
    x[vb + i] = v;
  }
  for (int i = 0; i < SubstepShape::nv; ++i) {
    double v = x[vb + i];
    if (i)
      v -= dot4(L + i * SubstepShape::nv, x + vb, i);
    v /= L[i * SubstepShape::nv + i];
    if (!isfinite(v)) {
      status[w] = 5;
      return;
    }
    x[vb + i] = v;
    fwd[vb + i] = v;
  }
  for (int i = SubstepShape::nv - 1; i >= 0; --i) {
    double v = x[vb + i];
    for (int j = i + 1; j < SubstepShape::nv; ++j)
      v -= L[j * SubstepShape::nv + i] * x[vb + j];
    v /= L[i * SubstepShape::nv + i];
    if (!isfinite(v)) {
      status[w] = 5;
      return;
    }
    x[vb + i] = v;
  }
  for (int i = 0; i < SubstepShape::nv; ++i) {
    grad[vb + i] = scratch_grad[vb + i];
    Mgrad[vb + i] = x[vb + i];
    candidate[vb + i] = copysign(x[vb + i], -copysign(1.0, x[vb + i]));
  }
}

#if LIBERO_NEWTON_SCALAR_IO
#else
// Pair only independent element updates. Shifted aliases retain scalar lane
// ownership; checks and alpha==0 remain before every write.
static __device__ inline bool n_state_disjoint(const void *a, size_t an, const void *b, size_t bn) {
  const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return !an || !bn || (x <= y ? y - x >= an : x - y >= bn);
}
static __device__ inline void n_state_axpy(double *dst, const double *src, int n, double a) {
  if (dst == src || n_state_disjoint(dst, size_t(n) * 8, src, size_t(n) * 8)) {
    int i = 2 * int(threadIdx.x);
    for (; i + 1 < n; i += 2 * int(blockDim.x)) {
      double2 x = libero_vec::load_pair(dst + i), y = libero_vec::load_pair(src + i);
      libero_vec::store_pair(dst + i, x.x + a * y.x, x.y + a * y.y);
    }
    if (i < n)
      dst[i] += a * src[i];
  } else
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      dst[i] += a * src[i];
}
#endif // LIBERO_NEWTON_SCALAR_IO
static __device__ void warm_select(int w, int *active, int *nefc, double *wc, double *sc, double *warm,
                                   double *smooth, double *sf, int *ss, double *sq, double *qacc,
                                   double *force, int *state, double *qfrc, int *choice, int *status, int nv,
                                   int rows) {
  if (status[w] || !active[w])
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  int n = nefc[w], v = w * nv, r = w * rows;
  if (n <= 0 || n > rows) {
    status[w] = 2;
    return;
  }
  if (!isfinite(wc[w]) || !isfinite(sc[w])) {
    status[w] = 3;
    return;
  }
  for (int i = 0; i < nv; ++i)
    if (!isfinite(warm[v + i]) || !isfinite(smooth[v + i]) || !isfinite(sq[v + i])) {
      status[w] = 3;
      return;
    }
  for (int i = 0; i < n; ++i)
    if (!isfinite(sf[r + i])) {
      status[w] = 3;
      return;
    }
  int select = wc[w] > sc[w];
  choice[w] = select;
  for (int i = 0; i < nv; ++i) {
    qacc[v + i] = select ? smooth[v + i] : warm[v + i];
    qfrc[v + i] = sq[v + i];
  }
  for (int i = 0; i < n; ++i) {
    force[r + i] = sf[r + i];
    state[r + i] = ss[r + i];
  }
}
static __device__ void commit_step(int w, int *active, int *nefc, double *alpha, double *search, double *mv,
                                   double *jv, double *qacc, double *ma, double *jar, int *committed,
                                   int *status, int nv, int rows) {
  if (status[w] || !active[w])
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  int n = nefc[w], v = w * nv, r = w * rows;
  if (n <= 0 || n > rows) {
    status[w] = 2;
    return;
  }
  double a = alpha[w];
  if (!isfinite(a)) {
    status[w] = 3;
    return;
  }
  if (a == 0) {
    committed[w] = 0;
    return;
  }
  for (int i = 0; i < nv; ++i)
    if (!isfinite(search[v + i]) || !isfinite(mv[v + i]) || !isfinite(qacc[v + i]) || !isfinite(ma[v + i]) ||
        !isfinite(qacc[v + i] + a * search[v + i]) || !isfinite(ma[v + i] + a * mv[v + i])) {
      status[w] = 3;
      return;
    }
  for (int i = 0; i < n; ++i)
    if (!isfinite(jv[r + i]) || !isfinite(jar[r + i]) || !isfinite(jar[r + i] + a * jv[r + i])) {
      status[w] = 3;
      return;
    }
  committed[w] = 1;
  for (int i = 0; i < nv; ++i)
    qacc[v + i] += a * search[v + i];
  for (int i = 0; i < nv; ++i)
    ma[v + i] += a * mv[v + i];
  for (int i = 0; i < n; ++i)
    jar[r + i] += a * jv[r + i];
}
static __device__ void stats_stop(int w, int *active, int *nefc, int *stage, int *committed, int *oldstate,
                                  int *state, double *grad, double *oldcost, double *cost, double *inertia,
                                  double *tolerance, double *slope, int *nactive, int *nupdate, int *neval,
                                  int *iteration, int *base, int *maxiter, double *sf, int *si, int *nchange,
                                  double *improvement, double *gradient, int *stop, int *reason,
                                  int *out_iter, int *status, int nv, int rows,
                                  int *precomputed_change = nullptr, double *precomputed_norm = nullptr,
                                  bool gradient_scanned = false, const int *residual_roundoff_ok = nullptr,
                                  const NewtonOptions *options = nullptr) {
  if (status[w] || !active[w])
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  int n = nefc[w], it = iteration[w], b = base[w];
  if (n <= 0 || n > rows || it < 0 || b < 0 || b > 2147483647 - it || it >= maxiter[w] || maxiter[w] < 0 ||
      maxiter[w] > 1000000) {
    status[w] = 2;
    return;
  }
  if (stage[w] != 1 || committed[w] != 1) {
    status[w] = 4;
    return;
  }
  if (!isfinite(oldcost[w]) || !isfinite(cost[w]) || !isfinite(inertia[w]) || !isfinite(tolerance[w]) ||
      !isfinite(slope[w]) || inertia[w] <= 0 || tolerance[w] < 0) {
    status[w] = 3;
    return;
  }
  if (nactive[w] < 0 || nupdate[w] < 0 || neval[w] < 0) {
    status[w] = 2;
    return;
  }
  int change = precomputed_change ? precomputed_change[0] : 0;
  if (!precomputed_change)
    for (int j = 0; j < n; ++j)
      if (oldstate[w * rows + j] != state[w * rows + j])
        ++change;
  // Internal norm production visits every gradient element. A finite sum of
  // squared components implies finite operands; avoid a second shared/global
  // load pass only for that call path. Scalar callers retain their guard.
  if (!gradient_scanned)
    for (int j = 0; j < nv; ++j)
      if (!isfinite(grad[w * nv + j])) {
        status[w] = 3;
        return;
      }
  double d;
  if (precomputed_norm)
    d = precomputed_norm[0];
  else {
    // Stats starts each lane at +0 (unlike mju_dot's lane priming).
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int j = 0;
    double *g = grad + w * nv;
    for (; j <= nv - 4; j += 4) {
      s0 += g[j] * g[j];
      s1 += g[j + 1] * g[j + 1];
      s2 += g[j + 2] * g[j + 2];
      s3 += g[j + 3] * g[j + 3];
    }
    d = (s0 + s2) + (s1 + s3);
    if (nv - j == 3)
      d += g[j] * g[j] + g[j + 1] * g[j + 1] + g[j + 2] * g[j + 2];
    else if (nv - j == 2)
      d += g[j] * g[j] + g[j + 1] * g[j + 1];
    else if (nv - j == 1)
      d += g[j] * g[j];
  }
  if (!isfinite(d)) {
    status[w] = 3;
    return;
  }
  double scale = inertia[w] * double(nv), imp = (oldcost[w] - cost[w]) / scale, gn = sqrt(d) / scale;
  if (!isfinite(imp) || !isfinite(gn)) {
    status[w] = 3;
    return;
  }
  int slot = b + it;
  if (slot < 0) {
    status[w] = 2;
    return;
  }
  if (slot < 1000) {
    int f = (w * 1000 + slot) * 3, i = (w * 1000 + slot) * 4;
    sf[f] = imp;
    sf[f + 1] = gn;
    sf[f + 2] = slope[w];
    si[i] = nactive[w];
    si[i + 1] = change;
    si[i + 2] = neval[w];
    si[i + 3] = nupdate[w];
  }
  // Match mj_solCGNewton: either scaled improvement or gradient is small.
  const bool cpu_stop = imp < tolerance[w] || gn < tolerance[w];
  bool selected_stop = cpu_stop;
  const bool audit = options && options->audit;
  const bool legacy = options && options->stop_policy == int(StopPolicy::RoundoffResidual);
  if (legacy || audit) {
    const double magnitude = fmax(fabs(oldcost[w]), fabs(cost[w]));
    const double resolution = 8.0 * 2.2204460492503131e-16 * magnitude;
    const double delta = oldcost[w] - cost[w];
    const bool ambiguous = fabs(delta) <= resolution;
    const bool resolved_small = delta >= 0.0 && !ambiguous && imp < tolerance[w];
    const bool roundoff_small = ambiguous && residual_roundoff_ok && residual_roundoff_ok[0];
    const bool legacy_stop = resolved_small || roundoff_small || gn < tolerance[w];
    if (legacy)
      selected_stop = legacy_stop;
    if (audit) {
      int64_t *counters = options->audit;
      ++counters[1];            // committed iterations
      counters[3] += ambiguous; // residual admission triggers
      if (cpu_stop != legacy_stop && counters[4] + counters[5] == 0 && options->first_disagreement) {
        double *event = options->first_disagreement;
        event[0] = oldcost[w];
        event[1] = cost[w];
        event[2] = imp;
        event[3] = gn;
        event[4] = tolerance[w];
        event[5] = it;
        event[6] = n;
        event[7] = counters[0];
      }
      counters[4] += cpu_stop && !legacy_stop;
      counters[5] += legacy_stop && !cpu_stop;
    }
  }
  int next = it + 1, st = 0, re = 0;
  if (selected_stop) {
    st = 1;
    re = 1;
  } else if (next >= maxiter[w]) {
    st = 1;
    re = 2;
  }
  nchange[w] = change;
  improvement[w] = imp;
  gradient[w] = gn;
  stop[w] = st;
  reason[w] = re;
  out_iter[w] = next;
}

// The actual solver calls this with all 32/64/128 block threads. Validation and
// stopping remain in stats_stop, so legal stop tests and write priority stay put.
static __device__ void
stats_stop_block(int w, int *active, int *nefc, int *stage, int *committed, int *oldstate, int *state,
                 double *grad, double *oldcost, double *cost, double *inertia, double *tolerance,
                 double *slope, int *nactive, int *nupdate, int *neval, int *iteration, int *base,
                 int *maxiter, double *sf, int *si, int *nchange, double *improvement, double *gradient,
                 int *stop, int *reason, int *out_iter, int *status, int nv, int rows,
                 int *precomputed_change = nullptr, const double *residual_Ma = nullptr,
                 const double *residual_smooth = nullptr, const double *residual_J = nullptr,
                 const double *residual_force = nullptr, const NewtonOptions *options = nullptr) {
  __shared__ double norm_warp[4], norm;
  __shared__ int changes_warp[4], changes;
  __shared__ int residual_bad_warp[4], residual_ok;
  if (status[w] || !active[w])
    return;
  const int lane = libero_warp::lane(), wid = libero_warp::warp();
  constexpr int warps = flash_libero::SubstepShape::threads / 32;
  // Counts may be malformed: don't dereference any rows until bounds are safe.
  const int n = nefc[w];
  double value = 0.;
  int change = 0;
  // The componentwise residual is experimental policy/audit work, not a
  // requirement of CPU Newton. The predicate is uniform for the whole world.
  const bool need_residual =
      options && (options->stop_policy == int(StopPolicy::RoundoffResidual) || options->audit);
  if (need_residual) {
    // A standard gamma_k bound covers the product/accumulation in J^T f,
    // the packed Mq product and the two residual subtractions. This is a
    // componentwise backward-error contract, not a claim of small forward error.
    const double mag = fmax(fabs(oldcost[w]), fabs(cost[w]));
    const bool ambiguous = fabs(oldcost[w] - cost[w]) <= 8.0 * 2.2204460492503131e-16 * mag;
    int bad = 0;
    if (ambiguous && residual_Ma && residual_smooth && residual_J && residual_force && n > 0 && n <= rows) {
      const double eps = 1.1102230246251565e-16;
      const double k = double(n + 2 * nv + 8), gamma = (k * eps) / (1.0 - k * eps);
      for (int j = threadIdx.x; j < nv; j += blockDim.x) {
        double sum = fabs(residual_Ma[w * nv + j]) + fabs(residual_smooth[w * nv + j]);
        for (int r = 0; r < n; ++r)
          sum += fabs(residual_J[(w * rows + r) * nv + j] * residual_force[w * rows + r]);
        // Bound the positive summation's own rounding before using its scale.
        const double limit = tolerance[w] * inertia[w] * double(nv) + gamma * sum / (1.0 - gamma);
        if (!isfinite(sum) || !isfinite(limit) || !isfinite(grad[w * nv + j]) ||
            fabs(grad[w * nv + j]) > limit)
          bad = 1;
      }
    } else
      bad = 1;
    bad = int(libero_warp::any(bad != 0));
    if (lane == 0)
      residual_bad_warp[wid] = bad;
    __syncthreads();
    if (wid == 0) {
      bad = int(libero_warp::any(lane < warps && residual_bad_warp[lane] != 0));
      if (lane == 0)
        residual_ok = !bad;
    }
    __syncthreads();
  }

  // The fused caller already has a block barrier after this stage. Keep small
  // sums in registers, with no shared partials or additional block barriers.
  // NV is the compiled capacity; using it also removes this entire candidate
  // path from larger kernels, where paired timings showed no improvement.
  if (SubstepShape::nv <= 32 && nv <= 32 && (precomputed_change || n <= 64)) {
    if (wid != 0)
      return;
    if (active[w] == 1 && n > 0 && n <= rows) {
      for (int j = lane; j < nv; j += 32) {
        double g = grad[w * nv + j];
        value += g * g;
      }
      if (!precomputed_change)
        for (int j = lane; j < n; j += 32)
          change += oldstate[w * rows + j] != state[w * rows + j];
    }
    value = libero_warp::sum(value);
    if (precomputed_change)
      change = precomputed_change[0];
    else
      change = libero_warp::sum(change);
    if (lane == 0)
      stats_stop(w, active, nefc, stage, committed, oldstate, state, grad, oldcost, cost, inertia, tolerance,
                 slope, nactive, nupdate, neval, iteration, base, maxiter, sf, si, nchange, improvement,
                 gradient, stop, reason, out_iter, status, nv, rows, &change, &value, true, &residual_ok,
                 options);
    return;
  }
  if (active[w] == 1 && n > 0 && n <= rows) {
    for (int j = threadIdx.x; j < nv; j += blockDim.x) {
      double g = grad[w * nv + j];
      value += g * g;
    }
    if (!precomputed_change)
      for (int j = threadIdx.x; j < n; j += blockDim.x)
        change += oldstate[w * rows + j] != state[w * rows + j];
  }
  value = libero_warp::sum(value);
  change = libero_warp::sum(change);
  if (lane == 0) {
    norm_warp[wid] = value;
    changes_warp[wid] = change;
  }
  __syncthreads();
  if (wid == 0) {
    value = libero_warp::sum(lane < warps ? norm_warp[lane] : 0.);
    change = libero_warp::sum(lane < warps ? changes_warp[lane] : 0);
    if (lane == 0) {
      norm = value;
      changes = precomputed_change ? precomputed_change[0] : change;
    }
  }
  // norm/changes are produced and consumed by lane zero. The solve loop
  // publishes its stopping decision after this collective returns.
  if (threadIdx.x == 0)
    stats_stop(w, active, nefc, stage, committed, oldstate, state, grad, oldcost, cost, inertia, tolerance,
               slope, nactive, nupdate, neval, iteration, base, maxiter, sf, si, nchange, improvement,
               gradient, stop, reason, out_iter, status, nv, rows, &changes, &norm, true, &residual_ok,
               options);
}

static __device__ void finalize(int w, int *active, int *exit, int *iteration, double *qacc, double *warm,
                                int *iter, int *nnz, int *status, int nv) {
  if (status[w] || !active[w])
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  if (exit[w] != 0 && exit[w] != 1) {
    status[w] = 4;
    return;
  }
  if (!exit[w])
    return;
  int it = iteration[w];
  if (it < 0 || iter[w] < 0 || iter[w] > 2147483647 - it) {
    status[w] = 2;
    return;
  }
  for (int j = 0; j < nv; ++j)
    if (!isfinite(qacc[w * nv + j])) {
      status[w] = 3;
      return;
    }
  iter[w] += it;
  nnz[w] = nv * nv;
  for (int j = 0; j < nv; ++j)
    warm[w * nv + j] = qacc[w * nv + j];
}

// All-thread transactional entry points. Scalar entry points above remain the
// reference for standalone callers. The caller publishes each stage with its
// existing block barrier; no destination is touched before collective checks.
static __device__ bool n5_any_bad(bool bad) {
  return __syncthreads_or(bad) != 0;
}
static __device__ void warm_select_block(int w, int *active, int *nefc, double *wc, double *sc, double *warm,
                                         double *smooth, double *sf, int *ss, double *sq, double *qacc,
                                         double *force, int *state, double *qfrc, int *choice, int *status,
                                         int nv, int rows) {
  __shared__ int run;
  if (threadIdx.x == 0) {
    run = !status[w] && active[w];
  }
  __syncthreads();
  if (!run)
    return;
  if (threadIdx.x == 0) {
    if (active[w] != 1)
      status[w] = 1;
    else if (nefc[w] <= 0 || nefc[w] > rows)
      status[w] = 2;
    else if (!isfinite(wc[w]) || !isfinite(sc[w]))
      status[w] = 3;
  }
  __syncthreads();
  if (status[w])
    return;
  const int n = nefc[w], v = w * nv, r = w * rows;
  bool bad = false;
  for (int i = threadIdx.x; i < nv; i += blockDim.x)
    bad |= !isfinite(warm[v + i]) || !isfinite(smooth[v + i]) || !isfinite(sq[v + i]);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    bad |= !isfinite(sf[r + i]);
  if (n5_any_bad(bad)) {
    if (threadIdx.x == 0)
      status[w] = 3;
    return;
  }
  const int select = wc[w] > sc[w];
  if (threadIdx.x == 0)
    choice[w] = select;
#if LIBERO_NEWTON_SCALAR_IO
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    qacc[v + i] = select ? smooth[v + i] : warm[v + i];
    qfrc[v + i] = sq[v + i];
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    force[r + i] = sf[r + i];
    state[r + i] = ss[r + i];
  }
#else
  const double *selected = select ? smooth + v : warm + v;
  const bool independent =
      n_state_disjoint(qacc + v, size_t(nv) * 8, qfrc + v, size_t(nv) * 8) &&
      n_state_disjoint(qacc + v, size_t(nv) * 8, sq + v, size_t(nv) * 8) &&
      n_state_disjoint(qfrc + v, size_t(nv) * 8, selected, size_t(nv) * 8) &&
      (qacc + v == selected || n_state_disjoint(qacc + v, size_t(nv) * 8, selected, size_t(nv) * 8)) &&
      (qfrc + v == sq + v || n_state_disjoint(qfrc + v, size_t(nv) * 8, sq + v, size_t(nv) * 8));
  if (independent) {
    libero_newton_io::copy_block(selected, qacc + v, nv);
    libero_newton_io::copy_block(sq + v, qfrc + v, nv);
  } else
    for (int i = threadIdx.x; i < nv; i += blockDim.x) {
      qacc[v + i] = selected[i];
      qfrc[v + i] = sq[v + i];
    }
  const bool row_independent =
      n_state_disjoint(force + r, size_t(n) * 8, state + r, size_t(n) * 4) &&
      n_state_disjoint(force + r, size_t(n) * 8, ss + r, size_t(n) * 4) &&
      n_state_disjoint(state + r, size_t(n) * 4, sf + r, size_t(n) * 8) &&
      (force + r == sf + r || n_state_disjoint(force + r, size_t(n) * 8, sf + r, size_t(n) * 8)) &&
      (state + r == ss + r || n_state_disjoint(state + r, size_t(n) * 4, ss + r, size_t(n) * 4));
  if (row_independent) {
    libero_newton_io::copy_block(sf + r, force + r, n);
    libero_newton_io::copy_block(ss + r, state + r, n);
  } else
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      force[r + i] = sf[r + i];
      state[r + i] = ss[r + i];
    }
#endif // LIBERO_NEWTON_SCALAR_IO
}
static __device__ void commit_step_block(int w, int *active, int *nefc, double *alpha, double *search,
                                         double *mv, double *jv, double *qacc, double *ma, double *jar,
                                         int *committed, int *status, int nv, int rows) {
  __shared__ int run;
  if (threadIdx.x == 0) {
    run = !status[w] && active[w];
  }
  __syncthreads();
  if (!run)
    return;
  if (threadIdx.x == 0) {
    if (active[w] != 1)
      status[w] = 1;
    else if (nefc[w] <= 0 || nefc[w] > rows)
      status[w] = 2;
    else if (!isfinite(alpha[w]))
      status[w] = 3;
  }
  __syncthreads();
  if (status[w])
    return;
  const int n = nefc[w], v = w * nv, r = w * rows;
  const double a = alpha[w];
  if (a == 0) {
    if (threadIdx.x == 0)
      committed[w] = 0;
    return;
  }
  bool bad = false;
  for (int i = threadIdx.x; i < nv; i += blockDim.x)
    bad |= !isfinite(search[v + i]) || !isfinite(mv[v + i]) || !isfinite(qacc[v + i]) ||
           !isfinite(ma[v + i]) || !isfinite(qacc[v + i] + a * search[v + i]) ||
           !isfinite(ma[v + i] + a * mv[v + i]);
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    bad |= !isfinite(jv[r + i]) || !isfinite(jar[r + i]) || !isfinite(jar[r + i] + a * jv[r + i]);
  if (n5_any_bad(bad)) {
    if (threadIdx.x == 0)
      status[w] = 3;
    return;
  }
  if (threadIdx.x == 0)
    committed[w] = 1;
#if LIBERO_NEWTON_SCALAR_IO
  for (int i = threadIdx.x; i < nv; i += blockDim.x)
    qacc[v + i] += a * search[v + i];
  for (int i = threadIdx.x; i < nv; i += blockDim.x)
    ma[v + i] += a * mv[v + i];
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    jar[r + i] += a * jv[r + i];
#else
  const double *spans[6] = {qacc + v, search + v, ma + v, mv + v, jar + r, jv + r};
  bool independent = true;
  for (int x = 0; x < 6; ++x)
    for (int y = x + 1; y < 6; ++y)
      independent &=
          n_state_disjoint(spans[x], size_t(x < 4 ? nv : n) * 8, spans[y], size_t(y < 4 ? nv : n) * 8);
  if (independent) {
    n_state_axpy(qacc + v, search + v, nv, a);
    n_state_axpy(ma + v, mv + v, nv, a);
    n_state_axpy(jar + r, jv + r, n, a);
  } else {
    for (int i = threadIdx.x; i < nv; i += blockDim.x)
      qacc[v + i] += a * search[v + i];
    for (int i = threadIdx.x; i < nv; i += blockDim.x)
      ma[v + i] += a * mv[v + i];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      jar[r + i] += a * jv[r + i];
  }
#endif // LIBERO_NEWTON_SCALAR_IO
}
// Fused-owned views are disjoint and entry counts have already been admitted.
// CPU mj_solCGNewton updates these three vectors directly; error recovery is
// performed by the enclosing mj_step-equivalent path, not a preflight AXPY.
static __device__ void commit_step_core_block(int w, int *active, int *nefc, const double *alpha,
                                              const double *search, const double *mv, const double *jv,
                                              double *qacc, double *ma, double *jar, int *committed,
                                              int nv, int rows) {
  if (!active[w])
    return;
  const int n = nefc[w], v = w * nv, r = w * rows;
  const double a = alpha[w];
  if (threadIdx.x == 0)
    committed[w] = a != 0.0;
  if (a == 0.0)
    return;
  n_state_axpy(qacc + v, search + v, nv, a);
  n_state_axpy(ma + v, mv + v, nv, a);
  n_state_axpy(jar + r, jv + r, n, a);
}

// The projection/constraint producers own schema validation. The fused Hessian
// stage needs fresh status/route, not another scan of their numeric outputs.
static __device__ void hessian_prepare_core_block(int w, int *status, int *route) {
  if (threadIdx.x == 0) {
    status[w] = 0;
    route[w] = 0;
  }
}

static __device__ void finalize_block(int w, int *active, int *exit, int *iteration, double *qacc,
                                      double *warm, int *iter, int *nnz, int *status, int nv) {
  __shared__ int run;
  if (threadIdx.x == 0) {
    run = !status[w] && active[w];
  }
  __syncthreads();
  if (!run)
    return;
  if (threadIdx.x == 0) {
    if (active[w] != 1)
      status[w] = 1;
    else if (exit[w] != 0 && exit[w] != 1)
      status[w] = 4;
    else if (exit[w] && (iteration[w] < 0 || iter[w] < 0 || iter[w] > 2147483647 - iteration[w]))
      status[w] = 2;
  }
  __syncthreads();
  if (status[w] || !exit[w])
    return;
  bool bad = false;
  for (int j = threadIdx.x; j < nv; j += blockDim.x)
    bad |= !isfinite(qacc[w * nv + j]);
  if (n5_any_bad(bad)) {
    if (threadIdx.x == 0)
      status[w] = 3;
    return;
  }
  if (threadIdx.x == 0) {
    iter[w] += iteration[w];
    nnz[w] = nv * nv;
  }
#if LIBERO_NEWTON_SCALAR_IO
  for (int j = threadIdx.x; j < nv; j += blockDim.x)
    warm[w * nv + j] = qacc[w * nv + j];
#else
  if (warm == qacc || n_state_disjoint(warm + w * nv, size_t(nv) * 8, qacc + w * nv, size_t(nv) * 8))
    libero_newton_io::copy_block(qacc + w * nv, warm + w * nv, nv);
  else
    for (int j = threadIdx.x; j < nv; j += blockDim.x)
      warm[w * nv + j] = qacc[w * nv + j];
#endif // LIBERO_NEWTON_SCALAR_IO
}

static __device__ void jacvec_sub(int w, int *active, int *n, double *J, double *q, double *aref, double *out,
                                  int rows) {
  if (!active[w])
    return;
  for (int r = 0; r < n[w]; ++r)
    out[w * rows + r] =
        dot4(J + (w * rows + r) * SubstepShape::nv, q + w * SubstepShape::nv, SubstepShape::nv) -
        aref[w * rows + r];
}
static __device__ void mulM(int w, int *active, double *q, double *M, int *madr, int *parent, int *simple,
                            double *out) {
  if (active[w])
    mass_multiply(q + w * SubstepShape::nv, M + w * SubstepShape::packed_mass, madr, parent, simple,
                  out + w * SubstepShape::nv);
}
// Address-only half-open span test. Unsigned distances avoid end-address
// overflow and avoid relational comparison of unrelated C++ pointers.
static __device__ inline bool mass_spans_overlap(const void *a, unsigned long long na, const void *b,
                                                 unsigned long long nb) {
  if (!na || !nb)
    return false;
  const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
  return pa <= pb ? pb - pa < na : pa - pb < nb;
}
// Collective planned product. Preserve original scalar behavior for exact or
// shifted input/output aliases; the qualification does not read any payload.
static __device__ void mulM_plan_block(int w, const int *active, const double *q, const double *M,
                                       const int *madr, const int *parent, const int *simple, double *out,
                                       const int *offsets, const int *coefficient, const int *source,
                                       const int *assign) {
  if (!active[w])
    return;
  const double *x = q + w * SubstepShape::nv;
  const double *qm = M + w * SubstepShape::packed_mass;
  double *result = out + w * SubstepShape::nv;
  const bool disjoint =
      !mass_spans_overlap(result, SubstepShape::nv * sizeof(double), x, SubstepShape::nv * sizeof(double)) &&
      !mass_spans_overlap(result, SubstepShape::nv * sizeof(double), qm,
                          SubstepShape::packed_mass * sizeof(double)) &&
      !mass_spans_overlap(result, SubstepShape::nv * sizeof(double), madr, SubstepShape::nv * sizeof(int)) &&
      !mass_spans_overlap(result, SubstepShape::nv * sizeof(double), parent,
                          SubstepShape::nv * sizeof(int)) &&
      !mass_spans_overlap(result, SubstepShape::nv * sizeof(double), simple, SubstepShape::nv * sizeof(int));
  if (offsets && disjoint)
    libero_packed_mass::multiply_plan(x, qm, offsets, coefficient, source, assign, result, SubstepShape::nv);
  else if (threadIdx.x == 0)
    mass_multiply(x, qm, madr, parent, simple, result);
}
// The warm and smooth candidates have already consumed exactly these operands.
// Reuse the selected candidate, without replacing M*q by an approximate force.
static __device__ void warm_init_jacvec(int *active, int *count, int *choice, double *warm, double *smooth,
                                        double *out) {
  if (!active[0])
    return;
  const double *selected = choice[0] ? smooth : warm;
  for (int r = threadIdx.x; r < count[0]; r += blockDim.x)
    out[r] = selected[r];
}
static __device__ void warm_init_mass_prepared(const int *active, const int *choice, const double *warm,
                                               const double *q, const double *M, double *out,
                                               const NewtonModelData &m) {
  if (!active[0])
    return;
  if (!choice[0]) {
    for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
      out[j] = warm[j];
  } else
    libero_packed_mass::multiply_plan(q, M, m.mass_plan_offsets, m.mass_plan_coefficient, m.mass_plan_source,
                                      m.mass_plan_assign, out, SubstepShape::nv);
}
static __device__ void gauss_compute(int w, int *active, double *Ma, double *smooth, double *q, double *qs,
                                     double *cost, double *quad0, int *add) {
  if (!active[w])
    return;
  double s = 0;
  for (int j = 0; j < SubstepShape::nv; ++j) {
    int i = w * SubstepShape::nv + j;
    s += .5 * (Ma[i] - smooth[i]) * (q[i] - qs[i]);
  }
  quad0[w] = s;
  if (add[w])
    cost[w] += s;
}

// A nonstandard scalar objective over two difference vectors; no BLAS matrix
// operation is replaced. Small vectors stay in warp 0; each lane accumulates
// one DOF. Larger vectors retain the measured block reduction. The caller's
// stage barrier publishes the lane-0 result.
static __device__ void gauss_compute_block(int w, int *active, double *Ma, double *smooth, double *q,
                                           double *qs, double *cost, double *quad0, bool add_cost) {
  __shared__ double warp_partial[4];
  if (!active[w])
    return;
  double value = 0.;
  if (SubstepShape::nv <= 64) {
    if (threadIdx.x >= 32)
      return;
    for (int j = threadIdx.x; j < SubstepShape::nv; j += 32) {
      int i = w * SubstepShape::nv + j;
      value += .5 * (Ma[i] - smooth[i]) * (q[i] - qs[i]);
    }
    value = libero_warp::sum(value);
    if (threadIdx.x == 0) {
      quad0[w] = value;
      if (add_cost)
        cost[w] += value;
    }
    return;
  }
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x) {
    int i = w * SubstepShape::nv + j;
    value += .5 * (Ma[i] - smooth[i]) * (q[i] - qs[i]);
  }
  value = libero_warp::sum(value);
  const int lane = libero_warp::lane(), wid = libero_warp::warp(), warps = blockDim.x / 32;
  if (lane == 0)
    warp_partial[wid] = value;
  __syncthreads();
  if (wid == 0) {
    value = libero_warp::sum(lane < warps ? warp_partial[lane] : 0.);
    if (lane == 0) {
      quad0[w] = value;
      if (add_cost)
        cost[w] += value;
    }
  }
}

static __device__ void count_active_cone(int w, int *active, int *state, int *n, int *na, int *nc, int rows) {
  if (!active[w])
    return;
  int a = 0, c = 0;
  for (int j = 0; j < n[w]; ++j) {
    int s = state[w * rows + j];
    if (s)
      ++a;
    if (s == 4)
      ++c;
  }
  na[w] = a;
  nc[w] = c;
}
static __device__ void save_old(int w, int *active, int *n, int *state, double *cost, int *old,
                                double *oldcost, int rows) {
#if LIBERO_NEWTON_SCALAR_IO
  if (!active[w])
    return;
  for (int j = 0; j < n[w]; ++j)
    old[w * rows + j] = state[w * rows + j];
  oldcost[w] = cost[w];
#else
  if (!active[w])
    return;
  if (old == state || n_state_disjoint(old + w * rows, size_t(n[w]) * 4, state + w * rows, size_t(n[w]) * 4))
    libero_newton_io::copy_serial(state + w * rows, old + w * rows, n[w]);
  else
    for (int j = 0; j < n[w]; ++j)
      old[w * rows + j] = state[w * rows + j];
  oldcost[w] = cost[w];
#endif // LIBERO_NEWTON_SCALAR_IO
}
static __device__ void direction_commit(int w, int *active, double *candidate, double *search) {
  if (active[w])
    for (int j = 0; j < SubstepShape::nv; ++j)
      search[w * SubstepShape::nv + j] = candidate[w * SubstepShape::nv + j];
}
static __device__ void empty_fast_path(int w, int *empty, double *smooth, double *q, double *warm,
                                       double *force, int *iter) {
  if (!empty[w])
    return;
  for (int j = 0; j < SubstepShape::nv; ++j) {
    int i = w * SubstepShape::nv + j;
    q[i] = smooth[i];
    warm[i] = smooth[i];
    force[i] = 0.0;
  }
  iter[w] = 0;
}
#if LIBERO_NEWTON_SCALAR_IO
static __device__ void copy_qacc(int w, int *active, double *src, double *dst) {
  if (active[w])
    for (int j = 0; j < SubstepShape::nv; ++j)
      dst[w * SubstepShape::nv + j] = src[w * SubstepShape::nv + j];
}
#else
static __device__ void copy_qacc(int w, int *active, double *src, double *dst) {
  if (active[w]) {
    if (src == dst || n_state_disjoint(src + w * SubstepShape::nv, SubstepShape::nv * sizeof(double),
                                       dst + w * SubstepShape::nv, SubstepShape::nv * sizeof(double)))
      libero_newton_io::copy_serial(src + w * SubstepShape::nv, dst + w * SubstepShape::nv, SubstepShape::nv);
    else
      for (int j = 0; j < SubstepShape::nv; ++j)
        dst[w * SubstepShape::nv + j] = src[w * SubstepShape::nv + j];
  }
}
#endif // LIBERO_NEWTON_SCALAR_IO
static __device__ void copy_cost(int w, int *active, double *src, double *dst) {
  if (active[w])
    dst[w] = src[w];
}

// Collective copy with no internal barriers. The caller synchronizes before
// consuming the result or reusing its storage. Shifted aliases retain the
// original forward scalar copy; identical spans need no writes.
template <class T> static __device__ __forceinline__ void ordered_copy_block(const T *src, T *dst, int n) {
  if (src == dst || n <= 0)
    return;
  const uintptr_t a = reinterpret_cast<uintptr_t>(src), b = reinterpret_cast<uintptr_t>(dst);
  const size_t bytes = size_t(n) * sizeof(T);
  if (a <= b ? b - a >= bytes : a - b >= bytes)
    libero_newton_io::copy_block(src, dst, size_t(n));
  else if (threadIdx.x == 0)
    for (int j = 0; j < n; ++j)
      dst[j] = src[j];
}
static __device__ void copy_qacc_block(int w, int *active, double *src, double *dst) {
  if (active[w])
    for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x)
      dst[w * SubstepShape::nv + i] = src[w * SubstepShape::nv + i];
}
static __device__ void direction_commit_block(int w, int *active, double *candidate, double *search) {
  if (active[w])
    for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x)
      search[w * SubstepShape::nv + i] = candidate[w * SubstepShape::nv + i];
}
static __device__ void save_old_block(int w, int *active, int *n, int *state, double *cost, int *old,
                                      double *oldcost, int rows) {
  if (!active[w])
    return;
  ordered_copy_block(state + w * rows, old + w * rows, n[w]);
  if (threadIdx.x == 0)
    oldcost[w] = cost[w];
}
#if LIBERO_NEWTON_SCALAR_IO
static __device__ void zero_rows(int w, int *active, int *n, double *out, int rows) {
  if (active[w])
    for (int j = 0; j < n[w]; ++j)
      out[w * rows + j] = 0.0;
}
#else
static __device__ void zero_rows(int w, int *active, int *n, double *out, int rows) {
  if (active[w])
    libero_newton_io::fill_serial(out + w * rows, n[w], 0.0);
}
#endif // LIBERO_NEWTON_SCALAR_IO
static __device__ void solver_iter_reset(int w, int *active, int *iter) {
  if (active[w])
    iter[w] = 0;
}
static __device__ void finalize_exit(int w, int *newton, int *error, int *exit, int *active) {
  exit[w] = active[w] = (newton[w] != 0 && error[w] == 0) ? 1 : 0;
}
#if LIBERO_NEWTON_TRACE
static __device__ void seed_eval_block(int w, double *eval, int *tag) {
  libero_newton_io::fill_block(tag + w * 52, 52, -101);
  libero_newton_io::fill_block(eval + w * 208, 208, 97.0);
}
#endif
static __device__ void tape_ls_input(int w, int slot, int slots, double *quad0, double *tape) {
  if (slot < slots)
    tape[w * slots + slot] = quad0[w];
}
static __device__ void parallel_tape_ls_buffers(int tid, int slot, int slots, int rows, double *mv,
                                                double *jv, double *quad, double *qg, double *tm, double *tj,
                                                double *tq, double *tg, int nv) {
  if (slot >= slots)
    return;
  int stride = nv + 4 * rows + 3, w = tid / stride, i = tid % stride;
  if (i < nv)
    tm[(w * slots + slot) * nv + i] = mv[w * nv + i];
  else if (i < nv + rows)
    tj[(w * slots + slot) * rows + i - nv] = jv[w * rows + i - nv];
  else if (i < nv + 4 * rows)
    tq[(w * slots + slot) * rows * 3 + i - nv - rows] = quad[w * rows * 3 + i - nv - rows];
  else
    tg[(w * slots + slot) * 3 + i - nv - 4 * rows] = qg[w * 3 + i - nv - 4 * rows];
}
// An inactive source slot repeats these same buffers, including capacity tails.
// Load each source cell once and fan it out; never omit observable history cells.
static __device__ void inactive_tape_ls_buffers(int begin, int end, int slots, int rows, double *mv,
                                                double *jv, double *quad, double *qg, double *tm, double *tj,
                                                double *tq, double *tg, int nv) {
#if LIBERO_NEWTON_SCALAR_IO
  if (end > slots)
    end = slots;
  int stride = nv + 4 * rows + 3;
  for (int i = threadIdx.x; i < stride; i += blockDim.x) {
    double value;
    double *dst;
    int width, cell;
    if (i < nv) {
      value = mv[i];
      dst = tm;
      width = nv;
      cell = i;
    } else if (i < nv + rows) {
      cell = i - nv;
      value = jv[cell];
      dst = tj;
      width = rows;
    } else if (i < nv + 4 * rows) {
      cell = i - nv - rows;
      value = quad[cell];
      dst = tq;
      width = 3 * rows;
    } else {
      cell = i - nv - 4 * rows;
      value = qg[cell];
      dst = tg;
      width = 3;
    }
    for (int slot = begin; slot < end; ++slot)
      dst[slot * width + cell] = value;
#else
  if (end > slots)
    end = slots;
  // Each packet stays within one history field and each destination slot.
  // Capacity tails are exported even after the world becomes inactive.
  for (int field = 0; field < 4; ++field) {
    const double *src = field == 0 ? mv : (field == 1 ? jv : (field == 2 ? quad : qg));
    double *dst = field == 0 ? tm : (field == 1 ? tj : (field == 2 ? tq : tg));
    const int width = field == 0 ? nv : (field == 1 ? rows : (field == 2 ? 3 * rows : 3));
    int i = 2 * int(threadIdx.x);
    for (; i + 1 < width; i += 2 * int(blockDim.x)) {
      const double2 value = libero_vec::load_pair(src + i);
      for (int slot = begin; slot < end; ++slot)
        libero_vec::store_pair(dst + slot * width + i, value.x, value.y);
    }
    if (i < width) {
      const double value = src[i];
      for (int slot = begin; slot < end; ++slot)
        dst[slot * width + i] = value;
    }
#endif // LIBERO_NEWTON_SCALAR_IO
  }
}
static __device__ void tape_commit(int w, int slot, int slots, int *active, double *alpha, double *slope,
                                   double *lineslope, double *improvement, double *gradient, double *cost,
                                   double *oldcost, double *quad0, int *LSiter, int *LSresult, int *committed,
                                   int *route, int *nupdate, int *nactive, int *ncone, int *nchange,
                                   int *iteration, int *reason, int *error, int *s07, int *s08b,
                                   int *stage_status, int *s08c, double *tf, int *ti, int *ts, double *eval,
                                   int *tag, double *te, int *tt, bool export_trace = true) {
  if (slot >= slots)
    return;
  int f = (w * slots + slot) * 8, i = (w * slots + slot) * 12, s = (w * slots + slot) * 6;
  tf[f] = alpha[w];
  tf[f + 1] = slope[w];
  tf[f + 2] = lineslope[w];
  tf[f + 3] = improvement[w];
  tf[f + 4] = gradient[w];
  tf[f + 5] = cost[w];
  tf[f + 6] = oldcost[w];
  tf[f + 7] = quad0[w];
  ti[i] = active[w];
  ti[i + 1] = LSiter[w];
  ti[i + 2] = LSresult[w];
  ti[i + 3] = committed[w];
  ti[i + 4] = route[w];
  ti[i + 5] = nupdate[w];
  ti[i + 6] = nactive[w];
  ti[i + 7] = ncone[w];
  ti[i + 8] = nchange[w];
  ti[i + 9] = iteration[w];
  ti[i + 10] = reason[w];
  ti[i + 11] = error[w];
  ts[s] = s07[w];
  ts[s + 1] = s08b[w];
  ts[s + 2] = stage_status[w];
  ts[s + 3] = s08c[w];
  if (export_trace)
    for (int e = 0; e < 52; ++e) {
      for (int c = 0; c < 4; ++c)
        te[((w * slots + slot) * 52 + e) * 4 + c] = eval[(w * 52 + e) * 4 + c];
      tt[(w * slots + slot) * 52 + e] = tag[w * 52 + e];
    }
}
// Collective trace export; leader retains the small mixed scalar record.
// Caller enters uniformly and supplies its publication barrier. Status tail
// cells ts[4:6] remain untouched exactly as the scalar history writer does.
static __device__ void tape_commit_block(int w, int slot, int slots, int *active, double *alpha,
                                         double *slope, double *lineslope, double *improvement,
                                         double *gradient, double *cost, double *oldcost, double *quad0,
                                         int *LSiter, int *LSresult, int *committed, int *route, int *nupdate,
                                         int *nactive, int *ncone, int *nchange, int *iteration, int *reason,
                                         int *error, int *s07, int *s08b, int *stage_status, int *s08c,
                                         double *tf, int *ti, int *ts, double *eval, int *tag, double *te,
                                         int *tt) {
  if (slot >= slots)
    return;
  if (threadIdx.x == 0)
    tape_commit(w, slot, slots, active, alpha, slope, lineslope, improvement, gradient, cost, oldcost, quad0,
                LSiter, LSresult, committed, route, nupdate, nactive, ncone, nchange, iteration, reason,
                error, s07, s08b, stage_status, s08c, tf, ti, ts, eval, tag, te, tt, false);
#if LIBERO_NEWTON_SCALAR_IO
  for (int e = threadIdx.x; e < 52 * 4; e += blockDim.x)
    te[(w * slots + slot) * 52 * 4 + e] = eval[w * 52 * 4 + e];
  for (int e = threadIdx.x; e < 52; e += blockDim.x)
    tt[(w * slots + slot) * 52 + e] = tag[w * 52 + e];
#else
  libero_newton_io::copy_block(eval + w * 52 * 4, te + (w * slots + slot) * 52 * 4, 52 * 4);
  libero_newton_io::copy_block(tag + w * 52, tt + (w * slots + slot) * 52, 52);
#endif // LIBERO_NEWTON_SCALAR_IO
}
static __device__ void validate_entry_counts(int w, int *n, int *ne, int *nf, int *nc, int *empty,
                                             int *newton, int *active, int *error, int *item, int rows,
                                             int contacts) {
  int bad = -1;
  if (n[w] < 0 || n[w] > rows)
    bad = 0;
  else if (nc[w] < 0 || nc[w] > contacts)
    bad = 3;
  else if (ne[w] < 0 || ne[w] > n[w])
    bad = 1;
  else if (nf[w] < 0 || nf[w] > n[w] - ne[w])
    bad = 2;
  if (bad >= 0) {
    if (!error[w]) {
      error[w] = 4002;
      item[w] = bad;
    }
    empty[w] = newton[w] = active[w] = 0;
  }
}
static __device__ void validate_contact_rows(int w, int *n, int *ne, int *nf, int *nc, int *typ, int *cid,
                                             int *rd, double *rm, double *rf, int *dim, double *mu,
                                             double *fr, int *newton, int *active, int *error, int *item,
                                             int rows, int contacts) {
  if (!newton[w] || error[w])
    return;
  int rb = w * rows, cb = w * contacts;
  for (int i = ne[w] + nf[w]; i < n[w];) {
    int t = typ[rb + i];
    if (t == 7) {
      int c = cid[rb + i];
      if (c < 0 || c >= nc[w])
        return;
      int d = dim[cb + c];
      double m = mu[cb + c];
      if ((d != 1 && d != 3 && d != 4 && d != 6) || i + d > n[w] || m <= 0 || !isfinite(m))
        return;
      for (int j = 0; j < d; ++j)
        if (typ[rb + i + j] != 7 || cid[rb + i + j] != c)
          return;
      for (int j = 0; j < d - 1; ++j)
        if (fr[(cb + c) * 5 + j] <= 0 || !isfinite(fr[(cb + c) * 5 + j]))
          return;
      i += d;
    } else if (t >= 0 && t <= 6)
      ++i;
    else
      return;
  }
  for (int i = ne[w] + nf[w]; i < n[w];) {
    int r = rb + i;
    if (typ[r] != 7) {
      ++i;
      continue;
    }
    int c = cb + cid[r], d = dim[c];
    bool bad = rd[r] != d || rm[r] != mu[c];
    for (int j = 0; j < d - 1; ++j)
      if (rf[r * 5 + j] != fr[c * 5 + j])
        bad = true;
    if (bad) {
      error[w] = 8001;
      item[w] = i;
      active[w] = newton[w] = 0;
      return;
    }
    i += d;
  }
}
static __device__ void final_event(int w, double *q, int *empty, int *newton, int *exit, int *error,
                                   int *status, int64_t *attempted, double *candidate, int64_t *stamp,
                                   int *completed, int *origin, int *stage, int *rejected, int nv) {
  completed[w] = origin[w] = stage[w] = rejected[w] = 0;
  stamp[w] = attempted[w];
  if (empty[w] == 1 && !error[w]) {
    completed[w] = origin[w] = stage[w] = 1;
  } else if (newton[w] == 1 && exit[w] == 1) {
    stage[w] = 2;
    if (!error[w] && !status[w])
      completed[w] = origin[w] = 1;
    else if (status[w] == 3)
      rejected[w] = 1;
  }
  if (completed[w] || rejected[w])
    for (int j = 0; j < nv; ++j)
      candidate[w * nv + j] = q[w * nv + j];
}
static __device__ void publish(int w, const int *upstream, const int *enabled, int *error, int *valid,
                               int *written) {
  if (enabled[w] && !upstream[w] && !error[w])
    valid[w] = written[w] = 1;
}
static __device__ void publication_gate(int w, int *valid, int *diagnostic) {
  if (diagnostic[w])
    valid[w] = 0;
}
static __device__ void finish_publication(int w, int *valid, int *written, int64_t *expected,
                                          int64_t *epoch) {
  written[w] = valid[w];
  if (valid[w])
    epoch[w] = expected[w];
}

// Parse dimensions in order, rather than inferring groups from cid changes:
// adjacent groups may reference the same contact. No row metadata is inspected
// by a lane until the entire schema has passed the original ordered guard.

// Lane 0 only. No block synchronization or payload reads by other lanes occur
// until the caller publishes this entry and the prepared head lists together.
static __device__ bool prepare_newton_entry(const NewtonWorldData &p, NewtonControl &c,
                                            bool reuse_layout) {
  CF_NEWTON_PROFILE_SCOPE(EntrySchema);
  validate_entry_counts(0, p.nefc, &c.ne(), &c.nf(), &c.ncon(), &c.empty(), &c.newton_world(), &c.active(),
                        &c.err_code(), &c.error_item(), SubstepShape::rows, SubstepShape::contacts);
  if (!c.newton_world() || c.err_code())
    return false;
  if (reuse_layout) {
    c.invariant_valid() = 1;
    return true;
  }
  *reinterpret_cast<int *>(p.precomputed + NewtonCoefficientLayout::layout_valid) = 0;
  const int n = c.count(), eq = c.ne(), fc = c.nf();
  int sizes[7] = {}, nh = 0;
  for (int r = 0; r < n;) {
    const bool contact = r >= eq + fc && p.efc_type[r] == 7;
    int d = 1;
    if (contact) {
      const int id = p.efc_id[r];
      if (id < 0 || id >= c.ncon())
        return false;
      d = p.contact_dim[id];
      const double mu = p.contact_mu[id];
      if ((d != 3 && d != 4 && d != 6) || r + d > n || mu <= 0 || !isfinite(mu))
        return false;
      for (int j = 0; j < d; ++j)
        if (p.efc_type[r + j] != 7 || p.efc_id[r + j] != id)
          return false;
      for (int j = 0; j < d - 1; ++j)
        if (p.contact_friction[id * 5 + j] <= 0 || !isfinite(p.contact_friction[id * 5 + j]))
          return false;
    } else if (r >= eq + fc && (p.efc_type[r] < 0 || p.efc_type[r] > 6))
      return false;
    p.constraint_group_heads[r] = r;
    for (int j = 1; j < d; ++j)
      p.constraint_group_heads[r + j] = -1;
    p.constraint_head_rows[nh++] = r;
    const int bucket =
        r < eq ? 0 : (r < eq + fc ? 1 : (!contact ? 2 : (d == 1 ? 3 : (d == 3 ? 4 : (d == 4 ? 5 : 6)))));
    ++sizes[bucket];
    r += d;
  }
  p.constraint_head_count[0] = nh;
  int cursor[7];
  p.constraint_typed_head_offsets[0] = 0;
  for (int bucket = 0; bucket < 7; ++bucket) {
    cursor[bucket] = p.constraint_typed_head_offsets[bucket];
    p.constraint_typed_head_offsets[bucket + 1] = cursor[bucket] + sizes[bucket];
  }
  for (int k = 0; k < nh; ++k) {
    const int r = p.constraint_head_rows[k];
    const bool contact = r >= eq + fc && p.efc_type[r] == 7;
    const int d = contact ? p.contact_dim[p.efc_id[r]] : 1;
    const int bucket =
        r < eq ? 0 : (r < eq + fc ? 1 : (!contact ? 2 : (d == 1 ? 3 : (d == 3 ? 4 : (d == 4 ? 5 : 6)))));
    p.constraint_typed_head_rows[cursor[bucket]++] = r;
  }
  c.invariant_valid() = 1;
  return true;
}

// Warp 0 only compares published contact heads. Other warps can independently
// produce tile offsets, then all lanes rendezvous at the caller's exit barrier.
static __device__ void validate_contact_metadata_warp(const NewtonWorldData &p, NewtonControl &c) {
  if (threadIdx.x >= 32 || !c.invariant_valid())
    return;
  int first = 2147483647;
  for (int k = threadIdx.x; k < p.constraint_head_count[0]; k += 32) {
    const int r = p.constraint_head_rows[k];
    if (r < c.ne() + c.nf() || p.efc_type[r] != 7)
      continue;
    const int id = p.efc_id[r], d = p.contact_dim[id];
    bool bad = p.cone_dim[r] != d || p.cone_mu[r] != p.contact_mu[id];
    for (int j = 0; j < d - 1; ++j)
      bad |= p.cone_friction[r * 5 + j] != p.contact_friction[id * 5 + j];
    if (bad)
      first = min(first, r);
  }
  first = libero_warp::min(first);
  if (threadIdx.x == 0 && first != 2147483647) {
    c.err_code() = 8001;
    c.error_item() = first;
    c.active() = c.newton_world() = 0;
  }
}

namespace libero_cuda {
static __device__ __forceinline__ double square(double x) { return x * x; }
static __device__ __forceinline__ double clamp(double x, double lo, double hi) {
  return fmin(hi, fmax(lo, x));
}
static __device__ __forceinline__ double dot3(const double *a, const double *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static __device__ __forceinline__ void cross3(const double *a, const double *b, double *out) {
  double x = a[1] * b[2] - a[2] * b[1], y = a[2] * b[0] - a[0] * b[2];
  double z = a[0] * b[1] - a[1] * b[0];
  out[0] = x;
  out[1] = y;
  out[2] = z;
}
static __device__ __forceinline__ void copy3(const double *x, double *out) {
  const double a = x[0], b = x[1], c = x[2];
  out[0] = a;
  out[1] = b;
  out[2] = c;
}
// Quaternions use MuJoCo's (w,x,y,z) convention; rotate3 assumes unit length.
static __device__ __forceinline__ void quat_mul(const double *a, const double *b, double *out) {
  double q[4] = {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                 a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                 a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                 a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
  for (int i = 0; i < 4; ++i)
    out[i] = q[i];
}
static __device__ __forceinline__ void quat_rotate3(const double *q, const double *v, double *out) {
  double uv[3], uuv[3];
  cross3(q + 1, v, uv);
  cross3(q + 1, uv, uuv);
  for (int i = 0; i < 3; ++i)
    out[i] = v[i] + 2 * (q[0] * uv[i] + uuv[i]);
}
static __device__ __forceinline__ double dot6(const double *a, const double *b) {
  return dot3(a, b) + dot3(a + 3, b + 3);
}
// Spatial motion vectors are angular then linear, as in MuJoCo cvel.
static __device__ __forceinline__ void cross_motion(const double *a, const double *b, double *out) {
  double angular[3], linear[3], extra[3];
  cross3(a, b, angular);
  cross3(a, b + 3, linear);
  cross3(a + 3, b, extra);
  for (int i = 0; i < 3; ++i) {
    out[i] = angular[i];
    out[i + 3] = linear[i] + extra[i];
  }
}
static __device__ __forceinline__ void cross_force(const double *a, const double *b, double *out) {
  double angular[3], linear[3], extra[3];
  cross3(a, b, angular);
  cross3(a + 3, b + 3, extra);
  cross3(a, b + 3, linear);
  for (int i = 0; i < 3; ++i) {
    out[i] = angular[i] + extra[i];
    out[i + 3] = linear[i];
  }
}
static __device__ __forceinline__ int lane() { return threadIdx.x; }
static __device__ __forceinline__ int width() { return blockDim.x; }
} // namespace libero_cuda



// A block-local C = sum_r (J_r^T D_r) J_r. Only admitted quadratic rows
// enter the dense BLAS tile. Input packing and sparse M expansion are hand code.

// Dense reconstruction from the admitted lower factor. Upper-triangle owner
// bits are ignored. This makes the batch cone path independent of the current
// source matrix/factor lifetime and keeps base H available for the next update.
static __device__ void hessian_from_factor_blas_dx(const double *factor, double *H, double *workspace,
                                                   int *algorithm_diagnostic) {
  double *C = workspace;
  double *A = C + SubstepShape::nv * SubstepShape::nv;
  double *B = A + SubstepShape::nv * flash_libero::NewtonLayout::hessian_tile_rows;
  libero_newton_io::fill_block(C, SubstepShape::nv * SubstepShape::nv, 0.0);
  __syncthreads();
  for (int begin = 0; begin < SubstepShape::nv; begin += flash_libero::NewtonLayout::hessian_tile_rows) {
    for (int z = threadIdx.x; z < SubstepShape::nv * flash_libero::NewtonLayout::hessian_tile_rows;
         z += blockDim.x) {
      const int i = z / flash_libero::NewtonLayout::hessian_tile_rows,
                k = begin + z % flash_libero::NewtonLayout::hessian_tile_rows;
      const double v = k < SubstepShape::nv && k <= i ? factor[i * SubstepShape::nv + k] : 0.0;
      A[z] = v;
      B[(z % flash_libero::NewtonLayout::hessian_tile_rows) * SubstepShape::nv + i] = v;
    }
    __syncthreads();
    dx_hessian_gemm(A, B, C);
    if (threadIdx.x == 0 && algorithm_diagnostic)
      ++algorithm_diagnostic[3];
  }
  for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x)
    H[z] = z / SubstepShape::nv >= z % SubstepShape::nv ? C[z] : factor[z];
  __syncthreads();
}

// A pending U tile is a separate owner: cone POTRF/L^T J scratch may overwrite
// workspace between appends. Only this call consumes/clears the pending tile.
static __device__ void cone_gram_append_blas_dx(const double *pending, int used, double *H, double *workspace,
                                                int *algorithm_diagnostic) {
  double *C = workspace;
  double *A = C + SubstepShape::nv * SubstepShape::nv;
  double *B = A + SubstepShape::nv * flash_libero::NewtonLayout::hessian_tile_rows;
  libero_newton_io::fill_block(C, SubstepShape::nv * SubstepShape::nv, 0.0);
  for (int z = threadIdx.x; z < SubstepShape::nv * flash_libero::NewtonLayout::hessian_tile_rows;
       z += blockDim.x) {
    const int k = z / SubstepShape::nv, j = z % SubstepShape::nv;
    const double v = k < used ? pending[z] : 0.0;
    B[z] = v;
    A[j * flash_libero::NewtonLayout::hessian_tile_rows + k] = v;
  }
  __syncthreads();
  dx_hessian_gemm(A, B, C);
  for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x)
    if (z / SubstepShape::nv >= z % SubstepShape::nv)
      H[z] += C[z];
  if (threadIdx.x == 0 && algorithm_diagnostic)
    ++algorithm_diagnostic[2];
  __syncthreads();
}

static __device__ void hessian_rebuild_blas_dx(int *count, double *J, double *D, int *state, double *qm,
                                               int *madr, int *parent, int *simple, double *H,
                                               double *workspace, int *diagnostic, double *atomic_scratch,
                                               jacobian_cache::View cache) {
#if FLASH_NEWTON_ATOMIC_REDUCTIONS
  static_assert(NewtonLayout::mathdx_cells >= atomic_reduction::gram_block_cells(SubstepShape::nv),
                "Sparse Gram must fit the existing MathDx arena");
  if (reduction_policy.cache_enabled && cache.valid())
    jacobian_cache::gram(cache, D, state, count[0], atomic_scratch, workspace);
  else
    atomic_reduction::gram_block(J, D, state, count[0], SubstepShape::nv, atomic_scratch,
                                 workspace, NewtonLayout::mathdx_cells);
  for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x) {
    const int i = z / SubstepShape::nv, j = z % SubstepShape::nv;
    if (j > i) continue;
    const int adr = mass_index(i, j, madr, parent, simple);
    // Preserve Gram -> add M -> mirror, including the existing cone stage.
    const double value = atomic_scratch[i * (i + 1) / 2 + j] + (adr >= 0 ? qm[adr] : 0.0);
    H[z] = value;
    if (i != j) H[j * SubstepShape::nv + i] = value;
  }
  if (threadIdx.x == 0) {
    ++diagnostic[0];
    // Logical source row-tile count, also retained by native GEMV diagnostics.
    diagnostic[1] += (count[0] + NewtonLayout::hessian_tile_rows - 1) / NewtonLayout::hessian_tile_rows;
  }
  __syncthreads();
#else
  double *C = workspace;
  double *A = C + SubstepShape::nv * SubstepShape::nv;
  double *B = A + SubstepShape::nv * flash_libero::NewtonLayout::hessian_tile_rows;
  for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x)
    C[z] = 0;
  // The first A/B publication also publishes C. With zero live rows the
  // reconstruction below reads each C cell in its original producer lane.
  const int lane = libero_warp::lane(), warp = libero_warp::warp();
  constexpr int warps = SubstepShape::threads / 32;
  for (int begin = 0; begin < count[0]; begin += flash_libero::NewtonLayout::hessian_tile_rows) {
    // One warp owns one J row. Broadcast its scalar parameters in registers;
    // there is no row_D/row_admitted shared producer/consumer boundary.
    for (int k = warp; k < flash_libero::NewtonLayout::hessian_tile_rows; k += warps) {
      const int r = begin + k;
      double d = 0.0;
      if (lane == 0 && r < count[0] && state[r] == 1)
        d = D[r];
      d = __shfl_sync(libero_warp::mask, d, 0);
      const bool admitted = d != 0.0;
      // Masked rows may legally hold nonfinite J and must not be read.
#if LIBERO_NEWTON_SCALAR_IO
      for (int i = lane; i < SubstepShape::nv; i += 32) {
        const double v = admitted ? J[r * SubstepShape::nv + i] : 0.0;
        A[i * flash_libero::NewtonLayout::hessian_tile_rows + k] = admitted ? v * d : 0.0;
        B[k * SubstepShape::nv + i] = v;
      }
#else
      for (int i = 2 * lane; i < SubstepShape::nv; i += 64) {
        if (i + 1 < SubstepShape::nv) {
          const double2 v = admitted ? libero_vec::load_pair(J + r * SubstepShape::nv + i) : make_double2(0, 0);
          A[i * flash_libero::NewtonLayout::hessian_tile_rows + k] = admitted ? v.x * d : 0.0;
          A[(i + 1) * flash_libero::NewtonLayout::hessian_tile_rows + k] = admitted ? v.y * d : 0.0;
          libero_vec::store_pair(B + k * SubstepShape::nv + i, v.x, v.y);
        } else {
          const double v = admitted ? J[r * SubstepShape::nv + i] : 0.0;
          A[i * flash_libero::NewtonLayout::hessian_tile_rows + k] = admitted ? v * d : 0.0;
          B[k * SubstepShape::nv + i] = v;
        }
      }
#endif // LIBERO_NEWTON_SCALAR_IO
    }
    __syncthreads();
    dx_hessian_gemm(A, B, C);
    if (threadIdx.x == 0)
      ++diagnostic[1];
  }
  for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x) {
    int i = z / SubstepShape::nv, j = z % SubstepShape::nv;
    if (j > i)
      continue;
    int adr = mass_index(i, j, madr, parent, simple);
    // Add sparse M after Gram summation, following the original source.
    double v = C[z] + (adr >= 0 ? qm[adr] : 0);
    H[z] = v;
    if (i != j)
      H[j * SubstepShape::nv + i] = v;
  }
  if (threadIdx.x == 0)
    ++diagnostic[0];
  __syncthreads();
#endif
}

static __device__ void hessian_assemble_blas_dx(int w, int *count, int *active, int kind, double *J,
                                                double *D, int *state, double *qm, int *madr, int *parent,
                                                int *simple, double *H, int *status, int rows,
                                                double *workspace, int *diagnostic, double *atomic_scratch,
                                               jacobian_cache::View cache) {
  // All inputs are already sliced to one world by the persistent wrapper.
  // The admitted solve caller needs no extra exit barrier: rebuild publishes H.
  if (!active[0] || kind != 1 || status[0])
    return;
  hessian_rebuild_blas_dx(count, J, D, state, qm, madr, parent, simple, H, workspace, diagnostic, atomic_scratch, cache);
}

// Collective contract: every exit closes the admission-flag read lifetime;
// active calls also publish results and release staging. No caller exit barrier.
static __device__ void jacvec_sub_blas_dx(int w, int *active, int *count, double *J, double *q, double *aref,
                                          double *out, int rows, double *workspace, int *diagnostic) {
  if (!active[0]) {
    __syncthreads();
    return;
  }
  double *A = workspace;
  double *B = A + 16 * SubstepShape::nv;
  double *C = B + SubstepShape::nv;
#if LIBERO_NEWTON_SCALAR_IO
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
    B[j] = q[j];
#else
  // Copies guard the actual shared subsegment, which may be only 8B aligned.
  libero_newton_io::copy_block(q, B, SubstepShape::nv);
#endif // LIBERO_NEWTON_SCALAR_IO
  __syncthreads();
  for (int begin = 0; begin < count[0]; begin += 16) {
#if LIBERO_NEWTON_SCALAR_IO
    for (int z = threadIdx.x; z < 16 * SubstepShape::nv; z += blockDim.x) {
      int r = begin + z / SubstepShape::nv;
      A[z] = r < count[0] ? J[r * SubstepShape::nv + z % SubstepShape::nv] : 0;
#else
    constexpr int row_pairs = (SubstepShape::nv + 1) / 2;
    for (int z = threadIdx.x; z < 16 * row_pairs; z += blockDim.x) {
      const int k = z / row_pairs, i = 2 * (z % row_pairs), r = begin + k;
      if (i + 1 < SubstepShape::nv) {
        const double2 v =
            r < count[0] ? libero_vec::load_pair(J + r * SubstepShape::nv + i) : make_double2(0, 0);
        libero_vec::store_pair(A + k * SubstepShape::nv + i, v.x, v.y);
      } else
        A[k * SubstepShape::nv + i] = r < count[0] ? J[r * SubstepShape::nv + i] : 0;
#endif // LIBERO_NEWTON_SCALAR_IO
    }
#if LIBERO_NEWTON_SCALAR_IO
    for (int r = threadIdx.x; r < 16; r += blockDim.x)
      C[r] = 0;
#else
    libero_newton_io::fill_block(C, 16, 0.0);
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
    dx_newton_jv(A, B, C);
#if LIBERO_NEWTON_SCALAR_IO
    for (int r = threadIdx.x; r < 16; r += blockDim.x)
      if (begin + r < count[0])
        out[begin + r] = C[r] - aref[begin + r];
#else
    for (int p = threadIdx.x; p < 8; p += blockDim.x) {
      const int r = 2 * p;
      if (begin + r + 1 < count[0]) {
        const double2 c = libero_vec::load_pair(C + r);
        const double2 a = libero_vec::load_pair(aref + begin + r);
        libero_vec::store_pair(out + begin + r, c.x - a.x, c.y - a.y);
      } else if (begin + r < count[0])
        out[begin + r] = C[r] - aref[begin + r];
    }
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
  }
  if (threadIdx.x == 0)
    ++diagnostic[3];
  __syncthreads();
}

// Two columns: smooth and warmstart. The J tile is staged once for both.
static __device__ void jacvec_candidates_mathdx(const int *active, const int *count, const double *J,
                                                 const double *smooth, const double *warm, const double *aref,
                                                 double *smooth_residual, double *warm_residual,
                                                 double *workspace, int *diagnostic) {
  if (!active[0]) {
    __syncthreads();
    return;
  }
  double *A = workspace, *B = A + 16 * SubstepShape::nv, *C = B + 2 * SubstepShape::nv;
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x) {
    B[2 * j] = smooth[j];
    B[2 * j + 1] = warm[j];
  }
  __syncthreads();
  for (int begin = 0; begin < count[0]; begin += 16) {
    for (int z = threadIdx.x; z < 16 * SubstepShape::nv; z += blockDim.x) {
      const int r = begin + z / SubstepShape::nv;
      A[z] = r < count[0] ? J[r * SubstepShape::nv + z % SubstepShape::nv] : 0.0;
    }
    // Keep C initialized: the installed Dx epilogue may stage its old cells
    // even though beta=0 discards their values.
    for (int z = threadIdx.x; z < 32; z += blockDim.x)
      C[z] = 0.0;
    __syncthreads();
    dx_newton_candidates(A, B, C); // publishes C and releases A/B readers
    for (int i = threadIdx.x; i < 16; i += blockDim.x)
      if (begin + i < count[0]) {
        const double reference = aref[begin + i];
        smooth_residual[begin + i] = C[2 * i] - reference;
        warm_residual[begin + i] = C[2 * i + 1] - reference;
      }
    __syncthreads(); // closes C readers before the next tile reuses it
  }
  if (threadIdx.x == 0)
    diagnostic[3] += 2;
  __syncthreads();
}

// Shape-specialized subwarps share each J load between the two candidates.
static __device__ void jacvec_candidates_blas_dx(const int *active, const int *count, const double *J,
                                                 const double *smooth, const double *warm, const double *aref,
                                                 double *smooth_residual, double *warm_residual,
                                                 double *workspace, int *diagnostic) {
#if FLASH_NEWTON_NATIVE_GEMV
  if (active[0]) {
    constexpr int lanes = SubstepShape::nv <= 32 ? 8 : 16;
    gemv::candidates<SubstepShape::nv,lanes>(J,smooth,warm,aref,count[0],smooth_residual,warm_residual);
    if (threadIdx.x == 0) diagnostic[3] += 2;
  }
  __syncthreads();
#else
  jacvec_candidates_mathdx(active,count,J,smooth,warm,aref,smooth_residual,warm_residual,workspace,diagnostic);
#endif
}

// Same collective exit contract as jacvec_sub_blas_dx, including zero rows.
static __device__ void constraint_transpose_mathdx(int w, int *active, int *count, double *J, double *force,
                                                    double *qfrc, int rows, int nv,
                                                    double *workspace, int *diagnostic) {
  // The caller can immediately change active after this call. Even a no-op
  // exit must finish both warps' flag reads before that next writer runs.
  if (!active[0]) {
    __syncthreads();
    return;
  }
  double *C = workspace;
  double *A = C + SubstepShape::nv;
  double *B = A + SubstepShape::nv * flash_libero::NewtonLayout::hessian_tile_rows;
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
    C[j] = 0;
  __syncthreads();
  for (int begin = 0; begin < count[0]; begin += flash_libero::NewtonLayout::hessian_tile_rows) {
    // Load the force tile once. The block-wide vote also publishes B
    // before any lane reads it. No Jacobian load precedes this decision.
    bool nonzero = false;
    for (int k = threadIdx.x; k < flash_libero::NewtonLayout::hessian_tile_rows; k += blockDim.x) {
      const int r = begin + k;
      const double f = r < count[0] ? force[r] : 0;
      B[k] = f;
      nonzero |= f != 0;
    }
    // Preserve row/tile order. A zero tile contributes no arithmetic;
    // importantly, even NaN Jacobian rows remain completely unread.
    if (!__syncthreads_or(nonzero))
      continue;
#if LIBERO_NEWTON_SCALAR_IO
    for (int z = threadIdx.x; z < SubstepShape::nv * flash_libero::NewtonLayout::hessian_tile_rows;
         z += blockDim.x) {
#else
    constexpr int row_pairs = (SubstepShape::nv + 1) / 2;
    for (int z = threadIdx.x; z < row_pairs * flash_libero::NewtonLayout::hessian_tile_rows;
         z += blockDim.x) {
#endif // LIBERO_NEWTON_SCALAR_IO                                                                                      \
       // Read global J in its contiguous [k][j] order, transposing only                                               \
       // this block-local tile into the existing compact MathDx A.
#if LIBERO_NEWTON_SCALAR_IO
      const int k = z / SubstepShape::nv, j = z % SubstepShape::nv, r = begin + k;
#else
      const int k = z / row_pairs, j = 2 * (z % row_pairs), r = begin + k;
#endif // LIBERO_NEWTON_SCALAR_IO
      const double f = B[k];
      // Zero force intentionally ignores its corresponding Jacobian row.
#if LIBERO_NEWTON_SCALAR_IO
      A[j * flash_libero::NewtonLayout::hessian_tile_rows + k] = f != 0 ? J[r * SubstepShape::nv + j] : 0;
#else
      if (j + 1 < SubstepShape::nv) {
        const double2 v = f != 0 ? libero_vec::load_pair(J + r * SubstepShape::nv + j) : make_double2(0, 0);
        A[j * flash_libero::NewtonLayout::hessian_tile_rows + k] = v.x;
        A[(j + 1) * flash_libero::NewtonLayout::hessian_tile_rows + k] = v.y;
      } else
        A[j * flash_libero::NewtonLayout::hessian_tile_rows + k] = f != 0 ? J[r * SubstepShape::nv + j] : 0;
#endif // LIBERO_NEWTON_SCALAR_IO
    }
    __syncthreads();
    dx_newton_transpose(A, B, C);
  }
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
    qfrc[j] = C[j];
  if (threadIdx.x == 0)
    ++diagnostic[4];
  __syncthreads();
}

// Small live ranges accumulate columns directly. Larger ranges and NV<=32
// partition rows across warps, then reduce only warps*NV private partials.
static __device__ void constraint_transpose_blas_dx(int w, int *active, int *count, double *J, double *force,
                                                    double *qfrc, int rows, int nv,
                                                    double *workspace, int *diagnostic, double *atomic_scratch,
                                               jacobian_cache::View cache) {
#if FLASH_NEWTON_ATOMIC_REDUCTIONS
  if (active[0]) {
    const TransposeMethod method = reduction_policy.transpose;
    const bool cached = reduction_policy.cache_enabled && cache.valid();
    if (cached && method >= TransposeMethod::Csc1) {
      const int lanes = method == TransposeMethod::Csc1 ? 1 : method == TransposeMethod::Csc8 ? 8 : 4;
      jacobian_cache::transpose(cache, J, force, count[0], qfrc, lanes);
    } else if (method == TransposeMethod::Atomic) {
      atomic_reduction::transpose(J, force, count[0], SubstepShape::nv, qfrc, atomic_scratch);
    } else {
      auto selected = transpose_reduction::Method::Direct;
      if (method == TransposeMethod::Shared32) selected = transpose_reduction::Method::Shared32;
      else if (method == TransposeMethod::Shared16) selected = transpose_reduction::Method::Shared16;
      else if (method == TransposeMethod::Shared8) selected = transpose_reduction::Method::Shared8;
      else if (method == TransposeMethod::Auto && blockDim.x >= 128 && count[0] >= 128)
        selected = transpose_reduction::Method::Shared16;
      transpose_reduction::dense(J, force, count[0], SubstepShape::nv, qfrc, workspace, selected);
    }
    if (threadIdx.x == 0) ++diagnostic[4];
  }
  __syncthreads(); // publish cooperative output copies, including skipped calls
#elif FLASH_NEWTON_NATIVE_GEMV
  if (active[0]) {
    gemv::transpose<SubstepShape::nv>(J,force,count[0],qfrc,workspace);
    if (threadIdx.x == 0) ++diagnostic[4];
  }
  __syncthreads(); // publishes output and closes active/staging readers
#else
  constraint_transpose_mathdx(w,active,count,J,force,qfrc,rows,nv,workspace,diagnostic);
#endif
}

static __device__ void cone_ltj_blas_dx(int begin, int dim, double *J, double *local, double *ltj,
                                        double *workspace, int *diagnostic) {
  double *C = workspace;
  double *A = C + 6 * SubstepShape::nv;
  double *B = A + 36;
  for (int z = threadIdx.x; z < 36; z += blockDim.x) {
    int r = z / 6, k = z % 6;
    A[z] = (r < dim && k < dim && k >= r) ? local[k * dim + r] : 0;
  }
  for (int z = threadIdx.x; z < 6 * SubstepShape::nv; z += blockDim.x) {
    int r = z / SubstepShape::nv;
    B[z] = r < dim ? J[(begin + r) * SubstepShape::nv + z % SubstepShape::nv] : 0;
    C[z] = 0;
  }
  __syncthreads();
  dx_newton_cone_ltj(A, B, C);
  for (int z = threadIdx.x; z < dim * SubstepShape::nv; z += blockDim.x)
    ltj[z] = C[z];
  if (threadIdx.x == 0)
    ++diagnostic[5];
  __syncthreads();
}

// Both calls are collective block operations; matrix and RHS are shared memory.
#if LIBERO_NEWTON_ISLANDS
namespace libero_islands {
static __device__ bool direction_with_forward(const double *, const double *, double *, double *, double *,
                                              int *);
}
#endif

static __device__ void direction_blas_dx(int w, double *Ma, double *smooth, double *constraint, double *H,
                                         double *HC, int *ncone, int *nefc, int *active, int *upstream,
                                         double *grad, double *Mgrad, double *candidate, int *status,
                                         double *scratch_grad, double *x, double *fwd, double *workspace,
                                         int *blas_diag, bool base_shared = false, bool cone_shared = false,
                                         int *islands_diag = nullptr) {
  const int lane = threadIdx.x;
  const int vb = w * SubstepShape::nv;
  const double *L = (ncone[w] > 0 ? HC : H) + w * SubstepShape::nv * SubstepShape::nv;
  bool resident = ncone[w] > 0 ? cone_shared : base_shared;
  double *lower = resident ? const_cast<double *>(L) : workspace;
  // Each Dx operand begins at a 16-byte aligned address. An odd NV gives
  // NV*NV FP64 cells an 8-byte remainder; pad only the staging subview.
  // The factor itself retains compact row-major lda=NV.
  constexpr int matrix_cells = (SubstepShape::nv * SubstepShape::nv + 1) & ~1;
  double *rhs = workspace + (resident ? 0 : matrix_cells);
  __shared__ int first_bad;
  if (lane == 0) {
    status[w] = upstream[w];
    if (!status[w] && active[w]) {
      if (active[w] != 1)
        status[w] = 1;
      else if (nefc[w] < 0 || ncone[w] < 0 || ncone[w] > nefc[w])
        status[w] = 2;
    }
  }
  __syncthreads();
  if (status[w] || !active[w])
    return;
  if (lane < 32) {
    bool bad = false, pivot = false;
    for (int i = lane; i < SubstepShape::nv; i += 32) {
      bad |= !isfinite(Ma[vb + i]) || !isfinite(smooth[vb + i]) || !isfinite(constraint[vb + i]);
      pivot |= L[i * SubstepShape::nv + i] == 0;
    }
    for (int z = lane; z < SubstepShape::nv * SubstepShape::nv; z += 32)
      if (z % SubstepShape::nv <= z / SubstepShape::nv)
        bad |= !isfinite(L[z]);
    bad = libero_warp::any(bad);
    pivot = libero_warp::any(pivot);
    if (lane == 0)
      status[w] = bad ? 3 : (pivot ? 4 : 0);
  }
  __syncthreads();
  if (status[w])
    return;
  if (lane < 32) {
    int bad = libero_warp::no_index;
    for (int i = lane; i < SubstepShape::nv; i += 32) {
      double v = (Ma[vb + i] - smooth[vb + i]) - constraint[vb + i];
      if (!isfinite(v) && i < bad)
        bad = i;
    }
    bad = libero_warp::min(bad);
    if (lane == 0) {
      first_bad = bad;
      if (bad != libero_warp::no_index)
        status[w] = 5;
    }
  }
  __syncthreads();
  // Preserve the original diagnostic prefix, including on overflow failure.
  for (int i = lane; i < SubstepShape::nv; i += blockDim.x)
    if (i < first_bad) {
      double v = (Ma[vb + i] - smooth[vb + i]) - constraint[vb + i];
      scratch_grad[vb + i] = v;
      x[vb + i] = v;
    }
  __syncthreads();
  if (status[w])
    return;
#if LIBERO_NEWTON_ISLANDS
  // The full gradient/entry contract above is unchanged. Rebuild exact current
  // factor islands now: rank updates may have introduced new coupling since
  // the preceding factor call. Success publishes both forward/final scratch;
  // failure holds them and can safely use the source full-NV diagnostic route.
  const bool island_solved =
      libero_islands::direction_with_forward(L, x + vb, x + vb, fwd + vb, workspace, islands_diag);
  if (island_solved) {
    if (lane == 0)
      ++blas_diag[2];
    for (int i = lane; i < SubstepShape::nv; i += blockDim.x) {
      grad[vb + i] = scratch_grad[vb + i];
      Mgrad[vb + i] = x[vb + i];
      candidate[vb + i] = copysign(x[vb + i], -copysign(1.0, x[vb + i]));
    }
    __syncthreads();
    return;
  }
#endif
  if (!resident)
    for (int z = lane; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x)
      lower[z] = z % SubstepShape::nv <= z / SubstepShape::nv ? L[z] : 0.0;
  for (int i = lane; i < SubstepShape::nv; i += blockDim.x)
    rhs[i] = x[vb + i];
  if (lane == 0)
    ++blas_diag[2];
  __syncthreads();
  dx_direction_forward(lower, rhs);
  __syncthreads();
  if (lane < 32) {
    int bad = libero_warp::no_index;
    for (int i = lane; i < SubstepShape::nv; i += 32)
      if (!isfinite(rhs[i]) && i < bad)
        bad = i;
    bad = libero_warp::min(bad);
    if (lane == 0) {
      first_bad = bad;
      if (bad != libero_warp::no_index)
        status[w] = 5;
    }
  }
  __syncthreads();
  for (int i = lane; i < SubstepShape::nv; i += blockDim.x)
    if (i < first_bad) {
      x[vb + i] = rhs[i];
      fwd[vb + i] = rhs[i];
    }
  __syncthreads();
  if (status[w])
    return;
  dx_direction_backward(lower, rhs);
  __syncthreads();
  if (lane < 32) {
    int bad = libero_warp::no_index;
    // Reverse solve commits a suffix; reduce reverse ordinal rather than index.
    for (int i = lane; i < SubstepShape::nv; i += 32)
      if (!isfinite(rhs[i]) && SubstepShape::nv - 1 - i < bad)
        bad = SubstepShape::nv - 1 - i;
    bad = libero_warp::min(bad);
    if (lane == 0) {
      first_bad = bad;
      if (bad != libero_warp::no_index)
        status[w] = 5;
    }
  }
  __syncthreads();
  for (int i = lane; i < SubstepShape::nv; i += blockDim.x)
    if (SubstepShape::nv - 1 - i < first_bad)
      x[vb + i] = rhs[i];
  __syncthreads();
  if (status[w])
    return;
  for (int i = lane; i < SubstepShape::nv; i += blockDim.x) {
    grad[vb + i] = scratch_grad[vb + i];
    Mgrad[vb + i] = x[vb + i];
    candidate[vb + i] = copysign(x[vb + i], -copysign(1.0, x[vb + i]));
  }
  __syncthreads();
}

// CPU CGupdateGradient with the existing cooperative FP64 triangular solves.
// Checked direction_blas_dx above remains available for standalone diagnostics.
static __device__ void direction_core_blas_dx(unsigned char *shared, const NewtonWorldData &p,
                                              double *smooth_force, bool publish_search = true) {
  double *shared_mass_acceleration = shared_at<double>(shared, NewtonLayout::mass_acceleration);
  double *shared_constraint_force = shared_at<double>(shared, NewtonLayout::constraint_force);
  double *shared_hessian = shared_at<double>(shared, NewtonLayout::hessian);
  double *shared_gradient = shared_at<double>(shared, NewtonLayout::gradient);
  double *shared_search = shared_at<double>(shared, NewtonLayout::search);
  double *shared_mathdx = shared_at<double>(shared, NewtonLayout::mathdx);
  auto &ctl = newton_control(shared);
  // The qualified solve calls this collective only after successful Hessian
  // factorization. It produces no status code or leader-written run flag.
  const int vb = 0 * SubstepShape::nv;
  const double *factor =
      (ctl.ncone() > 0 ? p.Hcone : shared_hessian) + 0 * SubstepShape::nv * SubstepShape::nv;
  const bool resident = ctl.ncone() > 0 ? false : true;
  constexpr int matrix_cells = (SubstepShape::nv * SubstepShape::nv + 1) & ~1;
  double *lower = resident ? const_cast<double *>(factor) : shared_mathdx;
  double *rhs = shared_mathdx + (resident ? 0 : matrix_cells);
  if (!resident)
    for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x)
      lower[z] = z % SubstepShape::nv <= z / SubstepShape::nv ? factor[z] : 0.0;
  for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x) {
    const double g =
        (shared_mass_acceleration[vb + i] - smooth_force[vb + i]) - shared_constraint_force[vb + i];
    shared_gradient[vb + i] = g;
    rhs[i] = g;
  }
  if (threadIdx.x == 0)
    ++p.blas_diagnostic[2];
  __syncthreads();
  dx_direction_potrs(lower, rhs);
  __syncthreads();
  for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x) {
    const double value = rhs[i];
    shared_search[vb + i] = copysign(value, -copysign(1.0, value));
  }
  // The initial direction is consumed immediately by line search. Iteration
  // statistics consume only the earlier gradient; their exit barrier can
  // publish search without a separate rendezvous here.
  if (publish_search)
    __syncthreads();
}

// Project-written collective evaluation of the retained MuJoCo S07 equations.
// All block threads enter; shared cells and global owners are explicit.
// group_head is a world-local map: a head stores its row and a follower stores -1.
static_assert(flash_libero::SubstepShape::threads >= 32 && flash_libero::SubstepShape::threads <= 1024 &&
                  flash_libero::SubstepShape::threads % 32 == 0,
              "S07 collectives require complete warps");
static __device__ P s07_eval_point_block(unsigned char *shared, const NewtonModelData &model,
                                         const NewtonWorldData &world, const double *smooth_force, int w,
                                         int rows, int nefc, int ne, int nf, double alpha, int tag,
                                         int *niter, const int *group_head,
                                         const int *compact_heads = nullptr, int head_count = 0,
                                         const double *head_mu = nullptr) {
  CF_NEWTON_PROFILE_SCOPE(SearchPoint);
  auto &ctl = newton_control(shared);
  __shared__ double partial[6][(flash_libero::SubstepShape::threads + 31) / 32];
  __shared__ P point;
  __shared__ int next_iter;
  if (!group_head && !compact_heads) {
    if (threadIdx.x == 0) {
      point = eval_point(shared, model, world, smooth_force, w, rows, nefc, ne, nf, alpha, tag, niter);
      next_iter = *niter;
    }
    __syncthreads();
    P result = point;
    *niter = next_iter;
    __syncthreads();
    return result;
  }
  double t0 = 0, t1 = 0, t2 = 0, cost = 0, d0 = 0, d1 = 0;
  const int base = w * rows, qbase = w * rows * 3;
  // Immutable compact plan is produced once per invocation, not per alpha.
  const int groups = compact_heads ? head_count : nefc;
  // Each lane consumes its own stable head directly. There is no block-wide
  // staging dependency between independent groups; preserve per-lane rank and
  // summation order, and retain the existing warp/block reduction below.
  for (int rank = threadIdx.x; rank < groups; rank += blockDim.x) {
    const int i = compact_heads ? compact_heads[rank] : rank;
    if (!compact_heads && group_head[i] != i)
      continue;
    const double *q = world.quad + qbase + 3 * i;
    bool add = false;
    if (i < ne)
      add = true;
    else if (i < ne + nf) {
      const double start = world.jar[base + i], dir = world.Jv[base + i];
      const double x = start + alpha * dir, f = world.loss[base + i], Rf = row_friction_limit(world, base + i);
      if (-Rf < x && x < Rf)
        add = true;
      else if (x <= -Rf) {
        t0 += f * (-0.5 * Rf - start);
        t1 += -f * dir;
      } else {
        t0 += f * (-0.5 * Rf + start);
        t1 += f * dir;
      }
    } else if (world.efc_type[base + i] == 7) {
      const double mu = head_mu && compact_heads ? head_mu[rank] : world.cone_mu[base + i];
#if LIBERO_NEWTON_SCALAR_IO
      const double U0 = q[3], V0 = q[4], UU = q[5], UV = q[6], VV = q[7], Dm = q[8];
#else
      // Cone heads own all three quad records (nine doubles). Guard
      // alignment on each actual pair; never read a follower as a head.
      const double2 uv0 = libero_vec::load_pair(q + 3);
      const double2 uuv = libero_vec::load_pair(q + 5);
      const double2 vdm = libero_vec::load_pair(q + 7);
      const double U0 = uv0.x, V0 = uv0.y, UU = uuv.x, UV = uuv.y, VV = vdm.x, Dm = vdm.y;
#endif // LIBERO_NEWTON_SCALAR_IO
      const double N = U0 + alpha * V0, Tsqr = UU + alpha * (2.0 * UV + alpha * VV);
      if (Tsqr <= 0) {
        if (N < 0)
          add = true;
      } else {
        const double T = __dsqrt_rn(Tsqr);
        if (N >= mu * T) {
        } else if (mu * N + T <= 0)
          add = true;
        else {
          const double N1 = V0, T1 = __ddiv_rn(UV + alpha * VV, T);
          const double T2 = __ddiv_rn(VV, T) - __ddiv_rn((UV + alpha * VV) * T1, T * T);
          cost += 0.5 * Dm * (N - mu * T) * (N - mu * T);
          d0 += Dm * (N - mu * T) * (N1 - mu * T1);
          d1 += Dm * ((N1 - mu * T1) * (N1 - mu * T1) + (N - mu * T) * (-mu * T2));
        }
      }
    } else if (world.jar[base + i] + alpha * world.Jv[base + i] < 0)
      add = true;
#if LIBERO_NEWTON_SCALAR_IO
    if (add) {
      t0 += q[0];
      t1 += q[1];
      t2 += q[2];
    }
#else
    if (add) {
      // Load only an admitted record, preserving skipped-term behavior.
      const double2 q01 = libero_vec::load_pair(q);
      t0 += q01.x;
      t1 += q01.y;
      t2 += q[2];
    }
#endif // LIBERO_NEWTON_SCALAR_IO
  }
  double values[6] = {t0, t1, t2, cost, d0, d1};
  const int lane = libero_warp::lane(), warp = libero_warp::warp();
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    const double value = libero_warp::sum_to_leader(values[k]);
    if (lane == 0)
      partial[k][warp] = value;
  }
  __syncthreads();
  // Each warp consumes the same small partial array and keeps its P/niter in
  // registers. No leader-written shared P needs a separate publication.
#pragma unroll
  for (int k = 0; k < 6; ++k)
    values[k] = libero_warp::sum(lane < (blockDim.x + 31) / 32 ? partial[k][lane] : 0.0);
  // Close every warp's partial reads before the next alpha reuses the array.
  __syncthreads();
  t0 = (&ctl.quad0())[w] + values[0];
  t1 = world.quadGauss[w * 3 + 1] + values[1];
  t2 = world.quadGauss[w * 3 + 2] + values[2];
  cost = values[3] + (alpha * alpha * t2 + alpha * t1 + t0);
  d0 = values[4] + (2.0 * alpha * t2 + t1);
  d1 = values[5] + 2.0 * t2;
  if (d1 <= 0)
    d1 = 1e-15;
  if (threadIdx.x == 0) {
    if (!isfinite(alpha) || !isfinite(cost) || !isfinite(d0) || !isfinite(d1))
      (&ctl.s07_status())[w] = 5;
#if LIBERO_NEWTON_TRACE
    const int idx = *niter;
    world.eval_trace[(w * 52 + idx) * 4] = alpha;
    world.eval_trace[(w * 52 + idx) * 4 + 1] = cost;
    world.eval_trace[(w * 52 + idx) * 4 + 2] = d0;
    world.eval_trace[(w * 52 + idx) * 4 + 3] = d1;
    world.eval_tag[w * 52 + idx] = tag;
#endif
  }
  ++*niter;
  return {alpha, cost, d0, d1};
}

static __device__ int s07_update_bracket_block(unsigned char *shared, const NewtonModelData &model,
                                               const NewtonWorldData &world, const double *smooth_force,
                                               int w, int rows, int nefc, int ne, int nf, P *p,
                                               const P *candidates, P *next, int *niter, int tag,
                                               const int *group_head, const int *compact_heads = nullptr,
                                               int head_count = 0, const double *head_mu = nullptr) {
  int flag = 0;
  for (int i = 0; i < 3; ++i) {
    if (p->d0 < 0 && candidates[i].d0 < 0 && p->d0 < candidates[i].d0) {
      *p = candidates[i];
      flag = 1;
    } else if (p->d0 > 0 && candidates[i].d0 > 0 && p->d0 > candidates[i].d0) {
      *p = candidates[i];
      flag = 2;
    }
  }
  // p and candidates are broadcasts of leader evaluations, so this branch is
  // uniform. Trace order and the 50-evaluation limit remain source-ordered.
  if (flag)
    *next = s07_eval_point_block(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                                 p->alpha - __ddiv_rn(p->d0, p->d1), tag, niter, group_head, compact_heads,
                                 head_count, head_mu);
  return flag;
}

static __device__ void s07_finish_block(unsigned char *shared, const NewtonModelData &model,
                                        const NewtonWorldData &world, const double *smooth_force, int w,
                                        double alpha, int niter, int result, double slope) {
  if (threadIdx.x == 0)
    finish(shared, model, world, smooth_force, w, alpha, niter, result, slope);
}

// Handwritten S07 device adapter. Domain preparation and line-search retain
// scalar packed-mass ownership; row groups and line-search evaluation are collective.
// The line-search helpers access shared offsets and prepared global owners directly.

// Warp 0 computes up to three independent dot products, one eight-lane group
// per product. Four lanes retain mju_dot's ordered accumulator streams; lane 4
// handles its 1/2/3-element tail. Only the group leader writes the result.
// No shared storage or barrier: the caller publishes the outputs once.
template <int Products>
static __device__ void search_dots_warp(const double *x, const double *y0, const double *y1, const double *y2,
                                        int n, double *out) {
  static_assert(Products >= 1 && Products <= 3, "at most three dot products");
  if (threadIdx.x >= 32)
    return;
  const int group = threadIdx.x / 8, lane = threadIdx.x % 8;
  const int end = n - n % 4;
  double partial = 0.0;
  if (group < Products) {
    const double *y = group == 0 ? y0 : (group == 1 ? y1 : y2);
    if (lane < 4 && end) {
      partial = x[lane] * y[lane];
      for (int i = lane + 4; i < end; i += 4)
        partial += x[i] * y[i];
    } else if (lane == 4) {
      const int tail = n - end;
      if (tail == 3)
        partial = x[end] * y[end] + x[end + 1] * y[end + 1] + x[end + 2] * y[end + 2];
      else if (tail == 2)
        partial = x[end] * y[end] + x[end + 1] * y[end + 1];
      else if (tail == 1)
        partial = x[end] * y[end];
    }
  }
  const double r0 = __shfl_sync(0xffffffffu, partial, 0, 8);
  const double r1 = __shfl_sync(0xffffffffu, partial, 1, 8);
  const double r2 = __shfl_sync(0xffffffffu, partial, 2, 8);
  const double r3 = __shfl_sync(0xffffffffu, partial, 3, 8);
  const double tail = __shfl_sync(0xffffffffu, partial, 4, 8);
  if (lane == 0 && group < Products) {
    double value = (r0 + r2) + (r1 + r3);
    if (n != end)
      value += tail;
    out[group] = value;
  }
}

// Only quantities depending on this search direction need shared storage.
struct S07PreparedDx {
  double gtol, slopescl, search_norm;
};

// A uniform register result replaces the shared ready flag. The DOF norm's
// producer is lane 0, also the sole root/scale producer; one publication suffices.
static __device__ bool s07_prepare_dx(unsigned char *shared, const NewtonModelData &model,
                                      const NewtonWorldData &world, const double *smooth_force,
                                      S07PreparedDx *prepared) {
  auto &ctl = newton_control(shared);
  const bool active = ctl.active() != 0;
  if (!active)
    return false;
  if (threadIdx.x == 0)
    ctl.s07_status() = 0;
  double *v = shared_at<double>(shared, NewtonLayout::search);
  double *result = shared_at<double>(shared, NewtonLayout::mass_search);
  search_dots_warp<1>(v, v, nullptr, nullptr, SubstepShape::nv, &prepared->search_norm);
  if (threadIdx.x == 0) {
    const double snorm = __dsqrt_rn(prepared->search_norm);
    prepared->search_norm = snorm;
    if (!isfinite(snorm))
      ctl.s07_status() = 5;
    else {
      world.quadGauss[0] = ctl.quad0();
      if (snorm < 1e-15)
        finish(shared, model, world, smooth_force, 0, 0.0, 0, 1, 1.0);
      else {
        const double mean = model.model_scalars[1];
        prepared->gtol = model.model_scalars[0] * 0.01 * snorm * mean * SubstepShape::nv;
        prepared->slopescl = __ddiv_rn(1.0, snorm * mean * SubstepShape::nv);
        if (!isfinite(prepared->gtol) || !isfinite(prepared->slopescl))
          ctl.s07_status() = 5;
      }
    }
  }
  __syncthreads(); // publish the norm and scales, not a separate ready flag
  const bool ready = ctl.s07_status() == 0 && prepared->search_norm >= 1e-15;
  if (!ready)
    return false;
  // Model plans and private shared operands are qualified once at binding.
  // No per-direction schema/overlap scan is needed for these owned arrays.
  if (model.mass_plan_offsets)
    libero_packed_mass::multiply_plan(v, world.qM, model.mass_plan_offsets,
        model.mass_plan_coefficient, model.mass_plan_source, model.mass_plan_assign,
        result, SubstepShape::nv);
  else
    libero_packed_mass::multiply_gather(v, world.qM, model.madr, model.parentid,
        model.simplenum, result, SubstepShape::nv);
  bool bad_product = false;
  for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x)
    bad_product |= !isfinite(result[i]);
  const bool bad = __syncthreads_or(bad_product);
  if (threadIdx.x == 0 && bad)
    ctl.s07_status() = 5;
  return !bad; // identical in every thread; caller owns final status publication
}

static __device__ double s07_tile_jv(int row, int begin, const double *tile, const double *Jv) {
  return row >= begin && row < begin + 16 ? tile[row - begin] : Jv[row];
}

// Consume the just-produced Dx output while it still resides in shared memory.
// A cone crossing a tile boundary waits until its last row is available; only
// its earlier (at most five) rows then need the global Jv mirror.
static __device__ bool s07_quad_group_values_dx(unsigned char *shared, const NewtonModelData &model,
                                                const NewtonWorldData &world, const double *smooth_force,
                                                int w, int i, int rows, int begin, const double *tile,
                                                double *q, int rank = -1, const int *head_dim = nullptr,
                                                const double *head_mu = nullptr,
                                                const double *head_friction = nullptr) {
  int base = w * rows;
  const double *Jv = world.Jv + base;
  double v = s07_tile_jv(i, begin, tile, Jv), j = world.jar[base + i], D = world.D[base + i];
  double DJ0 = D * j;
  q[0] = j * DJ0;
  q[1] = v * DJ0;
  q[2] = v * D * v;
  bool elliptic = world.efc_type[base + i] == 7;
  if (elliptic) {
    int dim = head_dim && rank >= 0 ? head_dim[rank] : world.cone_dim[base + i];
    double mu = head_mu && rank >= 0 ? head_mu[rank] : world.cone_mu[base + i];
    for (int k = 1; k < dim; ++k) {
      double jk = world.jar[base + i + k], vk = s07_tile_jv(i + k, begin, tile, Jv),
             Dk = world.D[base + i + k];
      double DJk = Dk * jk;
      q[0] += jk * DJk;
      q[1] += vk * DJk;
      q[2] += vk * Dk * vk;
    }
    double U0 = j * mu, V0 = v * mu, UU = 0, UV = 0, VV = 0;
    for (int k = 1; k < dim; ++k) {
      double f = head_friction && rank >= 0 ? head_friction[(k - 1) * rows + rank]
                                            : world.cone_friction[(base + i) * 5 + k - 1];
      double U = world.jar[base + i + k] * f, V = s07_tile_jv(i + k, begin, tile, Jv) * f;
      UU += U * U;
      UV += U * V;
      VV += V * V;
    }
    q[3] = U0;
    q[4] = V0;
    q[5] = UU;
    q[6] = UV;
    q[7] = VV;
    q[8] = coefficient_plane(world.precomputed, NewtonCoefficientLayout::cone_dm)[i];
  }
  q[0] *= 0.5;
  q[2] *= 0.5;
#pragma unroll
  for (int k = 0; k < 9; ++k)
    if (k < (elliptic ? 9 : 3) && !isfinite(q[k]))
      return false;
  return true;
}

static __device__ bool s07_quad_group_dx(unsigned char *shared, const NewtonModelData &model,
                                         const NewtonWorldData &world, const double *smooth_force, int w,
                                         int i, int rows, int begin, const double *tile) {
  return s07_quad_group_values_dx(shared, model, world, smooth_force, w, i, rows, begin, tile,
                                  world.quad + w * rows * 3 + 3 * i);
}

// At most 21 potential heads: the 16-row tile plus five preceding rows.
// Validate before ordered-prefix commit, preserving the original failed group's
// full private quad write while keeping all later groups held.
static __device__ void s07_quad_tile_block(unsigned char *shared, const NewtonModelData &model,
                                           const NewtonWorldData &world, const double *smooth_force, int w,
                                           int rows, int begin, int produced, const double *tile,
                                           const int *group_head, const int *compact_heads = nullptr,
                                           const int *tile_head_offsets = nullptr,
                                           const int *head_dim = nullptr, const double *head_mu = nullptr,
                                           const double *head_friction = nullptr) {
  auto &ctl = newton_control(shared);
  __shared__ int first_bad;
  // Offsets assign each group to the tile containing its final row.
  // Every head appears once; cross-tile cones retain their earlier Jv mirror.
  const bool planned = compact_heads && tile_head_offsets;
  const int tile_index = begin / 16;
  const int first = planned ? tile_head_offsets[tile_index] : (begin > 5 ? begin - 5 : 0);
  const int end = planned ? tile_head_offsets[tile_index + 1] : 0;
  const int rank = first + threadIdx.x;
  const int i = planned ? (rank < end ? compact_heads[rank] : -1) : rank;
  double q[9];
  int width = 0;
  bool owned = false, valid = true;
  if (i >= 0 && i < produced && (planned || group_head[i] == i)) {
    const int dim = world.efc_type[w * rows + i] == 7 ? world.cone_dim[w * rows + i] : 1;
    if (i + dim > begin && i + dim <= produced) {
      owned = true;
      width = dim == 1 ? 3 : 9;
      valid = s07_quad_group_values_dx(shared, model, world, smooth_force, w, i, rows, begin, tile, q,
                                       planned ? rank : -1, head_dim, head_mu, head_friction);
    }
  }
  if (libero_warp::warp() == 0) {
    const int bad = libero_warp::first_index(owned && !valid, i);
    if (libero_warp::lane() == 0)
      first_bad = bad;
  }
  __syncthreads();
  if (owned && i <= first_bad) {
#if LIBERO_NEWTON_SCALAR_IO
#else
    // Pair only within this head's owned quad record; its odd tail stays scalar.
#endif // LIBERO_NEWTON_SCALAR_IO
#pragma unroll
#if LIBERO_NEWTON_SCALAR_IO
    for (int k = 0; k < 9; ++k)
      if (k < width) {
        world.quad[w * rows * 3 + 3 * i + k] = q[k];
#else
    for (int k = 0; k < 8; k += 2)
      if (k + 1 < width) {
        libero_vec::store_pair(world.quad + w * rows * 3 + 3 * i + k, q[k], q[k + 1]);
#endif // LIBERO_NEWTON_SCALAR_IO
      }
#if LIBERO_NEWTON_SCALAR_IO
#else
    world.quad[w * rows * 3 + 3 * i + width - 1] = q[width - 1];
#endif // LIBERO_NEWTON_SCALAR_IO
  }
  __syncthreads();
  if (threadIdx.x == 0 && first_bad != libero_warp::no_index)
    (&ctl.s07_status())[w] = 5;
  __syncthreads();
}

static __device__ void
s07_jv_blas_dx(unsigned char *shared, const NewtonModelData &model, const NewtonWorldData &world,
               const double *smooth_force, int w, int nefc, int rows, double *workspace, int *diagnostic,
               bool prepare_quad = false, const int *group_head = nullptr, const int *compact_heads = nullptr,
               const int *tile_head_offsets = nullptr, const int *head_dim = nullptr,
               const double *head_mu = nullptr, const double *head_friction = nullptr) {
  auto &ctl = newton_control(shared);
  double *shared_search = shared_at<double>(shared, NewtonLayout::search);
#if FLASH_NEWTON_NATIVE_GEMV
  double *C = workspace;
#else
  double *A = workspace;
  double *B = A + 16 * SubstepShape::nv;
  double *C = B + SubstepShape::nv;
#endif
  const double *v = shared_search + w * SubstepShape::nv;
  const double *J = world.J + w * rows * SubstepShape::nv;
  double *Jv = world.Jv + w * rows;
  __shared__ int next_group;
  if (threadIdx.x == 0)
    next_group = 0;
#if !FLASH_NEWTON_NATIVE_GEMV
#if LIBERO_NEWTON_SCALAR_IO
  for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
    B[j] = v[j];
#else
  libero_newton_io::copy_block(v, B, SubstepShape::nv);
#endif // LIBERO_NEWTON_SCALAR_IO
  __syncthreads();
#endif
  for (int begin = 0; begin < nefc; begin += 16) {
#if FLASH_NEWTON_NATIVE_GEMV
    constexpr int lanes = SubstepShape::nv <= 32 ? 8 : 16;
    gemv::tile<SubstepShape::nv,lanes>(J,v,begin,nefc,C);
    __syncthreads();
#else
#if LIBERO_NEWTON_SCALAR_IO
    for (int z = threadIdx.x; z < 16 * SubstepShape::nv; z += blockDim.x) {
      int r = begin + z / SubstepShape::nv;
      A[z] = r < nefc ? J[r * SubstepShape::nv + z % SubstepShape::nv] : 0;
#else
    constexpr int row_pairs = (SubstepShape::nv + 1) / 2;
    // Pair only inside valid rows, including when an odd NV shifts alignment.
    for (int z = threadIdx.x; z < 16 * row_pairs; z += blockDim.x) {
      const int k = z / row_pairs, i = 2 * (z % row_pairs), r = begin + k;
      if (i + 1 < SubstepShape::nv) {
        const double2 pair =
            r < nefc ? libero_vec::load_pair(J + r * SubstepShape::nv + i) : make_double2(0, 0);
        libero_vec::store_pair(A + k * SubstepShape::nv + i, pair.x, pair.y);
      } else
        A[k * SubstepShape::nv + i] = r < nefc ? J[r * SubstepShape::nv + i] : 0;
#endif // LIBERO_NEWTON_SCALAR_IO
    }
#if LIBERO_NEWTON_SCALAR_IO
    for (int r = threadIdx.x; r < 16; r += blockDim.x)
      C[r] = 0;
#else
    libero_newton_io::fill_block(C, 16, 0.0);
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
    dx_newton_jv(A, B, C);
#endif
    // The admitted tile plan assigns each cone to its last-row tile. A tile
    // spans 16 rows plus at most five earlier rows, so <=21 heads fit warp0.
    // Keep producer, mirror, validation and quad commit in this loop: only
    // C's publication above and its reader-release below need CTA barriers.
    if (prepare_quad && group_head && compact_heads && tile_head_offsets) {
      if (threadIdx.x < 32) {
        CF_NEWTON_PROFILE_SCOPE(SearchQuadratic);
        const int lane = libero_warp::lane();
        const int produced = min(begin + 16, nefc), live = produced - begin;
        if (lane < live)
          Jv[begin + lane] = C[lane];
        // Preserve the old failure boundary: mirror the whole live tile, but
        // do not write any quad if any of its produced Jv values is nonfinite.
        const bool bad_jv = libero_warp::any(lane < live && !isfinite(C[lane]));
        if (bad_jv) {
          if (lane == 0)
            ctl.s07_status() = 5;
        } else {
          const int tile_index = begin / 16;
          const int rank = tile_head_offsets[tile_index] + lane;
          const int end = tile_head_offsets[tile_index + 1];
          const int i = rank < end ? compact_heads[rank] : -1;
          double q[9];
          int width = 0;
          bool owned = false, valid = true;
          if (i >= 0 && i < produced) {
            const int dim = world.efc_type[w * rows + i] == 7 ? world.cone_dim[w * rows + i] : 1;
            if (i + dim > begin && i + dim <= produced) {
              owned = true;
              width = dim == 1 ? 3 : 9;
              valid = s07_quad_group_values_dx(shared, model, world, smooth_force, w, i, rows,
                  begin, C, q, rank, head_dim, head_mu, head_friction);
            }
          }
          const int first_bad = libero_warp::first_index(owned && !valid, i);
          // Keep the failed head's full record and all preceding records;
          // later heads retain their old values, as in the ordered adapter.
          if (owned && i <= first_bad) {
#if LIBERO_NEWTON_SCALAR_IO
            for (int k = 0; k < width; ++k)
              world.quad[w * rows * 3 + 3 * i + k] = q[k];
#else
            for (int k = 0; k + 1 < width; k += 2)
              libero_vec::store_pair(world.quad + w * rows * 3 + 3 * i + k, q[k], q[k + 1]);
            world.quad[w * rows * 3 + 3 * i + width - 1] = q[width - 1];
#endif
          }
          if (lane == 0 && first_bad != libero_warp::no_index)
            ctl.s07_status() = 5;
        }
      }
      // Publish quad/status and earlier Jv rows for crossing cones; no lane
      // may overwrite C for the next tile until warp0 has finished reading it.
      __syncthreads();
      if (ctl.s07_status())
        return;
      continue;
    }
    // Generic callers without the admitted compact plan retain the original
    // collective adapter and its broader synchronization contract.
#if LIBERO_NEWTON_SCALAR_IO
    for (int r = threadIdx.x; r < 16; r += blockDim.x)
      if (begin + r < nefc)
        Jv[begin + r] = C[r];
#else
    const int produced = begin + 16 < nefc ? 16 : nefc - begin;
    libero_newton_io::copy_block(C, Jv + begin, produced);
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
    if (prepare_quad && group_head) {
      __shared__ int bad_jv;
      if (libero_warp::warp() == 0) {
        int r = libero_warp::lane();
        bool bad = r < 16 && begin + r < nefc && !isfinite(C[r]);
        bool any = libero_warp::any(bad);
        if (r == 0)
          bad_jv = any;
      }
      __syncthreads();
      if (bad_jv) {
        if (threadIdx.x == 0)
          (&ctl.s07_status())[w] = 5;
      } else {
        CF_NEWTON_PROFILE_SCOPE(SearchQuadratic);
        s07_quad_tile_block(shared, model, world, smooth_force, w, rows, begin,
                            begin + 16 < nefc ? begin + 16 : nefc, C, group_head, compact_heads,
                            tile_head_offsets, head_dim, head_mu, head_friction);
      }
    } else if (prepare_quad && threadIdx.x == 0) {
      for (int r = 0; r < 16 && begin + r < nefc; ++r)
        if (!isfinite(C[r])) {
          (&ctl.s07_status())[w] = 5;
          break;
        }
      int produced = begin + 16 < nefc ? begin + 16 : nefc;
      while (!(&ctl.s07_status())[w] && next_group < produced) {
        int i = next_group, dim = world.efc_type[w * rows + i] == 7 ? world.cone_dim[w * rows + i] : 1;
        if (i + dim > produced)
          break;
        if (!s07_quad_group_dx(shared, model, world, smooth_force, w, i, rows, begin, C)) {
          (&ctl.s07_status())[w] = 5;
          break;
        }
        next_group += dim;
      }
    }
    __syncthreads();
    if (prepare_quad && (&ctl.s07_status())[w])
      return; // uniform, after the barrier
  }
  if (threadIdx.x == 0)
    ++diagnostic[6];
  __syncthreads();
}

static __device__ void s07_quad_scalar_dx(unsigned char *shared, const NewtonModelData &model,
                                          const NewtonWorldData &world, const double *smooth_force, int w,
                                          int nefc, int rows, int base, const double *Jv) {
  auto &ctl = newton_control(shared);
  double *qbase = world.quad + w * rows * 3;
  for (int i = 0; i < nefc; ++i) {
    double *q = qbase + 3 * i;
    bool elliptic = world.efc_type[base + i] == 7;
    double DJ0 = world.D[base + i] * world.jar[base + i];
    q[0] = world.jar[base + i] * DJ0;
    q[1] = Jv[i] * DJ0;
    q[2] = Jv[i] * world.D[base + i] * Jv[i];
    if (elliptic) {
      int dim = world.cone_dim[base + i];
      double mu = world.cone_mu[base + i];
      for (int j = 1; j < dim; ++j) {
        double DJj = world.D[base + i + j] * world.jar[base + i + j];
        q[0] += world.jar[base + i + j] * DJj;
        q[1] += Jv[i + j] * DJj;
        q[2] += Jv[i + j] * world.D[base + i + j] * Jv[i + j];
      }
      double U0 = world.jar[base + i] * mu, V0 = Jv[i] * mu, UU = 0, UV = 0, VV = 0;
      for (int j = 1; j < dim; ++j) {
        double U = world.jar[base + i + j] * world.cone_friction[(base + i) * 5 + j - 1];
        double V = Jv[i + j] * world.cone_friction[(base + i) * 5 + j - 1];
        UU += U * U;
        UV += U * V;
        VV += V * V;
      }
      q[3] = U0;
      q[4] = V0;
      q[5] = UU;
      q[6] = UV;
      q[7] = VV;
      q[8] = __ddiv_rn(world.D[base + i], mu * mu * (1 + mu * mu));
      i += dim - 1;
    }
    q[0] *= 0.5;
    q[2] *= 0.5;
    for (int k = 0; k < 3; ++k)
      if (!isfinite(q[k])) {
        (&ctl.s07_status())[w] = 5;
        return;
      }
    if (elliptic)
      for (int k = 3; k < 9; ++k)
        if (!isfinite(q[k])) {
          (&ctl.s07_status())[w] = 5;
          return;
        }
  }
}

static __device__ void s07_continue_dx(unsigned char *shared, const NewtonModelData &model,
                                       const NewtonWorldData &world, const double *smooth_force, int w,
                                       int nv, int rows, const S07PreparedDx *prepared,
                                       bool quad_ready = false) {
  auto &ctl = newton_control(shared);
  double *shared_search = shared_at<double>(shared, NewtonLayout::search);
  double *shared_mass_search = shared_at<double>(shared, NewtonLayout::mass_search);
  double *shared_mass_acceleration = shared_at<double>(shared, NewtonLayout::mass_acceleration);
  const int nefc = ctl.count(), ne = ctl.ne(), nf = ctl.nf(), base = 0;
  double gtol = prepared->gtol, slopescl = prepared->slopescl;
  const double *v = shared_search + w * nv;
  const double *Mv = shared_mass_search + w * nv;
  const double *Jv = world.Jv + base;
  // The fast tile consumer already checked each produced C cell. Avoid a
  // second global Jv scan; compatible retains its original validation order.
  __shared__ int gate_bad, jv_bad[(flash_libero::SubstepShape::threads + 31) / 32];
  __shared__ double gaussian_dots[3];
  search_dots_warp<3>(v, shared_mass_acceleration + w * nv, smooth_force + w * nv, Mv, nv, gaussian_dots);
  if (!quad_ready) {
    bool local_bad = false;
    for (int i = threadIdx.x; i < nefc; i += blockDim.x)
      if (!isfinite(Jv[i]))
        local_bad = true;
    const bool warp_bad = libero_warp::any(local_bad);
    if (libero_warp::lane() == 0)
      jv_bad[libero_warp::warp()] = warp_bad;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    gate_bad = 0;
    if (!quad_ready)
      for (int i = 0; i < (blockDim.x + 31) / 32; ++i)
        if (jv_bad[i])
          gate_bad = 1;
    if (!gate_bad) {
      double *qG = world.quadGauss + w * 3;
      qG[1] = gaussian_dots[0] - gaussian_dots[1];
      qG[2] = 0.5 * gaussian_dots[2];
      if (!isfinite(qG[1]) || !isfinite(qG[2]))
        gate_bad = 1;
    }
    if (gate_bad)
      (&ctl.s07_status())[w] = 5;
  }
  __syncthreads();
  if (gate_bad)
    return;
  if (!quad_ready && world.constraint_group_heads) {
    for (int begin = 0; begin < nefc; begin += 16) {
      s07_quad_tile_block(shared, model, world, smooth_force, w, rows, begin,
                          begin + 16 < nefc ? begin + 16 : nefc, Jv + begin, world.constraint_group_heads,
                          world.constraint_head_rows, world.constraint_tile_head_offset, nullptr,
                          nullptr, nullptr);
      if ((&ctl.s07_status())[w])
        return;
    }
  } else if (!quad_ready) {
    if (threadIdx.x == 0)
      s07_quad_scalar_dx(shared, model, world, smooth_force, w, nefc, rows, base, Jv);
    __syncthreads();
    if ((&ctl.s07_status())[w])
      return;
  }
  int niter = 0;
  P p0 = s07_eval_point_block(shared, model, world, smooth_force, w, rows, nefc, ne, nf, 0.0, 1, &niter,
                              world.constraint_group_heads, world.constraint_head_rows, world.constraint_head_count[0],
                              nullptr);
  P p1 = s07_eval_point_block(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                              p0.alpha - __ddiv_rn(p0.d0, p0.d1), 2, &niter, world.constraint_group_heads,
                              world.constraint_head_rows, world.constraint_head_count[0], nullptr);
  if (p0.cost < p1.cost)
    p1 = p0;
  if (fabs(p1.d0) < gtol) {
    s07_finish_block(shared, model, world, smooth_force, w, p1.alpha, niter, p1.alpha == 0 ? 2 : 0,
                     fabs(p1.d0) * slopescl);
    return;
  }
  int dir = p1.d0 < 0 ? 1 : -1, p2update = 0;
  P p2 = {};
  while (p1.d0 * dir <= -gtol && niter < 50) {
    p2 = p1;
    p2update = 1;
    p1 = s07_eval_point_block(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                              p1.alpha - __ddiv_rn(p1.d0, p1.d1), 3, &niter, world.constraint_group_heads,
                              world.constraint_head_rows, world.constraint_head_count[0], nullptr);
    if (fabs(p1.d0) < gtol) {
      s07_finish_block(shared, model, world, smooth_force, w, p1.alpha, niter, 0, fabs(p1.d0) * slopescl);
      return;
    }
  }
  if (niter >= 50) {
    s07_finish_block(shared, model, world, smooth_force, w, p1.alpha, niter, 3, fabs(p1.d0) * slopescl);
    return;
  }
  if (!p2update) {
    s07_finish_block(shared, model, world, smooth_force, w, p1.alpha, niter, 6, fabs(p1.d0) * slopescl);
    return;
  }
  P p2next = p1;
  P p1next = s07_eval_point_block(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                                  p1.alpha - __ddiv_rn(p1.d0, p1.d1), 4, &niter, world.constraint_group_heads,
                                  world.constraint_head_rows, world.constraint_head_count[0], nullptr);
  while (niter < 50) {
    P pmid = s07_eval_point_block(shared, model, world, smooth_force, w, rows, nefc, ne, nf,
                                  0.5 * (p1.alpha + p2.alpha), 5, &niter, world.constraint_group_heads,
                                  world.constraint_head_rows, world.constraint_head_count[0], nullptr);
    P candidates[3] = {p1next, p2next, pmid};
    double bestcost = 0;
    int bestind = -1;
    for (int i = 0; i < 3; ++i)
      if (fabs(candidates[i].d0) < gtol && (bestind == -1 || candidates[i].cost < bestcost)) {
        bestcost = candidates[i].cost;
        bestind = i;
      }
    if (bestind >= 0) {
      s07_finish_block(shared, model, world, smooth_force, w, candidates[bestind].alpha, niter, 0,
                       fabs(candidates[bestind].d0) * slopescl);
      return;
    }
    int b1 = s07_update_bracket_block(shared, model, world, smooth_force, w, rows, nefc, ne, nf, &p1,
                                      candidates, &p1next, &niter, 6, world.constraint_group_heads,
                                      world.constraint_head_rows, world.constraint_head_count[0], nullptr);
    int b2 = s07_update_bracket_block(shared, model, world, smooth_force, w, rows, nefc, ne, nf, &p2,
                                      candidates, &p2next, &niter, 7, world.constraint_group_heads,
                                      world.constraint_head_rows, world.constraint_head_count[0], nullptr);
    if (!b1 && !b2) {
      s07_finish_block(shared, model, world, smooth_force, w, pmid.alpha, niter, pmid.cost < p0.cost ? 0 : 7,
                       fabs(pmid.d0) * slopescl);
      return;
    }
  }
  if (p1.cost <= p2.cost && p1.cost < p0.cost)
    s07_finish_block(shared, model, world, smooth_force, w, p1.alpha, niter, 4, fabs(p1.d0) * slopescl);
  else if (p2.cost <= p1.cost && p2.cost < p0.cost)
    s07_finish_block(shared, model, world, smooth_force, w, p2.alpha, niter, 4, fabs(p2.d0) * slopescl);
  else
    s07_finish_block(shared, model, world, smooth_force, w, 0.0, niter, 5, 1.0);
}

// The solve loop provides the final publication barrier after this call.
static __device__ void search_kernel_blas_dx(unsigned char *shared, const NewtonModelData &model,
                                             const NewtonWorldData &world,
                                             const double *smooth_force) {
  auto &ctl = newton_control(shared);
  __shared__ S07PreparedDx prepared;
  bool ready;
  {
    CF_NEWTON_PROFILE_SCOPE(SearchPrepare);
    ready = s07_prepare_dx(shared, model, world, smooth_force, &prepared);
  }
  if (!ready)
    return;
  double *workspace = shared_at<double>(shared, NewtonLayout::mathdx);
  {
    CF_NEWTON_PROFILE_SCOPE(SearchJacobian);
    s07_jv_blas_dx(shared, model, world, smooth_force, 0, ctl.count(), SubstepShape::rows,
        workspace, world.blas_diagnostic, true,
        world.constraint_group_heads, world.constraint_head_rows,
        world.constraint_tile_head_offset, nullptr, nullptr, nullptr);
  }
  if (!ctl.s07_status()) {
    CF_NEWTON_PROFILE_SCOPE(SearchContinue);
    s07_continue_dx(shared, model, world, smooth_force, 0, SubstepShape::nv,
        SubstepShape::rows, &prepared, true);
  }
}

// Collective FP64 Cholesky, embedded in the persistent Newton block.
// All block threads must enter every function in this file together.

// The device bridge and caller both use FMA-enabled arithmetic.
#if LIBERO_NEWTON_ISLANDS
namespace libero_islands {
static __device__ bool factor(double *, double *, int *, const double *, const int *, int, const int *,
                              const int *, const double *, const int *, const int *, const int *, int *);
}
#endif

// All block lanes participate. Warp votes cover strided matrix/input scans;
// a block vote publishes the uniform decision across multiple warps.
static __device__ inline bool factor_any_bad(bool bad) { return __syncthreads_or(bad); }

template <int N> static __device__ void block_potrf(double *H, double *temporary, int *diagnostic) {
  __shared__ int info;

  // Keep original operands intact until POTRF is admitted. This also keeps
  // the unused upper triangle intact, including caller-owned dirty bits.
#if LIBERO_NEWTON_SCALAR_IO
  for (int i = threadIdx.x; i < N * N; i += blockDim.x)
    temporary[i] = H[i];
#else
  libero_newton_io::copy_block(H, temporary, size_t(N) * N);
#endif // LIBERO_NEWTON_SCALAR_IO
  __syncthreads();
  if constexpr (N == SubstepShape::nv)
    dx_potrf_nv(temporary, &info);
  else if constexpr (N == 1)
    dx_potrf_1(temporary, &info);
  else if constexpr (N == 3)
    dx_potrf_3(temporary, &info);
  else if constexpr (N == 4)
    dx_potrf_4(temporary, &info);
  else if constexpr (N == 6)
    dx_potrf_6(temporary, &info);
  __syncthreads();

  if (threadIdx.x == 0) {
    ++diagnostic[N == SubstepShape::nv ? 0 : 1];
    if (info != 0)
      ++diagnostic[3];
  }
  // info and the factor were published together above. Each lane retains a
  // private predicate; one block vote returns the same decision in every lane.
  // MuJoCo's MINVAL regularization still requires the untouched source matrix.
  bool bad = info != 0;
  if (!bad) {
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      const double d = temporary[j * N + j];
      bad |= !isfinite(d) || d * d < 1e-15;
    }
  }
  const bool use_source = factor_any_bad(bad);
  if (use_source) {
    if (threadIdx.x == 0) {
      ++diagnostic[2];
      chol_factor(H, N, N);
    }
  } else {
#if LIBERO_NEWTON_SCALAR_IO
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      if (i / N >= i % N)
        H[i] = temporary[i];
#else
    // Each lane owns complete lower-row spans. No packet reaches the
    // caller-owned upper triangle, including the odd-row scalar tail.
    for (int row = threadIdx.x; row < N; row += blockDim.x)
      libero_newton_io::copy_serial(temporary + row * N, H + row * N, row + 1);
#endif // LIBERO_NEWTON_SCALAR_IO
  }
  __syncthreads();
}

static __device__ void local_potrf(int n, double *H, double *temporary, int *diagnostic) {
  // Contact dimensionality is a runtime value; descriptors are compile-time.
  switch (n) {
  case 1:
    block_potrf<1>(H, temporary, diagnostic);
    break;
  case 3:
    block_potrf<3>(H, temporary, diagnostic);
    break;
  case 4:
    block_potrf<4>(H, temporary, diagnostic);
    break;
  case 6:
    block_potrf<6>(H, temporary, diagnostic);
    break;
  }
}

static __device__ bool hessian_potrf_dispatch_dx(double *H, double *temporary, int *diagnostic, double *J,
                                                 int *state, int count, int *id, int *dim, double *qm,
                                                 int *madr, int *parent, int *simple,
                                                 int *islands_diagnostic) {
  CF_NEWTON_PROFILE_SCOPE(Potrf);
#if LIBERO_NEWTON_ISLANDS
  // factor owns both exact-block admission and its complete dense fallback.
  // The complete authoritative lower matrix is sufficient and is rebuilt
  // for every factor call. Do not introduce reads of otherwise-unused qM
  // (kind 2) or zero-weight quadratic Jacobian rows.
  return libero_islands::factor(H, temporary, diagnostic, nullptr, nullptr, 0, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr, islands_diagnostic);
#else
  block_potrf<SubstepShape::nv>(H, temporary, diagnostic);
  return true;
#endif
}

#if LIBERO_HESSIAN_BLAS
static __device__ void hessian_cone_batch_dx(int *count, int *state, int *id, int *dim, double *contact_H,
                                             double *J, double *H, double *HC, double *local, double *ltj,
                                             int *nupdate, double *temporary, int *diagnostic,
                                             int *blas_diagnostic, double *pending, int *algorithm_diagnostic,
                                             double *qm, int *madr, int *parent, int *simple,
                                             int *islands_diagnostic, int *factor_status) {
  // H remains the quadratic/base factor. The final HC lower factor is built
  // from exactly that effective base and the same regularized contact factors.
  // H/HC upper owner bits are never operands and retain the source values.
  hessian_from_factor_blas_dx(H, HC, temporary, algorithm_diagnostic);
  int used = 0;
  for (int i = 0; i < count[0];) {
    if (state[i] != 4) {
      ++i;
      continue;
    }
    const int c = id[i], d = dim[c];
    libero_newton_io::copy_block(contact_H + c * 36, local, size_t(d) * d);
    __syncthreads();
    local_potrf(d, local, temporary, diagnostic);
    cone_ltj_blas_dx(i, d, J, local, ltj, temporary, blas_diagnostic);
    for (int r = 0; r < d;) {
      const int take = min(d - r, flash_libero::NewtonLayout::hessian_tile_rows - used);
      for (int z = threadIdx.x; z < take * SubstepShape::nv; z += blockDim.x)
        pending[used * SubstepShape::nv + z] = ltj[r * SubstepShape::nv + z];
      __syncthreads();
      used += take;
      r += take;
      if (used == flash_libero::NewtonLayout::hessian_tile_rows) {
        cone_gram_append_blas_dx(pending, used, HC, temporary, algorithm_diagnostic);
        used = 0;
      }
    }
    if (threadIdx.x == 0)
      nupdate[0] += d;
    __syncthreads();
    i += d;
  }
  if (used)
    cone_gram_append_blas_dx(pending, used, HC, temporary, algorithm_diagnostic);
  const bool success = hessian_potrf_dispatch_dx(HC, temporary, diagnostic, J, state, count[0], id, dim, qm,
                                                 madr, parent, simple, islands_diagnostic);
  if (threadIdx.x == 0 && !success && factor_status)
    factor_status[0] = 5;
  if (threadIdx.x == 0 && algorithm_diagnostic)
    ++algorithm_diagnostic[0];
  __syncthreads();
}
#endif

static __device__ void hessian_cone_dx(int rows, int contacts, int *count, int *state, int *id, int *dim,
                                       double *contact_H, double *J, double *H, double *HC, double *local,
                                       double *ltj, double *vec, int *nupdate, double *temporary,
                                       int *diagnostic, int *blas_diagnostic = nullptr,
                                       double *pending = nullptr, int *algorithm_diagnostic = nullptr,
                                       double *qm = nullptr, int *madr = nullptr, int *parent = nullptr,
                                       int *simple = nullptr, int *islands_diagnostic = nullptr,
                                       int *factor_status = nullptr) {
  CF_NEWTON_PROFILE_SCOPE(ConeFactor);
#if LIBERO_HESSIAN_BLAS && LIBERO_CONE_HESSIAN_STRATEGY != 0
  int ranks = 0;
  // ncone counts rows, but this standalone helper deliberately has no ncone
  // parameter. Count validated state==4 rows without reading inactive J.
  if constexpr (LIBERO_CONE_HESSIAN_STRATEGY == 2)
    for (int r = 0; r < count[0]; ++r)
      ranks += state[r] == 4;
  const bool batch = pending && (LIBERO_CONE_HESSIAN_STRATEGY == 1 || ranks >= LIBERO_CONE_BATCH_THRESHOLD);
  if (batch) {
    hessian_cone_batch_dx(count, state, id, dim, contact_H, J, H, HC, local, ltj, nupdate, temporary,
                          diagnostic, blas_diagnostic, pending, algorithm_diagnostic, qm, madr, parent,
                          simple, islands_diagnostic, factor_status);
    return;
  }
#endif
  if (threadIdx.x == 0 && algorithm_diagnostic)
    ++algorithm_diagnostic[1];
#if LIBERO_NEWTON_SCALAR_IO
  for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x)
    HC[z] = H[z];
#else
  libero_newton_io::copy_block(H, HC, size_t(SubstepShape::nv) * SubstepShape::nv);
#endif // LIBERO_NEWTON_SCALAR_IO
  __syncthreads();
  for (int i = 0; i < count[0];) {
    if (state[i] != 4) {
      ++i;
      continue;
    }
    int c = id[i], d = dim[c];
#if LIBERO_NEWTON_SCALAR_IO
    for (int z = threadIdx.x; z < d * d; z += blockDim.x)
      local[z] = contact_H[c * 36 + z];
#else
    libero_newton_io::copy_block(contact_H + c * 36, local, size_t(d) * d);
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
    local_potrf(d, local, temporary, diagnostic);
#if LIBERO_HESSIAN_BLAS
    cone_ltj_blas_dx(i, d, J, local, ltj, temporary, blas_diagnostic);
#endif
#if !LIBERO_HESSIAN_BLAS
    if (threadIdx.x == 0) {
      for (int z = 0; z < d * SubstepShape::nv; ++z)
        ltj[z] = 0.0;
      for (int r = 0; r < d; ++r)
        for (int c0 = 0; c0 <= r; ++c0) {
          double s = local[r * d + c0];
          for (int k = 0; k < SubstepShape::nv; ++k)
            ltj[c0 * SubstepShape::nv + k] += J[(i + r) * SubstepShape::nv + k] * s;
        }
    }
    __syncthreads();
#endif
    if constexpr (SubstepShape::nv <= 32) {
      if (threadIdx.x == 0) {
        for (int r = 0; r < d; ++r) {
#if LIBERO_NEWTON_SCALAR_IO
          for (int k = 0; k < SubstepShape::nv; ++k)
            vec[k] = ltj[r * SubstepShape::nv + k];
#else
          libero_newton_io::copy_serial(ltj + r * SubstepShape::nv, vec, SubstepShape::nv);
#endif // LIBERO_NEWTON_SCALAR_IO
          chol_update(HC, vec, 1);
        }
      }
    } else {
      for (int r = 0; r < d; ++r) {
#if LIBERO_NEWTON_SCALAR_IO
        for (int k = threadIdx.x; k < SubstepShape::nv; k += blockDim.x)
          vec[k] = ltj[r * SubstepShape::nv + k];
#else
        libero_newton_io::copy_block(ltj + r * SubstepShape::nv, vec, SubstepShape::nv);
#endif // LIBERO_NEWTON_SCALAR_IO
        __syncthreads();
        chol_update_block(HC, vec, 1);
      }
    }
    if (threadIdx.x == 0)
      nupdate[0] += d;
    __syncthreads();
    i += d;
  }
}

static __device__ int compact_changed_rows(const int *oldstate, const int *state, int count, int *rows) {
  CF_NEWTON_PROFILE_SCOPE(ChangedRows);
  __shared__ int total;
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    int size = 0;
    for (int base = 0; base < count; base += 32) {
      const int r = base + lane;
      const bool changed = r < count && ((oldstate[r] == 1) != (state[r] == 1));
      const unsigned mask = __ballot_sync(0xffffffffu, changed);
      if (changed)
        rows[size + __popc(mask & ((1u << lane) - 1))] = r;
      size += __popc(mask);
    }
    if (lane == 0)
      total = size;
  }
  __syncthreads(); // ordered list and length become visible to both warps
  return total;
}

static __device__ void hessian_factor_update_dx(unsigned char *shared, const NewtonModelData &model,
                                                const NewtonWorldData &p, int kind) {
  double *shared_hessian = shared_at<double>(shared, NewtonLayout::hessian);
  double *shared_mathdx = shared_at<double>(shared, NewtonLayout::mathdx);
  double *shared_local_factor = shared_at<double>(shared, NewtonLayout::local_factor);
  double *shared_local_product = shared_at<double>(shared, NewtonLayout::local_product);
  double *shared_update_vector = shared_at<double>(shared, NewtonLayout::update_vector);
  auto &ctl = newton_control(shared);
  // Python supplies per-world pointer views, so this collective uses w=0.
  // Admitted calls publish their factor/status on every numerical exit. The
  // solve loop can consume them directly without another caller-side barrier.
  if (!ctl.active() || ctl.status())
    return;
  __shared__ int lost, forced, changed_rows;
  int k = kind;
  if (threadIdx.x == 0 && p.hessian_algorithm_diagnostic)
    p.hessian_algorithm_diagnostic[6] += ctl.ncone();
  if (k == 1) {
    const bool success = hessian_potrf_dispatch_dx(shared_hessian, shared_mathdx, p.potrf_diagnostic, p.J,
                                                   p.state, ctl.count(), p.efc_id, p.contact_dim, p.qM,
                                                   model.madr, model.parentid, model.simplenum, nullptr);
    if (threadIdx.x == 0 && !success)
      ctl.status() = 5;
    __syncthreads();
    if (ctl.status())
      return;
    if (threadIdx.x == 0) {
      ctl.nnz() = SubstepShape::nv * SubstepShape::nv;
      ctl.nupdate() = 0;
    }
    __syncthreads();
    if (ctl.ncone())
      hessian_cone_dx(SubstepShape::rows, SubstepShape::contacts, (&ctl.count()), p.state, p.efc_id,
                      p.contact_dim, p.contact_H, p.J, shared_hessian, p.Hcone, shared_local_factor,
                      shared_local_product, shared_update_vector, (&ctl.nupdate()), shared_mathdx,
                      p.potrf_diagnostic, p.blas_diagnostic, p.cone_pending_rows,
                      p.hessian_algorithm_diagnostic, p.qM, model.madr, model.parentid, model.simplenum,
                      nullptr, (&ctl.status()));
    if (threadIdx.x == 0) {
      ctl.route() = ctl.ncone() ? 2 : 1;
      ctl.nupdate() = ctl.count();
    }
  } else if (k == 2) {
    hessian_cone_dx(SubstepShape::rows, SubstepShape::contacts, (&ctl.count()), p.state, p.efc_id,
                    p.contact_dim, p.contact_H, p.J, shared_hessian, p.Hcone, shared_local_factor,
                    shared_local_product, shared_update_vector, (&ctl.nupdate()), shared_mathdx,
                    p.potrf_diagnostic, p.blas_diagnostic, p.cone_pending_rows,
                    p.hessian_algorithm_diagnostic, p.qM, model.madr, model.parentid, model.simplenum,
                    nullptr, (&ctl.status()));
    if (threadIdx.x == 0)
      ctl.route() = 6;
  } else {
    const int changes = compact_changed_rows(p.oldstate, p.state, ctl.count(), p.changed_rows);
    if (threadIdx.x == 0) {
      // Threshold 0 disables this policy. It is a configurable algorithm
      // crossover experiment, not a measured platform tuning result.
      forced = p.hessian_upper_backup && LIBERO_RANK_REBUILD_THRESHOLD > 0 &&
               changes >= LIBERO_RANK_REBUILD_THRESHOLD;
      lost = forced;
      changed_rows = changes;
      ctl.nupdate() = 0;
      if (p.hessian_algorithm_diagnostic) {
        p.hessian_algorithm_diagnostic[5] += changes;
        p.hessian_algorithm_diagnostic[4] += lost;
      }
    }
    __syncthreads();
    if (!lost) {
      CF_NEWTON_PROFILE_SCOPE(RankUpdates);
      if constexpr (SubstepShape::nv <= 32) {
        if (threadIdx.x == 0) {
          for (int k = 0; k < changes; ++k) {
            const int r = p.changed_rows[k];
            const bool after = p.state[r] == 1;
            const double s = row_sqrt_d(p, r);
            for (int j = 0; j < SubstepShape::nv; ++j)
              shared_update_vector[j] = p.J[r * SubstepShape::nv + j] * s;
            const int rank = chol_update(shared_hessian, shared_update_vector, after ? 1 : 0);
            ++ctl.nupdate();
            if (rank < SubstepShape::nv) {
              lost = 1;
              break;
            }
          }
        }
      } else {
        for (int k = 0; k < changes; ++k) {
          const int r = p.changed_rows[k];
          const bool after = p.state[r] == 1;
          const double s = row_sqrt_d(p, r);
          for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
            shared_update_vector[j] = p.J[r * SubstepShape::nv + j] * s;
          __syncthreads();
          const int rank = chol_update_block(shared_hessian, shared_update_vector, after ? 1 : 0);
          if (threadIdx.x == 0) {
            ++ctl.nupdate();
            if (rank < SubstepShape::nv)
              lost = 1;
          }
          __syncthreads();
          if (lost)
            break;
        }
      }
    }
    __syncthreads();
    if (lost) {
      bool bad = false;
      for (int z = threadIdx.x; z < SubstepShape::packed_mass; z += blockDim.x)
        bad |= !isfinite(p.qM[z]);
      for (int r = 0; r < ctl.count(); ++r) {
        if (p.state[r] != 1)
          continue;
        if (threadIdx.x == 0)
          bad |= p.D[r] < 0 || !isfinite(p.D[r]);
        if (p.D[r] != 0)
          for (int j = threadIdx.x; j < SubstepShape::nv; j += blockDim.x)
            bad |= !isfinite(p.J[r * SubstepShape::nv + j]);
      }
      const bool bad_input = factor_any_bad(bad);
      if (threadIdx.x == 0 && bad_input)
        ctl.status() = 4;
      __syncthreads();
    }
    if (ctl.status())
      return; // uniform block return after the barrier
    if (lost) {
      CF_NEWTON_PROFILE_SCOPE(Rebuild);
      if (forced) {
        for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x)
          if (z / SubstepShape::nv < z % SubstepShape::nv)
            p.hessian_upper_backup[z] = shared_hessian[z];
        __syncthreads();
      }
#if LIBERO_HESSIAN_BLAS
      hessian_rebuild_blas_dx((&ctl.count()), p.J, p.D, p.state, p.qM, model.madr, model.parentid,
                              model.simplenum, shared_hessian, shared_mathdx, p.blas_diagnostic, atomic_workspace(p), jacobian_view(p));
#else
      for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x) {
        int i = z / SubstepShape::nv, j = z % SubstepShape::nv;
        if (j > i)
          continue;
        double v = assembly_cell(0, i, j, SubstepShape::rows, (&ctl.count()), p.J, p.D, p.state, p.qM,
                                 model.madr, model.parentid, model.simplenum);
        shared_hessian[z] = v;
        if (i != j)
          shared_hessian[j * SubstepShape::nv + i] = v;
      }
      __syncthreads();
#endif
      const bool success = hessian_potrf_dispatch_dx(shared_hessian, shared_mathdx, p.potrf_diagnostic, p.J,
                                                     p.state, ctl.count(), p.efc_id, p.contact_dim, p.qM,
                                                     model.madr, model.parentid, model.simplenum, nullptr);
      if (threadIdx.x == 0 && !success)
        ctl.status() = 5;
      __syncthreads();
      if (ctl.status())
        return;
      if (forced) {
        for (int z = threadIdx.x; z < SubstepShape::nv * SubstepShape::nv; z += blockDim.x)
          if (z / SubstepShape::nv < z % SubstepShape::nv)
            shared_hessian[z] = p.hessian_upper_backup[z];
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        ctl.nnz() = SubstepShape::nv * SubstepShape::nv;
        if (forced)
          ctl.nupdate() = changed_rows;
        else if (ctl.ncone())
          ctl.nupdate() = 0;
      }
      __syncthreads();
      if (ctl.ncone())
        hessian_cone_dx(SubstepShape::rows, SubstepShape::contacts, (&ctl.count()), p.state, p.efc_id,
                        p.contact_dim, p.contact_H, p.J, shared_hessian, p.Hcone, shared_local_factor,
                        shared_local_product, shared_update_vector, (&ctl.nupdate()), shared_mathdx,
                        p.potrf_diagnostic, p.blas_diagnostic, p.cone_pending_rows,
                        p.hessian_algorithm_diagnostic, p.qM, model.madr, model.parentid, model.simplenum,
                        nullptr, (&ctl.status()));
      if (threadIdx.x == 0) {
        if (forced)
          ctl.route() = ctl.ncone() ? 4 : 3;
        else {
          ctl.nupdate() = ctl.count();
          ctl.route() = 5;
        }
      }
    } else {
      if (ctl.ncone())
        hessian_cone_dx(SubstepShape::rows, SubstepShape::contacts, (&ctl.count()), p.state, p.efc_id,
                        p.contact_dim, p.contact_H, p.J, shared_hessian, p.Hcone, shared_local_factor,
                        shared_local_product, shared_update_vector, (&ctl.nupdate()), shared_mathdx,
                        p.potrf_diagnostic, p.blas_diagnostic, p.cone_pending_rows,
                        p.hessian_algorithm_diagnostic, p.qM, model.madr, model.parentid, model.simplenum,
                        nullptr, (&ctl.status()));
      if (threadIdx.x == 0)
        ctl.route() = ctl.ncone() ? 4 : 3;
    }
  }
  __syncthreads();
  // Pivot handling and rank-loss fallback are part of the factor algorithm.
  // A second full H/HC finite scan is a checked-entry diagnostic, not math.
}

// The immutable prepared plan and output ownership establish nonaliasing.
static __device__ void warm_mass_cost_prepared(const NewtonModelData &m, const NewtonWorldData &p,
                                               unsigned char *shared, const double *smooth_force,
                                               const double *smooth_acceleration) {
  double *shared_warmstart = shared_at<double>(shared, NewtonLayout::warmstart);
  double *shared_warm_mass = shared_at<double>(shared, NewtonLayout::warm_mass);
  auto &c = newton_control(shared);
  if (threadIdx.x >= 32 || !c.newton_world())
    return;
  double cost = 0.0;
  for (int row = threadIdx.x; row < SubstepShape::nv; row += 32) {
    double value = 0.0;
    for (int e = m.mass_plan_offsets[row]; e < m.mass_plan_offsets[row + 1]; ++e) {
      const double term = p.qM[m.mass_plan_coefficient[e]] * shared_warmstart[m.mass_plan_source[e]];
      if (m.mass_plan_assign[e])
        value = term;
      else
        value += term;
    }
    shared_warm_mass[row] = value;
    cost += .5 * (value - smooth_force[row]) * (shared_warmstart[row] - smooth_acceleration[row]);
  }
  cost = libero_warp::sum(cost);
  if (threadIdx.x == 0) {
    c.quad0() = cost;
  }
}

// Cost-only warmstart comparison. Follow the same typed-head order as the
// full constraint pass, but do not materialize force, state, J^T force or H.
// The selected candidate receives one complete update immediately afterwards.
static __device__ inline void accumulate_candidate_cost(
    const NewtonWorldData &p, const double *jar, int r, int ne, int nf, double &cost) {
  const double v = jar[r], D = p.D[r];
  if (r < ne) {
    cost += .5 * D * v * v;
  } else if (r < ne + nf) {
    const double R = p.R[r], f = p.loss[r];
    const double limit = row_friction_limit(p, r);
    if (v <= -limit)
      cost += -.5 * R * f * f - f * v;
    else if (v >= limit)
      cost += -.5 * R * f * f + f * v;
    else
      cost += .5 * D * v * v;
  } else if (p.efc_type[r] != 7) {
    if (v < 0.0)
      cost += .5 * D * v * v;
  } else {
    const int contact = p.efc_id[r], dim = p.contact_dim[contact];
    const double mu = p.contact_mu[contact];
    const double N = v * mu, T = cone_norm(jar + r, p.contact_friction + contact * 5, dim);
    if (N >= mu * T || (T <= 0 && N >= 0))
      return;
    if (mu * N + T <= 0 || (T <= 0 && N < 0)) {
      for (int j = 0; j < dim; ++j)
        cost += .5 * p.D[r + j] * jar[r + j] * jar[r + j];
    } else {
      const double dm = coefficient_plane(p.precomputed, NewtonCoefficientLayout::cone_dm)[r];
      const double residual = N - mu * T;
      cost += .5 * dm * residual * residual;
    }
  }
}

static __device__ void candidate_costs_block(unsigned char *shared, const NewtonWorldData &p) {
  auto &ctl = newton_control(shared);
  if (!ctl.active())
    return;
  constexpr int warps = SubstepShape::threads / 32;
  __shared__ double warm_partial[warps], smooth_partial[warps];
  double warm = 0.0, smooth = 0.0;
  const int ne = ctl.ne(), nf = ctl.nf();
  for (int bucket = 0; bucket < 7; ++bucket) {
    const int begin = p.constraint_typed_head_offsets[bucket];
    const int end = p.constraint_typed_head_offsets[bucket + 1];
    for (int k = begin + threadIdx.x; k < end; k += blockDim.x) {
      const int r = p.constraint_typed_head_rows[k];
      accumulate_candidate_cost(p, p.jar_warm, r, ne, nf, warm);
      accumulate_candidate_cost(p, p.efc_b, r, ne, nf, smooth);
    }
  }
  warm = libero_warp::sum(warm);
  smooth = libero_warp::sum(smooth);
  const int lane = libero_warp::lane(), warp = libero_warp::warp();
  if (lane == 0) {
    warm_partial[warp] = warm;
    smooth_partial[warp] = smooth;
  }
  __syncthreads(); // also publishes the earlier warm mass/Gauss producer
  if (warp == 0) {
    warm = libero_warp::sum(lane < warps ? warm_partial[lane] : 0.0);
    smooth = libero_warp::sum(lane < warps ? smooth_partial[lane] : 0.0);
    if (lane == 0) {
      ctl.warm_cost() = warm + ctl.quad0();
      ctl.smooth_cost() = smooth;
    }
  }
}

static __device__ void select_candidate_block(unsigned char *shared, const double *warm,
                                               const double *smooth) {
  auto &ctl = newton_control(shared);
  if (!ctl.newton_world())
    return;
  // Costs are the mathematical inputs to this choice. Candidate force/state
  // are deliberately absent; they are not consumed until the selected update.
  const bool finite = isfinite(ctl.warm_cost()) && isfinite(ctl.smooth_cost());
  if (!finite) {
    if (threadIdx.x == 0)
      ctl.s08a_status() = 3;
    return;
  }
  const bool choose_smooth = ctl.warm_cost() > ctl.smooth_cost();
  if (threadIdx.x == 0)
    ctl.choice() = choose_smooth;
  double *qacc = shared_at<double>(shared, NewtonLayout::qacc);
  const double *selected = choose_smooth ? smooth : warm;
  for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x)
    qacc[i] = selected[i];
}

__device__ __noinline__ void solve_newton(unsigned char *shared, const NewtonModelData &model,
                                                    const NewtonWorldData &p, const int *entry_upstream,
                                                    const int *entry_enabled, const int *entry_publication,
                                                    const int64_t *entry_plan, const NewtonOptions &options) {
  const int w = blockIdx.y;
  CF_NEWTON_PROFILE_BEGIN(entry_enabled[w] != 0);
  auto &c = newton_control(shared);
  const uint64_t position_generation = p.position_generation[0];
  const bool refresh_coefficients = !p.position_cached[0] ||
      *reinterpret_cast<const uint64_t *>(p.precomputed + NewtonCoefficientLayout::generation) !=
          position_generation;
#if FLASH_NEWTON_ATOMIC_REDUCTIONS
  const bool jacobian_prepared = jacobian_view(p).prepared(p.nefc[0], position_generation);
#endif
  const int *layout_counts = reinterpret_cast<const int *>(
      p.precomputed + NewtonCoefficientLayout::layout_counts);
  // Capture before the existing entry barrier, just like coefficient refresh.
  // All four counts belong to the current position/constraint producer.
  const bool reuse_layout = p.position_cached[0] &&
      *reinterpret_cast<const int *>(p.precomputed + NewtonCoefficientLayout::layout_valid) == 1 &&
      *reinterpret_cast<const uint64_t *>(p.precomputed + NewtonCoefficientLayout::layout_generation) ==
          position_generation &&
      layout_counts[0] == p.nefc[0] && layout_counts[1] == p.ne[0] &&
      layout_counts[2] == p.nf[0] && layout_counts[3] == p.ncon[0];
  // Each lane accesses its DOF; these names are addresses, not copied vectors.
  double *constraint_force = shared_at<double>(shared, NewtonLayout::constraint_force);
  double *gradient = shared_at<double>(shared, NewtonLayout::gradient);
  double *hessian = shared_at<double>(shared, NewtonLayout::hessian);
  double *mass_acceleration = shared_at<double>(shared, NewtonLayout::mass_acceleration);
  double *mass_search = shared_at<double>(shared, NewtonLayout::mass_search);
  double *qacc = shared_at<double>(shared, NewtonLayout::qacc);
  double *search = shared_at<double>(shared, NewtonLayout::search);
  double *warm_mass = shared_at<double>(shared, NewtonLayout::warm_mass);
  double *warmstart = shared_at<double>(shared, NewtonLayout::warmstart);
  NewtonOptions execution = options;
  if (execution.audit)
    execution.audit += w * 8;
  if (execution.first_disagreement)
    execution.first_disagreement += w * 8;
  double *smooth_acceleration = execution.smooth_acceleration ? execution.smooth_acceleration : p.qacc_smooth;
  double *smooth_force = execution.smooth_force ? execution.smooth_force : p.qfrc_smooth;
  // Cooperative zero initialization, followed by one uniform header load.
  for (int i = threadIdx.x; i < NewtonControl::integer_count; i += blockDim.x)
    c.integers[i] = 0;
  for (int i = threadIdx.x; i < NewtonControl::real_count; i += blockDim.x)
    c.reals[i] = 0.0;
  __syncthreads();
  if (threadIdx.x == 0) {
    c.empty() = entry_enabled[w] && !entry_upstream[w] && p.nefc[0] == 0;
    c.newton_world() = c.active() = entry_enabled[w] && !entry_upstream[w] && p.nefc[0] > 0;
    c.err_code() = entry_upstream[w];
    c.error_item() = -1;
    c.solver_iter() = p.canonical_solver_iter[0];
    c.solver_nnz() = p.canonical_solver_nnz[0];
    c.count() = p.nefc[0];
    c.ne() = p.ne[0];
    c.nf() = p.nf[0];
    c.ncon() = p.ncon[0];
    c.stage() = 1;
    c.maxiter_arr() = 100;
    c.diagnostic() = entry_publication[w];
    c.meaninertia() = model.model_scalars[1];
    c.tolerance() = model.model_scalars[0];
    c.expected = entry_plan[w];
    const bool schema_ready = prepare_newton_entry(p, c, reuse_layout);
    if (c.newton_world() && !schema_ready) {
      if (!c.err_code())
        c.err_code() = NewtonError::constraint + 4;
      c.active() = c.newton_world() = 0;
    }
    if (entry_enabled[w]) {
      ++p.attempted[0];
      if (execution.audit)
        ++execution.audit[0];
    }
  }
  // DOF lanes load only their own vector cells. Large constraint work remains
  // global and is traversed cooperatively over its actual live row count.
  for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x) {
    qacc[i] = p.canonical_qacc[i];
    warmstart[i] = p.canonical_qacc_warmstart[i];
    constraint_force[i] = p.canonical_qfrc_constraint[i];
  }
  const int live_rows = max(0, min(p.nefc[0], SubstepShape::rows));
  libero_newton_io::copy_block(p.external_r_ef, p.force, live_rows);
  libero_newton_io::copy_block(p.canonical_state, p.state, live_rows);
  __syncthreads();
  double *dx_matrix = shared_at<double>(shared, NewtonLayout::mathdx);
  if (c.invariant_valid() && !reuse_layout)
    constraint_tile_head_offsets_block(c.count(), c.ne(), c.nf(), 0, 0, p.efc_type, p.efc_id, p.contact_dim,
                                       p.constraint_head_rows, p.constraint_head_count[0],
                                       p.constraint_tile_head_offset);
  if (!reuse_layout)
    validate_contact_metadata_warp(p, c);
  prepare_row_coefficients(shared, p, refresh_coefficients, position_generation);
  // The metadata validator's writer is lane zero of warp zero. The caller's
  // existing publication barrier closes both tile production and this header.
  if (threadIdx.x == 0 && !reuse_layout && c.invariant_valid() && !c.err_code()) {
    int *counts = reinterpret_cast<int *>(p.precomputed + NewtonCoefficientLayout::layout_counts);
    counts[0] = c.count(); counts[1] = c.ne(); counts[2] = c.nf(); counts[3] = c.ncon();
    *reinterpret_cast<uint64_t *>(p.precomputed + NewtonCoefficientLayout::layout_generation) =
        position_generation;
    *reinterpret_cast<int *>(p.precomputed + NewtonCoefficientLayout::layout_valid) = 1;
  }
  if (c.empty()) {
    for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x) {
      qacc[i] = smooth_acceleration[i];
      warmstart[i] = smooth_acceleration[i];
      constraint_force[i] = 0.0;
    }
    if (threadIdx.x == 0)
      c.solver_iter() = 0;
  }
  __syncthreads(); // publishes entry metadata, coefficients, tile offsets and empty-world output
#if FLASH_NEWTON_ATOMIC_REDUCTIONS
  // All lanes captured reuse/refresh before entry. A successful cached prefix
  // alone is insufficient: every constraint count and layout verdict must
  // match as well. Failure/overflow is a prepared fallback, not a per-solve retry.
  if (reduction_policy.cache_enabled && c.active() && c.invariant_valid() && !c.err_code()) {
    const auto cache = jacobian_view(p);
    if (refresh_coefficients || !reuse_layout || !jacobian_prepared) {
      CF_NEWTON_PROFILE_SCOPE(JacobianCache);
      jacobian_cache::prepare(cache, p.J, p.D, c.count(), position_generation, dx_matrix);
      __syncthreads(); // publish CSC and prepared header before consumers
    }
  }
#endif

  // Every condition below is uniform after the producer's numerical barrier.
  // Failure/convergence exits this body immediately; all lanes rejoin before
  // finalization. active stays immutable throughout the numerical body.
  CF_NEWTON_PROFILE_VALUE(Rows, c.count());
  CF_NEWTON_PROFILE_VALUE(Contacts, c.ncon());
  CF_NEWTON_PROFILE_VALUE(FrictionRows, c.nf());
  CF_NEWTON_PROFILE_VALUE(CoefficientRefresh, refresh_coefficients && c.invariant_valid());
  CF_NEWTON_PROFILE_VALUE(LayoutRefresh, !reuse_layout && c.invariant_valid() && !c.err_code());
  int iteration_slot = 0;
  do {
    if (!c.active())
      break;
    if (threadIdx.x == 0)
      solver_iter_reset(0, &c.newton_world(), &c.solver_iter());
    CF_NEWTON_PROFILE_STAGE(WarmJacobian);
    jacvec_candidates_blas_dx(&c.newton_world(), p.nefc, p.J, smooth_acceleration, warmstart, p.aref, p.efc_b,
                              p.jar_warm, dx_matrix, p.blas_diagnostic);
    // Gauss and both cost sums share the candidate reduction's publication.
    CF_NEWTON_PROFILE_STAGE(WarmCosts);
    warm_mass_cost_prepared(model, p, shared, smooth_force, smooth_acceleration);
    candidate_costs_block(shared, p);
    __syncthreads();
    select_candidate_block(shared, warmstart, smooth_acceleration);
    __syncthreads(); // selected qacc, choice and candidate verdict
    if (c.s08a_status()) {
      if (threadIdx.x == 0 && !c.err_code())
        c.err_code() = NewtonError::commit + c.s08a_status();
      break;
    }
    CF_NEWTON_PROFILE_STAGE(InitialResiduals);
    warm_init_mass_prepared(&c.active(), &c.choice(), warm_mass, qacc, p.qM, mass_acceleration, model);
    warm_init_jacvec(&c.active(), p.nefc, &c.choice(), p.jar_warm, p.efc_b, p.jar);
    __syncthreads();
    CF_NEWTON_PROFILE_STAGE(InitialConstraint);
    constraint_update_core_block(shared, p, true, p.jar, true, false);
    __syncthreads();
    // Transpose's exit publishes these leader records together with J^T f.
    hessian_prepare_core_block(0, &c.status(), &c.route());
    CF_NEWTON_PROFILE_STAGE(InitialTranspose);
    constraint_transpose_blas_dx(0, &c.active(), &c.count(), p.J, p.force, constraint_force,
                                 SubstepShape::rows, SubstepShape::nv, dx_matrix, p.blas_diagnostic, atomic_workspace(p), jacobian_view(p));
    CF_NEWTON_PROFILE_STAGE(InitialAssembly);
    gauss_compute_block(0, &c.active(), mass_acceleration, smooth_force, qacc, smooth_acceleration, &c.cost(),
                        &c.quad0(), true);
    // Gauss writes cost/quad0 independently of H and MathDx staging; the
    // assembly/factor collectives publish them before direction/search.
    hessian_assemble_blas_dx(0, &c.count(), &c.active(), 1, p.J, p.D, p.state, p.qM, model.madr, model.parentid,
                             model.simplenum, hessian, &c.status(), SubstepShape::rows, dx_matrix,
                             p.blas_diagnostic, atomic_workspace(p), jacobian_view(p));
    CF_NEWTON_PROFILE_STAGE(InitialFactor);
    hessian_factor_update_dx(shared, model, p, 1);
    if (c.status()) {
      if (threadIdx.x == 0 && !c.err_code())
        c.err_code() = NewtonError::hessian + c.status();
      break;
    }
    CF_NEWTON_PROFILE_STAGE(InitialDirection);
    direction_core_blas_dx(shared, p, smooth_force); // owns its exit publication

    // CPU order: line search -> commit -> constraint/Hessian -> direction
    // -> stopping rule. Every solve and completed warmstart update is retained.
    for (; iteration_slot < 100; ++iteration_slot) {
#if LIBERO_NEWTON_TRACE
      seed_eval_block(0, p.eval_trace, p.eval_tag);
      __syncthreads();
#endif
      CF_NEWTON_PROFILE_STAGE(LineSearch);
      search_kernel_blas_dx(shared, model, p, smooth_force);
      __syncthreads();
      CF_NEWTON_PROFILE_ADD(SearchEvaluations, c.LSiter());
      if (threadIdx.x == 0 && execution.audit) {
        execution.audit[2] += c.LSiter();
        execution.audit[7] += c.alpha() == 0.0;
      }
      if (c.s07_status() || c.alpha() == 0.0) {
        if (threadIdx.x == 0) {
          if (c.s07_status() && !c.err_code())
            c.err_code() = 1000 + c.s07_status();
          else if (!c.s07_status())
            c.stop_reason() = 10;
        }
        break;
      }
      CF_NEWTON_PROFILE_STAGE(CommitSave);
      commit_step_core_block(0, &c.active(), p.nefc, &c.alpha(), search, mass_search, p.Jv, qacc,
                             mass_acceleration, p.jar, &c.committed(), SubstepShape::nv, SubstepShape::rows);
      // Independent destinations: save_old reads neither qacc, Ma nor jar.
      save_old_block(0, &c.active(), p.nefc, p.state, &c.cost(), p.oldstate, &c.oldcost(), SubstepShape::rows);
      __syncthreads();
      CF_NEWTON_PROFILE_STAGE(IterConstraint);
      constraint_update_core_block(shared, p, true, p.jar, true, true);
      __syncthreads();
      hessian_prepare_core_block(0, &c.status(), &c.route());
      CF_NEWTON_PROFILE_STAGE(IterTranspose);
      constraint_transpose_blas_dx(0, &c.active(), &c.count(), p.J, p.force, constraint_force,
                                   SubstepShape::rows, SubstepShape::nv, dx_matrix, p.blas_diagnostic, atomic_workspace(p), jacobian_view(p));
      CF_NEWTON_PROFILE_STAGE(IterPrepare);
      gauss_compute_block(0, &c.active(), mass_acceleration, smooth_force, qacc, smooth_acceleration, &c.cost(),
                          &c.quad0(), true);
      CF_NEWTON_PROFILE_STAGE(IterFactor);
      hessian_factor_update_dx(shared, model, p, 3);
      if (c.status()) {
        if (threadIdx.x == 0 && !c.err_code())
          c.err_code() = NewtonError::hessian + c.status();
        break;
      }
      CF_NEWTON_PROFILE_STAGE(IterDirection);
      direction_core_blas_dx(shared, p, smooth_force, false);
      CF_NEWTON_PROFILE_STAGE(Statistics);
      stats_stop_block(0, &c.active(), p.nefc, &c.stage(), &c.committed(), p.oldstate, p.state, gradient,
                       &c.oldcost(), &c.cost(), &c.meaninertia(), &c.tolerance(), &c.slope(), &c.nactive(),
                       &c.nupdate(), &c.LSiter(), &c.iteration(), &c.solver_iter_base(), &c.maxiter_arr(),
                       p.solver_stats_f, p.solver_stats_i, &c.out_nchange(), &c.out_improvement(),
                       &c.out_gradient(), &c.out_stop(), &c.out_reason(), &c.iteration(), &c.s08c_status(),
                       SubstepShape::nv, SubstepShape::rows, &c.changed_count(), mass_acceleration,
                       smooth_force, p.J, p.force, &execution);
      __syncthreads();
      if (c.s08c_status() || (c.out_stop() && (c.out_reason() == 1 || c.out_reason() == 2))) {
        if (threadIdx.x == 0) {
          if (c.s08c_status() && !c.err_code())
            c.err_code() = 3000 + c.s08c_status();
          else if (!c.s08c_status())
            c.stop_reason() = c.out_reason();
        }
        break;
      }
    }
  } while (false);

  CF_NEWTON_PROFILE_STAGE(Finalize);
  // Close every lane's status/active reads before the leader clears scratch
  // flags or prepares finalization. No lane leaves the collective early.
  __syncthreads();
  if (threadIdx.x == 0) {
    c.active() = 0;
    if (iteration_slot < 100) {
      c.s07_status() = 0;
      c.status() = 0;
      c.route() = 0;
    }
  }
#if LIBERO_NEWTON_TRACE
  if (iteration_slot < 100)
    seed_eval_block(0, p.eval_trace, p.eval_tag);
#endif
  // Finalize acceleration/warmstart and publish the solver outputs.
  if (threadIdx.x == 0)
    finalize_exit(0, &c.newton_world(), &c.err_code(), &c.solver_exit(), &c.solver_finalize_active());
  finalize_block(0, &c.solver_finalize_active(), &c.solver_exit(), &c.iteration(), qacc, warmstart,
                 &c.solver_iter(), &c.solver_nnz(), &c.s08d_status(), SubstepShape::nv);
  // These records have one writer. Publish once when other lanes consume &c.valid().
  if (threadIdx.x == 0) {
    if (c.solver_finalize_active() && c.s08d_status()) {
      if (!c.err_code())
        c.err_code() = NewtonError::finalize + c.s08d_status();
      c.solver_finalize_active() = 0;
    }
    if (entry_enabled[w] && execution.audit)
      execution.audit[6] += c.iteration();
    final_event(0, qacc, &c.empty(), &c.newton_world(), &c.solver_exit(), &c.err_code(), &c.s08d_status(),
                p.attempted, p.solver_candidate, p.solver_candidate_attempt, p.solver_candidate_completed,
                p.solver_candidate_origin, p.solver_source_stage, p.solver_rejected_final_entry,
                SubstepShape::nv);
    publish(0, entry_upstream + w, entry_enabled + w, &c.err_code(), &c.valid(), &c.written());
    publication_gate(0, &c.valid(), &c.diagnostic());
    finish_publication(0, &c.valid(), &c.written(), (&c.expected), p.n_written);
  }
  __syncthreads();
  if (c.valid()) {
    for (int i = threadIdx.x; i < SubstepShape::nv; i += blockDim.x) {
      p.canonical_qacc[i] = qacc[i];
      p.canonical_qacc_warmstart[i] = warmstart[i];
      p.canonical_qfrc_constraint[i] = constraint_force[i];
    }
    libero_newton_io::copy_block(p.force, p.external_r_ef, p.nefc[0]);
    libero_newton_io::copy_block(p.state, p.canonical_state, p.nefc[0]);
    if (threadIdx.x == 0) {
      p.canonical_solver_iter[0] = c.solver_iter();
      p.canonical_solver_nnz[0] = c.solver_nnz();
    }
    libero_newton_io::copy_block(p.solver_stats_f, p.canonical_solver_stats_f, 3 * min(c.iteration(), 1000));
    libero_newton_io::copy_block(p.solver_stats_i, p.canonical_solver_stats_i, 4 * min(c.iteration(), 1000));
  }
  if (threadIdx.x == 0) {
    p.err_code[0] = c.err_code();
    p.valid[0] = c.valid();
    p.diagnostic[0] = c.diagnostic();
  }
  __syncthreads();
  CF_NEWTON_PROFILE_VALUE(Iterations, c.iteration());
  CF_NEWTON_PROFILE_VALUE(StopReason, c.stop_reason());
  CF_NEWTON_PROFILE_VALUE(Error, c.err_code());
  CF_NEWTON_PROFILE_END();
}

} // namespace physics_newton


#if defined(FLASH_NEWTON_PROFILE)
namespace physics_newton::profile_detail {
__device__ Configuration configuration{};
}
extern "C" int flash_libero_newton_profile_configure(void *pointer, int capacity) {
  if (pointer && capacity <= 0) return int(cudaErrorInvalidValue);
  const physics_newton::profile_detail::Configuration config{static_cast<uint64_t *>(pointer), capacity};
  return int(cudaMemcpyToSymbol(physics_newton::profile_detail::configuration, &config, sizeof(config)));
}
extern "C" int flash_libero_newton_profile_header_cells() {
  return physics_newton::profile_detail::header_cells;
}
extern "C" int flash_libero_newton_profile_record_cells() {
  return physics_newton::profile_detail::record_cells;
}
extern "C" int flash_libero_newton_profile_phase_count() {
  return int(physics_newton::profile_detail::Phase::Count);
}
extern "C" const char *flash_libero_newton_profile_phase_name(int phase) {
  static const char *names[] = {
    "entry", "warm_jacobian", "warm_costs", "initial_residuals", "initial_constraint",
    "initial_transpose", "initial_assembly", "initial_factor", "initial_direction",
    "line_search", "commit_save", "iter_constraint", "iter_transpose", "iter_prepare",
    "iter_factor", "iter_direction", "statistics", "finalize", "total",
    "entry_schema", "coefficients", "search_prepare", "search_jacobian", "search_continue",
    "search_point", "changed_rows", "rank_updates", "rebuild", "cone_factor", "rank_update", "potrf", "search_quadratic", "jacobian_cache"
  };
  static_assert(sizeof(names)/sizeof(names[0]) == int(physics_newton::profile_detail::Phase::Count));
  return phase >= 0 && phase < int(physics_newton::profile_detail::Phase::Count) ? names[phase] : nullptr;
}
extern "C" int flash_libero_newton_profile_info_count() {
  return int(physics_newton::profile_detail::Info::Count);
}
extern "C" const char *flash_libero_newton_profile_info_name(int field) {
  static const char *names[] = {"rows", "contacts", "friction_rows", "iterations", "search_evaluations",
    "coefficient_refresh", "stop_reason", "error", "sm", "enabled", "layout_refresh"};
  static_assert(sizeof(names)/sizeof(names[0]) == int(physics_newton::profile_detail::Info::Count));
  return field >= 0 && field < int(physics_newton::profile_detail::Info::Count) ? names[field] : nullptr;
}
#endif

#undef CF_NEWTON_PROFILE_BEGIN
#undef CF_NEWTON_PROFILE_STAGE
#undef CF_NEWTON_PROFILE_VALUE
#undef CF_NEWTON_PROFILE_ADD
#undef CF_NEWTON_PROFILE_END
#undef CF_NEWTON_PROFILE_SCOPE

extern "C" int flash_libero_newton_atomic_reductions() { return FLASH_NEWTON_ATOMIC_REDUCTIONS; }
extern "C" size_t flash_libero_newton_device_workspace_bytes() {
  return physics_newton::NewtonDeviceWorkspaceLayout::bytes;
}
