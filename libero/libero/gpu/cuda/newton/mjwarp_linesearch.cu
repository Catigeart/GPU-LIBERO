// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Readable CUDA adaptation of MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf:
// support.py::mul_m_kernel, solver.py::_linesearch_jv_fused_kernel and
// _linesearch_iterative_kernel (dense, elliptic, full DOFs, nonincremental).
// Warp 1.18 tile reductions use descending shuffle-down then lane-0 broadcast.
#include "mjwarp_linesearch.cuh"
#include "mjwarp_products.cuh"
#include <cstdio>

#ifndef FLASH_MJW_LINESEARCH_SOURCE_ID
#error "Build through flash_mjwarp_linesearch_fp32"
#endif

namespace flash::mjwarp {
namespace {
constexpr unsigned FullWarp = 0xffffffffu;
constexpr int Elliptic = 7, LinesearchOverflow = 1 << 10;
constexpr float NoiseEps = 8.f * 1.1920929e-7f;

// One thread gathers a sparse symmetric mass row, in the upstream index order.
__global__ void mass_product(LinesearchInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.nv) return;
  int w = t / a.nv, i = t % a.nv;
  if (a.done[w]) return;
  a.mv[t] = mass_row_dot(i, a.mass_row, a.mass_col, a.mass_address,
                        a.mass + w * a.mass_entries, a.search + w * a.nv);
}

__global__ void zero_jv(LinesearchInputs a) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.rows) return;
  if (!a.done[t / a.rows] && t % a.rows < a.nefc[t / a.rows]) a.jv[t] = 0.f;
}

// Above NV=50 upstream assigns 20 DOFs per row-part, then atomic-adds the
// partials. Preserve that FP32 grouping, active-row bounds and skipped holds.
__global__ void jacobian_product(LinesearchInputs a, int parts, int chunk) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= a.worlds * a.rows * parts) return;
  int part = t % parts, row = (t / parts) % a.rows, w = t / (parts * a.rows);
  if (a.done[w] || row >= a.nefc[w]) return;
  float sum = jacobian_partial_dot(a.nv, part * chunk, chunk,
      a.jacobian + (w * a.rows + row) * a.pad, a.search + w * a.nv);
  if (parts == 1) a.jv[w * a.rows + row] = sum;
  else atomicAdd(a.jv + w * a.rows + row, sum);
}

void launch_jv(const LinesearchInputs& a, cudaStream_t stream) {
  int chunk = a.nv > 50 ? 20 : 50, parts = (a.nv + chunk - 1) / chunk;
  if (parts > 1) zero_jv<<<(a.worlds * a.rows + 255) / 256, 256, 0, stream>>>(a);
  jacobian_product<<<(a.worlds * a.rows * parts + 255) / 256, 256, 0, stream>>>(a, parts, chunk);
}

// Large-NV products share a world CTA and the current search vector. Jv has
// one writer per complete row, so active-row zeroing/partial atomics disappear.
template<int Threads,bool Shared>
__global__ void world_products(LinesearchInputs a) {
  const int w=blockIdx.x,x=threadIdx.x;
  if(a.done[w]) return;
  __shared__ float scratch[Shared ? 128 : 1];
  const float* selected=Shared ? scratch : a.search+w*a.nv;
  if constexpr(Shared) {
    for(int i=x;i<a.nv;i+=Threads) scratch[i]=a.search[w*a.nv+i];
    __syncthreads();
  }
  for(int i=x;i<a.nv;i+=Threads)
    a.mv[w*a.nv+i]=mass_row_dot(i,a.mass_row,a.mass_col,a.mass_address,a.mass+w*a.mass_entries,selected);
  const int lane=x%4;const unsigned mask=0xfu<<((x%32)/4*4);
  for(int row=x/4;row<min(a.rows,a.nefc[w]);row+=Threads/4) {
    const int at=w*a.rows+row;float value=0.f;
    for(int i=lane;i<a.nv;i+=4) value+=a.jacobian[at*a.pad+i]*selected[i];
    value+=__shfl_down_sync(mask,value,2,4);value+=__shfl_down_sync(mask,value,1,4);
    if(!lane) a.jv[at]=value;
  }
}

// A point is (cost, first derivative, second derivative). Quadratic coefficient
// triples are separately stored as [q0,q1,q2] in the existing ctx.quad array.
struct Point {
  float cost, slope, curvature;
  __device__ Point operator+(Point b) const {
    return {cost + b.cost, slope + b.slope, curvature + b.curvature};
  }
  __device__ void operator+=(Point b) { *this = *this + b; }
};

__device__ float safe_div(float x, float y) { return x / (y != 0.f ? y : 1.e-15f); }
__device__ float sum_warp(float x) {
  for (int d = 16; d > 0; d /= 2) x += __shfl_down_sync(FullWarp, x, d);
  return __shfl_sync(FullWarp, x, 0);
}
__device__ Point sum_warp(Point p) {
  return {sum_warp(p.cost), sum_warp(p.slope), sum_warp(p.curvature)};
}
__device__ Point quadratic(float q1, float q2, float alpha) {
  float aq2 = alpha * q2;
  return {alpha * aq2 + alpha * q1, 2.f * aq2 + q1, 2.f * q2};
}
__device__ bool in_bracket(Point x, Point y) {
  return (x.slope < y.slope && y.slope < 0.f) || (x.slope > y.slope && y.slope > 0.f);
}

// One row's loaded data. Only primary elliptic rows contribute to line cost;
// secondary rows contribute through their primary's cone coefficients.
struct Row {
  int kind;  // 0 equality, 1 frictionloss, 2 inequality, 3 primary cone, 4 secondary
  float x, v, d, loss;
  float q0, q1, q2, u0, v0, uu, uv, vv, dm, mu;
  float cost0, t0, r0;
  int state0;
};

__device__ Row load_row(const LinesearchInputs& a, int w, int row) {
  Row r{};
  int at = w * a.rows + row;
  if (row < a.ne[w]) r.kind = 0;
  else if (row < a.ne[w] + a.nf[w]) r.kind = 1;
  else if (a.row_type[at] != Elliptic) r.kind = 2;
  else {
    int c = a.row_contact[at];
    if (row != a.contact_address[c * 6]) { r.kind = 4; return r; }
    r.kind = 3;
    int r1 = w * a.rows + a.contact_address[c * 6 + 1];
    int r2 = w * a.rows + a.contact_address[c * 6 + 2];
    r.q0 = a.quad[3 * at]; r.q1 = a.quad[3 * at + 1]; r.q2 = a.quad[3 * at + 2];
    r.u0 = a.quad[3 * r1]; r.v0 = a.quad[3 * r1 + 1]; r.uu = a.quad[3 * r1 + 2];
    r.uv = a.quad[3 * r2]; r.vv = a.quad[3 * r2 + 1]; r.dm = a.quad[3 * r2 + 2];
    r.mu = a.contact_friction[c * 5] * a.impratio[w % a.impratio_count];
    if (r.uu <= 0.f) {
      r.state0 = r.u0 < 0.f ? 1 : 0;
      r.cost0 = r.state0 == 1 ? r.q0 : 0.f;
    } else {
      r.t0 = sqrtf(r.uu);
      if (r.u0 >= r.mu * r.t0) r.state0 = 0;
      else if (r.mu * r.u0 + r.t0 <= 0.f) { r.state0 = 1; r.cost0 = r.q0; }
      else { r.state0 = 4; r.r0 = r.u0 - r.mu * r.t0; r.cost0 = 0.5f * r.dm * r.r0 * r.r0; }
    }
    return r;
  }
  r.x = a.jaref[at]; r.v = a.jv[at]; r.d = a.diagonal[at];
  if (r.kind == 1) r.loss = a.frictionloss[at];
  return r;
}

__device__ Point direct_zero(const Row& r) {
  float vd = r.v * r.d;
  return {0.5f * r.d * r.x * r.x, vd * r.x, r.v * vd};
}
__device__ Point friction_point(const Row& r, float x) {
  float rf = safe_div(r.loss, r.d);
  if (-rf < x && x < rf) {
    float vd = r.v * r.d;
    return {0.5f * r.d * x * x, vd * x, r.v * vd};
  }
  if (x <= -rf) return {r.loss * (-0.5f * rf - x), -r.loss * r.v, 0.f};
  return {r.loss * (-0.5f * rf + x), r.loss * r.v, 0.f};
}
__device__ Point evaluate_zero(const Row& r) {
  if (r.kind == 4) return {};
  if (r.kind == 3) {
    if (r.state0 == 1) return {r.q0, r.q1, 2.f * r.q2};
    if (r.state0 == 4) {
      float ti = 1.f / r.t0, t1 = r.uv * ti, t2 = (r.vv - t1 * t1) * ti;
      float r1 = r.v0 - r.mu * t1;
      return {r.cost0, r.dm * r.r0 * r1, r.dm * (r1 * r1 - r.mu * r.r0 * t2)};
    }
    return {};
  }
  if (r.kind == 1) return friction_point(r, r.x);
  if (r.kind == 2 && r.x >= 0.f) return {};
  return direct_zero(r);
}

__device__ Point cone_quadratic_shifted(const Row& r, float alpha, float n, float tsqr) {
  float aq2 = alpha * r.q2;
  float cost = alpha * (aq2 + r.q1);
  if (r.state0 == 4) {
    float boundary = r.mu * r.u0 + r.t0;
    cost += 0.5f * r.dm * boundary * boundary;
  } else if (r.state0 == 0)
    cost = 0.5f * r.dm * (1.f + r.mu * r.mu) * (n * n + fmaxf(tsqr, 0.f));
  return {cost, 2.f * aq2 + r.q1, 2.f * r.q2};
}

__device__ Point evaluate_shifted(const Row& r, float alpha) {
  if (r.kind == 4) return {};
  if (r.kind == 3) {
    float n = r.u0 + alpha * r.v0;
    float tsqr_delta = alpha * (2.f * r.uv + alpha * r.vv), tsqr = r.uu + tsqr_delta;
    if (tsqr <= 0.f) {
      if (n < 0.f) return cone_quadratic_shifted(r, alpha, n, tsqr);
    } else {
      float t = sqrtf(tsqr);
      if (n >= r.mu * t) {}
      else if (r.mu * n + t <= 0.f) return cone_quadratic_shifted(r, alpha, n, tsqr);
      else {
        float ti = 1.f / t, t1 = (r.uv + alpha * r.vv) * ti;
        float t2 = (r.vv - t1 * t1) * ti;
        float residual = n - r.mu * t, r1 = r.v0 - r.mu * t1, cost;
        if (r.state0 == 4) {
          // Rationalized norm difference: essential for a small FP32 step.
          float tdelta = tsqr_delta / (t + r.t0), rdelta = alpha * r.v0 - r.mu * tdelta;
          cost = 0.5f * r.dm * rdelta * (2.f * r.r0 + rdelta);
        } else if (r.state0 == 1) {
          float aq2 = alpha * r.q2, boundary = r.mu * n + t;
          cost = alpha * (aq2 + r.q1) - 0.5f * r.dm * boundary * boundary;
        } else cost = 0.5f * r.dm * residual * residual;
        return {cost, r.dm * residual * r1, r.dm * (r1 * r1 + residual * (-r.mu * t2))};
      }
    }
    return {-r.cost0, 0.f, 0.f};
  }
  if (r.kind == 1) {
    Point pt = friction_point(r, r.x + alpha * r.v);
    pt.cost -= friction_point(r, r.x).cost;
    return pt;
  }
  float quad0 = 0.5f * r.d * r.x * r.x;
  float cost0 = r.kind == 2 && r.x >= 0.f ? 0.f : quad0;
  if (r.kind == 2 && r.x + alpha * r.v >= 0.f) return {-cost0, 0.f, 0.f};
  float vd = r.v * r.d, hessian = r.v * vd, ah = alpha * hessian;
  return {alpha * (vd * r.x + 0.5f * ah) + (quad0 - cost0), vd * r.x + ah, hessian};
}

// One warp per world, matching the admitted MJWarp block dimension. Scalar
// search state is replicated per lane; no thread-0 state mailbox/barriers.
template<bool PrepareMass=false>
__global__ void iterative_search(LinesearchInputs a) {
  int w = blockIdx.x, tid = threadIdx.x;
  if (a.done[w]) return;
  int ne = a.ne[w], count = min(a.rows, a.nefc[w]);
  if constexpr(PrepareMass)
    for(int i=tid;i<a.nv;i+=32)
      a.mv[w*a.nv+i]=mass_row_dot(i,a.mass_row,a.mass_col,a.mass_address,
                                a.mass+w*a.mass_entries,a.search+w*a.nv);
  if (a.nv <= 50) {
    for (int row = tid; row < count; row += 32) {
      float sum = jacobian_partial_dot(a.nv, 0, a.nv,
          a.jacobian + (w * a.rows + row) * a.pad, a.search + w * a.nv);
      a.jv[w * a.rows + row] = sum;
    }
    __syncthreads();
  }
  float impratio = a.impratio[w % a.impratio_count];
  for (int row = tid; row < count; row += 32) {
    int at = w * a.rows + row;
    if (a.row_type[at] != Elliptic) continue;
    int c = a.row_contact[at];
    if (c >= a.contact_count[0] || row != a.contact_address[c * 6]) continue;
    float x = a.jaref[at], v = a.jv[at], d = a.diagonal[at], vd = v * d;
    float q0 = 0.5f * x * x * d, q1 = vd * x, q2 = 0.5f * v * vd;
    float mu = a.contact_friction[c * 5] * impratio;
    float u0 = x * mu, v0 = v * mu, uu = 0.f, uv = 0.f, vv = 0.f;
    for (int j = 1; j < a.contact_dimension[c]; ++j) {
      int next = a.contact_address[c * 6 + j];
      if (next < 0) continue;
      int t = w * a.rows + next;
      float jv = a.jv[t], jaref = a.jaref[t], dj = a.diagonal[t], djx = dj * jaref;
      q0 += 0.5f * jaref * djx; q1 += jv * djx; q2 += 0.5f * jv * dj * jv;
      float friction = a.contact_friction[c * 5 + j - 1];
      float uj = jaref * friction, vj = jv * friction;
      uu += uj * uj; uv += uj * vj; vv += vj * vj;
    }
    int one = w * a.rows + a.contact_address[c * 6 + 1];
    int two = w * a.rows + a.contact_address[c * 6 + 2];
    a.quad[3 * at] = q0; a.quad[3 * at + 1] = q1; a.quad[3 * at + 2] = q2;
    a.quad[3 * one] = u0; a.quad[3 * one + 1] = v0; a.quad[3 * one + 2] = uu;
    a.quad[3 * two] = uv; a.quad[3 * two + 1] = vv;
    float mu2 = mu * mu;
    a.quad[3 * two + 2] = d / (mu2 * (1.f + mu2));
  }
  __syncthreads();
  float tolerance = a.tolerance[w % a.tolerance_count], lstol = a.ls_tolerance[w % a.ls_tolerance_count];
  float scale = a.inertia[w % a.inertia_count] * float(a.nv);
  float gtol = fmaxf(tolerance * lstol * sqrtf(a.search_dot[w]) * scale, 1.e-6f);
  float gauss1 = 0.f, gauss2 = 0.f;
  for (int i = tid; i < a.nv; i += 32) {
    int at = w * a.nv + i;
    gauss1 += a.search[at] * (a.ma[at] - a.smooth[at]);
    gauss2 += 0.5f * a.search[at] * a.mv[at];
  }
  for (int row = tid; row < ne; row += 32) {
    int at = w * a.rows + row;
    float vd = a.jv[at] * a.diagonal[at];
    gauss1 += vd * a.jaref[at]; gauss2 += 0.5f * a.jv[at] * vd;
  }
  gauss1 = sum_warp(gauss1); gauss2 = sum_warp(gauss2);
  Point local{};
  for (int row = ne + tid; row < count; row += 32) local += evaluate_zero(load_row(a, w, row));
  Point rows = sum_warp(local), p0 = Point{0.f, gauss1, 2.f * gauss2} + rows;
  Point delta{0.f, p0.slope, p0.curvature};
  float q1_abs = sqrtf(2.f * fmaxf(rows.cost, 0.f) * fmaxf(rows.curvature, 0.f)) + fabsf(gauss1);
  float gtol_accept = fmaxf(gtol, NoiseEps * q1_abs), alpha = 0.f, improvement = 0.f;
  bool converged = fabsf(p0.slope) < NoiseEps * q1_abs;
  float initial_alpha = -safe_div(p0.slope, p0.curvature);
  Point initial{};
  if (!converged) {
    local = {};
    for (int row = ne + tid; row < count; row += 32) local += evaluate_shifted(load_row(a, w, row), initial_alpha);
    initial = quadratic(gauss1, gauss2, initial_alpha) + sum_warp(local);
    if (fabsf(initial.slope) < gtol_accept && initial.cost < 0.f) {
      converged = true; alpha = initial_alpha; improvement = -initial.cost;
    }
  }
  if (!converged) {
    bool less = initial.slope < p0.slope;
    Point lo = less ? initial : delta, hi = less ? delta : initial;
    float lo_alpha = less ? initial_alpha : 0.f, hi_alpha = less ? 0.f : initial_alpha;
    for (int iteration = 0; iteration < a.iterations; ++iteration) {
      float ln = lo_alpha - safe_div(lo.slope, lo.curvature);
      float hn = hi_alpha - safe_div(hi.slope, hi.curvature);
      float mid_alpha = 0.5f * (lo_alpha + hi_alpha);
      Point ll{}, lh{}, lm{};
      for (int row = ne + tid; row < count; row += 32) {
        Row r = load_row(a, w, row);
        ll += evaluate_shifted(r, ln); lh += evaluate_shifted(r, hn); lm += evaluate_shifted(r, mid_alpha);
      }
      Point next_lo = quadratic(gauss1, gauss2, ln) + sum_warp(ll);
      Point next_hi = quadratic(gauss1, gauss2, hn) + sum_warp(lh);
      Point mid = quadratic(gauss1, gauss2, mid_alpha) + sum_warp(lm);
      bool conv_lo = fabsf(next_lo.slope) < gtol_accept && next_lo.cost < 0.f;
      bool conv_hi = fabsf(next_hi.slope) < gtol_accept && next_hi.cost < 0.f;
      bool conv_mid = fabsf(mid.slope) < gtol_accept && mid.cost < 0.f;
      bool any = conv_lo || conv_hi || conv_mid, swap_lo = false, swap_hi = false;
      if (any) {
        Point best{1.e10f, 0.f, 0.f}; float best_alpha = 0.f;
        if (conv_lo && next_lo.cost < best.cost) { best = next_lo; best_alpha = ln; }
        if (conv_hi && next_hi.cost < best.cost) { best = next_hi; best_alpha = hn; }
        if (conv_mid && mid.cost < best.cost) { best = mid; best_alpha = mid_alpha; }
        lo = hi = best; lo_alpha = hi_alpha = best_alpha;
      } else {
        bool s = in_bracket(lo, next_lo);
        if (s) { lo = next_lo; lo_alpha = ln; } swap_lo |= s;
        s = in_bracket(lo, mid);
        if (s) { lo = mid; lo_alpha = mid_alpha; } swap_lo |= s;
        s = in_bracket(lo, next_hi);
        if (s) { lo = next_hi; lo_alpha = hn; } swap_lo |= s;
        s = in_bracket(hi, next_hi) || (hi.slope < 0.f && next_hi.slope > 0.f);
        if (s) { hi = next_hi; hi_alpha = hn; } swap_hi |= s;
        s = in_bracket(hi, mid);
        if (s) { hi = mid; hi_alpha = mid_alpha; } swap_hi |= s;
        s = in_bracket(hi, next_lo);
        if (s) { hi = next_lo; hi_alpha = ln; } swap_hi |= s;
      }
      bool ls_done = any || (!swap_lo && !swap_hi)
          || (lo.cost < 0.f && lo.slope < 0.f && lo.slope > -gtol)
          || (hi.cost < 0.f && hi.slope > 0.f && hi.slope < gtol);
      if (lo.cost < 0.f || hi.cost < 0.f) {
        bool best_lo = lo.cost < hi.cost;
        alpha = best_lo ? lo_alpha : hi_alpha;
        improvement = -(best_lo ? lo.cost : hi.cost);
      }
      if (ls_done) { converged = true; break; }
    }
  }
  for (int i = tid; i < a.nv; i += 32) {
    int at = w * a.nv + i;
    a.qacc[at] += alpha * a.search[at]; a.ma[at] += alpha * a.mv[at];
  }
  for (int row = tid; row < count; row += 32) {
    int at = w * a.rows + row;
    a.jaref[at] += alpha * a.jv[at];
  }
  if (tid == 0) {
    a.improvement[w] = improvement; a.alpha[w] = alpha;
    if (!converged) {
      if (a.warn_overflow & LinesearchOverflow)
        printf("linesearch iterations limit reached - please increase ls_iterations beyond %u\n"
               "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.LS_ITERATIONS (or = 0 for all)\n", a.iterations);
      atomicOr(a.overflow + w, LinesearchOverflow);
    }
  }
}
}  // namespace
}  // namespace flash::mjwarp

extern "C" {
const char* flash_mjwarp_linesearch_source_id() { return FLASH_MJW_LINESEARCH_SOURCE_ID; }
std::size_t flash_mjwarp_linesearch_arguments_bytes() { return sizeof(flash::mjwarp::LinesearchInputs); }
int flash_mjwarp_linesearch_arch() { return FLASH_MJW_LINESEARCH_ARCH; }
int flash_mjwarp_linesearch_launch(const flash::mjwarp::LinesearchInputs* input, int stage, cudaStream_t stream) {
  if (!input || stage < 0 || stage > 3) return int(cudaErrorInvalidValue);
  const auto& a = *input;
  if (a.worlds < 1 || a.nv < 1 || a.rows < 1) return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (stage == 0 || stage == 1) {
    mass_product<<<(a.worlds * a.nv + 255) / 256, 256, 0, stream>>>(a);
    if (a.nv > 50) launch_jv(a, stream);
  }
  if (stage == 3) launch_jv(a, stream);
  if (stage == 0 || stage == 2) iterative_search<false><<<a.worlds, 32, 0, stream>>>(a);
  return int(cudaGetLastError());
}
int flash_mjwarp_linesearch_fused(const flash::mjwarp::LinesearchInputs* input,cudaStream_t stream) {
  if(!input || input->worlds<1 || input->nv<1 || input->rows<1 || input->pad<input->nv) return cudaErrorInvalidValue;
  const auto& a=*input;using namespace flash::mjwarp;
  if(a.nv<=50) iterative_search<true><<<a.worlds,32,0,stream>>>(a);
  else {
    if(a.nv<=128) world_products<1024,true><<<a.worlds,1024,0,stream>>>(a);
    else world_products<1024,false><<<a.worlds,1024,0,stream>>>(a);
    iterative_search<false><<<a.worlds,32,0,stream>>>(a);
  }
  return cudaGetLastError();
}

}
