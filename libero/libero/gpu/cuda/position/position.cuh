#include "precision.cuh"
#pragma once
#ifdef LIBERO_POSITION_DATAFLOW
#define LIBERO_POSITION_STAGE_INLINE __forceinline__
#else
#define LIBERO_POSITION_STAGE_INLINE
#endif
#ifndef LIBERO_POSITION_BODY_DIAGNOSTICS
#define LIBERO_POSITION_BODY_DIAGNOSTICS 1
#endif
// Handwritten MuJoCo rigid-body position recurrence. Arrays retain world-major ownership.
namespace libero_position {
// Intermediate traces are optional. body_norm_branch remains a live error
// carrier in both profiles and is never disabled with diagnostic recording.
constexpr bool body_diagnostics = LIBERO_POSITION_BODY_DIAGNOSTICS != 0;
// Fixed local vectors must remain scalar expressions: converting their
// addresses to integers for overlap/alignment tests can materialize them in
// thread-local memory. Small copies keep the original forward-copy order.
__device__ __forceinline__ void copy(real_t *d, const real_t *s, int n) {
  if (n <= 16) {
    for (int i = 0; i < n; ++i)
      d[i] = s[i];
    return;
  }
  const uintptr_t db = reinterpret_cast<uintptr_t>(d), sb = reinterpret_cast<uintptr_t>(s);
  const size_t bytes = size_t(n) * sizeof(real_t);
  if (db != sb && (db > sb ? db - sb : sb - db) < bytes) {
    for (int i = 0; i < n; ++i)
      d[i] = s[i];
  } else
    libero_vec::copy_serial(s, d, n);
}
__device__ __forceinline__ void zero(real_t *d, int n) {
  if (n <= 16) {
    for (int i = 0; i < n; ++i)
      d[i] = REAL_C(0.0);
    return;
  }
  libero_vec::fill_serial(d, n, REAL_C(0.0));
}
__device__ __forceinline__ bool finite(const real_t *d, int n) {
  if (n <= 16) {
    bool ok = true;
    for (int i = 0; i < n; ++i)
      ok &= isfinite(d[i]);
    return ok;
  }
  return !libero_vec::any_nonfinite_lane(d, n, 0, 1);
}
__device__ __forceinline__ bool identity(const real_t *q) {
  return q[0] == 1 && q[1] == 0 && q[2] == 0 && q[3] == 0;
}
__device__ __forceinline__ real_t norm(const real_t *q) {
  return sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}
// CPU mju_normalize4 uses one reciprocal. Preserve the tiny-norm fallback.
// These free-object dimension families contain resting thin book contacts.
// Its end-state force/pose error is sensitive to reciprocal multiplication;
// retain the previous GPU division order there. Other families preprocess to
// the same arithmetic as before this compatibility qualification.
__device__ __forceinline__ void normalize(real_t *q, real_t n) {
  if (n < REAL_C(1e-15)) {
    q[0] = 1;
    q[1] = q[2] = q[3] = 0;
  }
#if defined(CF_PM_STATIC_NV) && (CF_PM_STATIC_NV == 21 || CF_PM_STATIC_NV == 27)
  else {
    for (int k = 0; k < 4; ++k)
      q[k] /= n;
  }
#else
  else if constexpr (body_diagnostics) {
    for (int k = 0; k < 4; ++k)
      q[k] /= n;
  } else if (n != REAL_C(1.0)) {
    const real_t scale = REAL_C(1.0) / n;
    for (int k = 0; k < 4; ++k)
      q[k] *= scale;
  }
#endif
}
__device__ __forceinline__ void mulquat(const real_t *a, const real_t *b, real_t *c) {
  const real_t aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const real_t bw = b[0], bx = b[1], by = b[2], bz = b[3];
  const real_t w = aw * bw - ax * bx - ay * by - az * bz;
  const real_t x = aw * bx + ax * bw + ay * bz - az * by;
  const real_t y = aw * by + ay * bw + az * bx - ax * bz;
  const real_t z = aw * bz + az * bw + ax * by - ay * bx;
  c[0] = w;
  c[1] = x;
  c[2] = y;
  c[3] = z;
}
__device__ inline void rotquat(const real_t *v, const real_t *q, real_t *o) {
  if (identity(q)) {
    copy(o, v, 3);
    return;
  }
  real_t t[3] = {q[0] * v[0] + q[2] * v[2] - q[3] * v[1], q[0] * v[1] + q[3] * v[0] - q[1] * v[2],
                 q[0] * v[2] + q[1] * v[1] - q[2] * v[0]};
  real_t r[3] = {v[0] + 2 * (q[2] * t[2] - q[3] * t[1]), v[1] + 2 * (q[3] * t[0] - q[1] * t[2]),
                 v[2] + 2 * (q[1] * t[1] - q[2] * t[0])};
  copy(o, r, 3);
}

__device__ __forceinline__ void rotmat(const real_t *a, const real_t *v, real_t *o) {
  const real_t x = v[0], y = v[1], z = v[2];
  const real_t r0 = a[0] * x + a[1] * y + a[2] * z;
  const real_t r1 = a[3] * x + a[4] * y + a[5] * z;
  const real_t r2 = a[6] * x + a[7] * y + a[8] * z;
  o[0] = r0;
  o[1] = r1;
  o[2] = r2;
}
__device__ __forceinline__ void quatmat(const real_t *q, real_t *a) {
  if (identity(q)) {
    zero(a, 9);
    a[0] = a[4] = a[8] = 1;
    return;
  }
  const real_t w = q[0], x = q[1], y = q[2], z = q[3];
  a[0] = w * w + x * x - y * y - z * z;
  a[4] = w * w - x * x + y * y - z * z;
  a[8] = w * w - x * x - y * y + z * z;
  a[1] = 2 * (x * y - w * z);
  a[2] = 2 * (x * z + w * y);
  a[3] = 2 * (x * y + w * z);
  a[5] = 2 * (y * z - w * x);
  a[6] = 2 * (x * z - w * y);
  a[7] = 2 * (y * z + w * x);
}
struct BodyInputs {
  real_t *qpos0;
  real_t *bpos;
  real_t *bquat;
  real_t *jaxis;
  real_t *jpos;
  real_t *qtmp;
  real_t *mqtmp;
  real_t *mocap_pos;

  int *parent;
  int *jnum;
  int *jadr;
  int *mocapid;
  int *jtype;
  int *qadr;
};
struct BodyOutputs {
  real_t *xpos;
  real_t *xquat;
  real_t *xmat;
  real_t *xanchor;
  real_t *xaxis;
  real_t *pre_pos;
  real_t *pre_quat;
  real_t *joint_pos;
  real_t *joint_quat;
  real_t *body_norm_before;
  real_t *trig_arg;
  real_t *trig_sin;
  real_t *trig_cos;

  int *body_route;
  int *joint_route;
  int *body_mat_identity;
  int *joint_rot_identity;
  int *hinge_zero;
  int *offcenter_rot_identity;
  int *body_norm_branch;
};
__device__ __forceinline__ int body_one(int b, const BodyInputs &in, const BodyOutputs &out) {
  real_t *qpos0 = in.qpos0;
  real_t *bpos = in.bpos;
  real_t *bquat = in.bquat;
  real_t *jaxis = in.jaxis;
  real_t *jpos = in.jpos;
  real_t *qtmp = in.qtmp;
  real_t *mqtmp = in.mqtmp;
  real_t *mocap_pos = in.mocap_pos;
  int *parent = in.parent;
  int *jnum = in.jnum;
  int *jadr = in.jadr;
  int *mocapid = in.mocapid;
  int *jtype = in.jtype;
  int *qadr = in.qadr;
  real_t *xpos = out.xpos;
  real_t *xquat = out.xquat;
  real_t *xmat = out.xmat;
  real_t *xanchor = out.xanchor;
  real_t *xaxis = out.xaxis;
  real_t *pre_pos = out.pre_pos;
  real_t *pre_quat = out.pre_quat;
  real_t *joint_pos = out.joint_pos;
  real_t *joint_quat = out.joint_quat;
  real_t *body_norm_before = out.body_norm_before;
  real_t *trig_arg = out.trig_arg;
  real_t *trig_sin = out.trig_sin;
  real_t *trig_cos = out.trig_cos;
  int *body_route = out.body_route;
  int *joint_route = out.joint_route;
  int *body_mat_identity = out.body_mat_identity;
  int *joint_rot_identity = out.joint_rot_identity;
  int *hinge_zero = out.hinge_zero;
  int *offcenter_rot_identity = out.offcenter_rot_identity;
  int *body_norm_branch = out.body_norm_branch;
  real_t p[3], q[4];
  int a = jadr[b];
  if (jnum[b] == 1 && jtype[a] == 0) {
    int qa = qadr[a];
    copy(p, qtmp + qa, 3);
    copy(q, qtmp + qa + 3, 4);
    if constexpr (body_diagnostics) {
      body_route[b] = 1;
      copy(pre_pos + 3 * b, p, 3);
      copy(pre_quat + 4 * b, q, 4);
    }
    copy(xanchor + 3 * a, p, 3);
    xaxis[3 * a + 2] = 1;
    if constexpr (body_diagnostics) {
      copy(joint_pos + 3 * a, p, 3);
      copy(joint_quat + 4 * a, q, 4);
    }
  } else {
    int pid = parent[b], mi = mocapid[b];
    const real_t *bp = mi >= 0 ? mocap_pos + 3 * mi : bpos + 3 * b;
    const real_t *bq = mi >= 0 ? mqtmp + 4 * mi : bquat + 4 * b;
    if (body_diagnostics && mi >= 0)
      body_route[b] = 2;
    rotmat(xmat + 9 * pid, bp, p);
    for (int k = 0; k < 3; ++k)
      p[k] += xpos[3 * pid + k];
    mulquat(xquat + 4 * pid, bq, q);
    if constexpr (body_diagnostics) {
      copy(pre_pos + 3 * b, p, 3);
      copy(pre_quat + 4 * b, q, 4);
    }
    for (int j = a; j < a + jnum[b]; ++j) {
      int t = jtype[j], qa = qadr[j];
      real_t axis[3], anchor[3], v[3];
      if constexpr (body_diagnostics)
        joint_rot_identity[j] = identity(q);
      rotquat(jaxis + 3 * j, q, axis);
      rotquat(jpos + 3 * j, q, anchor);
      for (int k = 0; k < 3; ++k)
        anchor[k] += p[k];
      if (!finite(axis, 3) || !finite(anchor, 3)) {
        return 5;
      }
      if (t == 2) {
        real_t d = qtmp[qa] - qpos0[qa];
        for (int k = 0; k < 3; ++k)
          p[k] += axis[k] * d;
      } else if (t == 1 || t == 3) {
        real_t local[4];
        if (t == 1)
          copy(local, qtmp + qa, 4);
        else {
          real_t angle = qtmp[qa] - qpos0[qa];
          if (angle == 0) {
            if constexpr (body_diagnostics)
              hinge_zero[j] = 1;
            local[0] = 1;
            local[1] = local[2] = local[3] = 0;
          } else {
            real_t z = angle * REAL_C(.5);
            if (!isfinite(z)) {
              return 2;
            }
            real_t sn = sin(z), cs = cos(z);
            if (!isfinite(sn) || !isfinite(cs)) {
              return 5;
            }
            if constexpr (body_diagnostics) {
              trig_arg[j] = z;
              trig_sin[j] = sn;
              trig_cos[j] = cs;
            }
            local[0] = cs;
            for (int k = 0; k < 3; ++k)
              local[k + 1] = jaxis[3 * j + k] * sn;
          }
        }
        mulquat(q, local, q);
        if constexpr (body_diagnostics)
          offcenter_rot_identity[j] = identity(q);
        rotquat(jpos + 3 * j, q, v);
        for (int k = 0; k < 3; ++k)
          p[k] += (anchor[k] - p[k]) - v[k];
      } else {
        return 3;
      }
      if (!finite(p, 3) || !finite(q, 4)) {
        return 5;
      }
      copy(xanchor + 3 * j, anchor, 3);
      copy(xaxis + 3 * j, axis, 3);
      if constexpr (body_diagnostics) {
        copy(joint_pos + 3 * j, p, 3);
        copy(joint_quat + 4 * j, q, 4);
        joint_route[j] = t;
      }
    }
  }
  real_t n = norm(q);
  if (!finite(p, 3) || !isfinite(n)) {
    return 5;
  }
  if constexpr (body_diagnostics)
    body_norm_before[b] = n;
  body_norm_branch[b] = (n < REAL_C(1e-15));
  normalize(q, n);
  copy(xpos + 3 * b, p, 3);
  copy(xquat + 4 * b, q, 4);
  if constexpr (body_diagnostics)
    body_mat_identity[b] = identity(q);
  quatmat(q, xmat + 9 * b);
  if (!finite(xmat + 9 * b, 9)) {
    return 5;
  }
  return 0;
}

// Scalar ordered body entry retained as fallback/reference.
__device__ LIBERO_POSITION_STAGE_INLINE void
body(int w, real_t *qpos0, int *parent, int *jnum, int *jadr, int *mocapid, real_t *bpos,
     real_t *bquat, int *jtype, int *qadr, real_t *jaxis, real_t *jpos, int nq, int nb, int nj,
     int nm, int *active, real_t *qpos, real_t *mocap_pos, real_t *mocap_quat, real_t *qtmp,
     real_t *mqtmp, real_t *xpos, real_t *xquat, real_t *xmat, real_t *xanchor, real_t *xaxis,
     real_t *world_xipos, real_t *world_ximat, int *status, real_t *norm_before,
     real_t *mocap_before, real_t *pre_pos, real_t *pre_quat, real_t *joint_pos, real_t *joint_quat,
     real_t *body_norm_before, int *body_route, int *joint_route, int *body_mat_identity,
     int *joint_rot_identity, int *hinge_zero, int *offcenter_rot_identity, real_t *trig_arg,
     real_t *trig_sin, real_t *trig_cos, int *norm_branch, int *mocap_branch, int *body_norm_branch,
     bool initialize_only = false) {
  qpos += w * nq;
  qtmp += w * nq;
  mocap_pos += w * nm * 3;
  mocap_quat += w * nm * 4;
  mqtmp += w * nm * 4;
  xpos += w * nb * 3;
  xquat += w * nb * 4;
  xmat += w * nb * 9;
  xanchor += w * nj * 3;
  xaxis += w * nj * 3;
  world_xipos += w * 3;
  world_ximat += w * 9;
  norm_before += w * nj;
  mocap_before += w * nm;
  pre_pos += w * nb * 3;
  pre_quat += w * nb * 4;
  joint_pos += w * nj * 3;
  joint_quat += w * nj * 4;
  body_norm_before += w * nb;
  body_route += w * nb;
  joint_route += w * nj;
  body_mat_identity += w * nb;
  joint_rot_identity += w * nj;
  hinge_zero += w * nj;
  offcenter_rot_identity += w * nj;
  trig_arg += w * nj;
  trig_sin += w * nj;
  trig_cos += w * nj;
  norm_branch += w * nj;
  mocap_branch += w * nm;
  body_norm_branch += w * nb;
  // Every owned field is cleared even on an inactive world. Caller normalization commits only on
  // success.
  zero(xpos, nb * 3);
  zero(xquat, nb * 4);
  zero(xmat, nb * 9);
  zero(xanchor, nj * 3);
  zero(xaxis, nj * 3);
  zero(qtmp, nq);
  zero(mqtmp, nm * 4);
  zero(world_xipos, 3);
  zero(world_ximat, 9);
  if constexpr (body_diagnostics) {
    zero(norm_before, nj);
    zero(mocap_before, nm);
    zero(pre_pos, nb * 3);
    zero(pre_quat, nb * 4);
    zero(joint_pos, nj * 3);
    zero(joint_quat, nj * 4);
    zero(body_norm_before, nb);
    zero(trig_arg, nj);
    zero(trig_sin, nj);
    zero(trig_cos, nj);
  }
  for (int i = 0; i < nb; ++i) {
    body_norm_branch[i] = 0;
    if constexpr (body_diagnostics)
      body_route[i] = body_mat_identity[i] = 0;
  }
  for (int i = 0; i < nj; ++i)
    if constexpr (body_diagnostics)
      joint_route[i] = joint_rot_identity[i] = hinge_zero[i] = offcenter_rot_identity[i] =
          norm_branch[i] = 0;
  for (int i = 0; i < nm; ++i)
    if constexpr (body_diagnostics)
      mocap_branch[i] = 0;
  status[w] = 0;
  if (active[w] == 0)
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  if (!finite(qpos, nq) || !finite(mocap_pos, nm * 3) || !finite(mocap_quat, nm * 4)) {
    status[w] = 4;
    return;
  }
  for (int j = 0; j < nj; ++j)
    if (jtype[j] == 3 && !isfinite((qpos[qadr[j]] - qpos0[qadr[j]]) * REAL_C(.5))) {
      status[w] = 2;
      return;
    }
  xquat[0] = 1;
  quatmat(xquat, xmat);
  world_ximat[0] = world_ximat[4] = world_ximat[8] = 1;
  copy(qtmp, qpos, nq);
  copy(mqtmp, mocap_quat, nm * 4);
  for (int j = 0; j < nj; ++j)
    if (jtype[j] == 0 || jtype[j] == 1) {
      int a = qadr[j] + (jtype[j] == 0 ? 3 : 0);
      real_t n = norm(qtmp + a);
      if (!isfinite(n)) {
        status[w] = 5;
        return;
      }
      if constexpr (body_diagnostics) {
        norm_before[j] = n;
        norm_branch[j] = (n < REAL_C(1e-15));
      }
      normalize(qtmp + a, n);
    }
  for (int i = 0; i < nm; ++i) {
    real_t n = norm(mqtmp + 4 * i);
    if (!isfinite(n)) {
      status[w] = 5;
      return;
    }
    if constexpr (body_diagnostics) {
      mocap_before[i] = n;
      mocap_branch[i] = (n < REAL_C(1e-15));
    }
    normalize(mqtmp + 4 * i, n);
  }
  if (initialize_only)
    return;
  for (int b = 1; b < nb; ++b) {
    real_t p[3], q[4];
    int a = jadr[b];
    if (jnum[b] == 1 && jtype[a] == 0) {
      int qa = qadr[a];
      copy(p, qtmp + qa, 3);
      copy(q, qtmp + qa + 3, 4);
      if constexpr (body_diagnostics) {
        body_route[b] = 1;
        copy(pre_pos + 3 * b, p, 3);
        copy(pre_quat + 4 * b, q, 4);
      }
      copy(xanchor + 3 * a, p, 3);
      xaxis[3 * a + 2] = 1;
      if constexpr (body_diagnostics) {
        copy(joint_pos + 3 * a, p, 3);
        copy(joint_quat + 4 * a, q, 4);
      }
    } else {
      int pid = parent[b], mi = mocapid[b];
      const real_t *bp = mi >= 0 ? mocap_pos + 3 * mi : bpos + 3 * b;
      const real_t *bq = mi >= 0 ? mqtmp + 4 * mi : bquat + 4 * b;
      if (body_diagnostics && mi >= 0)
        body_route[b] = 2;
      rotmat(xmat + 9 * pid, bp, p);
      for (int k = 0; k < 3; ++k)
        p[k] += xpos[3 * pid + k];
      mulquat(xquat + 4 * pid, bq, q);
      if constexpr (body_diagnostics) {
        copy(pre_pos + 3 * b, p, 3);
        copy(pre_quat + 4 * b, q, 4);
      }
      for (int j = a; j < a + jnum[b]; ++j) {
        int t = jtype[j], qa = qadr[j];
        real_t axis[3], anchor[3], v[3];
        if constexpr (body_diagnostics)
          joint_rot_identity[j] = identity(q);
        rotquat(jaxis + 3 * j, q, axis);
        rotquat(jpos + 3 * j, q, anchor);
        for (int k = 0; k < 3; ++k)
          anchor[k] += p[k];
        if (!finite(axis, 3) || !finite(anchor, 3)) {
          status[w] = 5;
          return;
        }
        if (t == 2) {
          real_t d = qtmp[qa] - qpos0[qa];
          for (int k = 0; k < 3; ++k)
            p[k] += axis[k] * d;
        } else if (t == 1 || t == 3) {
          real_t local[4];
          if (t == 1)
            copy(local, qtmp + qa, 4);
          else {
            real_t angle = qtmp[qa] - qpos0[qa];
            if (angle == 0) {
              if constexpr (body_diagnostics)
                hinge_zero[j] = 1;
              local[0] = 1;
              local[1] = local[2] = local[3] = 0;
            } else {
              real_t z = angle * REAL_C(.5);
              if (!isfinite(z)) {
                status[w] = 2;
                return;
              }
              real_t sn = sin(z), cs = cos(z);
              if (!isfinite(sn) || !isfinite(cs)) {
                status[w] = 5;
                return;
              }
              if constexpr (body_diagnostics) {
                trig_arg[j] = z;
                trig_sin[j] = sn;
                trig_cos[j] = cs;
              }
              local[0] = cs;
              for (int k = 0; k < 3; ++k)
                local[k + 1] = jaxis[3 * j + k] * sn;
            }
          }
          mulquat(q, local, q);
          if constexpr (body_diagnostics)
            offcenter_rot_identity[j] = identity(q);
          rotquat(jpos + 3 * j, q, v);
          for (int k = 0; k < 3; ++k)
            p[k] += (anchor[k] - p[k]) - v[k];
        } else {
          status[w] = 3;
          return;
        }
        if (!finite(p, 3) || !finite(q, 4)) {
          status[w] = 5;
          return;
        }
        copy(xanchor + 3 * j, anchor, 3);
        copy(xaxis + 3 * j, axis, 3);
        if constexpr (body_diagnostics) {
          copy(joint_pos + 3 * j, p, 3);
          copy(joint_quat + 4 * j, q, 4);
          joint_route[j] = t;
        }
      }
    }
    real_t n = norm(q);
    if (!finite(p, 3) || !isfinite(n)) {
      status[w] = 5;
      return;
    }
    if constexpr (body_diagnostics)
      body_norm_before[b] = n;
    body_norm_branch[b] = (n < REAL_C(1e-15));
    normalize(q, n);
    copy(xpos + 3 * b, p, 3);
    copy(xquat + 4 * b, q, 4);
    if constexpr (body_diagnostics)
      body_mat_identity[b] = identity(q);
    quatmat(q, xmat + 9 * b);
    if (!finite(xmat + 9 * b, 9)) {
      status[w] = 5;
      return;
    }
  }
}
__device__ LIBERO_POSITION_STAGE_INLINE void
body_block(int w, real_t *qpos0, int *parent, int *jnum, int *jadr, int *mocapid, real_t *bpos,
           real_t *bquat, int *jtype, int *qadr, real_t *jaxis, real_t *jpos, int nq, int nb,
           int nj, int nm, int *active, real_t *qpos, real_t *mocap_pos, real_t *mocap_quat,
           real_t *qtmp, real_t *mqtmp, real_t *xpos, real_t *xquat, real_t *xmat, real_t *xanchor,
           real_t *xaxis, real_t *world_xipos, real_t *world_ximat, int *status,
           real_t *norm_before, real_t *mocap_before, real_t *pre_pos, real_t *pre_quat,
           real_t *joint_pos, real_t *joint_quat, real_t *body_norm_before, int *body_route,
           int *joint_route, int *body_mat_identity, int *joint_rot_identity, int *hinge_zero,
           int *offcenter_rot_identity, real_t *trig_arg, real_t *trig_sin, real_t *trig_cos,
           int *norm_branch, int *mocap_branch, int *body_norm_branch, const int *level_offsets,
           const int *level_bodies, int level_count) {
  if (threadIdx.x == 0)
    body(w, qpos0, parent, jnum, jadr, mocapid, bpos, bquat, jtype, qadr, jaxis, jpos, nq, nb, nj,
         nm, active, qpos, mocap_pos, mocap_quat, qtmp, mqtmp, xpos, xquat, xmat, xanchor, xaxis,
         world_xipos, world_ximat, status, norm_before, mocap_before, pre_pos, pre_quat, joint_pos,
         joint_quat, body_norm_before, body_route, joint_route, body_mat_identity,
         joint_rot_identity, hinge_zero, offcenter_rot_identity, trig_arg, trig_sin, trig_cos,
         norm_branch, mocap_branch, body_norm_branch, true);
  __syncthreads();
  if (active[w] != 1 || status[w])
    return;
  const BodyInputs in = BodyInputs{qpos0,
                                   bpos,
                                   bquat,
                                   jaxis,
                                   jpos,
                                   qtmp + w * (nq),
                                   mqtmp + w * (nm * 4),
                                   mocap_pos + w * (nm * 3),
                                   parent,
                                   jnum,
                                   jadr,
                                   mocapid,
                                   jtype,
                                   qadr};
  const BodyOutputs out = BodyOutputs{xpos + w * (nb * 3),
                                      xquat + w * (nb * 4),
                                      xmat + w * (nb * 9),
                                      xanchor + w * (nj * 3),
                                      xaxis + w * (nj * 3),
                                      pre_pos + w * (nb * 3),
                                      pre_quat + w * (nb * 4),
                                      joint_pos + w * (nj * 3),
                                      joint_quat + w * (nj * 4),
                                      body_norm_before + w * (nb),
                                      trig_arg + w * (nj),
                                      trig_sin + w * (nj),
                                      trig_cos + w * (nj),
                                      body_route + w * (nb),
                                      joint_route + w * (nj),
                                      body_mat_identity + w * (nb),
                                      joint_rot_identity + w * (nj),
                                      hinge_zero + w * (nj),
                                      offcenter_rot_identity + w * (nj),
                                      body_norm_branch + w * (nb)};
  for (int level = 0; level < level_count; ++level) {
    for (int at = level_offsets[level] + threadIdx.x; at < level_offsets[level + 1];
         at += blockDim.x) {
      const int b = level_bodies[at];
      const int blocked = out.body_norm_branch[parent[b]];
      if (blocked < 0) {
        out.body_norm_branch[b] = blocked;
        continue;
      }
      const int code = body_one(b, in, out);
      if (code)
        out.body_norm_branch[b] = -code;
    }
    __syncthreads();
  }
  // Levels reorder independent bodies, so publish the earliest ORIGINAL body.
  if (threadIdx.x == 0)
    for (int b = 1; b < nb; ++b)
      if (out.body_norm_branch[b] < 0) {
        status[w] = -out.body_norm_branch[b];
        break;
      }
  __syncthreads();
}
__device__ LIBERO_POSITION_STAGE_INLINE void
commit(int w, int nq, int nb, int nj, int nm, int *active, int *status, real_t *qpos, real_t *mq,
       real_t *qt, real_t *mqt, real_t *xp, real_t *xq, real_t *xm, real_t *anchor, real_t *axis,
       real_t *wip, real_t *wim, real_t *njnorm, real_t *mnorm, real_t *pp, real_t *pq, real_t *jp,
       real_t *jq, real_t *bnorm, int *br, int *jr, int *bm, int *ji, int *hz, int *oi, real_t *ta,
       real_t *ts, real_t *tc, int *nbch, int *mbch, int *bbch) {
  if (status[w] == 0) {
    if (active[w] == 1) {
      copy(qpos + w * nq, qt + w * nq, nq);
      copy(mq + w * nm * 4, mqt + w * nm * 4, nm * 4);
    }
    return;
  }
  zero(qt + w * nq, nq);
  zero(mqt + w * nm * 4, nm * 4);
  zero(xp + w * nb * 3, nb * 3);
  zero(xq + w * nb * 4, nb * 4);
  zero(xm + w * nb * 9, nb * 9);
  zero(anchor + w * nj * 3, nj * 3);
  zero(axis + w * nj * 3, nj * 3);
  zero(wip + w * 3, 3);
  zero(wim + w * 9, 9);
  if constexpr (body_diagnostics) {
    zero(njnorm + w * nj, nj);
    zero(mnorm + w * nm, nm);
    zero(pp + w * nb * 3, nb * 3);
    zero(pq + w * nb * 4, nb * 4);
    zero(jp + w * nj * 3, nj * 3);
    zero(jq + w * nj * 4, nj * 4);
    zero(bnorm + w * nb, nb);
    zero(ta + w * nj, nj);
    zero(ts + w * nj, nj);
    zero(tc + w * nj, nj);
  }
  for (int i = 0; i < nb; ++i) {
    bbch[w * nb + i] = 0;
    if constexpr (body_diagnostics)
      br[w * nb + i] = bm[w * nb + i] = 0;
  }
  for (int i = 0; i < nj; ++i)
    if constexpr (body_diagnostics)
      jr[w * nj + i] = ji[w * nj + i] = hz[w * nj + i] = oi[w * nj + i] = nbch[w * nj + i] = 0;
  for (int i = 0; i < nm; ++i)
    if constexpr (body_diagnostics)
      mbch[w * nm + i] = 0;
}
__device__ LIBERO_POSITION_STAGE_INLINE void after_body(int w, int *active, int *status, int *fa) {
  fa[w] = (active[w] == 1 && status[w] == 0);
}
__device__ LIBERO_POSITION_STAGE_INLINE void frame_status(int w, int *active, int *status) {
  status[w] = (active[w] != 0 && active[w] != 1);
}
__device__ inline void frame(const real_t *bp, const real_t *bq, const real_t *bm, const real_t *lp,
                             const real_t *lq, real_t *p, real_t *m, real_t *tmp) {
  rotmat(bm, lp, p);
  for (int k = 0; k < 3; ++k)
    p[k] += bp[k];
  mulquat(bq, lq, tmp);
  quatmat(tmp, m);
}
__device__ LIBERO_POSITION_STAGE_INLINE void frames_body(int w, int *active, real_t *bp, real_t *bq,
                                                         real_t *bm, real_t *ip, real_t *iq,
                                                         int *same, real_t *op, real_t *om,
                                                         real_t *tmp, int nb) {
  for (int b = threadIdx.x; b < nb; b += blockDim.x) {
    int id = w * nb + b;
    zero(tmp + 4 * id, 4);
    if (active[w] != 1 || b == 0) {
      zero(op + 3 * id, 3);
      zero(om + 9 * id, 9);
      if (active[w] == 1 && b == 0)
        om[9 * id] = om[9 * id + 4] = om[9 * id + 8] = 1;
    } else if (same[b] == 1) {
      copy(op + 3 * id, bp + 3 * id, 3);
      copy(om + 9 * id, bm + 9 * id, 9);
    } else
      frame(bp + 3 * id, bq + 4 * id, bm + 9 * id, ip + 3 * b, iq + 4 * b, op + 3 * id, om + 9 * id,
            tmp + 4 * id);
  }
}
__device__ LIBERO_POSITION_STAGE_INLINE void
frames_objects(int w, int *active, real_t *bp, real_t *bq, real_t *bm, real_t *ip, real_t *im,
               int *body, real_t *lp, real_t *lq, int *same, real_t *op, real_t *om, real_t *tmp,
               int nb, int no) {
  for (int i = threadIdx.x; i < no; i += blockDim.x) {
    int id = w * no + i, b = w * nb + body[i];
    zero(tmp + 4 * id, 4);
    if (active[w] != 1) {
      zero(op + 3 * id, 3);
      zero(om + 9 * id, 9);
    } else if (same[i] == 0)
      frame(bp + 3 * b, bq + 4 * b, bm + 9 * b, lp + 3 * i, lq + 4 * i, op + 3 * id, om + 9 * id,
            tmp + 4 * id);
    else {
      copy(op + 3 * id, (same[i] == 1 ? bp : ip) + 3 * b, 3);
      copy(om + 9 * id, (same[i] == 1 ? bm : im) + 9 * b, 9);
    }
  }
}
__device__ LIBERO_POSITION_STAGE_INLINE void
check_frames(int w, int *active, int *status, real_t *ip, real_t *im, real_t *gp, real_t *gm,
             real_t *sp, real_t *sm, real_t *bt, real_t *gt, real_t *st, int nb, int ng, int ns) {
  // Whole-block entry: world-level admission is uniform across all lanes.
  if (active[w] != 1 || status[w] != 0)
    return;
  bool bad = false;
  const real_t *arrays[9] = {ip + w * nb * 3, im + w * nb * 9, gp + w * ng * 3,
                             gm + w * ng * 9, sp + w * ns * 3, sm + w * ns * 9,
                             bt + w * nb * 4, gt + w * ng * 4, st + w * ns * 4};
  const int sizes[9] = {nb * 3, nb * 9, ng * 3, ng * 9, ns * 3, ns * 9, nb * 4, ng * 4, ns * 4};
  for (int a = 0; a < 9; ++a)
    bad |= libero_vec::any_nonfinite_block_lane(arrays[a], sizes[a]);
  const int invalid = __syncthreads_or(libero_warp::any(bad));
  if (threadIdx.x == 0 && invalid)
    status[w] = 2;
  __syncthreads();
}
__device__ LIBERO_POSITION_STAGE_INLINE void after_frames(int w, int *active, int *status,
                                                          int *ma) {
  ma[w] = (active[w] == 1 && status[w] == 0);
}
__device__ LIBERO_POSITION_STAGE_INLINE void position_finish(int w, int *active, int *bs, int *fs,
                                                             int *ma, int *ms, int *mc, int *mk,
                                                             int *stage, int *code, int *kind) {
  stage[w] = code[w] = kind[w] = 0;
  if (active[w] == 0)
    return;
  if (bs[w] != 0) {
    stage[w] = 1;
    code[w] = bs[w];
    kind[w] = 3;
    return;
  }
  if (fs[w] != 0 || ma[w] != 1) {
    stage[w] = 2;
    code[w] = fs[w] ? fs[w] : 1;
    kind[w] = 3;
    return;
  }
  if (ms[w] != 0) {
    stage[w] = ms[w] + 2;
    code[w] = mc[w];
  }
  kind[w] = mk[w];
}
__device__ LIBERO_POSITION_STAGE_INLINE void dynamics_gate(int w, int *active, int *error,
                                                           int64_t *state, int64_t *source,
                                                           int64_t *epoch, int *valid,
                                                           int64_t *written, int *gate) {
  gate[w] = 0;
  if (active[w] != 1 || error[w] != 0)
    return;
  if (state[w] != source[w]) {
    error[w] = 203;
    return;
  }
  gate[w] = 1;
}
__device__ LIBERO_POSITION_STAGE_INLINE void dynamics_finish(int w, int *gate, int *status,
                                                             int64_t *epoch, int *valid,
                                                             int64_t *written, int *error) {
  valid[w] = 0;
  if (gate[w] != 1 || error[w] != 0)
    return;
  if (status[w] != 0 && status[w] != 25) {
    error[w] = 210;
    return;
  }
  written[w] = epoch[w];
  valid[w] = 1;
}
__device__ LIBERO_POSITION_STAGE_INLINE void transmission(int w, real_t *q, int *active, int *qa,
                                                          int *da, real_t *gear, real_t *length,
                                                          real_t *moment, int *status, int nq,
                                                          int nv, int nu) {
  q += w * nq;
  length += w * nu;
  moment += w * nu * nv;
  status[w] = 0;
  zero(length, nu);
  zero(moment, nu * nv);
  if (active[w] == 0)
    return;
  if (active[w] != 1) {
    status[w] = 1;
    return;
  }
  if (!finite(q, nq)) {
    status[w] = 2;
    return;
  }
  for (int i = 0; i < nu; ++i) {
    if (qa[i] < 0 || qa[i] >= nq || da[i] < 0 || da[i] >= nv) {
      status[w] = 3;
      return;
    }
    real_t v = q[qa[i]] * gear[i];
    if (!isfinite(v)) {
      status[w] = 4;
      return;
    }
    length[i] = v;
    moment[i * nv + da[i]] = gear[i];
  }
}
__device__ LIBERO_POSITION_STAGE_INLINE void dynamics_status(int w, int *kind, int *code, int *tr,
                                                             int *out) {
  out[w] = kind[w] == 2 ? 25 : kind[w] == 3 ? 30 + code[w] : tr[w];
}
} // namespace libero_position

// The dataflow caller owns a whole world block. Its geometry hook receives the
// freshly computed local frame before it is published to the durable arrays.
// Bounds PCA needs all positions; the default hook deliberately does no work.
#ifndef LIBERO_POSITION_GEOM_HOOK
#define LIBERO_POSITION_GEOM_HOOK(world, geom, position, matrix) ((void)0)
#endif
namespace libero_position {
__device__ __forceinline__ void copy_block(real_t *destination, const real_t *source, int n) {
  libero_vec::copy_block(source, destination, n);
}
__device__ __forceinline__ void
frames_objects_dataflow(int w, int *active, real_t *bp, real_t *bq, real_t *bm, real_t *ip,
                        real_t *im, int *body, real_t *lp, real_t *lq, int *same, real_t *op,
                        real_t *om, real_t *tmp, int nb, int no, int geometry, int logical_world,
                        real_t *local_geom_positions = nullptr, bool local_geom_planes = false) {
  for (int i = threadIdx.x; i < no; i += blockDim.x) {
    const int id = w * no + i, b = w * nb + body[i];
    // These small arrays are thread-local and expire after one object.
    real_t position[3] = {}, matrix[9] = {}, quaternion[4] = {};
    if (active[w] == 1) {
      if (same[i] == 0)
        frame(bp + 3 * b, bq + 4 * b, bm + 9 * b, lp + 3 * i, lq + 4 * i, position, matrix,
              quaternion);
      else {
        copy(position, (same[i] == 1 ? bp : ip) + 3 * b, 3);
        copy(matrix, (same[i] == 1 ? bm : im) + 9 * b, 9);
      }
      if (geometry)
        LIBERO_POSITION_GEOM_HOOK(logical_world, i, position, matrix);
    }
    if (geometry && local_geom_positions)
      for (int c = 0; c < 3; ++c)
        local_geom_positions[local_geom_planes ? c * no + i : 3 * i + c] = position[c];
    copy(op + 3 * id, position, 3);
    copy(om + 9 * id, matrix, 9);
    copy(tmp + 4 * id, quaternion, 4);
  }
}

template <int Mode>
__device__ __forceinline__ void
frame_bucket(int w, int *active, real_t *bp, real_t *bq, real_t *bm, real_t *ip, real_t *im,
             int *body, real_t *lp, real_t *lq, real_t *op, real_t *om, real_t *tmp, int nb, int no,
             int geometry, int logical_world, real_t *local_geom_positions, const int *ids,
             int begin, int end, bool local_geom_planes) {
  for (int at = begin + threadIdx.x; at < end; at += blockDim.x) {
    const int i = ids[at], id = w * no + i, b = w * nb + body[i];
    real_t position[3] = {}, matrix[9] = {}, quaternion[4] = {};
    if (active[w] == 1) {
      if constexpr (Mode == 0)
        frame(bp + 3 * b, bq + 4 * b, bm + 9 * b, lp + 3 * i, lq + 4 * i, position, matrix,
              quaternion);
      else {
        copy(position, (Mode == 1 ? bp : ip) + 3 * b, 3);
        copy(matrix, (Mode == 1 ? bm : im) + 9 * b, 9);
      }
      if (geometry)
        LIBERO_POSITION_GEOM_HOOK(logical_world, i, position, matrix);
    }
    if (geometry && local_geom_positions)
      for (int c = 0; c < 3; ++c)
        local_geom_positions[local_geom_planes ? c * no + i : 3 * i + c] = position[c];
    copy(op + 3 * id, position, 3);
    copy(om + 9 * id, matrix, 9);
    copy(tmp + 4 * id, quaternion, 4);
  }
}
__device__ __forceinline__ void frames_objects_dataflow_typed(
    int w, int *active, real_t *bp, real_t *bq, real_t *bm, real_t *ip, real_t *im, int *body,
    real_t *lp, real_t *lq, int *same, real_t *op, real_t *om, real_t *tmp, int nb, int no,
    int geometry, int logical_world, real_t *local_geom_positions, const int *frame_offsets,
    const int *frame_indices, bool local_geom_planes = false) {
  frame_bucket<0>(w, active, bp, bq, bm, ip, im, body, lp, lq, op, om, tmp, nb, no, geometry,
                  logical_world, local_geom_positions, frame_indices, frame_offsets[0],
                  frame_offsets[1], local_geom_planes);
  frame_bucket<1>(w, active, bp, bq, bm, ip, im, body, lp, lq, op, om, tmp, nb, no, geometry,
                  logical_world, local_geom_positions, frame_indices, frame_offsets[1],
                  frame_offsets[2], local_geom_planes);
  frame_bucket<2>(w, active, bp, bq, bm, ip, im, body, lp, lq, op, om, tmp, nb, no, geometry,
                  logical_world, local_geom_positions, frame_indices, frame_offsets[2],
                  frame_offsets[3], local_geom_planes);
}
__device__ LIBERO_POSITION_STAGE_INLINE void
frames_objects_typed(int w, int *active, real_t *bp, real_t *bq, real_t *bm, real_t *ip, real_t *im,
                     int *body, real_t *lp, real_t *lq, int *same, real_t *op, real_t *om,
                     real_t *tmp, int nb, int no, const int *frame_offsets,
                     const int *frame_indices) {
  frames_objects_dataflow_typed(w, active, bp, bq, bm, ip, im, body, lp, lq, same, op, om, tmp, nb,
                                no, 0, w, nullptr, frame_offsets, frame_indices);
}
template <bool Copy>
__device__ __forceinline__ void
body_frame_bucket(int w, int *active, real_t *bp, real_t *bq, real_t *bm, real_t *ip, real_t *iq,
                  real_t *op, real_t *om, real_t *tmp, int nb, const int *ids, int begin, int end) {
  for (int at = begin + threadIdx.x; at < end; at += blockDim.x) {
    const int b = ids[at], id = w * nb + b;
    zero(tmp + 4 * id, 4);
    if (active[w] != 1 || b == 0) {
      zero(op + 3 * id, 3);
      zero(om + 9 * id, 9);
      if (active[w] == 1 && b == 0)
        om[9 * id] = om[9 * id + 4] = om[9 * id + 8] = 1;
    } else if constexpr (Copy) {
      copy(op + 3 * id, bp + 3 * id, 3);
      copy(om + 9 * id, bm + 9 * id, 9);
    } else
      frame(bp + 3 * id, bq + 4 * id, bm + 9 * id, ip + 3 * b, iq + 4 * b, op + 3 * id, om + 9 * id,
            tmp + 4 * id);
  }
}
__device__ LIBERO_POSITION_STAGE_INLINE void
frames_body_typed(int w, int *active, real_t *bp, real_t *bq, real_t *bm, real_t *ip, real_t *iq,
                  int *same, real_t *op, real_t *om, real_t *tmp, int nb, const int *frame_offsets,
                  const int *frame_indices) {
  body_frame_bucket<false>(w, active, bp, bq, bm, ip, iq, op, om, tmp, nb, frame_indices,
                           frame_offsets[0], frame_offsets[1]);
  body_frame_bucket<true>(w, active, bp, bq, bm, ip, iq, op, om, tmp, nb, frame_indices,
                          frame_offsets[1], frame_offsets[2]);
  body_frame_bucket<false>(w, active, bp, bq, bm, ip, iq, op, om, tmp, nb, frame_indices,
                           frame_offsets[2], frame_offsets[3]);
}
} // namespace libero_position
