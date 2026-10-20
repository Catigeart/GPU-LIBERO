#include "precision.cuh"
// Direct CUDA port of MuJoCo 2.3.7 scalar collision mathematics.
// Sources and notices: libero/libero/gpu/THIRD_PARTY_NOTICES.md.
// Original Apache-2.0 notices remain below; no Warp-generated arithmetic.
#pragma once
#include "native/native_source.cuh"
#define mjGETINFO const real_t* pos1=d->geom_xpos+3*g1; const real_t* pos2=d->geom_xpos+3*g2; const real_t* mat1=d->geom_xmat+9*g1; const real_t* mat2=d->geom_xmat+9*g2; const real_t* size1=m->geom_size+3*g1; const real_t* size2=m->geom_size+3*g2;
__device__ real_t mju_clip(real_t a,real_t lo,real_t hi){return a<lo?lo:(a>hi?hi:a);}
__device__ void mju_clampVec(real_t* a,const real_t* bound,int n){for(int i=0;i<n;i++)a[i]=mju_clip(a[i],-bound[i],bound[i]);}
// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.













//--------------------------- plane collisions -----------------------------------------------------

// plane : sphere (actual implementation, can be called with modified parameters)
static __device__ int _PlaneSphere(mjContact* con, mjtNum margin,
                        const mjtNum* pos1, const mjtNum* mat1, const mjtNum* size1,
                        const mjtNum* pos2, const mjtNum* mat2, const mjtNum* size2) {
  // set normal
  con[0].frame[0] = mat1[2];
  con[0].frame[1] = mat1[5];
  con[0].frame[2] = mat1[8];

  // compute distance, return if too large
  mjtNum tmp[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};
  mjtNum cdist = mju_dot3(tmp, con[0].frame);
  if (cdist > margin + size2[0]) {
    return 0;
  }

  // depth and position
  con[0].dist = cdist - size2[0];
  mju_scl3(tmp, con[0].frame, -con[0].dist/2 - size2[0]);
  mju_add3(con[0].pos, pos2, tmp);

  mju_zero3(con[0].frame+3);
  return 1;
}



// plane : sphere
__device__ int mjc_PlaneSphere(const mjModel* m, const mjData* d,
                    mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO
  return _PlaneSphere(con, margin, pos1, mat1, size1, pos2, mat2, size2);
}



// plane : capsule
__device__ int mjc_PlaneCapsule(const mjModel* m, const mjData* d,
                     mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO

  // get capsule axis, segment = scaled axis
  mjtNum axis[3] = {mat2[2], mat2[5], mat2[8]};
  mjtNum segment[3] = {size2[1]*axis[0], size2[1]*axis[1], size2[1]*axis[2]};

  // get point 1, do sphere-plane test
  mjtNum pos[3];
  mju_add3(pos, pos2, segment);
  int n1 = _PlaneSphere(con, margin, pos1, mat1, size1, pos, mat2, size2);

  // get point 2, do sphere-plane test
  mju_sub3(pos, pos2, segment);
  int n2 = _PlaneSphere(con+n1, margin, pos1, mat1, size1, pos, mat2, size2);

  // align contact frames with capsule axis
  if (n1) {
    mju_copy3(con->frame+3, axis);
  }
  if (n2) {
    mju_copy3((con+n1)->frame+3, axis);
  }

  return n1+n2;
}



// plane : cylinder
__device__ int mjc_PlaneCylinder(const mjModel* m, const mjData* d,
                      mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO
  mjtNum normal[3] = {mat1[2], mat1[5], mat1[8]};
  mjtNum axis[3] = {mat2[2], mat2[5], mat2[8]};

  // project, make sure axis points towards plane
  mjtNum prjaxis = mju_dot3(normal, axis);
  if (prjaxis > 0) {
    mju_scl3(axis, axis, -1);
    prjaxis = -prjaxis;
  }

  // compute normal distance to cylinder center
  mjtNum vec[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};
  mjtNum dist0 = mju_dot3(vec, normal);

  // remove component of -normal along axis, compute length
  mju_scl3(vec, axis, prjaxis);
  mju_subFrom3(vec, normal);
  mjtNum len_sqr = mju_dot3(vec, vec);

  // general configuration: normalize vector, scale by radius
  if (len_sqr >= mjMINVAL*mjMINVAL) {
    mjtNum scl = size2[0]/mju_sqrt(len_sqr);
    vec[0] *= scl;
    vec[1] *= scl;
    vec[2] *= scl;
  }

  // disk parallel to plane: pick x-axis of cylinder, scale by radius
  else {
    vec[0] = mat2[0]*size2[0];
    vec[1] = mat2[3]*size2[0];
    vec[2] = mat2[6]*size2[0];
  }

  // project vector on normal
  mjtNum prjvec = mju_dot3(vec, normal);

  // scale axis by half-length
  mju_scl3(axis, axis, size2[1]);
  prjaxis *= size2[1];

  // check first point, construct contact
  int cnt = 0;
  if (dist0 + prjaxis + prjvec <= margin) {
    con[cnt].dist = dist0 + prjaxis + prjvec;
    mju_add3(con[cnt].pos, pos2, vec);
    mju_addTo3(con[cnt].pos, axis);
    mju_addToScl3(con[cnt].pos, normal, -con[cnt].dist*REAL_C(0.5));
    mju_copy3(con[cnt].frame, normal);
    mju_zero3(con[cnt].frame+3);
    cnt++;
  } else {
    return 0;  // nearest point is above margin: no contacts
  }

  // check second point, construct contact
  if (dist0 - prjaxis + prjvec <= margin) {
    con[cnt].dist = dist0 - prjaxis + prjvec;
    mju_add3(con[cnt].pos, pos2, vec);
    mju_subFrom3(con[cnt].pos, axis);
    mju_addToScl3(con[cnt].pos, normal, -con[cnt].dist*REAL_C(0.5));
    mju_copy3(con[cnt].frame, normal);
    mju_zero3(con[cnt].frame+3);
    cnt++;
  }

  // try to add triangle points on side closer to plane
  mjtNum prjvec1 = -prjvec*REAL_C(0.5);
  if (dist0 + prjaxis + prjvec1 <= margin) {
    // compute sideways vector: vec1
    mjtNum vec1[3];
    mju_cross(vec1, vec, axis);
    mju_normalize3(vec1);
    mju_scl3(vec1, vec1, size2[0]*mju_sqrt(REAL_C(3.0))/2);

    // add point A
    con[cnt].dist = dist0 + prjaxis + prjvec1;
    mju_add3(con[cnt].pos, pos2, vec1);
    mju_addTo3(con[cnt].pos, axis);
    mju_addToScl3(con[cnt].pos, vec, -REAL_C(0.5));
    mju_addToScl3(con[cnt].pos, normal, -con[cnt].dist*REAL_C(0.5));
    mju_copy3(con[cnt].frame, normal);
    mju_zero3(con[cnt].frame+3);
    cnt++;

    // add point B
    con[cnt].dist = dist0 + prjaxis + prjvec1;
    mju_sub3(con[cnt].pos, pos2, vec1);
    mju_addTo3(con[cnt].pos, axis);
    mju_addToScl3(con[cnt].pos, vec, -REAL_C(0.5));
    mju_addToScl3(con[cnt].pos, normal, -con[cnt].dist*REAL_C(0.5));
    mju_copy3(con[cnt].frame, normal);
    mju_zero3(con[cnt].frame+3);
    cnt++;
  }

  return cnt;
}



// plane : box
__device__ int mjc_PlaneBox(const mjModel* m, const mjData* d,
                 mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO

  // get normal, difference between centers, normal distance
  mjtNum norm[3] = {mat1[2], mat1[5], mat1[8]};
  mjtNum dif[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};
  mjtNum dist = mju_dot3(dif, norm);

  // test all corners, pick bottom 4
  int cnt = 0;
  for (int i=0; i < 8; i++) {
    // get corner in local coordinates
    mjtNum vec[3];
    vec[0] = (i&1 ? size2[0] : -size2[0]);
    vec[1] = (i&2 ? size2[1] : -size2[1]);
    vec[2] = (i&4 ? size2[2] : -size2[2]);

    // get corner in global coordinates relative to box center
    mjtNum corner[3];
    mju_rotVecMat(corner, vec, mat2);

    // compute distance to plane, skip if too far or pointing up
    mjtNum ldist = mju_dot3(norm, corner);
    if (dist + ldist > margin || ldist > 0) {
      continue;
    }

    // construct contact
    con[cnt].dist = dist + ldist;
    mju_copy3(con[cnt].frame, norm);
    mju_zero3(con[cnt].frame+3);
    mju_addTo3(corner, pos2);
    mju_scl3(vec, norm, -con[cnt].dist/2);
    mju_add3(con[cnt].pos, corner, vec);

    // count; max is 4
    if (++cnt >= 4) {
      return 4;
    }
  }

  return cnt;
}



//--------------------------- sphere and capsule collisions ----------------------------------------

// sphere : sphere (actual implementation, can be called with modified parameters)
static __device__ int _SphereSphere(mjContact* con, mjtNum margin,
                         const mjtNum* pos1, const mjtNum* mat1, const mjtNum* size1,
                         const mjtNum* pos2, const mjtNum* mat2, const mjtNum* size2) {
  // check bounding spheres (this is called from other functions)
  mjtNum dif[3] = {pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]};
  mjtNum cdist_sqr = mju_dot3(dif, dif);
  mjtNum min_dist = margin + size1[0] + size2[0];
  if (cdist_sqr > min_dist*min_dist) {
    return 0;
  }

  // depth and normal
  con[0].dist = mju_sqrt(cdist_sqr) - size1[0] - size2[0];
  mju_sub3(con[0].frame, pos2, pos1);
  mjtNum len = mju_normalize3(con[0].frame);

  // if centers are the same, norm = cross-product of z axes
  //  if z axes are parallel, norm = [1;0;0]
  if (len < mjMINVAL) {
    mjtNum axis1[3] = {mat1[2], mat1[5], mat1[8]};
    mjtNum axis2[3] = {mat2[2], mat2[5], mat2[8]};
    mju_cross(con[0].frame, axis1, axis2);
    mju_normalize3(con[0].frame);
  }

  // position
  mju_scl3(con[0].pos, con[0].frame, size1[0] + con[0].dist/2);
  mju_addTo3(con[0].pos, pos1);

  mju_zero3(con[0].frame+3);
  return 1;
}



// sphere : sphere
__device__ int mjc_SphereSphere(const mjModel* m, const mjData* d,
                     mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO
  return _SphereSphere(con, margin, pos1, mat1, size1, pos2, mat2, size2);
}



// sphere : capsule
__device__ int mjc_SphereCapsule(const mjModel* m, const mjData* d,
                      mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO

  // get capsule length and axis
  mjtNum len = size2[1];
  mjtNum axis[3] = {mat2[2], mat2[5], mat2[8]};

  // find projection, clip to segment
  mjtNum vec[3] = {pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]};
  mjtNum x = mju_clip(mju_dot3(axis, vec), -len, len);

  // find nearest point on segment, do sphere-sphere test
  mju_scl3(vec, axis, x);
  mju_addTo3(vec, pos2);
  return _SphereSphere(con, margin, pos1, mat1, size1, vec, mat2, size2);
}



// sphere : cylinder
__device__ int mjc_SphereCylinder(const mjModel* m, const mjData* d,
                       mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO

  // get cylinder sizes and axis
  mjtNum radius = size2[0];
  mjtNum height = size2[1];
  mjtNum axis[3] = {mat2[2], mat2[5], mat2[8]};

  // find sphere projection onto cylinder axis and plane
  mjtNum vec[3] = {pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]};
  mjtNum x = mju_dot3(axis, vec);
  mjtNum a_proj[3], p_proj[3];
  mju_scl3(a_proj, axis, x);
  mju_sub3(p_proj, vec, a_proj);
  mjtNum p_proj_sqr = mju_dot3(p_proj, p_proj);

  // get collision type
  int collide_side = mju_abs(x) < height;
  int collide_cap = p_proj_sqr < radius*radius;
  if (collide_side && collide_cap) {  // deep penetration (sphere origin inside cylinder)
    mjtNum dist_cap = height - mju_abs(x);
    mjtNum dist_radius = radius - mju_sqrt(p_proj_sqr);
    if (dist_cap < dist_radius) {  // disable one collision type
      collide_side = 0;
    } else {
      collide_cap = 0;
    }
  }

  // side collision: use sphere-sphere
  if (collide_side) {
    mju_addTo3(a_proj, pos2);
    return _SphereSphere(con, margin, pos1, mat1, size1, a_proj, mat2, size2);
  }

  // cap collision: use plane-sphere
  if (collide_cap) {
    const mjtNum flipmat[9] = {
      -mat2[0], mat2[1], -mat2[2],
      -mat2[3], mat2[4], -mat2[5],
      -mat2[6], mat2[7], -mat2[8]
    };
    const mjtNum* mat_cap;
    mjtNum pos_cap[3];
    if (x > 0) {  // top cap
      mju_addScl3(pos_cap, pos2, axis, height);
      mat_cap = mat2;
    } else {      // bottom cap
      mju_addScl3(pos_cap, pos2, axis, -height);
      mat_cap = flipmat;
    }
    int ncon = _PlaneSphere(con, margin, pos_cap, mat_cap, size2, pos1, mat1, size1);
    if (ncon) {
      // flip frame normal (because mjGEOM_PLANE < mjGEOM_SPHERE < mjGEOM_CYLINDER)
      mju_scl3(con->frame, con->frame, -1);
    }
    return ncon;
  }

  // otherwise corner collision: use sphere-sphere
  mju_scl3(p_proj, p_proj, size2[0] / mju_sqrt(p_proj_sqr));  // denominator cannot be 0
  mju_scl3(vec, axis, x > 0 ? height : -height);
  mju_addTo3(vec, p_proj);
  mju_addTo3(vec, pos2);

  // sphere-sphere with point sphere at the corner
  mjtNum size_zero[1] = {0};
  return _SphereSphere(con, margin, pos1, mat1, size1, vec, mat2, size_zero);
}



// capsule : capsule
__device__ int mjc_CapsuleCapsule(const mjModel* m, const mjData* d,
                       mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO

  // get capsule axes (scaled) and center difference
  mjtNum axis1[3] = {mat1[2] * size1[1], mat1[5] * size1[1], mat1[8] * size1[1]};
  mjtNum axis2[3] = {mat2[2] * size2[1], mat2[5] * size2[1], mat2[8] * size2[1]};
  mjtNum dif[3] = {pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]};

  // compute matrix coefficients and determinant
  mjtNum ma =  mju_dot3(axis1, axis1);
  mjtNum mb = -mju_dot3(axis1, axis2);
  mjtNum mc =  mju_dot3(axis2, axis2);
  mjtNum u  = -mju_dot3(axis1, dif);
  mjtNum v  =  mju_dot3(axis2, dif);
  mjtNum det = ma*mc - mb*mb;

  // general configuration (non-parallel axes)
  if (fabs(det) >= mjMINVAL) {
    // find projections, clip to segments
    mjtNum x1 = (mc*u - mb*v) / det;
    mjtNum x2 = (ma*v - mb*u) / det;

    if (x1 > 1) {
      x1 = 1;
      x2 = (v - mb) / mc;
    } else if (x1 < -1) {
      x1 = -1;
      x2 = (v + mb) / mc;
    }
    if (x2 > 1) {
      x2 = 1;
      x1 = mju_clip((u - mb) / ma, -1, 1);
    } else if (x2 < -1) {
      x2 = -1;
      x1 = mju_clip((u + mb) / ma, -1, 1);
    }

    // find nearest points, do sphere-sphere test
    mjtNum vec1[3], vec2[3];
    mju_scl3(vec1, axis1, x1);
    mju_addTo3(vec1, pos1);
    mju_scl3(vec2, axis2, x2);
    mju_addTo3(vec2, pos2);

    return _SphereSphere(con, margin, vec1, mat1, size1, vec2, mat2, size2);
  }

  // parallel axes
  else {
    // x1 = 1
    mjtNum vec1[3];
    mju_add3(vec1, pos1, axis1);
    mjtNum x2 = mju_clip((v - mb) / mc, -1, 1);

    mjtNum vec2[3];
    mju_scl3(vec2, axis2, x2);
    mju_addTo3(vec2, pos2);
    int n1 = _SphereSphere(con, margin, vec1, mat1, size1, vec2, mat2, size2);

    // x1 = -1
    mju_sub3(vec1, pos1, axis1);
    x2 = mju_clip((v + mb) / mc, -1, 1);
    mju_scl3(vec2, axis2, x2);
    mju_addTo3(vec2, pos2);
    int n2 = _SphereSphere(con+n1, margin, vec1, mat1, size1, vec2, mat2, size2);

    // return if two contacts already found
    if (n1+n2 >= 2) {
      return n1+n2;
    }

    // x2 = 1
    mju_add3(vec2, pos2, axis2);
    mjtNum x1 = mju_clip((u - mb) / ma, -1, 1);
    mju_scl3(vec1, axis1, x1);
    mju_addTo3(vec1, pos1);
    int n3 = _SphereSphere(con+n1+n2, margin, vec1, mat1, size1, vec2, mat2, size2);

    // return if two contacts already found
    if (n1+n2+n3 >= 2) {
      return n1+n2+n3;
    }

    // x2 = -1
    mju_sub3(vec2, pos2, axis2);
    x1 = mju_clip((u + mb) / ma, -1, 1);
    mju_scl3(vec1, axis1, x1);
    mju_addTo3(vec1, pos1);
    int n4 = _SphereSphere(con+n1+n2+n3, margin, vec1, mat1, size1, vec2, mat2, size2);

    return n1+n2+n3+n4;
  }
}

static __device__ int _SphereBox(mjContact* con, mjtNum mindist,
                      const mjtNum* pos1, const mjtNum* mat1, const mjtNum* size1,
                      const mjtNum* pos2, const mjtNum* mat2, const mjtNum* size2)
{
  int i, k;
  mjtNum tmp[3], center[3], clamped[3], deepest[3], nearest[3];
  mjtNum pos[3];
  mjtNum dist, closest;


  mju_sub3(tmp, pos1, pos2);
  mju_rotVecMatT(center, tmp, mat2);

  mju_copy(clamped, center, 3);
  mju_clampVec(clamped, size2, 3);

  mju_copy(deepest, center, 3);
  mju_sub3(tmp, clamped, center);
  dist = mju_normalize3(tmp);

  if (dist - size1[0] > mindist)
    return 0;


  // sphere center inside box
  if (dist <= mjMINVAL) {
    closest = (size2[0] + size2[1] + size2[2]) * 2;

    for (i = 0; i < 6; i++)
      if (closest > fabs((i % 2 ? 1 : -1)*size2[i / 2] - center[i / 2]))
      {
        closest = fabs((i % 2 ? 1 : -1) * size2[i / 2] - center[i / 2]);
        k = i;
      }

    mju_zero3(nearest);
    nearest[k / 2] = (k % 2 ? -1 : 1);

    mju_copy3(pos, center);
    mju_addToScl3(pos, nearest, (size1[0] - closest) / 2);
    mju_rotVecMat(con[0].frame, nearest, mat2);
  } else {
    mju_addToScl3(deepest, tmp, size1[0]);
    mju_zero3(pos);
    mju_addToScl3(pos, clamped, REAL_C(0.5));
    mju_addToScl3(pos, deepest, REAL_C(0.5));
    mju_rotVecMat(con[0].frame, tmp, mat2);
  }

  mju_rotVecMat(tmp, pos, mat2);
  mju_add3(con[0].pos, tmp, pos2);
  con[0].dist = dist - size1[0];
  mju_zero3(con[0].frame + 3);

  return 1;
}
__device__ int mjc_SphereBox(const mjModel* m, const mjData* d, mjContact* con,
                  int g1, int g2, mjtNum margin)
{
  mjGETINFO;

  return _SphereBox(con, margin, pos1, mat1, size1, pos2, mat2, size2);
}
struct mjcCapsuleBoxCandidate {
  mjtNum distance,segment,box;
  int valid,type,face,corner,edge;
};
__device__ int mjc_CapsuleBox(const mjModel* m, const mjData* d, mjContact* con,
                   int g1, int g2, mjtNum margin,
                   const mjcCapsuleBoxCandidate* candidates=nullptr)
{
  mjGETINFO

  mjtNum tmp1[3], tmp2[3], tmp3[3], halfaxis[3], axis[3], dif[3];
  mjtNum pos[3];          // position of capsule in box-local frame

  mjtNum halflength;      // half of capsule's length
  mjtNum bestdist;        // closest contact point distance
  mjtNum bestdistmax;     // init value for bestdist
  mjtNum bestsegmentpos;  // between -1 and 1 :  which point on the segment is closest to the box
  mjtNum secondpos;       // distance of 2nd contact position on capsule segment from the first
  mjtNum dist;
  mjtNum bestboxpos;      // closest contact point, position on the box's edge
  mjtNum mul, e1, e2, dp, de;
  // mjtNum penetration;

  mjtNum ma, mb, mc, u, v, det, x1, x2, idet;  // linelinedist temps

  int s1, s2;        // hold linelinedist info
  int i, j, c1, c2;  // temporary variables
  int cltype = -4;   // closest type
  int clface;        // closest face
  int clcorner = 0;  // closest corner (0..7 in binary)
  int cledge;        // closest edge axis
  int axisdir;       // direction of capsule axis in relation to the box
  int n;             // number of contacts
  int ax1, ax2, ax;  // axis temporaries


  halflength = size1[1];
  secondpos = -4;  // initialize to no 2nd contact (valid values are between -1 and 1)

  mju_sub3(tmp1, pos1, pos2);       // bring capsule to box-local frame (center's box is at (0,0,0))
  mju_rotVecMatT(pos, tmp1, mat2);  // and axis parralel to world

  tmp1[0] = mat1[2];  // capsule's axis
  tmp1[1] = mat1[5];
  tmp1[2] = mat1[8];

  mju_rotVecMatT(axis, tmp1, mat2);      // do the same for the capsule axis
  mju_scl3(halfaxis, axis, halflength);  // scale to get actual capsule half-axis

  axisdir = 0;
  if (halfaxis[0] > 0)
    axisdir += 1;
  if (halfaxis[1] > 0)
    axisdir += 2;
  if (halfaxis[2] > 0)
    axisdir += 4;

  // under this notion    "axisdir" and "7-axisdir" point in opposite directions,
  // essentially the same for a capsule

  bestdistmax = margin + 2 * (size1[0] + halflength + size2[0] + size2[1] +
                              size2[2]);  // initialize bestdist
  bestdist = bestdistmax;
  bestsegmentpos = 0;

  mju_zero3(tmp2);

  // test to see if maybe the a face of the box is closest to the capsule
  for (i = -1; i <= 1; i += 2) {
    if(candidates){
      const mjcCapsuleBoxCandidate& candidate=candidates[(i+1)/2];
      if(candidate.valid && candidate.distance<bestdist){
        bestdist=candidate.distance;bestsegmentpos=candidate.segment;
        cltype=candidate.type;clface=candidate.face;
      }
      continue;
    }
    mju_copy3(tmp1, pos);
    mju_addToScl3(tmp1, halfaxis, i);
    mju_copy3(tmp2, tmp1);

    for (c1 = 0, j = 0, c2 = -1; j < 3; j++) {
      if (tmp1[j] < -size2[j]) {
        c1++;
        c2 = j;
        tmp1[j] = -size2[j];
      } else if (tmp1[j] > size2[j]) {
        c1++;
        c2 = j;
        tmp1[j] = size2[j];
      }
    }

    if (c1 > 1)
      continue;

    mju_subFrom3(tmp1, tmp2);
    dist = mju_dot3(tmp1, tmp1);

    if (dist < bestdist) {
      bestdist = dist;
      bestsegmentpos = i;
      cltype = -2 + i;
      clface = c2;
    }
  }

  mju_zero3(tmp2);

  for (j = 0; j < 3; j++) {
    for (i = 0; i < 8; i++) {
      if ((i & (1 << j)) == 0) {
        if(candidates){
          const int ordinal=(i&((1<<j)-1))|((i>>(j+1))<<j);
          const mjcCapsuleBoxCandidate& candidate=candidates[2+4*j+ordinal];
          if(candidate.valid && candidate.distance<bestdist-mjMINVAL){
            bestdist=candidate.distance;bestsegmentpos=candidate.segment;
            bestboxpos=candidate.box;clcorner=candidate.corner;
            cledge=candidate.edge;cltype=candidate.type;
          }
          continue;
        }
        // trick to get a corner
        tmp3[0] = ((i & 1) ? 1 : -1) * size2[0];
        tmp3[1] = ((i & 2) ? 1 : -1) * size2[1];
        tmp3[2] = ((i & 4) ? 1 : -1) * size2[2];
        tmp3[j] = 0;

        // tmp3 is the starting point on the box
        // tmp2 is the direction along the "j"-th axis
        // pos is the capsule's center
        // halfaxis is the capsule direction

        // find closest point between capsule and the edge

        mju_sub3(dif, tmp3, pos);

        ma = size2[j] * size2[j];
        mb = -size2[j] * halfaxis[j];
        mc = size1[1] * size1[1];

        u = -size2[j] * dif[j];
        v = mju_dot3(halfaxis, dif);

        det = ma * mc - mb * mb;
        if (fabs(det) < mjMINVAL)
          continue;
        idet = 1 / det;


        // sX : X=1 means middle of segment. X=0 or 2 one or the other end

        x1 = (mc * u - mb * v) * idet;
        x2 = (ma * v - mb * u) * idet;

        s1 = s2 = 1;

        if (x1 > 1) {
          x1 = 1;
          s1 = 2;
          x2 = (v - mb) * (1 / mc);
        } else if (x1 < -1) {
          x1 = -1;
          s1 = 0;
          x2 = (v + mb) * (1 / mc);
        }

        if (x2 > 1) {
          x2 = 1;
          s2 = 2;
          x1 = (u - mb) * (1 / ma);
          if (x1 > 1)
            x1 = 1, s1 = 2;
          else if (x1 < -1)
            x1 = -1, s1 = 0;
        } else if (x2 < -1) {
          x2 = -1;
          s2 = 0;
          x1 = (u + mb) * (1 / ma);
          if (x1 > 1)
            x1 = 1, s1 = 2;
          else if (x1 < -1)
            x1 = -1, s1 = 0;
        }

        mju_sub3(dif, tmp3, pos);

        mju_addToScl3(dif, halfaxis, -x2);
        dif[j] += size2[j] * x1;

        tmp1[2] = mju_dot3(dif, dif);

        c1 = s1 * 3 + s2;


        // the -MINVAL might not be necessary. Fixes numerical problem when axis is numerically
        // parallel to the box
        if (tmp1[2] < bestdist - mjMINVAL) {
          bestdist = tmp1[2];
          bestsegmentpos = x2;
          bestboxpos = x1;

          // c1<6 means that closest point on the box is at the lower end
          // or in the middle of the edge
          c2 = c1 / 6;

          clcorner = i + (1 << j) * c2;  // which corner is the closest
          cledge = j;   // which axis
          cltype = c1;  // save clamped info
        }
      }
    }
  }


  // penetration = -bestdist;


  for (j = 0; j < 3; j++) {
    if (j == 2) {
      typedef union {
        struct {
          mjtNum x, y;
        };
        mjtNum c[2];
      } d2;
      d2 p, s, dd /*, c, tmp1*/;
      mjtNum uu, vv, w, ee1, best /* ,e2 */, l /* , e3, e4 */;

      bestdist = bestdistmax;

      p.x = pos[0];
      p.y = pos[1];
      dd.x = halfaxis[0];
      dd.y = halfaxis[1];
      s.x = size2[0];
      s.y = size2[1];

      l = sqrt(dd.x * dd.x + dd.y * dd.y);

      uu = dd.x * s.y;
      vv = dd.y * s.x;
      w = dd.x * p.y - dd.y * p.x;


      best = -1;

      ee1 = +uu - vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 0;
        }
      }
      ee1 = -uu - vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 1;
        }
      }
      ee1 = +uu + vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 2;
        }
      }
      ee1 = -uu + vv;
      if ((ee1 < 0) == (w < 0)) {
        if (best < mju_abs(ee1)) {
          best = mju_abs(ee1);
          c1 = 3;
        }
      }

      // c.x = s.x * ((c1 / 2) ? -1 : 1);
      // c.y = s.y * ((c1 % 2) ? -1 : 1);

      ee1 = fabs(w) / l;
      // e2 = best / l;

      // printf("%g %g      %g %g     %g %g\n",c.x,c.y,d.x,d.y,e1,e2);

      // tmp1.x = c.x - p.x;
      // tmp1.y = c.y - p.y;
      ee1 = dd.x * dd.x + dd.y * dd.y;
      // e2 = tmp1.x * d.x + tmp1.y * d.y;
      // e3 = e2 / e1;


      // printf("%g %g      %g %g     %g %g %g \n",c.x,c.y,d.x,d.y,e1,e2,e3);

      ee1 = p.x + (+s.y - p.y) / dd.y * dd.x;
      // e2 = p.x + (-s.y - p.y) / d.y * d.x;
      // e3 = p.y + (+s.x - p.x) / d.x * d.y;
      // e4 = p.y + (-s.x - p.x) / d.x * d.y;


      // printf("%g %g     %g %g\n",e1,e2,e3,e4);
    }
  }

  // goto skip;   // allow only the closest contact

  // cltype: -3 -1 : face is closest to the capsule
  // cltype: 0..8 : edge is closest to the capsule
  // cltype/3==0 means the lower corner is closest to the capsule (note that edges include corners)
  // cltype/3==2 means the upper corner is closest to the capsule (note that edges include corners)
  // cltype/3==1 means the middle of the edge is closest to the capsule
  // cltype%3==0 means the lower corner is closest to the box (note that edges include corners)
  // cltype%3==2 means the upper corner is closest to the box (note that edges include corners)
  // cltype%3==1 means the middle of the capsule is closest to the box


  // invalid type
  if (cltype == -4)
    return 0;

  if (cltype >= 0 && cltype / 3 != 1) {  // closest to a corner of the box
    c1 = axisdir ^ clcorner;

    // hack to find the relative orientation of capsule and corner
    // there are 2 cases:
    //    1: pointing to or away from the corner
    //    2: oriented along a face or an edge


    if (c1 == 0 || c1 == 7)
      goto skip;  // case 1: no chance of additional contact

    if (c1 == 1 || c1 == 2 || c1 == 4) {
      mul = 1;
      de = 1 - bestsegmentpos;
      dp = 1 + bestsegmentpos;
    }

    if (c1 == 3 || c1 == 5 || c1 == 6) {
      mul = -1;
      c1 = 7 - c1;
      dp = 1 - bestsegmentpos;
      de = 1 + bestsegmentpos;
    }

    // "de" and "dp" distance from first closest point on the capsule to both ends of it
    // mul is a direction along the capsule's axis

    if (c1 == 1)
      ax = 0, ax1 = 1, ax2 = 2;
    if (c1 == 2)
      ax = 1, ax1 = 2, ax2 = 0;
    if (c1 == 4)
      ax = 2, ax1 = 0, ax2 = 1;



    if (axis[ax]*axis[ax] > REAL_C(0.5)) {  // second point along the edge of the box
      secondpos = de;  // initial position from the
      e1 = 2 * size2[ax] / fabs(halfaxis[ax]);

      if (e1 < secondpos) {
        secondpos = e1;  // we overshoot, move back to the  other corner of the edge
      }
      secondpos *= mul;
    } else {  // second point along a face of the box
      secondpos = dp;

      // check for overshoot again

      e1 = 2 * size2[ax1] / fabs(halfaxis[ax1]);
      if (e1 < secondpos)
        secondpos = e1;

      e1 = 2 * size2[ax2] / fabs(halfaxis[ax2]);
      if (e1 < secondpos)
        secondpos = e1;

      secondpos *= -mul;
    }
  } else if (cltype >= 0 && cltype / 3 == 1) {  // we are on box's edge
    // hacks to find the relative orientation of capsule and edge
    // there are 2 cases:
    //    c1= 2^n: edge and capsule are oriented in a T configuaration (no more contacts
    //    c1!=2^n: oriented in a cross X configuration

    c1 = axisdir ^ clcorner;  // same trick

    c1 &= 7 - (1 << cledge);  // even more hacks

    // printf("%d %d %d %d    %lf %lf %lf\n",
    //        axisdir,clcorner,c1,cledge,halfaxis[0],halfaxis[1],halfaxis[2]);

    if (c1 != 1 && c1 != 2 && c1 != 4)
      goto skip;


    if (cledge == 0)
      ax1 = 1, ax2 = 2;
    if (cledge == 1)
      ax1 = 2, ax2 = 0;
    if (cledge == 2)
      ax1 = 0, ax2 = 1;
    ax = cledge;


    // Then it finds with which face the capsule has a lower angle and switches the axis names

    if (fabs(axis[ax1]) > fabs(axis[ax2]))
      ax1 = ax2;
    ax2 = 3 - ax - ax1;

    // keep track of the axis orientation (mul will tell us which direction along the capsule to
    // find the second point) you can notice all other references to the axis "halfaxis" are with
    // absolute value

    if (c1 & (1 << ax2)) {
      mul = 1;
      secondpos = 1 - bestsegmentpos;
    } else {
      mul = -1;
      secondpos = 1 + bestsegmentpos;
    }


    // now we have to find out whether we point towards the opposite side or towards one of the
    // sides and also find the farthest point along the capsule that is above the box

    e1 = 2 * size2[ax2] / fabs(halfaxis[ax2]);
    if (e1 < secondpos)
      secondpos = e1;

    if (((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0))  // that is insane
      e2 = 1 - bestboxpos;
    else
      e2 = 1 + bestboxpos;

    e1 = size2[ax] * e2 / fabs(halfaxis[ax]);

    if (e1 < secondpos)
      secondpos = e1;

    secondpos *= mul;
  } else if (cltype < 0) {
    // similarly we handle the case when one capsule's end is closest to a face of the box
    // and find where is the other end pointing to and clamping to the farthest point
    // of the capsule that's above the box

    if (clface == -1)
      goto skip;  // here the closest point is inside the box, no need for a second point
    if (cltype == -3)
      mul = 1;
    else
      mul = -1;

    secondpos = 2;

    mju_copy3(tmp1, pos);
    mju_addToScl3(tmp1, halfaxis, -mul);

    for (i = 0; i < 3; i++) {
      if (i != clface) {
        e1 = (size2[i] - tmp1[i]) / halfaxis[i] * mul;
        if (e1 > 0)
          if (e1 < secondpos)
            secondpos = e1;

        e1 = (-size2[i] - tmp1[i]) / halfaxis[i] * mul;
        if (e1 > 0)
          if (e1 < secondpos)
            secondpos = e1;
      }
    }
    secondpos *= mul;
  }


skip:

  // create sphere in original orientation at first contact point
  mju_copy3(tmp1, pos);
  mju_addToScl3(tmp1, halfaxis, bestsegmentpos);
  mju_rotVecMat(tmp2, tmp1, mat2);
  mju_addTo3(tmp2, pos2);

  // collide with
  n = _SphereBox(con, margin, tmp2, mat1, size1, pos2, mat2, size2);


  if (secondpos > -3) {  // secondpos was modified
    mju_copy3(tmp1, pos);
    mju_addToScl3(tmp1, halfaxis, secondpos + bestsegmentpos);  // note the summation
    mju_rotVecMat(tmp2, tmp1, mat2);
    mju_addTo3(tmp2, pos2);
    n += _SphereBox(con + n, margin, tmp2, mat1, size1, pos2, mat2, size2);
  }

  return n;
}
const int maxplanemesh = 3; const real_t tolplanemesh = REAL_C(0.3);
static __device__ int addplanemesh(mjContact* con, const float vertex[3],
                        const mjtNum pos1[3], const mjtNum normal1[3],
                        const mjtNum pos2[3], const mjtNum mat2[9],
                        const mjtNum first[3], mjtNum rbound) {
  // compute point in global coordinates
  mjtNum pnt[3], v[3] = {vertex[0], vertex[1], vertex[2]};
  mju_rotVecMat(pnt, v, mat2);
  mju_addTo3(pnt, pos2);

  // skip if too close to first contact
  if (mju_dist3(pnt, first) < tolplanemesh*rbound) {
    return 0;
  }

  // pnt-pos difference vector
  mjtNum dif[3];
  mju_sub3(dif, pnt, pos1);

  // set distance
  con->dist = mju_dot3(normal1, dif);

  // set position
  mju_copy3(con->pos, pnt);
  mju_addToScl3(con->pos, normal1, -REAL_C(0.5)*con->dist);

  // set frame
  mju_copy3(con->frame, normal1);
  mju_zero3(con->frame+3);

  return 1;
}
__device__ int mjc_PlaneConvex(const mjModel* m, const mjData* d,
                    mjContact* con, int g1, int g2, mjtNum margin) {
  mjGETINFO
  mjtNum dist, dif[3], normal[3] = {mat1[2], mat1[5], mat1[8]};
  ccd_vec3_t dir, vec;
  mjtCCD obj = {m, d, g2, -1, 0, {1, 0, 0, 0}};

  // get support point in -normal direction
  ccdVec3Set(&dir, -mat1[2], -mat1[5], -mat1[8]);
  // Preserve the existing missing-mesh zero-support behavior without indexing
  // mesh arrays through dataid=-1. Ellipsoids still use their normal support.
  if (m->geom_type[g2] == mjGEOM_MESH && m->geom_dataid[g2] < 0) {
    mju_copy3(vec.v, pos2);
  } else {
    mjccd_support(&obj, &dir, &vec);
    if (*m->adapter_error) return -6;
  }

  // compute normal distance, return if too far
  mju_sub3(dif, vec.v, pos1);
  dist = mju_dot3(normal, dif);
  if (dist > margin) {
    return 0;
  }

  // fill in contact data
  con->dist = dist;
  mju_copy3(con->pos, vec.v);
  mju_addToScl3(con->pos, normal, -REAL_C(0.5)*dist);
  mju_copy3(con->frame, normal);
  mju_zero3(con->frame+3);

  //--------------- add all/connected vertices below margin
  float* vertdata;
  int graphadr, numvert, locid;
  int *vert_edgeadr, *vert_globalid, *edge_localid;
  mjtNum vdot;
  int count = 1, g = g2;

  // g is an ellipsoid: no need for further mesh-specific processing
  if (m->geom_dataid[g] == -1) {
    return count;
  }

  // init
  vertdata = m->mesh_vert + 3*m->mesh_vertadr[m->geom_dataid[g]];

  // express dir in geom local frame
  mjtNum locdir[3];
  mju_rotVecMatT(locdir, dir.v, d->geom_xmat+9*g);

  // inclusion threshold along locdir, relative to geom2 center
  mju_sub3(dif, pos2, pos1);
  mjtNum threshold = mju_dot3(normal, dif) - margin;

  // no graph data: exhaustive search
  if (m->mesh_graphadr[m->geom_dataid[g]] < 0) {
    // search all vertices, find best
    for (int i=0; i < m->mesh_vertnum[m->geom_dataid[g]] && count < maxplanemesh; i++) {
      // vdot = dot(vertex, dir)
      vdot = locdir[0] * (mjtNum)vertdata[3*i] +
             locdir[1] * (mjtNum)vertdata[3*i+1] +
             locdir[2] * (mjtNum)vertdata[3*i+2];

      // detect contact, skip best
      if (vdot > threshold && i != obj.meshindex) {
        count += addplanemesh(con+count, vertdata+3*i,
                              pos1, normal, pos2, mat2,
                              con->pos, m->geom_rbound[g2]);
      }
    }
  }

  // use graph data
  else if (obj.meshindex >= 0) {
    // get info
    graphadr = m->mesh_graphadr[m->geom_dataid[g]];
    numvert = m->mesh_graph[graphadr];
    vert_edgeadr = m->mesh_graph + graphadr + 2;
    vert_globalid = m->mesh_graph + graphadr + 2 + numvert;
    edge_localid = m->mesh_graph + graphadr + 2 + 2*numvert;

    // look for contacts in ibest neighborhood
    int i = vert_edgeadr[obj.meshindex];
    while ((locid=edge_localid[i]) >= 0 && count < maxplanemesh) {
      // vdot = dot(vertex, dir)
      vdot = locdir[0] * (mjtNum)vertdata[3*vert_globalid[locid]] +
             locdir[1] * (mjtNum)vertdata[3*vert_globalid[locid]+1] +
             locdir[2] * (mjtNum)vertdata[3*vert_globalid[locid]+2];

      // detect contact
      if (vdot > threshold) {
        count += addplanemesh(con+count, vertdata+3*vert_globalid[locid],
                              pos1, normal, pos2, mat2,
                              con->pos, m->geom_rbound[g2]);
      }

      // advance to next edge
      i++;
    }
  }

  return count;
}
// Separate full-warp/query entry. The scalar entry above remains safe for the
// original one-thread/query scheduler. Only lane 0 mutates contact records.
__device__ int mjc_PlaneConvex_warp(const mjModel* m,const mjData* d,
                                  mjContact* con,int g1,int g2,mjtNum margin) {
  const int lane=threadIdx.x&31;
  const unsigned mask=0xffffffffu;
  mjGETINFO
  mjtNum normal[3]={mat1[2],mat1[5],mat1[8]};
  ccd_vec3_t dir,vec;
  ccdVec3Set(&dir,-normal[0],-normal[1],-normal[2]);
  mjtCCD obj={m,d,g2,-1,0,{1,0,0,0}};
  int count=1,result=1,meshindex=-1;
  if(lane==0){
    if(m->geom_type[g2]==mjGEOM_MESH && m->geom_dataid[g2]<0)mju_copy3(vec.v,pos2);
    else {mjccd_support(&obj,&dir,&vec);if(*m->adapter_error)result=-6;}
    meshindex=obj.meshindex;
    if(result==1){
      mjtNum dif[3];mju_sub3(dif,vec.v,pos1);
      const mjtNum dist=mju_dot3(normal,dif);
      if(dist>margin)result=0;
      else {
        con->dist=dist;mju_copy3(con->pos,vec.v);
        mju_addToScl3(con->pos,normal,-REAL_C(0.5)*dist);
        mju_copy3(con->frame,normal);mju_zero3(con->frame+3);
      }
    }
  }
  result=__shfl_sync(mask,result,0);
  meshindex=__shfl_sync(mask,meshindex,0);
  if(result!=1)return result;
  const int mesh=m->geom_dataid[g2];
  if(mesh==-1)return 1;
  float* vertices=m->mesh_vert+3*m->mesh_vertadr[mesh];
  mjtNum locdir[3],dif[3];mju_rotVecMatT(locdir,dir.v,mat2);
  mju_sub3(dif,pos2,pos1);
  const mjtNum threshold=mju_dot3(normal,dif)-margin;
  if(m->mesh_graphadr[mesh]<0){
    const int n=m->mesh_vertnum[mesh];
    for(int base=0;base<n && count<maxplanemesh;base+=32){
      const int i=base+lane;
      float xyz[3];native_mesh_vertex_tile(vertices,base,n,32,mask,xyz);
      mjtNum vdot=0;
      if(i<n)vdot=locdir[0]*(mjtNum)xyz[0]+
                     locdir[1]*(mjtNum)xyz[1]+
                     locdir[2]*(mjtNum)xyz[2];
      unsigned candidates=__ballot_sync(mask,i<n && i!=meshindex && vdot>threshold);
      if(lane==0){
        // addplanemesh can reject a candidate as too close. Replay every
        // selected original ordinal until the successful-contact cap is met.
        while(candidates && count<maxplanemesh){
          const int chosen=__ffs(candidates)-1;
          count+=addplanemesh(con+count,vertices+3*(base+chosen),pos1,normal,
                             pos2,mat2,con->pos,m->geom_rbound[g2]);
          candidates&=candidates-1;
        }
      }
      count=__shfl_sync(mask,count,0);
    }
  }else if(meshindex>=0){
    // Graph traversal retains its exact sentinel-read and successful-contact
    // early-stop boundary. Do not prefetch neighbors beyond that boundary.
    if(lane==0){
      const int graph=m->mesh_graphadr[mesh],nv=m->mesh_graph[graph];
      int* edges=m->mesh_graph+graph+2;
      int* ids=edges+nv;int* local=ids+nv;
      int i=edges[meshindex],id;
      while((id=local[i])>=0 && count<maxplanemesh){
        const int vertex=ids[id];
        const mjtNum vdot=locdir[0]*(mjtNum)vertices[3*vertex]+
                          locdir[1]*(mjtNum)vertices[3*vertex+1]+
                          locdir[2]*(mjtNum)vertices[3*vertex+2];
        if(vdot>threshold)count+=addplanemesh(con+count,vertices+3*vertex,
          pos1,normal,pos2,mat2,con->pos,m->geom_rbound[g2]);
        ++i;
      }
    }
    count=__shfl_sync(mask,count,0);
  }
  return count;
}

// Full-warp/query entry: evaluate independent endpoint/edge geometry first,
// then replay the original ordered comparisons and contact production in lane
// 0. Feature-dependent later stages are not speculatively evaluated.
__device__ int mjc_CapsuleBox_warp(const mjModel* m,const mjData* d,mjContact* con,
                                 int g1,int g2,mjtNum margin) {
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  // NativePlan supports at most 256 threads. Each warp has private candidate
  // storage and synchronizes only its own query, never the entire block.
  __shared__ mjcCapsuleBoxCandidate work[8][14];
  mjGETINFO
  if(lane<14){
    mjcCapsuleBoxCandidate candidate={};
    mjtNum tmp[3],pos[3],axis[3],halfaxis[3];
    mju_sub3(tmp,pos1,pos2);mju_rotVecMatT(pos,tmp,mat2);
    tmp[0]=mat1[2];tmp[1]=mat1[5];tmp[2]=mat1[8];
    mju_rotVecMatT(axis,tmp,mat2);mju_scl3(halfaxis,axis,size1[1]);
    if(lane<2){
      const int side=lane?1:-1;
      mjtNum end[3];mju_copy3(tmp,pos);mju_addToScl3(tmp,halfaxis,side);
      mju_copy3(end,tmp);int outside=0,face=-1;
      for(int j=0;j<3;++j){
        if(tmp[j]<-size2[j]){++outside;face=j;tmp[j]=-size2[j];}
        else if(tmp[j]>size2[j]){++outside;face=j;tmp[j]=size2[j];}
      }
      if(outside<=1){
        mju_subFrom3(tmp,end);candidate.distance=mju_dot3(tmp,tmp);
        candidate.segment=side;candidate.type=-2+side;
        candidate.face=face;candidate.valid=1;
      }
    }else{
      const int axis_id=(lane-2)/4,ordinal=(lane-2)%4;
      const int corner=(ordinal&((1<<axis_id)-1))|
                       ((ordinal>>axis_id)<<(axis_id+1));
      mjtNum origin[3]={((corner&1)?1:-1)*size2[0],
                        ((corner&2)?1:-1)*size2[1],
                        ((corner&4)?1:-1)*size2[2]};
      origin[axis_id]=0;
      mjtNum dif[3];mju_sub3(dif,origin,pos);
      const mjtNum ma=size2[axis_id]*size2[axis_id];
      const mjtNum mb=-size2[axis_id]*halfaxis[axis_id];
      const mjtNum mc=size1[1]*size1[1];
      const mjtNum u=-size2[axis_id]*dif[axis_id],v=mju_dot3(halfaxis,dif);
      const mjtNum det=ma*mc-mb*mb;
      if(!(fabs(det)<mjMINVAL)){
        const mjtNum inv=1/det;
        mjtNum x1=(mc*u-mb*v)*inv,x2=(ma*v-mb*u)*inv;
        int s1=1,s2=1;
        if(x1>1){x1=1;s1=2;x2=(v-mb)*(1/mc);}
        else if(x1<-1){x1=-1;s1=0;x2=(v+mb)*(1/mc);}
        if(x2>1){
          x2=1;s2=2;x1=(u-mb)*(1/ma);
          if(x1>1){x1=1;s1=2;}else if(x1<-1){x1=-1;s1=0;}
        }else if(x2<-1){
          x2=-1;s2=0;x1=(u+mb)*(1/ma);
          if(x1>1){x1=1;s1=2;}else if(x1<-1){x1=-1;s1=0;}
        }
        mju_sub3(dif,origin,pos);mju_addToScl3(dif,halfaxis,-x2);
        dif[axis_id]+=size2[axis_id]*x1;
        candidate.distance=mju_dot3(dif,dif);candidate.segment=x2;
        candidate.box=x1;candidate.type=s1*3+s2;
        candidate.corner=corner+(1<<axis_id)*(candidate.type/6);
        candidate.edge=axis_id;candidate.valid=1;
      }
    }
    work[warp][lane]=candidate;
  }
  __syncwarp(0xffffffffu);
  int result=0;
  if(lane==0)result=mjc_CapsuleBox(m,d,con,g1,g2,margin,work[warp]);
  result=__shfl_sync(0xffffffffu,result,0);
  // All lanes have passed the leader consumption before this warp can reuse
  // its scratch records for another query.
  __syncwarp(0xffffffffu);
  return result;
}
