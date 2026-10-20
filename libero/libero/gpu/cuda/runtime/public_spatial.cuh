#include "precision.cuh"
#pragma once
#ifndef SPATIAL_NV
#error SPATIAL_NV must be the native model velocity dimension
#endif
namespace public_spatial_cuda {
// Derived cdof/positions keep the age of the last explicit GPU forward, while
// qvel below is the current live state, matching original get_*_xvel methods.
__device__ inline void jacobian(int w, int*active, int*kind, int*index,
    int*status, int*rootid, int*ancestor, int*geom_bodyid, int*site_bodyid,
    real_t*cdof, real_t*com, real_t*body_pos, real_t*geom_pos, real_t*site_pos,
    real_t*jp, real_t*jr, int nb, int ng, int ns) {
  if (active[w] != 1) return;
  const int k=kind[w], id=index[w];
  const int count=k==0?nb:k==1?ng:k==2?ns:0;
  if (id<0 || id>=count) { status[w]=1; return; }
  const int body=k==0?id:k==1?geom_bodyid[id]:site_bodyid[id];
  if (body<0 || body>=nb || rootid[body]<0 || rootid[body]>=nb) {
    status[w]=2; return;
  }
  const real_t*point=(k==0?body_pos:k==1?geom_pos:site_pos)+3*(w*count+id);
  const real_t*center=com+3*(w*nb+rootid[body]);
  const real_t dx=point[0]-center[0],dy=point[1]-center[1],dz=point[2]-center[2];
  for (int v=0;v<SPATIAL_NV;++v) {
    const real_t*d=cdof+6*(w*SPATIAL_NV+v);
    const bool present=ancestor[body*SPATIAL_NV+v]==1;
    const real_t ax=present?d[0]:REAL_C(0.),ay=present?d[1]:REAL_C(0.),az=present?d[2]:REAL_C(0.);
    const int base=w*3*SPATIAL_NV+v;
    jr[base]=ax; jr[base+SPATIAL_NV]=ay; jr[base+2*SPATIAL_NV]=az;
    jp[base]=present?d[3]+ay*dz-az*dy:REAL_C(0.);
    jp[base+SPATIAL_NV]=present?d[4]+az*dx-ax*dz:REAL_C(0.);
    jp[base+2*SPATIAL_NV]=present?d[5]+ax*dy-ay*dx:REAL_C(0.);
  }
  status[w]=0;
}
__device__ void velocity(int w,int*active,int*status,real_t*jp,real_t*jr,
                        real_t*qvel,real_t*vp,real_t*vr);
}
