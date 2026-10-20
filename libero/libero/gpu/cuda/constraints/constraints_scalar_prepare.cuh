#include "precision.cuh"
// S05 cold scalar-parameter prototype. It is deliberately not selected by
// hot impedance until the complete static/options content lease is available.
#pragma once
namespace cf_s {
__device__ void scalar_parameter_prepare(int w,const real_t* ref,const real_t* ip,
  const int* kind,real_t* normalized,int* friction_route,int records,
  real_t timestep,int refsafe){
  if(w)return; // one cold model preparation, shared across resident worlds
  for(int id=threadIdx.x;id<records;id+=blockDim.x){
    real_t r0=ref[id*2],r1=ref[id*2+1];
    if((r0>REAL_C(0.))!=(r1>REAL_C(0.))){r0=REAL_C(.02);r1=REAL_C(1.);}
    if(refsafe&&r0>REAL_C(0.))r0=fmax(r0,REAL_C(2.)*timestep);
    const real_t p0=fmin(REAL_C(.9999),fmax(REAL_C(.0001),ip[id*5]));
    const real_t p1=fmin(REAL_C(.9999),fmax(REAL_C(.0001),ip[id*5+1]));
    const real_t width=fmax(REAL_C(0.),ip[id*5+2]);
    const real_t mid=fmin(REAL_C(.9999),fmax(REAL_C(.0001),ip[id*5+3]));
    const real_t power=fmax(REAL_C(1.),ip[id*5+4]);
    const real_t k=kind[id]!=1?(r0>REAL_C(0.)?REAL_C(1.)/fmax(REAL_C(1e-15),p1*p1*r0*r0*r1*r1):-r0/fmax(REAL_C(1e-15),p1*p1)):REAL_C(0.);
    const real_t b=r1>REAL_C(0.)?REAL_C(2.)/fmax(REAL_C(1e-15),p1*r0):-r1/fmax(REAL_C(1e-15),p1);
    real_t imp=REAL_C(0.);int route=0;
    if(p0==p1||width<=REAL_C(1e-15))imp=REAL_C(.5)*(p0+p1);
    else {imp=p0;route=2;} // source scalar friction pos==margin==0
    real_t* output=normalized+id*12;
    output[0]=r0;output[1]=r1;output[2]=p0;output[3]=p1;
    output[4]=width;output[5]=mid;output[6]=power;
    output[7]=k;output[8]=b;output[9]=imp;output[10]=REAL_C(0.);output[11]=REAL_C(0.);
    friction_route[id]=route;
  }
}
}
