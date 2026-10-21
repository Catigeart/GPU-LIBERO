// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned collision_core.py: material mixing and pooled contact publication.
#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace flash::mjwarp {
struct ContactParameter {const float* data; int rows;};
struct ContactInputs {
  int worlds,geoms,explicit_pairs,capacity,address_columns;
  ContactParameter size,solmix,solref,solimp,friction,margin,gap,adhesion;
  ContactParameter pair_solref,pair_solreffriction,pair_solimp,pair_friction,pair_margin,pair_gap,pair_adhesion;
  const int *geom_type,*condim,*priority,*pair_dim,*geom_dataid;
  int dataid_rows;
  const int *mesh_vertadr,*mesh_vertnum,*mesh_graphadr,*mesh_graph,*mesh_polynum,*mesh_polyadr;
  const int *mesh_polyvertadr,*mesh_polyvertnum,*mesh_polyvert,*mesh_polymapadr,*mesh_polymapnum,*mesh_polymap;
  const float *mesh_vert,*mesh_polynormal,*xpos,*xmat;
  const int *candidate_count,*candidate_geom,*candidate_ids,*candidate_world;
  int *contact_count,*contact_dim,*contact_geom,*contact_world,*contact_type,*contact_collisionid,*contact_address;
  float *contact_dist,*contact_pos,*contact_frame,*contact_margin,*contact_friction,*contact_solref,*contact_solreffriction,*contact_solimp,*contact_adhesion;
};
struct CollisionVec3 {
  float v[3];
  __device__ float& operator[](int i) {return v[i];}
  __device__ float operator[](int i) const {return v[i];}
};
using CV=CollisionVec3;
__device__ inline CV cv(float x,float y,float z) {return {{x,y,z}};}
__device__ inline CV operator+(CV a,CV b) {return cv(a[0]+b[0],a[1]+b[1],a[2]+b[2]);}
__device__ inline CV operator-(CV a,CV b) {return cv(a[0]-b[0],a[1]-b[1],a[2]-b[2]);}
__device__ inline CV operator-(CV a) {return cv(-a[0],-a[1],-a[2]);}
__device__ inline CV operator*(CV a,float s) {return cv(a[0]*s,a[1]*s,a[2]*s);}
__device__ inline CV operator*(float s,CV a) {return a*s;}
__device__ inline float cdot(CV a,CV b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ inline CV ccross(CV a,CV b) {return cv(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]);}
__device__ inline CV cload(const float* a,int i=0) {return cv(a[3*i],a[3*i+1],a[3*i+2]);}
__device__ inline CV ccolumn(const float* R,int k) {return cv(R[k],R[3+k],R[6+k]);}
__device__ inline CV crot(const float* R,CV x) {return cv(cdot(cload(R),x),cdot(cload(R,1),x),cdot(cload(R,2),x));}
__device__ inline CV crotT(const float* R,CV x) {return cv(cdot(ccolumn(R,0),x),cdot(ccolumn(R,1),x),cdot(ccolumn(R,2),x));}
__device__ inline CV cnormalize(CV a) {float n=sqrtf(cdot(a,a));return n>0 ? a*(1.0f/n) : cv(0,0,0);}
__device__ inline void contact_frame(CV a,CV* frame) {
  a=cnormalize(a);
  CV b=(-.5f<a[1] && a[1]<.5f) ? cv(0,1,0) : cv(0,0,1);
  b=cnormalize(b-a*cdot(a,b));
  if (sqrtf(cdot(a,a))==0) b=cv(0,0,0);
  frame[0]=a;frame[1]=b;frame[2]=ccross(a,b);
}
__device__ inline const float* contact_parameter(ContactParameter a,int w,int count,int index,int width) {
  return a.data+((w%a.rows)*count+index)*width;
}
struct ContactMaterial {
  int dim;
  float margin,gap,adhesion,friction[5],solref[2],solreffriction[2],solimp[5];
};
__device__ inline ContactMaterial contact_material(const ContactInputs& a,int w,int g1,int g2,int pair) {
  ContactMaterial p{};
  if (pair>=0) {
    p.dim=a.pair_dim[pair];
    p.margin=*contact_parameter(a.pair_margin,w,a.explicit_pairs,pair,1);
    p.gap=*contact_parameter(a.pair_gap,w,a.explicit_pairs,pair,1);
    p.adhesion=*contact_parameter(a.pair_adhesion,w,a.explicit_pairs,pair,1);
    for (int k=0;k<5;++k) {
      p.friction[k]=contact_parameter(a.pair_friction,w,a.explicit_pairs,pair,5)[k];
      p.solimp[k]=contact_parameter(a.pair_solimp,w,a.explicit_pairs,pair,5)[k];
    }
    for (int k=0;k<2;++k) {
      p.solref[k]=contact_parameter(a.pair_solref,w,a.explicit_pairs,pair,2)[k];
      p.solreffriction[k]=contact_parameter(a.pair_solreffriction,w,a.explicit_pairs,pair,2)[k];
    }
  } else {
    p.margin=*contact_parameter(a.margin,w,a.geoms,g1,1)+*contact_parameter(a.margin,w,a.geoms,g2,1);
    p.gap=*contact_parameter(a.gap,w,a.geoms,g1,1)+*contact_parameter(a.gap,w,a.geoms,g2,1);
    float m1=*contact_parameter(a.solmix,w,a.geoms,g1,1),m2=*contact_parameter(a.solmix,w,a.geoms,g2,1),mix;
    const float *f1=contact_parameter(a.friction,w,a.geoms,g1,3),*f2=contact_parameter(a.friction,w,a.geoms,g2,3);
    float f[3],a1=*contact_parameter(a.adhesion,w,a.geoms,g1,1),a2=*contact_parameter(a.adhesion,w,a.geoms,g2,1);
    if (a.priority[g1]>a.priority[g2]) {
      mix=1; p.dim=a.condim[g1];p.adhesion=a1;
      for (int k=0;k<3;++k) f[k]=f1[k];
    } else if (a.priority[g2]>a.priority[g1]) {
      mix=0;p.dim=a.condim[g2];p.adhesion=a2;
      for (int k=0;k<3;++k) f[k]=f2[k];
    } else {
      mix=m1+m2!=0 ? m1/(m1+m2) : 0;
      if (m1<1.e-15f && m2<1.e-15f) mix=.5f;
      if (m1<1.e-15f && m2>=1.e-15f) mix=0;
      if (m1>=1.e-15f && m2<1.e-15f) mix=1;
      p.dim=a.condim[g1]>a.condim[g2] ? a.condim[g1] : a.condim[g2];p.adhesion=a1+a2;
      for (int k=0;k<3;++k) f[k]=fmaxf(f1[k],f2[k]);
    }
    p.friction[0]=f[0];p.friction[1]=f[0];p.friction[2]=f[1];p.friction[3]=f[2];p.friction[4]=f[2];
    const float *r1=contact_parameter(a.solref,w,a.geoms,g1,2),*r2=contact_parameter(a.solref,w,a.geoms,g2,2);
    const float *i1=contact_parameter(a.solimp,w,a.geoms,g1,5),*i2=contact_parameter(a.solimp,w,a.geoms,g2,5);
    // Pinned GPU semantics test both reference formats even at unequal
    // priorities; CPU2.3.7 instead copies the priority winner outright.
    for (int k=0;k<2;++k) p.solref[k]=r1[0]>0 && r2[0]>0 ? mix*r1[k]+(1-mix)*r2[k] : fminf(r1[k],r2[k]);
    for (int k=0;k<5;++k) p.solimp[k]=mix*i1[k]+(1-mix)*i2[k];
  }
  for (int k=0;k<5;++k) p.friction[k]=fmaxf(1.e-5f,p.friction[k]);
  return p;
}
__device__ inline int publish_contact(const ContactInputs& a,const ContactMaterial& p,int w,int g1,int g2,int pair,int sensor,int id,float dist,CV pos,const CV* frame) {
  bool active=dist<p.margin || (p.adhesion!=0 && dist<p.margin+p.gap);
  bool detected=dist<p.margin+p.gap;
  if ((pair==-2 || !detected) && sensor==-1) return 0;
  int kind=(pair>=-1 && detected ? 1 : 0) | (sensor>=0 ? 2 : 0);
  int slot=atomicAdd(a.contact_count,1);
  if (slot>=a.capacity) return 0;
  a.contact_dist[slot]=dist;a.contact_margin[slot]=p.margin;a.contact_adhesion[slot]=p.adhesion;
  a.contact_dim[slot]=p.adhesion!=0 && dist>=p.margin ? 1 : p.dim;
  a.contact_geom[2*slot]=g1;a.contact_geom[2*slot+1]=g2;a.contact_world[slot]=w;a.contact_type[slot]=kind;a.contact_collisionid[slot]=id;
  for (int k=0;k<3;++k) {
    a.contact_pos[3*slot+k]=pos[k];
    for (int j=0;j<3;++j) a.contact_frame[9*slot+3*k+j]=frame[k][j];
  }
  for (int k=0;k<5;++k) {a.contact_friction[5*slot+k]=p.friction[k];a.contact_solimp[5*slot+k]=p.solimp[k];}
  for (int k=0;k<2;++k) {a.contact_solref[2*slot+k]=p.solref[k];a.contact_solreffriction[2*slot+k]=p.solreffriction[k];}
  for (int k=0;k<a.address_columns;++k) a.contact_address[slot*a.address_columns+k]=-1;
  return int(active);
}
}
