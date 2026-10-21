// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned collision_gjk.py signed-volume distance subalgorithm. Vertex order is
// the GJK caller's order; this is not a generic unordered-simplex QP interface.
#pragma once
#include "mjwarp_convex_geometry.cuh"
namespace flash::mjwarp {
constexpr float convex_float_max=1.e30f,convex_minval=1.e-15f;
struct ConvexWeights {
  float v[4];
  __device__ float& operator[](int k) {return v[k];}
  __device__ float operator[](int k) const {return v[k];}
};
__device__ inline int convex_same_sign(float a,float b) {return (a>0 && b>0) || (a<0 && b<0);}
__device__ inline CV project_origin_line(CV a,CV b) {CV difference=b-a;float scale=-cdot(b,difference)/cdot(difference,difference);return b+scale*difference;}
__device__ inline CV project_origin_plane(CV a,CV b,CV c,bool& degenerate) {
  CV d21=b-a,d31=c-a,d32=c-b,n=ccross(d32,d21);float nv=cdot(n,b),nn=cdot(n,n);degenerate=false;
  if (nn==0) {degenerate=true;return cv(0,0,0);}
  if (nv!=0 && nn>convex_minval) return (nv/nn)*n;
  n=ccross(d21,d31);nv=cdot(n,a);nn=cdot(n,n);
  if (nn==0) {degenerate=true;return cv(0,0,0);}
  if (nv!=0 && nn>convex_minval) return (nv/nn)*n;
  n=ccross(d31,d32);nv=cdot(n,c);nn=cdot(n,n);return (nv/nn)*n;
}
__device__ inline ConvexWeights simplex_line(CV a,CV b) {
  CV projection=project_origin_line(a,b);float maximum=a[0]-b[0];int index=0;
  for (int k=1;k<3;++k) {float value=a[k]-b[k];if (fabsf(value)>=fabsf(maximum)) {maximum=value;index=k;}}
  float c1=projection[index]-b[index],c2=a[index]-projection[index];
  if (convex_same_sign(maximum,c1) && convex_same_sign(maximum,c2)) return {{c1/maximum,c2/maximum,0,0}};
  return {{0,1,0,0}};
}
__device__ inline float simplex_minor(CV a,CV b,CV c,int i,int j) {
  // Translation-invariant determinant: avoid six nearly cancelling absolute
  // coordinate products for a small triangle far from the origin.
  CV u=b-a,v=c-a;return u[i]*v[j]-u[j]*v[i];
}
__device__ inline float simplex_area(CV p,CV a,CV b,int i,int j) {
  CV u=a-p,v=b-p;return u[i]*v[j]-u[j]*v[i];
}
__device__ inline ConvexWeights simplex_triangle(CV a,CV b,CV c) {
  bool degenerate;CV projection=project_origin_plane(a,b,c,degenerate);
  if (degenerate) return simplex_line(a,b);
  float m14=simplex_minor(a,b,c,1,2),m24=simplex_minor(a,b,c,0,2),m34=simplex_minor(a,b,c,0,1),maximum;int i,j;
  if (fabsf(m14)>=fabsf(m24) && fabsf(m14)>=fabsf(m34)) {maximum=m14;i=1;j=2;}
  else if (fabsf(m24)>=fabsf(m34)) {maximum=m24;i=0;j=2;} else {maximum=m34;i=0;j=1;}
  float c31=simplex_area(projection,b,c,i,j),c32=simplex_area(projection,c,a,i,j),c33=simplex_area(projection,a,b,i,j);
  int comp1=convex_same_sign(maximum,c31),comp2=convex_same_sign(maximum,c32),comp3=convex_same_sign(maximum,c33);
  if (comp1 && comp2 && comp3) return {{c31/maximum,c32/maximum,c33/maximum,0}};
  ConvexWeights weights{{0,0,0,0}};float best=convex_float_max;
  if (!comp1) {ConvexWeights sub=simplex_line(b,c);CV x=sub[0]*b+sub[1]*c;weights={{0,sub[0],sub[1],0}};best=cdot(x,x);}
  if (!comp2) {ConvexWeights sub=simplex_line(a,c);CV x=sub[0]*a+sub[1]*c;float d=cdot(x,x);if (d<best) {weights={{sub[0],0,sub[1],0}};best=d;}}
  if (!comp3) {ConvexWeights sub=simplex_line(a,b);CV x=sub[0]*a+sub[1]*b;float d=cdot(x,x);if (d<best) weights={{sub[0],sub[1],0,0}};}
  return weights;
}
__device__ inline float simplex_det(CV a,CV b,CV c) {return cdot(a,ccross(b,c));}
__device__ inline ConvexWeights simplex_tetrahedron(CV a,CV b,CV c,CV d) {
  float c41=-simplex_det(b,c,d),c42=simplex_det(a,c,d),c43=-simplex_det(a,b,d),c44=simplex_det(a,b,c),det=c41+c42+c43+c44;
  int comp1=convex_same_sign(det,c41),comp2=convex_same_sign(det,c42),comp3=convex_same_sign(det,c43),comp4=convex_same_sign(det,c44);
  if (comp1 && comp2 && comp3 && comp4) return {{c41/det,c42/det,c43/det,c44/det}};
  ConvexWeights weights{{0,0,0,0}};float best=convex_float_max;
  if (!comp1) {ConvexWeights sub=simplex_triangle(b,c,d);CV x=sub[0]*b+sub[1]*c+sub[2]*d;weights={{0,sub[0],sub[1],sub[2]}};best=cdot(x,x);}
  if (!comp2) {ConvexWeights sub=simplex_triangle(a,c,d);CV x=sub[0]*a+sub[1]*c+sub[2]*d;float distance=cdot(x,x);if (distance<best) {weights={{sub[0],0,sub[1],sub[2]}};best=distance;}}
  if (!comp3) {ConvexWeights sub=simplex_triangle(a,b,d);CV x=sub[0]*a+sub[1]*b+sub[2]*d;float distance=cdot(x,x);if (distance<best) {weights={{sub[0],sub[1],0,sub[2]}};best=distance;}}
  if (!comp4) {ConvexWeights sub=simplex_triangle(a,b,c);CV x=sub[0]*a+sub[1]*b+sub[2]*c;float distance=cdot(x,x);if (distance<best) weights={{sub[0],sub[1],sub[2],0}};}
  return weights;
}
__device__ inline ConvexWeights simplex_distance(int count,const CV* s) {
  if (count==4) return simplex_tetrahedron(s[0],s[1],s[2],s[3]);
  if (count==3) return simplex_triangle(s[0],s[1],s[2]);
  if (count==2) return simplex_line(s[0],s[1]);
  return {{1,0,0,0}};
}
__device__ inline CV simplex_combine(int count,ConvexWeights w,const CV* s) {
  if (count==1) return w[0]*s[0];
  if (count==2) return w[0]*s[0]+w[1]*s[1];
  if (count==3) return w[0]*s[0]+w[1]*s[1]+w[2]*s[2];
  return w[0]*s[0]+w[1]*s[1]+w[2]*s[2]+w[3]*s[3];
}
}
