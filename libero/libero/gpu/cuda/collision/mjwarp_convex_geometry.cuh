// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned collision_gjk.py shape support. Geom cache lives within one GJK call.
#pragma once
#include "mjwarp_contact.cuh"
namespace flash::mjwarp {
struct ConvexGeom {
  CV pos,size;
  const float* rot;
  const float* vertices;
  const int* graph;
  int vertex_address,vertex_count,graph_address,cached_index;
  float margin;
};
struct ConvexSupport {CV point;int cached_index,vertex_index;};
__device__ inline float convex_sign(float x) {return x>0 ? 1.0f : x<0 ? -1.0f : 0.0f;}
__device__ inline ConvexSupport convex_support(const ConvexGeom& g,int type,CV direction) {
  ConvexSupport sp{cv(0,0,0),-1,-1};
  if (type==2) {sp.point=g.pos+(g.size[0]+.5f*g.margin)*direction;return sp;}
  CV local=crotT(g.rot,direction),res=cv(0,0,0);
  if (type==6) {
    // EPA uses vertex_index to recover actual edges and faces. At an
    // exactly zero component either extreme is a valid support maximizer;
    // choose the negative endpoint so bit0 still denotes -size. Returning
    // sign(0)*size == 0 with that bit would mislabel a face-interior point
    // as a corner and corrupt subsequent feature recovery.
    CV sign=cv(local[0]>0 ? 1 : -1,local[1]>0 ? 1 : -1,local[2]>0 ? 1 : -1);
    for (int k=0;k<3;++k) res[k]=sign[k]*g.size[k];
    sp.point=crot(g.rot,res)+g.pos;sp.vertex_index=int(sign[0]>0)+2*int(sign[1]>0)+4*int(sign[2]>0);
  } else if (type==3) {
    res=local*g.size[0];res[2]+=convex_sign(local[2])*g.size[1];sp.point=crot(g.rot,res)+g.pos;
  } else if (type==4) {
    for (int k=0;k<3;++k) res[k]=local[k]*g.size[k];res=cnormalize(res);
    for (int k=0;k<3;++k) res[k]*=g.size[k];sp.point=crot(g.rot,res)+g.pos;
  } else if (type==5) {
    float length=sqrtf(local[0]*local[0]+local[1]*local[1]);
    if (length>1.e-15f) {float scale=g.size[0]/length;res[0]=local[0]*scale;res[1]=local[1]*scale;}
    // The pinned cylinder selects the positive cap when local.z is zero;
    // capsule support instead uses sign(z), which gives the segment center.
    res[2]=local[2]>=0 ? g.size[1] : -g.size[1];sp.vertex_index=local[2]>=0 ? 0 : 1;sp.point=crot(g.rot,res)+g.pos;
  } else if (type==7) {
    float maximum=-1.e30f;
    if (g.graph_address==-1 || g.vertex_count<10) {
      if (g.cached_index>-1) {sp.cached_index=g.cached_index;sp.point=cload(g.vertices,g.cached_index);maximum=cdot(sp.point,local);}
      for (int i=0;i<g.vertex_count;++i) {
        CV vertex=cload(g.vertices,g.vertex_address+i);float support=cdot(vertex,local);
        if (support>maximum) {maximum=support;sp.point=vertex;sp.cached_index=g.vertex_address+i;}
      }
      sp.vertex_index=sp.cached_index-g.vertex_address;
    } else {
      int graph=g.graph_address,vertices=g.graph[graph],edgeadr=graph+2,globaladr=graph+2+vertices,localadr=graph+2+2*vertices;
      int previous=-1,best=g.cached_index>-1 ? g.cached_index : 0;
      maximum=cdot(local,cload(g.vertices,g.vertex_address+g.graph[globaladr+best]));
      while (best!=previous) {
        previous=best;int i=g.graph[edgeadr+best],neighbor=g.graph[localadr+i];
        while (neighbor>=0) {
          int vertex=g.graph[globaladr+neighbor];float support=cdot(local,cload(g.vertices,g.vertex_address+vertex));
          if (support>maximum) {best=neighbor;maximum=support;}
          neighbor=g.graph[localadr+(++i)];
        }
      }
      sp.cached_index=best;sp.vertex_index=g.graph[globaladr+best];sp.point=cload(g.vertices,g.vertex_address+sp.vertex_index);
    }
    sp.point=crot(g.rot,sp.point)+g.pos;
  }
  if (g.margin>0) sp.point=sp.point+direction*(.5f*g.margin);
  return sp;
}
}
