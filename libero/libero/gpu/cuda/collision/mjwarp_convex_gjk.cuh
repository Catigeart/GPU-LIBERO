// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned collision_gjk.py gjk and discrete direction tuning. Geometry and
// simplex storage are private to each thread/invocation, not shared per world.
#pragma once
#include "mjwarp_convex_simplex.cuh"
namespace flash::mjwarp {
struct ConvexGJKResult {
  bool separated;
  float dist;
  CV x1,x2;
  int dim;
  CV simplex[4],simplex1[4],simplex2[4];
  int vertex1[4],vertex2[4],index1,index2;
};
// Optional function-probe certificate. Production calls pass nullptr; it does
// not add numerical checks, physical state or shared-memory synchronization.
struct ConvexGJKCertificate {CV direction;float lower;int valid,stop,iterations,support_queries;};
__device__ inline CV convex_div(CV x,float value) {return cv(x[0]/value,x[1]/value,x[2]/value);}
__device__ inline void gjk_support_pair(const ConvexGeom& g1,const ConvexGeom& g2,int type1,int type2,CV x,float norm,const CV* simplex,int count,bool discrete,ConvexSupport& a,ConvexSupport& b,CV& support_direction) {
  CV direction=convex_div(x,norm);
  if (discrete && norm<1.e-4f) {
    if (count==2) {
      CV edge=simplex[1]-simplex[0];float length2=cdot(edge,edge);
      if (length2>1.e-30f) {
        float projection=cdot(direction,edge)/length2;direction=direction-projection*edge;float length=sqrtf(cdot(direction,direction));
        if (length>convex_minval) direction=convex_div(direction,length);
      }
    } else if (count==3) {
      CV e1=simplex[1]-simplex[0],e2=simplex[2]-simplex[0],normal=ccross(e1,e2);float length=sqrtf(cdot(normal,normal));
      if (length>convex_minval) direction=convex_div(convex_sign(cdot(direction,normal))*normal,length);
    }
  }
  support_direction=direction;
  a=convex_support(g1,type1,-direction);b=convex_support(g2,type2,direction);
}
__device__ inline ConvexGJKResult gjk_distance(float tolerance,int max_iterations,ConvexGeom g1,ConvexGeom g2,CV initial1,CV initial2,int type1,int type2,float cutoff,bool discrete,ConvexGJKCertificate* certificate=nullptr) {
  CV simplex[4]{},s1[4]{},s2[4]{};int vertex1[4]{},vertex2[4]{},count=0;ConvexWeights weights{{1,0,0,0}};
  float epsilon=discrete ? 0.0f : .5f*tolerance*tolerance,min_norm=discrete ? convex_minval : tolerance;
  CV x=initial1-initial2;float norm=sqrtf(cdot(x,x)),previous_norm=0;
  if (certificate) *certificate={};
  for (int iteration=0;iteration<max_iterations;++iteration) {
    if (certificate) certificate->iterations=iteration+1;
    if (norm<min_norm || fabsf(previous_norm-norm)<convex_minval) {if (certificate) certificate->stop=norm<min_norm ? 1 : 2;break;}
    ConvexSupport a,b;CV support_direction;gjk_support_pair(g1,g2,type1,type2,x,norm,simplex,count,discrete,a,b,support_direction);
    if (certificate) certificate->support_queries+=2;
    s1[count]=a.point;g1.cached_index=a.cached_index;vertex1[count]=a.vertex_index;
    s2[count]=b.point;g2.cached_index=b.cached_index;vertex2[count]=b.vertex_index;
    simplex[count]=s1[count]-s2[count];
    if (cdot(x,x-simplex[count])<epsilon) {if (certificate) certificate->stop=3;break;}
    float lower=cdot(x,simplex[count]);
    bool cutoff_reached=(cutoff==0 && lower>0) || (cutoff!=0 && cutoff<convex_float_max && lower>0 && lower>=cutoff*norm);
    CV certificate_direction=x;float certificate_norm=norm;
    if (cutoff_reached && discrete && norm<1.e-4f && (count==2 || count==3)) {
      // A support point is extremal in the direction actually queried. The
      // discrete tuning above can rotate that direction away from x: dot(x,w)
      // then is NOT a separating lower bound. Keep the fast untuned path and
      // certify a proposed tuned exit with its own support plane instead.
      lower=cdot(support_direction,simplex[count]);
      certificate_norm=sqrtf(cdot(support_direction,support_direction));
      certificate_direction=support_direction;
      cutoff_reached=certificate_norm>convex_minval && lower>0
          && (cutoff==0 || (cutoff<convex_float_max && lower>=cutoff*certificate_norm));
    }
    if (cutoff_reached) {
      ConvexGJKResult result{};result.separated=true;result.dim=0;result.dist=convex_float_max;result.index1=g1.cached_index;result.index2=g2.cached_index;
      if (certificate) {certificate->direction=convex_div(certificate_direction,certificate_norm);certificate->lower=lower/certificate_norm;certificate->valid=1;certificate->stop=cutoff==0 ? 4 : 5;}
      return result;
    }
    weights=simplex_distance(count+1,simplex);count=0;
    for (int i=0;i<4;++i) {
      if (weights[i]==0) continue;
      simplex[count]=simplex[i];s1[count]=s1[i];s2[count]=s2[i];vertex1[count]=vertex1[i];vertex2[count]=vertex2[i];weights[count]=weights[i];++count;
    }
    if (count<1) {if (certificate) certificate->stop=6;break;}
    x=simplex_combine(count,weights,simplex);previous_norm=norm;norm=sqrtf(cdot(x,x));
    if (count==4) {if (certificate) certificate->stop=7;break;}
  }
  ConvexGJKResult result{};result.separated=false;result.x1=count==0 ? initial1 : simplex_combine(count,weights,s1);result.x2=count==0 ? initial2 : simplex_combine(count,weights,s2);
  if (norm>0) {
    CV direction=convex_div(x,norm);ConvexSupport a=convex_support(g1,type1,-direction),b=convex_support(g2,type2,direction);float lower=cdot(x,a.point-b.point);result.separated=lower>0;
    // The original post-test does not update geom cached indices.
    if (certificate) {certificate->direction=direction;certificate->lower=lower/norm;certificate->valid=1;certificate->support_queries+=2;}
  }
  result.dist=count==4 && !result.separated ? 0.0f : norm;result.dim=count;
  for (int i=0;i<4;++i) {result.simplex[i]=simplex[i];result.simplex1[i]=s1[i];result.simplex2[i]=s2[i];result.vertex1[i]=vertex1[i];result.vertex2[i]=vertex2[i];}
  result.index1=g1.cached_index;result.index2=g2.cached_index;return result;
}
}
