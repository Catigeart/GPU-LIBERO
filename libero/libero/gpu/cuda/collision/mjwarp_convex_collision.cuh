// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned rigid CCD caller for the seven convex pair families in LIBERO130.
// Types are launched serially and share retained global EPA scratch, matching
// the original per-type counter/capacity ownership. No shared-memory storage.
#pragma once
#include "mjwarp_box_manifold.cuh"
namespace flash::mjwarp {
struct ConvexCollisionInputs {
  ContactInputs contact;
  ContactParameter tolerance;
  int iterations,epa_iterations,naccd,vertex_capacity,face_capacity,horizon_capacity,warn;
  int active_types,type_pair[14],type_index[7];
  int *nccd,*overflow;
  float *vertices,*projection,*norm2;
  int *vertex_index,*face,*horizon;
};
__device__ inline ConvexGeom collision_convex_geom(const ContactInputs& a,int w,int geom,int type,CV position,float margin) {
  int mesh=a.geom_dataid[(w%a.dataid_rows)*a.geoms+geom];
  return {position,cload(contact_parameter(a.size,w,a.geoms,geom,3)),a.xmat+9*(w*a.geoms+geom),a.mesh_vert,a.mesh_graph,
    type==7 ? a.mesh_vertadr[mesh] : -1,type==7 ? a.mesh_vertnum[mesh] : 0,type==7 ? a.mesh_graphadr[mesh] : -1,-1,margin};
}
__device__ inline void convex_margin_gap(const ContactInputs& a,int w,int g1,int g2,int pair,float& margin,float& gap) {
  if (pair>=0) {
    margin=*contact_parameter(a.pair_margin,w,a.explicit_pairs,pair,1);gap=*contact_parameter(a.pair_gap,w,a.explicit_pairs,pair,1);
  } else {
    margin=*contact_parameter(a.margin,w,a.geoms,g1,1)+*contact_parameter(a.margin,w,a.geoms,g2,1);
    gap=*contact_parameter(a.gap,w,a.geoms,g1,1)+*contact_parameter(a.gap,w,a.geoms,g2,1);
  }
}
__device__ inline int convex_scratch_slot(const ConvexCollisionInputs& a,int type,int world) {
  int slot=atomicAdd(a.nccd+type,1);
  if (slot>=a.naccd) {
    if (a.warn&(1<<4)) printf("CCD overflow - please increase naccdmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CCD (or = 0 for all)\n",unsigned(a.naccd));
    atomicOr(a.overflow+world,1<<4);return -1;
  }
  return slot;
}
__device__ inline ConvexPolytope convex_scratch(const ConvexCollisionInputs& a,int slot) {
  return {a.vertices+3*slot*a.vertex_capacity,a.projection+3*slot*a.face_capacity,a.norm2+slot*a.face_capacity,
    a.vertex_index+slot*a.vertex_capacity,a.face+slot*a.face_capacity,a.horizon+slot*a.horizon_capacity,a.face_capacity,a.horizon_capacity,0,0,0,0,{}};
}
template<int Type1,int Type2>
__global__ void convex_contacts(ConvexCollisionInputs a,int type_index) {
  const ContactInputs& c=a.contact;
  for (int id=blockIdx.x*blockDim.x+threadIdx.x;id<min(*c.candidate_count,c.capacity);id+=gridDim.x*blockDim.x) {
    int g1=c.candidate_geom[2*id],g2=c.candidate_geom[2*id+1];
    if (c.geom_type[g1]!=Type1 || c.geom_type[g2]!=Type2) continue;
    int w=c.candidate_world[id],pair=c.candidate_ids[2*id],sensor=c.candidate_ids[2*id+1];float margin,gap;
    convex_margin_gap(c,w,g1,g2,pair,margin,gap);
    // Perform CCD in a translated frame, preserving original precision and
    // leaving public geometry poses untouched.
    CV origin=cload(c.xpos,w*c.geoms+g1),position=cload(c.xpos,w*c.geoms+g2)-origin;
    ConvexGeom one=collision_convex_geom(c,w,g1,Type1,cv(0,0,0),margin),two=collision_convex_geom(c,w,g2,Type2,position,margin);
    float tolerance=a.tolerance.data[w%a.tolerance.rows],cutoff=sensor>=0 ? 1.e32f : gap;
    ConvexPhase phase=gjk_phase(tolerance,cutoff,a.iterations,one,two,Type1,Type2,one.pos,two.pos);
    ConvexEPAResult result{phase.distance,1,phase.x1,phase.x2,-1};ConvexPolytope p{};int slot=-1;
    if (phase.needs_epa) {
      slot=convex_scratch_slot(a,type_index,w);if (slot<0) continue;p=convex_scratch(a,slot);
      result=epa_phase(tolerance,a.epa_iterations,phase.result,phase.geom1,phase.geom2,Type1,Type2,p,(a.warn&(1<<8))!=0,w,a.overflow,phase.require_enclosing);
    }
    if (result.distance>=gap && sensor<0) continue;
    // GJK/EPA use shapes expanded by half-margin on both sides. Public contact
    // distances are uninflated, as expected by row construction.
    float distance=result.distance+margin;ConvexManifold manifold{};manifold.count=result.ncontact;manifold.one[0]=result.x1;manifold.two[0]=result.x2;
    if constexpr (Type1==6 && Type2==6) {
      if (!phase.needs_epa && distance<0 && margin==0 && phase.result.dim>0) {
        slot=convex_scratch_slot(a,type_index,w);if (slot<0) continue;p=convex_scratch(a,slot);result.face=gjk_simplex_to_epa(phase.result,p);
      }
      if (result.face>=0) {
        manifold=box_multicontact(p,result.face,result.x1,result.x2,phase.geom1,phase.geom2);
        if (manifold.count<1) {manifold.count=1;manifold.one[0]=result.x1;manifold.two[0]=result.x2;}
      }
    }
    ContactMaterial material=contact_material(c,w,g1,g2,pair);CV frame[3];
    contact_frame(distance<=margin ? manifold.one[0]-manifold.two[0] : manifold.two[0]-manifold.one[0],frame);
    for (int i=0;i<manifold.count;++i) publish_contact(c,material,w,g1,g2,pair,sensor,i,manifold.count>1 ? manifold.distance[i] : distance,.5f*(manifold.one[i]+manifold.two[i])+origin,frame);
  }
}
}
