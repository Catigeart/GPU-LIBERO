// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Readable convex numerical leaves, isolated function probes and an explicit
// qualified-pair production collision entry. Selection remains a host policy.
#include "mjwarp_convex.cuh"
namespace flash::mjwarp {
namespace {
__device__ ConvexGeom probe_geometry(const ConvexGJKInputs& a,int w,int geom,CV position,int cache,float margin) {
  int type=a.geom_type[geom],mesh=a.geom_dataid[(w%a.dataid_rows)*a.geoms+geom];
  return {position,cload(a.size,(w%a.size_rows)*a.geoms+geom),a.xmat+9*(w*a.geoms+geom),a.vertices,a.mesh_graph,
    type==7 ? a.mesh_vertadr[mesh] : -1,type==7 ? a.mesh_vertnum[mesh] : 0,type==7 ? a.mesh_graphadr[mesh] : -1,cache,margin};
}
__global__ void manifold_points(ConvexManifoldInputs all) {
  ConvexGJKInputs a=all.query;
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.requests;i+=gridDim.x*blockDim.x) {
    int w=a.world[i],g1=a.pair[2*i],g2=a.pair[2*i+1];
    ConvexGeom one=probe_geometry(a,w,g1,cload(a.position1,i),a.cache1[i],a.margin1[i]);
    ConvexGeom two=probe_geometry(a,w,g2,cload(a.position2,i),a.cache2[i],a.margin2[i]);
    // The private algorithm only reads the const frozen polytope slice.
    ConvexPolytope p{};p.vertices=const_cast<float*>(all.vertices+3*i*all.vertex_capacity);p.vertex_index=const_cast<int*>(all.vertex_index+i*all.vertex_capacity);p.face=const_cast<int*>(all.face+i*all.face_capacity);
    ConvexManifold result=box_multicontact(p,all.face_index[i],cload(all.x1,i),cload(all.x2,i),one,two);all.count[i]=result.count;
    for (int k=0;k<4;++k) {convex_store(all.witness1,4*i+k,result.one[k]);convex_store(all.witness2,4*i+k,result.two[k]);all.distance[4*i+k]=result.distance[k];}
  }
}
__global__ void ccd_points(ConvexCCDInputs all) {
  ConvexGJKInputs a=all.query;
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.requests;i+=gridDim.x*blockDim.x) {
    int w=a.world[i],g1=a.pair[2*i],g2=a.pair[2*i+1],type1=a.geom_type[g1],type2=a.geom_type[g2];
    ConvexGeom one=probe_geometry(a,w,g1,cload(a.position1,i),a.cache1[i],a.margin1[i]);
    ConvexGeom two=probe_geometry(a,w,g2,cload(a.position2,i),a.cache2[i],a.margin2[i]);
    ConvexPhase phase{};
    if (all.use_frozen_gjk) {
      ConvexGJKResult r{};r.dim=all.frozen.count[i];r.dist=all.frozen.distance[i];r.x1=cload(all.frozen.x1,i);r.x2=cload(all.frozen.x2,i);
      for (int j=0;j<4;++j) {r.simplex[j]=cload(all.frozen.simplex,4*i+j);r.simplex1[j]=cload(all.frozen.simplex1,4*i+j);r.simplex2[j]=cload(all.frozen.simplex2,4*i+j);r.vertex1[j]=all.frozen.vertex1[4*i+j];r.vertex2[j]=all.frozen.vertex2[4*i+j];}
      phase={true,r.dist,r.x1,r.x2,r,one,two};
    } else phase=gjk_phase(a.tolerance[i],a.cutoff[i],a.iterations[i],one,two,type1,type2,cload(a.initial1,i),cload(a.initial2,i));
    ConvexPolytope p{all.vertices+3*i*all.vertex_capacity,all.projection+3*i*all.face_capacity,all.norm2+i*all.face_capacity,
      all.vertex_index+i*all.vertex_capacity,all.face+i*all.face_capacity,all.horizon+i*all.horizon_capacity,all.face_capacity,all.horizon_capacity,0,0,0,0,{}};
    ConvexEPAResult result{phase.distance,1,phase.x1,phase.x2,-1};
    if (phase.needs_epa) result=epa_phase(a.tolerance[i],all.epa_iterations,phase.result,phase.geom1,phase.geom2,type1,type2,p,all.warn!=0,w,all.overflow,phase.require_enclosing);
    all.needs_epa_out[i]=int(phase.needs_epa);all.ncontact_out[i]=result.ncontact;all.face_index_out[i]=result.face;all.nvert_out[i]=p.nvert;all.nface_out[i]=p.nface;all.status_out[i]=p.status;
    a.separated_out[i]=int(phase.result.separated);a.dim_out[i]=phase.result.dim;a.index1_out[i]=phase.geom1.cached_index;a.index2_out[i]=phase.geom2.cached_index;
    a.dist_out[i]=result.distance;convex_store(a.x1_out,i,result.x1);convex_store(a.x2_out,i,result.x2);
    for (int j=0;j<4;++j) {
      a.vertex1_out[4*i+j]=phase.result.vertex1[j];a.vertex2_out[4*i+j]=phase.result.vertex2[j];
      convex_store(a.simplex_out,4*i+j,phase.result.simplex[j]);convex_store(a.simplex1_out,4*i+j,phase.result.simplex1[j]);convex_store(a.simplex2_out,4*i+j,phase.result.simplex2[j]);
    }
  }
}
__global__ void gjk_points(ConvexGJKInputs a) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.requests;i+=gridDim.x*blockDim.x) {
    int w=a.world[i],g1=a.pair[2*i],g2=a.pair[2*i+1];
    ConvexGeom one=probe_geometry(a,w,g1,cload(a.position1,i),a.cache1[i],a.margin1[i]);
    ConvexGeom two=probe_geometry(a,w,g2,cload(a.position2,i),a.cache2[i],a.margin2[i]);ConvexGJKCertificate certificate{};
    ConvexGJKResult result=gjk_distance(a.tolerance[i],a.iterations[i],one,two,cload(a.initial1,i),cload(a.initial2,i),a.geom_type[g1],a.geom_type[g2],a.cutoff[i],a.discrete[i]!=0,&certificate);
    a.separated_out[i]=int(result.separated);a.dim_out[i]=result.dim;a.index1_out[i]=result.index1;a.index2_out[i]=result.index2;a.dist_out[i]=result.dist;
    a.valid_out[i]=certificate.valid;a.stop_out[i]=certificate.stop;a.iterations_out[i]=certificate.iterations;a.queries_out[i]=certificate.support_queries;a.lower_out[i]=certificate.lower;
    for (int k=0;k<3;++k) {a.x1_out[3*i+k]=result.x1[k];a.x2_out[3*i+k]=result.x2[k];a.direction_out[3*i+k]=certificate.direction[k];}
    for (int j=0;j<4;++j) {
      a.vertex1_out[4*i+j]=result.vertex1[j];a.vertex2_out[4*i+j]=result.vertex2[j];
      for (int k=0;k<3;++k) {a.simplex_out[12*i+3*j+k]=result.simplex[j][k];a.simplex1_out[12*i+3*j+k]=result.simplex1[j][k];a.simplex2_out[12*i+3*j+k]=result.simplex2[j][k];}
    }
  }
}
__global__ void simplex_points(ConvexSimplexInputs a) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.requests;i+=gridDim.x*blockDim.x) {
    CV simplex[4];for (int k=0;k<4;++k) simplex[k]=cload(a.simplex,4*i+k);
    ConvexWeights weights=simplex_distance(a.count[i],simplex);CV point=simplex_combine(a.count[i],weights,simplex);int mask=0;
    for (int k=0;k<4;++k) {a.weights[4*i+k]=weights[k];if (weights[k]!=0) mask|=1<<k;}
    for (int k=0;k<3;++k) a.point[3*i+k]=point[k];a.distance[i]=sqrtf(cdot(point,point));a.mask[i]=mask;
  }
}
__global__ void support_points(ConvexSupportInputs a) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.requests;i+=gridDim.x*blockDim.x) {
    int w=a.world[i],geom=a.geom[i],type=a.geom_type[geom],mesh=a.geom_dataid[(w%a.dataid_rows)*a.geoms+geom];
    ConvexGeom g{cload(a.xpos,w*a.geoms+geom),cload(a.size,(w%a.size_rows)*a.geoms+geom),a.xmat+9*(w*a.geoms+geom),a.vertices,a.mesh_graph,
      type==7 ? a.mesh_vertadr[mesh] : -1,type==7 ? a.mesh_vertnum[mesh] : 0,type==7 ? a.mesh_graphadr[mesh] : -1,a.cached[i],a.margin[i]};
    ConvexSupport out=convex_support(g,type,cload(a.direction,i));
    for (int k=0;k<3;++k) a.point[3*i+k]=out.point[k];a.cached_out[i]=out.cached_index;a.vertex_out[i]=out.vertex_index;
  }
}
}
}
extern "C" {
const char* flash_mjwarp_convex_source_id() {return FLASH_MJW_CONVEX_SOURCE_ID;}
std::size_t flash_mjwarp_convex_arguments_bytes() {return sizeof(flash::mjwarp::ConvexSupportInputs);}
int flash_mjwarp_convex_arch() {return FLASH_MJW_CONVEX_ARCH;}
std::size_t flash_mjwarp_convex_collision_arguments_bytes() {return sizeof(flash::mjwarp::ConvexCollisionInputs);}
int flash_mjwarp_convex_collision_launch(const flash::mjwarp::ConvexCollisionInputs* a,cudaStream_t stream) {
  if (!a || a->active_types<0 || a->active_types>7 || a->naccd<0 || a->contact.capacity<0) return int(cudaErrorInvalidValue);
  if (!a->active_types) return int(cudaSuccess);
  cudaError_t error=cudaMemsetAsync(a->nccd,0,66*sizeof(int),stream);if (error!=cudaSuccess) return int(error);
  int blocks=(a->contact.capacity+127)/128;if (blocks>4096) blocks=4096;
  if (!blocks) return int(cudaSuccess);
  for (int i=0;i<a->active_types;++i) {
    // One precompiled specialization per admitted family. Serial stream order
    // lets every type reuse the same EPA pool without cross-type aliasing.
    switch (8*a->type_pair[2*i]+a->type_pair[2*i+1]) {
      case 29: flash::mjwarp::convex_contacts<3,5><<<blocks,128,0,stream>>>(*a,a->type_index[i]);break;
      case 31: flash::mjwarp::convex_contacts<3,7><<<blocks,128,0,stream>>>(*a,a->type_index[i]);break;
      case 46: flash::mjwarp::convex_contacts<5,6><<<blocks,128,0,stream>>>(*a,a->type_index[i]);break;
      case 47: flash::mjwarp::convex_contacts<5,7><<<blocks,128,0,stream>>>(*a,a->type_index[i]);break;
      case 54: flash::mjwarp::convex_contacts<6,6><<<blocks,128,0,stream>>>(*a,a->type_index[i]);break;
      case 55: flash::mjwarp::convex_contacts<6,7><<<blocks,128,0,stream>>>(*a,a->type_index[i]);break;
      case 63: flash::mjwarp::convex_contacts<7,7><<<blocks,128,0,stream>>>(*a,a->type_index[i]);break;
      default:return int(cudaErrorInvalidValue);
    }
    error=cudaGetLastError();if (error!=cudaSuccess) return int(error);
  }
  return int(cudaSuccess);
}
std::size_t flash_mjwarp_convex_manifold_arguments_bytes() {return sizeof(flash::mjwarp::ConvexManifoldInputs);}
int flash_mjwarp_convex_manifold_launch(const flash::mjwarp::ConvexManifoldInputs* a,cudaStream_t stream) {
  if (!a || a->query.worlds<1 || a->query.requests<0 || a->vertex_capacity<6 || a->face_capacity<1) return int(cudaErrorInvalidValue);
  if (a->query.requests) {int blocks=(a->query.requests+127)/128;if (blocks>4096) blocks=4096;flash::mjwarp::manifold_points<<<blocks,128,0,stream>>>(*a);}
  return int(cudaGetLastError());
}
std::size_t flash_mjwarp_convex_ccd_arguments_bytes() {return sizeof(flash::mjwarp::ConvexCCDInputs);}
int flash_mjwarp_convex_ccd_launch(const flash::mjwarp::ConvexCCDInputs* a,cudaStream_t stream) {
  if (!a || a->query.worlds<1 || a->query.geoms<1 || a->query.requests<0 || a->epa_iterations<0 || a->vertex_capacity<10+2*a->epa_iterations || a->face_capacity<6 || a->horizon_capacity<1) return int(cudaErrorInvalidValue);
  if (a->query.requests) {int blocks=(a->query.requests+127)/128;if (blocks>4096) blocks=4096;flash::mjwarp::ccd_points<<<blocks,128,0,stream>>>(*a);}
  return int(cudaGetLastError());
}
std::size_t flash_mjwarp_convex_gjk_arguments_bytes() {return sizeof(flash::mjwarp::ConvexGJKInputs);}
int flash_mjwarp_convex_gjk_launch(const flash::mjwarp::ConvexGJKInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->geoms<1 || a->requests<0) return int(cudaErrorInvalidValue);
  if (a->requests) {int blocks=(a->requests+127)/128;if (blocks>4096) blocks=4096;flash::mjwarp::gjk_points<<<blocks,128,0,stream>>>(*a);}
  return int(cudaGetLastError());
}
std::size_t flash_mjwarp_convex_simplex_arguments_bytes() {return sizeof(flash::mjwarp::ConvexSimplexInputs);}
int flash_mjwarp_convex_simplex_launch(const flash::mjwarp::ConvexSimplexInputs* a,cudaStream_t stream) {
  if (!a || a->requests<0) return int(cudaErrorInvalidValue);
  if (a->requests) {int blocks=(a->requests+127)/128;if (blocks>4096) blocks=4096;flash::mjwarp::simplex_points<<<blocks,128,0,stream>>>(*a);}
  return int(cudaGetLastError());
}
int flash_mjwarp_convex_support_launch(const flash::mjwarp::ConvexSupportInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->geoms<1 || a->requests<0) return int(cudaErrorInvalidValue);
  if (a->requests) {int blocks=(a->requests+127)/128;if (blocks>4096) blocks=4096;flash::mjwarp::support_points<<<blocks,128,0,stream>>>(*a);}
  return int(cudaGetLastError());
}
}
