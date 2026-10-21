// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Read-current rigid kinematic queries; one thread per world/DOF or sensor.
#include "mjwarp_queries.cuh"
#include "mjwarp_spatial.cuh"
namespace flash::mjwarp {
namespace {
__device__ inline void point_jacobian(const QueryGeometry& g,int world,int body,int dof,const float* point,float* linear,float* angular) {
  if (!g.ancestor[body*g.ancestor_stride+dof]) {
    for (int k=0;k<3;++k) linear[k]=angular[k]=0;
    return;
  }
  const float* cdof=g.cdof+6*(world*g.dofs+dof);
  const float* com=g.com+3*(world*g.bodies+g.root[body]);float offset[3],cross[3];
  for (int k=0;k<3;++k) {offset[k]=point[k]-com[k];angular[k]=cdof[k];}
  cross3(cdof,offset,cross);
  for (int k=0;k<3;++k) linear[k]=cdof[k+3]+cross[k];
}
__global__ void point_jacobians(JacobianInputs a) {
  const QueryGeometry& g=a.geometry;
  for (long long index=blockIdx.x*blockDim.x+threadIdx.x;index<g.worlds*g.dofs;index+=gridDim.x*blockDim.x) {
    int i=int(index);
    int world=i/g.dofs,dof=i%g.dofs;float linear[3],angular[3];point_jacobian(g,world,a.body[world],dof,a.point+3*world,linear,angular);
    for (int k=0;k<3;++k) {
      int index=(world*3+k)*g.dofs+dof;
      if (a.position) a.position[index]=linear[k];
      if (a.rotation) a.rotation[index]=angular[k];
    }
  }
}
__global__ void force_torque_sensors(ForceSensorInputs a) {
  for (long long index=blockIdx.x*blockDim.x+threadIdx.x;index<a.worlds*a.sensors;index+=gridDim.x*blockDim.x) {
    int i=int(index);
    int world=i/a.sensors,sensor=i%a.sensors,site=a.object[sensor],body=a.site_body[site];
    const float* wrench=a.wrench+6*(world*a.bodies+body);const float* rotation=a.site_rotation+9*(world*a.sites+site);float value[3];
    if (a.type[sensor]==4) {
      for (int k=0;k<3;++k) value[k]=wrench[k+3];
    } else {
      const float* com=a.com+3*(world*a.bodies+a.root[body]);const float* position=a.site_position+3*(world*a.sites+site);float offset[3],torque[3];
      for (int k=0;k<3;++k) offset[k]=position[k]-com[k];
      cross3(offset,wrench+3,torque);
      for (int k=0;k<3;++k) value[k]=wrench[k]-torque[k];
    }
    float cutoff=a.cutoff[sensor];int address=world*a.channels+a.address[sensor];
    for (int k=0;k<3;++k) {
      float out=rotation[k]*value[0]+rotation[3+k]*value[1]+rotation[6+k]*value[2];
      if (cutoff>0) {
        if (a.datatype[sensor]==0) out=fminf(cutoff,fmaxf(-cutoff,out));
        else if (a.datatype[sensor]==1) out=fminf(cutoff,out);
      }
      a.output[address+k]=out;
    }
  }
}
__global__ void rigid_gravity_projection(GravityInputs a) {
  const QueryGeometry& g=a.geometry;int bodies=g.bodies-1;
  for (long long index=blockIdx.x*blockDim.x+threadIdx.x;index<g.worlds*bodies*g.dofs;index+=gridDim.x*blockDim.x) {
    int i=int(index);
    int dof=i%g.dofs,body=(i/g.dofs)%bodies+1,world=i/(g.dofs*bodies);float coefficient=a.coefficient[(world%a.coefficient_rows)*g.bodies+body];
    if (!coefficient) continue;
    const float* gravity=a.gravity+3*(world%a.gravity_rows);float mass=a.mass[(world%a.mass_rows)*g.bodies+body],force[3],linear[3],angular[3];
    for (int k=0;k<3;++k) force[k]=-gravity[k]*mass*coefficient;
    point_jacobian(g,world,body,dof,a.position+3*(world*g.bodies+body),linear,angular);
    float result=linear[0]*force[0]+linear[1]*force[1]+linear[2]*force[2];atomicAdd(a.output+world*g.dofs+dof,result);
  }
}
int blocks(int count) {return min((count+127)/128,4096);}
}
}
extern "C" {
const char* flash_mjwarp_queries_source_id() {return FLASH_MJW_QUERIES_SOURCE_ID;}
int flash_mjwarp_queries_arch() {return FLASH_MJW_QUERIES_ARCH;}
std::size_t flash_mjwarp_queries_jacobian_arguments_bytes() {return sizeof(flash::mjwarp::JacobianInputs);}
std::size_t flash_mjwarp_queries_sensor_arguments_bytes() {return sizeof(flash::mjwarp::ForceSensorInputs);}
std::size_t flash_mjwarp_queries_gravity_arguments_bytes() {return sizeof(flash::mjwarp::GravityInputs);}
int flash_mjwarp_queries_jacobian_launch(const flash::mjwarp::JacobianInputs* a,cudaStream_t stream) {
  if (!a || a->geometry.worlds<1 || a->geometry.bodies<1 || a->geometry.dofs<0) return int(cudaErrorInvalidValue);
  int count=a->geometry.worlds*a->geometry.dofs;
  if (count && (a->position || a->rotation)) flash::mjwarp::point_jacobians<<<flash::mjwarp::blocks(count),128,0,stream>>>(*a);
  return int(cudaGetLastError());
}
int flash_mjwarp_queries_sensor_launch(const flash::mjwarp::ForceSensorInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->bodies<1 || a->sensors<0 || a->sites<0 || a->channels<0) return int(cudaErrorInvalidValue);
  int count=a->worlds*a->sensors;
  if (count && !a->disabled) flash::mjwarp::force_torque_sensors<<<flash::mjwarp::blocks(count),128,0,stream>>>(*a);
  return int(cudaGetLastError());
}
int flash_mjwarp_queries_gravity_launch(const flash::mjwarp::GravityInputs* a,cudaStream_t stream) {
  if (!a || a->geometry.worlds<1 || a->geometry.bodies<1 || a->geometry.dofs<0 || a->gravity_rows<1 || a->mass_rows<1 || a->coefficient_rows<1) return int(cudaErrorInvalidValue);
  int count=a->geometry.worlds*(a->geometry.bodies-1)*a->geometry.dofs;
  if (count) flash::mjwarp::rigid_gravity_projection<<<flash::mjwarp::blocks(count),128,0,stream>>>(*a);
  return int(cudaGetLastError());
}
}
