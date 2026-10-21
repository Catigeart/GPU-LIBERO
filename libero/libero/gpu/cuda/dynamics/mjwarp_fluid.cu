// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned passive.py inertia-box fluid and support.py::_apply_ft(False).
#include "mjwarp_fluid.cuh"
#include "mjwarp_passive.cuh"
#include <cmath>
namespace flash::mjwarp {
namespace {
struct V {float x,y,z;};
__device__ inline V load3(const float* p,int i) {p+=3*i;return {p[0],p[1],p[2]};}
__device__ inline V operator-(V a,V b) {return {a.x-b.x,a.y-b.y,a.z-b.z};}
__device__ inline V cross(V a,V b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
__device__ inline V rotate(const float* r,V v) {
  return {r[0]*v.x+r[1]*v.y+r[2]*v.z,r[3]*v.x+r[4]*v.y+r[5]*v.z,r[6]*v.x+r[7]*v.y+r[8]*v.z};
}
__device__ inline V rotate_transpose(const float* r,V v) {
  return {r[0]*v.x+r[3]*v.y+r[6]*v.z,r[1]*v.x+r[4]*v.y+r[7]*v.z,r[2]*v.x+r[5]*v.y+r[8]*v.z};
}

__global__ void inertia_box_force(FluidInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.bodies) return;
  int w=t/a.bodies,b=t%a.bodies;
  float* output=a.body_force+6*t;
  float mass=a.mass[(w%a.mass_rows)*a.bodies+b];
  constexpr float minval=1.0e-15f,pi=3.14159265358979323846f;
  if (b==0 || mass<minval) {
    for (int k=0;k<6;++k) output[k]=0.0f;
    return;
  }
  const float* rotation=a.ximat+9*t;
  const float* velocity=a.cvel+6*t;
  V angular={velocity[0],velocity[1],velocity[2]},linear={velocity[3],velocity[4],velocity[5]};
  V offset=load3(a.xipos,t)-load3(a.subtree_com,w*a.bodies+a.root[b]);
  // cvel uses the root-CoM origin; translate to this body's inertial origin.
  V local_angular=rotate_transpose(rotation,angular);
  V local_linear=rotate_transpose(rotation,linear-cross(offset,angular));
  V wind=load3(a.wind,w%a.wind_rows);
  if (wind.x || wind.y || wind.z) local_linear=local_linear-rotate_transpose(rotation,wind);
  float density=a.density[w%a.density_rows],viscosity=a.viscosity[w%a.viscosity_rows];
  float box[3]={0.0f,0.0f,0.0f};
  if (viscosity>0.0f || density>0.0f) {
    V inertia=load3(a.inertia,(w%a.inertia_rows)*a.bodies+b);
    float scale=6.0f/mass;
    box[0]=sqrtf(fmaxf(minval,inertia.y+inertia.z-inertia.x)*scale);
    box[1]=sqrtf(fmaxf(minval,inertia.x+inertia.z-inertia.y)*scale);
    box[2]=sqrtf(fmaxf(minval,inertia.x+inertia.y-inertia.z)*scale);
  }
  float angular_local[3]={local_angular.x,local_angular.y,local_angular.z};
  float linear_local[3]={local_linear.x,local_linear.y,local_linear.z};
  float force[3]={0.0f,0.0f,0.0f},torque[3]={0.0f,0.0f,0.0f};
  if (viscosity>0.0f) {
    float diameter=(box[0]+box[1]+box[2])/3.0f;
    float diameter3=powf(diameter,3.0f);
    for (int k=0;k<3;++k) {
      torque[k]=-angular_local[k]*diameter3*pi*viscosity;
      force[k]=-3.0f*linear_local[k]*diameter*pi*viscosity;
    }
  }
  if (density>0.0f) {
    float power4[3]={powf(box[0],4.0f),powf(box[1],4.0f),powf(box[2],4.0f)};
    float scale=density/64.0f;
    for (int k=0;k<3;++k) {
      int i=(k+1)%3,j=(k+2)%3;
      // Keep x/z factor order as in the pinned component expressions.
      if (k==1) {i=0;j=2;}
      force[k]-=0.5f*density*box[i]*box[j]*fabsf(linear_local[k])*linear_local[k];
      torque[k]-=box[k]*(power4[i]+power4[j])*fabsf(angular_local[k])*angular_local[k]*scale;
    }
  }
  V world_force=rotate(rotation,{force[0],force[1],force[2]});
  V world_torque=rotate(rotation,{torque[0],torque[1],torque[2]});
  // Wrench storage is (force,torque); cvel/cdof storage is (angular,linear).
  output[0]=world_force.x;output[1]=world_force.y;output[2]=world_force.z;
  output[3]=world_torque.x;output[4]=world_torque.y;output[5]=world_torque.z;
}

__global__ void project_fluid(FluidInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.nv) return;
  int w=t/a.nv,d=t%a.nv,body=a.dof_body[d];
  const float* motion=a.cdof+6*t;
  float jac[6]={motion[3],motion[4],motion[5],motion[0],motion[1],motion[2]};
  V angular={motion[0],motion[1],motion[2]};
  V root_com=load3(a.subtree_com,w*a.bodies+a.root[body]);
  float sum=0.0f;
  for (int b=body;b<body+a.subtree_size[body];++b) {
    const float* force=a.body_force+6*(w*a.bodies+b);
    if (force[0]==0.0f && force[1]==0.0f && force[2]==0.0f && force[3]==0.0f && force[4]==0.0f && force[5]==0.0f) continue;
    V shift=cross(angular,load3(a.xipos,w*a.bodies+b)-root_com);
    float dot6=0.0f;
    for (int k=0;k<6;++k) dot6+=jac[k]*force[k];
    float dot3=shift.x*force[0]+shift.y*force[1]+shift.z*force[2];
    sum+=dot6+dot3;
  }
  a.qfrc[t]=sum;
}
// One world CTA produces current body wrenches and consumes them in the
// complete per-DOF subtree projection. No contact-pool work is included.
template<int Threads,bool Shared,bool Compose=false>
__global__ void world_fluid(FluidInputs a,PassiveInputs p,float* adhesion) {
  __shared__ float scratch[Shared ? 6*64 : 1];
  const int w=blockIdx.x,x=threadIdx.x;
  float* storage=Shared ? scratch : a.body_force+6*w*a.bodies;
  for(int b=x;b<a.bodies;b+=Threads) {
    const int t=w*a.bodies+b;
    float* output=storage+6*b;
    float mass=a.mass[(w%a.mass_rows)*a.bodies+b];
    constexpr float minval=1.0e-15f,pi=3.14159265358979323846f;
    if (b==0 || mass<minval) {
      for (int k=0;k<6;++k) output[k]=0.0f;
      continue;
    }
    const float* rotation=a.ximat+9*t;
    const float* velocity=a.cvel+6*t;
    V angular={velocity[0],velocity[1],velocity[2]},linear={velocity[3],velocity[4],velocity[5]};
    V offset=load3(a.xipos,t)-load3(a.subtree_com,w*a.bodies+a.root[b]);
    // cvel uses the root-CoM origin; translate to this body's inertial origin.
    V local_angular=rotate_transpose(rotation,angular);
    V local_linear=rotate_transpose(rotation,linear-cross(offset,angular));
    V wind=load3(a.wind,w%a.wind_rows);
    if (wind.x || wind.y || wind.z) local_linear=local_linear-rotate_transpose(rotation,wind);
    float density=a.density[w%a.density_rows],viscosity=a.viscosity[w%a.viscosity_rows];
    float box[3]={0.0f,0.0f,0.0f};
    if (viscosity>0.0f || density>0.0f) {
      V inertia=load3(a.inertia,(w%a.inertia_rows)*a.bodies+b);
      float scale=6.0f/mass;
      box[0]=sqrtf(fmaxf(minval,inertia.y+inertia.z-inertia.x)*scale);
      box[1]=sqrtf(fmaxf(minval,inertia.x+inertia.z-inertia.y)*scale);
      box[2]=sqrtf(fmaxf(minval,inertia.x+inertia.y-inertia.z)*scale);
    }
    float angular_local[3]={local_angular.x,local_angular.y,local_angular.z};
    float linear_local[3]={local_linear.x,local_linear.y,local_linear.z};
    float force[3]={0.0f,0.0f,0.0f},torque[3]={0.0f,0.0f,0.0f};
    if (viscosity>0.0f) {
      float diameter=(box[0]+box[1]+box[2])/3.0f;
      float diameter3=powf(diameter,3.0f);
      for (int k=0;k<3;++k) {
        torque[k]=-angular_local[k]*diameter3*pi*viscosity;
        force[k]=-3.0f*linear_local[k]*diameter*pi*viscosity;
      }
    }
    if (density>0.0f) {
      float power4[3]={powf(box[0],4.0f),powf(box[1],4.0f),powf(box[2],4.0f)};
      float scale=density/64.0f;
      for (int k=0;k<3;++k) {
        int i=(k+1)%3,j=(k+2)%3;
        // Keep x/z factor order as in the pinned component expressions.
        if (k==1) {i=0;j=2;}
        force[k]-=0.5f*density*box[i]*box[j]*fabsf(linear_local[k])*linear_local[k];
        torque[k]-=box[k]*(power4[i]+power4[j])*fabsf(angular_local[k])*angular_local[k]*scale;
      }
    }
    V world_force=rotate(rotation,{force[0],force[1],force[2]});
    V world_torque=rotate(rotation,{torque[0],torque[1],torque[2]});
    // Wrench storage is (force,torque); cvel/cdof storage is (angular,linear).
    output[0]=world_force.x;output[1]=world_force.y;output[2]=world_force.z;
    output[3]=world_torque.x;output[4]=world_torque.y;output[5]=world_torque.z;
  }
  __syncthreads();
  if constexpr(Shared)
    for(int i=x;i<6*a.bodies;i+=Threads) a.body_force[6*w*a.bodies+i]=storage[i];
  for(int d=x;d<a.nv;d+=Threads) {
    const int t=w*a.nv+d,body=a.dof_body[d];
    if constexpr(Threads>=512) if(a.subtree_size[body]>=64) continue;
    const float* motion=a.cdof+6*t;
    float jac[6]={motion[3],motion[4],motion[5],motion[0],motion[1],motion[2]};
    V angular={motion[0],motion[1],motion[2]};
    V root_com=load3(a.subtree_com,w*a.bodies+a.root[body]);
    float sum=0.0f;
    for (int b=body;b<body+a.subtree_size[body];++b) {
      const float* force=storage+6*b;
      if (force[0]==0.0f && force[1]==0.0f && force[2]==0.0f && force[3]==0.0f && force[4]==0.0f && force[5]==0.0f) continue;
      V shift=cross(angular,load3(a.xipos,w*a.bodies+b)-root_com);
      float dot6=0.0f;
      for (int k=0;k<6;++k) dot6+=jac[k]*force[k];
      float dot3=shift.x*force[0]+shift.y*force[1]+shift.z*force[2];
      sum+=dot6+dot3;
    }
    a.qfrc[t]=sum;
    if constexpr(Compose) {
      adhesion[t]=0.f;
      float value=p.spring[t];value+=p.damper[t];
      if(p.gravity_enabled && !p.actgravcomp[p.dof_joint[d]]) value+=p.gravcomp[t];
      if(p.has_fluid) value+=sum;
      if(p.has_adhesion) value+=0.f;
      p.passive[t]=value;
    }
  }
  if constexpr(Threads>=512) {
    // Long subtrees are rare stragglers. A whole warp consumes their body
    // contributions; short rows above keep one independent DOF per thread.
    const int lane=x%32,warp=x/32;
    for(int d=warp;d<a.nv;d+=Threads/32) {
      const int body=a.dof_body[d];
      if(a.subtree_size[body]<64) continue;
      const float* motion=a.cdof+6*(w*a.nv+d);
      const float jac[6]={motion[3],motion[4],motion[5],motion[0],motion[1],motion[2]};
      const V angular={motion[0],motion[1],motion[2]};
      const V root_com=load3(a.subtree_com,w*a.bodies+a.root[body]);
      float sum=0.f;
      for(int b=body+lane;b<body+a.subtree_size[body];b+=32) {
        const float* force=storage+6*b;
        if(force[0]==0 && force[1]==0 && force[2]==0 && force[3]==0 && force[4]==0 && force[5]==0) continue;
        const V shift=cross(angular,load3(a.xipos,w*a.bodies+b)-root_com);
        float dot6=0.f;
        for(int k=0;k<6;++k) dot6+=jac[k]*force[k];
        const float dot3=shift.x*force[0]+shift.y*force[1]+shift.z*force[2];
        sum+=dot6+dot3;
      }
      for(int offset=16;offset;offset/=2) sum+=__shfl_down_sync(0xffffffffu,sum,offset);
      if(!lane) {
        const int t=w*a.nv+d;a.qfrc[t]=sum;
        if constexpr(Compose) {
          adhesion[t]=0.f;
          float value=p.spring[t];value+=p.damper[t];
          if(p.gravity_enabled && !p.actgravcomp[p.dof_joint[d]]) value+=p.gravcomp[t];
          if(p.has_fluid) value+=sum;
          if(p.has_adhesion) value+=0.f;
          p.passive[t]=value;
        }
      }
    }
  }

}

}
}
extern "C" {
const char* flash_mjwarp_fluid_source_id() {return FLASH_MJW_FLUID_SOURCE_ID;}
std::size_t flash_mjwarp_fluid_arguments_bytes() {return sizeof(flash::mjwarp::FluidInputs);}
int flash_mjwarp_fluid_arch() {return FLASH_MJW_FLUID_ARCH;}
int flash_mjwarp_fluid_launch(const flash::mjwarp::FluidInputs* a,int phase,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->bodies<2 || a->nv<1 || a->mass_rows<1 || a->inertia_rows<1 ||
      a->wind_rows<1 || a->density_rows<1 || a->viscosity_rows<1) return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase==0) inertia_box_force<<<(a->worlds*a->bodies+255)/256,256,0,stream>>>(*a);
  else if (phase==1) project_fluid<<<(a->worlds*a->nv+255)/256,256,0,stream>>>(*a);
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
int flash_mjwarp_fluid_fused(const flash::mjwarp::FluidInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->bodies<2 || a->nv<1 || a->mass_rows<1 || a->inertia_rows<1 ||
      a->wind_rows<1 || a->density_rows<1 || a->viscosity_rows<1) return cudaErrorInvalidValue;
  if(a->bodies<=64) flash::mjwarp::world_fluid<128,true><<<a->worlds,128,0,stream>>>(*a,{},nullptr);
  else if(a->bodies>128 || a->nv>128) flash::mjwarp::world_fluid<512,false><<<a->worlds,512,0,stream>>>(*a,{},nullptr);
  else flash::mjwarp::world_fluid<128,false><<<a->worlds,128,0,stream>>>(*a,{},nullptr);
  return cudaGetLastError();
}

std::size_t flash_mjwarp_fluid_passive_bytes() {return sizeof(flash::mjwarp::PassiveInputs);}
int flash_mjwarp_fluid_passive(const flash::mjwarp::FluidInputs* a,
                             const flash::mjwarp::PassiveInputs* p,float* adhesion,cudaStream_t stream) {
  if (!a || !p || a->worlds<1 || a->bodies<2 || a->nv<1 || a->mass_rows<1 || a->inertia_rows<1 ||
      a->wind_rows<1 || a->density_rows<1 || a->viscosity_rows<1 || a->worlds!=p->worlds ||
      a->nv!=p->nv || p->joints<1 || !p->has_fluid || p->fluid!=a->qfrc || !adhesion || adhesion!=p->adhesion) return cudaErrorInvalidValue;
  if(a->bodies<=64) flash::mjwarp::world_fluid<128,true,true><<<a->worlds,128,0,stream>>>(*a,*p,adhesion);
  else if(a->bodies>128 || a->nv>128) flash::mjwarp::world_fluid<512,false,true><<<a->worlds,512,0,stream>>>(*a,*p,adhesion);
  else flash::mjwarp::world_fluid<128,false,true><<<a->worlds,128,0,stream>>>(*a,*p,adhesion);
  return cudaGetLastError();
}

}
