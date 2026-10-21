// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned MJWarp constraint.py: rigid dense Euler row construction.
// Shared row mathematics stays visible here; no upstream numerical kernel call.
#include "mjwarp_constraint_build.cuh"
#include <cmath>

namespace flash::mjwarp {
namespace {
struct V { float x,y,z; };
struct KBI { float k,b,imp; };
__device__ inline V load3(const float* p,int i) {p+=3*i;return {p[0],p[1],p[2]};}
__device__ inline V operator-(V a,V b) {return {a.x-b.x,a.y-b.y,a.z-b.z};}
__device__ inline V operator+(V a,V b) {return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ inline V cross(V a,V b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
__device__ inline float dot(V a,V b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ inline float clamp(float x,float lo,float hi) {return fminf(fmaxf(x,lo),hi);}
__device__ inline const float* parameter(ConstraintBroadcast p,int w,int width,int components,int i=0) {
  return p.data+((w%p.rows)*width+i)*components;
}

__device__ KBI impedance(int refsafe,float timestep,const float* ref,const float* imp,float position) {
  constexpr float minval=1.0e-15f,minimp=0.0001f,maximp=0.9999f;
  float time=refsafe ? fmaxf(ref[0],2.0f*timestep) : ref[0];
  float lo=clamp(imp[0],minimp,maximp),hi=clamp(imp[1],minimp,maximp);
  float width=fmaxf(minval,imp[2]),mid=clamp(imp[3],minimp,maximp),power=fmaxf(1.0f,imp[4]);
  float k=ref[0]<=0.0f ? -ref[0]/fmaxf(minval,hi*hi) : 1.0f/fmaxf(minval,hi*hi*time*time*ref[1]*ref[1]);
  float b=ref[1]<=0.0f ? -ref[1]/fmaxf(minval,hi) : 2.0f/fmaxf(minval,hi*time);
  float x=fabsf(position)/width,result;
  if (lo==hi || imp[2]<=minval) result=0.5f*(lo+hi);
  else if (x<=0.0f) result=lo;
  else if (x>=1.0f) result=hi;
  else if (power==1.0f) result=lo+x*(hi-lo);
  else if (x<=mid) result=clamp(lo+(1.0f/powf(mid,power-1.0f))*powf(x,power)*(hi-lo),lo,hi);
  else result=clamp(lo+(1.0f-(1.0f/powf(1.0f-mid,power-1.0f))*powf(1.0f-x,power))*(hi-lo),lo,hi);
  return {k,b,result};
}

__device__ void publish_row(ConstraintBuildInputs a,int w,int r,float position,float impedance_position,
    float weight,const float* ref,const float* imp,float margin,float velocity,float friction,int type,int id) {
  KBI kbi=impedance(a.refsafe,parameter(a.timestep,w,1,1)[0],ref,imp,impedance_position);
  float k=type==1 ? 0.0f : kbi.k;
  int t=w*a.rows+r;
  a.row_D[t]=1.0f/fmaxf(weight*(1.0f-kbi.imp)/kbi.imp,1.0e-15f);
  a.row_aref[t]=-k*kbi.imp*position-kbi.b*velocity;
  a.row_vel[t]=velocity;a.row_pos[t]=position+margin;a.row_margin[t]=margin;
  a.row_friction[t]=friction;a.row_type[t]=type;a.row_id[t]=id;
}

// Dense construction leaves sparse-only row metadata untouched. All counters
// reset even when constraints are disabled. Contact-enabled Jqvel is cleared
// across its complete allocated extent, including inactive rows.
__global__ void reset_counts(ConstraintBuildInputs a) {
  int stride=a.rows>0 ? a.rows : 1;
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*stride) return;
  int w=t/stride,r=t%stride;
  if (r==0) {a.ne[w]=0;a.nf[w]=0;a.nl[w]=0;a.nefc[w]=0;a.jtdaj_count[w]=0;}
  if (a.constraints_enabled && a.contacts_enabled && r<a.rows) a.Jqvel[t]=0.0f;
}

__global__ void dof_friction_rows(ConstraintBuildInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.nv) return;
  int w=t/a.nv,d=t%a.nv;
  float friction=parameter(a.dof_friction,w,a.nv,1,d)[0];
  if (friction<=0.0f) return;
  atomicAdd(a.nf+w,1);int r=atomicAdd(a.nefc+w,1);
  if (r>=a.rows) return;
  float* J=a.J+(w*a.padded_rows+r)*a.padded_nv;
  for (int i=0;i<a.nv;++i) J[i]=0.0f;
  J[d]=1.0f;
  publish_row(a,w,r,0.0f,0.0f,parameter(a.dof_weight,w,a.nv,1,d)[0],
    parameter(a.dof_ref,w,a.nv,2,d),parameter(a.dof_imp,w,a.nv,5,d),0.0f,a.qvel[t],friction,1,d);
}

__global__ void scalar_limit_rows(ConstraintBuildInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.limited) return;
  int w=t/a.limited,j=a.limited_joint[t%a.limited],d=a.joint_dof[j];
  const float* range=parameter(a.joint_range,w,a.joints,2,j);
  float q=a.qpos[w*a.nq+a.joint_qpos[j]],margin=parameter(a.joint_margin,w,a.joints,1,j)[0];
  float lower=q-range[0],upper=range[1]-q,position=fminf(lower,upper)-margin;
  if (!(position<0.0f)) return;
  atomicAdd(a.nl+w,1);int r=atomicAdd(a.nefc+w,1);
  if (r>=a.rows) return;
  float sign=float(lower<upper)*2.0f-1.0f;
  float* J=a.J+(w*a.padded_rows+r)*a.padded_nv;
  for (int i=0;i<a.nv;++i) J[i]=0.0f;
  J[d]=sign;
  publish_row(a,w,r,position,position,parameter(a.dof_weight,w,a.nv,1,d)[0],
    parameter(a.joint_ref,w,a.joints,2,j),parameter(a.joint_imp,w,a.joints,5,j),margin,sign*a.qvel[w*a.nv+d],0.0f,3,j);
}

__global__ void allocate_contacts(ConstraintBuildInputs a) {
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  if (c>=a.contacts || c>=a.contact_count[0] || !(a.contact_type[c]&1)) return;
  float position=a.contact_dist[c]-a.contact_margin[c];
  if (!(position<0.0f || (a.adhesion_enabled && a.contact_adhesion[c]!=0.0f))) return;
  int dim=a.contact_dim[c],count=a.cone==1 || dim==1 ? dim : 2*(dim-1),w=a.contact_world[c];
  int base=atomicAdd(a.nefc+w,count);
  for (int i=0;i<count;++i) {
    int r=base+i;
    a.contact_address[c*a.address_columns+i]=r<a.rows ? r : -1;
    if (r<a.rows) a.row_id[w*a.rows+r]=c;
  }
}

__device__ V linear_jacobian(ConstraintBuildInputs a,int w,int b,int d,V point) {
  if (!a.ancestor[b*a.padded_nv+d]) return {0.0f,0.0f,0.0f};
  const float* motion=a.cdof+6*(w*a.nv+d);
  V angular={motion[0],motion[1],motion[2]},linear={motion[3],motion[4],motion[5]};
  V offset=point-load3(a.subtree_com,w*a.bodies+a.root[b]);
  return linear+cross(angular,offset);
}
__device__ V angular_jacobian(ConstraintBuildInputs a,int w,int b,int d) {
  if (!a.ancestor[b*a.padded_nv+d]) return {0.0f,0.0f,0.0f};
  const float* motion=a.cdof+6*(w*a.nv+d);
  return {motion[0],motion[1],motion[2]};
}

// A CTA owns one contact constraint row. Lanes own DOFs, then reduce J*qvel.
// Shared storage is eight floats (one per possible warp), independent of NV/N.
__global__ void contact_jacobian(ConstraintBuildInputs a) {
  int w=blockIdx.x/a.rows,r=blockIdx.x%a.rows,lane=threadIdx.x;
  if (r<a.ne[w]+a.nf[w]+a.nl[w] || r>=a.nefc[w]) return;
  int c=a.row_id[w*a.rows+r],dim=a.contact_dim[c];
  int body1=a.geom_body[a.contact_geom[2*c]],body2=a.geom_body[a.contact_geom[2*c+1]];
  int component=r-a.contact_address[c*a.address_columns];
  V point=load3(a.contact_pos,c),normal=load3(a.contact_frame,3*c);
  float sum=0.0f;
  for (int d=lane;d<a.padded_nv;d+=blockDim.x) {
    float value=0.0f;
    if (d<a.nv) {
      V linear=linear_jacobian(a,w,body2,d,point)-linear_jacobian(a,w,body1,d,point);
      V angular=angular_jacobian(a,w,body2,d)-angular_jacobian(a,w,body1,d);
      if (a.cone==1) value=dot(component<3 ? linear : angular,load3(a.contact_frame,3*c+component%3));
      else {
        value=dot(linear,normal);
        if (dim>1) {
          int direction=component/2+1;
          float friction=a.contact_friction[5*c+direction-1]*(1.0f-2.0f*float(component&1));
          value+=dot(direction<3 ? linear : angular,load3(a.contact_frame,3*c+direction%3))*friction;
        }
      }
      sum+=value*a.qvel[w*a.nv+d];
    }
    a.J[(w*a.padded_rows+r)*a.padded_nv+d]=value;
  }
  for (int offset=16;offset>0;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
  __shared__ float warp_sum[8];
  if ((lane&31)==0) warp_sum[lane/32]=sum;
  __syncthreads();
  if (lane<32) {
    sum=lane<blockDim.x/32 ? warp_sum[lane] : 0.0f;
    for (int offset=16;offset>0;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if (lane==0) a.Jqvel[w*a.rows+r]=sum;
  }
}

__global__ void contact_reference_rows(ConstraintBuildInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.contacts*a.address_columns) return;
  int c=t/a.address_columns,component=t%a.address_columns;
  if (c>=a.contact_count[0] || !(a.contact_type[c]&1)) return;
  int dim=a.contact_dim[c],count=a.cone==1 || dim==1 ? dim : 2*(dim-1);
  if (component>=count) return;
  int r=a.contact_address[t];
  if (r<0) return;
  int w=a.contact_world[c],body1=a.geom_body[a.contact_geom[2*c]],body2=a.geom_body[a.contact_geom[2*c+1]];
  const float* weights=parameter(a.body_weight,w,a.bodies,2);
  float weight=weights[2*body1]+weights[2*body2];
  float ratio=parameter(a.impratio_invsqrt,w,1,1)[0],scale=1.0f;
  const float *friction=a.contact_friction+5*c,*ref=a.contact_ref+2*c,*imp=a.contact_imp+5*c;
  float position=a.contact_dist[c]-a.contact_margin[c],pos_aref=position,velocity=a.Jqvel[w*a.rows+r];
  if (a.cone==1 && component>0) {
    scale=ratio*ratio;
    if (component>1) scale*=friction[0]*friction[0]/(friction[component-1]*friction[component-1]);
    pos_aref=0.0f;
  } else if (a.cone==0 && dim>1) {
    weight=weight+friction[0]*friction[0]*weight;
    weight=weight*2.0f*friction[0]*friction[0]*ratio*ratio;
  }
  int type=dim==1 ? 5 : a.cone==1 ? 7 : 6;
  publish_row(a,w,r,pos_aref,position,weight,ref,imp,a.contact_margin[c],velocity,0.0f,type,c);
  int row=w*a.rows+r;
  if (a.cone==1 && component>0) {
    a.row_D[row]/=scale;
    const float* friction_ref=a.contact_ref_friction+2*c;
    if (friction_ref[0]!=0.0f || friction_ref[1]!=0.0f) ref=friction_ref;
    float damping=impedance(a.refsafe,parameter(a.timestep,w,1,1)[0],ref,imp,position).b;
    a.row_aref[row]=-damping*velocity;
  }
  if (a.adhesion_enabled && a.contact_adhesion[c]!=0.0f && (component==0 || a.cone==0) && a.row_D[row]>0.0f) {
    float adhesion=a.contact_adhesion[c];
    if (a.cone==0 && dim>1) adhesion/=float(2*(dim-1));
    a.row_aref[row]+=(1.0f/a.row_D[row])*adhesion;
  }
}
}
}

extern "C" {
const char* flash_mjwarp_constraint_build_source_id() {return FLASH_MJW_CONSTRAINT_BUILD_SOURCE_ID;}
std::size_t flash_mjwarp_constraint_build_arguments_bytes() {return sizeof(flash::mjwarp::ConstraintBuildInputs);}
int flash_mjwarp_constraint_build_arch() {return FLASH_MJW_CONSTRAINT_BUILD_ARCH;}
int flash_mjwarp_constraint_build_launch(const flash::mjwarp::ConstraintBuildInputs* a,int phase,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->nv<1 || a->rows<0 || a->contacts<0 || a->padded_nv<a->nv || a->padded_rows<a->rows) return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase==0) reset_counts<<<(a->worlds*(a->rows>0 ? a->rows : 1)+255)/256,256,0,stream>>>(*a);
  else if (phase==1) dof_friction_rows<<<(a->worlds*a->nv+255)/256,256,0,stream>>>(*a);
  else if (phase==2) {if (a->limited) scalar_limit_rows<<<(a->worlds*a->limited+255)/256,256,0,stream>>>(*a);}
  else if (phase==3) {if (a->contacts) allocate_contacts<<<(a->contacts+255)/256,256,0,stream>>>(*a);}
  else if (phase==4) {if (a->rows) contact_jacobian<<<a->worlds*a->rows,128,0,stream>>>(*a);}
  else if (phase==5) {if (a->contacts) contact_reference_rows<<<(a->contacts*a->address_columns+255)/256,256,0,stream>>>(*a);}
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
}
