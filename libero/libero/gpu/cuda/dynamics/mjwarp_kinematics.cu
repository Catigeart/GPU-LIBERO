// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned smooth.py rigid kinematics; branch-private prefixes and unique outputs.
#include "mjwarp_kinematics.cuh"
#include <cmath>
#ifndef FLASH_MJW_KINEMATICS_SOURCE_ID
#error "Build through flash_mjwarp_kinematics_fp32"
#endif
namespace flash::mjwarp {
namespace {
struct V { float x,y,z; };
struct Q { float w,x,y,z; };
__device__ inline V operator+(V a,V b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__device__ inline V operator-(V a,V b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ inline V operator*(float a,V b) { return {a*b.x,a*b.y,a*b.z}; }
__device__ inline float dot(V a,V b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ inline V cross(V a,V b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
__device__ inline V load3(const float* p,int i) { p+=3*i;return {p[0],p[1],p[2]}; }
__device__ inline Q load4(const float* p,int i) { p+=4*i;return {p[0],p[1],p[2],p[3]}; }
__device__ inline void store3(float* p,int i,V v) { p+=3*i;p[0]=v.x;p[1]=v.y;p[2]=v.z; }
__device__ inline void store4(float* p,int i,Q q) { p+=4*i;p[0]=q.w;p[1]=q.x;p[2]=q.y;p[3]=q.z; }
__device__ inline Q normalize(Q q) {
  float length=sqrtf(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z);
  if (length>0.0f) {float r=1.0f/length;return {q.w*r,q.x*r,q.y*r,q.z*r};}
  // Pinned Warp storage convention, also when components are treated as wxyz.
  return {0.0f,0.0f,0.0f,1.0f};
}
__device__ inline Q mul(Q a,Q b) {
  return {a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
    a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
    a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
    a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w};
}
__device__ inline V rotate(V v,Q q) {
  V u={q.x,q.y,q.z};
  V r=2.0f*(dot(u,v)*u)+(q.w*q.w-dot(u,u))*v;
  return r+(2.0f*q.w)*cross(u,v);
}
__device__ inline Q axis_angle(V axis,float angle) {
  float s,c;sincosf(angle*0.5f,&s,&c);
  return {c,axis.x*s,axis.y*s,axis.z*s};
}
__device__ inline void matrix(float* p,int i,Q q) {
  p+=9*i;
  float a=q.w*q.w,b=q.w*q.x,c=q.w*q.y,d=q.w*q.z;
  float e=q.x*q.x,f=q.x*q.y,h=q.x*q.z,j=q.y*q.y,k=q.y*q.z,l=q.z*q.z;
  p[0]=a+e-j-l;p[1]=2.0f*(f-d);p[2]=2.0f*(h+c);
  p[3]=2.0f*(f+d);p[4]=a-e+j-l;p[5]=2.0f*(k-b);
  p[6]=2.0f*(h-c);p[7]=2.0f*(k+b);p[8]=a-e-j+l;
}

__global__ void body_poses(KinematicsInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.branches) return;
  int w=t/a.branches,branch=t%a.branches;
  // Complete root-to-leaf paths were validated cold. Read only world pose;
  // carry every parent pose privately instead of reading another CTA's output.
  V parent_pos=load3(a.xpos,w*a.bodies);
  Q parent_quat=load4(a.xquat,w*a.bodies);
  for (int at=a.branch_start[branch];at<a.branch_start[branch+1];++at) {
    int body=a.branch_body[at],first=a.jadr[body],count=a.jnum[body];
    bool publish=a.owner[body]==branch;
    V pos;Q quat;
    if (count==1 && a.kind[first]==0) {
      int q= w*a.nq+a.qadr[first];
      pos={a.qpos[q],a.qpos[q+1],a.qpos[q+2]};
      quat=normalize({a.qpos[q+3],a.qpos[q+4],a.qpos[q+5],a.qpos[q+6]});
      if (publish) {
        store3(a.anchor,w*a.joints+first,pos);
        store3(a.axis,w*a.joints+first,load3(a.jnt_axis,(w%a.jnt_axis_rows)*a.joints+first));
      }
    } else {
      int mocap=a.mocap[body];
      if (mocap>=0) {pos=load3(a.mocap_pos,w*a.mocaps+mocap);quat=load4(a.mocap_quat,w*a.mocaps+mocap);}
      else {pos=load3(a.body_pos,(w%a.body_pos_rows)*a.bodies+body);quat=load4(a.body_quat,(w%a.body_quat_rows)*a.bodies+body);}
      pos=rotate(pos,parent_quat)+parent_pos;quat=mul(parent_quat,quat);
      for (int j=first;j<first+count;++j) {
        int kind=a.kind[j],q=a.qadr[j];
        V local_pos=load3(a.jnt_pos,(w%a.jnt_pos_rows)*a.joints+j);
        V local_axis=load3(a.jnt_axis,(w%a.jnt_axis_rows)*a.joints+j);
        V anchor=rotate(local_pos,quat)+pos,axis=rotate(local_axis,quat);
        if (kind==1) {
          const float* x=a.qpos+w*a.nq+q;
          quat=mul(quat,normalize({x[0],x[1],x[2],x[3]}));
          pos=anchor-rotate(local_pos,quat);
        } else if (kind==2) {
          float displacement=a.qpos[w*a.nq+q]-a.qpos0[(w%a.qpos0_rows)*a.nq+q];
          pos=pos+displacement*axis;
        } else if (kind==3) {
          float angle=a.qpos[w*a.nq+q]-a.qpos0[(w%a.qpos0_rows)*a.nq+q];
          quat=mul(quat,axis_angle(local_axis,angle));pos=anchor-rotate(local_pos,quat);
        }
        if (publish) {store3(a.anchor,w*a.joints+j,anchor);store3(a.axis,w*a.joints+j,axis);}
      }
      quat=normalize(quat);
    }
    parent_pos=pos;parent_quat=quat;
    if (publish) {store3(a.xpos,w*a.bodies+body,pos);store4(a.xquat,w*a.bodies+body,quat);}
  }
}
__global__ void body_matrices(KinematicsInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t<a.worlds*a.bodies) matrix(a.xmat,t,load4(a.xquat,t));
}
__global__ void inertial_frames(KinematicsInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.bodies) return;
  int w=t/a.bodies,b=t%a.bodies;Q q=load4(a.xquat,t);
  store3(a.xipos,t,load3(a.xpos,t)+rotate(load3(a.ipos,(w%a.ipos_rows)*a.bodies+b),q));
  matrix(a.ximat,t,mul(q,load4(a.iquat,(w%a.iquat_rows)*a.bodies+b)));
}
__global__ void geom_frames(KinematicsInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.geoms) return;
  int w=t/a.geoms,g=t%a.geoms,b=a.geom_body[g];
  // Exactly retain make_data's static-geom ownership/hold rule.
  if (a.weld[b]==0 && a.mocap[a.root[b]]==-1) return;
  Q q=load4(a.xquat,w*a.bodies+b);
  store3(a.geom_xpos,t,load3(a.xpos,w*a.bodies+b)+rotate(load3(a.gpos,(w%a.gpos_rows)*a.geoms+g),q));
  matrix(a.geom_xmat,t,mul(q,load4(a.gquat,(w%a.gquat_rows)*a.geoms+g)));
}
__global__ void site_frames(KinematicsInputs a) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t>=a.worlds*a.sites) return;
  int w=t/a.sites,s=t%a.sites,b=a.site_body[s];Q q=load4(a.xquat,w*a.bodies+b);
  store3(a.site_xpos,t,load3(a.xpos,w*a.bodies+b)+rotate(load3(a.spos,(w%a.spos_rows)*a.sites+s),q));
  matrix(a.site_xmat,t,mul(q,load4(a.squat,(w%a.squat_rows)*a.sites+s)));
}
// Lane0 of each warp traverses a complete root-to-leaf path, publishing
// canonical poses. Its warp then cooperates on disjoint derived transforms.
// Disjoint canonical owners remove the world-wide publication barrier.
template<int Warps>
__global__ void branch_kinematics(KinematicsInputs a,AttachedTopology attached) {
  const int groups=(a.branches+Warps-1)/Warps;
  const int w=blockIdx.x/groups,branch=(blockIdx.x%groups)*Warps+threadIdx.x/32,lane=threadIdx.x&31;
  if (branch>=a.branches) return; // warp-uniform; there are no block barriers
  if (lane==0) {
    // Complete root-to-leaf paths were validated cold. Read only world pose;
    // carry every parent pose privately instead of reading another CTA's output.
    V parent_pos=load3(a.xpos,w*a.bodies);
    Q parent_quat=load4(a.xquat,w*a.bodies);
    for (int at=a.branch_start[branch];at<a.branch_start[branch+1];++at) {
      int body=a.branch_body[at],first=a.jadr[body],count=a.jnum[body];
      bool publish=a.owner[body]==branch;
      V pos;Q quat;
      if (count==1 && a.kind[first]==0) {
        int q= w*a.nq+a.qadr[first];
        pos={a.qpos[q],a.qpos[q+1],a.qpos[q+2]};
        quat=normalize({a.qpos[q+3],a.qpos[q+4],a.qpos[q+5],a.qpos[q+6]});
        if (publish) {
          store3(a.anchor,w*a.joints+first,pos);
          store3(a.axis,w*a.joints+first,load3(a.jnt_axis,(w%a.jnt_axis_rows)*a.joints+first));
        }
      } else {
        int mocap=a.mocap[body];
        if (mocap>=0) {pos=load3(a.mocap_pos,w*a.mocaps+mocap);quat=load4(a.mocap_quat,w*a.mocaps+mocap);}
        else {pos=load3(a.body_pos,(w%a.body_pos_rows)*a.bodies+body);quat=load4(a.body_quat,(w%a.body_quat_rows)*a.bodies+body);}
        pos=rotate(pos,parent_quat)+parent_pos;quat=mul(parent_quat,quat);
        for (int j=first;j<first+count;++j) {
          int kind=a.kind[j],q=a.qadr[j];
          V local_pos=load3(a.jnt_pos,(w%a.jnt_pos_rows)*a.joints+j);
          V local_axis=load3(a.jnt_axis,(w%a.jnt_axis_rows)*a.joints+j);
          V anchor=rotate(local_pos,quat)+pos,axis=rotate(local_axis,quat);
          if (kind==1) {
            const float* x=a.qpos+w*a.nq+q;
            quat=mul(quat,normalize({x[0],x[1],x[2],x[3]}));
            pos=anchor-rotate(local_pos,quat);
          } else if (kind==2) {
            float displacement=a.qpos[w*a.nq+q]-a.qpos0[(w%a.qpos0_rows)*a.nq+q];
            pos=pos+displacement*axis;
          } else if (kind==3) {
            float angle=a.qpos[w*a.nq+q]-a.qpos0[(w%a.qpos0_rows)*a.nq+q];
            quat=mul(quat,axis_angle(local_axis,angle));pos=anchor-rotate(local_pos,quat);
          }
          if (publish) {store3(a.anchor,w*a.joints+j,anchor);store3(a.axis,w*a.joints+j,axis);}
        }
        quat=normalize(quat);
      }
      parent_pos=pos;parent_quat=quat;
      if (publish) {store3(a.xpos,w*a.bodies+body,pos);store4(a.xquat,w*a.bodies+body,quat);}
    }
  }
  __syncwarp(); // every owned pose is ready, without waiting for other branches
  const int start=a.branch_start[branch];
  for (int at=start-(branch==0)+lane;at<a.branch_start[branch+1];at+=32) {
    const int b=at<start ? 0 : a.branch_body[at],t=w*a.bodies+b;
    if (b && a.owner[b]!=branch) continue;
    Q q=load4(a.xquat,t);
    matrix(a.xmat,t,q);
    store3(a.xipos,t,load3(a.xpos,t)+rotate(load3(a.ipos,(w%a.ipos_rows)*a.bodies+b),q));
    matrix(a.ximat,t,mul(q,load4(a.iquat,(w%a.iquat_rows)*a.bodies+b)));
  }
  for (int at=attached.geom_start[branch]+lane;at<attached.geom_start[branch+1];at+=32) {
    const int g=attached.geom_id[at],b=a.geom_body[g],t=w*a.geoms+g;
    if (a.weld[b]==0 && a.mocap[a.root[b]]==-1) continue;
    Q q=load4(a.xquat,w*a.bodies+b);
    store3(a.geom_xpos,t,load3(a.xpos,w*a.bodies+b)+rotate(load3(a.gpos,(w%a.gpos_rows)*a.geoms+g),q));
    matrix(a.geom_xmat,t,mul(q,load4(a.gquat,(w%a.gquat_rows)*a.geoms+g)));
  }
  for (int at=attached.site_start[branch]+lane;at<attached.site_start[branch+1];at+=32) {
    const int site=attached.site_id[at],b=a.site_body[site],t=w*a.sites+site;
    Q q=load4(a.xquat,w*a.bodies+b);
    store3(a.site_xpos,t,load3(a.xpos,w*a.bodies+b)+rotate(load3(a.spos,(w%a.spos_rows)*a.sites+site),q));
    matrix(a.site_xmat,t,mul(q,load4(a.squat,(w%a.squat_rows)*a.sites+site)));
  }
}

}
}
extern "C" {
const char* flash_mjwarp_kinematics_source_id() { return FLASH_MJW_KINEMATICS_SOURCE_ID; }
std::size_t flash_mjwarp_kinematics_arguments_bytes() { return sizeof(flash::mjwarp::KinematicsInputs); }
int flash_mjwarp_kinematics_arch() { return FLASH_MJW_KINEMATICS_ARCH; }
int flash_mjwarp_kinematics_launch(const flash::mjwarp::KinematicsInputs* a,int phase,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->bodies<2 || a->joints<1 || a->nq<1 || a->branches<1 || a->geoms<0 || a->sites<0 || a->mocaps<0)
    return int(cudaErrorInvalidValue);
  using namespace flash::mjwarp;
  if (phase==0) body_poses<<<(a->worlds*a->branches+255)/256,256,0,stream>>>(*a);
  else if (phase==1) body_matrices<<<(a->worlds*a->bodies+255)/256,256,0,stream>>>(*a);
  else if (phase==2) inertial_frames<<<(a->worlds*a->bodies+255)/256,256,0,stream>>>(*a);
  else if (phase==3) {if (!a->geoms) return 0;geom_frames<<<(a->worlds*a->geoms+255)/256,256,0,stream>>>(*a);}
  else if (phase==4) {if (!a->sites) return 0;site_frames<<<(a->worlds*a->sites+255)/256,256,0,stream>>>(*a);}
  else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
std::size_t flash_mjwarp_kinematics_attached_bytes() {return sizeof(flash::mjwarp::AttachedTopology);}
int flash_mjwarp_kinematics_fused(const flash::mjwarp::KinematicsInputs* a,
    const flash::mjwarp::AttachedTopology* attached,cudaStream_t stream) {
  if (!a || !attached || a->worlds<1 || a->branches<1) return cudaErrorInvalidValue;
  flash::mjwarp::branch_kinematics<4><<<a->worlds*((a->branches+3)/4),128,0,stream>>>(*a,*attached);
  return cudaGetLastError();
}

}
