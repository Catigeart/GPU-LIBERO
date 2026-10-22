// Copyright 2025 The Newton Developers
// Copyright 2026 Flash-LIBERO contributors
// SPDX-License-Identifier: Apache-2.0
// FP64 adaptation of cuda/fp32/fused_motion_passive.cu. One world block owns
// motion, complete bias RNE and joint/fluid passive production. Publication
// remains in the residual kernel at the native FP64 admission points.
#include "fp64_motion_passive.cuh"
#include <cmath>

namespace flash_libero {
namespace {
struct V { double x, y, z; };
__device__ V load3(const double *p) { return {p[0], p[1], p[2]}; }
__device__ V operator-(V a, V b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__device__ V cross(V a, V b) {
  return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
__device__ V rotate(const double *r, V v) {
  return {r[0]*v.x+r[1]*v.y+r[2]*v.z, r[3]*v.x+r[4]*v.y+r[5]*v.z,
          r[6]*v.x+r[7]*v.y+r[8]*v.z};
}
__device__ V rotate_transpose(const double *r, V v) {
  return {r[0]*v.x+r[3]*v.y+r[6]*v.z, r[1]*v.x+r[4]*v.y+r[7]*v.z,
          r[2]*v.x+r[5]*v.y+r[8]*v.z};
}
__device__ bool nonfinite(const double *p, int n) {
  bool bad=false;
  for(int i=threadIdx.x;i<n;i+=blockDim.x) bad |= !isfinite(p[i]);
  return bad;
}
} // namespace

__global__ __launch_bounds__(512) void fp64_motion_passive_kernel(const SubstepArguments *arguments,
                                          const int *plan, int *ready, SubstepEntry entry) {
  const auto &a=*arguments;
  const int w=blockIdx.y, lane=threadIdx.x, warp=lane/32, component=lane&31;
  if(lane==0) ready[w]=0;
  if(entry!=SubstepEntry::Start && a.continuation[w]!=int(entry)) return;
  const auto &v=a.body_velocity;
  const auto &b=a.bias_forces;
  const auto &p=a.joint_passive;
  const auto &f=a.fluid_passive;
  const int nb=v.nb, nv=v.nv, branches=plan[0];
  // Unsupported/exceptional inputs execute the retained native producer in
  // the residual. No gate, public valid flag or error is changed here.
  const auto &gate=a.velocity_gate;
  const bool admitted=gate.active[w]==1 && !gate.error[w] && gate.state[w]==gate.source[w] &&
      (gate.require!=1 || (gate.valid[w]==1 && gate.written[w]==gate.epoch[w]));
  if(a.state.requested_mask[w]==0 || !admitted ||
     a.state.first_error[w] || f.diagnostic || nb>64 || !p.joint_disjoint) return;
  const auto &fw=a.forward_workspace;
  const auto *cache=fw.metadata+size_t(w)*ForwardCacheWords;
  const bool reuse=entry!=SubstepEntry::Start && a.state.requested_mask[w]==1 &&
      a.begin.active[w]==1 && !a.position.work[w] && cache[ForwardCacheValid]==1 &&
      cache[ForwardCacheState]==uint64_t(a.state.epoch[w]) &&
      cache[ForwardCachePosition]==fw.position_generation[w];
  if(reuse) return;
  const int *starts=plan+1, *paths=starts+branches+1;
  const int *owner=paths+starts[branches], *interval=owner+nb;
  const double *qvel=v.qvel+size_t(w)*nv, *cdof=v.cdof+size_t(w)*nv*6;
  const double *inertia=v.cinert+size_t(w)*nb*10;
  const double *qpos=p.qpos+size_t(w)*p.nq;
  const double *com=f.subtree_com+size_t(w)*nb*3, *pos=f.xipos+size_t(w)*nb*3;
  const double *mat=f.ximat+size_t(w)*nb*9;
  const bool fluid=!f.passive_disabled && (f.opt[0]>0.0 || f.opt[1]>0.0);
  bool bad=nonfinite(qvel,nv)|nonfinite(cdof,nv*6)|nonfinite(inertia,nb*10)|nonfinite(qpos,p.nq);
  if(fluid) bad |= nonfinite(com,nb*3)|nonfinite(pos,nb*3)|nonfinite(mat,nb*9);
  if(__syncthreads_or(bad)) return;

  constexpr int Slots=6*64;
  __shared__ double scratch[2*Slots];
  const bool packed=18*nb<=2*Slots;
  double *raw=scratch, *velocity=scratch+(packed?6*nb:Slots);
  double *fluid_force=packed?scratch+12*nb:fw.fluid+size_t(w)*(9*nb+nv);
  double *cvel=v.cvel+size_t(w)*nb*6, *dot=v.cdot+size_t(w)*nv*6;
  double *cacc=b.cacc+size_t(w)*nb*6, *frc=b.frc+size_t(w)*nb*6;
  double *passive=p.out+size_t(w)*nv, *bias=b.result+size_t(w)*nv;
  double *cache_acc=v.cache_acc+size_t(w)*nb*6, *cache_force=v.cache_force+size_t(w)*nb*6;
  if(lane<6) {
    cvel[lane]=velocity[lane]=raw[lane]=0.0;
    fluid_force[lane]=0.0;
    cacc[lane]=cache_acc[lane*nb]=lane>=3&&!v.disabled?-v.gravity[lane-3]:0.0;
    cache_force[lane*nb]=0.0;
  }
  // One joint owns its disjoint DOF interval. Native inputs already encode
  // disable policies. The retained native implementation has translational
  // free-joint springs and scalar hinge/slide springs; no modern polynomial
  // or adhesion extension is introduced by this numerical port.
  for(int j=lane;j<p.njnt;j+=blockDim.x) {
    const int d=p.dadr[j], q=p.qadr[j], kind=p.jtype[j];
    const int width=kind==0?6:kind==1?3:1;
    for(int k=0;k<width;++k) {
      double value=0.0;
      if((kind==0&&k<3)||kind>=2) value-=p.stiffness[j]*(qpos[q+k]-p.spring[q+k]);
      value-=p.damping[d+k]*qvel[d+k];
      passive[d+k]=value;
    }
  }
  // Match FP32's warp-per-branch organization. Only its first eight lanes
  // carry spatial components; shared prefixes have one canonical writer.
  for(int branch=warp;branch<branches;branch+=blockDim.x/32) {
    if(component<8) {
      double vel=0.0, acc=component>=3&&component<6&&!v.disabled?-v.gravity[component-3]:0.0;
      for(int at=starts[branch];at<starts[branch+1];++at) {
        const int body=paths[at];
        const bool publish=owner[body]==branch;
        double acc_delta=0.0;
        int j=0;
        while(j<v.bn[body]) {
          int dof=v.ba[body]+j;
          const int kind=v.jt[v.dj[dof]];
          if(kind==0) {
            double delta=0.0;
            for(int k=0;k<3;++k) if(component<6) {
              delta+=cdof[6*(dof+k)+component]*qvel[dof+k];
              if(publish) dot[6*(dof+k)+component]=0.0;
            }
            vel+=delta; dof+=3; j+=3;
          }
          const int width=kind<=1?3:1, axis=component%3, i=(axis+1)%3, k=(axis+2)%3;
          const double vi=__shfl_sync(0xffu,vel,i,8), vk=__shfl_sync(0xffu,vel,k,8);
          const double li=__shfl_sync(0xffu,vel,i+3,8), lk=__shfl_sync(0xffu,vel,k+3,8);
          double delta=0.0;
          for(int t=0;t<width;++t) if(component<6) {
            const double *motion=cdof+6*(dof+t);
            double derivative=vi*motion[k]-vk*motion[i];
            if(component>=3) derivative=(li*motion[k]-lk*motion[i])+(vi*motion[k+3]-vk*motion[i+3]);
            if(publish) dot[6*(dof+t)+component]=derivative;
            delta+=motion[component]*qvel[dof+t];
            acc_delta+=derivative*qvel[dof+t];
          }
          vel+=delta; j+=width;
        }
        acc+=acc_delta;
        if(publish&&component<6) {
          velocity[6*body+component]=cvel[6*body+component]=vel;
          cacc[6*body+component]=cache_acc[component*nb+body]=acc;
        }
      }
    }
  }
  __syncthreads(); // all canonical body velocities/accelerations are current

  for(int body=1+lane;body<nb;body+=blockDim.x) {
    const double *in=inertia+10*body, *acc=cacc+6*body, *vel=velocity+6*body;
    double momentum[6], force[6];
    // Compressed spatial inertia times acceleration and velocity.
    for(int pass=0;pass<2;++pass) {
      const double *x=pass?vel:acc;
      double *y=pass?momentum:force;
      y[0]=in[0]*x[0]+in[3]*x[1]+in[4]*x[2]-in[8]*x[4]+in[7]*x[5];
      y[1]=in[3]*x[0]+in[1]*x[1]+in[5]*x[2]+in[8]*x[3]-in[6]*x[5];
      y[2]=in[4]*x[0]+in[5]*x[1]+in[2]*x[2]-in[7]*x[3]+in[6]*x[4];
      y[3]=in[8]*x[1]-in[7]*x[2]+in[9]*x[3];
      y[4]=in[6]*x[2]-in[8]*x[0]+in[9]*x[4];
      y[5]=in[7]*x[0]-in[6]*x[1]+in[9]*x[5];
    }
    const V angular=load3(vel), linear=load3(vel+3);
    const V ma=load3(momentum), ml=load3(momentum+3);
    const V angular_cross=cross(angular,ma), linear_cross=cross(linear,ml), translation_cross=cross(angular,ml);
    const double correction[6]={angular_cross.x+linear_cross.x,angular_cross.y+linear_cross.y,
        angular_cross.z+linear_cross.z,translation_cross.x,translation_cross.y,translation_cross.z};
    for(int k=0;k<6;++k) raw[6*body+k]=cache_force[k*nb+body]=force[k]+correction[k];

    double *out=fluid_force+6*body;
    for(int k=0;k<6;++k) out[k]=0.0;
    if(!fluid || f.mass[body]<1.e-15) continue;
    const double *rotation=mat+9*body;
    const V offset=load3(pos+3*body)-load3(com+3*f.rootid[body]);
    const V object_linear=rotate_transpose(rotation,linear-cross(offset,angular));
    const V wind=rotate_transpose(rotation,load3(f.wind));
    const V local_linear=object_linear-wind, local_angular=rotate_transpose(rotation,angular);
    const double l[3]={local_linear.x,local_linear.y,local_linear.z};
    const double r[3]={local_angular.x,local_angular.y,local_angular.z};
    const double *box=f.box+3*body;
    double torque[3]={0.0,0.0,0.0}, drag[3]={0.0,0.0,0.0};
    constexpr double pi=3.14159265358979323846;
    if(f.opt[0]>0.0) {
      const double diameter=(box[0]+box[1]+box[2])/3.0;
      const double angular_scale=-pi*diameter*diameter*diameter*f.opt[0], linear_scale=-3.0*pi*diameter*f.opt[0];
      for(int k=0;k<3;++k) { torque[k]=r[k]*angular_scale; drag[k]=l[k]*linear_scale; }
    }
    if(f.opt[1]>0.0) for(int k=0;k<3;++k) {
      int i=(k+1)%3,j=(k+2)%3;
      if(k==1) { i=0;j=2; }
      const double x=box[i],y=box[j];
      drag[k]-=0.5*f.opt[1]*x*y*fabs(l[k])*l[k];
      torque[k]-=f.opt[1]*box[k]*(x*x*x*x+y*y*y*y)*fabs(r[k])*r[k]/64.0;
    }
    const V world_force=rotate(rotation,{drag[0],drag[1],drag[2]});
    const V world_torque=rotate(rotation,{torque[0],torque[1],torque[2]});
    out[0]=world_force.x;out[1]=world_force.y;out[2]=world_force.z;
    out[3]=world_torque.x;out[4]=world_torque.y;out[5]=world_torque.z;
  }
  __syncthreads(); // velocity arena is dead; raw body and fluid forces are live
  double *composite=velocity;
  for(int work=lane;work<6*nb;work+=blockDim.x) {
    const int body=work/6,k=work%6;
    double value=body?raw[work]:0.0;
    if(body) for(int node=body+1;node<body+interval[body];++node) value+=raw[6*node+k];
    composite[work]=frc[work]=value;
  }
  __syncthreads(); // all composite forces are ready for DOF projection
  bad=false;
  for(int dof=lane;dof<nv;dof+=blockDim.x) {
    const double *motion=cdof+6*dof, *force=composite+6*b.db[dof];
    double value=0.0;
    for(int k=0;k<6;++k) value+=motion[k]*force[k];
    bias[dof]=value;
    bad |= !isfinite(value);
    // CPU2.3.7 applies body fluid forces in ascending order, force then torque.
    // Only the reduction organization changes; retain that accumulation here.
    double total=passive[dof];
    if(fluid) for(int body=1;body<nb;++body) {
      if(f.mass[body]<1.e-15 || !f.ancestor[body*nv+dof]) continue;
      const double *wrench=fluid_force+6*body;
      const V offset=load3(pos+3*body)-load3(com+3*f.rootid[body]);
      const V shift=cross(load3(motion),offset);
      const double jp[3]={motion[3]+shift.x,motion[4]+shift.y,motion[5]+shift.z};
      double qforce=0.0,qtorque=0.0;
      for(int k=0;k<3;++k) if(wrench[k]!=0.0) qforce+=jp[k]*wrench[k];
      total+=qforce;
      for(int k=0;k<3;++k) if(wrench[3+k]!=0.0) qtorque+=motion[k]*wrench[3+k];
      total+=qtorque;
    }
    passive[dof]=total;
    bad |= !isfinite(total);
  }
  bad |= nonfinite(cvel,nb*6)|nonfinite(dot,nv*6)|nonfinite(cacc,nb*6)|nonfinite(frc,nb*6);
  if(fluid) bad |= nonfinite(fluid_force,nb*6);
  // A failed candidate is not admitted. The native producer overwrites its
  // working values and preserves ordered exceptional writes/error handling.
  if(__syncthreads_or(bad)) return;
  if(lane==0) {
    v.status[w]=p.status[w]=f.status[w]=b.status[w]=0;
    v.cache_status[w]=0;v.cache_valid[w]=1;
    v.cache_state[w]=v.state_epoch[w];v.cache_forward[w]=v.forward_epoch[w];
    ready[w]=1;
  }
}
} // namespace flash_libero
