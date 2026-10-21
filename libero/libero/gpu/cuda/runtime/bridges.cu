// Copyright 2026 Flash-LIBERO contributors
// Readable CUDA implementations of the maintained control/observable bridges.
// These are transport/sampling operations, not new physics stages.
#include "bridge_types.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace flash::bridge {

struct Observe {
  int worlds,ns,width,stride;
  const int *op,*offset,*size,*index_offset,*indices,*dep0,*dep1,*dep2,*presence,*enabled;
  const float* periods;
  const float* source;
  float *cache,*observed,*sidequat,*elapsed;
  int *sampled,*present;
  const int* mask;
  int* events;
};
struct Scene {
  int worlds,cameras,stride;
  const float* source;
  float* scene;
  const float* periods;
  const int* enabled;
  float* elapsed;
  int *sampled,*generation;
  const int* parents;
  int *parent_generation,*captured;
  const int* mask;
};

__global__ void points(int worlds,int sites,int site,const float* position,float* output) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;if (w>=worlds) return;
  for (int k=0;k<3;++k) output[3*w+k]=position[3*(w*sites+site)+k];
}
__global__ void gather(Gather a) {
  int w=blockIdx.x,x=threadIdx.x;
  if (x<49) { int at=a.mi[x];a.out_mass[49*w+x]=at>=0 ? a.mass[w*a.nmass+at] : 0.f; }
  if (x<42) {
    int axis=x/7,dof=a.vi[x%7];
    a.out_jac[42*w+x]=(axis<3 ? a.jp[(3*w+axis)*a.nv+dof] : a.jr[(3*w+axis-3)*a.nv+dof]);
  }
  if (x<7) {
    a.out_q[7*w+x]=a.qpos[w*a.nq+a.qi[x]];a.out_v[7*w+x]=a.qvel[w*a.nv+a.vi[x]];
    a.out_bias[7*w+x]=a.bias[w*a.nv+a.vi[x]];
  }
  if (x<3) a.out_pos[3*w+x]=a.position[3*(w*a.nsites+a.site)+x];
  if (x<9) a.out_rot[9*w+x]=a.rotation[9*(w*a.nsites+a.site)+x];
  if (x<6) {
    float value=0;
    for (int k=0;k<a.nv;++k) value+=(x<3 ? a.jp[(3*w+x)*a.nv+k] : a.jr[(3*w+x-3)*a.nv+k])*a.qvel[w*a.nv+k];
    a.out_vel[6*w+x]=value;
  }
}
__global__ void scatter(int worlds,int width,const int* ids,const float* source,float* target,int stride) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if (t>=worlds*width) return;
  target[(t/width)*stride+ids[t%width]]=source[t];
}
__global__ void masked(int worlds,int width,const int* mask,const float* source,float* target) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if (t<worlds*width && mask[t/width]) target[t]=source[t];
}
__global__ void bootstrap(int worlds,const int* mask,const float* pos,const float* rot,
    float* goal_pos,float* goal_rot,float* gripper,int* error,int* needs,int* valid) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;if (w>=worlds || !mask[w]) return;
  for (int k=0;k<3;++k) goal_pos[3*w+k]=pos[3*w+k];
  for (int k=0;k<9;++k) goal_rot[9*w+k]=rot[9*w+k];
  gripper[2*w]=gripper[2*w+1]=0;error[w]=0;needs[w]=valid[w]=1;
}
__global__ void count(int worlds,int kind,const int* mask,std::int64_t* integrations,
    std::int64_t* epoch,std::int64_t* resets,std::int64_t* attempts,std::int64_t* positions) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;if (w>=worlds) return;
  if (kind==0) { ++integrations[w];++epoch[w];++attempts[w];++positions[w]; }
  else if (kind==1) { ++attempts[w];++positions[w]; }
  else if (mask[w]) { ++epoch[w];resets[w]+=kind==3; }
}
__global__ void pack(int worlds,int width,int source_stride,int target_stride,int offset,
    const float* source,float* target) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if (t<worlds*width) target[(t/width)*target_stride+offset+t%width]=source[(t/width)*source_stride+t%width];
}

// Preserve the sampler's robosuite FP32 quaternion convention and its fixed
// FP64-epsilon cutoff. This is independent of physics quaternion integration.
__device__ void rotation32(const float* q,float* r) {
  float v[4]={q[3],q[0],q[1],q[2]},n=0;
  for (int i=0;i<4;++i) n+=v[i]*v[i];
  for (int i=0;i<9;++i) r[i]=i%4==0 ? 1.f : 0.f;
  if (n>=float(4.*2.220446049250313e-16)) {
    float scale=sqrtf(2.f/n);for (int i=0;i<4;++i) v[i]*=scale;
    r[0]=1-v[2]*v[2]-v[3]*v[3];r[1]=v[1]*v[2]-v[3]*v[0];r[2]=v[1]*v[3]+v[2]*v[0];
    r[3]=v[1]*v[2]+v[3]*v[0];r[4]=1-v[1]*v[1]-v[3]*v[3];r[5]=v[2]*v[3]-v[1]*v[0];
    r[6]=v[1]*v[3]-v[2]*v[0];r[7]=v[2]*v[3]+v[1]*v[0];r[8]=1-v[1]*v[1]-v[2]*v[2];
  }
}
__device__ void matrix_quaternion32(const float* m,float* out) {
  float k[16]={m[0]-m[4]-m[8],m[1]+m[3],m[2]+m[6],m[7]-m[5],
      m[1]+m[3],m[4]-m[0]-m[8],m[5]+m[7],m[2]-m[6],
      m[2]+m[6],m[5]+m[7],m[8]-m[0]-m[4],m[3]-m[1],
      m[7]-m[5],m[2]-m[6],m[3]-m[1],m[0]+m[4]+m[8]};
  for (int i=0;i<16;++i) k[i]/=3.f;
  int largest=0;for (int i=1;i<4;++i) if (k[5*i]>k[5*largest]) largest=i;
  float q[4]={};q[largest]=1;
  for (int it=0;it<24;++it) {
    float v[4]={},norm=0;
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) v[i]+=k[4*i+j]*q[j];
    for (int i=0;i<4;++i) norm+=v[i]*v[i];norm=sqrtf(norm);
    for (int i=0;i<4;++i) q[i]=v[i]/norm;
  }
  for (int i=0;i<4;++i) out[i]=q[3]<0 ? -q[i] : q[i];
}
__global__ void observe(Observe a,float dt,int force) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;if (w>=a.worlds || !a.mask[w]) return;
  int b=w*a.width,sb=w*a.ns;
  for (int s=0;s<a.ns;++s) {
    a.events[sb+s]=0;if (!a.enabled[s]) continue;
    float elapsed=a.elapsed[sb+s]+dt;a.elapsed[sb+s]=elapsed;
    bool take=(!a.sampled[sb+s] && a.periods[s]>=elapsed)||force;
    bool fallback=elapsed>=a.periods[s] && !a.sampled[sb+s];
    if (take || fallback) {
      int at=b+a.offset[s],d0=b+a.dep0[s],d1=b+a.dep1[s],d2=b+a.dep2[s];
      for (int j=0;j<a.size[s];++j) {
        float value=0;
        if (a.op[s]==0) value=a.source[w*a.stride+a.indices[a.index_offset[s]+j]];
        else if (a.op[s]==1 && a.present[sb+a.presence[3*s]]) value=cosf(a.cache[d0+j]);
        else if (a.op[s]==2 && a.present[sb+a.presence[3*s]]) value=sinf(a.cache[d0+j]);
        a.cache[at+j]=value;
      }
      if (a.op[s]==3) {
        if (!a.present[sb+a.presence[3*s+1]] || !a.present[sb+a.presence[3*s]]) {
          for (int row=0;row<4;++row) a.cache[at+5*row]=1;
        } else {
          float r[9];rotation32(a.cache+d1,r);
          for (int row=0;row<3;++row) {
            float t=0;
            for (int col=0;col<3;++col) {a.cache[at+4*row+col]=r[3*col+row];t-=r[3*col+row]*a.cache[d0+col];}
            a.cache[at+4*row+3]=t;
          }
          a.cache[at+15]=1;
        }
      } else if (a.op[s]==4) {
        if (a.present[sb+a.presence[3*s]] && a.present[sb+a.presence[3*s+1]] && a.present[sb+a.presence[3*s+2]]) {
          float ro[9],rr[9];rotation32(a.cache+d1,ro);
          for (int row=0;row<3;++row) {
            float t=a.cache[d2+4*row+3];
            for (int col=0;col<3;++col) {
              t+=a.cache[d2+4*row+col]*a.cache[d0+col];float value=0;
              for (int k=0;k<3;++k) value+=a.cache[d2+4*row+k]*ro[3*k+col];rr[3*row+col]=value;
            }
            a.cache[at+row]=t;
          }
          matrix_quaternion32(rr,a.sidequat+b+a.offset[s]);
        }
      } else if (a.op[s]==5 && a.present[sb+a.presence[3*s]]) {
        for (int j=0;j<4;++j) a.cache[at+j]=a.sidequat[d0+j];
      }
      for (int j=0;j<a.size[s];++j) a.observed[at+j]=a.cache[at+j];
      a.present[sb+s]=a.sampled[sb+s]=a.events[sb+s]=1;
    }
    if (elapsed>=a.periods[s]) {a.elapsed[sb+s]=fmodf(elapsed,a.periods[s]);a.sampled[sb+s]=0;}
  }
}
__global__ void reset_observation(int worlds,int ns,int width,const int* mask,
    float* elapsed,int* present,float* observed,float* cache,float* sidequat) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;if (w>=worlds || !mask[w]) return;
  // Observable.reset preserves sampled flags in the original implementation.
  for (int s=0;s<ns;++s) {elapsed[w*ns+s]=0;present[w*ns+s]=0;}
  for (int j=0;j<width;++j) observed[w*width+j]=cache[w*width+j]=sidequat[w*width+j]=0;
}
__global__ void capture_scene(Scene a,float dt,int force) {
  int w=blockIdx.x*blockDim.x+threadIdx.x;if (w>=a.worlds || !a.mask[w]) return;
  for (int c=0;c<a.cameras;++c) {
    int s=w*a.cameras+c;if (!a.enabled[c]) continue;
    float e=a.elapsed[s]+dt;
    if ((!a.sampled[s] && a.periods[c]>=e)||force||(e>=a.periods[c] && !a.sampled[s])) {
      int p=a.parents[c],parent=w*a.cameras+p;
      const float* source=p>=0 ? a.scene+parent*a.stride : a.source+w*a.stride;
      for (int j=0;j<a.stride;++j) a.scene[s*a.stride+j]=source[j];
      a.parent_generation[s]=p>=0 && a.captured[parent] ? a.generation[parent] : 0;
      ++a.generation[s];a.captured[s]=a.sampled[s]=1;
    }
    if (e>=a.periods[c]) {e=fmodf(e,a.periods[c]);a.sampled[s]=0;}
    a.elapsed[s]=e;
  }
}
__global__ void scene_transform(int worlds,int bodies,int count,int rotation,const int* mask,
    const int* bodyid,const float* local_pos,const float* local_axes,const float* body_pos,
    const float* body_mat,float* output_pos,float* output_axes) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if (t>=worlds*count || !mask[t/count]) return;
  int object=t%count,body=(t/count)*bodies+bodyid[object];
  for (int i=0;i<3;++i) {
    float position=body_pos[3*body+i];
    for (int j=0;j<3;++j) position+=body_mat[9*body+3*i+j]*local_pos[3*object+j];output_pos[3*t+i]=position;
    for (int j=0;j<(rotation ? 3 : 1);++j) {
      float value=0;
      for (int k=0;k<3;++k) value+=body_mat[9*body+3*i+k]*local_axes[object*(rotation ? 9 : 3)+k*(rotation ? 3 : 1)+j];
      output_axes[t*(rotation ? 9 : 3)+i*(rotation ? 3 : 1)+j]=value;
    }
  }
}
}

extern "C" {
const char* flash_bridge_source_id() { return FLASH_BRIDGE_SOURCE_ID; }
std::size_t flash_bridge_gather_bytes() { return sizeof(flash::bridge::Gather); }
std::size_t flash_bridge_observe_bytes() { return sizeof(flash::bridge::Observe); }
std::size_t flash_bridge_scene_bytes() { return sizeof(flash::bridge::Scene); }
int flash_bridge_points(int worlds,int sites,int site,const float* position,float* output,cudaStream_t stream) {
  flash::bridge::points<<<(worlds+127)/128,128,0,stream>>>(worlds,sites,site,position,output);return cudaGetLastError();
}
int flash_bridge_gather(const flash::bridge::Gather* a,cudaStream_t stream) {
  flash::bridge::gather<<<a->worlds,64,0,stream>>>(*a);return cudaGetLastError();
}
int flash_bridge_scatter(int worlds,int width,const int* ids,const float* source,float* target,int stride,cudaStream_t stream) {
  flash::bridge::scatter<<<(worlds*width+255)/256,256,0,stream>>>(worlds,width,ids,source,target,stride);return cudaGetLastError();
}
int flash_bridge_masked(int worlds,int width,const int* mask,const float* source,float* target,cudaStream_t stream) {
  if (width) flash::bridge::masked<<<(worlds*width+255)/256,256,0,stream>>>(worlds,width,mask,source,target);return cudaGetLastError();
}
int flash_bridge_bootstrap(int worlds,const int* mask,const float* pos,const float* rot,float* goal_pos,float* goal_rot,float* gripper,int* error,int* needs,int* valid,cudaStream_t stream) {
  flash::bridge::bootstrap<<<(worlds+127)/128,128,0,stream>>>(worlds,mask,pos,rot,goal_pos,goal_rot,gripper,error,needs,valid);return cudaGetLastError();
}
int flash_bridge_count(int worlds,int kind,const int* mask,std::int64_t* integration,std::int64_t* epoch,std::int64_t* resets,std::int64_t* attempts,std::int64_t* positions,cudaStream_t stream) {
  flash::bridge::count<<<(worlds+127)/128,128,0,stream>>>(worlds,kind,mask,integration,epoch,resets,attempts,positions);return cudaGetLastError();
}
int flash_bridge_pack(int worlds,int width,int source_stride,int target_stride,int offset,const float* source,float* target,cudaStream_t stream) {
  if (width) flash::bridge::pack<<<(worlds*width+255)/256,256,0,stream>>>(worlds,width,source_stride,target_stride,offset,source,target);return cudaGetLastError();
}
int flash_bridge_observe(const flash::bridge::Observe* a,float dt,int force,cudaStream_t stream) {
  flash::bridge::observe<<<(a->worlds+127)/128,128,0,stream>>>(*a,dt,force);return cudaGetLastError();
}
int flash_bridge_reset_observation(int worlds,int ns,int width,const int* mask,float* elapsed,int* present,float* observed,float* cache,float* sidequat,cudaStream_t stream) {
  flash::bridge::reset_observation<<<(worlds+127)/128,128,0,stream>>>(worlds,ns,width,mask,elapsed,present,observed,cache,sidequat);return cudaGetLastError();
}
int flash_bridge_capture_scene(const flash::bridge::Scene* a,float dt,int force,cudaStream_t stream) {
  flash::bridge::capture_scene<<<(a->worlds+127)/128,128,0,stream>>>(*a,dt,force);return cudaGetLastError();
}
int flash_bridge_scene_transform(int worlds,int bodies,int count,int rotation,const int* mask,const int* bodyid,const float* local_pos,const float* local_axes,const float* body_pos,const float* body_mat,float* output_pos,float* output_axes,cudaStream_t stream) {
  if (count) flash::bridge::scene_transform<<<(worlds*count+127)/128,128,0,stream>>>(worlds,bodies,count,rotation,mask,bodyid,local_pos,local_axes,body_pos,body_mat,output_pos,output_axes);return cudaGetLastError();
}
}
