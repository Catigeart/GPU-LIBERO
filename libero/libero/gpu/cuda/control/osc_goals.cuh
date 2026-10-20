#include "precision.cuh"
#pragma once
#include "orientation_math.cuh"
extern "C" __device__ void osc_mm_3_3_3(real_t*,real_t*,real_t*);
namespace osc_goal_cuda {
__device__ inline void ori_error(const real_t*goal,const real_t*current,real_t*error){
 for(int i=0;i<3;i++)error[i]=0;
 for(int c=0;c<3;c++){
  error[0]+=REAL_C(.5)*(current[3+c]*goal[6+c]-current[6+c]*goal[3+c]);
  error[1]+=REAL_C(.5)*(current[6+c]*goal[c]-current[c]*goal[6+c]);
  error[2]+=REAL_C(.5)*(current[c]*goal[3+c]-current[3+c]*goal[c]);
 }
}
// Cold config consists of d input-lo/hi/output-lo/hi then 3D position and
// orientation limits. A policy update writes goals; ordinary substeps retain.
__device__ inline void begin(int w,int*active,int*policy,int*error,real_t*action,
 real_t*position,real_t*rotation,real_t*config,real_t*goal_pos,real_t*goal_rot,
 real_t*delta_rot,int*rotate,int d,int adim,int mode,int relative,int poslimit){
 rotate[w]=0;if(active[w]!=1||error[w]||policy[w]!=1)return;
 real_t command[6]={};int offset=6*mode;
 for(int i=0;i<d;i++){
  real_t x=action[w*adim+offset+i];
  if(relative){real_t lo=config[i],hi=config[d+i],ol=config[2*d+i],oh=config[3*d+i];
   command[i]=(fmin(fmax(x,lo),hi)-REAL_C(.5)*(hi+lo))*fabs(oh-ol)/fabs(hi-lo)+REAL_C(.5)*(oh+ol);}
  else command[i]=x;
 }
 for(int i=0;i<3;i++){
  real_t x=relative?position[w*3+i]+command[i]:command[i];
  if(poslimit)x=fmin(fmax(x,config[4*d+i]),config[4*d+3+i]);goal_pos[w*3+i]=x;
 }
 if(d==3){const real_t down[9]={0,1,0,1,0,0,0,0,-1};for(int i=0;i<9;i++)goal_rot[w*9+i]=down[i];rotate[w]=2;}
 else if(!relative){osc_geometry_cuda::axis_matrix(command+3,goal_rot+w*9);rotate[w]=2;}
 else if(command[3]!=0||command[4]!=0||command[5]!=0){osc_geometry_cuda::axis_matrix(command+3,delta_rot+w*9);rotate[w]=1;}
}
__device__ inline void product(int w,int*active,int*error,int*rotate,real_t*delta,real_t*current,real_t*goal){
 __shared__ int allowed;__shared__ real_t a[9],b[9],c[9];
 if(threadIdx.x==0)allowed=active[w]==1&&!error[w]&&rotate[w]==1;
 __syncthreads();if(!allowed)return;
 for(int i=threadIdx.x;i<9;i+=blockDim.x){a[i]=delta[w*9+i];b[i]=current[w*9+i];c[i]=0;}
 __syncthreads();osc_mm_3_3_3(a,b,c);
 for(int i=threadIdx.x;i<9;i+=blockDim.x)goal[w*9+i]=c[i];
}
__device__ inline void finish(int w,int*active,int*policy,int*error,int*rotate,
 real_t*config,real_t*goal_pos,real_t*goal_rot,real_t*current_rot,
 real_t*previous_pos,real_t*start_pos,real_t*start_ori,real_t*target_ori,
 real_t*step,real_t*effective_pos,real_t*orientation_error,int d,
 int ori_limit,int interp_pos,int interp_ori,int total){
 if(active[w]!=1||error[w])return;
 if(policy[w]==1){
  if(ori_limit&&rotate[w]!=0)osc_geometry_cuda::limit_matrix(goal_rot+w*9,config+4*d+6);
  if(interp_pos){for(int i=0;i<3;i++){start_pos[w*3+i]=previous_pos[w*3+i];previous_pos[w*3+i]=goal_pos[w*3+i];}step[w*2]=0;}
  if(interp_ori){for(int i=0;i<3;i++)start_ori[w*3+i]=target_ori[w*3+i];
   ori_error(goal_rot+w*9,current_rot+w*9,target_ori+w*3);step[w*2+1]=0;}
 }
 for(int i=0;i<3;i++)effective_pos[w*3+i]=interp_pos?start_pos[w*3+i]+(goal_pos[w*3+i]-start_pos[w*3+i])/(total-step[w*2]):goal_pos[w*3+i];
 if(interp_ori)osc_geometry_cuda::interpolate_euler(start_ori+w*3,target_ori+w*3,(step[w*2+1]+1)/total,orientation_error+w*3);
 else ori_error(goal_rot+w*9,current_rot+w*9,orientation_error+w*3);
 if(interp_pos&&step[w*2]<total-1)step[w*2]++;
 if(interp_ori&&step[w*2+1]<total-1)step[w*2+1]++;
 for(int i=0;i<3;i++)if(!isfinite(effective_pos[w*3+i])||!isfinite(orientation_error[w*3+i])){error[w]=2;return;}
}
__device__ inline void reset_history(int w,int*mask,real_t*previous_pos,real_t*start_pos,
 real_t*start_ori,real_t*target_ori,real_t*step,real_t*base_gains,real_t*gains){
 if(mask[w]!=1)return;for(int i=0;i<3;i++)previous_pos[w*3+i]=start_pos[w*3+i]=start_ori[w*3+i]=target_ori[w*3+i]=0;
 step[w*2]=step[w*2+1]=0;for(int i=0;i<12;i++)gains[w*12+i]=base_gains[i];
}
}
