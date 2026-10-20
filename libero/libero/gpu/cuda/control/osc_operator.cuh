#include "precision.cuh"
#pragma once
#include "osc.cuh"
#include "osc_tasks.cuh"
extern "C" __device__ void osc_solve7x6(real_t*,real_t*,int*,int*);
extern "C" __device__ void osc_mm_6_1_7(real_t*,real_t*,real_t*);
#ifndef OSC_DIRECT_RHS
#define OSC_DIRECT_RHS 1
#endif
#ifndef OSC_DIRECT_N
#define OSC_DIRECT_N 1
#endif
#ifndef OSC_CACHE_ENABLED
#define OSC_CACHE_ENABLED 1
#endif
#define OSC_OPERATOR_CACHE_WIDTH (OSC_DIRECT_N ? 96 : 145)
// Explicit operator_v1 ABI: legacy matrix diagnostics are held, never published.
// All live fields, gates, errors, every-substep gains and gripper cadence remain.
namespace osc_cuda {
__device__ inline void operator_control(int w,real_t*mass,real_t*jacobian,real_t*q,real_t*qvel,real_t*bias,real_t*q0,
 real_t*ee_pos,real_t*ee_rot,real_t*ee_velocity,real_t*action,real_t*goal_pos,real_t*goal_rot,real_t*gripper,
 int*active,real_t*gains,real_t*range,real_t*ctrl,real_t*raw_tau,int*error,int*ranks,int*policy,int*needs,
 real_t*mass_inv,real_t*A_full,real_t*A_pos,real_t*A_ori,real_t*lambda_full,real_t*lambda_pos,real_t*lambda_ori,
 real_t*nullspace_matrix,real_t*position_error,real_t*orientation_error,real_t*quat_input,float*quat_cast,
 real_t*quat_result,int*quat_valid,real_t*nullspace_torque,real_t*clipped,int diagnostics,int uncouple, int* derived_valid=nullptr, real_t* derived_key=nullptr, real_t* derived_data=nullptr, int* derived_ranks=nullptr, int* cache_hits=nullptr,int* cache_misses=nullptr,int* task_stats=nullptr,real_t* persist_m=nullptr,real_t* persist_j=nullptr,real_t* local_state=nullptr) {
    __shared__ real_t m[49],lu[49],mi[49],j[42],jt[42],jw[42],a[36],ap[36],ar[36],lf[36],lp[36],lr[36];
    __shared__ real_t mjt[42],jbar[42],project[49],delta[9],current[9],gr[9],gp[3],desired[6],wrench[6];
    __shared__ real_t lp3[9],lr3[9],pd[7],pose[7],nt[7],tau[7],project_t[49],xt[42],lft[36],z6[6],y6[6],correction[7];
    __shared__ int info,piv[7],admitted,reuse,serial_load,task_rank[3],task_ok[3];
    if(threadIdx.x==0){
        admitted=active[w]!=0&&error[w]==0;info=0;
        serial_load=admitted&&load_alias(w,mass,jacobian,bias,q0,persist_m,persist_j);
    }
    __syncthreads();
    if(!admitted)return;
    if(serial_load){
        if(threadIdx.x==0){
            for(int r=0;r<7;r++){if(!isfinite(bias[w*7+r])||!isfinite(q0[w*7+r]))admitted=0;
                for(int c=0;c<7;c++){real_t v=mass[w*49+r*7+c];m[r*7+c]=v;lu[r*7+c]=v;if(!OSC_DIRECT_RHS)mi[r*7+c]=r==c?1:0;if(persist_m)persist_m[w*49+r*7+c]=v;if(!isfinite(v))admitted=0;}}
            for(int r=0;r<6;r++)for(int c=0;c<7;c++){real_t v=jacobian[w*42+r*7+c];j[r*7+c]=v;jt[c*6+r]=v;if(persist_j)persist_j[w*42+r*7+c]=v;if(!isfinite(v))admitted=0;}
            if(!admitted)error[w]=1;
        }
    }else if(threadIdx.x<32){
        bool bad=false;
        const int width=blockDim.x<32?blockDim.x:32;
        const unsigned participants=width==32?0xffffffffu:((1u<<width)-1u);
        for(int pair=threadIdx.x;pair<25;pair+=width){
            const int z=pair*2;
            if(z+1<49){const real2_t v=libero_vec::load_pair(mass+w*49+z);
                libero_vec::store_pair(m+z,v.x,v.y);libero_vec::store_pair(lu+z,v.x,v.y);
                if(!OSC_DIRECT_RHS){if(!OSC_DIRECT_RHS)mi[z]=z/7==z%7?1:0;mi[z+1]=(z+1)/7==(z+1)%7?1:0;}
                if(persist_m)libero_vec::store_pair(persist_m+w*49+z,v.x,v.y);
                bad=bad||!isfinite(v.x)||!isfinite(v.y);
            }else{const real_t v=mass[w*49+z];m[z]=lu[z]=v;if(!OSC_DIRECT_RHS)mi[z]=z/7==z%7?1:0;
                if(persist_m)persist_m[w*49+z]=v;bad=bad||!isfinite(v);}
        }
        for(int pair=threadIdx.x;pair<21;pair+=width){
            const int z=pair*2;const real2_t v=libero_vec::load_pair(jacobian+w*42+z);
            libero_vec::store_pair(j+z,v.x,v.y);
            jt[(z%7)*6+z/7]=v.x;jt[((z+1)%7)*6+(z+1)/7]=v.y;
            if(persist_j)libero_vec::store_pair(persist_j+w*42+z,v.x,v.y);
            bad=bad||!isfinite(v.x)||!isfinite(v.y);
        }
        bad=libero_vec::any_nonfinite_lane(bias+w*7,7,threadIdx.x,width)||bad;
        bad=libero_vec::any_nonfinite_lane(q0+w*7,7,threadIdx.x,width)||bad;
        const bool any_bad=__any_sync(participants,bad);
        if(threadIdx.x==0&&any_bad){admitted=0;error[w]=1;}
    }
    __syncthreads();if(!admitted)return;
    if(threadIdx.x<32){
        bool mismatch=!(OSC_CACHE_ENABLED&&derived_valid&&derived_valid[w]);
        const int width=blockDim.x<32?blockDim.x:32;
        const unsigned participants=width==32?0xffffffffu:((1u<<width)-1u);
        if(!mismatch){
            for(int pair=threadIdx.x;pair<46;pair+=width){
                const int z=pair*2;
                const real2_t key=z+1<91?libero_vec::load_pair(derived_key+w*91+z):make_real2(derived_key[w*91+z],0);
                const real_t x=z<49?m[z]:j[z-49];
                mismatch=mismatch||(real_bits(x)!=real_bits(key.x));
                if(z+1<91){const real_t y=z+1<49?m[z+1]:j[z+1-49];
                    mismatch=mismatch||(real_bits(y)!=real_bits(key.y));}
            }
        }
        const bool any_mismatch=__any_sync(participants,mismatch);
        if(threadIdx.x==0)reuse=!any_mismatch;
    }
    __syncthreads();
    if(threadIdx.x==0&&reuse){
        real_t* d=derived_data+w*OSC_OPERATOR_CACHE_WIDTH;
        libero_vec::copy_serial(d,mjt,42);libero_vec::copy_serial(d+42,lf,36);
        for(int z=0;z<36;z++)lp[z]=lr[z]=0;
        for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;
            lp[t]=lp3[z]=d[78+z];lr[t]=lr3[z]=d[87+z];}
        if(!OSC_DIRECT_N)libero_vec::copy_serial(d+96,project,49);
        for(int z=0;z<3;z++)ranks[w*3+z]=derived_ranks[w*3+z];
        if(cache_hits&&cache_hits[w]<INT32_MAX)cache_hits[w]++;
    }__syncthreads();
    if(!reuse){
    if(threadIdx.x==0&&cache_misses&&cache_misses[w]<INT32_MAX)cache_misses[w]++;
    if(OSC_DIRECT_RHS){
        for(int z=threadIdx.x;z<42;z+=blockDim.x)mjt[z]=jt[z];
        __syncthreads();osc_solve7x6(lu,mjt,piv,&info);
    }else osc_inverse7(lu,mi,piv,&info);
    if(threadIdx.x==0&&info!=0){error[w]=2;admitted=0;}
    __syncthreads();if(!admitted)return;
    if(OSC_DIRECT_RHS)osc_mm_6_6_7(j,mjt,a);
    else{osc_mm_6_7_7(j,mi,jw);osc_mm_6_6_7(jw,jt,a);osc_mm_7_6_7(mi,jt,mjt);}
    if(threadIdx.x==0){
        for(int z=0;z<36;z++){ap[z]=0;ar[z]=0;}
        for(int r=0;r<3;r++)for(int c=0;c<3;c++){ap[r*6+c]=a[r*6+c];ar[r*6+c]=a[(r+3)*6+c+3];}
    }__syncthreads();
    operator_tasks(w,a,ap,ar,lf,lp,lr,task_rank,task_ok,task_stats);
    if(threadIdx.x==0){
        for(int z=0;z<3;z++)ranks[w*3+z]=task_rank[z];
        if(!task_ok[0]||!task_ok[1]||!task_ok[2]){error[w]=4;admitted=0;}
        else for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;lp3[z]=lp[t];lr3[z]=lr[t];}
    }__syncthreads();if(!admitted)return;
    if(!OSC_DIRECT_N){osc_mm_7_6_6(mjt,lf,jbar);osc_mm_7_7_6(jbar,j,project);}
    if(threadIdx.x==0){
        if(!OSC_DIRECT_N)for(int r=0;r<7;r++)for(int c=0;c<7;c++)project[r*7+c]=(r==c?REAL_C(1.):REAL_C(0.))-project[r*7+c];
        if(OSC_CACHE_ENABLED&&derived_valid){real_t* d=derived_data+w*OSC_OPERATOR_CACHE_WIDTH;
            for(int z=0;z<49;z++)derived_key[w*91+z]=m[z];
            for(int z=0;z<42;z++){derived_key[w*91+49+z]=j[z];d[z]=mjt[z];}
            for(int z=0;z<36;z++)d[42+z]=lf[z];
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;d[78+z]=lp[t];d[87+z]=lr[t];}
            if(!OSC_DIRECT_N)for(int z=0;z<49;z++)d[96+z]=project[z];
            for(int z=0;z<3;z++)derived_ranks[w*3+z]=ranks[w*3+z];derived_valid[w]=1;
        }
    }__syncthreads();
    } // exact-input derived-matrix cache miss
    for(int z=threadIdx.x;z<42;z+=blockDim.x)xt[(z%6)*7+z/6]=mjt[z];
    for(int z=threadIdx.x;z<36;z+=blockDim.x)lft[(z%6)*6+z/6]=lf[z];
    if(!OSC_DIRECT_N)for(int z=threadIdx.x;z<49;z+=blockDim.x)project_t[(z%7)*7+z/7]=project[z];
    __syncthreads();
    if(threadIdx.x==0){
        for(int r=0;r<3;r++){gp[r]=goal_pos[w*3+r];for(int c=0;c<3;c++){gr[r*3+c]=goal_rot[w*9+r*3+c];current[r*3+c]=(local_state?local_state[17+r*3+c]:ee_rot[w*9+r*3+c]);}}
        quat_valid[w]=0;
        if(policy[w]){
            for(int r=0;r<3;r++)gp[r]=(local_state?local_state[14+r]:ee_pos[w*3+r])+REAL_C(.05)*clip(action[w*7+r],-1,1);
            real_t x=REAL_C(.5)*clip(action[w*7+3],-1,1),y=REAL_C(.5)*clip(action[w*7+4],-1,1),z=REAL_C(.5)*clip(action[w*7+5],-1,1);
            if(x!=0||y!=0||z!=0){rotation_delta(x,y,z,delta,quat_input+w*4,quat_cast+w*4);quat_valid[w]=1;for(int i=0;i<9;i++)quat_result[w*9+i]=delta[i];}
        }
    }__syncthreads();if(quat_valid[w])osc_mm_3_3_3(delta,current,gr);
    if(threadIdx.x==0){
        real_t er[3]={};
        for(int c=0;c<3;c++){er[0]+=REAL_C(.5)*(current[3+c]*gr[6+c]-current[6+c]*gr[3+c]);
            er[1]+=REAL_C(.5)*(current[6+c]*gr[c]-current[c]*gr[6+c]);er[2]+=REAL_C(.5)*(current[c]*gr[3+c]-current[3+c]*gr[c]);}
        for(int r=0;r<3;r++){position_error[w*3+r]=gp[r]-(local_state?local_state[14+r]:ee_pos[w*3+r]);orientation_error[w*3+r]=er[r];
            desired[r]=gains[w*12+r]*position_error[w*3+r]-gains[w*12+r+6]*(local_state?local_state[26+r]:ee_velocity[w*6+r]);
            desired[r+3]=gains[w*12+r+3]*er[r]-gains[w*12+r+9]*(local_state?local_state[26+r+3]:ee_velocity[w*6+r+3]);}
        for(int c=0;c<7;c++)pd[c]=10*(q0[w*7+c]-(local_state?local_state[0+c]:q[w*7+c]))-REAL_C(6.324555320336759)*(local_state?local_state[7+c]:qvel[w*7+c]);
    }__syncthreads();
    const bool split_mode=OSC_UNCOUPLE<0 ? uncouple!=0 : OSC_UNCOUPLE!=0;
    if(split_mode){osc_mm_3_1_3(lp3,desired,wrench);osc_mm_3_1_3(lr3,desired+3,wrench+3);}else osc_mm_6_1_6(lf,desired,wrench);
    osc_mm_7_1_7(m,pd,pose);
    // N^T v = v - J^T lambda_full^T X^T v. Neither M nor lambda is assumed symmetric.
    if(OSC_DIRECT_N){osc_mm_6_1_7(xt,pose,z6);osc_mm_6_1_6(lft,z6,y6);osc_mm_7_1_6(jt,y6,correction);
        for(int z=threadIdx.x;z<7;z+=blockDim.x)nt[z]=pose[z]-correction[z];__syncthreads();
    }else osc_mm_7_1_7(project_t,pose,nt);
    osc_mm_7_1_6(jt,wrench,tau);
    if(threadIdx.x==0){
        for(int r=0;r<3;r++){goal_pos[w*3+r]=gp[r];for(int c=0;c<3;c++)goal_rot[w*9+r*3+c]=gr[r*3+c];}
        for(int i=0;i<7;i++){nullspace_torque[w*7+i]=nt[i];real_t t=tau[i]+bias[w*7+i]+nt[i];raw_tau[w*7+i]=t;clipped[w*7+i]=clip(t,range[w*18+2*i],range[w*18+2*i+1]);ctrl[w*9+i]=clipped[w*7+i];}
        real_t a6=action[w*7+6],sgn=isnan(a6)?a6:(a6>0?REAL_C(1.):(a6<0?-REAL_C(1.):REAL_C(0.)));
        for(int i=0;i<2;i++){real_t value=clip(gripper[w*2+i]+REAL_C(.01)*(i==0?-REAL_C(1.):REAL_C(1.))*sgn,-1,1);gripper[w*2+i]=value;
            real_t lo=range[w*18+2*(i+7)],hi=range[w*18+2*(i+7)+1];ctrl[w*9+i+7]=(hi+lo)*REAL_C(.5)+(hi-lo)*REAL_C(.5)*value;}
        needs[w]=1;
    }__syncthreads();
}
__device__ inline void operator_fast(int w,real_t*mass,real_t*jacobian,real_t*q,real_t*qvel,real_t*bias,real_t*q0,
 real_t*ee_pos,real_t*ee_rot,real_t*ee_velocity,real_t*action,real_t*goal_pos,real_t*goal_rot,real_t*gripper,
 int*active,real_t*gains,real_t*range,real_t*ctrl,real_t*raw_tau,int*error,int*ranks,int*policy,int*needs,
 real_t*mass_inv,real_t*A_full,real_t*A_pos,real_t*A_ori,real_t*lambda_full,real_t*lambda_pos,real_t*lambda_ori,
 real_t*nullspace_matrix,real_t*position_error,real_t*orientation_error,real_t*quat_input,float*quat_cast,
 real_t*quat_result,int*quat_valid,real_t*nullspace_torque,real_t*clipped,int diagnostics,int uncouple, int* derived_valid, real_t* derived_key, real_t* derived_data, int* derived_ranks, int* cache_hits,int* cache_misses,int* task_stats, real_t* pm,real_t* pj,real_t* pq,real_t* pv,real_t* pp,real_t* pr,real_t* pe,int* cache_valid) {
    __shared__ int live;
    __shared__ real_t local_state[32];
    if(threadIdx.x==0){live=needs[w]!=0;
        if(active[w]&&!error[w]){
            if(live){
                // M/J materialization is paired with their shared operand load
                // inside control. Each physical matrix element is read once.
                const real_t* sources[5]={pq+w*7,pv+w*7,pp+w*3,pr+w*9,pe+w*6};
                real_t* targets[5]={q+w*7,qvel+w*7,ee_pos+w*3,ee_rot+w*9,ee_velocity+w*6};
                const int sizes[5]={7,7,3,9,6};
                bool alias=false;
                for(int i=0;i<5;i++)for(int k=0;k<5;k++)alias=alias||input_overlap(sources[i],sizes[i],targets[k],sizes[k]);
                for(int i=0;i<5;i++)for(int k=i+1;k<5;k++)alias=alias||input_overlap(targets[i],sizes[i],targets[k],sizes[k]);
                if(alias){
                for(int z=0;z<7;z++){real_t a=pq[w*7+z],b=pv[w*7+z];local_state[z]=q[w*7+z]=a;local_state[7+z]=qvel[w*7+z]=b;}
                for(int z=0;z<3;z++){real_t v=pp[w*3+z];local_state[14+z]=ee_pos[w*3+z]=v;}
                for(int z=0;z<9;z++){real_t v=pr[w*9+z];local_state[17+z]=ee_rot[w*9+z]=v;}
                for(int z=0;z<6;z++){real_t v=pe[w*6+z];local_state[26+z]=ee_velocity[w*6+z]=v;}
                }else{
                    state_field(sources[0],targets[0],local_state,7);
                    state_field(sources[1],targets[1],local_state+7,7);
                    state_field(sources[2],targets[2],local_state+14,3);
                    state_field(sources[3],targets[3],local_state+17,9);
                    state_field(sources[4],targets[4],local_state+26,6);
                }
                cache_valid[w]=1;needs[w]=0;
            }else if(!cache_valid[w])error[w]=1;
            else{
                state_field(q+w*7,nullptr,local_state,7);
                state_field(qvel+w*7,nullptr,local_state+7,7);
                state_field(ee_pos+w*3,nullptr,local_state+14,3);
                state_field(ee_rot+w*9,nullptr,local_state+17,9);
                state_field(ee_velocity+w*6,nullptr,local_state+26,6);
            }
        }
    }
    __syncthreads();
    operator_control(w,(live?pm:mass),(live?pj:jacobian),(live?pq:q),(live?pv:qvel),bias,q0,(live?pp:ee_pos),(live?pr:ee_rot),(live?pe:ee_velocity),action,goal_pos,goal_rot,gripper,active,gains,range,ctrl,raw_tau,error,ranks,policy,needs,mass_inv,A_full,A_pos,A_ori,lambda_full,lambda_pos,lambda_ori,nullspace_matrix,position_error,orientation_error,quat_input,quat_cast,quat_result,quat_valid,nullspace_torque,clipped,diagnostics,uncouple,derived_valid,derived_key,derived_data,derived_ranks,cache_hits,cache_misses,task_stats,live?mass:nullptr,live?jacobian:nullptr,local_state);
}
}
