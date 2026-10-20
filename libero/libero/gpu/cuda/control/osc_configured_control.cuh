#include "precision.cuh"
#pragma once
#include "osc.cuh"
namespace osc_goal_cuda {
using namespace osc_cuda;
__device__ inline void control(int w,real_t*mass,real_t*jacobian,real_t*q,real_t*qvel,real_t*bias,real_t*q0,
 real_t*ee_pos,real_t*ee_rot,real_t*ee_velocity,real_t*action,real_t*goal_pos,real_t*goal_rot,real_t*gripper,
 int*active,real_t*gains,real_t*range,real_t*ctrl,real_t*raw_tau,int*error,int*ranks,int*policy,int*needs,
 real_t*mass_inv,real_t*A_full,real_t*A_pos,real_t*A_ori,real_t*lambda_full,real_t*lambda_pos,real_t*lambda_ori,
 real_t*nullspace_matrix,real_t*position_error,real_t*orientation_error,real_t*quat_input,float*quat_cast,
 real_t*quat_result,int*quat_valid,real_t*nullspace_torque,real_t*clipped,int diagnostics,int uncouple,real_t*ori_override, int* derived_valid=nullptr, real_t* derived_key=nullptr, real_t* derived_data=nullptr, int* derived_ranks=nullptr, int* cache_hits=nullptr,real_t* persist_m=nullptr,real_t* persist_j=nullptr,real_t* local_state=nullptr) {
    __shared__ real_t m[49],lu[49],mi[49],j[42],jt[42],jw[42],a[36],ap[36],ar[36],lf[36],lp[36],lr[36];
    __shared__ real_t mjt[42],jbar[42],project[49],delta[9],current[9],gr[9],gp[3],desired[6],wrench[6];
    __shared__ real_t lp3[9],lr3[9],pd[7],pose[7],nt[7],tau[7],project_t[49];
    __shared__ int info,piv[7],admitted,reuse,serial_load;
    if(threadIdx.x==0){
        admitted=active[w]!=0&&error[w]==0;info=0;
        serial_load=admitted&&load_alias(w,mass,jacobian,bias,q0,persist_m,persist_j);
    }
    __syncthreads();
    if(!admitted)return;
    if(serial_load){
        if(threadIdx.x==0){
            for(int r=0;r<7;r++){if(!isfinite(bias[w*7+r])||!isfinite(q0[w*7+r]))admitted=0;
                for(int c=0;c<7;c++){real_t v=mass[w*49+r*7+c];m[r*7+c]=v;lu[r*7+c]=v;mi[r*7+c]=r==c?1:0;if(persist_m)persist_m[w*49+r*7+c]=v;if(!isfinite(v))admitted=0;}}
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
                mi[z]=z/7==z%7?1:0;mi[z+1]=(z+1)/7==(z+1)%7?1:0;
                if(persist_m)libero_vec::store_pair(persist_m+w*49+z,v.x,v.y);
                bad=bad||!isfinite(v.x)||!isfinite(v.y);
            }else{const real_t v=mass[w*49+z];m[z]=lu[z]=v;mi[z]=z/7==z%7?1:0;
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
        bool mismatch=!(derived_valid&&derived_valid[w]);
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
    if(threadIdx.x==0){
        if(reuse){real_t* d=derived_data+w*206;
            libero_vec::copy_serial(d,mi,49);libero_vec::copy_serial(d+157,project,49);
            libero_vec::copy_serial(d+49,a,36);libero_vec::copy_serial(d+103,lf,36);
            for(int z=0;z<36;z++)ap[z]=ar[z]=lp[z]=lr[z]=0;
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;
                ap[t]=d[85+z];ar[t]=d[94+z];lp[t]=d[139+z];lr[t]=d[148+z];lp3[z]=lp[t];lr3[z]=lr[t];}
            for(int z=0;z<3;z++)ranks[w*3+z]=derived_ranks[w*3+z];if(cache_hits)cache_hits[w]++;
        }
    }__syncthreads();
    if(!reuse){
    osc_inverse7(lu,mi,piv,&info);
    if(threadIdx.x==0&&info!=0){error[w]=2;admitted=0;}
    __syncthreads();if(!admitted)return;
    for(int z=threadIdx.x;z<49;z+=blockDim.x)mass_inv[w*49+z]=mi[z];
    osc_mm_6_7_7(j,mi,jw);osc_mm_6_6_7(jw,jt,a);
    if(threadIdx.x==0){
        for(int z=0;z<36;z++){ap[z]=0;ar[z]=0;}
        for(int r=0;r<3;r++)for(int c=0;c<3;c++){ap[r*6+c]=a[r*6+c];ar[r*6+c]=a[(r+3)*6+c+3];}
        int rf,rp,rr,of,op,orr;pinv(a,6,lf,rf,of);pinv(ap,3,lp,rp,op);pinv(ar,3,lr,rr,orr);
        ranks[w*3]=rf;ranks[w*3+1]=rp;ranks[w*3+2]=rr;
        if(!of||!op||!orr){error[w]=4;admitted=0;}
        else{
            for(int z=0;z<36;z++){A_full[w*36+z]=a[z];lambda_full[w*36+z]=lf[z];}
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,s=r*6+c;
                A_pos[w*9+z]=ap[s];A_ori[w*9+z]=ar[s];lambda_pos[w*9+z]=lp[s];lambda_ori[w*9+z]=lr[s];lp3[z]=lp[s];lr3[z]=lr[s];}
        }
    }__syncthreads();if(!admitted)return;
    osc_mm_7_6_7(mi,jt,mjt);osc_mm_7_6_6(mjt,lf,jbar);osc_mm_7_7_6(jbar,j,project);
    if(threadIdx.x==0){
        for(int r=0;r<7;r++)for(int c=0;c<7;c++)project[r*7+c]=(r==c?REAL_C(1.):REAL_C(0.))-project[r*7+c];
        if(derived_valid){real_t* d=derived_data+w*206;
            for(int z=0;z<49;z++){derived_key[w*91+z]=m[z];d[z]=mi[z];d[157+z]=project[z];}
            for(int z=0;z<42;z++)derived_key[w*91+49+z]=j[z];
            for(int z=0;z<36;z++){d[49+z]=a[z];d[103+z]=lf[z];}
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;d[85+z]=ap[t];d[94+z]=ar[t];d[139+z]=lp[t];d[148+z]=lr[t];}
            for(int z=0;z<3;z++)derived_ranks[w*3+z]=ranks[w*3+z];derived_valid[w]=1;
        }
    }__syncthreads();
    } // exact-input derived-matrix cache miss
    // Misses already materialized these diagnostics at their original points.
    // Only hits need replay; avoid extra writes on the compatible alias path.
    if(reuse){
        for(int z=threadIdx.x;z<49;z+=blockDim.x)mass_inv[w*49+z]=mi[z];
        if(threadIdx.x==0){
            for(int z=0;z<36;z++){A_full[w*36+z]=a[z];lambda_full[w*36+z]=lf[z];}
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){int z=r*3+c,t=r*6+c;A_pos[w*9+z]=ap[t];A_ori[w*9+z]=ar[t];lambda_pos[w*9+z]=lp[t];lambda_ori[w*9+z]=lr[t];}
        }
    }
    if(threadIdx.x==0){
        for(int r=0;r<7;r++)for(int c=0;c<7;c++){real_t v=project[r*7+c];project[r*7+c]=v;project_t[c*7+r]=v;if(diagnostics)nullspace_matrix[w*49+r*7+c]=v;}
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
        if(ori_override)for(int r=0;r<3;r++)er[r]=ori_override[w*3+r];
        for(int r=0;r<3;r++){position_error[w*3+r]=gp[r]-(local_state?local_state[14+r]:ee_pos[w*3+r]);orientation_error[w*3+r]=er[r];
            desired[r]=gains[w*12+r]*position_error[w*3+r]-gains[w*12+r+6]*(local_state?local_state[26+r]:ee_velocity[w*6+r]);
            desired[r+3]=gains[w*12+r+3]*er[r]-gains[w*12+r+9]*(local_state?local_state[26+r+3]:ee_velocity[w*6+r+3]);}
        for(int c=0;c<7;c++)pd[c]=10*(q0[w*7+c]-(local_state?local_state[0+c]:q[w*7+c]))-REAL_C(6.324555320336759)*(local_state?local_state[7+c]:qvel[w*7+c]);
    }__syncthreads();
    const bool split_mode=OSC_UNCOUPLE<0 ? uncouple!=0 : OSC_UNCOUPLE!=0;
    if(split_mode){osc_mm_3_1_3(lp3,desired,wrench);osc_mm_3_1_3(lr3,desired+3,wrench+3);}else osc_mm_6_1_6(lf,desired,wrench);
    osc_mm_7_1_7(m,pd,pose);osc_mm_7_1_7(project_t,pose,nt);osc_mm_7_1_6(jt,wrench,tau);
    if(threadIdx.x==0){
        for(int r=0;r<3;r++){goal_pos[w*3+r]=gp[r];for(int c=0;c<3;c++)goal_rot[w*9+r*3+c]=gr[r*3+c];}
        for(int i=0;i<7;i++){nullspace_torque[w*7+i]=nt[i];real_t t=tau[i]+bias[w*7+i]+nt[i];raw_tau[w*7+i]=t;clipped[w*7+i]=clip(t,range[w*18+2*i],range[w*18+2*i+1]);ctrl[w*9+i]=clipped[w*7+i];}
        real_t a6=action[w*7+6],sgn=isnan(a6)?a6:(a6>0?REAL_C(1.):(a6<0?-REAL_C(1.):REAL_C(0.)));
        for(int i=0;i<2;i++){real_t value=clip(gripper[w*2+i]+REAL_C(.01)*(i==0?-REAL_C(1.):REAL_C(1.))*sgn,-1,1);gripper[w*2+i]=value;
            real_t lo=range[w*18+2*(i+7)],hi=range[w*18+2*(i+7)+1];ctrl[w*9+i+7]=(hi+lo)*REAL_C(.5)+(hi-lo)*REAL_C(.5)*value;}
        needs[w]=1;
    }__syncthreads();
}
}
