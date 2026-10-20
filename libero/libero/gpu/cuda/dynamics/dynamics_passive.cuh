#include "precision.cuh"
#pragma once
#include <math.h>

// Full-world-block helpers; blockDim.x must be a positive multiple of 32.
// Warp collectives only exchange registers; shared publication uses block barriers.
namespace d_passive_detail {
__device__ inline int first_bad(int index) {
    __shared__ int first[32];
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int value=libero_warp::min(index);
    if(lane==0) first[warp]=value;
    __syncthreads();
    if(warp==0) {
        value=lane<(blockDim.x+31)/32 ? first[lane] : libero_warp::no_index;
        value=libero_warp::min(value);
        if(lane==0) first[0]=value;
    }
    __syncthreads();
    const int result=first[0];
    __syncthreads();
    return result;
}
__device__ inline bool bad(bool value) {
    return first_bad(value ? 0 : libero_warp::no_index)!=libero_warp::no_index;
}
}
__device__ inline void d_joint_passive_count(int w, real_t* qpos, real_t* qvel,
    int* active, int* jtype, int* qadr, int* dadr, real_t* stiffness,
    real_t* spring, real_t* damping, real_t* out, int* status,
    int nv, int nq, int njnt, int joint_disjoint=0,
    const int*joint_type_offsets=nullptr,const int*joint_type_indices=nullptr) {
    qpos += (long long)w*nq; qvel += (long long)w*nv; out += (long long)w*nv;
    if(threadIdx.x==0) status[w]=active[w]!=0 && active[w]!=1 ? 1 : 0;
    libero_vec::fill_block(out,nv,REAL_C(0.0));
    __syncthreads();
    if(active[w]!=1) return;
    bool invalid=false;
    invalid|=libero_vec::any_nonfinite_block_lane(qpos,nq);
    invalid|=libero_vec::any_nonfinite_block_lane(qvel,nv);
    if(d_passive_detail::bad(invalid)) {
        if(threadIdx.x==0) status[w]=2;
        __syncthreads(); return;
    }
    if(joint_disjoint && joint_type_offsets && joint_type_indices) {
        // Qualification proves unique output intervals. Type cohorts keep the
        // free/hinge/slide route uniform while each independent joint writes.
        for(int type=0;type<4;++type) {
            if(type==1)continue;
            for(int k=joint_type_offsets[type]+threadIdx.x;k<joint_type_offsets[type+1];k+=blockDim.x) {
                const int i=joint_type_indices[k],q=qadr[i],d=dadr[i];
                const real_t v=stiffness[i];
                if(type==0) {
                    // Two elements belong to this free joint; never cross a joint mask.
                    const real2_t qp=libero_vec::load_pair(qpos+q),sp=libero_vec::load_pair(spring+q);
                    const real2_t old=libero_vec::load_pair(out+d);
                    libero_vec::store_pair(out+d,old.x-v*(qp.x-sp.x),old.y-v*(qp.y-sp.y));
                    out[d+2]-=v*(qpos[q+2]-spring[q+2]);
                }
                else out[d]-=v*(qpos[q]-spring[q]);
            }
        }
    } else if(threadIdx.x==0)for(int i=0;i<njnt;++i) {
        // Malformed overlapping descriptors retain original ordered updates.
        int t=jtype[i],q=qadr[i],d=dadr[i];real_t v=stiffness[i];
        if(t==0)for(int axis=0;axis<3;++axis)out[d+axis]-=v*(qpos[q+axis]-spring[q+axis]);
        else if(t==2 || t==3)out[d]-=v*(qpos[q]-spring[q]);
    }
    __syncthreads();
    for(int k=2*threadIdx.x;k+1<nv;k+=2*blockDim.x) {
        const real2_t old=libero_vec::load_pair(out+k),d=libero_vec::load_pair(damping+k),v=libero_vec::load_pair(qvel+k);
        libero_vec::store_pair(out+k,old.x-d.x*v.x,old.y-d.y*v.y);
    }
    if((nv&1)&&threadIdx.x==0)out[nv-1]-=damping[nv-1]*qvel[nv-1];
    __syncthreads();
}

namespace d_passive_detail {
struct Vec { real_t v[3]; __device__ real_t& operator[](int i) {return v[i];}
    __device__ real_t operator[](int i) const {return v[i];} };
__device__ inline Vec cross(Vec a,Vec b) {
    return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};
}
__device__ inline Vec rot(const real_t* m,Vec v) {
    return {{m[0]*v[0]+m[1]*v[1]+m[2]*v[2],m[3]*v[0]+m[4]*v[1]+m[5]*v[2],m[6]*v[0]+m[7]*v[1]+m[8]*v[2]}};
}
__device__ inline Vec rot_t(const real_t* m,Vec v) {
    return {{m[0]*v[0]+m[3]*v[1]+m[6]*v[2],m[1]*v[0]+m[4]*v[1]+m[7]*v[2],m[2]*v[0]+m[5]*v[1]+m[8]*v[2]}};
}
}

namespace d_passive_detail {
// The body producer retains scalar arithmetic and diagnostic-prefix semantics.
__device__ inline bool fluid_body(int b, const real_t* cvel,const real_t* subtree_com,
    const real_t* xipos,const real_t* ximat,int* rootid,real_t* box,real_t* wind,
    real_t* opt,int diagnostic,real_t* stages,int stride,real_t* payload) {
    constexpr real_t pi=REAL_C(3.14159265358979323846);
        int root=rootid[b]; const real_t* m=ximat+9*b;
        Vec offset={{xipos[3*b]-subtree_com[3*root],xipos[3*b+1]-subtree_com[3*root+1],xipos[3*b+2]-subtree_com[3*root+2]}};
        const real2_t cv01=libero_vec::load_pair(cvel+6*b),cv23=libero_vec::load_pair(cvel+6*b+2),cv45=libero_vec::load_pair(cvel+6*b+4);
        Vec ang={{cv01.x,cv01.y,cv23.x}};
        Vec lin={{cv23.y,cv45.x,cv45.y}};
        Vec local_ang=rot_t(m,ang), ox=cross(offset,ang);
        for(int r=0;r<3;++r) lin[r]-=ox[r];
        Vec object_lin=rot_t(m,lin), local_wind=rot_t(m,{{wind[0],wind[1],wind[2]}});
        Vec local_lin; for(int r=0;r<3;++r) local_lin[r]=object_lin[r]-local_wind[r];
        for(int r=0;r<3;++r) if(!isfinite(offset[r]) || !isfinite(local_ang[r]) || !isfinite(object_lin[r]) || !isfinite(local_wind[r]) || !isfinite(local_lin[r])) {return false;}
        real_t* st=stages+b*stride;
        if(diagnostic!=0) for(int r=0;r<3;++r) {
            st[r]=box[3*b+r];st[3+r]=local_ang[r];st[6+r]=object_lin[r];
            st[9+r]=REAL_C(0.0);st[12+r]=local_wind[r];st[15+r]=local_ang[r];st[18+r]=local_lin[r];
        }
        const real2_t boxxy=libero_vec::load_pair(box+3*b);
        real_t x=boxxy.x,y=boxxy.y,z=box[3*b+2],visc=opt[0],density=opt[1];
        Vec torque={{REAL_C(0.),REAL_C(0.),REAL_C(0.)}},force={{REAL_C(0.),REAL_C(0.),REAL_C(0.)}};
        if(visc>REAL_C(0.0)) {
            real_t diam=(x+y+z)/REAL_C(3.0),ang_scl=-pi*diam*diam*diam*visc,lin_scl=-REAL_C(3.0)*pi*diam*visc;
            for(int r=0;r<3;++r) {torque[r]=local_ang[r]*ang_scl;force[r]=local_lin[r]*lin_scl;}
        }
        if(density>REAL_C(0.0)) {
            force[0]-=REAL_C(0.5)*density*y*z*fabs(local_lin[0])*local_lin[0];
            force[1]-=REAL_C(0.5)*density*x*z*fabs(local_lin[1])*local_lin[1];
            force[2]-=REAL_C(0.5)*density*x*y*fabs(local_lin[2])*local_lin[2];
            torque[0]-=density*x*(y*y*y*y+z*z*z*z)*fabs(local_ang[0])*local_ang[0]/REAL_C(64.0);
            torque[1]-=density*y*(x*x*x*x+z*z*z*z)*fabs(local_ang[1])*local_ang[1]/REAL_C(64.0);
            torque[2]-=density*z*(x*x*x*x+y*y*y*y)*fabs(local_ang[2])*local_ang[2]/REAL_C(64.0);
        }
        Vec wt=rot(m,torque),wf=rot(m,force);
        for(int r=0;r<3;++r) if(!isfinite(torque[r]) || !isfinite(force[r]) || !isfinite(wt[r]) || !isfinite(wf[r])) {return false;}
        if(diagnostic!=0) for(int r=0;r<3;++r) {st[21+r]=torque[r];st[24+r]=force[r];st[27+r]=wt[r];st[30+r]=wf[r];}
    for(int r=0;r<3;++r) {payload[r]=offset[r];payload[3+r]=wt[r];payload[6+r]=wf[r];}
    return true;
}
} // namespace d_passive_detail

__device__ inline void d_fluid_count(int w, real_t* cvel,real_t* subtree_com,
    real_t* xipos,real_t* ximat,real_t* cdof,int* active,int* rootid,real_t* mass,
    real_t* box,int* ancestor,real_t* wind,real_t* opt,int passive_disabled,
    int diagnostic,real_t* stages,real_t* qfrc,int* status,int nbody,int nv,
    const int*body_dof_offsets=nullptr,const int*body_dof_indices=nullptr) {
    using namespace d_passive_detail;
    cvel+=(long long)w*nbody*6; subtree_com+=(long long)w*nbody*3;
    xipos+=(long long)w*nbody*3; ximat+=(long long)w*nbody*9;
    cdof+=(long long)w*nv*6; qfrc+=(long long)w*nv;
    int stride=33+7*nv; stages+=(long long)w*nbody*stride;
    if(diagnostic!=0)libero_vec::fill_block(stages,(size_t)nbody*stride,REAL_C(0.0));
    __syncthreads();
    if(active[w]==0 || status[w]!=0) return;
    bool invalid=false;
    invalid|=libero_vec::any_nonfinite_block_lane(qfrc,nv);
    invalid|=libero_vec::any_nonfinite_block_lane(cvel,nbody*6);
    invalid|=libero_vec::any_nonfinite_block_lane(subtree_com,nbody*3);
    invalid|=libero_vec::any_nonfinite_block_lane(xipos,nbody*3);
    invalid|=libero_vec::any_nonfinite_block_lane(ximat,nbody*9);
    invalid|=libero_vec::any_nonfinite_block_lane(cdof,nv*6);
    if(d_passive_detail::bad(invalid)) {
        if(threadIdx.x==0) status[w]=2;
        __syncthreads(); return;
    }
    if(passive_disabled!=0 || (opt[0]<=REAL_C(0.0) && opt[1]<=REAL_C(0.0))) return;
    __shared__ real_t payload[9];
    __shared__ int body_ok;
    for(int b=1;b<nbody;++b) {
        if(mass[b]<REAL_C(1e-15)) continue;
        if(threadIdx.x==0) body_ok=fluid_body(b,cvel,subtree_com,xipos,ximat,
            rootid,box,wind,opt,diagnostic,stages,stride,payload);
        __syncthreads();
        if(!body_ok) {
            if(threadIdx.x==0) status[w]=2;
            __syncthreads(); return;
        }
        Vec offset,wt,wf;
        // One shared load per warp; every lane receives the same body payload.
        for(int r=0;r<3;++r) {
            real_t o=REAL_C(0.0),t=REAL_C(0.0),f=REAL_C(0.0);
            if((threadIdx.x&31)==0) {o=payload[r];t=payload[3+r];f=payload[6+r];}
            offset[r]=__shfl_sync(0xffffffffu,o,0);
            wt[r]=__shfl_sync(0xffffffffu,t,0);
            wf[r]=__shfl_sync(0xffffffffu,f,0);
        }
        real_t* st=stages+b*stride;
        const bool sparse=diagnostic==0 && body_dof_offsets && body_dof_indices;
        const int begin=sparse?body_dof_offsets[b]:0;
        const int count=sparse?body_dof_offsets[b+1]-begin:nv;
        for(int base=0;base<count;base+=blockDim.x) {
            const int ordinal=base+threadIdx.x;
            const int j=ordinal<count?(sparse?body_dof_indices[begin+ordinal]:ordinal):nv;
            Vec jp={{REAL_C(0.),REAL_C(0.),REAL_C(0.)}},jr={{REAL_C(0.),REAL_C(0.),REAL_C(0.)}};
            bool projection_bad=false, result_bad=false;
            real_t result=REAL_C(0.0);
            if(j<nv) {
                if(sparse || ancestor[b*nv+j]!=0) {
                    jr={{cdof[6*j],cdof[6*j+1],cdof[6*j+2]}};
                    Vec cp=cross(jr,offset);
                    for(int r=0;r<3;++r) jp[r]=cdof[6*j+3+r]+cp[r];
                }
                for(int r=0;r<3;++r) projection_bad|=!isfinite(jp[r]) || !isfinite(jr[r]);
                if(!projection_bad) {
                    real_t qforce=REAL_C(0.0),qtorque=REAL_C(0.0);
                    for(int r=0;r<3;++r) if(wf[r]!=REAL_C(0.0)) qforce+=jp[r]*wf[r];
                    result=qfrc[j]+qforce;
                    for(int r=0;r<3;++r) if(wt[r]!=REAL_C(0.0)) qtorque+=jr[r]*wt[r];
                    result+=qtorque;
                    result_bad=!isfinite(result);
                }
            }
            const int local=projection_bad ? 2*j : result_bad ? 2*j+1 : libero_warp::no_index;
            const int fail=d_passive_detail::first_bad(local);
            // Preserve the scalar prefix: bad projection writes nothing; a bad
            // result writes qfrc but does not write its final diagnostic slot.
            if(j<nv && 2*j<fail) {
                if(diagnostic!=0) for(int r=0;r<3;++r) {
                    st[33+nv*r+j]=jp[r];st[33+3*nv+nv*r+j]=jr[r];
                }
                qfrc[j]=result;
                if(diagnostic!=0 && 2*j+1<fail) st[33+6*nv+j]=result;
            }
            __syncthreads();
            if(fail!=libero_warp::no_index) {
                if(sparse)for(int k=threadIdx.x;k<fail/2;k+=blockDim.x)
                    if(ancestor[b*nv+k]==0)qfrc[k]=(qfrc[k]+REAL_C(0.0))+REAL_C(0.0);
                __syncthreads();
                if(threadIdx.x==0) status[w]=2;
                __syncthreads(); return;
            }
        }
        if(sparse)for(int j=threadIdx.x;j<nv;j+=blockDim.x)
            if(ancestor[b*nv+j]==0)qfrc[j]=(qfrc[j]+REAL_C(0.0))+REAL_C(0.0);
        __syncthreads();
    }
    __syncthreads();
}

__device__ inline void d_actuation_prepare(int w,real_t* length,real_t* moment,
    real_t* velocity,real_t* ctrl,int* active,int* vel_status,real_t* gain,
    real_t* bias0,real_t* bias1,real_t* bias2,int* bias_type,real_t* ctrl_lo,
    real_t* ctrl_hi,real_t* force_lo,real_t* force_hi,int* force_limited,
    int disable_flags,real_t* force,real_t* force_pre,real_t* ctrl_used,
    int* ctrl_route,int* force_route,real_t* qfrc_actuator,int* status,
    int* warn_badctrl_count,int* warn_badctrl_lastinfo,int* warning_number,
    int* warning_lastinfo,int nv,int nu) {
    length+=(long long)w*nu;moment+=(long long)w*nu*nv;velocity+=(long long)w*nu;
    ctrl+=(long long)w*nu;force+=(long long)w*nu;force_pre+=(long long)w*nu;
    ctrl_used+=(long long)w*nu;ctrl_route+=(long long)w*nu;force_route+=(long long)w*nu;
    qfrc_actuator+=(long long)w*nv;
    status[w]=0;warn_badctrl_count[w]=0;warn_badctrl_lastinfo[w]=-1;
    for(int j=0;j<nv;++j) qfrc_actuator[j]=REAL_C(0.0);
    for(int i=0;i<nu;++i) {force[i]=REAL_C(0.0);force_pre[i]=REAL_C(0.0);ctrl_used[i]=REAL_C(0.0);ctrl_route[i]=0;force_route[i]=0;}
    if(active[w]==0) return;
    if(active[w]!=1) {status[w]=1;return;}
    if(vel_status[w]!=0) {status[w]=2;return;}
    if((disable_flags&1024)!=0) return;
    int bad_index=-1;
    for(int i=0;i<nu;++i) {
        real_t u=ctrl[i];
        if((disable_flags&128)==0) {
            if(u<ctrl_lo[i]) {u=ctrl_lo[i];ctrl_route[i]=-1;}
            else if(u>ctrl_hi[i]) {u=ctrl_hi[i];ctrl_route[i]=1;}
        }
        if(bad_index<0 && (u!=u || u>REAL_C(1e10) || u< -REAL_C(1e10))) bad_index=i;
    }
    if(bad_index>=0) {warn_badctrl_count[w]=1;warn_badctrl_lastinfo[w]=bad_index;++warning_number[w];warning_lastinfo[w]=bad_index;}
    for(int i=0;i<nu;++i) {
        real_t u=REAL_C(0.0);
        if(bad_index<0) {u=ctrl[i];if(ctrl_route[i]==-1) u=ctrl_lo[i];else if(ctrl_route[i]==1) u=ctrl_hi[i];}
        ctrl_used[i]=u;
        real_t f=gain[i]*u;
        if(bias_type[i]==1) f+=bias0[i]+bias1[i]*length[i]+bias2[i]*velocity[i];
        force_pre[i]=f;
        if(force_limited[i]!=0) {
            if(f<force_lo[i]) {f=force_lo[i];force_route[i]=-1;}
            else if(f>force_hi[i]) {f=force_hi[i];force_route[i]=1;}
        }
        force[i]=f;
    }
}

// Large-NU preparation; the ordinary small actuator family retains scalar work.
__device__ inline void d_actuation_prepare_block(int w,real_t* length,real_t* moment,
    real_t* velocity,real_t* ctrl,int* active,int* vel_status,real_t* gain,
    real_t* bias0,real_t* bias1,real_t* bias2,int* bias_type,real_t* ctrl_lo,
    real_t* ctrl_hi,real_t* force_lo,real_t* force_hi,int* force_limited,
    int disable_flags,real_t* force,real_t* force_pre,real_t* ctrl_used,
    int* ctrl_route,int* force_route,real_t* qfrc_actuator,int* status,
    int* warn_badctrl_count,int* warn_badctrl_lastinfo,int* warning_number,
    int* warning_lastinfo,int nv,int nu) {
    length+=(long long)w*nu;moment+=(long long)w*nu*nv;velocity+=(long long)w*nu;
    ctrl+=(long long)w*nu;force+=(long long)w*nu;force_pre+=(long long)w*nu;
    ctrl_used+=(long long)w*nu;ctrl_route+=(long long)w*nu;force_route+=(long long)w*nu;
    qfrc_actuator+=(long long)w*nv;
    if(threadIdx.x==0){status[w]=0;warn_badctrl_count[w]=0;warn_badctrl_lastinfo[w]=-1;}
    libero_vec::fill_block(qfrc_actuator,nv,REAL_C(0.0));
    libero_vec::fill_block(force,nu,REAL_C(0.0));libero_vec::fill_block(force_pre,nu,REAL_C(0.0));
    libero_vec::fill_block(ctrl_used,nu,REAL_C(0.0));libero_vec::fill_block(ctrl_route,nu,0);
    libero_vec::fill_block(force_route,nu,0);
    __syncthreads();
    if(active[w]==0) return;
    if(active[w]!=1) {if(threadIdx.x==0)status[w]=1;return;}
    if(vel_status[w]!=0) {if(threadIdx.x==0)status[w]=2;return;}
    if((disable_flags&1024)!=0) return;
    int bad_index=libero_warp::no_index;
    for(int base=2*threadIdx.x;base<nu;base+=2*blockDim.x) {
        const bool pair=base+1<nu;
        const real2_t inputs=pair?libero_vec::load_pair(ctrl+base):make_real2(ctrl[base],REAL_C(0.0));
        for(int part=0;part<(pair?2:1);++part) {
        const int i=base+part;
        real_t u=part?inputs.y:inputs.x;
        if((disable_flags&128)==0) {
            if(u<ctrl_lo[i]) {u=ctrl_lo[i];ctrl_route[i]=-1;}
            else if(u>ctrl_hi[i]) {u=ctrl_hi[i];ctrl_route[i]=1;}
        }
        if(u!=u || u>REAL_C(1e10) || u< -REAL_C(1e10))bad_index=bad_index<i?bad_index:i;
        }
    }
    bad_index=d_passive_detail::first_bad(bad_index);
    if(threadIdx.x==0 && bad_index!=libero_warp::no_index) {warn_badctrl_count[w]=1;warn_badctrl_lastinfo[w]=bad_index;++warning_number[w];warning_lastinfo[w]=bad_index;}
    for(int base=2*threadIdx.x;base<nu;base+=2*blockDim.x) {
        const bool pair=base+1<nu;
        real_t fvals[2],prevals[2],used[2];
        const real2_t inputs=bad_index==libero_warp::no_index?
            (pair?libero_vec::load_pair(ctrl+base):make_real2(ctrl[base],REAL_C(0.0))):make_real2(REAL_C(0.0),REAL_C(0.0));
        for(int part=0;part<(pair?2:1);++part) {
        const int i=base+part;
        real_t u=REAL_C(0.0);
        if(bad_index==libero_warp::no_index) {u=part?inputs.y:inputs.x;if(ctrl_route[i]==-1) u=ctrl_lo[i];else if(ctrl_route[i]==1) u=ctrl_hi[i];}
        used[part]=u;
        real_t f=gain[i]*u;
        if(bias_type[i]==1) f+=bias0[i]+bias1[i]*length[i]+bias2[i]*velocity[i];
        prevals[part]=f;
        if(force_limited[i]!=0) {
            if(f<force_lo[i]) {f=force_lo[i];force_route[i]=-1;}
            else if(f>force_hi[i]) {f=force_hi[i];force_route[i]=1;}
        }
        fvals[part]=f;
        }
        if(pair) {
            libero_vec::store_pair(ctrl_used+base,used[0],used[1]);
            libero_vec::store_pair(force_pre+base,prevals[0],prevals[1]);
            libero_vec::store_pair(force+base,fvals[0],fvals[1]);
        } else {ctrl_used[base]=used[0];force_pre[base]=prevals[0];force[base]=fvals[0];}
    }
    __syncthreads();
}

#ifndef D_NO_ACTUATION_DX
extern __device__ void d_actuation_gemv_dx(const real_t*,const real_t*,real_t*);
#ifdef D_SCALAR_SUPPORT
extern __device__ void d_actuation_support_or_dx(const real_t*,const real_t*,real_t*);
#define D_ACTUATION_CONTRACT d_actuation_support_or_dx
#else
#define D_ACTUATION_CONTRACT d_actuation_gemv_dx
#endif

// All lanes in a world block must call this helper. Force clipping/warnings
// are scalar; the dense transpose contraction is performed by cuBLASDx.
__device__ inline void d_actuation_count(int w,real_t* length,real_t* moment,
    real_t* velocity,real_t* ctrl,int* active,int* vel_status,real_t* gain,
    real_t* bias0,real_t* bias1,real_t* bias2,int* bias_type,real_t* ctrl_lo,
    real_t* ctrl_hi,real_t* force_lo,real_t* force_hi,int* force_limited,
    int disable_flags,real_t* force,real_t* force_pre,real_t* ctrl_used,
    int* ctrl_route,int* force_route,real_t* qfrc_actuator,int* status,
    int* warn_badctrl_count,int* warn_badctrl_lastinfo,int* warning_number,
    int* warning_lastinfo,int nv,int nu) {
    __shared__ int run;
#ifdef D_FAST_DATAFLOW
    DForceShared& work=d_force_shared();
    libero_vec::fill_block(work.actuator,D_NV,REAL_C(0.0));
    const long long u=(long long)w*nu,v=(long long)w*nv;
    if(nu<=16) {
        if(threadIdx.x==0)d_actuation_prepare(0,length+u,moment+u*nv,velocity+u,ctrl+u,active+w,
            vel_status+w,gain,bias0,bias1,bias2,bias_type,ctrl_lo,ctrl_hi,
            force_lo,force_hi,force_limited,disable_flags,work.scalar_force,
            force_pre+u,ctrl_used+u,ctrl_route+u,force_route+u,qfrc_actuator+v,
            status+w,warn_badctrl_count+w,warn_badctrl_lastinfo+w,
            warning_number+w,warning_lastinfo+w,nv,nu);
    } else d_actuation_prepare_block(0,length+u,moment+u*nv,velocity+u,ctrl+u,active+w,
        vel_status+w,gain,bias0,bias1,bias2,bias_type,ctrl_lo,ctrl_hi,
        force_lo,force_hi,force_limited,disable_flags,work.scalar_force,
        force_pre+u,ctrl_used+u,ctrl_route+u,force_route+u,qfrc_actuator+v,
        status+w,warn_badctrl_count+w,warn_badctrl_lastinfo+w,
        warning_number+w,warning_lastinfo+w,nv,nu);
    __syncthreads();
    libero_vec::copy_block(work.scalar_force,force+u,nu);
#else
    if(nu<=16) {
        if(threadIdx.x==0)d_actuation_prepare(w,length,moment,velocity,ctrl,active,vel_status,gain,
            bias0,bias1,bias2,bias_type,ctrl_lo,ctrl_hi,force_lo,force_hi,
            force_limited,disable_flags,force,force_pre,ctrl_used,ctrl_route,
            force_route,qfrc_actuator,status,warn_badctrl_count,
            warn_badctrl_lastinfo,warning_number,warning_lastinfo,nv,nu);
    } else d_actuation_prepare_block(w,length,moment,velocity,ctrl,active,vel_status,gain,
        bias0,bias1,bias2,bias_type,ctrl_lo,ctrl_hi,force_lo,force_hi,
        force_limited,disable_flags,force,force_pre,ctrl_used,ctrl_route,
        force_route,qfrc_actuator,status,warn_badctrl_count,
        warn_badctrl_lastinfo,warning_number,warning_lastinfo,nv,nu);
#endif
    __syncthreads();
    if(threadIdx.x==0) {
        run=active[w]==1 && vel_status[w]==0 && (disable_flags&1024)==0 && status[w]==0;
        if(nv!=D_NV || nu!=D_NU){status[w]=2;run=0;}
    }
    __syncthreads();
    if(run) {
#ifdef D_FAST_DATAFLOW
        // Keep the contraction result live for smooth-force assembly. The
        // declared actuator diagnostic is still written once at this boundary.
        real_t* contracted=d_force_shared().actuator;
        D_ACTUATION_CONTRACT(moment+(long long)w*nu*nv,d_force_shared().scalar_force,contracted);
        libero_vec::copy_block(contracted,qfrc_actuator+(long long)w*nv,nv);
#else
        D_ACTUATION_CONTRACT(moment+(long long)w*nu*nv,force+(long long)w*nu,
            qfrc_actuator+(long long)w*nv);
#endif
        __syncthreads();
        bool invalid=false;
        for(int j=threadIdx.x;j<nv;j+=blockDim.x)
            invalid|=!isfinite(qfrc_actuator[(long long)w*nv+j]);
        if(d_passive_detail::bad(invalid) && threadIdx.x==0) status[w]=3;
    }
    __syncthreads();
}
#endif
