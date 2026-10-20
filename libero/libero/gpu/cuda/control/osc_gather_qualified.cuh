#include "precision.cuh"
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "../common/warp.cuh"
#include "../common/vector.cuh"

// Count-parametric site Jacobian and seven-joint controller projection.
// jacobian, selected and local_selected require the first complete warp.
// Other helpers execute on thread 0. Callers must provide stage block barriers.
namespace cf_osc_gather {
__device__ void gate(int w,const int32_t* mask,int32_t* error,
    const int64_t* state,const int64_t* source,const int64_t* epoch,
    const int32_t* valid,const int64_t* written,int32_t* active) {
    const int requested=mask[w];
    active[w]=0;
    if(requested!=1 || error[w]!=0) return;
    if(state[w]!=source[w]) {error[w]=203;return;}
    if(valid[w]==1 && written[w]==epoch[w]) active[w]=1;
}
__device__ void site(int w,const real_t* pos,const real_t* mat,
    const int32_t* active,real_t* outp,real_t* outm,int ns,int id) {
    if(active[w]!=1) return;
    for(int k=0;k<3;k++) outp[3*w+k]=pos[(w*ns+id)*3+k];
    for(int k=0;k<9;k++) outm[9*w+k]=mat[(w*ns+id)*9+k];
}
__device__ void jacobian(int w,const real_t* cdof,const real_t* subtree,
    const real_t* pos,const int32_t* active,const int32_t* ancestor,
    int root,int body,int nv,int nb,real_t* jp,real_t* jr,int32_t* status,
    const int32_t* body_dof_offsets=nullptr,const int32_t* body_dof_indices=nullptr) {
    if(threadIdx.x>=32) return;
    const int lane=libero_warp::lane();
    real_t* p=jp+w*3*nv; real_t* r=jr+w*3*nv;
    if(lane==0) status[w]=0;
    for(int k=lane;k<3*nv;k+=32) {p[k]=0;r[k]=0;}
    __syncwarp();
    if(active[w]==0) return;
    if(active[w]!=1) {if(lane==0)status[w]=4;return;}
    // One subtraction per offset, broadcast without changing arithmetic.
    real_t ox=0,oy=0,oz=0;
    if(lane==0){ox=pos[w*3]-subtree[(w*nb+root)*3];
        oy=pos[w*3+1]-subtree[(w*nb+root)*3+1];
        oz=pos[w*3+2]-subtree[(w*nb+root)*3+2];}
    ox=__shfl_sync(libero_warp::mask,ox,0);
    oy=__shfl_sync(libero_warp::mask,oy,0);
    oz=__shfl_sync(libero_warp::mask,oz,0);
    if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)) {if(lane==0)status[w]=5;return;}
    bool bad=false;
    // Optional immutable model CSR removes inactive-column branches. The
    // dense diagnostics still receive their required zero values above.
    const bool indexed=body_dof_offsets&&body_dof_indices;
    const int begin=indexed?body_dof_offsets[body]:0;
    const int end=indexed?body_dof_offsets[body+1]:nv;
    for(int z=begin+lane;z<end;z+=32) {
        const int k=indexed?body_dof_indices[z]:z;
        if(!indexed&&ancestor[body*nv+k]!=1)continue;
        const real_t* d=cdof+(w*nv+k)*6;
        const real_t vx=d[3]+(d[1]*oz-d[2]*oy);
        const real_t vy=d[4]+(d[2]*ox-d[0]*oz);
        const real_t vz=d[5]+(d[0]*oy-d[1]*ox);
        bool ok=isfinite(vx)&&isfinite(vy)&&isfinite(vz);
        for(int j=0;j<6;j++) ok=ok&&isfinite(d[j]);
        bad=bad||!ok;
        if(ok){for(int j=0;j<3;j++) r[j*nv+k]=d[j];
            p[k]=vx;p[nv+k]=vy;p[2*nv+k]=vz;}
    }
    // Failure clears the entire diagnostic, including columns completed by
    // other lanes. Register voting alone does not order global writes.
    const bool failed=libero_warp::any(bad);
    __syncwarp();
    if(failed){if(lane==0)status[w]=5;
        for(int k=lane;k<3*nv;k+=32){p[k]=0;r[k]=0;}}
    __syncwarp();
}
__device__ void route(int w,const int32_t* active,const int32_t* status,int32_t* out) {
    out[w]=(active[w]==1 && status[w]==0)?1:0;
}
__device__ void selected(int w,const real_t* qpos,const real_t* qvel,
    const real_t* pos,const real_t* rot,const real_t* jp,const real_t* jr,
    const real_t* qm,const real_t* qbias,const int32_t* active,
    const int32_t* qi,const int32_t* vi,const int32_t* sparse,int nq,int nv,int nm,
    real_t* ee_pos,real_t* ee_mat,real_t* vp,real_t* vo,real_t* q,real_t* qd,
    real_t* J,real_t* M,real_t* bias,int32_t* status,
    const int32_t* body_dof_offsets=nullptr,const int32_t* body_dof_indices=nullptr,
    int body=0,int producer_finite=0,int sparse_velocity=0,
    const int32_t* cold_qi=nullptr,const int32_t* cold_vi=nullptr,const int32_t* cold_mass=nullptr) {
    if(threadIdx.x>=32) return;
    const int lane=libero_warp::lane();
    if(lane==0) status[w]=0;
    libero_vec::fill_lane(ee_pos+w*3,3,REAL_C(0.0),lane,32);
    libero_vec::fill_lane(vp+w*3,3,REAL_C(0.0),lane,32);
    libero_vec::fill_lane(vo+w*3,3,REAL_C(0.0),lane,32);
    libero_vec::fill_lane(ee_mat+w*9,9,REAL_C(0.0),lane,32);
    libero_vec::fill_lane(q+w*7,7,REAL_C(0.0),lane,32);
    libero_vec::fill_lane(qd+w*7,7,REAL_C(0.0),lane,32);
    libero_vec::fill_lane(bias+w*7,7,REAL_C(0.0),lane,32);
    libero_vec::fill_lane(J+w*42,42,REAL_C(0.0),lane,32);
    libero_vec::fill_lane(M+w*49,49,REAL_C(0.0),lane,32);
    __syncwarp();
    if(active[w]==0) return;
    if(active[w]!=1) {if(lane==0)status[w]=1;return;}
    bool bad=false;
    bad=libero_vec::any_nonfinite_lane(qpos+w*nq,nq,lane,32);
    bad=libero_vec::any_nonfinite_lane(qvel+w*nv,nv,lane,32)||bad;
    bad=libero_vec::any_nonfinite_lane(qbias+w*nv,nv,lane,32)||bad;
    // Only local_selected can issue this certificate: same pointers/extents,
    // successful producer, same invocation and no intervening writer. Its host
    // installer additionally proves that no input/output ranges alias.
    if(!producer_finite)for(int r=0;r<3;r++) {
        bad=libero_vec::any_nonfinite_lane(jp+(w*3+r)*nv,nv,lane,32)||bad;
        bad=libero_vec::any_nonfinite_lane(jr+(w*3+r)*nv,nv,lane,32)||bad;
    }
    bad=libero_vec::any_nonfinite_lane(qm+w*nm,nm,lane,32)||bad;
    bad=libero_vec::any_nonfinite_lane(pos+w*3,3,lane,32)||bad;
    bad=libero_vec::any_nonfinite_lane(rot+w*9,9,lane,32)||bad;
    if(libero_warp::any(bad)) {if(lane==0)status[w]=2;return;}
    libero_vec::copy_lane(pos+w*3,ee_pos+w*3,3,lane,32);
    libero_vec::copy_lane(rot+w*9,ee_mat+w*9,9,lane,32);
    // Retain the accepted sequential GEMV. This patch changes gathers and
    // validation only; it does not replace standard BLAS with a warp sum.
    if(lane==0) for(int r=0;r<3;r++) {
        real_t a=0,b=0;
        if(sparse_velocity&&body_dof_offsets&&body_dof_indices){
            for(int z=body_dof_offsets[body];z<body_dof_offsets[body+1];z++){
                const int k=body_dof_indices[z];a+=jp[(w*3+r)*nv+k]*qvel[w*nv+k];b+=jr[(w*3+r)*nv+k]*qvel[w*nv+k];}
        }else for(int k=0;k<nv;k++) {a+=jp[(w*3+r)*nv+k]*qvel[w*nv+k];b+=jr[(w*3+r)*nv+k]*qvel[w*nv+k];}
        vp[w*3+r]=a;vo[w*3+r]=b;
    }
    // Guard every used descriptor value in the current invocation. Arbitrary
    // in-place external writes invalidate this shortcut and use the live route.
    bool changed=!(cold_qi&&cold_vi&&cold_mass);
    if(!changed){
        for(int z=lane;z<7;z+=32)changed=changed||qi[z]!=cold_qi[z]||vi[z]!=cold_vi[z];
        for(int z=lane;z<49;z+=32)changed=changed||sparse[vi[z/7]*nv+vi[z%7]]!=cold_mass[z];
    }
    const bool mapped=!libero_warp::any(changed);
    const int32_t* qindex=mapped?cold_qi:qi;
    const int32_t* vindex=mapped?cold_vi:vi;
    // Pair destinations only inside each seven-column row. Indirect sources
    // use wide reads solely when both live selectors are consecutive.
    for(int pair=lane;pair<4;pair+=32) {
        const int i=pair*2, v0=vindex[i], q0=qindex[i];
        if(i+1<7) {
            const int v1=vindex[i+1],q1=qindex[i+1];
            real2_t x=q1==q0+1?libero_vec::load_pair(qpos+w*nq+q0):make_real2(qpos[w*nq+q0],qpos[w*nq+q1]);
            libero_vec::store_pair(q+w*7+i,x.x,x.y);
            x=v1==v0+1?libero_vec::load_pair(qvel+w*nv+v0):make_real2(qvel[w*nv+v0],qvel[w*nv+v1]);
            libero_vec::store_pair(qd+w*7+i,x.x,x.y);
            x=v1==v0+1?libero_vec::load_pair(qbias+w*nv+v0):make_real2(qbias[w*nv+v0],qbias[w*nv+v1]);
            libero_vec::store_pair(bias+w*7+i,x.x,x.y);
            for(int r=0;r<3;r++) {
                x=v1==v0+1?libero_vec::load_pair(jp+(w*3+r)*nv+v0):make_real2(jp[(w*3+r)*nv+v0],jp[(w*3+r)*nv+v1]);
                libero_vec::store_pair(J+w*42+r*7+i,x.x,x.y);
                x=v1==v0+1?libero_vec::load_pair(jr+(w*3+r)*nv+v0):make_real2(jr[(w*3+r)*nv+v0],jr[(w*3+r)*nv+v1]);
                libero_vec::store_pair(J+w*42+(r+3)*7+i,x.x,x.y);
            }
        } else {
            q[w*7+i]=qpos[w*nq+q0];qd[w*7+i]=qvel[w*nv+v0];bias[w*7+i]=qbias[w*nv+v0];
            for(int r=0;r<3;r++){J[w*42+r*7+i]=jp[(w*3+r)*nv+v0];J[w*42+(r+3)*7+i]=jr[(w*3+r)*nv+v0];}
        }
    }
    // The flattened pair may cross a row boundary; both sparse addresses
    // are checked independently. Missing cells retain the zero initialization.
    for(int pair=lane;pair<25;pair+=32) {
        const int z=pair*2, row=vi[z/7],col=vi[z%7];
        const int a0=mapped?cold_mass[z]:sparse[row*nv+col];
        if(z+1<49){
            const int a1=mapped?cold_mass[z+1]:sparse[vi[(z+1)/7]*nv+vi[(z+1)%7]];
            if(a0>=0&&a0<nm-1&&a1==a0+1){const real2_t x=libero_vec::load_pair(qm+w*nm+a0);libero_vec::store_pair(M+w*49+z,x.x,x.y);}
            else {if(a0>=0)M[w*49+z]=qm[w*nm+a0];if(a1>=0)M[w*49+z+1]=qm[w*nm+a1];}
        }else if(a0>=0)M[w*49+z]=qm[w*nm+a0];
    }
    __syncwarp();
}
__device__ void status(int w,const int32_t* producer,const int32_t* adapter,int32_t* out) {
    out[w]=producer[w]?10+producer[w]:adapter[w];
}
__device__ void publish_velocity(int w,const int32_t* active,const int32_t* status,
    const real_t* vp,const real_t* vo,real_t* target,int32_t* first) {
    if(active[w]!=1) return;
    if(status[w]!=0) {if(first[w]==0) first[w]=202;return;}
    for(int k=0;k<3;k++) {target[w*6+k]=vp[w*3+k];target[w*6+k+3]=vo[w*3+k];}
}
__device__ void live_bias(int w,const int32_t* mask,const int32_t* first,
    const int64_t* state,const int64_t* source,const int64_t* epoch,
    const int32_t* valid,const int64_t* written,const real_t* bias,real_t* selected,
    const int32_t* indices,int32_t* parent_valid,int64_t* parent_written,int nv) {
    parent_valid[w]=0;
    if(mask[w]!=1 || first[w]!=0 || state[w]!=source[w]) return;
    if(valid[w]!=1 || written[w]!=epoch[w]) return;
    const uintptr_t src=(uintptr_t)(bias+w*nv),dst=(uintptr_t)(selected+w*7);
    const bool overlap=src<=dst?dst-src<(uintptr_t)nv*sizeof(real_t):src-dst<7*sizeof(real_t);
    if(overlap){for(int k=0;k<7;k++)selected[w*7+k]=bias[w*nv+indices[k]];}
    else {
    int k=0;
    for(;k+1<7;k+=2){const int i=indices[k],j=indices[k+1];
        const real2_t x=j==i+1?libero_vec::load_pair(bias+w*nv+i):make_real2(bias[w*nv+i],bias[w*nv+j]);
        libero_vec::store_pair(selected+w*7+k,x.x,x.y);}
    selected[w*7+k]=bias[w*nv+indices[k]];
    }
    parent_written[w]=epoch[w];parent_valid[w]=1;
}

// Cooperative local adapter: columns are written once and validated before
// selected gathers. The accepted sequential velocity accumulation is retained.
#ifndef OSC_NV
#define OSC_NV 1
#endif
__device__ void local_selected(int w,const real_t* cdof,const real_t* subtree,
 const real_t* pos,const int32_t* active,const int32_t* ancestor,
 int root,int body,int nv,int nb,real_t* jp,real_t* jr,int32_t* jac_status,
 int32_t* adapter_active,const real_t* qpos,const real_t* qvel,const real_t* rot,
 const real_t* qm,const real_t* qbias,const int32_t* qi,const int32_t* vi,
 const int32_t* sparse,int nq,int nm,real_t* ep,real_t* er,real_t* vp,real_t* vo,
 real_t* q,real_t* qd,real_t* J,real_t* M,real_t* bias,int32_t* adapter_status,int32_t* out_status,
 const int32_t* body_dof_offsets=nullptr,const int32_t* body_dof_indices=nullptr,
 int producer_finite=0,int sparse_velocity=0,
 const int32_t* cold_qi=nullptr,const int32_t* cold_vi=nullptr,const int32_t* cold_mass=nullptr){
 if(threadIdx.x>=32) return;
 jacobian(w,cdof,subtree,pos,active,ancestor,root,body,nv,nb,jp,jr,jac_status,body_dof_offsets,body_dof_indices);
 __syncwarp();
 if(libero_warp::lane()==0)route(w,active,jac_status,adapter_active);
 __syncwarp();
 selected(w,qpos,qvel,pos,rot,jp,jr,qm,qbias,adapter_active,qi,vi,sparse,nq,nv,nm,
          ep,er,vp,vo,q,qd,J,M,bias,adapter_status,body_dof_offsets,body_dof_indices,
          body,producer_finite,sparse_velocity,cold_qi,cold_vi,cold_mass);
 __syncwarp();
 if(libero_warp::lane()==0)status(w,jac_status,adapter_status,out_status);
}
}
