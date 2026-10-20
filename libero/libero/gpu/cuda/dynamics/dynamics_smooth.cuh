#include "precision.cuh"
#pragma once
#include <cmath>
#include <cstddef>
#include "../common/packed_mass.cuh"
#include "../common/warp.cuh"

// Smooth combine and external-force projection use one full warp per world.
// Their callers must enter with all block threads and rendezvous between stages.
// Other helpers in this header retain the owning-lane contract.
// Topology is immutable, qualified by the host before launch. World arrays use
// their original compact strides. Compilation permits FMA contraction.
__device__ inline void d_three_rn(int w, const real_t* passive, const real_t* bias,
    const real_t* applied, const real_t* actuator, const int* active,
    int* entry_status, real_t* smooth, real_t* seed, int nv) {
    // All block threads enter; warp 0 owns ordered DOF tiles.
    if(threadIdx.x>=32)return;
    const int lane=libero_warp::lane();
    const size_t v=size_t(w)*nv;
    if(lane==0)entry_status[w]=0;
    for(int i=lane;i<nv;i+=32)smooth[v+i]=seed[v+i]=REAL_C(0.0);
    __syncwarp();
    if(active[w]==0)return;
    if(active[w]!=1){if(lane==0)entry_status[w]=1;return;}
    bool bad=false;
    for(int i=lane;i<nv;i+=32)
        bad|=!isfinite(passive[v+i])||!isfinite(bias[v+i])||
             !isfinite(applied[v+i])||!isfinite(actuator[v+i]);
    if(libero_warp::any(bad)){if(lane==0)entry_status[w]=3;return;}
    for(int base=0;base<nv;base+=32){
        const int i=base+lane;
        real_t force=REAL_C(0.0);
        if(i<nv){force=passive[v+i]-bias[v+i];force+=applied[v+i];force+=actuator[v+i];}
        const int first=libero_warp::first_index(i<nv&&!isfinite(force),i);
        if(i<nv&&i<first)smooth[v+i]=seed[v+i]=force;
        __syncwarp();
        if(first!=libero_warp::no_index){if(lane==0)entry_status[w]=4;return;}
    }
}

__device__ inline void d_seed_rhs(int w, const real_t* qfrc, real_t* rhs,
    const int* active, const int* entry_status, const int* d02_status, int nv) {
    const size_t v=size_t(w)*nv, r=size_t(w)*4*nv;
    libero_vec::fill_serial(rhs+r,4*nv,REAL_C(0.0));
    if(active[w]!=1||entry_status[w]!=0||d02_status[w]!=0) return;
    if(d_copy_overlap(qfrc+v,rhs+r,nv))
        for(int i=0;i<nv;++i)rhs[r+i]=qfrc[v+i];
    else libero_vec::copy_serial(qfrc+v,rhs+r,nv);
}

__device__ inline void d_route(int w, const int* entry_status,
    const int* d02_status, const int* solve_status, const int* active,
    const real_t* solution, real_t* smooth_force, real_t* smooth_rn,
    real_t* qacc_smooth, int* status, int nv) {
    const size_t v=size_t(w)*nv,r=size_t(w)*4*nv;
    int s=0;
    if(active[w]!=0) {
        if(entry_status[w]!=0) s=entry_status[w];
        else if(d02_status[w]!=0) s=10+d02_status[w];
        else if(solve_status[w]==5) s=25;
        else if(solve_status[w]!=0) s=20+solve_status[w];
    }
    status[w]=s;
    const bool publish=active[w]!=0&&(s==0||s==25);
    if(publish) {
        if(d_copy_overlap(solution+r,qacc_smooth+v,nv))
            for(int i=0;i<nv;++i)qacc_smooth[v+i]=solution[r+i];
        else libero_vec::copy_serial(solution+r,qacc_smooth+v,nv);
    } else {
        libero_vec::fill_serial(qacc_smooth+v,nv,REAL_C(0.0));
        libero_vec::fill_serial(smooth_force+v,nv,REAL_C(0.0));
        libero_vec::fill_serial(smooth_rn+v,nv,REAL_C(0.0));
    }
}

__device__ inline void d_project(int w, const real_t* seed, const real_t* xfrc,
    const real_t* xipos, const real_t* subtree_com, const real_t* cdof,
    const int* body_ids, const int* body_count, const int* active,
    const int* body_rootid, const int* ancestor, real_t* scratch, real_t* stages,
    real_t* qfrc, int* status, int nbody, int nv, int diagnostic,
    const int*body_dof_offsets=nullptr,const int*body_dof_indices=nullptr) {
    // One full warp cooperates. Sequential bodies and ordered DOF tiles retain
    // the scalar accumulation order and precise private failure prefix.
    if(threadIdx.x>=32)return;
    const int lane=libero_warp::lane();
    const size_t v=size_t(w)*nv,b=size_t(w)*nbody;
    if(lane==0)status[w]=0;
    if(active[w]!=0&&active[w]!=1){if(lane==0)status[w]=1;return;}
    if(active[w]==0)return;
    if(body_count[w]!=nbody-1){if(lane==0)status[w]=2;return;}
    bool bad=false;
    for(int i=lane;i<nbody-1;i+=32)bad|=body_ids[size_t(w)*(nbody-1)+i]!=i+1;
    if(libero_warp::any(bad)){if(lane==0)status[w]=3;return;}
    bad=false;for(int j=lane;j<nv;j+=32)bad|=!isfinite(seed[v+j]);
    if(libero_warp::any(bad)){if(lane==0)status[w]=4;return;}
    bad=false;
    for(int i=lane;i<(nbody-1)*6;i+=32)bad|=!isfinite(xfrc[(b+1)*6+i]);
    if(libero_warp::any(bad)){if(lane==0)status[w]=5;return;}
    // Only nonzero external-force bodies admit positional/ancestor operands.
    for(int body=1;body<nbody;++body){
        const real_t* f=xfrc+(b+body)*6;
        int nonzero=0;
        if(lane==0)for(int k=0;k<6;++k)nonzero|=f[k]!=REAL_C(0.0);
        nonzero=__shfl_sync(libero_warp::mask,nonzero,0);
        if(!nonzero)continue;
        const int root=body_rootid[body];bad=false;
        if(lane<3)bad=!isfinite(xipos[(b+body)*3+lane])||!isfinite(subtree_com[(b+root)*3+lane]);
        if(body_dof_offsets && body_dof_indices) {
            for(int k=body_dof_offsets[body]+lane;k<body_dof_offsets[body+1];k+=32)
                for(int r=0;r<6;++r)bad|=!isfinite(cdof[(v+body_dof_indices[k])*6+r]);
        } else for(int j=lane;j<nv;j+=32)if(ancestor[size_t(body)*nv+j]!=0)
            for(int k=0;k<6;++k)bad|=!isfinite(cdof[(v+j)*6+k]);
        if(libero_warp::any(bad)){if(lane==0)status[w]=6;return;}
    }
    for(int j=lane;j<nv;j+=32)scratch[v+j]=seed[v+j];
    __syncwarp();
    for(int body=1;body<nbody;++body){
        real_t f[6],offset[3];int nonzero=0;
        for(int k=0;k<6;++k){real_t value=lane==0?xfrc[(b+body)*6+k]:REAL_C(0.0);
            f[k]=__shfl_sync(libero_warp::mask,value,0);nonzero|=f[k]!=REAL_C(0.0);}
        if(!nonzero)continue;
        const int root=body_rootid[body];
        for(int k=0;k<3;++k){real_t value=lane==0?xipos[(b+body)*3+k]-subtree_com[(b+root)*3+k]:REAL_C(0.0);
            offset[k]=__shfl_sync(libero_warp::mask,value,0);}
        if(!isfinite(offset[0])||!isfinite(offset[1])||!isfinite(offset[2])){
            if(lane==0)status[w]=7;return;}
        // Diagnostic rows retain the dense route. The qualified CSR route
        // schedules ascending active DOFs; inactive columns still perform
        // their original two +0 updates, including signed-zero normalization.
        const bool sparse=!diagnostic && body_dof_offsets && body_dof_indices;
        const int begin=sparse?body_dof_offsets[body]:0;
        const int count=sparse?body_dof_offsets[body+1]-begin:nv;
        for(int base=0;base<count;base+=32){
            const int ordinal=base+lane;
            const int j=ordinal<count?(sparse?body_dof_indices[begin+ordinal]:ordinal):nv;
            real_t jp[3]={REAL_C(0.0),REAL_C(0.0),REAL_C(0.0)},jr[3]={REAL_C(0.0),REAL_C(0.0),REAL_C(0.0)};
            real_t force=REAL_C(0.0),torque=REAL_C(0.0),after_force=REAL_C(0.0),after_torque=REAL_C(0.0);
            int step=5;
            if(j<nv){
                if(sparse || ancestor[size_t(body)*nv+j]!=0){
                    const real_t* c=cdof+(v+j)*6;
                    for(int k=0;k<3;++k)jr[k]=c[k];
                    jp[0]=c[3]+(jr[1]*offset[2]-jr[2]*offset[1]);
                    jp[1]=c[4]+(jr[2]*offset[0]-jr[0]*offset[2]);
                    jp[2]=c[5]+(jr[0]*offset[1]-jr[1]*offset[0]);
                    for(int k=0;k<3;++k)if(!isfinite(jp[k]))step=0;
                }
                if(step==5){
                    for(int k=0;k<3;++k)if(f[k]!=REAL_C(0.0))force+=jp[k]*f[k];
                    if(!isfinite(force))step=1;
                }
                if(step==5){after_force=scratch[v+j]+force;if(!isfinite(after_force))step=2;}
                if(step==5){
                    for(int k=0;k<3;++k)if(f[k+3]!=REAL_C(0.0))torque+=jr[k]*f[k+3];
                    if(!isfinite(torque))step=3;
                }
                if(step==5){after_torque=after_force+torque;if(!isfinite(after_torque))step=4;}
            }
            const int first=libero_warp::first_index(j<nv&&step!=5,j);
            if(j<nv&&j<=first){
                real_t* t=diagnostic?stages+(b+body)*13*nv+j:nullptr;
                if(diagnostic&&step>=1)for(int k=0;k<3;++k){
                    t[size_t(k)*nv]=offset[k];t[size_t(k+3)*nv]=jp[k];t[size_t(k+6)*nv]=jr[k];}
                if(step>=2)scratch[v+j]=after_force;
                if(diagnostic&&step>=3){t[9*nv]=force;t[10*nv]=after_force;}
                if(step>=4)scratch[v+j]=after_torque;
                if(diagnostic&&step==5){t[11*nv]=torque;t[12*nv]=after_torque;}
            }
            __syncwarp();
            if(first!=libero_warp::no_index){
                if(sparse)for(int k=lane;k<first;k+=32)
                    if(ancestor[size_t(body)*nv+k]==0)scratch[v+k]=(scratch[v+k]+REAL_C(0.0))+REAL_C(0.0);
                __syncwarp();
                if(lane==0)status[w]=7;return;
            }
        }
        if(sparse)for(int j=lane;j<nv;j+=32)
            if(ancestor[size_t(body)*nv+j]==0)scratch[v+j]=(scratch[v+j]+REAL_C(0.0))+REAL_C(0.0);
        __syncwarp();
    }
    for(int j=lane;j<nv;j+=32)qfrc[v+j]=scratch[v+j];
    __syncwarp();
}

__device__ inline void d_mass_clear(int w,int nv,int nm,real_t* ld,real_t* inv,
    real_t* sqrt_inv,real_t* solution,int* pivot,int* warning_count,
    int* warning_last,bool preserve_events) {
    const size_t v=size_t(w)*nv,m=size_t(w)*nm,r=size_t(w)*4*nv;
    for(int a=0;a<nm;++a) ld[m+a]=REAL_C(0.0);
    for(int i=0;i<nv;++i) {
        inv[v+i]=sqrt_inv[v+i]=REAL_C(0.0);
        if(!preserve_events) pivot[v+i]=0;
    }
    for(int i=0;i<4*nv;++i) solution[r+i]=REAL_C(0.0);
    if(!preserve_events) {warning_count[w]=0;warning_last[w]=-1;}
}

// Packed tree LDL^T, not dense POTRF. No dense expansion or external LAPACK
// replacement: each packed row contains its diagonal followed by ancestors.
__device__ inline void d_factor_solve(int w,const real_t* qm,const real_t* rhs,
    const int* count,const int* active,const int* madr,const int* parent,
    real_t* ld,real_t* inv,real_t* sqrt_inv,real_t* solution,int* pivot,
    int* status,int* warning_count,int* warning_last,int nv,int nm) {
    status[w]=0;
    d_mass_clear(w,nv,nm,ld,inv,sqrt_inv,solution,pivot,warning_count,warning_last,false);
    if(active[w]==0) return;
    if(active[w]!=1) {status[w]=1;return;}
    const int n=count[w];
    if(n<1||n>4) {status[w]=2;return;}
    const size_t v=size_t(w)*nv,m=size_t(w)*nm,r0=size_t(w)*4*nv;
    for(int a=0;a<nm;++a) if(!isfinite(qm[m+a])) {status[w]=3;return;}
    for(int a=0;a<4*nv;++a) if(!isfinite(rhs[r0+a])) {status[w]=3;return;}
    real_t* L=ld+m;real_t* x=solution+r0;
    bool bad=!libero_packed_mass::factor(qm+m,L,inv+v,sqrt_inv+v,pivot+v,
        warning_count[w],warning_last[w],nv,nm,madr,parent);
    for(int r=0;r<n;++r) {
        for(int i=0;i<nv;++i)x[r*nv+i]=rhs[r0+r*nv+i];
        bad|=!libero_packed_mass::solve_inplace(x+r*nv,L,inv+v,nv,madr,parent);
    }
    if(bad) {
        d_mass_clear(w,nv,nm,ld,inv,sqrt_inv,solution,pivot,warning_count,warning_last,true);
        status[w]=4;
    } else if(warning_count[w]!=0) status[w]=5;
}

#ifdef D_FAST_DATAFLOW
// The original externally visible work arrays are written once at the tail.
// All stage-to-stage operands, the single active RHS and candidate acceleration
// remain in shared memory. Failure still isolates durable public outputs.
__device__ inline void d_smooth_chain(int w,const real_t*passive,const real_t*bias,
    const real_t*applied,const real_t*actuator,const int*active,int*entry_status,
    real_t*smooth,real_t*seed,int nv,const real_t*xfrc,const real_t*xipos,
    const real_t*subtree_com,const real_t*cdof,const int*body_ids,const int*body_count,
    const int*body_rootid,const int*ancestor,real_t*scratch,real_t*stages,real_t*qfrc,
    int*d02_status,int nbody,const real_t*qm,const int*count,const int*madr,
    const int*parent,real_t*ld,real_t*inv,real_t*sqrt_inv,real_t*solution,int*pivot,
    int*solve_status,int*warning_count,int*warning_last,int nm,real_t*rhs,
    real_t*qacc_smooth,int*status,const real_t*source_ld,const real_t*source_inv,
    const real_t*source_sqrt_inv,const int*source_pivot,const int*source_status,
    const int*source_warning_count,const int*source_warning_last,const int*source_active,
    const int*position_valid,const int64_t*position_written,const int64_t*forward_epoch,
    int64_t*factor_reuse_count,
    const int*body_dof_offsets=nullptr,const int*body_dof_indices=nullptr) {
    DForceShared& s=d_force_shared();
    const size_t v=size_t(w)*nv,b=size_t(w)*nbody,m=size_t(w)*nm,r=size_t(w)*4*nv;
    // scratch is a diagnostic with partial-failure semantics; preserve its old
    // value until the original projection's validation has admitted writes.
    libero_vec::copy_block(scratch+v,s.scratch,nv);
    libero_vec::copy_block(qfrc+v,s.force,nv);
    __syncthreads();
    d_three_rn(0,passive+v,bias+v,applied+v,s.actuator,active+w,
               entry_status+w,s.smooth,s.seed,nv);
    __syncthreads();
    d_project(0,s.seed,xfrc+b*6,xipos+b*3,subtree_com+b*3,cdof+v*6,
        body_ids+size_t(w)*(nbody-1),body_count+w,active+w,body_rootid,ancestor,
        s.scratch,stages,s.force,d02_status+w,nbody,nv,0,body_dof_offsets,body_dof_indices);
    __syncthreads();
    __shared__ int solve,warnings,last,reuse;
    if(threadIdx.x==0){solve=0;warnings=0;last=-1;reuse=0;}
    for(int i=threadIdx.x;i<nv;i+=blockDim.x) {
        s.rhs[i]=(active[w]==1&&entry_status[w]==0&&d02_status[w]==0)?s.force[i]:REAL_C(0.0);
        s.acceleration[i]=REAL_C(0.0);s.inv[i]=s.sqrt_inv[i]=REAL_C(0.0);s.pivot[i]=0;
    }
    libero_vec::fill_block(s.ld,nm,REAL_C(0.0));
    __syncthreads();
    if(threadIdx.x==0&&active[w]!=0) {
        if(active[w]!=1)solve=1;
        else if(count[w]<1||count[w]>4)solve=2;
        else if(!d_finite(qm+m,nm)||!d_finite(s.rhs,nv))solve=3;
        else {
            reuse=position_valid[w]==1&&position_written[w]==forward_epoch[w]&&
                source_active[w]==1&&(source_status[w]==0||source_status[w]==5)&&
                d_finite(source_ld+m,nm)&&d_finite(source_inv+v,nv)&&
                d_finite(source_sqrt_inv+v,nv);
            if(reuse){warnings=source_warning_count[w];last=source_warning_last[w];++factor_reuse_count[w];}
        }
    }
    __syncthreads();
    if(active[w]==1&&solve==0) {
        libero_packed_mass::ForestPlan forest;
        bool factor_ok=true;
        if(reuse) {
            libero_vec::copy_block(source_ld+m,s.ld,nm);
            libero_vec::copy_block(source_inv+v,s.inv,nv);
            libero_vec::copy_block(source_sqrt_inv+v,s.sqrt_inv,nv);
            libero_vec::copy_block(source_pivot+v,s.pivot,nv);
            __syncthreads();forest=libero_packed_mass::inspect_forest_block(nv,parent);
        } else factor_ok=libero_packed_mass::factor_block(qm+m,s.ld,s.inv,s.sqrt_inv,
            s.pivot,warnings,last,nv,nm,madr,parent,forest);
        const bool solve_ok=libero_packed_mass::solve_rhs_block(s.rhs,s.acceleration,
            s.ld,s.inv,1,nv,madr,parent,forest);
        if(threadIdx.x==0) {
            if(!factor_ok||!solve_ok) {
                libero_vec::fill_serial(s.ld,nm,REAL_C(0.0));
                for(int i=0;i<nv;++i)s.inv[i]=s.sqrt_inv[i]=s.acceleration[i]=REAL_C(0.0);
                solve=4;
            } else if(warnings!=0)solve=5;
        }
    }
    __syncthreads();
    if(threadIdx.x!=0)return;
    solve_status[w]=solve;warning_count[w]=warnings;warning_last[w]=last;
    // Route into the same local force/acceleration consumed by public commit.
    d_route(0,entry_status+w,d02_status+w,solve_status+w,active+w,
            s.acceleration,s.force,s.smooth,s.acceleration,status+w,nv);
    libero_vec::copy_serial(s.ld,ld+m,nm);
    for(int i=0;i<nv;++i) {
        smooth[v+i]=s.smooth[i];seed[v+i]=s.seed[i];scratch[v+i]=s.scratch[i];
        qfrc[v+i]=s.force[i];qacc_smooth[v+i]=s.acceleration[i];
        rhs[r+i]=s.rhs[i];solution[r+i]=(solve==0||solve==5)?s.acceleration[i]:REAL_C(0.0);
        inv[v+i]=s.inv[i];sqrt_inv[v+i]=s.sqrt_inv[i];pivot[v+i]=s.pivot[i];
    }
    // The API retains four RHS diagnostic columns. Only the first is consumed
    // by smooth acceleration; unused columns keep their original zero content.
    libero_vec::fill_serial(rhs+r+nv,3*nv,REAL_C(0.0));
    libero_vec::fill_serial(solution+r+nv,3*nv,REAL_C(0.0));
}
#endif
