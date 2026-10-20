#include "precision.cuh"
#include "io_policy.cuh"
#pragma once
#ifndef DATAFLOW_FAST_PROFILE
#define DATAFLOW_FAST_PROFILE 0
#endif
// Handwritten S07 device adapter. Domain preparation and line-search retain
// scalar packed-mass ownership; row groups and line-search evaluation are collective.
// Include after the retained native S07 helpers (S07Args/P/finish/eval_point).
struct S07PreparedDx {
    int ready,nefc,ne,nf,base;
    const int* group_head;
    const int* compact_heads;
    const int* tile_head_offsets;
    int head_count;
    const int* head_dim;
    const real_t* head_mu;
    const real_t* head_friction;
    real_t gtol,slopescl;
};

__device__ void s07_prepare_scalar_dx(S07Args a,int w,int nv,int nM,int rows,
    int tracecap,S07PreparedDx* prepared) {
    prepared->ready=0;
    a.status[w]=0;
    if(a.active[w]==0) return;
    if(a.active[w]!=1) {a.status[w]=1;return;}
    int nefc=a.nefc[w],ne=a.ne[w],nf=a.nf[w];
    if(nefc<0||nefc>rows||ne<0||nf<0||ne>nefc||nf>nefc-ne) {a.status[w]=2;return;}
    if(tracecap<52) {a.status[w]=3;return;}
    int base=w*rows;
    for(int i=0;i<nefc;++i) {
        if(a.efc_type[base+i]==7) {
            int dim=a.cone_dim[base+i];
            if(i<ne+nf||(dim!=3&&dim!=4&&dim!=6)||i+dim>nefc||a.cone_mu[base+i]<=0) {
                a.status[w]=4;return;
            }
            i+=dim-1;
        }
    }
    const real_t* v=a.search+w*nv;
    real_t snorm=real_sqrt_rn(dot_strided(v,1,v,1,nv));
    if(!isfinite(snorm)) {a.status[w]=5;return;}
    a.quadGauss[w*3]=a.quad0[w];
    if(snorm<REAL_C(1e-15)) {finish(a,w,REAL_C(0.0),0,1,REAL_C(1.0));return;}
    real_t mean=a.model_scalars[1];
    real_t gtol=a.model_scalars[0]*REAL_C(0.01)*snorm*mean*(nv>1?nv:1);
    real_t slopescl=real_div_rn(REAL_C(1.0),snorm*mean*(nv>1?nv:1));
    if(!isfinite(gtol)||!isfinite(slopescl)) {a.status[w]=5;return;}
    real_t* Mv=a.Mv+w*nv;
    const real_t* M=a.qM+w*nM;
    for(int i=0;i<nv;++i) Mv[i]=0;
    for(int i=0;i<nv;++i) {
        int adr=a.madr[i];
        if(a.simplenum[i]) Mv[i]=M[adr]*v[i];
        else {
            Mv[i]+=M[adr]*v[i];
            int j=a.parent[i];++adr;
            while(j>=0) {
                Mv[i]+=M[adr]*v[j];Mv[j]+=M[adr]*v[i];
                j=a.parent[j];++adr;
            }
        }
    }
    for(int i=0;i<nv;++i) if(!isfinite(Mv[i])) {a.status[w]=5;return;}
    prepared->nefc=nefc;prepared->ne=ne;prepared->nf=nf;prepared->base=base;
    prepared->gtol=gtol;prepared->slopescl=slopescl;prepared->ready=1;
}

// All threads enter. The scalar setup retains packed-scatter and dot order.
__device__ void s07_prepare_dx(S07Args a,int w,int nv,int nM,int rows,
    int tracecap,S07PreparedDx* prepared,const int* group_head=nullptr,
    const int* compact_heads=nullptr,const int* compact_count=nullptr,
    const int* tile_head_offsets=nullptr,const int* head_dim=nullptr,const real_t* head_mu=nullptr,
    const real_t* head_friction=nullptr) {
    if(threadIdx.x==0){
        prepared->group_head=group_head;
        prepared->head_dim=head_dim;
        prepared->head_mu=head_mu;prepared->head_friction=head_friction;
        prepared->compact_heads=compact_heads&&compact_count?compact_heads:nullptr;
        prepared->head_count=prepared->compact_heads?compact_count[0]:0;
        prepared->tile_head_offsets=prepared->compact_heads?tile_head_offsets:nullptr;
        s07_prepare_scalar_dx(a,w,nv,nM,rows,tracecap,prepared);
        if(prepared->ready&&prepared->compact_heads&&
            (prepared->head_count<0||prepared->head_count>prepared->nefc)){
            a.status[w]=4;prepared->ready=0;
        }
    }
    __syncthreads();
}

__device__ real_t s07_tile_jv(int row,int begin,const real_t* tile,const real_t* Jv){
    return row>=begin&&row<begin+16?tile[row-begin]:Jv[row];
}

// Consume the just-produced Dx output while it still resides in shared memory.
// A cone crossing a tile boundary waits until its last row is available; only
// its earlier (at most five) rows then need the global Jv mirror.
__device__ bool s07_quad_group_values_dx(S07Args a,int w,int i,int rows,int begin,const real_t* tile,real_t* q,int rank=-1,
    const int* head_dim=nullptr,const real_t* head_mu=nullptr,
    const real_t* head_friction=nullptr){
    int base=w*rows;const real_t* Jv=a.Jv+base;
    real_t v=s07_tile_jv(i,begin,tile,Jv),j=a.Jaref[base+i],D=a.D[base+i];
    real_t DJ0=D*j;
    q[0]=j*DJ0;q[1]=v*DJ0;q[2]=v*D*v;
    bool elliptic=a.efc_type[base+i]==7;
    if(elliptic){int dim=head_dim&&rank>=0?head_dim[rank]:a.cone_dim[base+i];
      real_t mu=head_mu&&rank>=0?head_mu[rank]:a.cone_mu[base+i];
      for(int k=1;k<dim;++k){real_t jk=a.Jaref[base+i+k],vk=s07_tile_jv(i+k,begin,tile,Jv),Dk=a.D[base+i+k];
        real_t DJk=Dk*jk;q[0]+=jk*DJk;q[1]+=vk*DJk;q[2]+=vk*Dk*vk;}
      real_t U0=j*mu,V0=v*mu,UU=0,UV=0,VV=0;
      for(int k=1;k<dim;++k){real_t f=head_friction&&rank>=0?head_friction[(k-1)*rows+rank]:a.cone_friction[(base+i)*5+k-1];
        real_t U=a.Jaref[base+i+k]*f,V=s07_tile_jv(i+k,begin,tile,Jv)*f;
        UU+=U*U;UV+=U*V;VV+=V*V;}
      q[3]=U0;q[4]=V0;q[5]=UU;q[6]=UV;q[7]=VV;
      q[8]=real_div_rn(D,mu*mu*(1+mu*mu));
    }
    q[0]*=REAL_C(0.5);q[2]*=REAL_C(0.5);
    #pragma unroll
    for(int k=0;k<9;++k)if(k<(elliptic?9:3)&&!isfinite(q[k]))return false;
    return true;
}

__device__ bool s07_quad_group_dx(S07Args a,int w,int i,int rows,int begin,const real_t* tile){
    return s07_quad_group_values_dx(a,w,i,rows,begin,tile,a.quad+w*rows*3+3*i);
}

// At most 21 potential heads: the 16-row tile plus five preceding rows.
// Validate before ordered-prefix commit, preserving the original failed group's
// full private quad write while keeping all later groups held.
__device__ void s07_quad_tile_block(S07Args a,int w,int rows,int begin,int produced,
    const real_t* tile,const int* group_head,
    const int* compact_heads=nullptr,const int* tile_head_offsets=nullptr,
    const int* head_dim=nullptr,
    const real_t* head_mu=nullptr,const real_t* head_friction=nullptr) {
    __shared__ int first_bad;
    // Offsets assign each group to the tile containing its final row.
    // Every head appears once; cross-tile cones retain their earlier Jv mirror.
    const bool planned=compact_heads&&tile_head_offsets;
    const int tile_index=begin/16;
    const int first=planned?tile_head_offsets[tile_index]:(begin>5?begin-5:0);
    const int end=planned?tile_head_offsets[tile_index+1]:0;
    const int rank=first+threadIdx.x;
    const int i=planned?(rank<end?compact_heads[rank]:-1):rank;
    real_t q[9];int width=0;bool owned=false,valid=true;
    if(i>=0&&i<produced&&(planned||group_head[i]==i)){
        const int dim=a.efc_type[w*rows+i]==7?a.cone_dim[w*rows+i]:1;
        if(i+dim>begin&&i+dim<=produced){
            owned=true;width=dim==1?3:9;
            valid=s07_quad_group_values_dx(a,w,i,rows,begin,tile,q,planned?rank:-1,head_dim,head_mu,head_friction);
        }
    }
    if(libero_warp::warp()==0){
        const int bad=libero_warp::first_index(owned&&!valid,i);
        if(libero_warp::lane()==0)first_bad=bad;
    }
    __syncthreads();
    if(owned&&i<=first_bad){
#if LIBERO_NEWTON_SCALAR_IO
#else
        // Pair only within this head's owned quad record; its odd tail stays scalar.
#endif // LIBERO_NEWTON_SCALAR_IO
        #pragma unroll
#if LIBERO_NEWTON_SCALAR_IO
        for(int k=0;k<9;++k)if(k<width){
            a.quad[w*rows*3+3*i+k]=q[k];
#else
        for(int k=0;k<8;k+=2)if(k+1<width){
            libero_vec::store_pair(a.quad+w*rows*3+3*i+k,q[k],q[k+1]);
#endif // LIBERO_NEWTON_SCALAR_IO
        }
#if LIBERO_NEWTON_SCALAR_IO
#else
        a.quad[w*rows*3+3*i+width-1]=q[width-1];
#endif // LIBERO_NEWTON_SCALAR_IO
    }
    __syncthreads();
    if(threadIdx.x==0&&first_bad!=libero_warp::no_index)a.status[w]=5;
    __syncthreads();
}

__device__ void s07_jv_blas_dx(S07Args a,int w,int nefc,int rows,
    real_t* workspace,int* diagnostic,bool prepare_quad=false,const int* group_head=nullptr,
    const int* compact_heads=nullptr,const int* tile_head_offsets=nullptr,
    const int* head_dim=nullptr,
    const real_t* head_mu=nullptr,const real_t* head_friction=nullptr) {
    real_t* A=workspace;real_t* B=A+16*NV;real_t* C=B+NV;
    const real_t* v=a.search+w*NV;
    const real_t* J=a.J+w*rows*NV;
    real_t* Jv=a.Jv+w*rows;
    __shared__ int next_group;
    if(threadIdx.x==0)next_group=0;
#if LIBERO_NEWTON_SCALAR_IO
    for(int j=threadIdx.x;j<NV;j+=blockDim.x) B[j]=v[j];
#else
    libero_newton_io::copy_block(v,B,NV);
#endif // LIBERO_NEWTON_SCALAR_IO
    __syncthreads();
    for(int begin=0;begin<nefc;begin+=16) {
#if LIBERO_NEWTON_SCALAR_IO
        for(int z=threadIdx.x;z<16*NV;z+=blockDim.x) {
            int r=begin+z/NV;A[z]=r<nefc?J[r*NV+z%NV]:0;
#else
        constexpr int row_pairs=(NV+1)/2;
        // Pair only inside valid rows, including when an odd NV shifts alignment.
        for(int z=threadIdx.x;z<16*row_pairs;z+=blockDim.x) {
            const int k=z/row_pairs,i=2*(z%row_pairs),r=begin+k;
            if(i+1<NV) {
                const real2_t pair=r<nefc?libero_vec::load_pair(J+r*NV+i):make_real2(0,0);
                libero_vec::store_pair(A+k*NV+i,pair.x,pair.y);
            } else A[k*NV+i]=r<nefc?J[r*NV+i]:0;
#endif // LIBERO_NEWTON_SCALAR_IO
        }
#if LIBERO_NEWTON_SCALAR_IO
        for(int r=threadIdx.x;r<16;r+=blockDim.x) C[r]=0;
#else
        libero_newton_io::fill_block(C,16,REAL_C(0.0));
#endif // LIBERO_NEWTON_SCALAR_IO
        __syncthreads();dx_newton_jv(A,B,C);
#if LIBERO_NEWTON_SCALAR_IO
        for(int r=threadIdx.x;r<16;r+=blockDim.x)
            if(begin+r<nefc) Jv[begin+r]=C[r];
#else
        const int produced=begin+16<nefc?16:nefc-begin;
        libero_newton_io::copy_block(C,Jv+begin,produced);
#endif // LIBERO_NEWTON_SCALAR_IO
        __syncthreads();
        if(prepare_quad&&group_head){
            __shared__ int bad_jv;
            if(libero_warp::warp()==0){
                int r=libero_warp::lane();
                bool bad=r<16&&begin+r<nefc&&!isfinite(C[r]);
                bool any=libero_warp::any(bad);
                if(r==0)bad_jv=any;
            }
            __syncthreads();
            if(bad_jv){if(threadIdx.x==0)a.status[w]=5;}
            else s07_quad_tile_block(a,w,rows,begin,
                begin+16<nefc?begin+16:nefc,C,group_head,compact_heads,tile_head_offsets,
                head_dim,head_mu,head_friction);
        } else if(prepare_quad&&threadIdx.x==0){
            for(int r=0;r<16&&begin+r<nefc;++r)
                if(!isfinite(C[r])){a.status[w]=5;break;}
            int produced=begin+16<nefc?begin+16:nefc;
            while(!a.status[w]&&next_group<produced){
                int i=next_group,dim=a.efc_type[w*rows+i]==7?a.cone_dim[w*rows+i]:1;
                if(i+dim>produced)break;
                if(!s07_quad_group_dx(a,w,i,rows,begin,C)){a.status[w]=5;break;}
                next_group+=dim;
            }
        }
        __syncthreads();
        if(prepare_quad&&a.status[w])return; // uniform, after the barrier
    }
    if(threadIdx.x==0) ++diagnostic[6];
    __syncthreads();
}

__device__ void s07_quad_scalar_dx(S07Args a,int w,int nefc,int rows,int base,const real_t* Jv){
    real_t* qbase=a.quad+w*rows*3;
    for(int i=0;i<nefc;++i) {
        real_t* q=qbase+3*i;
        bool elliptic=a.efc_type[base+i]==7;
        real_t DJ0=a.D[base+i]*a.Jaref[base+i];
        q[0]=a.Jaref[base+i]*DJ0;q[1]=Jv[i]*DJ0;q[2]=Jv[i]*a.D[base+i]*Jv[i];
        if(elliptic) {
            int dim=a.cone_dim[base+i];real_t mu=a.cone_mu[base+i];
            for(int j=1;j<dim;++j) {
                real_t DJj=a.D[base+i+j]*a.Jaref[base+i+j];
                q[0]+=a.Jaref[base+i+j]*DJj;q[1]+=Jv[i+j]*DJj;
                q[2]+=Jv[i+j]*a.D[base+i+j]*Jv[i+j];
            }
            real_t U0=a.Jaref[base+i]*mu,V0=Jv[i]*mu,UU=0,UV=0,VV=0;
            for(int j=1;j<dim;++j) {
                real_t U=a.Jaref[base+i+j]*a.cone_friction[(base+i)*5+j-1];
                real_t V=Jv[i+j]*a.cone_friction[(base+i)*5+j-1];
                UU+=U*U;UV+=U*V;VV+=V*V;
            }
            q[3]=U0;q[4]=V0;q[5]=UU;q[6]=UV;q[7]=VV;
            q[8]=real_div_rn(a.D[base+i],mu*mu*(1+mu*mu));i+=dim-1;
        }
        q[0]*=REAL_C(0.5);q[2]*=REAL_C(0.5);
        for(int k=0;k<3;++k) if(!isfinite(q[k])) {a.status[w]=5;return;}
        if(elliptic) for(int k=3;k<9;++k) if(!isfinite(q[k])) {a.status[w]=5;return;}
    }
}

__device__ void s07_continue_dx(S07Args a,int w,int nv,int rows,
    const S07PreparedDx* prepared,bool quad_ready=false) {
    int nefc=prepared->nefc,ne=prepared->ne,nf=prepared->nf,base=prepared->base;
    real_t gtol=prepared->gtol,slopescl=prepared->slopescl;
    const real_t* v=a.search+w*nv;
    const real_t* Mv=a.Mv+w*nv;
    const real_t* Jv=a.Jv+base;
    // The fast tile consumer already checked each produced C cell. Avoid a
    // second global Jv scan; compatible retains its original validation order.
    __shared__ int gate_bad,jv_bad[(DX_THREADS+31)/32];
    if(!quad_ready){
        bool local_bad=false;
        for(int i=threadIdx.x;i<nefc;i+=blockDim.x)if(!isfinite(Jv[i]))local_bad=true;
        const bool warp_bad=libero_warp::any(local_bad);
        if(libero_warp::lane()==0)jv_bad[libero_warp::warp()]=warp_bad;
        __syncthreads();
    }
    if(threadIdx.x==0){
        gate_bad=0;
        if(!quad_ready)for(int i=0;i<(blockDim.x+31)/32;++i)if(jv_bad[i])gate_bad=1;
        if(!gate_bad){
            real_t* qG=a.quadGauss+w*3;
            qG[1]=dot_strided(v,1,a.Ma+w*nv,1,nv)-dot_strided(v,1,a.smooth+w*nv,1,nv);
            qG[2]=REAL_C(0.5)*dot_strided(v,1,Mv,1,nv);
            if(!isfinite(qG[1])||!isfinite(qG[2]))gate_bad=1;
        }
        if(gate_bad)a.status[w]=5;
    }
    __syncthreads();if(gate_bad)return;
    if(!quad_ready&&prepared->group_head){
        for(int begin=0;begin<nefc;begin+=16){
            s07_quad_tile_block(a,w,rows,begin,begin+16<nefc?begin+16:nefc,
                Jv+begin,prepared->group_head,prepared->compact_heads,prepared->tile_head_offsets,
                prepared->head_dim,prepared->head_mu,prepared->head_friction);
            if(a.status[w])return;
        }
    }else if(!quad_ready){
        if(threadIdx.x==0)s07_quad_scalar_dx(a,w,nefc,rows,base,Jv);
        __syncthreads();if(a.status[w])return;
    }
    int niter=0;
    P p0=s07_eval_point_block(a,w,rows,nefc,ne,nf,REAL_C(0.0),1,&niter,prepared->group_head,prepared->compact_heads,prepared->head_count,prepared->head_mu);
    P p1=s07_eval_point_block(a,w,rows,nefc,ne,nf,p0.alpha-real_div_rn(p0.d0,p0.d1),2,&niter,prepared->group_head,prepared->compact_heads,prepared->head_count,prepared->head_mu);
    if(p0.cost<p1.cost) p1=p0;
    if(fabs(p1.d0)<gtol) {
        s07_finish_block(a,w,p1.alpha,niter,p1.alpha==0?2:0,fabs(p1.d0)*slopescl);return;
    }
    int dir=p1.d0<0?1:-1,p2update=0;P p2={};
    while(p1.d0*dir<=-gtol&&niter<50) {
        p2=p1;p2update=1;
        p1=s07_eval_point_block(a,w,rows,nefc,ne,nf,p1.alpha-real_div_rn(p1.d0,p1.d1),3,&niter,prepared->group_head,prepared->compact_heads,prepared->head_count,prepared->head_mu);
        if(fabs(p1.d0)<gtol) {s07_finish_block(a,w,p1.alpha,niter,0,fabs(p1.d0)*slopescl);return;}
    }
    if(niter>=50) {s07_finish_block(a,w,p1.alpha,niter,3,fabs(p1.d0)*slopescl);return;}
    if(!p2update) {s07_finish_block(a,w,p1.alpha,niter,6,fabs(p1.d0)*slopescl);return;}
    P p2next=p1;
    P p1next=s07_eval_point_block(a,w,rows,nefc,ne,nf,p1.alpha-real_div_rn(p1.d0,p1.d1),4,&niter,prepared->group_head,prepared->compact_heads,prepared->head_count,prepared->head_mu);
    while(niter<50) {
        P pmid=s07_eval_point_block(a,w,rows,nefc,ne,nf,REAL_C(0.5)*(p1.alpha+p2.alpha),5,&niter,prepared->group_head,prepared->compact_heads,prepared->head_count,prepared->head_mu);
        P candidates[3]={p1next,p2next,pmid};real_t bestcost=0;int bestind=-1;
        for(int i=0;i<3;++i)
            if(fabs(candidates[i].d0)<gtol&&(bestind==-1||candidates[i].cost<bestcost)) {
                bestcost=candidates[i].cost;bestind=i;
            }
        if(bestind>=0) {
            s07_finish_block(a,w,candidates[bestind].alpha,niter,0,fabs(candidates[bestind].d0)*slopescl);return;
        }
        int b1=s07_update_bracket_block(a,w,rows,nefc,ne,nf,&p1,candidates,&p1next,&niter,6,prepared->group_head,prepared->compact_heads,prepared->head_count,prepared->head_mu);
        int b2=s07_update_bracket_block(a,w,rows,nefc,ne,nf,&p2,candidates,&p2next,&niter,7,prepared->group_head,prepared->compact_heads,prepared->head_count,prepared->head_mu);
        if(!b1&&!b2) {
            s07_finish_block(a,w,pmid.alpha,niter,pmid.cost<p0.cost?0:7,fabs(pmid.d0)*slopescl);return;
        }
    }
    if(p1.cost<=p2.cost&&p1.cost<p0.cost) s07_finish_block(a,w,p1.alpha,niter,4,fabs(p1.d0)*slopescl);
    else if(p2.cost<=p1.cost&&p2.cost<p0.cost) s07_finish_block(a,w,p2.alpha,niter,4,fabs(p2.d0)*slopescl);
    else s07_finish_block(a,w,REAL_C(0.0),niter,5,REAL_C(1.0));
}

__device__ void search_kernel_blas_dx(int w,S07Args a,int nworld,int nv,int nM,
    int rows,int tracecap,real_t* workspace,int* diagnostic,const int* group_head=nullptr,
    const int* compact_heads=nullptr,const int* compact_count=nullptr,
    const int* tile_head_offsets=nullptr,const int* head_dim=nullptr,const real_t* head_mu=nullptr,
    const real_t* head_friction=nullptr) {
    if(w>=nworld) return;
    __shared__ S07PreparedDx prepared;
    s07_prepare_dx(a,w,nv,nM,rows,tracecap,&prepared,group_head,compact_heads,compact_count,tile_head_offsets,head_dim,head_mu,head_friction);
    __syncthreads();
    if(!prepared.ready) return;
    s07_jv_blas_dx(a,w,prepared.nefc,rows,workspace,diagnostic,DATAFLOW_FAST_PROFILE!=0,prepared.group_head,prepared.compact_heads,prepared.tile_head_offsets,prepared.head_dim,prepared.head_mu,prepared.head_friction);
    if(!a.status[w]) s07_continue_dx(a,w,nv,rows,&prepared,DATAFLOW_FAST_PROFILE!=0);
    __syncthreads();
}
