#include "precision.cuh"
#include "io_policy.cuh"
#pragma once
// Project-written collective evaluation of the retained MuJoCo S07 equations.
// Include after S07Args/P/eval_point and common/warp.cuh. All block threads enter;
// group_head is a world-local map: a head stores its row and a follower stores -1.
static_assert(DX_THREADS>=32&&DX_THREADS<=1024&&DX_THREADS%32==0,
              "S07 collectives require complete warps");
__device__ P s07_eval_point_block(const S07Args& a,int w,int rows,int nefc,
    int ne,int nf,real_t alpha,int tag,int* niter,const int* group_head,
    const int* compact_heads=nullptr,int head_count=0,const real_t* head_mu=nullptr) {
    __shared__ real_t partial[6][(DX_THREADS+31)/32];
    __shared__ P point;
    __shared__ int next_iter;
    if(!group_head&&!compact_heads) {
        if(threadIdx.x==0) {
            point=eval_point(a,w,rows,nefc,ne,nf,alpha,tag,niter);
            next_iter=*niter;
        }
        __syncthreads();
        P result=point;*niter=next_iter;
        __syncthreads();return result;
    }
    real_t t0=0,t1=0,t2=0,cost=0,d0=0,d1=0;
    const int base=w*rows,qbase=w*rows*3;
    // Immutable compact plan is produced once per invocation, not per alpha.
    const int groups=compact_heads?head_count:nefc;
    // Each lane consumes its own stable head directly. There is no block-wide
    // staging dependency between independent groups; preserve per-lane rank and
    // summation order, and retain the existing warp/block reduction below.
    for(int rank=threadIdx.x;rank<groups;rank+=blockDim.x) {
        const int i=compact_heads?compact_heads[rank]:rank;
        if(!compact_heads&&group_head[i]!=i)continue;
        const real_t* q=a.quad+qbase+3*i;
        bool add=false;
        if(i<ne) add=true;
        else if(i<ne+nf) {
            const real_t start=a.Jaref[base+i],dir=a.Jv[base+i];
            const real_t x=start+alpha*dir,f=a.loss[base+i],Rf=a.R[base+i]*f;
            if(-Rf<x&&x<Rf)add=true;
            else if(x<=-Rf){t0+=f*(-REAL_C(0.5)*Rf-start);t1+=-f*dir;}
            else{t0+=f*(-REAL_C(0.5)*Rf+start);t1+=f*dir;}
        } else if(a.efc_type[base+i]==7) {
            const real_t mu=head_mu&&compact_heads?head_mu[rank]:a.cone_mu[base+i];
#if LIBERO_NEWTON_SCALAR_IO
            const real_t U0=q[3],V0=q[4],UU=q[5],UV=q[6],VV=q[7],Dm=q[8];
#else
            // Cone heads own all three quad records (nine doubles). Guard
            // alignment on each actual pair; never read a follower as a head.
            const real2_t uv0=libero_vec::load_pair(q+3);
            const real2_t uuv=libero_vec::load_pair(q+5);
            const real2_t vdm=libero_vec::load_pair(q+7);
            const real_t U0=uv0.x,V0=uv0.y,UU=uuv.x,UV=uuv.y,VV=vdm.x,Dm=vdm.y;
#endif // LIBERO_NEWTON_SCALAR_IO
            const real_t N=U0+alpha*V0,Tsqr=UU+alpha*(REAL_C(2.0)*UV+alpha*VV);
            if(Tsqr<=0){if(N<0)add=true;}
            else {
                const real_t T=real_sqrt_rn(Tsqr);
                if(N>=mu*T){}
                else if(mu*N+T<=0)add=true;
                else {
                    const real_t N1=V0,T1=real_div_rn(UV+alpha*VV,T);
                    const real_t T2=real_div_rn(VV,T)-real_div_rn((UV+alpha*VV)*T1,T*T);
                    cost+=REAL_C(0.5)*Dm*(N-mu*T)*(N-mu*T);
                    d0+=Dm*(N-mu*T)*(N1-mu*T1);
                    d1+=Dm*((N1-mu*T1)*(N1-mu*T1)+(N-mu*T)*(-mu*T2));
                }
            }
        } else if(a.Jaref[base+i]+alpha*a.Jv[base+i]<0)add=true;
#if LIBERO_NEWTON_SCALAR_IO
        if(add){t0+=q[0];t1+=q[1];t2+=q[2];}
#else
        if(add){
            // Load only an admitted record, preserving skipped-term behavior.
            const real2_t q01=libero_vec::load_pair(q);
            t0+=q01.x;t1+=q01.y;t2+=q[2];
        }
#endif // LIBERO_NEWTON_SCALAR_IO
    }
    real_t values[6]={t0,t1,t2,cost,d0,d1};
    const int lane=libero_warp::lane(),warp=libero_warp::warp();
    #pragma unroll
    for(int k=0;k<6;++k){
        const real_t value=libero_warp::sum_to_leader(values[k]);
        if(lane==0)partial[k][warp]=value;
    }
    __syncthreads();
    if(warp==0){
        #pragma unroll
        for(int k=0;k<6;++k){
            const real_t value=libero_warp::sum_to_leader(
                lane<(blockDim.x+31)/32?partial[k][lane]:REAL_C(0.0));
            if(lane==0)values[k]=value;
        }
        if(lane==0){
            t0=a.quad0[w]+values[0];
            t1=a.quadGauss[w*3+1]+values[1];
            t2=a.quadGauss[w*3+2]+values[2];
            cost=values[3]+(alpha*alpha*t2+alpha*t1+t0);
            d0=values[4]+(REAL_C(2.0)*alpha*t2+t1);d1=values[5]+REAL_C(2.0)*t2;
            if(d1<=0)d1=REAL_C(1e-15);
            if(!isfinite(alpha)||!isfinite(cost)||!isfinite(d0)||!isfinite(d1))a.status[w]=5;
            point={alpha,cost,d0,d1};
            const int idx=*niter;
            a.eval_trace[(w*52+idx)*4]=alpha;
            a.eval_trace[(w*52+idx)*4+1]=cost;
            a.eval_trace[(w*52+idx)*4+2]=d0;
            a.eval_trace[(w*52+idx)*4+3]=d1;
            a.eval_tag[w*52+idx]=tag;next_iter=idx+1;
        }
    }
    __syncthreads();
    P result=point;*niter=next_iter;
    __syncthreads();return result;
}

__device__ int s07_update_bracket_block(const S07Args& a,int w,int rows,int nefc,
    int ne,int nf,P* p,const P* candidates,P* next,int* niter,int tag,
    const int* group_head,const int* compact_heads=nullptr,int head_count=0,const real_t* head_mu=nullptr) {
    int flag=0;
    for(int i=0;i<3;++i){
        if(p->d0<0&&candidates[i].d0<0&&p->d0<candidates[i].d0){*p=candidates[i];flag=1;}
        else if(p->d0>0&&candidates[i].d0>0&&p->d0>candidates[i].d0){*p=candidates[i];flag=2;}
    }
    // p and candidates are broadcasts of leader evaluations, so this branch is
    // uniform. Trace order and the 50-evaluation limit remain source-ordered.
    if(flag)*next=s07_eval_point_block(a,w,rows,nefc,ne,nf,
        p->alpha-real_div_rn(p->d0,p->d1),tag,niter,group_head,compact_heads,head_count,head_mu);
    return flag;
}

__device__ void s07_finish_block(const S07Args& a,int w,real_t alpha,int niter,
    int result,real_t slope){
    if(threadIdx.x==0)finish(a,w,alpha,niter,result,slope);
}
