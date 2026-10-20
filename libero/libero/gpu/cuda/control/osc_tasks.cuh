#include "precision.cuh"
#pragma once
#include "osc.cuh"
#include "osc_pinv_observed.cuh"
extern "C" __device__ void osc_task_inverse6(real_t*,real_t*,int*,int*);
extern "C" __device__ void osc_task_inverse3(real_t*,real_t*,int*,int*);
#ifndef OSC_PINV_PARALLEL
#define OSC_PINV_PARALLEL 0
#endif
#ifndef OSC_TASK_SOLVE
#define OSC_TASK_SOLVE 0
#endif
#ifndef OSC_TASK_TELEMETRY
#define OSC_TASK_TELEMETRY 0
#endif
#ifndef OSC_TASK_CONDITION_MAX
#define OSC_TASK_CONDITION_MAX REAL_C(1000000.0)
#endif
namespace osc_cuda {
// Only the explicit operator profile uses this optional full-rank route.
// A residual proves an inverse exists; the conservative condition certificate
// stays far below the retained singular-value cutoff. Any failure uses pinv.
__device__ inline void certified_task_inverse(const real_t* a,int n,real_t* out,
                                             int& rank,int& ok,int& accepted,int* sweeps=nullptr){
    __shared__ real_t scaled[36],factor[36],inverse[36],scale;
    __shared__ int pivots[6],info,eligible;
    if(threadIdx.x==0){
        bool input_finite=true;
        scale=0;for(int r=0;r<n;r++)for(int c=0;c<n;c++){
            input_finite=input_finite&&isfinite(a[r*6+c]);scale=fmax(scale,fabs(a[r*6+c]));}
        eligible=input_finite&&scale>0&&isfinite(scale);accepted=0;info=0;if(sweeps)*sweeps=0;
        for(int z=0;z<n*n;z++){int r=z/n,c=z%n;scaled[z]=factor[z]=eligible?a[r*6+c]/scale:0;inverse[z]=r==c?1:0;}
    }__syncthreads();
    if(eligible){if(n==6)osc_task_inverse6(factor,inverse,pivots,&info);else osc_task_inverse3(factor,inverse,pivots,&info);}
    if(threadIdx.x==0&&eligible&&info==0){
        real_t an1=0,ani=0,bn1=0,bni=0,rn1=0,rni=0,round1=0,roundi=0;
        const real_t u=REAL_C(1.11022302462515654042e-16);
        const real_t gamma=(2*n+4)*u/(1-(2*n+4)*u);
        bool finite=true;
        for(int r=0;r<n;r++){
            real_t ar=0,ac=0,br=0,bc=0,rr=0,rc=0,er=0,ec=0;
            for(int c=0;c<n;c++){
                ar+=fabs(scaled[r*n+c]);ac+=fabs(scaled[c*n+r]);
                br+=fabs(inverse[r*n+c]);bc+=fabs(inverse[c*n+r]);
                real_t row=0,col=0,absrow=0,abscol=0;
                for(int k=0;k<n;k++){
                    real_t x=scaled[r*n+k]*inverse[k*n+c],y=scaled[c*n+k]*inverse[k*n+r];
                    row+=x;col+=y;absrow+=fabs(x);abscol+=fabs(y);
                }
                real_t delta=r==c?1:0;
                real_t row_error=gamma*(delta+absrow),col_error=gamma*(delta+abscol);
                rr+=fabs(delta-row)+row_error;rc+=fabs(delta-col)+col_error;
                er+=row_error;ec+=col_error;
                finite=finite&&isfinite(inverse[r*n+c])&&isfinite(inverse[r*n+c]/scale);
            }
            an1=fmax(an1,ac);ani=fmax(ani,ar);bn1=fmax(bn1,bc);bni=fmax(bni,br);
            rn1=fmax(rn1,rc);rni=fmax(rni,rr);round1=fmax(round1,ec);roundi=fmax(roundi,er);
            finite=finite&&isfinite(ar)&&isfinite(ac)&&isfinite(br)&&isfinite(bc)&&
                isfinite(rr)&&isfinite(rc)&&isfinite(er)&&isfinite(ec);
        }
        // A*B near I gives ||A^-1|| <= ||B||/(1-||I-A*B||).
        // sqrt(kappa_1*kappa_inf) bounds kappa_2, preserving full rank.
        const real_t sum_upper=1+gamma;
        rn1*=sum_upper;rni*=sum_upper;
        const real_t lu_gamma=(3*n+5)*u/(1-(3*n+5)*u);
        const real_t budget1=lu_gamma*(an1*bn1*sum_upper*sum_upper+1)+2*round1*sum_upper;
        const real_t budgeti=lu_gamma*(ani*bni*sum_upper*sum_upper+1)+2*roundi*sum_upper;
        const real_t cond1=an1*bn1/(1-rn1)*sum_upper*sum_upper;
        const real_t condi=ani*bni/(1-rni)*sum_upper*sum_upper;
        const real_t cond=sqrt(cond1*condi);
        // The existence certificate alone is insufficient: a finite, large
        // residual must not license an inaccurate inverse. Require the local
        // FP64 residual policy budget as well; every failure invokes pinv below.
        accepted=finite&&isfinite(rn1)&&isfinite(rni)&&rn1>=0&&rni>=0&&
            rn1<REAL_C(0.125)&&rni<REAL_C(0.125)&&isfinite(budget1)&&isfinite(budgeti)&&
            rn1<=budget1&&rni<=budgeti&&isfinite(cond1)&&isfinite(condi)&&isfinite(cond)&&
            cond1<=OSC_TASK_CONDITION_MAX&&condi<=OSC_TASK_CONDITION_MAX&&cond<=OSC_TASK_CONDITION_MAX;
        if(accepted){rank=n;ok=1;for(int z=0;z<36;z++)out[z]=0;
            for(int r=0;r<n;r++)for(int c=0;c<n;c++)out[r*6+c]=inverse[r*n+c]/scale;}
    }__syncthreads();
    if(threadIdx.x==0&&!accepted){if(sweeps)pinv_observed(a,n,out,rank,ok,sweeps);else pinv(a,n,out,rank,ok);}
    __syncthreads();
}
__device__ inline void operator_tasks(int w,real_t* a,real_t* ap,real_t* ar,
    real_t* lf,real_t* lp,real_t* lr,int* rank,int* ok,int* stats){
    if(OSC_TASK_SOLVE){
        __shared__ int accepted[3];
        certified_task_inverse(a,6,lf,rank[0],ok[0],accepted[0],OSC_TASK_TELEMETRY&&stats?stats+w*9+6:nullptr);
        certified_task_inverse(ap,3,lp,rank[1],ok[1],accepted[1],OSC_TASK_TELEMETRY&&stats?stats+w*9+7:nullptr);
        certified_task_inverse(ar,3,lr,rank[2],ok[2],accepted[2],OSC_TASK_TELEMETRY&&stats?stats+w*9+8:nullptr);
        if(threadIdx.x==0&&stats)for(int z=0;z<3;z++){
            if(accepted[z]&&stats[w*9+z]<INT32_MAX)stats[w*9+z]++;
            if(!accepted[z]&&stats[w*9+3+z]<INT32_MAX)stats[w*9+3+z]++;
        }
    }else if(OSC_PINV_PARALLEL){
        // No warp reduction: each task retains the exact serial pair/sweep order.
        // All threads rejoin before any collective library operation.
        if(threadIdx.x==0){if(OSC_TASK_TELEMETRY&&stats)pinv_observed(a,6,lf,rank[0],ok[0],stats+w*9+6);else pinv(a,6,lf,rank[0],ok[0]);}
        if(threadIdx.x==32){
            if(OSC_TASK_TELEMETRY&&stats){pinv_observed(ap,3,lp,rank[1],ok[1],stats+w*9+7);pinv_observed(ar,3,lr,rank[2],ok[2],stats+w*9+8);}
            else{pinv(ap,3,lp,rank[1],ok[1]);pinv(ar,3,lr,rank[2],ok[2]);}
        }
        __syncthreads();
    }else{
        if(threadIdx.x==0){
            if(OSC_TASK_TELEMETRY&&stats){pinv_observed(a,6,lf,rank[0],ok[0],stats+w*9+6);pinv_observed(ap,3,lp,rank[1],ok[1],stats+w*9+7);pinv_observed(ar,3,lr,rank[2],ok[2],stats+w*9+8);}
            else{pinv(a,6,lf,rank[0],ok[0]);pinv(ap,3,lp,rank[1],ok[1]);pinv(ar,3,lr,rank[2],ok[2]);}
        }
        __syncthreads();
    }
}
}
