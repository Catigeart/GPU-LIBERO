#pragma once

// Warp 0 computes up to three independent dot products, one eight-lane group
// per product. Four lanes retain mju_dot's ordered accumulator streams; lane 4
// handles its 1/2/3-element tail. Only the group leader writes the result.
// No shared storage or barrier: the caller publishes the outputs once.
template<int Products>
static __device__ void search_dots_warp(const double* x,const double* y0,
    const double* y1,const double* y2,int n,double* out) {
    static_assert(Products>=1&&Products<=3,"at most three dot products");
    if(threadIdx.x>=32)return;
    const int group=threadIdx.x/8,lane=threadIdx.x%8;
    const int end=n-n%4;
    double partial=0.0;
    if(group<Products) {
        const double* y=group==0?y0:(group==1?y1:y2);
        if(lane<4&&end) {
            partial=x[lane]*y[lane];
            for(int i=lane+4;i<end;i+=4)partial+=x[i]*y[i];
        } else if(lane==4) {
            const int tail=n-end;
            if(tail==3)partial=x[end]*y[end]+x[end+1]*y[end+1]+x[end+2]*y[end+2];
            else if(tail==2)partial=x[end]*y[end]+x[end+1]*y[end+1];
            else if(tail==1)partial=x[end]*y[end];
        }
    }
    const double r0=__shfl_sync(0xffffffffu,partial,0,8);
    const double r1=__shfl_sync(0xffffffffu,partial,1,8);
    const double r2=__shfl_sync(0xffffffffu,partial,2,8);
    const double r3=__shfl_sync(0xffffffffu,partial,3,8);
    const double tail=__shfl_sync(0xffffffffu,partial,4,8);
    if(lane==0&&group<Products) {
        double value=(r0+r2)+(r1+r3);
        if(n!=end)value+=tail;
        out[group]=value;
    }
}
