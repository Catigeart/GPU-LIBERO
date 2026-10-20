#include "precision.cuh"
#pragma once

// Exact coordinate grouping is a combinatorial operation: no tolerance, hash
// collision, or floating-point arithmetic is introduced by this helper.
#ifndef LIBERO_CF_DEDUP_SORT_MIN_RECORDS
#define LIBERO_CF_DEDUP_SORT_MIN_RECORDS 16
#endif

namespace collision_cuda {
__device__ __forceinline__ bool cf_raw_position_less(const real_t* raw,int a,int b){
    for(int k=1;k<=3;++k){
        const real_t x=raw[a*10+k],y=raw[b*10+k];
        if(x<y)return true;
        if(y<x)return false;
    }
    // Finite inputs only. Numeric equality deliberately identifies +0/-0.
    return a<b;
}
__device__ __forceinline__ bool cf_raw_position_equal(const real_t* raw,int a,int b){
    return raw[a*10+1]==raw[b*10+1]&&raw[a*10+2]==raw[b*10+2]&&
           raw[a*10+3]==raw[b*10+3];
}
__device__ __forceinline__ void cf_raw_heap_sift(const real_t* raw,int* order,int root,int end){
    while(root*2+1<end){
        int child=root*2+1;
        if(child+1<end&&cf_raw_position_less(raw,order[child],order[child+1]))++child;
        if(!cf_raw_position_less(raw,order[root],order[child]))return;
        const int t=order[root];order[root]=order[child];order[child]=t;root=child;
    }
}
// Input has already passed the original validation of all 10 fields. Output
// ordinals remain increasing; the greatest ordinal of each xyz group survives.
__device__ __forceinline__ int cf_raw_keep_last(const real_t* raw,int nr,bool dedup,int* kept){
    if(!dedup){for(int r=0;r<nr;++r)kept[r]=r;return nr;}
    if(LIBERO_CF_DEDUP_SORT_MIN_RECORDS<=0||nr<LIBERO_CF_DEDUP_SORT_MIN_RECORDS){
        int out=0;
        for(int r=0;r<nr;++r){
            bool discard=false;
            for(int t=r+1;t<nr;++t)if(cf_raw_position_equal(raw,r,t)){discard=true;break;}
            if(!discard)kept[out++]=r;
        }
        return out;
    }
    // Completely coincident rows were already linear in the old early-exit
    // predicate. Preserve that favorable case before paying sorting work.
    bool all_same=true;
    for(int r=1;r<nr&&all_same;++r)all_same=cf_raw_position_equal(raw,0,r);
    if(all_same){kept[0]=nr-1;return 1;}
    int order[50];
    for(int r=0;r<nr;++r){order[r]=r;kept[r]=0;}
    for(int root=nr/2-1;root>=0;--root)cf_raw_heap_sift(raw,order,root,nr);
    for(int end=nr-1;end>0;--end){
        const int t=order[0];order[0]=order[end];order[end]=t;
        cf_raw_heap_sift(raw,order,0,end);
    }
    for(int j=0;j<nr;++j)
        if(j+1==nr||!cf_raw_position_equal(raw,order[j],order[j+1]))kept[order[j]]=1;
    // out<=r, so compaction cannot overwrite an unread membership flag.
    int out=0;for(int r=0;r<nr;++r)if(kept[r])kept[out++]=r;
    return out;
}
} // namespace collision_cuda
