#include "precision.cuh"
#pragma once
#include <stdint.h>
#include <math.h>
#include "../common/vector.cuh"
namespace collision_cuda {
__device__ void cf_input_error(int w,int* active,int* stage,int* code,int* item) {
    if(!code[w] && active[w]!=0 && active[w]!=1){stage[w]=1;code[w]=1;item[w]=w;}
}
__device__ void cf_effective(int w,int* active,int* code,int* effective) {
    effective[w]=code[w] ? 0:active[w];
}
__device__ void cf_reset_error(int w,int* reset,int* stage,int* code,int* item) {
    if(reset[w]==1){stage[w]=0;code[w]=0;item[w]=-1;}
}
__device__ void cf_stage_error(int w,int* status,int* active,int stage_id,int a,int b,int c,
        int* stage,int* code,int* item,int* effective) {
    if(active[w]==1 && !code[w] && status[w]){
        stage[w]=stage_id; const int s=status[w]; code[w]=(s==a||s==b||s==c)?3:9;item[w]=0;
    }
    effective[w]=code[w]?0:active[w];
}
__device__ void cf_detect_error(int w,int* status,int* det_item,int* active,
        int* stage,int* code,int* item,int* effective) {
    if(active[w]==1 && !code[w] && status[w]){stage[w]=6;code[w]=status[w];item[w]=det_item[w];}
    effective[w]=code[w]?0:active[w];
}
template<class T> __device__ void cf_copy(int w,T* src,T* dst,int stride){
    if(stride>0)libero_vec::copy_block(src+w*stride,dst+w*stride,(size_t)stride);
}
__device__ void cf_copy_i32(int w,int* s,int* d,int n){cf_copy(w,s,d,n);}
__device__ void cf_copy_f64(int w,real_t* s,real_t* d,int n){cf_copy(w,s,d,n);}
__device__ void cf_copy_f32(int w,float* s,float* d,int n){cf_copy(w,s,d,n);}
template<class T> __device__ void cf_restore(int w,int* active,T* src,T* dst,int stride){
    if(active[w]==0)cf_copy(w,src,dst,stride);
}
__device__ void cf_restore_i32(int w,int* a,int* s,int* d,int n){cf_restore(w,a,s,d,n);}
__device__ void cf_restore_f64(int w,int* a,real_t* s,real_t* d,int n){cf_restore(w,a,s,d,n);}
__device__ void cf_restore_f32(int w,int* a,float* s,float* d,int n){cf_restore(w,a,s,d,n);}
__device__ void cf_req_dispatch(int w,int* requests,int* count,int* active,int* raw,real_t* usermargin,
        int* counts,int* dispatch_active,int* origin,int* segments,int* pair_start,int* pair_end,
        int* geom_stage,int* trailing_start,int* trailing_end,int* pair1,int* pair2,
        int req_slots,int slots,int npair,int stage_slots){
    dispatch_active[w]=active[w];counts[w]=0;
    if(!active[w])return;
    const int rc=count[w];if(rc<0||rc>req_slots){counts[w]=-1;return;}
    int n=0;
    for(int s=0;s<segments[w];++s){
        int ix=w*stage_slots+s;
        for(int p=pair_start[ix];p<pair_end[ix];++p){
            if(n>=slots){counts[w]=-2;return;}
            int q=w*slots+n;raw[3*q]=p;raw[3*q+1]=-1;raw[3*q+2]=0;
            usermargin[q]=0;origin[2*q]=-n-1;origin[2*q+1]=-1;++n;
        }
        int gs=geom_stage[ix];if(gs<0)continue;
        for(int r=0;r<rc;++r){
            int ri=(w*req_slots+r)*4;if(requests[ri]!=gs)continue;
            if(n>=slots){counts[w]=-2;return;}
            int q=w*slots+n;raw[3*q]=requests[ri+1];raw[3*q+1]=requests[ri+2];raw[3*q+2]=0;
            usermargin[q]=0;origin[2*q]=gs;origin[2*q+1]=r;++n;
        }
    }
    for(int p=trailing_start[w];p<trailing_end[w];++p){
        if(n>=slots){counts[w]=-2;return;}
        int q=w*slots+n;raw[3*q]=p;raw[3*q+1]=-1;raw[3*q+2]=0;
        usermargin[q]=0;origin[2*q]=-n-1;origin[2*q+1]=-1;++n;
    }
    counts[w]=n;
}

// Tree traversal emits requests in nondecreasing stage order. Preserve a
// fallback for unusual standalone inputs instead of changing their domain.
__device__ void cf_req_dispatch_ordered(int w,int* requests,int* count,int* active,int* raw,real_t* usermargin,
        int* counts,int* dispatch_active,int* origin,int* segments,int* pair_start,int* pair_end,
        int* geom_stage,int* trailing_start,int* trailing_end,int* pair1,int* pair2,
        int req_slots,int slots,int npair,int stage_slots){
    const int rc=count[w];
    bool ordered=rc>=0&&rc<=req_slots;
    if(ordered)for(int r=1;r<rc;++r)
        if(requests[(w*req_slots+r)*4]<requests[(w*req_slots+r-1)*4]){ordered=false;break;}
    // The cursor is valid only for the chain's strictly increasing geometry
    // stages. A standalone caller may replay or reorder a stage.
    int previous_stage=-1;
    if(ordered)for(int s=0;s<segments[w];++s){
        const int gs=geom_stage[w*stage_slots+s];if(gs<0)continue;
        if(gs<=previous_stage){ordered=false;break;}previous_stage=gs;
    }
    if(!ordered){
        cf_req_dispatch(w,requests,count,active,raw,usermargin,counts,dispatch_active,origin,
            segments,pair_start,pair_end,geom_stage,trailing_start,trailing_end,pair1,pair2,
            req_slots,slots,npair,stage_slots);return;
    }
    dispatch_active[w]=active[w];counts[w]=0;if(!active[w])return;
    int n=0,r=0;
    for(int s=0;s<segments[w];++s){
        const int ix=w*stage_slots+s;
        for(int p=pair_start[ix];p<pair_end[ix];++p){
            if(n>=slots){counts[w]=-2;return;}
            const int q=w*slots+n;raw[3*q]=p;raw[3*q+1]=-1;raw[3*q+2]=0;
            usermargin[q]=0;origin[2*q]=-n-1;origin[2*q+1]=-1;++n;
        }
        const int gs=geom_stage[ix];if(gs<0)continue;
        while(r<rc&&requests[(w*req_slots+r)*4]<gs)++r;
        while(r<rc&&requests[(w*req_slots+r)*4]==gs){
            if(n>=slots){counts[w]=-2;return;}
            const int ri=(w*req_slots+r)*4,q=w*slots+n;
            raw[3*q]=requests[ri+1];raw[3*q+1]=requests[ri+2];raw[3*q+2]=0;
            usermargin[q]=0;origin[2*q]=gs;origin[2*q+1]=r;++n;++r;
        }
    }
    for(int p=trailing_start[w];p<trailing_end[w];++p){
        if(n>=slots){counts[w]=-2;return;}
        const int q=w*slots+n;raw[3*q]=p;raw[3*q+1]=-1;raw[3*q+2]=0;
        usermargin[q]=0;origin[2*q]=-n-1;origin[2*q+1]=-1;++n;
    }
    counts[w]=n;
}
// Complete-warp materialization of a contiguous request tile. Register rows
// are transposed only at the compatible AoS write boundary; no mirror is kept.
// All lanes execute every shuffle, including a partial final tile.
__device__ __forceinline__ void cf_dispatch_tile(int w,int slots,int first,int valid,
        int g1,int g2,int o0,int o1,int* raw,real_t* margin,int* origin){
    const int lane=threadIdx.x&31;
    for(int tile=0;tile<3;++tile){
        const int x=32*tile+lane,row=x/3,field=x%3;
        const int a=__shfl_sync(0xffffffffu,g1,row);
        const int b=__shfl_sync(0xffffffffu,g2,row);
        if(x<3*valid)raw[3*(w*slots+first)+x]=field==0?a:field==1?b:0;
    }
    for(int tile=0;tile<2;++tile){
        const int x=32*tile+lane,row=x/2,field=x&1;
        const int a=__shfl_sync(0xffffffffu,o0,row);
        const int b=__shfl_sync(0xffffffffu,o1,row);
        if(x<2*valid)origin[2*(w*slots+first)+x]=field?b:a;
    }
    if(lane<valid)margin[w*slots+first+lane]=0;
}
// Collective entry: one complete warp generates ordered segments; every block
// thread participates in the validation and final publication barrier.
__device__ void cf_req_dispatch_ordered_block(int w,int* requests,int* count,int* active,int* raw,real_t* usermargin,
        int* counts,int* dispatch_active,int* origin,int* segments,int* pair_start,int* pair_end,
        int* geom_stage,int* trailing_start,int* trailing_end,int* pair1,int* pair2,
        int req_slots,int slots,int npair,int stage_slots){
    __shared__ int req_order_bad[32],stage_order_bad;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,rc=count[w];
    // Retain the scalar domain for uncommon launch shapes and malformed counts.
    if(blockDim.x<32 || (blockDim.x&31) || rc<0 || rc>req_slots || slots<0){
        if(threadIdx.x==0)cf_req_dispatch_ordered(w,requests,count,active,raw,usermargin,counts,
            dispatch_active,origin,segments,pair_start,pair_end,geom_stage,trailing_start,
            trailing_end,pair1,pair2,req_slots,slots,npair,stage_slots);
        __syncthreads();return;
    }
    bool bad=false;
    for(int r=threadIdx.x+1;r<rc;r+=blockDim.x)
        bad|=requests[(w*req_slots+r)*4]<requests[(w*req_slots+r-1)*4];
    const int wb=__any_sync(0xffffffffu,bad);
    if(lane==0)req_order_bad[warp]=wb;
    if(threadIdx.x==0){
        int previous=-1;stage_order_bad=0;
        for(int s=0;s<segments[w];++s){
            const int gs=geom_stage[w*stage_slots+s];if(gs<0)continue;
            if(gs<=previous){stage_order_bad=1;break;}previous=gs;
        }
    }
    __syncthreads();
    bool ordered=!stage_order_bad;
    for(int k=0;k<(blockDim.x>>5);++k)ordered&=!req_order_bad[k];
    if(!ordered){
        if(threadIdx.x==0)cf_req_dispatch(w,requests,count,active,raw,usermargin,counts,
            dispatch_active,origin,segments,pair_start,pair_end,geom_stage,trailing_start,
            trailing_end,pair1,pair2,req_slots,slots,npair,stage_slots);
        __syncthreads();return;
    }
    if(threadIdx.x==0){dispatch_active[w]=active[w];counts[w]=0;}
    if(active[w] && warp==0){
        long long base=0;
        const long long ceiling=(long long)slots+1;
        for(int tile=0;tile<segments[w];tile+=32){
            const int s=tile+lane,ix=w*stage_slots+s;
            int p0=0,pn=0,r0=0,rn=0,gs=-1;
            if(s<segments[w]){
                p0=pair_start[ix];
                pn=pair_end[ix]>p0?pair_end[ix]-p0:0;
                gs=geom_stage[ix];
                if(gs>=0){
                    int lo=0,hi=rc;
                    while(lo<hi){int m=lo+(hi-lo)/2;if(requests[(w*req_slots+m)*4]<gs)lo=m+1;else hi=m;}
                    r0=lo;hi=rc;
                    while(lo<hi){int m=lo+(hi-lo)/2;if(requests[(w*req_slots+m)*4]<=gs)lo=m+1;else hi=m;}
                    rn=lo-r0;
                }
            }
            const long long length=(long long)pn+rn;
            long long inclusive=length;
            for(int d=1;d<32;d<<=1){
                const long long x=__shfl_up_sync(0xffffffffu,inclusive,d);
                if(lane>=d)inclusive+=x;
            }
            const long long offset=base+inclusive-length;
            const long long tile_length=__shfl_sync(0xffffffffu,inclusive,31);
            // Avoid one long segment monopolizing one lane: coalesced output
            // lanes find their segment with a fixed five-step prefix search.
            // Every lane enters every shuffle, including padded output lanes.
            if(__any_sync(0xffffffffu,length>64)){
                const long long writable=base<slots?
                    (tile_length<(long long)slots-base?tile_length:(long long)slots-base):0;
                for(long long output_tile=0;output_tile<writable;output_tile+=32){
                    const long long j=output_tile+lane;
                    const bool valid=j<writable;
                    int before=-1;
                    for(int step=16;step;step>>=1){
                        const int probe=before+step;
                        const long long boundary=__shfl_sync(0xffffffffu,inclusive,probe);
                        if(probe<32 && boundary<=j)before=probe;
                    }
                    const int owner=before+1<32?before+1:31;
                    const long long owner_end=__shfl_sync(0xffffffffu,inclusive,owner);
                    const long long owner_length=__shfl_sync(0xffffffffu,length,owner);
                    const long long local=j-(owner_end-owner_length);
                    const int op0=__shfl_sync(0xffffffffu,p0,owner);
                    const int opn=__shfl_sync(0xffffffffu,pn,owner);
                    const int or0=__shfl_sync(0xffffffffu,r0,owner);
                    const int ogs=__shfl_sync(0xffffffffu,gs,owner);
                    int eg1=0,eg2=-1,eo0=0,eo1=-1;
                    if(valid){
                        const int n=(int)(base+j);
                        if(local<opn){eg1=op0+(int)local;eo0=-n-1;}
                        else{const int r=or0+(int)(local-opn),ri=(w*req_slots+r)*4;
                            eg1=requests[ri+1];eg2=requests[ri+2];eo0=ogs;eo1=r;}
                    }
                    const int available=(int)(writable-output_tile<32?writable-output_tile:32);
                    cf_dispatch_tile(w,slots,(int)(base+output_tile),available,
                        eg1,eg2,eo0,eo1,raw,usermargin,origin);
                }
            }else{
                // Small segments retain the inexpensive lane-owned path.
                for(long long j=0;j<length && offset+j<slots;++j){
                    const int n=(int)(offset+j),q=w*slots+n;
                    if(j<pn){raw[3*q]=p0+(int)j;raw[3*q+1]=-1;origin[2*q]=-n-1;origin[2*q+1]=-1;}
                    else{const int r=r0+(int)(j-pn),ri=(w*req_slots+r)*4;
                        raw[3*q]=requests[ri+1];raw[3*q+1]=requests[ri+2];origin[2*q]=gs;origin[2*q+1]=r;}
                    raw[3*q+2]=0;usermargin[q]=0;
                }
            }
            base+=tile_length;
            if(base>slots){base=ceiling;break;}
        }
        const long long tail=trailing_end[w]>trailing_start[w]?
            (long long)trailing_end[w]-trailing_start[w]:0;
        const long long tail_writable=base<slots?
            (tail<(long long)slots-base?tail:(long long)slots-base):0;
        for(long long j=0;j<tail_writable;j+=32){
            const int n=(int)(base+j+lane);
            const int available=(int)(tail_writable-j<32?tail_writable-j:32);
            cf_dispatch_tile(w,slots,(int)(base+j),available,
                trailing_start[w]+(int)j+lane,-1,-n-1,-1,raw,usermargin,origin);
        }
        if(lane==0)counts[w]=base+tail>slots?-2:(int)(base+tail);
    }
    __syncthreads();
}
__device__ void cf_dispatch_calls(int w,int* out_i,real_t* out_f,int* index,int* count,int* active,
        int* status,int* origin,int* route,int stage_slots,int* g1,int* g2,int* t1,int* t2,
        real_t* margin,int* ipair,int* user,int* segment,int* sort,int* route_slot,int* ccount,
        int pairs,int slots){
    ccount[w]=0;if(!active[w])return;
    int n=count[w];if(status[w]||n<0||n>pairs){ccount[w]=-1;return;}ccount[w]=n;
    for(int j=0;j<n;++j){
        int q=index[w*pairs+j], b=(w*slots+q)*12, c=w*pairs+j;
        g1[c]=out_i[b+2];g2[c]=out_i[b+3];t1[c]=out_i[b+4];t2[c]=out_i[b+5];
        margin[c]=out_f[(w*slots+q)*2];ipair[c]=out_i[b+7];user[c]=0;route_slot[c]=q;
        int seg=origin[(w*slots+q)*2];segment[c]=seg;
        sort[c]=(seg>=0 && seg<stage_slots && route[w*stage_slots+seg]==1)?1:0;
    }
}

struct CandidatePair {
    int geom1,geom2,type1,type2,pair,segment,sort,route_slot;
    real_t margin;
};

__device__ void cf_dispatch_calls_block(int w,int* out_i,real_t* out_f,int* index,int* count,int* active,
        int* status,int* origin,int* route,int stage_slots,int* g1,int* g2,int* t1,int* t2,
        real_t* margin,int* ipair,int* user,int* segment,int* sort,int* route_slot,int* ccount,
        int pairs,int slots){
    const int n=count[w];
    if(threadIdx.x==0)ccount[w]=!active[w]?0:(status[w]||n<0||n>pairs?-1:n);
    __syncthreads();
    if(!active[w]||status[w]||n<0||n>pairs)return;
    for(int j=threadIdx.x;j<n;j+=blockDim.x){
        const int q=index[w*pairs+j],b=(w*slots+q)*12,c=w*pairs+j;
        CandidatePair p={out_i[b+2],out_i[b+3],out_i[b+4],out_i[b+5],out_i[b+7],
                         origin[(w*slots+q)*2],0,q,out_f[(w*slots+q)*2]};
        p.sort=p.segment>=0&&p.segment<stage_slots&&route[w*stage_slots+p.segment]==1?1:0;
        g1[c]=p.geom1;g2[c]=p.geom2;t1[c]=p.type1;t2[c]=p.type2;margin[c]=p.margin;
        ipair[c]=p.pair;user[c]=0;segment[c]=p.segment;sort[c]=p.sort;route_slot[c]=p.route_slot;
    }
}
__device__ void cf_body_counters(int w,int* count,int* segment,int* ipair,int* raw_count,int* active,
        int* mode,int* gates,int* broad,int* narrow,int pairs,int npair,int nbody){
    broad[w]=narrow[w]=0;
    if(active[w]!=1||gates[0]||gates[1]||!gates[2]||nbody<2)return;
    int n=count[w];if(n<0||n>pairs)return;
    if(mode[0]==1)broad[w]=npair;
    for(int j=0;j<n;++j){
        int q=w*pairs+j;
        if(ipair[q]>=0){if(mode[0]==1){++broad[w];if(raw_count[q]>0)++narrow[w];}continue;}
        bool first=true;for(int k=0;k<j;++k)if(ipair[w*pairs+k]<0 && segment[w*pairs+k]==segment[q])first=false;
        if(!first)continue;
        ++broad[w];bool entered=false;
        for(int k=j;k<n;++k)if(ipair[w*pairs+k]<0 && segment[w*pairs+k]==segment[q] && raw_count[w*pairs+k]>0)entered=true;
        if(entered)++narrow[w];
    }
}
// Normal tree output has monotonically increasing non-explicit segment IDs.
// Prove that property before using adjacent groups; standalone permutations
// retain the scalar unique-segment semantics.
__device__ void cf_body_counters_block(int w,int* count,int* segment,int* ipair,int* raw_count,int* active,
        int* mode,int* gates,int* broad,int* narrow,int pairs,int npair,int nbody){
    __shared__ int ordered,warp_broad[32],warp_narrow[32];
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5,n=count[w];
    if(threadIdx.x==0){
        broad[w]=narrow[w]=0;ordered=1;
        if(n>=0&&n<=pairs){
            int previous=0;bool seen=false;
            for(int j=0;j<n;++j)if(ipair[w*pairs+j]<0){
                int s=segment[w*pairs+j];if(seen&&s<previous){ordered=0;break;}
                previous=s;seen=true;
            }
        }
    }
    __syncthreads();
    if(active[w]!=1||gates[0]||gates[1]||!gates[2]||nbody<2||n<0||n>pairs)return;
    if(!ordered || blockDim.x<32 || (blockDim.x&31)){
        if(threadIdx.x==0)cf_body_counters(w,count,segment,ipair,raw_count,active,mode,gates,
            broad,narrow,pairs,npair,nbody);
        __syncthreads();return;
    }
    int b=0,t=0;
    for(int j=threadIdx.x;j<n;j+=blockDim.x){
        const int q=w*pairs+j;
        if(ipair[q]>=0){if(mode[0]==1){++b;t+=raw_count[q]>0;}continue;}
        int previous=j-1;while(previous>=0 && ipair[w*pairs+previous]>=0)--previous;
        if(previous>=0 && segment[w*pairs+previous]==segment[q])continue;
        ++b;bool entered=false;
        for(int k=j;k<n;++k){
            const int x=w*pairs+k;if(ipair[x]>=0)continue;
            if(segment[x]!=segment[q])break;
            entered|=raw_count[x]>0;
        }
        t+=entered;
    }
    for(int d=16;d;d>>=1){b+=__shfl_down_sync(0xffffffffu,b,d);t+=__shfl_down_sync(0xffffffffu,t,d);}
    if(lane==0){warp_broad[warp]=b;warp_narrow[warp]=t;}
    __syncthreads();
    if(warp==0){
        b=lane<(blockDim.x>>5)?warp_broad[lane]:0;
        t=lane<(blockDim.x>>5)?warp_narrow[lane]:0;
        for(int d=16;d;d>>=1){b+=__shfl_down_sync(0xffffffffu,b,d);t+=__shfl_down_sync(0xffffffffu,t,d);}
        if(lane==0){broad[w]=b+(mode[0]==1?npair:0);narrow[w]=t;}
    }
    __syncthreads();
}
__device__ void cf_chain_error(int w,int* bounds,int* broad,int* build,int* tree,int* dispatch,
        int* detect,int* detect_item,int* status,int* ncon,int* warnings,int capacity,
        int* active,int* stage,int* code,int* item){
    if(code[w]){status[w]=code[w];ncon[w]=0;return;}
    if(!active[w])return;
    if(active[w]!=1){stage[w]=1;code[w]=status[w]=1;item[w]=w;return;}
    int s=0,c=0;
    if(bounds[w]){s=1;c=9;}
    else if(broad[w]){s=2;c=broad[w]==1?3:9;}
    else if(build[w]){s=3;c=build[w]==3?3:9;}
    else if(tree[w]){s=4;c=(tree[w]==5||tree[w]==6||tree[w]==9)?3:9;}
    else if(dispatch[w]){s=5;c=dispatch[w]==4?3:9;}
    else if(detect[w]){s=6;c=detect[w];}
    else if(status[w]){s=7;c=8;}
    else if(warnings[w]||ncon[w]>capacity){s=7;c=3;}
    if(s){stage[w]=s;code[w]=c;item[w]=s==6?detect_item[w]:0;ncon[w]=0;
        if(s!=7 || c==3)status[w]=c;}
}
__device__ void cf_restore_contacts(int w,int* active,int* code,real_t* before_f,int* before_i,
        real_t* out_f,int* out_i,int capacity){
    if(!active[w]||code[w]){
        cf_copy(w,before_f,out_f,capacity*65);cf_copy(w,before_i,out_i,capacity*5);
    }
}
}
