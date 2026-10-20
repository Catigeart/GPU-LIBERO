#include "precision.cuh"
#pragma once
#include "collision_dedup.cuh"
#include <math.h>
namespace collision_cuda {
// A complete block owns a world; fixed-capacity rows retain source order.
template<bool Diagnostics>
__device__ void cf_calls_finalize_impl(int w,int* g1,int* g2,int* t1,int* t2,int* ipair,int* user,
        int* segment,int* sort,int* call_count,real_t* margin,real_t* raw,int* raw_count,int* det_status,
        real_t* fraw,int* fcount,int* fcall,int* factive,int* ftypes,int* fsegment,int* fsort,
        int* route,real_t* usermargin,int* counts,int* status,int* active,
        int worlds,int pairs,int max_raw,int raw_f){
    // Each warp votes over its strided detector calls. No detector status has
    // a distinct public error code, so an unordered boolean vote preserves
    // the previous first-failure result and leaves inactive rows untouched.
    __shared__ int cf_finalize_failed[32];
    bool failed=false;
    const int n=active[w]?call_count[w]:0;
    if(active[w] && n>=0){
        if(n<=pairs){
            for(int j=threadIdx.x;j<n;j+=blockDim.x)
                failed=failed || det_status[w*pairs+j]!=0;
        }else{
            // A malformed call count is a detector failure. Do not address
            // detector slots beyond the declared allocation, even if an
            // earlier slot might otherwise have supplied the same verdict.
            failed=true;
        }
    }
    // Loop trip counts differ between lanes. An instantaneous activemask can
    // describe only an arriving subset under independent thread scheduling;
    // name the complete physical warp (or final partial warp) explicitly.
    const int warp_threads=min(32,int(blockDim.x)-int(threadIdx.x>>5)*32);
    const unsigned participants=warp_threads==32?0xffffffffu:((1u<<warp_threads)-1u);
    const int warp_failed=__any_sync(participants,failed);
    if((threadIdx.x&31)==0)cf_finalize_failed[threadIdx.x>>5]=warp_failed;
    __syncthreads();
    if(threadIdx.x==0){
        factive[w]=active[w];fcall[w]=status[w]=counts[w]=0;
        if(active[w]){
            if(n<0)status[w]=1;
            else{for(int j=0;j<(blockDim.x+31)/32;++j)if(cf_finalize_failed[j]){factive[w]=0;status[w]=1;break;}
                if(!status[w])counts[w]=pairs*max_raw;}
        }
    }
    __syncthreads();
    if(factive[w] && !status[w] && counts[w]){
        // Copy immutable raw fields with adjacent lanes at adjacent addresses.
        // Route records retain their original ordinal and ownership below.
        const int live_calls=Diagnostics?pairs:n;
        if(fraw!=raw)for(int k=threadIdx.x;k<live_calls*max_raw*raw_f;k+=blockDim.x)
            fraw[w*pairs*max_raw*raw_f+k]=raw[w*pairs*max_raw*raw_f+k];
        // The live consumer reads one descriptor at each call's first row.
        // Keep the public sparse addressing, without expanding it 50 times
        // or writing unused call slots. Compatible diagnostics retain tails.
        for(int q=threadIdx.x;q<live_calls*(Diagnostics?max_raw:1);q+=blockDim.x){
            const int row=Diagnostics?q:q*max_raw;
            int slot=w*pairs+row/max_raw, rr=w*pairs*max_raw+row;
            // Cold retained alias mode uses the detector's public raw owner.
            // It must not turn every bridge row into a read/write self-copy.
            route[rr*4]=g1[slot];route[rr*4+1]=g2[slot];route[rr*4+2]=ipair[slot];route[rr*4+3]=user[slot];
            usermargin[rr]=margin[slot];
            if(row%max_raw==0){fcount[slot]=raw_count[slot];ftypes[slot*2]=t1[slot];ftypes[slot*2+1]=t2[slot];
                fsegment[slot]=segment[slot];fsort[slot]=sort[slot];}
        }
    }
    __syncthreads();
    if(threadIdx.x==0 && factive[w] && !status[w] && counts[w])fcall[w]=call_count[w];
}
template<class... Args>
__device__ __forceinline__ void cf_calls_finalize(Args... args){cf_calls_finalize_impl<true>(args...);}
template<class... Args>
__device__ __forceinline__ void cf_calls_finalize_live(Args... args){cf_calls_finalize_impl<false>(args...);}
__device__ void cf_dedup(int w,real_t* raw,int* raw_count,int* call_count,int* active,int* types,
        int* route,int* geom_type,int ngeom,real_t* compact,int* compact_count,int* keep_index,
        int* status,int pairs){
    for(int local=threadIdx.x;local<pairs;local+=blockDim.x){
        int call=w*pairs+local;compact_count[call]=status[call]=0;
        for(int r=0;r<50;++r){keep_index[call*50+r]=-1;for(int k=0;k<10;++k)compact[(call*50+r)*10+k]=0;}
        if(!active[w])continue;
        if(active[w]!=1){status[call]=1;continue;}
        int n=call_count[w];if(n<0||n>pairs){status[call]=2;continue;}if(local>=n)continue;
        int g1=route[call*50*4],g2=route[call*50*4+1];
        if(g1<0||g2<0||g1>=ngeom||g2>=ngeom){status[call]=5;continue;}
        if(types[call*2]!=geom_type[g1]||types[call*2+1]!=geom_type[g2]){status[call]=6;continue;}
        int nr=raw_count[call];if(nr<0||nr>50){status[call]=3;continue;}
        bool finite=true;for(int r=0;r<nr;++r)for(int k=0;k<10;++k)if(!isfinite(raw[(call*50+r)*10+k]))finite=false;
        if(!finite){status[call]=4;continue;}
        int kept[50];
        const int out=cf_raw_keep_last(raw+call*500,nr,
            types[call*2]==6&&types[call*2+1]==6,kept);
        for(int r=0;r<out;++r){
            keep_index[call*50+r]=kept[r];
            libero_vec::copy_serial(raw+(call*50+kept[r])*10,compact+(call*50+r)*10,10);
        }
        compact_count[call]=out;
    }
}
__device__ bool cf_contact_lt(int ga,int ha,int gb,int hb,int* type){
    int a=ga,b=ha,c=gb,d=hb;
    if(type[a]>type[b]){a=ha;b=ga;}if(type[c]>type[d]){c=hb;d=gb;}
    return a<c || (a==c && b<d);
}
template<int Fields=65>
__device__ void cf_contact_copy(int dst,int src,real_t* sf,int* si,real_t* df,int* di){
    for(int k=0;k+1<Fields;k+=2){const real2_t v=libero_vec::load_pair(sf+src*Fields+k);
        libero_vec::store_pair(df+dst*Fields+k,v.x,v.y);}
    if(Fields&1)df[dst*Fields+Fields-1]=sf[src*Fields+Fields-1];
    const int* s=si+src*5;int* d=di+dst*5;
    int4 v;
    if(libero_vec::aligned16(s))v=*reinterpret_cast<const int4*>(s);
    else{v.x=s[0];v.y=s[1];v.z=s[2];v.w=s[3];}
    if(libero_vec::aligned16(d))*reinterpret_cast<int4*>(d)=v;
    else{d[0]=v.x;d[1]=v.y;d[2]=v.z;d[3]=v.w;}
    d[4]=s[4];
}
template<int Fields=65>
__device__ void cf_contact_sort(int w,int start,int length,int capacity,int* type,
        real_t* cf,int* ci,real_t* sf,int* si,int* order=nullptr,int* order_scratch=nullptr){
    if(order&&order_scratch){
        const int base=w*capacity+start;
        for(int j=0;j<length;++j)order[base+j]=base+j;
        for(int width=1;width<length;width*=2){
            for(int left=0;left<length;left+=2*width){
                int mid=min(left+width,length),end=min(left+2*width,length),a=left,b=mid;
                for(int t=left;t<end;++t){
                    bool right=a>=mid;
                    if(!right&&b<end){int ai=order[base+a]*5,bi=order[base+b]*5;
                        right=cf_contact_lt(ci[bi+1],ci[bi+2],ci[ai+1],ci[ai+2],type);}
                    order_scratch[base+t]=order[base+(right?b++:a++)];
                }
            }
            for(int j=0;j<length;++j)order[base+j]=order_scratch[base+j];
        }
        // Stable key ordering moves only indices until the final permutation.
        for(int j=0;j<length;++j)cf_contact_copy<Fields>(base+j,order[base+j],cf,ci,sf,si);
        for(int j=0;j<length;++j)cf_contact_copy<Fields>(base+j,base+j,sf,si,cf,ci);
        return;
    }
    for(int width=1;width<length;width*=2){
        for(int left=0;left<length;left+=2*width){
            int mid=min(left+width,length),end=min(left+2*width,length),a=left,b=mid;
            for(int t=left;t<end;++t){
                bool right=a>=mid;
                if(!right && b<end){int ai=(w*capacity+start+a)*5,bi=(w*capacity+start+b)*5;
                    right=cf_contact_lt(ci[bi+1],ci[bi+2],ci[ai+1],ci[ai+2],type);}
                int src=right?b++:a++;
                cf_contact_copy<Fields>(w*capacity+start+t,w*capacity+start+src,cf,ci,sf,si);
            }
        }
        for(int j=0;j<length;++j)cf_contact_copy<Fields>(w*capacity+start+j,w*capacity+start+j,sf,si,cf,ci);
    }
}
// All block threads must enter this helper with the same segment. The ordered
// assembly replay still uses cf_contact_sort; switching that lane-0 call to
// this helper without a collective segment protocol would deadlock.
// Units never cross record boundaries; source/destination alignment may differ.
template<int Fields=65>
__device__ void cf_contact_gather_block(int base,int length,const int* permutation,
        const real_t* cf,const int* ci,real_t* sf,int* si){
    for(int unit=threadIdx.x;unit<length*((Fields+1)/2);unit+=blockDim.x){
        const int row=unit/((Fields+1)/2),k=(unit%((Fields+1)/2))*2,src=permutation[row];
        if(k+1<Fields){const real2_t v=libero_vec::load_pair(cf+src*Fields+k);
            libero_vec::store_pair(sf+(base+row)*Fields+k,v.x,v.y);}
        else sf[(base+row)*Fields+Fields-1]=cf[src*Fields+Fields-1];
    }
    for(int row=threadIdx.x;row<length;row+=blockDim.x){
        const int* s=ci+permutation[row]*5;int* d=si+(base+row)*5;int4 v;
        if(libero_vec::aligned16(s))v=*reinterpret_cast<const int4*>(s);
        else{v.x=s[0];v.y=s[1];v.z=s[2];v.w=s[3];}
        if(libero_vec::aligned16(d))*reinterpret_cast<int4*>(d)=v;
        else{d[0]=v.x;d[1]=v.y;d[2]=v.z;d[3]=v.w;}
        d[4]=s[4];
    }
}
template<int Fields=65>
__device__ void cf_contact_sort_block(int w,int start,int length,int capacity,int* type,
        real_t* cf,int* ci,real_t* sf,int* si,int* order=nullptr,int* order_scratch=nullptr){
    if(length<=0)return;
    if(length==1){
        // The indexed scalar path initializes order and its record scratch,
        // even for a singleton, but never writes order_scratch in that case.
        if(order && order_scratch){
            const int base=w*capacity+start;
            if(threadIdx.x==0)order[base]=base;
            libero_vec::copy_block(cf+base*Fields,sf+base*Fields,Fields);
            libero_vec::copy_block(ci+base*5,si+base*5,5);
            __syncthreads();
        }
        return;
    }
    if(blockDim.x<32 || (length>32 && !(order && order_scratch))){
        // The legacy no-index large-segment path has no rank buffer; preserve
        // its existing merge behavior rather than allocate new device storage.
        if(threadIdx.x==0)cf_contact_sort<Fields>(w,start,length,capacity,type,cf,ci,sf,si,order,order_scratch);
        __syncthreads();
        return;
    }
    const int base=w*capacity+start;
    if(length>32){
        for(int j=threadIdx.x;j<length;j+=blockDim.x)order[base+j]=base+j;
        __syncthreads();
        // Stable merge by independent source rank. Left-run records count
        // strictly smaller right keys; right-run records count smaller-or-
        // equal left keys. Equal keys therefore retain original order.
        for(int width=1;width<length;){
            for(int j=threadIdx.x;j<length;j+=blockDim.x){
                const int left=(j/(2*width))*(2*width);
                const int mid=min(left+width,length),end=min(left+2*width,length);
                const bool in_left=j<mid;
                const int src=order[base+j],key=src*5;
                int lo=in_left?mid:left,hi=in_left?end:mid;
                const int other_start=lo;
                while(lo<hi){
                    const int probe=lo+(hi-lo)/2,other=order[base+probe]*5;
                    const bool before=in_left?
                        cf_contact_lt(ci[other+1],ci[other+2],ci[key+1],ci[key+2],type):
                        !cf_contact_lt(ci[key+1],ci[key+2],ci[other+1],ci[other+2],type);
                    if(before)lo=probe+1;else hi=probe;
                }
                const int rank=left+(j-(in_left?left:mid))+(lo-other_start);
                order_scratch[base+rank]=src;
            }
            __syncthreads();
            for(int j=threadIdx.x;j<length;j+=blockDim.x)order[base+j]=order_scratch[base+j];
            __syncthreads();
            if(width>=(length+1)/2)break;
            width*=2;
        }
        cf_contact_gather_block<Fields>(base,length,order+base,cf,ci,sf,si);
        __syncthreads();
        libero_vec::copy_block(sf+base*Fields,cf+base*Fields,size_t(length)*Fields);
        libero_vec::copy_block(si+base*5,ci+base*5,size_t(length)*5);
        __syncthreads();
        return;
    }
    __shared__ int cf_short_permutation[32];
    if(threadIdx.x<32){
        const int lane=threadIdx.x;
        int a=0,b=0;
        if(lane<length){
            a=ci[(base+lane)*5+1];b=ci[(base+lane)*5+2];
            if(type[a]>type[b]){int t=a;a=b;b=t;}
        }
        int rank=0;
        // Every lane participates, including padding. The original ordinal
        // resolves equal keys exactly as the stable left-first merge does.
        for(int j=0;j<length;++j){
            const int other_a=__shfl_sync(0xffffffffu,a,j);
            const int other_b=__shfl_sync(0xffffffffu,b,j);
            if(lane<length && (other_a<a || (other_a==a &&
                    (other_b<b || (other_b==b && j<lane)))))++rank;
        }
        if(lane<length){
            if(order && order_scratch){
                order[base+rank]=base+lane;
                order_scratch[base+rank]=base+lane;
            }
            cf_short_permutation[rank]=base+lane;
        }
    }
    __syncthreads();
    cf_contact_gather_block<Fields>(base,length,cf_short_permutation,cf,ci,sf,si);
    __syncthreads();
    // Flatten record copies: sorting changes neither floating-point bits nor
    // the fields outside this segment.
    libero_vec::copy_block(sf+base*Fields,cf+base*Fields,size_t(length)*Fields);
    libero_vec::copy_block(si+base*5,ci+base*5,size_t(length)*5);
    __syncthreads();
}
__device__ real_t cf_norm3(real_t a,real_t b,real_t c){return sqrt((a*a+b*b)+c*c);}
__device__ void cf_assemble(int w,int* compact_count,int* keep_index,int* call_count,int* active,
        int* segment,int* sort_enabled,int* geom_type,real_t* params_f,int* params_i,int* dedup_status,
        real_t* contact_f,int* contact_i,real_t* scratch_f,int* scratch_i,int* ncon,int* attempts,
        int* appended,int* warnings,int* status,int pairs,int capacity,
        int* order=nullptr,int* order_scratch=nullptr){
    ncon[w]=warnings[w]=status[w]=0;
    for(int p=0;p<pairs;++p)attempts[w*pairs+p]=appended[w*pairs+p]=0;
    if(!active[w])return;if(active[w]!=1){status[w]=1;return;}
    int n=call_count[w];if(n<0||n>pairs){status[w]=2;return;}
    int start=0,current_segment=-1,current_sort=0;
    for(int p=0;p<n;++p){
        int q=w*pairs+p;
        if(dedup_status[q]){if(!status[w])status[w]=10+dedup_status[q];return;}
        int seg=segment[q];
        if(p==0){current_segment=seg;current_sort=sort_enabled[q];}
        if(seg!=current_segment){
            if(current_sort)cf_contact_sort(w,start,ncon[w]-start,capacity,geom_type,contact_f,contact_i,scratch_f,scratch_i,order,order_scratch);
            start=ncon[w];current_segment=seg;current_sort=sort_enabled[q];
        }
        for(int r=0;r<compact_count[q];++r){
            int slot=q*50+r,pf=slot*26;
            if(params_i[slot*5+4]){if(!status[w])status[w]=20+params_i[slot*5+4];return;}
            real_t x0=params_f[pf+4],x1=params_f[pf+5],x2=params_f[pf+6];
            real_t nx=cf_norm3(x0,x1,x2);
            if(nx<REAL_C(0.5)){if(!status[w])status[w]=30;return;}
            if(nx<REAL_C(1e-15)){x0=1;x1=x2=0;}else{real_t inv=1/nx;x0*=inv;x1*=inv;x2*=inv;}
            real_t y0=params_f[pf+7],y1=params_f[pf+8],y2=params_f[pf+9];
            if(cf_norm3(y0,y1,y2)<REAL_C(0.5)){y0=y1=y2=0;if(x1<REAL_C(0.5)&&x1>-REAL_C(0.5))y1=1;else y2=1;}
            real_t dot=(x0*y0+x1*y1)+x2*y2;y0-=x0*dot;y1-=x1*dot;y2-=x2*dot;
            real_t ny=cf_norm3(y0,y1,y2);
            if(ny<REAL_C(1e-15)){y0=1;y1=y2=0;}else{real_t inv=1/ny;y0*=inv;y1*=inv;y2*=inv;}
            real_t z0=x1*y2-x2*y1,z1=x2*y0-x0*y2,z2=x0*y1-x1*y0;
            ++attempts[q];if(ncon[w]>=capacity){++warnings[w];break;}
            int out=w*capacity+ncon[w],cf=out*65,ci=out*5;
            for(int k=0;k<4;++k)contact_f[cf+k]=params_f[pf+k];
            contact_f[cf+4]=x0;contact_f[cf+5]=x1;contact_f[cf+6]=x2;
            contact_f[cf+7]=y0;contact_f[cf+8]=y1;contact_f[cf+9]=y2;
            contact_f[cf+10]=z0;contact_f[cf+11]=z1;contact_f[cf+12]=z2;
            contact_f[cf+13]=params_f[pf+11];
            for(int k=0;k<5;++k)contact_f[cf+14+k]=params_f[pf+12+k];
            for(int k=0;k<2;++k){contact_f[cf+19+k]=params_f[pf+17+k];contact_f[cf+21+k]=params_f[pf+19+k];}
            for(int k=0;k<5;++k)contact_f[cf+23+k]=params_f[pf+21+k];
            for(int k=28;k<65;++k)contact_f[cf+k]=0;
            for(int k=0;k<4;++k)contact_i[ci+k]=params_i[slot*5+k];contact_i[ci+4]=-1;
            ++ncon[w];++appended[q];
        }
    }
    if(current_sort)cf_contact_sort(w,start,ncon[w]-start,capacity,geom_type,contact_f,contact_i,scratch_f,scratch_i,order,order_scratch);
}
}
