#include "precision.cuh"
#pragma once
#include "collision_dedup.cuh"
namespace collision_cuda {

// Local raw + pair parameters produce the final contact record. Returns the
// same frame error as ordered assembly, before its capacity check.
template<int Fields=65>
__device__ inline bool cf_candidate_contact(const real_t* raw,const PairParams& p,real_t* f,int* flags){
    real_t x0=raw[4],x1=raw[5],x2=raw[6];
    const real_t nx=cf_norm3(x0,x1,x2);
    if(nx<REAL_C(0.5))return false;
    if(nx<REAL_C(1e-15)){x0=1;x1=x2=0;}else{const real_t inv=1/nx;x0*=inv;x1*=inv;x2*=inv;}
    real_t y0=raw[7],y1=raw[8],y2=raw[9];
    if(cf_norm3(y0,y1,y2)<REAL_C(0.5)){y0=y1=y2=0;if(x1<REAL_C(0.5)&&x1>-REAL_C(0.5))y1=1;else y2=1;}
    const real_t dot=(x0*y0+x1*y1)+x2*y2;y0-=x0*dot;y1-=x1*dot;y2-=x2*dot;
    const real_t ny=cf_norm3(y0,y1,y2);
    if(ny<REAL_C(1e-15)){y0=1;y1=y2=0;}else{const real_t inv=1/ny;y0*=inv;y1*=inv;y2*=inv;}
    const real_t z0=x1*y2-x2*y1,z1=x2*y0-x0*y2,z2=x0*y1-x1*y0;
    if(!f)return true;
    libero_vec::copy_serial(raw,f,4);
    f[4]=x0;f[5]=x1;f[6]=x2;f[7]=y0;f[8]=y1;f[9]=y2;f[10]=z0;f[11]=z1;f[12]=z2;
    f[13]=p.f[11];
    libero_vec::copy_serial(p.f+12,f+14,14);
    if(Fields>28)libero_vec::fill_serial(f+28,Fields-28,REAL_C(0.0));
    for(int k=0;k<3;++k)flags[k]=p.flags[k];
    flags[3]=raw[0]<p.f[11]?0:1;flags[4]=-1;
    return true;
}

// One lane owns a call. Compatible fixed-capacity public diagnostics are
// materialized, while kept raw data feeds pair mixing and final contacts
// directly. A small deterministic prefix separates the two parallel passes.
template<bool Diagnostics>
__device__ void cf_dedup_params_impl(int w,
    real_t* raw,int* raw_count,int* call_count,int* active,int* types,int* route,int* geom_type,int ngeom,
    real_t* compact,int* compact_count,int* keep_index,int* status,int pairs,
    real_t* usermargin,int* counts,const int* priority,const int* condim,
    const real_t* geom_margin,const real_t* geom_gap,const real_t* geom_friction,const real_t* geom_solmix,
    const real_t* geom_solref,const real_t* geom_solimp,const real_t* pair_margin,const real_t* pair_gap,
    const int* pair_dim,const real_t* pair_friction,const real_t* pair_solref,const real_t* pair_solreffriction,
    const real_t* pair_solimp,const real_t* o_margin,const real_t* o_solref,const real_t* o_solimp,
    const int* override_enabled,real_t* out_f,int* out_i,int slots,int npair,
    int* offsets,int* frame_error,real_t* candidate_f,int* candidate_i,int capacity,
    const real_t* cached_f=nullptr,const int* cached_i=nullptr,int cached_geoms=0) {
    // Capacity diagnostics retain their exact tails, but tail initialization
    // no longer runs one 50-record serial copy loop per unused call.
    const int active_value=active[w],declared_calls=call_count[w];
    const int initial_status=active_value?(active_value!=1?1:
        (declared_calls<0||declared_calls>pairs?2:0)):0;
    const int effective_calls=active_value==1&&declared_calls>=0&&declared_calls<=pairs?
        declared_calls:0;
    const int materialized_calls=Diagnostics?pairs:effective_calls;
    for(int local=threadIdx.x;local<materialized_calls;local+=blockDim.x){
        const int call=w*pairs+local;
        status[call]=initial_status;compact_count[call]=0;frame_error[call]=-1;
    }
    if(Diagnostics){
        libero_vec::fill_block(keep_index+w*pairs*50,size_t(pairs)*50,-1);
        libero_vec::fill_block(compact+w*pairs*500,size_t(pairs)*500,REAL_C(0.0));
    }
    __syncthreads(); // initialization must finish before live rows overwrite it
    for(int local=threadIdx.x;local<effective_calls;local+=blockDim.x){
        const int call=w*pairs+local,g1=route[call*200],g2=route[call*200+1];
        int kept[50],out=0,s=0;
        if(g1<0||g2<0||g1>=ngeom||g2>=ngeom)s=5;
        else if(types[call*2]!=geom_type[g1]||types[call*2+1]!=geom_type[g2])s=6;
        else{
            const int nr=raw_count[call];
            if(nr<0||nr>50)s=3;
            else{
                for(int r=0;r<nr;++r)for(int k=0;k<10;++k)
                    if(!isfinite(raw[(call*50+r)*10+k]))s=4;
                if(!s)out=cf_raw_keep_last(raw+call*500,nr,
                    types[call*2]==6&&types[call*2+1]==6,kept);
            }
        }
        status[call]=s;compact_count[call]=out;
        for(int r=0;r<out;++r){
            const int row=call*50+r;keep_index[row]=kept[r];
            if(Diagnostics)libero_vec::copy_serial(raw+(call*50+kept[r])*10,compact+row*10,10);
        }
    }
    __syncthreads();
    // Calls retain their original order. Scan bounded counts within each warp,
    // then combine at most 32 warp totals; only the inter-tile carry saturates.
    // No global prefix addition can overflow when capacity approaches INT_MAX.
    __shared__ int cf_prefix_totals[32],cf_prefix_bases[32],cf_prefix_carry;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int nwarp=(blockDim.x+31)>>5;
    if(threadIdx.x==0)cf_prefix_carry=0;
    __syncthreads();
    for(int tile=0;tile<effective_calls;tile+=blockDim.x){
        const int local=tile+threadIdx.x;
        const int count=local<effective_calls?compact_count[w*pairs+local]:0;
        // Participants are physical lanes of this block, not an instantaneous
        // ITS arrival subset after the count load's divergent bounds branch.
        const int physical_lanes=blockDim.x-warp*32<32?blockDim.x-warp*32:32;
        const unsigned mask=physical_lanes==32?0xffffffffu:(1u<<physical_lanes)-1u;
        int inclusive=count;
        for(int delta=1;delta<32;delta<<=1){
            const int previous=__shfl_up_sync(mask,inclusive,delta);
            if(lane>=delta)inclusive+=previous;
        }
        // The last warp may be partial for diagnostic launchers. Its last
        // physical lane publishes the total; no shuffle reads absent lanes.
        if(lane==31||threadIdx.x==blockDim.x-1)cf_prefix_totals[warp]=inclusive;
        __syncthreads();
        if(threadIdx.x==0){
            int carry=cf_prefix_carry;
            for(int k=0;k<nwarp;++k){
                cf_prefix_bases[k]=carry;
                const int increment=cf_prefix_totals[k];
                carry=increment>=capacity-carry?capacity:carry+increment;
            }
            cf_prefix_carry=carry;
        }
        __syncthreads();
        if(local<effective_calls){
            const int base=cf_prefix_bases[warp],prefix=inclusive-count;
            offsets[w*pairs+local]=prefix>=capacity-base?capacity:base+prefix;
        }
        // Protect shared bases and totals from the next tile's producers.
        __syncthreads();
    }
    __syncthreads();
    // All unused calls have zero count; their compatible offsets equal the
    // saturated sum of live calls. Materialize them without collective scans.
    for(int local=effective_calls+threadIdx.x;local<materialized_calls;local+=blockDim.x)
        offsets[w*pairs+local]=cf_prefix_carry;
    __syncthreads();
    if(!Diagnostics){
        // Bound shared storage independently of world count and call capacity.
        // One lane mixes each nonempty call, then all lanes consume a compact
        // list of observed contact rows. Include each call's first overflowing
        // row: frame failure must precede that call's capacity warning.
        __shared__ PairParams live_params[32];
        __shared__ int live_prefix[33],live_rows[32],live_error[32];
        for(int tile=0;tile<effective_calls;tile+=32){
            const int local=tile+threadIdx.x,call=w*pairs+local;
            if(threadIdx.x<32){
                int rows=0;
                if(local<effective_calls){
                    const int nr=compact_count[call];
                    if(nr>0&&active[w]&&counts[w]>=0&&counts[w]<=slots){
                        const int* rr=route+call*200;
                        PairParams pp;
                        if(cached_geoms==ngeom&&rr[0]>=0&&rr[0]<ngeom&&rr[1]>=0&&rr[1]<ngeom&&
                                rr[2]==-1&&rr[3]==0){
                            const int entry=rr[0]*ngeom+rr[1];
                            libero_vec::fill_serial(pp.f,10,REAL_C(0.0));
                            libero_vec::copy_serial(cached_f+entry*16,pp.f+10,16);
                            pp.flags[0]=cached_i[entry*2];pp.flags[1]=rr[0];pp.flags[2]=rr[1];
                            pp.flags[3]=-1;pp.flags[4]=cached_i[entry*2+1];pp.route_valid=true;
                        }else pp=cf_pair_params(rr[0],rr[1],rr[2],rr[3],usermargin[call*50],
                            priority,condim,geom_margin,geom_gap,geom_friction,geom_solmix,geom_solref,geom_solimp,
                            pair_margin,pair_gap,pair_dim,pair_friction,pair_solref,pair_solreffriction,pair_solimp,
                            o_margin,o_solref,o_solimp,override_enabled,ngeom,npair);
                        live_params[threadIdx.x]=pp;
                        // Ordered replay only consumes the parameter error.
                        for(int r=0;r<nr;++r)out_i[(call*50+r)*5+4]=pp.flags[4];
                        if(!pp.flags[4]){
                            rows=min(nr,max(0,counts[w]-local*50));
                            const int remaining=capacity-offsets[call];
                            if(remaining<rows)rows=remaining+1;
                        }
                    }else if(nr>0){
                        for(int r=0;r<nr;++r)out_i[(call*50+r)*5+4]=1;
                    }
                }
                live_rows[threadIdx.x]=rows;live_error[threadIdx.x]=50;
            }
            __syncthreads();
            if(threadIdx.x==0){
                live_prefix[0]=0;
                for(int j=0;j<32;++j)live_prefix[j+1]=live_prefix[j]+live_rows[j];
            }
            __syncthreads();
            for(int row=threadIdx.x;row<live_prefix[32];row+=blockDim.x){
                int lo=0,hi=32;
                while(lo<hi){const int mid=(lo+hi)/2;
                    if(live_prefix[mid+1]<=row)lo=mid+1;else hi=mid;}
                const int call=w*pairs+tile+lo,r=row-live_prefix[lo];
                const int kept=keep_index[call*50+r],base=offsets[call];
                real_t contact[10];
                if(kept>=0)libero_vec::copy_serial(raw+(call*50+kept)*10,contact,10);
                else libero_vec::fill_serial(contact,10,REAL_C(0.0));
                const int target=w*capacity+base+r;
                const bool fits=base+r<capacity;
                if(!cf_candidate_contact<28>(contact,live_params[lo],
                        fits?candidate_f+target*28:nullptr,fits?candidate_i+target*5:nullptr))
                    atomicMin(live_error+lo,r);
            }
            __syncthreads();
            if(threadIdx.x<32&&local<effective_calls)
                frame_error[call]=live_error[threadIdx.x]<50?live_error[threadIdx.x]:-1;
            __syncthreads();
        }
        return;
    }
    // Eight lanes share one call's immutable pair mixing. The physical
    // prefix remains bounded by nr/capacity, while all 50 diagnostic rows
    // (including unused tails) are materialized by independent row owners.
    // Native collision launchers use whole warps (at most 256 threads).
    __shared__ PairParams cf_row_params[32];
    const int row_group=threadIdx.x/8,row_lane=threadIdx.x&7;
    const int row_groups=blockDim.x/8;
    for(int tile=0;tile<materialized_calls;tile+=row_groups){
        const int local=tile+row_group,call=w*pairs+local;
        if(row_lane==0&&local<materialized_calls&&active[w]&&counts[w]>=0&&counts[w]<=slots){
            const int* rr=route+call*200;
            cf_row_params[row_group]=cf_pair_params(rr[0],rr[1],rr[2],rr[3],usermargin[call*50],
                priority,condim,geom_margin,geom_gap,geom_friction,geom_solmix,geom_solref,geom_solimp,
                pair_margin,pair_gap,pair_dim,pair_friction,pair_solref,pair_solreffriction,pair_solimp,
                o_margin,o_solref,o_solimp,override_enabled,ngeom,npair);
        }
        __syncthreads();
        int first_frame_error=50;
        if(local<materialized_calls){
        const int base=offsets[call],nr=compact_count[call];
        const PairParams& pp=cf_row_params[row_group];
        // Frame math keeps one row per lane; diagnostic stores below use
        // flattened record/field ownership and do not need a full mirror.
        if(active[w]&&counts[w]>=0&&counts[w]<=slots&&!pp.flags[4]){
            // Validate the first overflowing frame before capacity rejection,
            // as ordered assembly does; later frames are never observed.
            int live_rows=min(nr,50);
            live_rows=min(live_rows,max(0,counts[w]-local*50));
            if(capacity-base<live_rows)live_rows=capacity-base+1;
            for(int r=row_lane;r<live_rows;r+=8){
                const int row=call*50+r,q=local*50+r,kept=keep_index[row];
                if(q<counts[w]&&r<nr&&!pp.flags[4]&&r<=capacity-base){
                    real_t contact[10];
                    if(kept>=0)libero_vec::copy_serial(raw+(call*50+kept)*10,contact,10);
                    else libero_vec::fill_serial(contact,10,REAL_C(0.0));
                    const int target=w*capacity+base+r;
                    const bool fits=base+r<capacity;
                    if(!cf_candidate_contact(contact,pp,fits?candidate_f+target*65:nullptr,
                            fits?candidate_i+target*5:nullptr)&&r<first_frame_error)first_frame_error=r;
                }
            }
        }
        if(Diagnostics)for(int r=row_lane;r<50;r+=8){
            const int row=call*50+r,q=local*50+r;
            real_t* dst=out_f+row*26;
            if(active[w]&&counts[w]>=0&&counts[w]<=slots&&q<counts[w]){
                const int kept=keep_index[row];
                if(pp.route_valid&&kept>=0)
                    libero_vec::copy_serial(raw+(call*50+kept)*10,dst,10);
                else libero_vec::fill_serial(dst,10,REAL_C(0.0));
                libero_vec::copy_serial(pp.f+10,dst+10,16);
            }else libero_vec::fill_serial(dst,26,REAL_C(0.0));
        }
        for(int field=row_lane;field<(Diagnostics?50:nr)*5;field+=8){
            const int r=field/5,k=field%5,row=call*50+r,q=local*50+r;
            int value=k<4?-1:0;
            if(active[w]){
                if(counts[w]<0||counts[w]>slots){if(k==4)value=1;}
                else if(q<counts[w]){
                    value=pp.flags[k];
                    if(k==3&&!pp.flags[4]){
                        const int kept=keep_index[row];
                        const real_t distance=kept>=0?raw[(call*50+kept)*10]:REAL_C(0.0);
                        value=distance<pp.f[11]?0:1;
                    }
                }
            }
            out_i[row*5+k]=value;
        }
        }
        const unsigned row_mask=0xffu<<((threadIdx.x&31)&~7);
        for(int delta=4;delta>0;delta>>=1){
            const int other=__shfl_down_sync(row_mask,first_frame_error,delta,8);
            if(row_lane+delta<8&&other<first_frame_error)first_frame_error=other;
        }
        if(row_lane==0&&local<materialized_calls)frame_error[call]=first_frame_error<50?first_frame_error:-1;
        // Prevent one tile from replacing shared parameters still in use.
        __syncthreads();
    }
}

// Both schedules share the same contact math, ordered prefix and rejection
// rules. Only the diagnostic form materializes unused capacity and params_f.
template<class... Args>
__device__ __forceinline__ void cf_dedup_params(Args... args){
    cf_dedup_params_impl<true>(args...);
}
template<class... Args>
__device__ __forceinline__ void cf_dedup_params_live(Args... args){
    cf_dedup_params_impl<false>(args...);
}

// Candidate math is already complete. Keep the original ordered rejection,
// attempts, per-call overflow and segment sort semantics without loading the
// public params_f diagnostics or recomputing contact frames.
template<int Fields=65>
__device__ void cf_assemble_candidates(int w,int* compact_count,int* call_count,int* active,
    int* segment,int* sort_enabled,int* geom_type,int* params_i,int* dedup_status,int* frame_error,
    real_t* candidate_f,int* candidate_i,real_t* scratch_f,int* scratch_i,int* ncon,int* attempts,
    int* appended,int* warnings,int* status,int pairs,int capacity,int* order,int* order_scratch){
    ncon[w]=warnings[w]=status[w]=0;
    for(int p=0;p<pairs;++p)attempts[w*pairs+p]=appended[w*pairs+p]=0;
    if(!active[w])return;if(active[w]!=1){status[w]=1;return;}
    const int n=call_count[w];if(n<0||n>pairs){status[w]=2;return;}
    int start=0,current_segment=-1,current_sort=0;
    for(int p=0;p<n;++p){
        const int call=w*pairs+p;
        if(dedup_status[call]){if(!status[w])status[w]=10+dedup_status[call];return;}
        const int seg=segment[call];
        if(p==0){current_segment=seg;current_sort=sort_enabled[call];}
        if(seg!=current_segment){
            if(current_sort)cf_contact_sort<Fields>(w,start,ncon[w]-start,capacity,geom_type,candidate_f,candidate_i,
                                               scratch_f,scratch_i,order,order_scratch);
            start=ncon[w];current_segment=seg;current_sort=sort_enabled[call];
        }
        for(int r=0;r<compact_count[call];++r){
            const int param_error=params_i[(call*50+r)*5+4];
            if(param_error){if(!status[w])status[w]=20+param_error;return;}
            if(frame_error[call]==r){if(!status[w])status[w]=30;return;}
            ++attempts[call];if(ncon[w]>=capacity){++warnings[w];break;}
            ++ncon[w];++appended[call];
        }
    }
    if(current_sort)cf_contact_sort<Fields>(w,start,ncon[w]-start,capacity,geom_type,candidate_f,candidate_i,
                                       scratch_f,scratch_i,order,order_scratch);
}

// Same ABI as the scalar replay, but all block threads enter. Only the leader
// advances ordered rejection/capacity state; segment sorting is collective.
// A failed call returns uniformly before sorting its current partial segment,
// exactly as the scalar prefix does. Shared commands live until the sort and
// all its scratch writes finish; there is no leader-only collective call.
template<int Fields=65>
__device__ void cf_assemble_candidates_block_legacy(int w,int* compact_count,int* call_count,int* active,
    int* segment,int* sort_enabled,int* geom_type,int* params_i,int* dedup_status,int* frame_error,
    real_t* candidate_f,int* candidate_i,real_t* scratch_f,int* scratch_i,int* ncon,int* attempts,
    int* appended,int* warnings,int* status,int pairs,int capacity,int* order,int* order_scratch){
    __shared__ int replay_n,replay_done,replay_start,replay_segment,replay_sort,replay_cursor;
    __shared__ int replay_sort_start,replay_sort_length,replay_do_sort;
    // Live diagnostics have no contract on unused call slots.
    const int clear_calls=Fields==28?max(0,min(call_count[w],pairs)):pairs;
    for(int p=threadIdx.x;p<clear_calls;p+=blockDim.x)
        attempts[w*pairs+p]=appended[w*pairs+p]=0;
    if(threadIdx.x==0){
        ncon[w]=warnings[w]=status[w]=0;
        replay_done=0;replay_n=call_count[w];replay_start=0;replay_cursor=0;
        replay_segment=-1;replay_sort=0;
        if(!active[w])replay_done=1;
        else if(active[w]!=1){status[w]=1;replay_done=1;}
        else if(replay_n<0||replay_n>pairs){status[w]=2;replay_done=1;}
    }
    __syncthreads();
    if(replay_done)return;
    // Advance a whole ordered segment between collective boundaries. There
    // is no inter-call dependency on another lane: candidate math is already
    // complete. This removes two or three CTA barriers per detector call.
    while(true){
        if(threadIdx.x==0){
            replay_do_sort=0;
            while(replay_cursor<replay_n){
                const int call=w*pairs+replay_cursor;
                // The scalar contract checks this before a segment change;
                // a failure must not sort the preceding partial segment.
                if(dedup_status[call]){status[w]=10+dedup_status[call];replay_done=1;break;}
                const int seg=segment[call];
                if(replay_cursor==0){replay_segment=seg;replay_sort=sort_enabled[call];}
                if(seg!=replay_segment){
                    replay_do_sort=replay_sort;
                    replay_sort_start=replay_start;
                    replay_sort_length=ncon[w]-replay_start;
                    replay_start=ncon[w];replay_segment=seg;replay_sort=sort_enabled[call];
                    if(replay_do_sort)break; // resume this call after sort
                }
                for(int r=0;r<compact_count[call];++r){
                    const int error=params_i[(call*50+r)*5+4];
                    if(error){status[w]=20+error;replay_done=1;break;}
                    if(frame_error[call]==r){status[w]=30;replay_done=1;break;}
                    ++attempts[call];
                    if(ncon[w]>=capacity){++warnings[w];break;}
                    ++ncon[w];++appended[call];
                }
                if(replay_done)break;
                ++replay_cursor;
            }
            if(!replay_done&&replay_cursor==replay_n){
                replay_do_sort=replay_sort;
                replay_sort_start=replay_start;replay_sort_length=ncon[w]-replay_start;
                replay_done=2; // final sort then uniform exit
            }
        }
        __syncthreads();
        if(replay_done==1)return;
        if(replay_do_sort)cf_contact_sort_block<Fields>(w,replay_sort_start,replay_sort_length,capacity,
            geom_type,candidate_f,candidate_i,scratch_f,scratch_i,order,order_scratch);
        __syncthreads();
        if(replay_done==2)return;
    }
}

#ifndef LIBERO_CF_REPLAY_AGGREGATE_MIN_CALLS
#define LIBERO_CF_REPLAY_AGGREGATE_MIN_CALLS 65
#endif
// A row's parameter/frame verdict is independent of the capacity prefix.
// Summarize it in parallel, then preserve the original ordered prefix with
// O(1) arithmetic per call. The first overflowing row is still checked before
// warnings/attempts change; error precedence and segment sorting are unchanged.
template<int Fields=65>
__device__ void cf_assemble_candidates_block(int w,int* compact_count,int* call_count,int* active,
    int* segment,int* sort_enabled,int* geom_type,int* params_i,int* dedup_status,int* frame_error,
    real_t* candidate_f,int* candidate_i,real_t* scratch_f,int* scratch_i,int* ncon,int* attempts,
    int* appended,int* warnings,int* status,int pairs,int capacity,int* order,int* order_scratch){
    const int n=call_count[w];
    if(LIBERO_CF_REPLAY_AGGREGATE_MIN_CALLS<=0||n<LIBERO_CF_REPLAY_AGGREGATE_MIN_CALLS||
            n>pairs||active[w]!=1||blockDim.x>256){
        cf_assemble_candidates_block_legacy<Fields>(w,compact_count,call_count,active,segment,sort_enabled,
            geom_type,params_i,dedup_status,frame_error,candidate_f,candidate_i,scratch_f,scratch_i,
            ncon,attempts,appended,warnings,status,pairs,capacity,order,order_scratch);
        return;
    }
    __shared__ int first_row[256],first_code[256];
    __shared__ int cursor,done,start,current_segment,current_sort,summary_base,prepare;
    __shared__ int do_sort,sort_start,sort_length;
    const int clear_calls=Fields==28?n:pairs;
    for(int p=threadIdx.x;p<clear_calls;p+=blockDim.x)
        attempts[w*pairs+p]=appended[w*pairs+p]=0;
    if(threadIdx.x==0){
        ncon[w]=warnings[w]=status[w]=0;cursor=done=start=0;
        current_segment=-1;current_sort=0;summary_base=-1;
    }
    __syncthreads();
    while(true){
        if(threadIdx.x==0){
            const int tile=(cursor/int(blockDim.x))*int(blockDim.x);
            prepare=tile!=summary_base;
            if(prepare)summary_base=tile;
        }
        __syncthreads();
        if(prepare){
            const int p=summary_base+threadIdx.x,call=w*pairs+p;
            int row=50,code=0;
            if(p<n){
                const int nr=compact_count[call];
                row=max(0,nr);
                for(int r=0;r<nr;++r){
                    const int error=params_i[(call*50+r)*5+4];
                    if(error){row=r;code=20+error;break;}
                    if(frame_error[call]==r){row=r;code=30;break;}
                }
            }
            first_row[threadIdx.x]=row;first_code[threadIdx.x]=code;
        }
        __syncthreads();
        if(threadIdx.x==0){
            do_sort=0;
            while(cursor<n&&cursor<summary_base+int(blockDim.x)){
                const int call=w*pairs+cursor;
                if(dedup_status[call]){status[w]=10+dedup_status[call];done=1;break;}
                const int seg=segment[call];
                if(cursor==0){current_segment=seg;current_sort=sort_enabled[call];}
                if(seg!=current_segment){
                    do_sort=current_sort;sort_start=start;sort_length=ncon[w]-start;
                    start=ncon[w];current_segment=seg;current_sort=sort_enabled[call];
                    if(do_sort)break;
                }
                const int nr=max(0,compact_count[call]),remaining=max(0,capacity-ncon[w]);
                const int slot=cursor-summary_base,row=first_row[slot];
                const int observed=remaining<nr?remaining+1:nr;
                if(first_code[slot]&&row<observed){
                    attempts[call]=appended[call]=row;ncon[w]+=row;
                    status[w]=first_code[slot];done=1;break;
                }
                const int accepted=min(nr,remaining);
                attempts[call]=accepted+(nr>remaining?1:0);appended[call]=accepted;
                ncon[w]+=accepted;if(nr>remaining)++warnings[w];
                ++cursor;
            }
            if(!done&&cursor==n){
                do_sort=current_sort;sort_start=start;sort_length=ncon[w]-start;done=2;
            }
        }
        __syncthreads();
        if(done==1)return;
        if(do_sort)cf_contact_sort_block<Fields>(w,sort_start,sort_length,capacity,geom_type,
            candidate_f,candidate_i,scratch_f,scratch_i,order,order_scratch);
        __syncthreads();
        if(done==2)return;
    }
}
// The private live record omits mu/H; public commit writes their required zeros.
template<class... Args>
__device__ __forceinline__ void cf_assemble_candidates_live(Args... args){
    cf_assemble_candidates_block<28>(args...);
}

}

