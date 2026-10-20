#include "precision.cuh"
// Handwritten S stages. CF_S_* macros supply only ABI signatures, never arithmetic.
// Every array is a contiguous raw pointer. World and model capacities are explicit.
#pragma once
#ifndef CF_S_KBIP_PLANES
#define CF_S_KBIP_PLANES 0
#endif
#ifndef CF_S_UNIFIED_METADATA
#define CF_S_UNIFIED_METADATA 0
#endif
#ifndef CF_S_INTERNAL_OWNER_RECEIPT
#define CF_S_INTERNAL_OWNER_RECEIPT 0
#endif
#ifndef CF_S_STATIC_CANDIDATES
#define CF_S_STATIC_CANDIDATES 0
#endif
// Common-zero is experimentally opt-in: current paired timings regress.
// Defining this to 1 retains the independent finite-range safety guard.
#ifndef CF_S_COMMON_ANCESTOR_ZERO
#define CF_S_COMMON_ANCESTOR_ZERO 0
#endif
namespace cf_s {
// Pair checks are restricted to the caller's valid contiguous span.
__device__ __forceinline__ bool bounded_nonfinite_lane(const real_t* p,int n,int first,int stride){
  bool bad=false;
  for(int z=2*first;z<n;z+=2*stride){
    if(z+1<n){const real2_t v=libero_vec::load_pair(p+z);
      bad|=!isfinite(v.x)||!isfinite(v.y)||fabs(v.x)>REAL_SAFE_MAGNITUDE||fabs(v.y)>REAL_SAFE_MAGNITUDE;
    }else bad|=!isfinite(p[z])||fabs(p[z])>REAL_SAFE_MAGNITUDE;
  }
  return bad;
}
__device__ __forceinline__ int kbip_address(int w,int r,int k,int rows){
#if CF_S_KBIP_PLANES
  return (w*4+k)*rows+r;
#else
  return (w*rows+r)*4+k;
#endif
}
__device__ void publish_exclude(int w,const int*active,const int*status,const int*committed,
 const int*ncon,const int*source,int*target,int capacity){
  if(active[w]!=1||status[w]||committed[w]!=1)return;
  int n=ncon[w];if(n<0||n>capacity)return;
  for(int c=threadIdx.x;c<n;c+=blockDim.x)target[w*capacity+c]=source[w*capacity+c];
}
__device__ void CF_S_position_gate {
#if CF_S_INTERNAL_OWNER_RECEIPT
  producer_receipt[w*4]=0;
#endif
  effective[w]=0; if(active[w]==0)return; committed[w]=0;
  if(status[w]==0)effective[w]=active[w];
}
__device__ void CF_S_reset_stage_status { s02_status[w]=s01_status[w]=s03_status[w]=0; }
__device__ void CF_S_gate_active_status { downstream[w]=(active[w]==1 && upstream_status[w]==0); }
__device__ void CF_S_contact_incoming { incoming[w]=nf[w]+nlimit[w]; }
__device__ void CF_S_merge_stage_status {
  int s=stage_status[w]; if(!s || committed[w]!=1)return;
  if(!public_status[w])public_status[w]=s==1?1:s==2?2:s==3?5:(s==4||s==5)?4:8;
  committed[w]=0;
}
// Complete-block vote: every lane enters, including padded lanes.
__device__ __forceinline__ bool s1_block_any(bool bad){
  __shared__ int votes[32],result;
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  bool v=libero_warp::any(bad);
  if(lane==0)votes[warp]=v;
  __syncthreads();
  if(warp==0){
    bool all=libero_warp::any(lane<(blockDim.x>>5)&&votes[lane]);
    if(lane==0)result=all;
  }
  __syncthreads();return result!=0;
}
__device__ void CF_S_produce {
  if(status[w] || !active[w])return;
  if(active[w]!=1){if(threadIdx.x==0)status[w]=1;return;}
  int offset=incoming[w],cap=capacity[w];
  if(offset<0||cap<0||offset>cap){if(threadIdx.x==0)status[w]=2;return;}
  bool bad=false;
  for(int k=threadIdx.x;k<NQ;k+=blockDim.x)bad|=!isfinite(qpos[w*NQ+k]);
  if(s1_block_any(bad)){if(threadIdx.x==0)status[w]=3;return;}
  bad=false;
  for(int j=threadIdx.x;j<NJ;j+=blockDim.x)
    bad|=limited[j]&&(jnt_type[j]!=2&&jnt_type[j]!=3);
  if(s1_block_any(bad)){if(threadIdx.x==0)status[w]=4;return;}
  __shared__ int friction_map[NV>0?NV:1],limit_map[2*NJ>0?2*NJ:1];
  __shared__ int fr,lr,nl_count;
  if(threadIdx.x==0){fr=lr=nl_count=0;}
  __syncthreads();
#if CF_S_STATIC_CANDIDATES
  // Exact predicate readset comparison catches arbitrary raw static writes.
  // Values/ranges/options remain current inputs; a drift uses the full scan.
  bool candidate_drift=false;
  for(int k=threadIdx.x;k<NV;k+=blockDim.x)
    candidate_drift|=int(frictionloss[k]>REAL_C(0.))!=cold_friction_mask[k];
  for(int j=threadIdx.x;j<NJ;j+=blockDim.x)
    candidate_drift|=int(limited[j]!=0)!=cold_limited_mask[j];
  const bool cold_valid=!s1_block_any(candidate_drift);
#endif
  // Warp 0 scans in native DOF / joint-side order. Integer scans preserve
  // row addresses exactly; all remaining warps join the row-write phase.
  if(threadIdx.x<32){
    int lane=threadIdx.x;
#if CF_S_STATIC_CANDIDATES
    if(cold_valid){
      for(int r=lane;r<cold_friction_count;r+=32)friction_map[r]=cold_friction_map[r];
      if(lane==0)fr=cold_friction_count;
    }else
#endif
    {
    for(int base=0;base<NV;base+=32){
      int i=base+lane,keep=i<NV&&frictionloss[i]>0;
      int prefix=libero_warp::exclusive_sum(keep),count=libero_warp::sum(keep);
      int start=fr;
      if(keep)friction_map[start+prefix]=i;
      __syncwarp();if(lane==0)fr=start+count;__syncwarp();
    }
    }
#if CF_S_STATIC_CANDIDATES
    const int candidate_count=cold_valid?cold_limit_count:2*NJ;
#else
    const int candidate_count=2*NJ;
#endif
    for(int base=0;base<candidate_count;base+=32){
      const int ordinal=base+lane;
#if CF_S_STATIC_CANDIDATES
      const int z=cold_valid&&ordinal<candidate_count?cold_limit_sides[ordinal]:ordinal;
#else
      const int z=ordinal;
#endif
      int j=z/2,side=z&1,keep=0,counted=0;
      if(ordinal<candidate_count&&limited[j]){
        real_t v=qpos[w*NQ+qposadr[j]],m=margins[j];
        real_t d=side?ranges[2*j+1]-v:v-ranges[2*j];
        counted=d<m;keep=!(d>=m);
      }
      // Keep the baseline's distinct comparisons even for NaN model data.
      int prefix=libero_warp::exclusive_sum(keep),count=libero_warp::sum(keep);
      int counted_sum=libero_warp::sum(counted),start=lr;
      if(keep)limit_map[start+prefix]=z;
      __syncwarp();if(lane==0){lr=start+count;nl_count+=counted_sum;}__syncwarp();
    }
  }
  __syncthreads();
  if(fr>NF_CAP||nl_count>NL_CAP||fr+nl_count>cap-offset){
    if(threadIdx.x==0)status[w]=5;return;
  }
  for(int r=threadIdx.x;r<fr;r+=blockDim.x){
    int i=friction_map[r],slot=w*NF_CAP+r;
    friction_pos[slot]=friction_margin[slot]=REAL_C(0.);friction_loss[slot]=frictionloss[i];
    friction_type[slot]=1;friction_id[slot]=i;friction_global_adr[slot]=offset+r;
#if CF_S_UNIFIED_METADATA
    const int q=w*S_ROWS+r;
    unified_pos[q]=unified_margin[q]=REAL_C(0.);unified_loss[q]=frictionloss[i];
    unified_type[q]=1;unified_id[q]=i;
#endif
  }
  for(int r=threadIdx.x;r<lr;r+=blockDim.x){
    int z=limit_map[r],j=z/2,side=z&1,slot=w*NL_CAP+r;
    real_t v=qpos[w*NQ+qposadr[j]];
    limit_pos[slot]=side?ranges[2*j+1]-v:v-ranges[2*j];
    limit_margin[slot]=margins[j];limit_loss[slot]=REAL_C(0.);
    limit_type[slot]=3;limit_id[slot]=j;limit_global_adr[slot]=offset+fr+r;
#if CF_S_UNIFIED_METADATA
    const int q=w*S_ROWS+fr+r;
    unified_pos[q]=limit_pos[slot];unified_margin[q]=margins[j];unified_loss[q]=REAL_C(0.);
    unified_type[q]=3;unified_id[q]=j;
#endif
  }
  for(int z=2*threadIdx.x;z<(fr+lr)*NV;z+=2*blockDim.x){
    real_t value[2];
    const int count=z+1<(fr+lr)*NV?2:1;
    for(int h=0;h<count;h++){
      const int r=(z+h)/NV,k=(z+h)%NV;
      if(r<fr)value[h]=k==friction_map[r]?REAL_C(1.):REAL_C(0.);
      else{const int local=r-fr,j=limit_map[local]/2,side=limit_map[local]&1;
        value[h]=k==dofadr[j]?(side?-REAL_C(1.):REAL_C(1.)):REAL_C(0.);}
    }
    real_t* out=unified_J+w*S_ROWS*NV+z;
    if(count==2)libero_vec::store_pair(out,value[0],value[1]);else *out=value[0];
#if CF_S_COMPATIBLE
    const int split=fr*NV;
    if(count==2&&(z+1<split||z>=split)){
      real_t* compatible=z<split?friction_J+w*NF_CAP*NV+z:limit_J+w*NL_CAP*NV+z-split;
      libero_vec::store_pair(compatible,value[0],value[1]);
    }else for(int h=0;h<count;h++){
      if(z+h<split)friction_J[w*NF_CAP*NV+z+h]=value[h];
      else limit_J[w*NL_CAP*NV+z+h-split]=value[h];
    }
#endif
  }
  __syncthreads();
  if(threadIdx.x==0){nf_delta[w]=fr;nlimit_delta[w]=lr;nefc_delta[w]=fr+lr;}
}
__device__ void CF_S_preflight {
  // All lanes participate, including the last tile's inactive contact lanes.
  // The launcher uses complete warps (32..1024 threads per block).
  __shared__ int tile_sums[32],tile_offsets[32],tile_errors[32];
  __shared__ int running_total,first_error;
  __shared__ int field_error[1024];
  __shared__ unsigned char field_eligible[1024];
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5,nwarp=blockDim.x>>5;
  if(threadIdx.x==0){status[w]=0;contact_queue_valid[w]=0;running_total=0;first_error=2147483647;}
  __syncthreads();
  if(!active[w])return;
  if(active[w]!=1){if(threadIdx.x==0)status[w]=1;return;}
  const int count=ncon[w],base=incoming[w],cap=capacity[w];
  if(count<0||count>maxcon){if(threadIdx.x==0)status[w]=2;return;}
  if(base<0||cap<base){if(threadIdx.x==0)status[w]=3;return;}
  for(int begin=0;begin<count;begin+=blockDim.x){
    const int c=begin+threadIdx.x,g=w*maxcon+c;
    int error=0,rows=0;
    if(c<count&&!exclude[g]){
      const int d=dim[g],g1=geom1[g],g2=geom2[g];
      if((d!=1&&d!=3&&d!=4&&d!=6)||g1<0||g1>=ngeom||g2<0||g2>=ngeom)
        error=4;
      if(!error){
        const int b1=geom_body[g1],b2=geom_body[g2];
        if(b1<0||b1>=nbody||b2<0||b2>=nbody)error=4;
      }
      // Numeric fields are checked cooperatively below; the structural
      // gate keeps the baseline read domain (excluded and invalid IDs held).
      if(!error&&nv>0)rows=d;
    }
    field_eligible[threadIdx.x]=c<count&&!exclude[g]&&!error;
    field_error[threadIdx.x]=0;
    __syncthreads();
    // Adjacent lanes check adjacent record fields. This retains the original
    // contact AoS owner while coalescing the 3/9-field numeric preflight.
    for(int z=threadIdx.x;z<blockDim.x*2;z+=blockDim.x){
      const int local=z/2,pair=z%2,cc=begin+local;
      if(field_eligible[local]&&bounded_nonfinite_lane(pos+(w*maxcon+cc)*3,3,pair,2))
        atomicOr(field_error+local,5);
    }
    for(int z=threadIdx.x;z<blockDim.x*5;z+=blockDim.x){
      const int local=z/5,pair=z%5,cc=begin+local;
      if(field_eligible[local]&&bounded_nonfinite_lane(frame+(w*maxcon+cc)*9,9,pair,5))
        atomicOr(field_error+local,5);
    }
    if(field_eligible[threadIdx.x]&&(!isfinite(dist[g])||!isfinite(margin[g])||
       fabs(dist[g])>REAL_SAFE_MAGNITUDE||fabs(margin[g])>REAL_SAFE_MAGNITUDE))atomicOr(field_error+threadIdx.x,5);
    __syncthreads();
    if(!error&&field_error[threadIdx.x]>0){error=5;rows=0;}
    // Encode the lane index before the error code: earliest contact wins,
    // including when its error differs from that of a later contact.
    const int encoded=error?int(threadIdx.x)*8+error:2147483647;
    const int offset=libero_warp::exclusive_sum(rows);
    const int sum=libero_warp::sum(rows),failure=libero_warp::min(encoded);
    if(lane==0){tile_sums[warp]=sum;tile_errors[warp]=failure;}
    __syncthreads();
    if(warp==0){
      const int warp_sum=lane<nwarp?tile_sums[lane]:0;
      const int warp_error=lane<nwarp?tile_errors[lane]:2147483647;
      const int warp_offset=libero_warp::exclusive_sum(warp_sum);
      const int block_error=libero_warp::min(warp_error);
      if(lane<nwarp)tile_offsets[lane]=warp_offset;
      if(lane==0)first_error=block_error;
    }
    __syncthreads();
    // The serial baseline writes rowstart before validating this contact.
    // Preserve that partial-write boundary and leave later contacts untouched.
    if(c<count&&(first_error==2147483647||int(threadIdx.x)<=first_error/8))
      rowstart[g]=running_total+tile_offsets[warp]+offset;
    __syncthreads();
    if(first_error!=2147483647){
      if(threadIdx.x==0)status[w]=first_error%8;
      return;
    }
    if(threadIdx.x==0)
      running_total+=tile_offsets[nwarp-1]+tile_sums[nwarp-1];
    __syncthreads();
  }
  const int total=running_total;
  if(total>maxrows||base>2147483647-total||base+total>cap){
    if(threadIdx.x==0)status[w]=6;
    return;
  }
  bool invalid=false;
  if(total>0){
    invalid|=bounded_nonfinite_lane(cdof+w*nv*6,nv*6,threadIdx.x,blockDim.x);
    invalid|=bounded_nonfinite_lane(subtree+w*nbody*3,nbody*3,threadIdx.x,blockDim.x);
  }
  const bool warp_invalid=libero_warp::any(invalid);
  if(lane==0)tile_errors[warp]=warp_invalid?1:0;
  __syncthreads();
  if(warp==0){
    const bool block_invalid=libero_warp::any(lane<nwarp&&tile_errors[lane]);
    if(lane==0){
      if(block_invalid)status[w]=5;
      else local_count[w]=total;
    }
  }
  __syncthreads();
  if(status[w])return;
  // Stable within each admissible dimension. The queue owns contact IDs,
  // never replacement row addresses: every consumer still scatters by rowstart.
  if(warp==0){
    if(lane==0)contact_dim_offsets[w*5]=0;
    __syncwarp(libero_warp::mask);
    for(int bin=0;bin<4;bin++){
      const int requested=bin==0?1:bin==1?3:bin==2?4:6;
      int cursor=contact_dim_offsets[w*5+bin];
      for(int begin=0;begin<count;begin+=32){
        const int c=begin+lane,g=w*maxcon+c;
        const int keep=c<count&&!exclude[g]&&nv>0&&dim[g]==requested;
        const int rank=libero_warp::exclusive_sum(keep),total=libero_warp::sum(keep);
        if(keep)contact_queue[w*maxcon+cursor+rank]=c;
        cursor+=total;
      }
      if(lane==0)contact_dim_offsets[w*5+bin+1]=cursor;
      __syncwarp(libero_warp::mask);
    }
    if(lane==0)contact_queue_valid[w]=1;
  }
  __syncthreads();
}
__device__ real_t body_component(const real_t*cdof,const real_t*subtree,const real_t*pos,
 const int*roots,const int*ancestor,int w,int c,int body,int j,int component,bool rotation,int nv,int nb,int nc){
  if(!ancestor[body*nv+j])return REAL_C(0.);
  const real_t*a=cdof+(w*nv+j)*6;
  if(rotation)return a[component];
  const real_t*p=pos+(w*nc+c)*3;const real_t*s=subtree+(w*nb+roots[body])*3;
  real_t o[3]={p[0]-s[0],p[1]-s[1],p[2]-s[2]};
  int u=(component+1)%3,v=(component+2)%3;
  return a[3+component]+(a[u]*o[v]-a[v]*o[u]);
}
__device__ __forceinline__ void body_components(const real_t*cdof,const real_t*subtree,const real_t*pos,
 const int*roots,const int*ancestor,int w,int c,int body,int j,bool rotation,int nv,int nb,int nc,
 real_t* translation,real_t* angular){
  if(!ancestor[body*nv+j]){
    for(int k=0;k<3;k++)translation[k]=angular[k]=REAL_C(0.);
    return;
  }
  const real_t*a=cdof+(w*nv+j)*6;
  const real_t*p=pos+(w*nc+c)*3;const real_t*s=subtree+(w*nb+roots[body])*3;
  real_t o[3]={p[0]-s[0],p[1]-s[1],p[2]-s[2]};
  for(int k=0;k<3;k++){
    int u=(k+1)%3,v=(k+2)%3;
    translation[k]=a[3+k]+(a[u]*o[v]-a[v]*o[u]);
    angular[k]=rotation?a[k]:REAL_C(0.);
  }
}
__device__ __forceinline__ void body_components_offset(const real_t*cdof,
 const int*ancestor,int w,int body,int j,bool rotation,int nv,const real_t*offset,
 real_t*translation,real_t*angular){
  if(!ancestor[body*nv+j]){
    for(int k=0;k<3;k++)translation[k]=angular[k]=REAL_C(0.);
    return;
  }
  const real_t*a=cdof+(w*nv+j)*6;
  for(int k=0;k<3;k++){
    int u=(k+1)%3,v=(k+2)%3;
    translation[k]=a[3+k]+(a[u]*offset[v]-a[v]*offset[u]);
    angular[k]=rotation?a[k]:REAL_C(0.);
  }
}
// Membership comes from the immutable ordered body-pair DOF table. A missing
// body contribution is explicitly zero and never reads an unrelated cdof.
__device__ __forceinline__ void body_components_known(const real_t* cdof,
 int w,int j,bool member,bool rotation,int nv,const real_t* offset,
 const real_t*planes,int pitch,int layout,int plane_pair,
 real_t* translation,real_t* angular){
  if(!member){for(int k=0;k<3;k++)translation[k]=angular[k]=REAL_C(0.);return;}
  real_t value[6];
  // An optional producer view is consumed only when explicitly bound. The
  // public AoS fallback and membership read domain remain available.
  if(layout&&planes){
    if(plane_pair>=0){
      for(int k=0;k<6;k++){
        const real2_t v=libero_vec::load_pair(planes+(w*6+k)*pitch+plane_pair);
        value[k]=j==plane_pair?v.x:v.y;
      }
    }else for(int k=0;k<6;k++)value[k]=planes[(w*6+k)*pitch+j];
  }else{
    const real_t* a=cdof+(w*nv+j)*6;
    for(int k=0;k<6;k+=2){const real2_t v=libero_vec::load_pair(a+k);value[k]=v.x;value[k+1]=v.y;}
  }
  const real_t* a=value;
  for(int k=0;k<3;k++){
    int u=(k+1)%3,v=(k+2)%3;
    translation[k]=a[3+k]+(a[u]*offset[v]-a[v]*offset[u]);
    angular[k]=rotation?a[k]:REAL_C(0.);
  }
}
#ifndef CF_S_WARPS
#define CF_S_WARPS 8
#endif
__device__ __forceinline__ bool shared_cdof_safe(const real_t* cdof,const real_t* planes,
 int w,int j,int nv,int pitch,int layout){
  // The production preflight already bounds |cdof| by REAL_SAFE_MAGNITUDE. Keep the
  // optimization independently safe as well: exceptional direct callers must
  // retain the old overflow/NaN arithmetic and later finite-check rejection.
  if(layout&&planes){
    for(int k=0;k<6;k++){
      const real_t value=planes[(w*6+k)*pitch+j];
      if(!isfinite(value)||fabs(value)>REAL_SAFE_MAGNITUDE)return false;
    }
  }else for(int k=0;k<6;k+=2){
    const real2_t value=libero_vec::load_pair(cdof+(w*nv+j)*6+k);
    if(!isfinite(value.x)||!isfinite(value.y)||fabs(value.x)>REAL_SAFE_MAGNITUDE||fabs(value.y)>REAL_SAFE_MAGNITUDE)return false;
  }
  return true;
}
__device__ void CF_S_jacobian_rows {
  __shared__ int pair_column[CF_S_WARPS][32],pair_member[CF_S_WARPS][32],pair_count[CF_S_WARPS];
  if(status[w]||active[w]!=1||nv<=0)return;
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  const int warps=(blockDim.x+31)>>5;
  // Even a partial final warp covers every DOF. All its live lanes execute
  // the broadcasts before entering the potentially uneven DOF tile loop.
  const int lanes=blockDim.x-warp*32<32?blockDim.x-warp*32:32;
  const unsigned mask=0xffffffffu>>(32-lanes);
  const bool queued=contact_queue_valid[w]==1;
  const int contact_count=queued?contact_dim_offsets[w*5+4]:ncon[w];
#if CF_S_CONTACT_TILES
  // A tile owns complete contacts, including every dense zero/J diagnostic
  // column and every cone row. Queue IDs remain stable original contact IDs.
  const int begin=tile*CF_S_CONTACT_TILE;
  const int end=min(begin+CF_S_CONTACT_TILE,contact_count);
#else
  const int begin=0,end=contact_count;
#endif
  for(int ordinal=begin+warp;ordinal<end;ordinal+=warps){
    const int c=queued?contact_queue[w*maxcon+ordinal]:ordinal;
    int g=w*maxcon+c;
    int skip=__shfl_sync(mask,lane==0?exclude[g]:0,0);
    if(skip)continue;
    int b1=0,b2=0,d=0,start=0;
    if(lane==0){b1=geom_body[geom1[g]];b2=geom_body[geom2[g]];d=dim[g];start=rowstart[g];}
    b1=__shfl_sync(mask,b1,0);b2=__shfl_sync(mask,b2,0);
    d=__shfl_sync(mask,d,0);start=__shfl_sync(mask,start,0);
#if !CF_S_COMPATIBLE && CF_S_COMMON_ANCESTOR_ZERO
    // For a shared ancestor of bodies in the same motion tree, both body
    // components are identical at this contact point. Their difference is
    // exactly +0 when their intermediate arithmetic is finite. Dense zeros above
    // already own these columns; compatible diagnostics still compute both.
    const bool common_root=__shfl_sync(mask,lane==0?int(roots[b1]==roots[b2]):0,0)!=0;
#endif
    real_t offset1[3],offset2[3],contact_frame[9];
    for(int k=0;k<3;k++){
      real_t one=REAL_C(0.),two=REAL_C(0.);
      if(lane==0){
        real_t p=pos[g*3+k];
        one=p-subtree[(w*S_NBODY+roots[b1])*3+k];
        two=p-subtree[(w*S_NBODY+roots[b2])*3+k];
      }
      offset1[k]=__shfl_sync(mask,one,0);
      offset2[k]=__shfl_sync(mask,two,0);
    }
    for(int k=0;k<9;k++)contact_frame[k]=__shfl_sync(mask,lane==0?frame[g*9+k]:REAL_C(0.),0);
#if !CF_S_COMPATIBLE && CF_S_COMMON_ANCESTOR_ZERO
    bool common_root_safe=common_root;
    for(int k=0;k<3;k++)common_root_safe&=isfinite(offset1[k])&&fabs(offset1[k])<=(REAL_C(2)*REAL_SAFE_MAGNITUDE);
    for(int k=0;k<9;k++)common_root_safe&=isfinite(contact_frame[k]);
#endif
    // Retain dense zero columns even though arithmetic visits only the ordered
    // union of the two ancestor sets. Rotational diagnostics for d<=3 remain
    // untouched, matching their original ownership boundary.
    for(int r=0;r<d;r++){
        libero_vec::fill_lane(unified_J+(w*S_ROWS+unified_incoming[w]+start+r)*nv,nv,REAL_C(0.),lane,lanes);
#if CF_S_COMPATIBLE
        libero_vec::fill_lane(J+(w*S_ROWS+start+r)*nv,nv,REAL_C(0.),lane,lanes);
#endif
    }
#if CF_S_COMPATIBLE
    for(int k=0;k<3;k++){
      const int z=(g*3+k)*nv;
      libero_vec::fill_lane(jac1p+z,nv,REAL_C(0.),lane,lanes);
      libero_vec::fill_lane(jac2p+z,nv,REAL_C(0.),lane,lanes);
      libero_vec::fill_lane(difp+z,nv,REAL_C(0.),lane,lanes);
      if(d>3){
        libero_vec::fill_lane(jac1r+z,nv,REAL_C(0.),lane,lanes);
        libero_vec::fill_lane(jac2r+z,nv,REAL_C(0.),lane,lanes);
        libero_vec::fill_lane(difr+z,nv,REAL_C(0.),lane,lanes);
      }
    }
#endif
    __syncwarp(mask);
    int next1=body_dof_offsets[b1],next2=body_dof_offsets[b2];
    const int end1=body_dof_offsets[b1+1],end2=body_dof_offsets[b2+1];
    // Cold model cache contains precisely the same ordered union/member bits.
    // A missing/budgeted-out pair retains the qualified CSR merge. Only integer
    // structure is reused: cdof, offsets and contact frames are always current.
    int cached_pair=lane==0&&pair_cache_enabled?body_pair_lookup[b1*S_NBODY+b2]:-1;
    cached_pair=__shfl_sync(mask,cached_pair,0);
    int cached_cursor=cached_pair>=0?pair_support_offsets[cached_pair]:0;
    const int cached_end=cached_pair>=0?pair_support_offsets[cached_pair+1]:0;
    while(true){
      if(cached_pair>=0){
        const int used=min(lanes,cached_end-cached_cursor);
        if(lane==0)pair_count[warp]=used;
        if(lane<used){
          pair_column[warp][lane]=pair_support_columns[cached_cursor+lane];
          pair_member[warp][lane]=pair_support_members[cached_cursor+lane];
        }
        cached_cursor+=used;
      }else if(lane==0){
        int used=0;
        while(used<lanes&&(next1<end1||next2<end2)){
          const int one=next1<end1?body_dof_indices[next1]:2147483647;
          const int two=next2<end2?body_dof_indices[next2]:2147483647;
          const int j=one<two?one:two;
          pair_column[warp][used]=j;
          pair_member[warp][used]=(one==j?1:0)|(two==j?2:0);
          if(one==j)next1++;if(two==j)next2++;
          used++;
        }
        pair_count[warp]=used;
      }
      __syncwarp(mask);
      const int used=pair_count[warp];
      if(!used)break;
      if(lane<used){
      const int j=pair_column[warp][lane],members=pair_member[warp][lane];
#if !CF_S_COMPATIBLE && CF_S_COMMON_ANCESTOR_ZERO
      // With B=REAL_SAFE_MAGNITUDE, |offset|<=2B and |cdof|<=B
      // bound cross products by 2B^2, below REAL_MAX in either precision. Outside
      // this sufficient bound, execute the accepted arithmetic unchanged.
      if(!(common_root_safe&&members==3&&shared_cdof_safe(cdof,cdof_planes,w,j,nv,cdof_pitch,cdof_layout))){
#endif
      // Plane pairs require two adjacent valid DOFs with identical membership.
      // Each lane still owns its original column and its original arithmetic.
      const int peer=lane^1;
      const int pairbase=j-(lane&1);
      const bool adjacent=pairbase>=0&&pairbase+1<nv&&pairbase+1<cdof_pitch&&
        peer<used&&pair_member[warp][peer]==members&&
        pair_column[warp][peer]==j+((lane&1)?-1:1);
      const int plane_pair=adjacent?pairbase:-1;
      real_t dp[3],dr[3],onep[3],twop[3],oner[3],twor[3];
      body_components_known(cdof,w,j,(members&1)!=0,d>3,nv,offset1,cdof_planes,cdof_pitch,cdof_layout,plane_pair,onep,oner);
      body_components_known(cdof,w,j,(members&2)!=0,d>3,nv,offset2,cdof_planes,cdof_pitch,cdof_layout,plane_pair,twop,twor);
      for(int k=0;k<3;k++){
        int z=(g*3+k)*nv+j;
        real_t one=onep[k],two=twop[k];dp[k]=two-one;
#if CF_S_COMPATIBLE
        jac1p[z]=one;jac2p[z]=two;difp[z]=dp[k];
#endif
        if(d>3){
          one=oner[k];two=twor[k];dr[k]=two-one;
#if CF_S_COMPATIBLE
          jac1r[z]=one;jac2r[z]=two;difr[z]=dr[k];
#endif
        }
      }
      for(int r=0;r<d;r++){
        real_t value=REAL_C(0.);int fr=r<3?r:r-3;
        for(int k=0;k<3;k++){real_t coeff=contact_frame[fr*3+k];if(coeff!=REAL_C(0.))value+=coeff*(r<3?dp[k]:dr[k]);}
        unified_J[(w*S_ROWS+unified_incoming[w]+start+r)*nv+j]=value;
#if CF_S_COMPATIBLE
        J[(w*S_ROWS+start+r)*nv+j]=value;
#endif
      }
#if !CF_S_COMPATIBLE && CF_S_COMMON_ANCESTOR_ZERO
      }
#endif
      }
      __syncwarp(mask);
    }
  }
}
__device__ void CF_S_metadata {
  if(status[w]||active[w]!=1)return;
  if(threadIdx.x==0)outgoing[w]=incoming[w]+local_count[w];
  const bool queued=contact_queue_valid[w]==1;
  const int contact_count=queued?contact_dim_offsets[w*5+4]:ncon[w];
  // nv==0 needs the original excluded-contact address update, so it remains
  // on the generic sequence even when the active math queue is empty.
  const int extent=nv>0?contact_count:ncon[w];
  for(int ordinal=threadIdx.x;ordinal<extent;ordinal+=blockDim.x){
    const int c=nv>0&&queued?contact_queue[w*S_CONTACTS+ordinal]:ordinal;
    int g=w*S_CONTACTS+c;if(exclude[g])continue;
    if(nv==0){out_address[g]=-1;out_exclude[g]=3;continue;}
    int start=rowstart[g],d=dim[g];out_address[g]=incoming[w]+start;
    const int q=w*S_ROWS+start;
    out_pos[q]=dist[g];out_margin[q]=margin[g];
    libero_vec::fill_serial(out_pos+q+1,d-1,REAL_C(0.));
    libero_vec::fill_serial(out_margin+q+1,d-1,REAL_C(0.));
    libero_vec::fill_serial(out_frictionloss+q,d,REAL_C(0.));
    libero_vec::fill_serial(out_type+q,d,d==1?5:7);
    libero_vec::fill_serial(out_id+q,d,c);
#if CF_S_UNIFIED_METADATA
    const int u=q+incoming[w];
    unified_pos[u]=dist[g];unified_margin[u]=margin[g];
    libero_vec::fill_serial(unified_pos+u+1,d-1,REAL_C(0.));
    libero_vec::fill_serial(unified_margin+u+1,d-1,REAL_C(0.));
    libero_vec::fill_serial(unified_loss+u,d,REAL_C(0.));
    libero_vec::fill_serial(unified_type+u,d,d==1?5:7);
    libero_vec::fill_serial(unified_id+u,d,c);
#endif
  }
  __syncthreads();
#if CF_S_INTERNAL_OWNER_RECEIPT
  // This receipt is reset by position_gate and consumed only by the following
  // assembler in the sealed producer schedule. It never authenticates mutable
  // public arrays for Newton/projection or survives a position invocation.
  if(threadIdx.x==0&&contact_queue_valid[w]==1){
    producer_receipt[w*4+1]=incoming[w];
    producer_receipt[w*4+2]=local_count[w];
    producer_receipt[w*4+3]=ncon[w];
    producer_receipt[w*4]=1;
  }
#endif
}
__device__ void CF_S_assemble {
  if(threadIdx.x==0)committed[w]=0;
  __syncthreads();if(status[w]||!active[w])return;
  if(active[w]!=1){if(threadIdx.x==0)status[w]=1;return;}
  int s=s02_status[w];if(s){if(threadIdx.x==0)status[w]=s==1?1:s==2?2:s==3?4:s==4?5:3;return;}
  s=s01_status[w];if(s){if(threadIdx.x==0)status[w]=s==1?1:(s==2||s==3)?2:s==4?5:s==5?4:3;return;}
  int f=nf[w],l=nlimit[w],n=s01_local_count[w],nc=ncon[w],total=f+l+n;
  if(f<0||f>nf_cap||l<0||l>nl_cap||n<0||nc<0||nc>maxcon){if(threadIdx.x==0)status[w]=2;return;}
  if(total>rows){if(threadIdx.x==0)status[w]=3;return;}
#if !CF_S_UNIFIED_METADATA
  libero_vec::copy_block(s02_friction_pos+w*nf_cap,efc_pos+w*rows+0,f);
  libero_vec::copy_block(s02_friction_margin+w*nf_cap,efc_margin+w*rows+0,f);
  libero_vec::copy_block(s02_friction_loss+w*nf_cap,efc_frictionloss+w*rows+0,f);
  libero_vec::copy_block(s02_friction_type+w*nf_cap,efc_type+w*rows+0,f);
  libero_vec::copy_block(s02_friction_id+w*nf_cap,efc_id+w*rows+0,f);
  libero_vec::copy_block(s02_limit_pos+w*nl_cap,efc_pos+w*rows+f,l);
  libero_vec::copy_block(s02_limit_margin+w*nl_cap,efc_margin+w*rows+f,l);
  libero_vec::copy_block(s02_limit_loss+w*nl_cap,efc_frictionloss+w*rows+f,l);
  libero_vec::copy_block(s02_limit_type+w*nl_cap,efc_type+w*rows+f,l);
  libero_vec::copy_block(s02_limit_id+w*nl_cap,efc_id+w*rows+f,l);
#endif
  __syncthreads();
  __shared__ int canonical;
  if(threadIdx.x==0){
    canonical=1;int cursor=0;
#if CF_S_INTERNAL_OWNER_RECEIPT
    const bool owned=producer_receipt[w*4]==1&&producer_receipt[w*4+1]==f+l&&
      producer_receipt[w*4+2]==n&&producer_receipt[w*4+3]==nc;
    if(!owned){
#endif
    for(int c=0;c<nc;c++){
      const int z=w*maxcon+c;if(s01_exclude[z])continue;
      const int d=s01_dim[z];
      if(d<1||d>6||s01_rowstart[z]!=cursor||d>n-cursor){canonical=0;break;}
      cursor+=d;
    }
    if(cursor!=n)canonical=0;
#if CF_S_INTERNAL_OWNER_RECEIPT
    }
#endif
  }
  __syncthreads();
  if(canonical){
#if !CF_S_UNIFIED_METADATA
    libero_vec::copy_block(s01_pos+w*rows,efc_pos+w*rows+f+l,n);
    libero_vec::copy_block(s01_margin+w*rows,efc_margin+w*rows+f+l,n);
    libero_vec::copy_block(s01_loss+w*rows,efc_frictionloss+w*rows+f+l,n);
    libero_vec::copy_block(s01_type+w*rows,efc_type+w*rows+f+l,n);
    libero_vec::copy_block(s01_id+w*rows,efc_id+w*rows+f+l,n);
#endif
    __syncthreads();
  }
  if(threadIdx.x!=0)return;
  if(!canonical)for(int c=0;c<nc;c++){
    int z=w*maxcon+c;if(s01_exclude[z])continue;int d=s01_dim[z],start=s01_rowstart[z];
    if(d<1||d>6){status[w]=5;return;}
    for(int r=0;r<d;r++){
      int local=start+r,g=w*rows+f+l+local,q=w*rows+local;
      if(f+l+local>=total){status[w]=2;return;}
      efc_pos[g]=s01_pos[q];efc_margin[g]=s01_margin[q];efc_frictionloss[g]=s01_loss[q];
      efc_type[g]=s01_type[q];efc_id[g]=s01_id[q];
    }
  }
  counts[w*5]=0;counts[w*5+1]=f;counts[w*5+2]=l;counts[w*5+3]=total;counts[w*5+4]=total*nv;
  ne_flat[w]=0;nf_flat[w]=f;count_flat[w]=total;committed[w]=1;
}
__device__ void CF_S_prediag {
  if(committed[w]!=1)return;const int n=counts[w*5+3];
  __shared__ int canonical;
  if(threadIdx.x==0){
    canonical=1;int previous_type=-1,previous_id=-1;
#if CF_S_INTERNAL_OWNER_RECEIPT
    const bool owned=producer_receipt[w*4]==1&&
      producer_receipt[w*4+1]+producer_receipt[w*4+2]==n&&producer_receipt[w*4+3]==ncon[w];
    if(!owned){
#endif
    for(int i=0;i<n;){
      const int q=w*rows+i,t=efc_type[q],id=efc_id[q];
      int d=t==7?dim[w*S_CONTACTS+id]:1;
      if((t!=1&&t!=3&&t!=5&&t!=7)||d<1||d>n-i){canonical=0;break;}
      // If externally modified followers no longer match the parsed owner,
      // retain the original leader interpretation and partial-write history.
      for(int j=1;j<d;j++)if(efc_type[q+j]!=t||efc_id[q+j]!=id)canonical=0;
      if(t==7&&previous_type==7&&previous_id==id)canonical=0;
      previous_type=t;previous_id=id;
      if(!canonical)break;i+=d;
    }
#if CF_S_INTERNAL_OWNER_RECEIPT
    }
#endif
  }
  __syncthreads();
  if(canonical){
    for(int r=threadIdx.x;r<n;r+=blockDim.x){
      const int q=w*rows+r,t=efc_type[q],id=efc_id[q];
      if(t==1)prediag[q]=dof_inv[id];
      else if(t==3)prediag[q]=dof_inv[jnt_dofadr[id]];
      else if(t!=7||r==0||efc_type[q-1]!=7||efc_id[q-1]!=id){
        // Canonical admission proves a unique group leader and all followers.
        const int b1=geom_body[geom1[w*S_CONTACTS+id]],b2=geom_body[geom2[w*S_CONTACTS+id]];
        const int d=t==7?dim[w*S_CONTACTS+id]:1;
        const real_t tran=body_inv[2*b1]+body_inv[2*b2];
        const int nt=d<3?d:3;
        libero_vec::fill_serial(prediag+q,nt,tran);
        if(d>3){const real_t rot=body_inv[2*b1+1]+body_inv[2*b2+1];
          libero_vec::fill_serial(prediag+q+nt,d-nt,rot);}
      }
    }
    __syncthreads();return;
  }
  if(threadIdx.x!=0)return;
  for(int i=0;i<n;){
    int q=w*rows+i,t=efc_type[q],id=efc_id[q];
    if(t==1){prediag[q]=dof_inv[id];i++;}
    else if(t==3){prediag[q]=dof_inv[jnt_dofadr[id]];i++;}
    else if(t==5||t==7){
      int b1=geom_body[geom1[w*S_CONTACTS+id]],b2=geom_body[geom2[w*S_CONTACTS+id]];
      real_t tran=body_inv[2*b1]+body_inv[2*b2],rot=body_inv[2*b1+1]+body_inv[2*b2+1];
      int d=t==5?1:dim[w*S_CONTACTS+id];for(int j=0;j<d;j++)prediag[q+j]=j<3?tran:rot;i+=d;
    }else{if(!status[w])status[w]=8;committed[w]=0;return;}
    if(i>n){if(!status[w])status[w]=2;committed[w]=0;return;}
  }
}
__device__ void CF_S_commit_position {
  if(ready[w]!=1||status[w])return;int n=sc[w*5+3];
  __shared__ int position_nonfinite;
  __shared__ int position_warp_nonfinite[32];
  bool thread_nonfinite=false;
  const int rowbase=w*rows;
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sp+rowbase,n);
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sm+rowbase,n);
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sf+rowbase,n);
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sd+rowbase,n);
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sr+rowbase,n);
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sD+rowbase,n);
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sPost+rowbase,n);
#if CF_S_KBIP_PLANES
  for(int k=0;k<4;k++)thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sk+(w*4+k)*rows,n);
#else
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sk+rowbase*4,n*4);
#endif
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sj+rowbase*nv,n*nv);
  for(int c=threadIdx.x;c<ncon[w];c+=blockDim.x){int g=w*contacts+c;thread_nonfinite|=!exclude[g]&&dim[g]>=2&&!isfinite(sMu[g]);}
  // All lanes, including those beyond the active rows, participate. One vote
  // per warp replaces atomics per failed row; warp 0 combines those votes.
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  bool warp_nonfinite=libero_warp::any(thread_nonfinite);
  if(lane==0)position_warp_nonfinite[warp]=warp_nonfinite;
  __syncthreads();
  if(warp==0){
    bool block_nonfinite=libero_warp::any(lane<(blockDim.x>>5)&&position_warp_nonfinite[lane]!=0);
    if(lane==0)position_nonfinite=block_nonfinite;
  }
  __syncthreads();
  if(position_nonfinite){if(threadIdx.x==0){status[w]=4;ready[w]=0;}return;}
  // No public write occurs until every row/contact passed. Errors have the
  // original common code 4, so a shared OR preserves deterministic reporting.
  libero_vec::copy_block(sj+rowbase*nv,pj+rowbase*nv,n*nv);
  libero_vec::copy_block(st+rowbase,pt+rowbase,n);
  libero_vec::copy_block(si+rowbase,pi+rowbase,n);
  libero_vec::copy_block(sp+rowbase,pp+rowbase,n);
  libero_vec::copy_block(sm+rowbase,pm+rowbase,n);
  libero_vec::copy_block(sf+rowbase,pf+rowbase,n);
  libero_vec::copy_block(sd+rowbase,pd+rowbase,n);
  libero_vec::copy_block(sr+rowbase,pr+rowbase,n);
  libero_vec::copy_block(sD+rowbase,pD+rowbase,n);
  libero_vec::copy_block(sPost+rowbase,pPost+rowbase,n);
  libero_vec::copy_block(sRoute+rowbase,pRoute+rowbase,n);
#if CF_S_KBIP_PLANES
  for(int r=threadIdx.x;r<n;r+=blockDim.x){
    const int g=rowbase+r;
    libero_vec::store_pair(pk+g*4,sk[kbip_address(w,r,0,rows)],sk[kbip_address(w,r,1,rows)]);
    libero_vec::store_pair(pk+g*4+2,sk[kbip_address(w,r,2,rows)],sk[kbip_address(w,r,3,rows)]);
  }
#else
  libero_vec::copy_block(sk+rowbase*4,pk+rowbase*4,n*4);
#endif
  for(int c=threadIdx.x;c<ncon[w];c+=blockDim.x){int g=w*contacts+c;if(!exclude[g]){pAddress[g]=sAddress[g];if(dim[g]>=2)pMu[g]=sMu[g];}}
  __syncthreads();
  if(threadIdx.x==0){for(int j=0;j<5;j++)pc[w*5+j]=sc[w*5+j];committed[w]=1;}
}
__device__ void CF_S_reference_gate {
  if(threadIdx.x==0)ready[w]=0;
  __syncthreads();if(!active[w])return;
  if(status[w]){if(threadIdx.x==0)committed[w]=0;return;}
  if(active[w]!=1){if(threadIdx.x==0){status[w]=1;committed[w]=0;}return;}
  if(committed[w]!=1){if(threadIdx.x==0){status[w]=9;committed[w]=0;}return;}
  int n=counts[w*5+3];if(n<0||n>rows){if(threadIdx.x==0){status[w]=2;committed[w]=0;}return;}
  if(!n)return;
  if(epoch[w]!=expected[w]){if(threadIdx.x==0){status[w]=9;committed[w]=0;}return;}
  bool bad=false;
  for(int j=threadIdx.x;j<nv;j+=blockDim.x)bad|=!isfinite(qvel[w*nv+j]);
  if(s1_block_any(bad)){if(threadIdx.x==0){status[w]=4;committed[w]=0;}return;}
  if(threadIdx.x==0)ready[w]=1;
}
// Definition supplied by constraints_reference.cuh: cuBLASDx GEMV tiles.
__device__ void CF_S_reference;
__device__ void CF_S_commit_reference {
  if(ready[w]!=1||status[w])return;int n=counts[w*5+3];
  __shared__ int reference_nonfinite;
  __shared__ int reference_warp_nonfinite[32];
  bool thread_nonfinite=false;
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sv+w*rows,n);
  thread_nonfinite|=libero_vec::any_nonfinite_block_lane(sa+w*rows,n);
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  bool warp_nonfinite=libero_warp::any(thread_nonfinite);
  if(lane==0)reference_warp_nonfinite[warp]=warp_nonfinite;
  __syncthreads();
  if(warp==0){
    bool block_nonfinite=libero_warp::any(lane<(blockDim.x>>5)&&reference_warp_nonfinite[lane]!=0);
    if(lane==0)reference_nonfinite=block_nonfinite;
  }
  __syncthreads();
  if(reference_nonfinite){if(threadIdx.x==0){status[w]=4;committed[w]=ready[w]=0;}return;}
  libero_vec::copy_block(sv+w*rows,pv+w*rows,n);
  libero_vec::copy_block(sa+w*rows,pa+w*rows,n);
  __syncthreads();
  if(threadIdx.x==0)ref_epoch[w]=epoch[w];
}
} // namespace cf_s
