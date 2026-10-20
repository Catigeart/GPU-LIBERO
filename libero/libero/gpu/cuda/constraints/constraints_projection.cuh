#include "precision.cuh"
// Typed S -> N metadata projection. Numeric rows already share their owners.
#pragma once
namespace cf_s {
__device__ void n_mask(int w,const int*active,const int*first,const int*smooth,
 const int*reference,const int*sstatus,int*admitted,int*error){
  admitted[w]=0;error[w]=first[w];if(!error[w]&&sstatus[w])error[w]=sstatus[w];
  if(active[w]==1&&!error[w]&&smooth[w]==1&&reference[w]==1)admitted[w]=1;
}
__device__ void project_counts(int w,const int*admitted,const int*source_counts,const int*source_ncon,
 int*nefc,int*ne,int*nf,int*ncon,int*count,int*empty,int*newton,int*active,int*status,int*item,int rows,int contacts,int nv){
  nefc[w]=ne[w]=nf[w]=ncon[w]=count[w]=empty[w]=newton[w]=active[w]=status[w]=0;item[w]=-1;
  if(admitted[w]!=1)return;
  ne[w]=source_counts[w*5];nf[w]=source_counts[w*5+1];nefc[w]=count[w]=source_counts[w*5+3];ncon[w]=source_ncon[w];
  if(!nefc[w])empty[w]=1;else if(nefc[w]>0)newton[w]=active[w]=1;
  int n=nefc[w],eq=ne[w],fr=nf[w],nl=source_counts[w*5+2],nc=ncon[w],bad=-1;
  if(n<0||n>rows)bad=3;
  else if(eq<0||eq>n)bad=0;
  else if(fr<0||fr>n-eq)bad=1;
  else if(nl<0||nl>n-eq-fr)bad=2;
  else if(source_counts[w*5+4]!=n*nv)bad=4;
  else if(nc<0||nc>contacts)bad=5;
  if(bad>=0){status[w]=9001;item[w]=bad;}
}
__device__ void project_cone_heads(int w,const int*admitted,const int*nefc,const int*ne,const int*nf,const int*ncon,
 const int*typ,const int*id,const int*dim,const real_t*mu,const real_t*friction,int*out_dim,real_t*out_mu,
 real_t*out_friction,int*status,int*item,int rows,int contacts){
  __shared__ int head[32],contact[32],width[32],kind[32],parse_code[32];
  __shared__ int cursor,used,finished;
  if(threadIdx.x==0){
    finished=0;cursor=ne[w]+nf[w];
    if(admitted[w]!=1||status[w])finished=1;
    else if(nefc[w]<0||nefc[w]>rows||ne[w]<0||nf[w]<0||cursor<0||cursor>nefc[w]){status[w]=9001;item[w]=3;finished=1;}
    else if(ncon[w]<0||ncon[w]>contacts){status[w]=9001;item[w]=5;finished=1;}
  }
  __syncthreads();
  const int rejected=finished;
  __syncthreads();
  if(rejected)return;
  const int n=nefc[w],start=ne[w]+nf[w],rb=w*rows,cb=w*contacts;
  // Two complete passes: no history write precedes global validation success.
  for(int pass=0;pass<2;pass++){
    if(threadIdx.x==0){cursor=start;finished=cursor==n;}
    __syncthreads();
    while(!finished){
      if(threadIdx.x==0){
        used=0;
        while(used<32&&cursor<n){
          const int k=used++,i=cursor,t=typ[rb+i];
          int c=-1,d=1,code=0;
          if(t==7||t==5){
            c=id[rb+i];
            if(c<0||c>=ncon[w])code=9002;
            else{
              d=dim[cb+c];
              if(t==7){if(d!=1&&d!=3&&d!=4&&d!=6)code=9003;else if(d>n-i)code=9004;}
              else if(d!=1)code=9003;
            }
          }else if(t<0||t>6)code=9005;
          head[k]=i;contact[k]=c;width[k]=d;kind[k]=t;parse_code[k]=code;
          if(code)break;
          cursor+=t==7?d:1;
        }
      }
      __syncthreads();
      if(threadIdx.x<32){
        const int k=threadIdx.x;
        int code=k<used?parse_code[k]:0,bad=k<used?head[k]:-1;
        if(k<used&&!code&&kind[k]==7){
          const int i=head[k],c=contact[k],d=width[k];
          if(!pass){
            for(int j=0;j<d;j++)if(typ[rb+i+j]!=7||id[rb+i+j]!=c){code=9004;bad=i+j;break;}
            if(!code&&(!isfinite(mu[cb+c])||mu[cb+c]<=REAL_C(0.)))code=9006;
            if(!code)for(int j=0;j<d-1;j++){const real_t f=friction[(cb+c)*5+j];if(!isfinite(f)||f<=REAL_C(0.)){code=9007;break;}}
          }else{
            out_dim[rb+i]=d;out_mu[rb+i]=mu[cb+c];
            for(int j=0;j<d-1;j++)out_friction[(rb+i)*5+j]=friction[(cb+c)*5+j];
          }
        }
        const int first=libero_warp::min(code?k:libero_warp::no_index);
        const int winner=first==libero_warp::no_index?0:first;
        const int rejection=__shfl_sync(libero_warp::mask,code,winner);
        const int rejection_item=__shfl_sync(libero_warp::mask,bad,winner);
        if(k==0){
          if(first!=libero_warp::no_index){status[w]=rejection;item[w]=rejection_item;finished=1;}
          else if(cursor==n)finished=1;
        }
      }
      __syncthreads();
    }
    __syncthreads();
    if(status[w])return;
  }
}
// This fast entry is bound only to the same S publication whose successful
// impedance invocation rebuilt this ascending list. Generic external callers
// retain project_cone_heads. Projection-specific numeric admission is repeated.
__device__ void project_cone_heads_verified(int w,const int*admitted,const int*nefc,const int*ne,const int*nf,const int*ncon,
 const int*typ,const int*id,const int*dim,const real_t*mu,const real_t*friction,int*out_dim,real_t*out_mu,
 real_t*out_friction,int*status,int*item,int rows,int contacts,
 const int*compact_heads,const int*compact_count,const int*compact_valid,
 const int*head_dim,real_t*head_mu,real_t*head_friction,int*head_cone_valid){
  if(threadIdx.x==0)head_cone_valid[w]=0;
  __syncthreads();
  if(compact_valid[w]!=1||compact_count[w]<0||compact_count[w]>rows){
    project_cone_heads(w,admitted,nefc,ne,nf,ncon,typ,id,dim,mu,friction,
      out_dim,out_mu,out_friction,status,item,rows,contacts);return;
  }
  __shared__ int begin,compatible;
  if(threadIdx.x==0){
    begin=0;compatible=1;
    if(admitted[w]==1&&!status[w]){
      const int n=nefc[w],start=ne[w]+nf[w];
      if(n<0||n>rows||ne[w]<0||nf[w]<0||start<0||start>n){status[w]=9001;item[w]=3;}
      else if(ncon[w]<0||ncon[w]>contacts){status[w]=9001;item[w]=5;}
      else{
        int lo=0,hi=compact_count[w];
        while(lo<hi){const int mid=(lo+hi)/2;if(compact_heads[w*rows+mid]<start)lo=mid+1;else hi=mid;}
        begin=lo;
        // Raw arrays remain externally addressable: prove that this invocation's
        // complete current row partition equals the stored list before reuse.
        // A changed follower, count, dim or start falls back to the generic
        // parser, which retains its original exact error/partial-write rules.
        int cursor=0;bool boundary=start==0;
        for(int ordinal=0;ordinal<compact_count[w]&&compatible;ordinal++){
          const int h=compact_heads[w*rows+ordinal];
          if(h!=cursor||h<0||h>=n){compatible=0;break;}
          if(h==start)boundary=true;
          int d=1;
          if(typ[w*rows+h]==7){
            const int c=id[w*rows+h];
            if(c<0||c>=ncon[w]){compatible=0;break;}
            d=dim[w*contacts+c];
            if(d!=head_dim[w*rows+ordinal]){compatible=0;break;}
            if((d!=1&&d!=3&&d!=4&&d!=6)||d>n-h){compatible=0;break;}
          }
          cursor+=d;
        }
        if(cursor!=n||(!boundary&&start!=n))compatible=0;
      }
    }
  }
  __syncthreads();
  if(admitted[w]!=1||status[w])return;
  if(!compatible){
    project_cone_heads(w,admitted,nefc,ne,nf,ncon,typ,id,dim,mu,friction,
      out_dim,out_mu,out_friction,status,item,rows,contacts);return;
  }
  const int rb=w*rows,cb=w*contacts,n=nefc[w],total=compact_count[w];
  // All group heads retain source order. Followers and scalar-type groups
  // never execute cone arithmetic. A block barrier protects each next batch.
  for(int base=begin;base<total;base+=32){
    __syncthreads();
    if(threadIdx.x<32){
      const int lane=threadIdx.x,ordinal=base+lane;
      int code=0,bad=-1;
      if(ordinal<total){
        const int i=compact_heads[rb+ordinal];bad=i;
        if(i<ne[w]+nf[w]||i>=n){code=9004;}
        else{
          const int t=typ[rb+i];
          if(t==7){
            const int c=id[rb+i];
            if(c<0||c>=ncon[w])code=9002;
            else{
              const int d=dim[cb+c];
              if(d!=1&&d!=3&&d!=4&&d!=6)code=9003;
              else if(d>n-i)code=9004;
              else{
                for(int j=0;j<d;j++)if(typ[rb+i+j]!=7||id[rb+i+j]!=c){code=9004;bad=i+j;break;}
                if(!code&&(!isfinite(mu[cb+c])||mu[cb+c]<=REAL_C(0.)))code=9006;
                if(!code)for(int j=0;j<d-1;j++){const real_t f=friction[(cb+c)*5+j];if(!isfinite(f)||f<=REAL_C(0.)){code=9007;break;}}
              }
            }
          }else if(t==5){
            const int c=id[rb+i];if(c<0||c>=ncon[w])code=9002;else if(dim[cb+c]!=1)code=9003;
          }else if(t<0||t>6)code=9005;
        }
      }
      const int first=libero_warp::min(code?lane:libero_warp::no_index);
      const int winner=first==libero_warp::no_index?0:first;
      const int rejection=__shfl_sync(libero_warp::mask,code,winner);
      const int rejection_item=__shfl_sync(libero_warp::mask,bad,winner);
      if(lane==0&&first!=libero_warp::no_index){status[w]=rejection;item[w]=rejection_item;}
    }
    __syncthreads();
    if(status[w])return;
  }
  __syncthreads();
  for(int ordinal=begin+threadIdx.x;ordinal<total;ordinal+=blockDim.x){
    const int i=compact_heads[rb+ordinal];
    if(typ[rb+i]!=7)continue;
    const int c=id[rb+i],d=dim[cb+c];
    out_dim[rb+i]=d;out_mu[rb+i]=mu[cb+c];
    head_mu[rb+ordinal]=mu[cb+c];
    // Wide contiguous records stay within the validated cone prefix. The
    // transposed private head metadata retains its existing scalar layout.
    int j=0;
    for(;j+1<d-1;j+=2){
      const real2_t f=libero_vec::load_pair(friction+(cb+c)*5+j);
      libero_vec::store_pair(out_friction+(rb+i)*5+j,f.x,f.y);
      head_friction[(w*5+j)*rows+ordinal]=f.x;
      head_friction[(w*5+j+1)*rows+ordinal]=f.y;
    }
    if(j<d-1){const real_t f=friction[(cb+c)*5+j];out_friction[(rb+i)*5+j]=f;head_friction[(w*5+j)*rows+ordinal]=f;}
  }
  __syncthreads();
  if(threadIdx.x==0)head_cone_valid[w]=1;
}
__device__ void increment_plan(int w,const int*mask,const int*first,int64_t*epoch){
  if(mask[w]==1&&!first[w])epoch[w]++;
}
}
