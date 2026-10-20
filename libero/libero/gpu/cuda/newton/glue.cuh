#include "precision.cuh"
#include "io_policy.cuh"
#pragma once
#include <stdint.h>
__device__ void jacvec_sub(int w,int* active,int* n,real_t* J,real_t* q,real_t* aref,real_t* out,int rows){
  if(!active[w])return;for(int r=0;r<n[w];++r)out[w*rows+r]=dot4(J+(w*rows+r)*NV,q+w*NV,NV)-aref[w*rows+r];
}
__device__ void mulM(int w,int* active,real_t* q,real_t* M,int* madr,int* parent,int* simple,real_t* out){
  if(active[w])mass_multiply(q+w*NV,M+w*NM,madr,parent,simple,out+w*NV);
}
// Address-only half-open span test. Unsigned distances avoid end-address
// overflow and avoid relational comparison of unrelated C++ pointers.
__device__ inline bool mass_spans_overlap(const void* a,unsigned long long na,
 const void* b,unsigned long long nb){
  if(!na||!nb)return false;
  const uintptr_t pa=(uintptr_t)a,pb=(uintptr_t)b;
  return pa<=pb?pb-pa<na:pa-pb<nb;
}
// Collective planned product. Preserve original scalar behavior for exact or
// shifted input/output aliases; the qualification does not read any payload.
__device__ void mulM_plan_block(int w,int* active,real_t* q,real_t* M,
 int* madr,int* parent,int* simple,real_t* out,const int* offsets,
 const int* coefficient,const int* source,const int* assign){
  if(!active[w])return;
  real_t* x=q+w*NV;real_t* qm=M+w*NM;real_t* result=out+w*NV;
  const bool disjoint=!mass_spans_overlap(result,NV*sizeof(real_t),x,NV*sizeof(real_t))&&
    !mass_spans_overlap(result,NV*sizeof(real_t),qm,NM*sizeof(real_t))&&
    !mass_spans_overlap(result,NV*sizeof(real_t),madr,NV*sizeof(int))&&
    !mass_spans_overlap(result,NV*sizeof(real_t),parent,NV*sizeof(int))&&
    !mass_spans_overlap(result,NV*sizeof(real_t),simple,NV*sizeof(int));
  if(offsets&&disjoint)
    libero_packed_mass::multiply_plan(x,qm,offsets,coefficient,source,assign,result,NV);
  else if(threadIdx.x==0)mass_multiply(x,qm,madr,parent,simple,result);
}
// The warm and smooth candidates have already consumed exactly these operands.
// Reuse the selected candidate, without replacing M*q by an approximate force.
__device__ void warm_init_jacvec(int* active,int* count,int* choice,
 real_t* warm,real_t* smooth,real_t* out){
  if(!active[0])return;const real_t* selected=choice[0]?smooth:warm;
  for(int r=threadIdx.x;r<count[0];r+=blockDim.x)out[r]=selected[r];
}
__device__ void warm_init_mass(int* active,int* choice,real_t* warm,
 real_t* q,real_t* M,int* madr,int* parent,int* simple,real_t* out,
 const int* offsets=nullptr,const int* coefficient=nullptr,
 const int* source=nullptr,const int* assign=nullptr){
  if(!active[0])return;
  if(!choice[0]){for(int j=threadIdx.x;j<NV;j+=blockDim.x)out[j]=warm[j];}
  else mulM_plan_block(0,active,q,M,madr,parent,simple,out,offsets,coefficient,source,assign);
}
__device__ void gauss_compute(int w,int* active,real_t* Ma,real_t* smooth,real_t* q,real_t* qs,real_t* cost,real_t* quad0,int* add){
  if(!active[w])return;real_t s=0;for(int j=0;j<NV;++j){int i=w*NV+j;s+=REAL_C(.5)*(Ma[i]-smooth[i])*(q[i]-qs[i]);}quad0[w]=s;if(add[w])cost[w]+=s;
}

// A nonstandard scalar objective over two difference vectors; no BLAS matrix
// operation is replaced. Small vectors stay in warp 0; each lane accumulates
// one DOF. Larger vectors retain the measured block reduction. The caller's
// stage barrier publishes the lane-0 result.
__device__ void gauss_compute_block(int w,int* active,real_t* Ma,real_t* smooth,real_t* q,real_t* qs,real_t* cost,real_t* quad0,int* add){
  __shared__ real_t warp_partial[4];
  if(!active[w])return;real_t value=REAL_C(0.);
  if(NV<=32){
    if(threadIdx.x>=32)return;
    for(int j=threadIdx.x;j<NV;j+=32){int i=w*NV+j;value+=REAL_C(.5)*(Ma[i]-smooth[i])*(q[i]-qs[i]);}
    value=libero_warp::sum(value);
    if(threadIdx.x==0){quad0[w]=value;if(add[w])cost[w]+=value;}
    return;
  }
  for(int j=threadIdx.x;j<NV;j+=blockDim.x){int i=w*NV+j;value+=REAL_C(.5)*(Ma[i]-smooth[i])*(q[i]-qs[i]);}
  value=libero_warp::sum(value);
  const int lane=libero_warp::lane(),wid=libero_warp::warp(),warps=blockDim.x/32;
  if(lane==0)warp_partial[wid]=value;
  __syncthreads();
  if(wid==0){value=libero_warp::sum(lane<warps?warp_partial[lane]:REAL_C(0.));
    if(lane==0){quad0[w]=value;if(add[w])cost[w]+=value;}
  }
}

__device__ void count_active_cone(int w,int* active,int* state,int* n,int* na,int* nc,int rows){
  if(!active[w])return;int a=0,c=0;for(int j=0;j<n[w];++j){int s=state[w*rows+j];if(s)++a;if(s==4)++c;}na[w]=a;nc[w]=c;
}
__device__ void save_old(int w,int* active,int* n,int* state,real_t* cost,int* old,real_t* oldcost,int rows){
#if LIBERO_NEWTON_SCALAR_IO
  if(!active[w])return;for(int j=0;j<n[w];++j)old[w*rows+j]=state[w*rows+j];oldcost[w]=cost[w];
#else
  if(!active[w])return;
  if(old==state||n_state_disjoint(old+w*rows,size_t(n[w])*4,state+w*rows,size_t(n[w])*4))
    libero_newton_io::copy_serial(state+w*rows,old+w*rows,n[w]);
  else for(int j=0;j<n[w];++j)old[w*rows+j]=state[w*rows+j];
  oldcost[w]=cost[w];
#endif // LIBERO_NEWTON_SCALAR_IO
}
__device__ void gate_ls(int w,int* active,real_t* alpha,int* status,int* reason,int* error){
  if(!active[w])return;if(status[w]){if(!error[w])error[w]=1000+status[w];active[w]=0;return;}if(alpha[w]==0){reason[w]=10;active[w]=0;}
}
__device__ void gate_status(int w,int* active,int* status,int* base,int* error){
  if(active[w]&&status[w]){if(!error[w])error[w]=base[w]+status[w];active[w]=0;}
}
__device__ void gate_stats(int w,int* active,int* stop,int* out_reason,int* status,int* reason,int* error){
  if(!active[w])return;if(status[w]){if(!error[w])error[w]=3000+status[w];active[w]=0;return;}
  if(stop[w]&&out_reason[w]==1){reason[w]=1;active[w]=0;}
}
__device__ void gate_maxiter(int w,int* active,int* stop,int* out_reason,int* reason){
  if(active[w]&&stop[w]&&out_reason[w]==2){reason[w]=2;active[w]=0;}
}
__device__ void direction_commit(int w,int* active,real_t* candidate,real_t* search){if(active[w])for(int j=0;j<NV;++j)search[w*NV+j]=candidate[w*NV+j];}
__device__ void empty_fast_path(int w,int* empty,real_t* smooth,real_t* q,real_t* warm,real_t* force,int* iter){
  if(!empty[w])return;for(int j=0;j<NV;++j){int i=w*NV+j;q[i]=smooth[i];warm[i]=smooth[i];force[i]=REAL_C(0.0);}iter[w]=0;
}
#if LIBERO_NEWTON_SCALAR_IO
__device__ void copy_qacc(int w,int* active,real_t* src,real_t* dst){if(active[w])for(int j=0;j<NV;++j)dst[w*NV+j]=src[w*NV+j];}
#else
__device__ void copy_qacc(int w,int* active,real_t* src,real_t* dst){if(active[w]){
  if(src==dst||n_state_disjoint(src+w*NV,NV*sizeof(real_t),dst+w*NV,NV*sizeof(real_t)))
    libero_newton_io::copy_serial(src+w*NV,dst+w*NV,NV);
  else for(int j=0;j<NV;++j)dst[w*NV+j]=src[w*NV+j];}}
#endif // LIBERO_NEWTON_SCALAR_IO
__device__ void copy_cost(int w,int* active,real_t* src,real_t* dst){if(active[w])dst[w]=src[w];}
#if LIBERO_NEWTON_SCALAR_IO
__device__ void zero_rows(int w,int* active,int* n,real_t* out,int rows){if(active[w])for(int j=0;j<n[w];++j)out[w*rows+j]=REAL_C(0.0);}
#else
__device__ void zero_rows(int w,int* active,int* n,real_t* out,int rows){if(active[w])libero_newton_io::fill_serial(out+w*rows,n[w],REAL_C(0.0));}
#endif // LIBERO_NEWTON_SCALAR_IO
__device__ void solver_iter_reset(int w,int* active,int* iter){if(active[w])iter[w]=0;}
__device__ void finalize_exit(int w,int* newton,int* error,int* exit,int* active){exit[w]=active[w]=(newton[w]!=0&&error[w]==0)?1:0;}
__device__ void seed_eval(int w,real_t* eval,int* tag){for(int j=0;j<52;++j){tag[w*52+j]=-101;for(int c=0;c<4;++c)eval[(w*52+j)*4+c]=REAL_C(97.0);}}
__device__ void tape_ls_input(int w,int slot,int slots,real_t* quad0,real_t* tape){if(slot<slots)tape[w*slots+slot]=quad0[w];}
__device__ void parallel_tape_ls_buffers(int tid,int slot,int slots,int rows,real_t* mv,real_t* jv,real_t* quad,real_t* qg,
 real_t* tm,real_t* tj,real_t* tq,real_t* tg,int nv){
  if(slot>=slots)return;int stride=nv+4*rows+3,w=tid/stride,i=tid%stride;
  if(i<nv)tm[(w*slots+slot)*nv+i]=mv[w*nv+i];
  else if(i<nv+rows)tj[(w*slots+slot)*rows+i-nv]=jv[w*rows+i-nv];
  else if(i<nv+4*rows)tq[(w*slots+slot)*rows*3+i-nv-rows]=quad[w*rows*3+i-nv-rows];
  else tg[(w*slots+slot)*3+i-nv-4*rows]=qg[w*3+i-nv-4*rows];
}
// An inactive source slot repeats these same buffers, including capacity tails.
// Load each source cell once and fan it out; never omit observable history cells.
__device__ void inactive_tape_ls_buffers(int begin,int end,int slots,int rows,
 real_t* mv,real_t* jv,real_t* quad,real_t* qg,real_t* tm,real_t* tj,real_t* tq,real_t* tg,int nv){
#if LIBERO_NEWTON_SCALAR_IO
  if(end>slots)end=slots;int stride=nv+4*rows+3;
  for(int i=threadIdx.x;i<stride;i+=blockDim.x){
    real_t value;real_t* dst;int width,cell;
    if(i<nv){value=mv[i];dst=tm;width=nv;cell=i;}
    else if(i<nv+rows){cell=i-nv;value=jv[cell];dst=tj;width=rows;}
    else if(i<nv+4*rows){cell=i-nv-rows;value=quad[cell];dst=tq;width=3*rows;}
    else{cell=i-nv-4*rows;value=qg[cell];dst=tg;width=3;}
    for(int slot=begin;slot<end;++slot)dst[slot*width+cell]=value;
#else
  if(end>slots)end=slots;
  // Each packet stays within one history field and each destination slot.
  // Capacity tails are exported even after the world becomes inactive.
  for(int field=0;field<4;++field){
    const real_t* src=field==0?mv:(field==1?jv:(field==2?quad:qg));
    real_t* dst=field==0?tm:(field==1?tj:(field==2?tq:tg));
    const int width=field==0?nv:(field==1?rows:(field==2?3*rows:3));
    int i=2*int(threadIdx.x);
    for(;i+1<width;i+=2*int(blockDim.x)){
      const real2_t value=libero_vec::load_pair(src+i);
      for(int slot=begin;slot<end;++slot)libero_vec::store_pair(dst+slot*width+i,value.x,value.y);
    }
    if(i<width){const real_t value=src[i];for(int slot=begin;slot<end;++slot)dst[slot*width+i]=value;}
#endif // LIBERO_NEWTON_SCALAR_IO
  }
}
__device__ void tape_commit(int w,int slot,int slots,int* active,real_t* alpha,real_t* slope,real_t* lineslope,
 real_t* improvement,real_t* gradient,real_t* cost,real_t* oldcost,real_t* quad0,int* LSiter,int* LSresult,int* committed,
 int* route,int* nupdate,int* nactive,int* ncone,int* nchange,int* iteration,int* reason,int* error,int* s07,int* s08b,int* stage_status,int* s08c,
 real_t* tf,int* ti,int* ts,real_t* eval,int* tag,real_t* te,int* tt,bool export_trace=true){
  if(slot>=slots)return;int f=(w*slots+slot)*8,i=(w*slots+slot)*12,s=(w*slots+slot)*6;
  tf[f]=alpha[w];tf[f+1]=slope[w];tf[f+2]=lineslope[w];tf[f+3]=improvement[w];tf[f+4]=gradient[w];tf[f+5]=cost[w];tf[f+6]=oldcost[w];tf[f+7]=quad0[w];
  ti[i]=active[w];ti[i+1]=LSiter[w];ti[i+2]=LSresult[w];ti[i+3]=committed[w];ti[i+4]=route[w];ti[i+5]=nupdate[w];ti[i+6]=nactive[w];ti[i+7]=ncone[w];ti[i+8]=nchange[w];ti[i+9]=iteration[w];ti[i+10]=reason[w];ti[i+11]=error[w];
  ts[s]=s07[w];ts[s+1]=s08b[w];ts[s+2]=stage_status[w];ts[s+3]=s08c[w];
  if(export_trace)for(int e=0;e<52;++e){for(int c=0;c<4;++c)te[((w*slots+slot)*52+e)*4+c]=eval[(w*52+e)*4+c];tt[(w*slots+slot)*52+e]=tag[w*52+e];}
}
// Collective trace export; leader retains the small mixed scalar record.
// Caller enters uniformly and supplies its publication barrier. Status tail
// cells ts[4:6] remain untouched exactly as the scalar history writer does.
__device__ void tape_commit_block(int w,int slot,int slots,int* active,real_t* alpha,real_t* slope,real_t* lineslope,
 real_t* improvement,real_t* gradient,real_t* cost,real_t* oldcost,real_t* quad0,int* LSiter,int* LSresult,int* committed,
 int* route,int* nupdate,int* nactive,int* ncone,int* nchange,int* iteration,int* reason,int* error,int* s07,int* s08b,int* stage_status,int* s08c,
 real_t* tf,int* ti,int* ts,real_t* eval,int* tag,real_t* te,int* tt){
  if(slot>=slots)return;
  if(threadIdx.x==0)tape_commit(w,slot,slots,active,alpha,slope,lineslope,improvement,gradient,cost,oldcost,quad0,LSiter,LSresult,committed,route,nupdate,nactive,ncone,nchange,iteration,reason,error,s07,s08b,stage_status,s08c,tf,ti,ts,eval,tag,te,tt,false);
#if LIBERO_NEWTON_SCALAR_IO
  for(int e=threadIdx.x;e<52*4;e+=blockDim.x)
    te[(w*slots+slot)*52*4+e]=eval[w*52*4+e];
  for(int e=threadIdx.x;e<52;e+=blockDim.x)
    tt[(w*slots+slot)*52+e]=tag[w*52+e];
#else
  libero_newton_io::copy_block(eval+w*52*4,te+(w*slots+slot)*52*4,52*4);
  libero_newton_io::copy_block(tag+w*52,tt+(w*slots+slot)*52,52);
#endif // LIBERO_NEWTON_SCALAR_IO
}
__device__ void validate_entry_counts(int w,int* n,int* ne,int* nf,int* nc,int* empty,int* newton,int* active,int* error,int* item,int rows,int contacts){
  int bad=-1;if(n[w]<0||n[w]>rows)bad=0;else if(nc[w]<0||nc[w]>contacts)bad=3;else if(ne[w]<0||ne[w]>n[w])bad=1;else if(nf[w]<0||nf[w]>n[w]-ne[w])bad=2;
  if(bad>=0){if(!error[w]){error[w]=4002;item[w]=bad;}empty[w]=newton[w]=active[w]=0;}
}
__device__ void validate_contact_rows(int w,int* n,int* ne,int* nf,int* nc,int* typ,int* cid,int* rd,real_t* rm,real_t* rf,
 int* dim,real_t* mu,real_t* fr,int* newton,int* active,int* error,int* item,int rows,int contacts){
  if(!newton[w]||error[w])return;int rb=w*rows,cb=w*contacts;
  for(int i=ne[w]+nf[w];i<n[w];){int t=typ[rb+i];if(t==7){int c=cid[rb+i];if(c<0||c>=nc[w])return;int d=dim[cb+c];real_t m=mu[cb+c];
    if((d!=1&&d!=3&&d!=4&&d!=6)||i+d>n[w]||m<=0||!isfinite(m))return;
    for(int j=0;j<d;++j)if(typ[rb+i+j]!=7||cid[rb+i+j]!=c)return;
    for(int j=0;j<d-1;++j)if(fr[(cb+c)*5+j]<=0||!isfinite(fr[(cb+c)*5+j]))return;i+=d;
  }else if(t>=0&&t<=6)++i;else return;}
  for(int i=ne[w]+nf[w];i<n[w];){int r=rb+i;if(typ[r]!=7){++i;continue;}int c=cb+cid[r],d=dim[c];bool bad=rd[r]!=d||rm[r]!=mu[c];
    for(int j=0;j<d-1;++j)if(rf[r*5+j]!=fr[c*5+j])bad=true;
    if(bad){error[w]=8001;item[w]=i;active[w]=newton[w]=0;return;}i+=d;
  }
}
__device__ void final_event(int w,real_t* q,int* empty,int* newton,int* exit,int* error,int* status,int64_t* attempted,
 real_t* candidate,int64_t* stamp,int* completed,int* origin,int* stage,int* rejected,int nv){
  completed[w]=origin[w]=stage[w]=rejected[w]=0;stamp[w]=attempted[w];
  if(empty[w]==1&&!error[w]){completed[w]=origin[w]=stage[w]=1;}
  else if(newton[w]==1&&exit[w]==1){stage[w]=2;if(!error[w]&&!status[w])completed[w]=origin[w]=1;else if(status[w]==3)rejected[w]=1;}
  if(completed[w]||rejected[w])for(int j=0;j<nv;++j)candidate[w*nv+j]=q[w*nv+j];
}
__device__ void publish(int w,const int* upstream,const int* enabled,int* error,int* valid,int* written){if(enabled[w]&&!upstream[w]&&!error[w])valid[w]=written[w]=1;}
__device__ void publication_gate(int w,int* valid,int* diagnostic){if(diagnostic[w])valid[w]=0;}
__device__ void finish_publication(int w,int* valid,int* written,int64_t* expected,int64_t* epoch){written[w]=valid[w];if(valid[w])epoch[w]=expected[w];}


// Parse dimensions in order, rather than inferring groups from cid changes:
// adjacent groups may reference the same contact. No row metadata is inspected
// by a lane until the entire schema has passed the original ordered guard.
__device__ void validate_contact_rows_block(int w,int* n,int* ne,int* nf,int* nc,int* typ,int* cid,int* rd,real_t* rm,real_t* rf,
 int* dim,real_t* mu,real_t* fr,int* newton,int* active,int* error,int* item,int rows,int contacts){
  if(!newton[w]||error[w])return;
  __shared__ int schema_valid,heads[32],head_count,next,first_bad;
  const int rb=w*rows,cb=w*contacts;
  if(threadIdx.x==0){
    schema_valid=1;
    for(int i=ne[w]+nf[w];i<n[w];){
      const int t=typ[rb+i];
      if(t==7){const int c=cid[rb+i];
        if(c<0||c>=nc[w]){schema_valid=0;break;}
        const int d=dim[cb+c];const real_t m=mu[cb+c];
        if((d!=1&&d!=3&&d!=4&&d!=6)||i+d>n[w]||m<=0||!isfinite(m)){schema_valid=0;break;}
        for(int j=0;j<d;++j)if(typ[rb+i+j]!=7||cid[rb+i+j]!=c){schema_valid=0;break;}
        if(!schema_valid)break;
        for(int j=0;j<d-1;++j)if(fr[(cb+c)*5+j]<=0||!isfinite(fr[(cb+c)*5+j])){schema_valid=0;break;}
        if(!schema_valid)break;i+=d;
      }else if(t>=0&&t<=6)++i;else{schema_valid=0;break;}
    }
    next=ne[w]+nf[w];
  }
  __syncthreads();if(!schema_valid)return;
  for(;;){
    if(threadIdx.x==0){head_count=0;first_bad=2147483647;
      while(next<n[w]&&head_count<32){const int i=next;
        if(typ[rb+i]!=7){++next;continue;}
        heads[head_count++]=i;next+=dim[cb+cid[rb+i]];}}
    __syncthreads();if(head_count==0)return;
    if(threadIdx.x<32){const int lane=libero_warp::lane();int bad_row=2147483647;
      if(lane<head_count){const int i=heads[lane],r=rb+i,c=cb+cid[r],d=dim[c];
        bool bad=rd[r]!=d||rm[r]!=mu[c];
        for(int j=0;j<d-1;++j)bad|=rf[r*5+j]!=fr[c*5+j];
        if(bad)bad_row=i;}
      bad_row=libero_warp::min(bad_row);if(lane==0)first_bad=bad_row;
    }
    __syncthreads();
    if(first_bad!=2147483647){if(threadIdx.x==0){error[w]=8001;item[w]=first_bad;active[w]=newton[w]=0;}return;}
  }
}
