#include "precision.cuh"
// Handwritten forward protocol and S-to-Newton projection. No physics oracle.
#pragma once
#include <stdint.h>
#include <math.h>
namespace forward_cuda {
static __device__ inline void increment(int w,const int* mask,const int* first,int64_t* expected) {
    if(mask[w]==1 && first[w]==0) ++expected[w];
}
static __device__ inline void collision_mask(int w,const int* active,const int* valid,
    const int64_t* written,const int64_t* expected,const int* first,int* result) {
    result[w]=active[w]==1 && valid[w]==1 && written[w]==expected[w] && first[w]==0;
}
static __device__ inline void reference_mask(int w,const int* valid,const int64_t* written,
    const int64_t* expected,const int* first,int* active) {
    if(valid[w]!=1 || written[w]!=expected[w] || first[w]!=0) active[w]=0;
}
static __device__ inline void invalidate(int w,int* valid) { valid[w]=0; }
static __device__ inline void publish(int w,const int* active,const int* nvalid,const int* nerror,
    const int* first,const int* projection,const int64_t* source,const int64_t* state,
    int* valid,int64_t* written,int* error) {
    valid[w]=0; error[w]=first[w];
    if(!error[w]) error[w]=nerror[w];
    if(!error[w]) error[w]=projection[w];
    if(active[w]==1 && !error[w] && nvalid[w]==1 && source[w]==state[w]) {
        written[w]=source[w]; valid[w]=1;
    }
}
static __device__ inline void begin(int w,const int* mask,int* error,const int64_t* state,
    int64_t* source,int64_t* epoch,int* active,int* position,int* velocity,
    int* bias,int* smooth,int64_t* expected) {
    active[w]=0;
    if(mask[w]==0) { increment(w,mask,error,expected); return; }
    position[w]=velocity[w]=bias[w]=smooth[w]=0;
    if(!error[w]) {
        if(mask[w]!=1 || state[w]<0) error[w]=201;
        else if(epoch[w]==INT64_MAX) error[w]=9;
        else { ++epoch[w]; source[w]=state[w]; active[w]=1; }
    }
    increment(w,mask,error,expected);
}
static __device__ inline void newton_mask(int w,const int* active,const int* first,
    const int* smooth,const int* reference,const int* status,int* admitted,int* error) {
    admitted[w]=0; error[w]=first[w];
    if(!error[w]) error[w]=status[w];
    if(active[w]==1 && !error[w] && smooth[w]==1 && reference[w]==1) admitted[w]=1;
}
static __device__ inline void counts(int w,const int* admitted,const int* source,const int* source_ncon,
    int* nefc,int* ne,int* nf,int* ncon,int* count,int* empty,int* newton,int* active,
    int* status,int* item,int rows,int contacts,int nv) {
    nefc[w]=ne[w]=nf[w]=ncon[w]=count[w]=0;
    empty[w]=newton[w]=active[w]=status[w]=0; item[w]=-1;
    if(admitted[w]!=1) return;
    ne[w]=source[w*5]; nf[w]=source[w*5+1];
    nefc[w]=count[w]=source[w*5+3]; ncon[w]=source_ncon[w];
    if(!nefc[w]) empty[w]=1;
    else if(nefc[w]>0) newton[w]=active[w]=1;
    int n=nefc[w],eq=ne[w],fr=nf[w],nl=source[w*5+2],nc=ncon[w];
    int bad=-1;
    if(n<0 || n>rows) bad=3;
    else if(eq<0 || eq>n) bad=0;
    else if(fr<0 || fr>n-eq) bad=1;
    else if(nl<0 || nl>n-eq-fr) bad=2;
    else if(source[w*5+4]!=n*nv) bad=4;
    else if(nc<0 || nc>contacts) bad=5;
    if(bad>=0) { status[w]=9001; item[w]=bad; }
}
static __device__ inline void cone_heads(int w,const int* admitted,const int* nefc,const int* ne,
    const int* nf,const int* ncon,const int* type,const int* id,const int* dim,
    const real_t* mu,const real_t* friction,int* cone_dim,real_t* cone_mu,
    real_t* cone_friction,int* status,int* item,int rows,int contacts) {
    if(admitted[w]!=1 || status[w]) return;
    int n=nefc[w],i=ne[w]+nf[w];
    if(n<0 || n>rows || ne[w]<0 || nf[w]<0 || i<0 || i>n) {
        status[w]=9001; item[w]=3; return;
    }
    if(ncon[w]<0 || ncon[w]>contacts) { status[w]=9001; item[w]=5; return; }
    const int off=w*rows,coff=w*contacts;
    while(i<n) {
        int t=type[off+i],c=id[off+i];
        if(t==7) {
            if(c<0 || c>=ncon[w]) {status[w]=9002;item[w]=i;return;}
            int d=dim[coff+c];
            if(d!=1 && d!=3 && d!=4 && d!=6) {status[w]=9003;item[w]=i;return;}
            if(d>n-i) {status[w]=9004;item[w]=i;return;}
            for(int k=0;k<d;++k) if(type[off+i+k]!=7 || id[off+i+k]!=c) {
                status[w]=9004;item[w]=i+k;return;
            }
            if(!isfinite(mu[coff+c]) || mu[coff+c]<=REAL_C(0.)) {status[w]=9006;item[w]=i;return;}
            for(int k=0;k<d-1;++k) {
                real_t v=friction[(coff+c)*5+k];
                if(!isfinite(v) || v<=REAL_C(0.)) {status[w]=9007;item[w]=i;return;}
            }
            i+=d;
        } else if(t>=0 && t<=6) {
            if(t==5) {
                if(c<0 || c>=ncon[w]) {status[w]=9002;item[w]=i;return;}
                if(dim[coff+c]!=1) {status[w]=9003;item[w]=i;return;}
            }
            ++i;
        } else {status[w]=9005;item[w]=i;return;}
    }
    // Validate every row first, preserving all cone history on malformed input.
    for(i=ne[w]+nf[w];i<n;) {
        if(type[off+i]==7) {
            int c=id[off+i],d=dim[coff+c];
            cone_dim[off+i]=d; cone_mu[off+i]=mu[coff+c];
            for(int k=0;k<d-1;++k) cone_friction[(off+i)*5+k]=friction[(coff+c)*5+k];
            i+=d;
        } else ++i;
    }
}
// Foundation gates retain private singleton protocol indices and public world.
static __device__ inline void gate_copy(int w,const int* admission,const int* count,
    const int64_t* epoch,const int64_t* expected,const int* producer_valid,
    const int64_t* event,int* errors,int* stage_out,int* local_out,int* item_out,
    int64_t* event_out,int* active,int* valid,int capacity,int stage,
    int* admitted,int* public_error,int* public_stage,int* public_local,
    int* public_item,int64_t* public_event,int world) {
    active[w]=valid[w]=0;
    int kind=0,local=0,item=-1;
    if(!errors[w] && admission[w]!=0) {
        if(admission[w]!=1) {kind=1;local=admission[w];}
        else if(count[w]<0) {kind=2;local=count[w];}
        else if(count[w]>capacity) {kind=3;local=count[w];item=capacity;}
        else if(producer_valid[w]!=1 || epoch[w]!=expected[w] || epoch[w]<0 || expected[w]<0) {
            kind=7;local=producer_valid[w];
        } else active[w]=1;
        if(kind) {errors[w]=kind;stage_out[w]=stage;local_out[w]=local;item_out[w]=item;event_out[w]=event[w];}
    }
    int out=world+w; admitted[out]=active[w]; public_error[out]=errors[w];
    public_stage[out]=stage_out[w]; public_local[out]=local_out[w];
    public_item[out]=item_out[w]; public_event[out]=event_out[w];
}
} // namespace forward_cuda
