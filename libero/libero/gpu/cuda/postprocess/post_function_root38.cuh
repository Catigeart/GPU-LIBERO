#include "precision.cuh"
#pragma once
#include "post_contact_index_original.cuh"
namespace post_root38_original { using namespace libero_post;
// Each source view is guarded independently; workspace starts require only
// double alignment. Plane-to-AoS pairs intentionally use separate scalar reads.
__device__ __forceinline__ bool post_finite(const real_t* p,int n){
 return !libero_vec::any_nonfinite_lane(p,n,0,1);
}
__device__ __forceinline__ void post_export(const real_t* src,real_t* dst,int n,int b,bool planes){
 if(!planes){libero_vec::copy_block(src,dst,n);return;}
 for(int pair=threadIdx.x;pair<n/2;pair+=blockDim.x){
  const int k=2*pair;
  const real_t x=src[(k%6)*b+k/6],y=src[((k+1)%6)*b+(k+1)/6];
  libero_vec::store_pair(dst+k,x,y);
 }
 if((n&1)&&threadIdx.x==0){const int k=n-1;dst[k]=src[(k%6)*b+k/6];}
}
// Exactly count valid values of an enabled, schema-validated contact only.
__device__ __forceinline__ void post_decode(const real_t* ef,int count,real_t* d){
 int k=0;
 for(;k+1<count;k+=2){const real2_t p=libero_vec::load_pair(ef+k);d[k]=p.x;d[k+1]=p.y;}
 if(k<count)d[k]=ef[k];
}
__device__ inline void rne_shared_scalar(int w,const int* active,const real_t* qv,const real_t* qa,const real_t* ci,const real_t* cv,const real_t* cd,const real_t* cdd,const real_t* xi,const real_t* com,const real_t* sp,const real_t* sm,const real_t* xf,const real_t* ef,const int* nc,const int* ne,const int* neq,const int* dim,const int* adr,const int* g1,const int* g2,const real_t* frame,const real_t* pos,const int* parent,const int* da,const int* dn,const int* root,const int* gb,const real_t* grav,int cc,int ec,real_t* acc,real_t* ext,real_t* intr,real_t* decoded,int* status,int b,int v,int ns,int ng,bool local_outputs,bool diagnostics,bool qacc_world_stride,const real_t* bias_acc,const real_t* bias_force,bool cache_ok,bool validated=false){
 status[w]=0;qv+=w*v;if(qacc_world_stride)qa+=w*v;ci+=w*b*10;cv+=w*b*6;cd+=w*v*6;cdd+=w*v*6;xi+=w*b*3;com+=w*b*3;sp+=w*ns*3;sm+=w*ns*9;xf+=w*b*6;ef+=w*ec;dim+=w*cc;adr+=w*cc;g1+=w*cc;g2+=w*cc;frame+=w*cc*9;pos+=w*cc*3;if(!local_outputs){acc+=w*b*6;ext+=w*b*6;intr+=w*b*6;}decoded+=w*cc*6;
 if(!validated){libero_vec::fill_serial(acc,b*6,REAL_C(0.0));libero_vec::fill_serial(ext,b*6,REAL_C(0.0));libero_vec::fill_serial(intr,b*6,REAL_C(0.0));if(diagnostics)libero_vec::fill_serial(decoded,cc*6,REAL_C(0.0));}
 if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}int n=nc[w],e=ne[w];if(n<0||n>cc||e<0||e>ec){status[w]=2;return;}if(neq[w]!=0){status[w]=6;return;}
 if(!validated){
 if(!post_finite(qv,v)||!post_finite(qa,v)||!post_finite(cd,6*v)||!post_finite(cdd,6*v)||!post_finite(ci,10*b)||!post_finite(cv,6*b)||!post_finite(xf,6*b)||!post_finite(xi,3*b)||!post_finite(com,3*b)||!post_finite(ef,e)||!post_finite(sp,3*ns)||!post_finite(sm,9*ns)){status[w]=4;return;}
 for(int j=0;j<n;j++)if(adr[j]>=0){if(dim[j]<1||dim[j]>6||adr[j]>e-dim[j]||g1[j]<0||g1[j]>=ng||g2[j]<0||g2[j]>=ng){status[w]=3;return;}if(!post_finite(frame+9*j,9)||!post_finite(pos+3*j,3)){status[w]=4;return;}}
 }
 for(int k=0;k<3;k++)acc[3+k]=-grav[k];
 for(int j=1;j<b;j++){bool nz=false;for(int k=0;k<6;k++)nz|=xf[6*j+k]!=0;if(nz)wrench(ext+6*j,xf+6*j,xf+6*j+3,com+3*root[j],xi+3*j,1);}
 for(int j=0;j<n;j++)if(adr[j]>=0){real_t d[6]={0,0,0,0,0,0};post_decode(ef+adr[j],dim[j],d);real_t fm[9],point[3];libero_vec::copy_serial(frame+9*j,fm,9);libero_vec::copy_serial(pos+3*j,point,3);if(diagnostics)libero_vec::copy_serial(d,decoded+6*j,6);real_t f[3],t[3];for(int k=0;k<3;k++){f[k]=fm[k]*d[0]+fm[3+k]*d[1]+fm[6+k]*d[2];t[k]=fm[k]*d[3]+fm[3+k]*d[4]+fm[6+k]*d[5];}for(int side=0;side<2;side++){int body=gb[side?g2[j]:g1[j]];if(body>0)wrench(ext+6*body,f,t,com+3*root[body],point,side?1:-1);}}
 if(cache_ok){
 // Cached velocity-only RNE; shared acc first holds the qacc-only increment.
 libero_vec::fill_serial(acc,6*b,REAL_C(0.0));
 bias_acc+=w*6*b;bias_force+=w*6*b;
 for(int j=1;j<b;j++)for(int k=0;k<6;k++)acc[6*j+k]=acc[6*parent[j]+k]+dof(cd,qa,da[j],dn[j],k);
 for(int j=1;j<b;j++){real_t f[6];inert(ci+10*j,acc+6*j,f);for(int k=0;k<6;k++)intr[6*j+k]=(bias_force[k*b+j]+f[k])-ext[6*j+k];}
 for(int k=0;k<6*b;k++)acc[k]+=bias_acc[(k%6)*b+k/6];
 }else{
 for(int j=1;j<b;j++){real_t* a=acc+6*j;const real_t* c=cv+6*j;for(int k=0;k<6;k++)a[k]=acc[6*parent[j]+k]+dof(cdd,qv,da[j],dn[j],k);for(int k=0;k<6;k++)a[k]+=dof(cd,qa,da[j],dn[j],k);real_t f[6],r[6],x[6];inert(ci+10*j,a,f);inert(ci+10*j,c,r);
 x[0]=-c[2]*r[1]+c[1]*r[2];x[1]=c[2]*r[0]-c[0]*r[2];x[2]=-c[1]*r[0]+c[0]*r[1];x[3]=-c[2]*r[4]+c[1]*r[5];x[4]=c[2]*r[3]-c[0]*r[5];x[5]=-c[1]*r[3]+c[0]*r[4];x[0]+=-c[5]*r[4]+c[4]*r[5];x[1]+=c[5]*r[3]-c[3]*r[5];x[2]+=-c[4]*r[3]+c[3]*r[4];for(int k=0;k<6;k++)intr[6*j+k]=(f[k]+x[k])-ext[6*j+k];}
 }
 for(int j=b-1;j>0;j--)for(int k=0;k<6;k++)intr[6*parent[j]+k]+=intr[6*j+k];
 if(!validated&&(!post_finite(acc,6*b)||!post_finite(ext,6*b)||!post_finite(intr,6*b))){status[w]=5;libero_vec::fill_serial(acc,6*b,REAL_C(0.0));libero_vec::fill_serial(ext,6*b,REAL_C(0.0));libero_vec::fill_serial(intr,6*b,REAL_C(0.0));}
}
// Register reduction followed by one shared result per warp. Full block call.
__device__ __forceinline__ int post_block_min(int value,int* scratch){
 const int r=libero_warp::min(value);
 if(libero_warp::lane()==0)scratch[libero_warp::warp()]=r;
 __syncthreads();
 int out=libero_warp::no_index;
 if(libero_warp::warp()==0)out=libero_warp::min(libero_warp::lane()<(int(blockDim.x)>>5)?scratch[libero_warp::lane()]:libero_warp::no_index);
 if(threadIdx.x==0)scratch[0]=out;
 __syncthreads();
 const int result=scratch[0];
 // No warp may overwrite scratch for the next collective before all read it.
 __syncthreads();
 return result;
}
__device__ __forceinline__ bool post_bad_strided(const real_t* values,int length){
 return libero_vec::any_nonfinite_block_lane(values,length);
}
// Validated independent bodies. Shared acc/ext/intr are [component][body].
// FP64 lanes span paired 32-bit banks; no float-style +1 padding assumption.
// Direct-child sums keep descending original body IDs.
__device__ inline void rne_body_ordered(int w,const real_t* qv,const real_t* qa,const real_t* ci,const real_t* cv,
 const real_t* cd,const real_t* cdd,const real_t* xi,const real_t* com,const real_t* xf,const real_t* ef,
 const int* nc,const int* dim,const int* adr,const int* g1,const int* g2,const real_t* frame,const real_t* pos,
 const int* parent,const int* da,const int* dn,const int* root,const int* gb,const real_t* grav,int cc,
 real_t* acc,real_t* ext,real_t* intr,real_t* decoded,int b,int v,bool diagnostics,bool qacc_world_stride,
 const real_t* bias_acc,const real_t* bias_force,bool cache_ok,int* body_index,int* body_offsets,int* body_counts,
 const int* level_offsets,const int* level_bodies,int nlevels,const int* children_offsets,const int* children_indices,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0){
 qv+=w*v;if(qacc_world_stride)qa+=w*v;ci+=w*b*10;cv+=w*b*6;cd+=w*v*6;cdd+=w*v*6;
 xi+=w*b*3;com+=w*b*3;xf+=w*b*6;dim+=w*cc;adr+=w*cc;g1+=w*cc;g2+=w*cc;
 frame+=w*cc*9;pos+=w*cc*3;decoded+=w*cc*6;
 body_index+=w*2*cc;body_offsets+=w*(b+1);body_counts+=w*b;
 post_contact_index(nc[w],cc,b,adr,g1,g2,gb,body_index,body_offsets,body_counts,
  event_ping?event_ping+w*2*cc:nullptr,event_pong?event_pong+w*2*cc:nullptr,contact_index_mode,radix_min_contacts,
  cached_event_body?cached_event_body+w*2*cc:nullptr,cached_contact_meta?cached_contact_meta+2*w:nullptr,topology_cache);
 // Bounded contact tile replaces repeated per-incidence wrench decoding.
 // Every body keeps its stable CSR cursor; tile progression cannot reorder sides.
 __shared__ real_t contact_tile[9*32];
 // Supported body levels fit one block in the qualified path. For larger body
 // sets the body owner loops below use the existing per-body path instead.
 if(b-1<=int(blockDim.x)){
  const int body=1+int(threadIdx.x);
  real_t local[6]={0,0,0,0,0,0};
  int cursor=body<b?body_offsets[body]:0;
  if(body<b){bool nz=false;for(int k=0;k<6;k++)nz|=xf[6*body+k]!=0;
   if(nz)wrench(local,xf+6*body,xf+6*body+3,com+3*root[body],xi+3*body,1);}
  for(int base=0;base<nc[w];base+=32){
   const int end=min(base+32,nc[w]);
   for(int j=base+int(threadIdx.x);j<end;j+=blockDim.x)if(adr[j]>=0){
    const int slot=j-base;real_t d[6]={0,0,0,0,0,0};
    post_decode(ef+adr[j],dim[j],d);real_t fm[9],point[3];libero_vec::copy_serial(frame+9*j,fm,9);libero_vec::copy_serial(pos+3*j,point,3);
    if(diagnostics)libero_vec::copy_serial(d,decoded+6*j,6);
    for(int k=0;k<3;k++){
     contact_tile[k*32+slot]=fm[k]*d[0]+fm[3+k]*d[1]+fm[6+k]*d[2];
     contact_tile[(3+k)*32+slot]=fm[k]*d[3]+fm[3+k]*d[4]+fm[6+k]*d[5];
     contact_tile[(6+k)*32+slot]=point[k];
    }
   }
   __syncthreads();
   if(body<b)while(cursor<body_offsets[body+1]){
    const int event=body_index[cursor],j=event>>1,side=event&1;
    if(j>=end)break;
    const int slot=j-base;real_t f[3],t[3],point[3];
    for(int k=0;k<3;k++){f[k]=contact_tile[k*32+slot];t[k]=contact_tile[(3+k)*32+slot];point[k]=contact_tile[(6+k)*32+slot];}
    wrench(local,f,t,com+3*root[body],point,side?1:-1);++cursor;
   }
   __syncthreads();
  }
  if(body<b)for(int k=0;k<6;k++)ext[k*b+body]=local[k];
 }else{
  // General body-count fallback keeps the existing stable incidence algorithm.
  if(diagnostics)for(int j=threadIdx.x;j<nc[w];j+=blockDim.x)if(adr[j]>=0)
   libero_vec::copy_serial(ef+adr[j],decoded+6*j,dim[j]);
  for(int body=1+threadIdx.x;body<b;body+=blockDim.x){
   real_t local[6]={0,0,0,0,0,0};bool nz=false;for(int k=0;k<6;k++)nz|=xf[6*body+k]!=0;
   if(nz)wrench(local,xf+6*body,xf+6*body+3,com+3*root[body],xi+3*body,1);
   for(int i=body_offsets[body];i<body_offsets[body+1];i++){
    const int event=body_index[i],j=event>>1,side=event&1;
    real_t d[6]={0,0,0,0,0,0};post_decode(ef+adr[j],dim[j],d);real_t fm[9],point[3];libero_vec::copy_serial(frame+9*j,fm,9);libero_vec::copy_serial(pos+3*j,point,3);real_t f[3],t[3];
    for(int k=0;k<3;k++){f[k]=fm[k]*d[0]+fm[3+k]*d[1]+fm[6+k]*d[2];t[k]=fm[k]*d[3]+fm[3+k]*d[4]+fm[6+k]*d[5];}
    wrench(local,f,t,com+3*root[body],point,side?1:-1);
   }
   for(int k=0;k<6;k++)ext[k*b+body]=local[k];
  }
 }
 if(threadIdx.x==0&&!cache_ok)for(int k=0;k<3;k++)acc[(3+k)*b]=-grav[k];
 __syncthreads();
 for(int level=0;level<nlevels;level++){
  for(int i=level_offsets[level]+threadIdx.x;i<level_offsets[level+1];i+=blockDim.x){
   const int j=level_bodies[i];
   for(int k=0;k<6;k++)acc[k*b+j]=acc[k*b+parent[j]]+dof(cache_ok?cd:cdd,cache_ok?qa:qv,da[j],dn[j],k);
   if(!cache_ok)for(int k=0;k<6;k++)acc[k*b+j]+=dof(cd,qa,da[j],dn[j],k);
  }
  __syncthreads();
 }
 if(cache_ok){bias_acc+=w*6*b;bias_force+=w*6*b;}
 for(int j=1+threadIdx.x;j<b;j+=blockDim.x){
  real_t f[6],a_body[6];for(int k=0;k<6;k++)a_body[k]=acc[k*b+j];inert(ci+10*j,a_body,f);
  if(cache_ok){for(int k=0;k<6;k++)intr[k*b+j]=(bias_force[k*b+j]+f[k])-ext[k*b+j];}
  else {
   const real_t* c=cv+6*j;real_t r[6],x[6];inert(ci+10*j,c,r);
   x[0]=-c[2]*r[1]+c[1]*r[2];x[1]=c[2]*r[0]-c[0]*r[2];x[2]=-c[1]*r[0]+c[0]*r[1];x[3]=-c[2]*r[4]+c[1]*r[5];x[4]=c[2]*r[3]-c[0]*r[5];x[5]=-c[1]*r[3]+c[0]*r[4];x[0]+=-c[5]*r[4]+c[4]*r[5];x[1]+=c[5]*r[3]-c[3]*r[5];x[2]+=-c[4]*r[3]+c[3]*r[4];
   for(int k=0;k<6;k++)intr[k*b+j]=(f[k]+x[k])-ext[k*b+j];
  }
 }
 // All body owners must finish inert(acc) before lanes add cached acceleration.
 __syncthreads();
 if(cache_ok)for(int k=threadIdx.x;k<6*b;k+=blockDim.x)acc[k]+=bias_acc[k];
 __syncthreads();
 for(int level=nlevels-1;level>=0;level--){
  for(int i=level_offsets[level]+threadIdx.x;i<level_offsets[level+1];i+=blockDim.x){
   const int j=level_bodies[i];
   for(int c=children_offsets[j];c<children_offsets[j+1];c++){
    const int child=children_indices[c];
    for(int k=0;k<6;k++)intr[k*b+j]+=intr[k*b+child];
   }
  }
  __syncthreads();
 }
 if(threadIdx.x==0)for(int c=children_offsets[0];c<children_offsets[1];c++){
  const int child=children_indices[c];for(int k=0;k<6;k++)intr[k*b]+=intr[k*b+child];
 }
 __syncthreads();
}

__device__ inline void rne_shared(int w,const int* active,const real_t* qv,const real_t* qa,const real_t* ci,const real_t* cv,const real_t* cd,const real_t* cdd,const real_t* xi,const real_t* com,const real_t* sp,const real_t* sm,const real_t* xf,const real_t* ef,const int* nc,const int* ne,const int* neq,const int* dim,const int* adr,const int* g1,const int* g2,const real_t* frame,const real_t* pos,const int* parent,const int* da,const int* dn,const int* root,const int* gb,const real_t* grav,int cc,int ec,real_t* acc,real_t* ext,real_t* intr,real_t* decoded,int* status,int b,int v,int ns,int ng,bool local_outputs,bool diagnostics,bool qacc_world_stride,const real_t* bias_acc,const real_t* bias_force,bool cache_ok,int* body_index=nullptr,int* body_offsets=nullptr,int* body_counts=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int nlevels=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0){

 // Collective entry: a whole block owns one world. All full warps participate.
 __shared__ int post_vote[32];
 if(threadIdx.x==0)status[w]=0;
 real_t* wa=acc+(local_outputs?0:w*b*6);
 real_t* we=ext+(local_outputs?0:w*b*6);
 real_t* wi=intr+(local_outputs?0:w*b*6);
 libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));
 if(diagnostics)libero_vec::fill_block(decoded+w*cc*6,6*cc,REAL_C(0.0));
 __syncthreads();
 if(active[w]==0)return;
 if(active[w]!=1){if(threadIdx.x==0)status[w]=1;return;}
 const int n=nc[w],e=ne[w];
 if(n<0||n>cc||e<0||e>ec){if(threadIdx.x==0)status[w]=2;return;}
 if(neq[w]!=0){if(threadIdx.x==0)status[w]=6;return;}
 bool bad=false;
 bad|=post_bad_strided(qv+w*v,v);
 bad|=post_bad_strided(qa+(qacc_world_stride?w*v:0),v);
 bad|=post_bad_strided(cd+w*v*6,6*v);
 bad|=post_bad_strided(cdd+w*v*6,6*v);
 bad|=post_bad_strided(ci+w*b*10,10*b);
 bad|=post_bad_strided(cv+w*b*6,6*b);
 bad|=post_bad_strided(xf+w*b*6,6*b);
 bad|=post_bad_strided(xi+w*b*3,3*b);
 bad|=post_bad_strided(com+w*b*3,3*b);
 bad|=post_bad_strided(ef+w*ec,e);
 bad|=post_bad_strided(sp+w*ns*3,3*ns);
 bad|=post_bad_strided(sm+w*ns*9,9*ns);
 const bool bad_input=post_block_min(bad?0:libero_warp::no_index,post_vote)==0;
 if(bad_input){if(threadIdx.x==0)status[w]=4;return;}
 int first=libero_warp::no_index;
 for(int j=threadIdx.x;j<n;j+=blockDim.x){
  const int z=w*cc+j;
  if(adr[z]<0)continue;
  // Schema wins over finite failure within a contact; earlier contacts win.
  const bool invalid=dim[z]<1||dim[z]>6||adr[z]>e-dim[z]||g1[z]<0||g1[z]>=ng||g2[z]<0||g2[z]>=ng;
  bool numeric=false;
  if(!invalid)numeric=!post_finite(frame+9*z,9)||!post_finite(pos+3*z,3);
  if(invalid||numeric){const int key=2*j+(invalid?0:1);first=first<key?first:key;}
 }
 first=post_block_min(first,post_vote);
 if(first!=libero_warp::no_index){if(threadIdx.x==0)status[w]=(first&1)?4:3;return;}
 // Parallel path is enabled only by proven cold topology and sized CSR owners.
 if(body_index&&body_offsets&&body_counts&&level_offsets&&level_bodies&&children_offsets&&children_indices){
  rne_body_ordered(w,qv,qa,ci,cv,cd,cdd,xi,com,xf,ef+w*ec,nc,dim,adr,g1,g2,frame,pos,parent,da,dn,root,gb,grav,cc,wa,we,wi,decoded,b,v,diagnostics,qacc_world_stride,bias_acc,bias_force,cache_ok,body_index,body_offsets,body_counts,level_offsets,level_bodies,nlevels,children_offsets,children_indices,event_ping,event_pong,contact_index_mode,radix_min_contacts,cached_event_body,cached_contact_meta,topology_cache);
 }else if(threadIdx.x==0)rne_shared_scalar(w,active,qv,qa,ci,cv,cd,cdd,xi,com,sp,sm,xf,ef,nc,ne,neq,dim,adr,g1,g2,frame,pos,parent,da,dn,root,gb,grav,cc,ec,acc,ext,intr,decoded,status,b,v,ns,ng,local_outputs,diagnostics,qacc_world_stride,bias_acc,bias_force,cache_ok,true);
 __syncthreads();
 bad=false;
 bad=post_bad_strided(wa,6*b)|post_bad_strided(we,6*b)|post_bad_strided(wi,6*b);
 if(post_block_min(bad?0:libero_warp::no_index,post_vote)==0){
  if(threadIdx.x==0)status[w]=5;
  libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));
 }
 __syncthreads();
}
__device__ inline void sensor_shared(int w,const real_t* fr,const real_t* com,const real_t* pos,const real_t* mat,const int* active,const int* type,const int* site,const int* adr,const int* sb,const int* br,real_t* result,int* status,int b,int ns,bool local_force,bool prevalidated,bool local_result,bool force_planes=false){
 status[w]=0;if(!local_result)result+=6*w;libero_vec::fill_serial(result,6,REAL_C(0.0));if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}if(!local_force)fr+=w*b*6;com+=w*b*3;pos+=w*ns*3;mat+=w*ns*9;
 if(!prevalidated&&(!post_finite(fr,6*b)||!post_finite(com,3*b)||!post_finite(pos,3*ns)||!post_finite(mat,9*ns))){status[w]=2;return;}
 for(int s=0;s<2;s++){int st=site[s],body=sb[st],root=br[body];real_t d[3],c[3],t[3];real_t body_force[6];for(int k=0;k<6;k++)body_force[k]=force_planes?fr[k*b+body]:fr[6*body+k];const real_t* f=body_force+3;for(int k=0;k<3;k++)d[k]=pos[3*st+k]-com[3*root+k];libero_cuda::cross3(d,f,c);for(int k=0;k<3;k++)t[k]=body_force[k]-c[k];const real_t* v=type[s]==4?f:t;for(int k=0;k<3;k++)result[adr[s]+k]=mat[9*st+k]*v[0]+mat[9*st+3+k]*v[1]+mat[9*st+6+k]*v[2];}
}

// Collective block-owned post pipeline. Each qualified spatial field is [6][NB];
// scalar fallback remains [NB][6], and the public exports remain [NB][6].
// qacc remains explicitly supplied; a joint N/post entry may bind its private output.
__device__ __forceinline__ void run_workspace(int w,const int* g_active,const int* g_first,const int* g_dv,const int* g_nv,const int* g_ne,const int64_t* g_nw,const int64_t* g_na,const int64_t* g_expected,const int64_t* g_state,const int64_t* g_consumed,const int64_t* g_forward,const int64_t* g_cf,int* g_g,const int* r_active,const real_t* r_qv,const real_t* r_qa,const real_t* r_ci,const real_t* r_cv,const real_t* r_cd,const real_t* r_cdd,const real_t* r_xi,const real_t* r_com,const real_t* r_sp,const real_t* r_sm,const real_t* r_xf,const real_t* r_ef,const int* r_nc,const int* r_ne,const int* r_neq,const int* r_dim,const int* r_adr,const int* r_g1,const int* r_g2,const real_t* r_frame,const real_t* r_pos,const int* r_parent,const int* r_da,const int* r_dn,const int* r_root,const int* r_gb,const real_t* r_grav,int r_cc,int r_ec,real_t* r_acc,real_t* r_ext,real_t* r_intr,real_t* r_decoded,int* r_status,int r_b,int r_v,int r_ns,int r_ng,const real_t* s_fr,const real_t* s_com,const real_t* s_pos,const real_t* s_mat,const int* s_active,const int* s_type,const int* s_site,const int* s_adr,const int* s_sb,const int* s_br,real_t* s_result,int* s_status,int s_b,int s_ns,const int* c_gate,const int* c_status,const int* c_ss,int* c_first,const int64_t* c_epoch,const real_t* c_ext,const real_t* c_intr,const real_t* c_sensor,real_t* c_pe,real_t* c_pi,real_t* c_ps,int* c_valid,int64_t* c_written,int c_bw,int c_sw,int diagnostics,real_t* workspace,bool qacc_world_stride=true,const real_t* bias_acc=nullptr,const real_t* bias_force=nullptr,bool cache_ok=false,int* body_index=nullptr,int* body_offsets=nullptr,int* body_counts=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int nlevels=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0){
 const bool work_planes=body_index&&body_offsets&&body_counts&&level_offsets&&level_bodies&&children_offsets&&children_indices;
 real_t* work_acc=workspace;real_t* work_ext=work_acc+6*r_b;
 real_t* work_int=work_ext+6*r_b;real_t* work_sensor=work_int+6*r_b;
 if(threadIdx.x==0)gate(w,g_active,g_first,g_dv,g_nv,g_ne,g_nw,g_na,g_expected,g_state,g_consumed,g_forward,g_cf,g_g);
 rne_shared(w,r_active,r_qv,r_qa,r_ci,r_cv,r_cd,r_cdd,r_xi,r_com,r_sp,r_sm,r_xf,r_ef,r_nc,r_ne,r_neq,r_dim,r_adr,r_g1,r_g2,r_frame,r_pos,r_parent,r_da,r_dn,r_root,r_gb,r_grav,r_cc,r_ec,work_acc,work_ext,work_int,r_decoded,r_status,r_b,r_v,r_ns,r_ng,true,diagnostics!=0,qacc_world_stride,bias_acc,bias_force,cache_ok,body_index,body_offsets,body_counts,level_offsets,level_bodies,nlevels,children_offsets,children_indices,event_ping,event_pong,contact_index_mode,radix_min_contacts,cached_event_body,cached_contact_meta,topology_cache);
 // All lanes remain live through publication. Only sensor math and state decisions
 // are leader-owned; their result is visible before the collective copies.
 if(threadIdx.x==0){
  sensor_shared(w,work_int,s_com,s_pos,s_mat,s_active,s_type,s_site,s_adr,s_sb,s_br,work_sensor,s_status,s_b,s_ns,true,r_status[w]==0,true,work_planes);
  if(r_active[w]==1&&r_status[w]==0){if(s_status[w]!=0)r_status[w]=4;else if(!post_finite(work_sensor,6)){r_status[w]=5;libero_vec::fill_serial(work_sensor,6,REAL_C(0.0));}}
 }
 __syncthreads();
 if(diagnostics){
  post_export(work_acc,r_acc+w*6*r_b,6*r_b,r_b,work_planes);
  post_export(work_ext,r_ext+w*6*r_b,6*r_b,r_b,work_planes);
  post_export(work_int,r_intr+w*6*r_b,6*r_b,r_b,work_planes);
  libero_vec::copy_block(work_sensor,s_result+6*w,6);
 }
 __syncthreads();
 if(threadIdx.x==0){
  c_valid[w]=0;
  if(c_gate[w]==1&&c_first[w]==0&&(r_status[w]!=0||s_status[w]!=0))c_first[w]=215;
 }
 __syncthreads();
 // These owner/world predicates are uniform. Blocked worlds retain all public
 // payload and written epochs; diagnostics retain their original private writes.
 if(c_gate[w]!=1||c_first[w]!=0)return;
 post_export(work_ext,c_pe+w*c_bw,c_bw,r_b,work_planes);
 post_export(work_int,c_pi+w*c_bw,c_bw,r_b,work_planes);
 libero_vec::copy_block(work_sensor,c_ps+w*c_sw,c_sw);
 __syncthreads();
 if(threadIdx.x==0){c_written[w]=c_epoch[w];c_valid[w]=1;}

}

__device__ __forceinline__ void run_original(int w,const int* g_active,const int* g_first,const int* g_dv,const int* g_nv,const int* g_ne,const int64_t* g_nw,const int64_t* g_na,const int64_t* g_expected,const int64_t* g_state,const int64_t* g_consumed,const int64_t* g_forward,const int64_t* g_cf,int* g_g,const int* r_active,const real_t* r_qv,const real_t* r_qa,const real_t* r_ci,const real_t* r_cv,const real_t* r_cd,const real_t* r_cdd,const real_t* r_xi,const real_t* r_com,const real_t* r_sp,const real_t* r_sm,const real_t* r_xf,const real_t* r_ef,const int* r_nc,const int* r_ne,const int* r_neq,const int* r_dim,const int* r_adr,const int* r_g1,const int* r_g2,const real_t* r_frame,const real_t* r_pos,const int* r_parent,const int* r_da,const int* r_dn,const int* r_root,const int* r_gb,const real_t* r_grav,int r_cc,int r_ec,real_t* r_acc,real_t* r_ext,real_t* r_intr,real_t* r_decoded,int* r_status,int r_b,int r_v,int r_ns,int r_ng,const real_t* s_fr,const real_t* s_com,const real_t* s_pos,const real_t* s_mat,const int* s_active,const int* s_type,const int* s_site,const int* s_adr,const int* s_sb,const int* s_br,real_t* s_result,int* s_status,int s_b,int s_ns,const int* c_gate,const int* c_status,const int* c_ss,int* c_first,const int64_t* c_epoch,const real_t* c_ext,const real_t* c_intr,const real_t* c_sensor,real_t* c_pe,real_t* c_pi,real_t* c_ps,int* c_valid,int64_t* c_written,int c_bw,int c_sw,int diagnostics,const real_t* cache_acc=nullptr,const real_t* cache_force=nullptr,const int* cache_status=nullptr,const int* cache_valid=nullptr,const int64_t* cache_state=nullptr,const int64_t* cache_forward=nullptr,int cache_gravity_disabled=0,int* body_index=nullptr,int* body_offsets=nullptr,int* body_counts=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int nlevels=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0){
 extern __shared__ real_t post_work[];
 run_workspace(w,g_active,g_first,g_dv,g_nv,g_ne,g_nw,g_na,g_expected,g_state,g_consumed,g_forward,g_cf,g_g,r_active,r_qv,r_qa,r_ci,r_cv,r_cd,r_cdd,r_xi,r_com,r_sp,r_sm,r_xf,r_ef,r_nc,r_ne,r_neq,r_dim,r_adr,r_g1,r_g2,r_frame,r_pos,r_parent,r_da,r_dn,r_root,r_gb,r_grav,r_cc,r_ec,r_acc,r_ext,r_intr,r_decoded,r_status,r_b,r_v,r_ns,r_ng,s_fr,s_com,s_pos,s_mat,s_active,s_type,s_site,s_adr,s_sb,s_br,s_result,s_status,s_b,s_ns,c_gate,c_status,c_ss,c_first,c_epoch,c_ext,c_intr,c_sensor,c_pe,c_pi,c_ps,c_valid,c_written,c_bw,c_sw,diagnostics,post_work,true,cache_acc,cache_force,cache_valid&&cache_status&&cache_state&&cache_forward&&cache_acc&&cache_force&&cache_gravity_disabled==0&&cache_valid[w]==1&&cache_status[w]==0&&cache_state[w]==g_state[w]&&cache_forward[w]==g_forward[w],body_index,body_offsets,body_counts,level_offsets,level_bodies,nlevels,children_offsets,children_indices,event_ping,event_pong,contact_index_mode,radix_min_contacts,cached_event_body,cached_contact_meta,topology_cache);
}
}
namespace post_root38_fast { using namespace libero_post;
// Each source view is guarded independently; workspace starts require only
// double alignment. Plane-to-AoS pairs intentionally use separate scalar reads.
__device__ __forceinline__ bool post_finite(const real_t* p,int n){
 return !libero_vec::any_nonfinite_lane(p,n,0,1);
}
__device__ __forceinline__ void post_export(const real_t* src,real_t* dst,int n,int b,bool planes){
 if(!planes){libero_vec::copy_block(src,dst,n);return;}
 for(int pair=threadIdx.x;pair<n/2;pair+=blockDim.x){
  const int k=2*pair;
  const real_t x=src[(k%6)*b+k/6],y=src[((k+1)%6)*b+(k+1)/6];
  libero_vec::store_pair(dst+k,x,y);
 }
 if((n&1)&&threadIdx.x==0){const int k=n-1;dst[k]=src[(k%6)*b+k/6];}
}
// Exactly count valid values of an enabled, schema-validated contact only.
__device__ __forceinline__ void post_decode(const real_t* ef,int count,real_t* d){
 int k=0;
 for(;k+1<count;k+=2){const real2_t p=libero_vec::load_pair(ef+k);d[k]=p.x;d[k+1]=p.y;}
 if(k<count)d[k]=ef[k];
}
__device__ inline void rne_shared_scalar(int w,const int* active,const real_t* qv,const real_t* qa,const real_t* ci,const real_t* cv,const real_t* cd,const real_t* cdd,const real_t* xi,const real_t* com,const real_t* sp,const real_t* sm,const real_t* xf,const real_t* ef,const int* nc,const int* ne,const int* neq,const int* dim,const int* adr,const int* g1,const int* g2,const real_t* frame,const real_t* pos,const int* parent,const int* da,const int* dn,const int* root,const int* gb,const real_t* grav,int cc,int ec,real_t* acc,real_t* ext,real_t* intr,real_t* decoded,int* status,int b,int v,int ns,int ng,bool local_outputs,bool diagnostics,bool qacc_world_stride,const real_t* bias_acc,const real_t* bias_force,bool cache_ok,bool validated=false){
 status[w]=0;qv+=w*v;if(qacc_world_stride)qa+=w*v;ci+=w*b*10;cv+=w*b*6;cd+=w*v*6;cdd+=w*v*6;xi+=w*b*3;com+=w*b*3;sp+=w*ns*3;sm+=w*ns*9;xf+=w*b*6;ef+=w*ec;dim+=w*cc;adr+=w*cc;g1+=w*cc;g2+=w*cc;frame+=w*cc*9;pos+=w*cc*3;if(!local_outputs){acc+=w*b*6;ext+=w*b*6;intr+=w*b*6;}decoded+=w*cc*6;
 if(!validated){libero_vec::fill_serial(acc,b*6,REAL_C(0.0));libero_vec::fill_serial(ext,b*6,REAL_C(0.0));libero_vec::fill_serial(intr,b*6,REAL_C(0.0));if(diagnostics)libero_vec::fill_serial(decoded,cc*6,REAL_C(0.0));}
 if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}int n=nc[w],e=ne[w];if(n<0||n>cc||e<0||e>ec){status[w]=2;return;}if(neq[w]!=0){status[w]=6;return;}
 if(!validated){
 if(!post_finite(qv,v)||!post_finite(qa,v)||!post_finite(cd,6*v)||!post_finite(cdd,6*v)||!post_finite(ci,10*b)||!post_finite(cv,6*b)||!post_finite(xf,6*b)||!post_finite(xi,3*b)||!post_finite(com,3*b)||!post_finite(ef,e)||!post_finite(sp,3*ns)||!post_finite(sm,9*ns)){status[w]=4;return;}
 for(int j=0;j<n;j++)if(adr[j]>=0){if(dim[j]<1||dim[j]>6||adr[j]>e-dim[j]||g1[j]<0||g1[j]>=ng||g2[j]<0||g2[j]>=ng){status[w]=3;return;}if(!post_finite(frame+9*j,9)||!post_finite(pos+3*j,3)){status[w]=4;return;}}
 }
 for(int k=0;k<3;k++)acc[3+k]=-grav[k];
 for(int j=1;j<b;j++){bool nz=false;for(int k=0;k<6;k++)nz|=xf[6*j+k]!=0;if(nz)wrench(ext+6*j,xf+6*j,xf+6*j+3,com+3*root[j],xi+3*j,1);}
 for(int j=0;j<n;j++)if(adr[j]>=0){real_t d[6]={0,0,0,0,0,0};post_decode(ef+adr[j],dim[j],d);real_t fm[9],point[3];libero_vec::copy_serial(frame+9*j,fm,9);libero_vec::copy_serial(pos+3*j,point,3);if(diagnostics)libero_vec::copy_serial(d,decoded+6*j,6);real_t f[3],t[3];for(int k=0;k<3;k++){f[k]=fm[k]*d[0]+fm[3+k]*d[1]+fm[6+k]*d[2];t[k]=fm[k]*d[3]+fm[3+k]*d[4]+fm[6+k]*d[5];}for(int side=0;side<2;side++){int body=gb[side?g2[j]:g1[j]];if(body>0)wrench(ext+6*body,f,t,com+3*root[body],point,side?1:-1);}}
 if(cache_ok){
 // Cached velocity-only RNE; shared acc first holds the qacc-only increment.
 libero_vec::fill_serial(acc,6*b,REAL_C(0.0));
 bias_acc+=w*6*b;bias_force+=w*6*b;
 for(int j=1;j<b;j++)for(int k=0;k<6;k++)acc[6*j+k]=acc[6*parent[j]+k]+dof(cd,qa,da[j],dn[j],k);
 for(int j=1;j<b;j++){real_t f[6];inert(ci+10*j,acc+6*j,f);for(int k=0;k<6;k++)intr[6*j+k]=(bias_force[k*b+j]+f[k])-ext[6*j+k];}
 for(int k=0;k<6*b;k++)acc[k]+=bias_acc[(k%6)*b+k/6];
 }else{
 for(int j=1;j<b;j++){real_t* a=acc+6*j;const real_t* c=cv+6*j;for(int k=0;k<6;k++)a[k]=acc[6*parent[j]+k]+dof(cdd,qv,da[j],dn[j],k);for(int k=0;k<6;k++)a[k]+=dof(cd,qa,da[j],dn[j],k);real_t f[6],r[6],x[6];inert(ci+10*j,a,f);inert(ci+10*j,c,r);
 x[0]=-c[2]*r[1]+c[1]*r[2];x[1]=c[2]*r[0]-c[0]*r[2];x[2]=-c[1]*r[0]+c[0]*r[1];x[3]=-c[2]*r[4]+c[1]*r[5];x[4]=c[2]*r[3]-c[0]*r[5];x[5]=-c[1]*r[3]+c[0]*r[4];x[0]+=-c[5]*r[4]+c[4]*r[5];x[1]+=c[5]*r[3]-c[3]*r[5];x[2]+=-c[4]*r[3]+c[3]*r[4];for(int k=0;k<6;k++)intr[6*j+k]=(f[k]+x[k])-ext[6*j+k];}
 }
 for(int j=b-1;j>0;j--)for(int k=0;k<6;k++)intr[6*parent[j]+k]+=intr[6*j+k];
 if(!validated&&(!post_finite(acc,6*b)||!post_finite(ext,6*b)||!post_finite(intr,6*b))){status[w]=5;libero_vec::fill_serial(acc,6*b,REAL_C(0.0));libero_vec::fill_serial(ext,6*b,REAL_C(0.0));libero_vec::fill_serial(intr,6*b,REAL_C(0.0));}
}
// Register reduction followed by one shared result per warp. Full block call.
__device__ __forceinline__ int post_block_min(int value,int* scratch){
 const int r=libero_warp::min(value);
 if(libero_warp::lane()==0)scratch[libero_warp::warp()]=r;
 __syncthreads();
 int out=libero_warp::no_index;
 if(libero_warp::warp()==0)out=libero_warp::min(libero_warp::lane()<(int(blockDim.x)>>5)?scratch[libero_warp::lane()]:libero_warp::no_index);
 if(threadIdx.x==0)scratch[0]=out;
 __syncthreads();
 const int result=scratch[0];
 // No warp may overwrite scratch for the next collective before all read it.
 __syncthreads();
 return result;
}
__device__ __forceinline__ bool post_bad_strided(const real_t* values,int length){
 return libero_vec::any_nonfinite_block_lane(values,length);
}
// Validated independent bodies. Shared acc/ext/intr are [component][body].
// FP64 lanes span paired 32-bit banks; no float-style +1 padding assumption.
// Direct-child sums keep descending original body IDs.
__device__ inline void rne_body_ordered(int w,const real_t* qv,const real_t* qa,const real_t* ci,const real_t* cv,
 const real_t* cd,const real_t* cdd,const real_t* xi,const real_t* com,const real_t* xf,const real_t* ef,
 const int* nc,const int* dim,const int* adr,const int* g1,const int* g2,const real_t* frame,const real_t* pos,
 const int* parent,const int* da,const int* dn,const int* root,const int* gb,const real_t* grav,int cc,
 real_t* acc,real_t* ext,real_t* intr,real_t* decoded,int b,int v,bool diagnostics,bool qacc_world_stride,
 const real_t* bias_acc,const real_t* bias_force,bool cache_ok,int* body_index,int* body_offsets,int* body_counts,
 const int* level_offsets,const int* level_bodies,int nlevels,const int* children_offsets,const int* children_indices,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0){
 qv+=w*v;if(qacc_world_stride)qa+=w*v;ci+=w*b*10;cv+=w*b*6;cd+=w*v*6;cdd+=w*v*6;
 xi+=w*b*3;com+=w*b*3;xf+=w*b*6;dim+=w*cc;adr+=w*cc;g1+=w*cc;g2+=w*cc;
 frame+=w*cc*9;pos+=w*cc*3;decoded+=w*cc*6;
 body_index+=w*2*cc;body_offsets+=w*(b+1);body_counts+=w*b;
 post_contact_index(nc[w],cc,b,adr,g1,g2,gb,body_index,body_offsets,body_counts,
  event_ping?event_ping+w*2*cc:nullptr,event_pong?event_pong+w*2*cc:nullptr,contact_index_mode,radix_min_contacts,
  cached_event_body?cached_event_body+w*2*cc:nullptr,cached_contact_meta?cached_contact_meta+2*w:nullptr,topology_cache);
 // Bounded contact tile replaces repeated per-incidence wrench decoding.
 // Every body keeps its stable CSR cursor; tile progression cannot reorder sides.
 __shared__ real_t contact_tile[9*32];
 // Supported body levels fit one block in the qualified path. For larger body
 // sets the body owner loops below use the existing per-body path instead.
 if(b-1<=int(blockDim.x)){
  const int body=1+int(threadIdx.x);
  real_t local[6]={0,0,0,0,0,0};
  int cursor=body<b?body_offsets[body]:0;
  if(body<b){bool nz=false;for(int k=0;k<6;k++)nz|=xf[6*body+k]!=0;
   if(nz)wrench(local,xf+6*body,xf+6*body+3,com+3*root[body],xi+3*body,1);}
  for(int base=0;base<nc[w];base+=32){
   const int end=min(base+32,nc[w]);
   for(int j=base+int(threadIdx.x);j<end;j+=blockDim.x)if(adr[j]>=0){
    const int slot=j-base;real_t d[6]={0,0,0,0,0,0};
    post_decode(ef+adr[j],dim[j],d);real_t fm[9],point[3];libero_vec::copy_serial(frame+9*j,fm,9);libero_vec::copy_serial(pos+3*j,point,3);
    if(diagnostics)libero_vec::copy_serial(d,decoded+6*j,6);
    for(int k=0;k<3;k++){
     contact_tile[k*32+slot]=fm[k]*d[0]+fm[3+k]*d[1]+fm[6+k]*d[2];
     contact_tile[(3+k)*32+slot]=fm[k]*d[3]+fm[3+k]*d[4]+fm[6+k]*d[5];
     contact_tile[(6+k)*32+slot]=point[k];
    }
   }
   __syncthreads();
   if(body<b)while(cursor<body_offsets[body+1]){
    const int event=body_index[cursor],j=event>>1,side=event&1;
    if(j>=end)break;
    const int slot=j-base;real_t f[3],t[3],point[3];
    for(int k=0;k<3;k++){f[k]=contact_tile[k*32+slot];t[k]=contact_tile[(3+k)*32+slot];point[k]=contact_tile[(6+k)*32+slot];}
    wrench(local,f,t,com+3*root[body],point,side?1:-1);++cursor;
   }
   __syncthreads();
  }
  if(body<b)for(int k=0;k<6;k++)ext[k*b+body]=local[k];
 }else{
  // General body-count fallback keeps the existing stable incidence algorithm.
  if(diagnostics)for(int j=threadIdx.x;j<nc[w];j+=blockDim.x)if(adr[j]>=0)
   libero_vec::copy_serial(ef+adr[j],decoded+6*j,dim[j]);
  for(int body=1+threadIdx.x;body<b;body+=blockDim.x){
   real_t local[6]={0,0,0,0,0,0};bool nz=false;for(int k=0;k<6;k++)nz|=xf[6*body+k]!=0;
   if(nz)wrench(local,xf+6*body,xf+6*body+3,com+3*root[body],xi+3*body,1);
   for(int i=body_offsets[body];i<body_offsets[body+1];i++){
    const int event=body_index[i],j=event>>1,side=event&1;
    real_t d[6]={0,0,0,0,0,0};post_decode(ef+adr[j],dim[j],d);real_t fm[9],point[3];libero_vec::copy_serial(frame+9*j,fm,9);libero_vec::copy_serial(pos+3*j,point,3);real_t f[3],t[3];
    for(int k=0;k<3;k++){f[k]=fm[k]*d[0]+fm[3+k]*d[1]+fm[6+k]*d[2];t[k]=fm[k]*d[3]+fm[3+k]*d[4]+fm[6+k]*d[5];}
    wrench(local,f,t,com+3*root[body],point,side?1:-1);
   }
   for(int k=0;k<6;k++)ext[k*b+body]=local[k];
  }
 }
 if(threadIdx.x==0&&!cache_ok)for(int k=0;k<3;k++)acc[(3+k)*b]=-grav[k];
 __syncthreads();
 for(int level=0;level<nlevels;level++){
  for(int i=level_offsets[level]+threadIdx.x;i<level_offsets[level+1];i+=blockDim.x){
   const int j=level_bodies[i];
   for(int k=0;k<6;k++)acc[k*b+j]=acc[k*b+parent[j]]+dof(cache_ok?cd:cdd,cache_ok?qa:qv,da[j],dn[j],k);
   if(!cache_ok)for(int k=0;k<6;k++)acc[k*b+j]+=dof(cd,qa,da[j],dn[j],k);
  }
  __syncthreads();
 }
 if(cache_ok){bias_acc+=w*6*b;bias_force+=w*6*b;}
 for(int j=1+threadIdx.x;j<b;j+=blockDim.x){
  real_t f[6],a_body[6];for(int k=0;k<6;k++)a_body[k]=acc[k*b+j];inert(ci+10*j,a_body,f);
  if(cache_ok){for(int k=0;k<6;k++)intr[k*b+j]=(bias_force[k*b+j]+f[k])-ext[k*b+j];}
  else {
   const real_t* c=cv+6*j;real_t r[6],x[6];inert(ci+10*j,c,r);
   x[0]=-c[2]*r[1]+c[1]*r[2];x[1]=c[2]*r[0]-c[0]*r[2];x[2]=-c[1]*r[0]+c[0]*r[1];x[3]=-c[2]*r[4]+c[1]*r[5];x[4]=c[2]*r[3]-c[0]*r[5];x[5]=-c[1]*r[3]+c[0]*r[4];x[0]+=-c[5]*r[4]+c[4]*r[5];x[1]+=c[5]*r[3]-c[3]*r[5];x[2]+=-c[4]*r[3]+c[3]*r[4];
   for(int k=0;k<6;k++)intr[k*b+j]=(f[k]+x[k])-ext[k*b+j];
  }
 }
 // All body owners must finish inert(acc) before lanes add cached acceleration.
 __syncthreads();
 if(cache_ok)for(int k=threadIdx.x;k<6*b;k+=blockDim.x)acc[k]+=bias_acc[k];
 __syncthreads();
 for(int level=nlevels-1;level>=0;level--){
  for(int i=level_offsets[level]+threadIdx.x;i<level_offsets[level+1];i+=blockDim.x){
   const int j=level_bodies[i];
   for(int c=children_offsets[j];c<children_offsets[j+1];c++){
    const int child=children_indices[c];
    for(int k=0;k<6;k++)intr[k*b+j]+=intr[k*b+child];
   }
  }
  __syncthreads();
 }
 if(threadIdx.x==0)for(int c=children_offsets[0];c<children_offsets[1];c++){
  const int child=children_indices[c];for(int k=0;k<6;k++)intr[k*b]+=intr[k*b+child];
 }
 __syncthreads();
}

__device__ inline void rne_shared(int w,const int* active,const real_t* qv,const real_t* qa,const real_t* ci,const real_t* cv,const real_t* cd,const real_t* cdd,const real_t* xi,const real_t* com,const real_t* sp,const real_t* sm,const real_t* xf,const real_t* ef,const int* nc,const int* ne,const int* neq,const int* dim,const int* adr,const int* g1,const int* g2,const real_t* frame,const real_t* pos,const int* parent,const int* da,const int* dn,const int* root,const int* gb,const real_t* grav,int cc,int ec,real_t* acc,real_t* ext,real_t* intr,real_t* decoded,int* status,int b,int v,int ns,int ng,bool local_outputs,bool diagnostics,bool qacc_world_stride,const real_t* bias_acc,const real_t* bias_force,bool cache_ok,int* body_index=nullptr,int* body_offsets=nullptr,int* body_counts=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int nlevels=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0){

 // Collective entry: a whole block owns one world. All full warps participate.
 __shared__ int post_vote[32];
 if(threadIdx.x==0)status[w]=0;
 real_t* wa=acc+(local_outputs?0:w*b*6);
 real_t* we=ext+(local_outputs?0:w*b*6);
 real_t* wi=intr+(local_outputs?0:w*b*6);
 const bool root_lazy=!diagnostics&&body_index&&body_offsets&&body_counts&&level_offsets&&level_bodies&&children_offsets&&children_indices;
 if(!root_lazy){libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));}
 else if(threadIdx.x==0)for(int k=0;k<6;k++){wa[k*b]=REAL_C(0.0);we[k*b]=REAL_C(0.0);wi[k*b]=REAL_C(0.0);}
 if(diagnostics)libero_vec::fill_block(decoded+w*cc*6,6*cc,REAL_C(0.0));
 __syncthreads();
 if(active[w]==0){if(root_lazy){libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));}return;}
 if(active[w]!=1){if(threadIdx.x==0)status[w]=1;if(root_lazy){libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));}return;}
 const int n=nc[w],e=ne[w];
 if(n<0||n>cc||e<0||e>ec){if(threadIdx.x==0)status[w]=2;if(root_lazy){libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));}return;}
 if(neq[w]!=0){if(threadIdx.x==0)status[w]=6;if(root_lazy){libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));}return;}
 bool bad=false;
 bad|=post_bad_strided(qv+w*v,v);
 bad|=post_bad_strided(qa+(qacc_world_stride?w*v:0),v);
 bad|=post_bad_strided(cd+w*v*6,6*v);
 bad|=post_bad_strided(cdd+w*v*6,6*v);
 bad|=post_bad_strided(ci+w*b*10,10*b);
 bad|=post_bad_strided(cv+w*b*6,6*b);
 bad|=post_bad_strided(xf+w*b*6,6*b);
 bad|=post_bad_strided(xi+w*b*3,3*b);
 bad|=post_bad_strided(com+w*b*3,3*b);
 bad|=post_bad_strided(ef+w*ec,e);
 bad|=post_bad_strided(sp+w*ns*3,3*ns);
 bad|=post_bad_strided(sm+w*ns*9,9*ns);
 const bool bad_input=post_block_min(bad?0:libero_warp::no_index,post_vote)==0;
 if(bad_input){if(threadIdx.x==0)status[w]=4;if(root_lazy){libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));}return;}
 int first=libero_warp::no_index;
 for(int j=threadIdx.x;j<n;j+=blockDim.x){
  const int z=w*cc+j;
  if(adr[z]<0)continue;
  // Schema wins over finite failure within a contact; earlier contacts win.
  const bool invalid=dim[z]<1||dim[z]>6||adr[z]>e-dim[z]||g1[z]<0||g1[z]>=ng||g2[z]<0||g2[z]>=ng;
  bool numeric=false;
  if(!invalid)numeric=!post_finite(frame+9*z,9)||!post_finite(pos+3*z,3);
  if(invalid||numeric){const int key=2*j+(invalid?0:1);first=first<key?first:key;}
 }
 first=post_block_min(first,post_vote);
 if(first!=libero_warp::no_index){if(threadIdx.x==0)status[w]=(first&1)?4:3;if(root_lazy){libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));}return;}
 // Parallel path is enabled only by proven cold topology and sized CSR owners.
 if(body_index&&body_offsets&&body_counts&&level_offsets&&level_bodies&&children_offsets&&children_indices){
  rne_body_ordered(w,qv,qa,ci,cv,cd,cdd,xi,com,xf,ef+w*ec,nc,dim,adr,g1,g2,frame,pos,parent,da,dn,root,gb,grav,cc,wa,we,wi,decoded,b,v,diagnostics,qacc_world_stride,bias_acc,bias_force,cache_ok,body_index,body_offsets,body_counts,level_offsets,level_bodies,nlevels,children_offsets,children_indices,event_ping,event_pong,contact_index_mode,radix_min_contacts,cached_event_body,cached_contact_meta,topology_cache);
 }else if(threadIdx.x==0)rne_shared_scalar(w,active,qv,qa,ci,cv,cd,cdd,xi,com,sp,sm,xf,ef,nc,ne,neq,dim,adr,g1,g2,frame,pos,parent,da,dn,root,gb,grav,cc,ec,acc,ext,intr,decoded,status,b,v,ns,ng,local_outputs,diagnostics,qacc_world_stride,bias_acc,bias_force,cache_ok,true);
 __syncthreads();
 bad=false;
 bad=post_bad_strided(wa,6*b)|post_bad_strided(we,6*b)|post_bad_strided(wi,6*b);
 if(post_block_min(bad?0:libero_warp::no_index,post_vote)==0){
  if(threadIdx.x==0)status[w]=5;
  libero_vec::fill_block(wa,6*b,REAL_C(0.0));libero_vec::fill_block(we,6*b,REAL_C(0.0));libero_vec::fill_block(wi,6*b,REAL_C(0.0));
 }
 __syncthreads();
}
__device__ inline void sensor_shared(int w,const real_t* fr,const real_t* com,const real_t* pos,const real_t* mat,const int* active,const int* type,const int* site,const int* adr,const int* sb,const int* br,real_t* result,int* status,int b,int ns,bool local_force,bool prevalidated,bool local_result,bool force_planes=false){
 status[w]=0;if(!local_result)result+=6*w;libero_vec::fill_serial(result,6,REAL_C(0.0));if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}if(!local_force)fr+=w*b*6;com+=w*b*3;pos+=w*ns*3;mat+=w*ns*9;
 if(!prevalidated&&(!post_finite(fr,6*b)||!post_finite(com,3*b)||!post_finite(pos,3*ns)||!post_finite(mat,9*ns))){status[w]=2;return;}
 for(int s=0;s<2;s++){int st=site[s],body=sb[st],root=br[body];real_t d[3],c[3],t[3];real_t body_force[6];for(int k=0;k<6;k++)body_force[k]=force_planes?fr[k*b+body]:fr[6*body+k];const real_t* f=body_force+3;for(int k=0;k<3;k++)d[k]=pos[3*st+k]-com[3*root+k];libero_cuda::cross3(d,f,c);for(int k=0;k<3;k++)t[k]=body_force[k]-c[k];const real_t* v=type[s]==4?f:t;for(int k=0;k<3;k++)result[adr[s]+k]=mat[9*st+k]*v[0]+mat[9*st+3+k]*v[1]+mat[9*st+6+k]*v[2];}
}

// Collective block-owned post pipeline. Each qualified spatial field is [6][NB];
// scalar fallback remains [NB][6], and the public exports remain [NB][6].
// qacc remains explicitly supplied; a joint N/post entry may bind its private output.
__device__ __forceinline__ void run_workspace(int w,const int* g_active,const int* g_first,const int* g_dv,const int* g_nv,const int* g_ne,const int64_t* g_nw,const int64_t* g_na,const int64_t* g_expected,const int64_t* g_state,const int64_t* g_consumed,const int64_t* g_forward,const int64_t* g_cf,int* g_g,const int* r_active,const real_t* r_qv,const real_t* r_qa,const real_t* r_ci,const real_t* r_cv,const real_t* r_cd,const real_t* r_cdd,const real_t* r_xi,const real_t* r_com,const real_t* r_sp,const real_t* r_sm,const real_t* r_xf,const real_t* r_ef,const int* r_nc,const int* r_ne,const int* r_neq,const int* r_dim,const int* r_adr,const int* r_g1,const int* r_g2,const real_t* r_frame,const real_t* r_pos,const int* r_parent,const int* r_da,const int* r_dn,const int* r_root,const int* r_gb,const real_t* r_grav,int r_cc,int r_ec,real_t* r_acc,real_t* r_ext,real_t* r_intr,real_t* r_decoded,int* r_status,int r_b,int r_v,int r_ns,int r_ng,const real_t* s_fr,const real_t* s_com,const real_t* s_pos,const real_t* s_mat,const int* s_active,const int* s_type,const int* s_site,const int* s_adr,const int* s_sb,const int* s_br,real_t* s_result,int* s_status,int s_b,int s_ns,const int* c_gate,const int* c_status,const int* c_ss,int* c_first,const int64_t* c_epoch,const real_t* c_ext,const real_t* c_intr,const real_t* c_sensor,real_t* c_pe,real_t* c_pi,real_t* c_ps,int* c_valid,int64_t* c_written,int c_bw,int c_sw,int diagnostics,real_t* workspace,bool qacc_world_stride=true,const real_t* bias_acc=nullptr,const real_t* bias_force=nullptr,bool cache_ok=false,int* body_index=nullptr,int* body_offsets=nullptr,int* body_counts=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int nlevels=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0){
 const bool work_planes=body_index&&body_offsets&&body_counts&&level_offsets&&level_bodies&&children_offsets&&children_indices;
 real_t* work_acc=workspace;real_t* work_ext=work_acc+6*r_b;
 real_t* work_int=work_ext+6*r_b;real_t* work_sensor=work_int+6*r_b;
 if(threadIdx.x==0)gate(w,g_active,g_first,g_dv,g_nv,g_ne,g_nw,g_na,g_expected,g_state,g_consumed,g_forward,g_cf,g_g);
 rne_shared(w,r_active,r_qv,r_qa,r_ci,r_cv,r_cd,r_cdd,r_xi,r_com,r_sp,r_sm,r_xf,r_ef,r_nc,r_ne,r_neq,r_dim,r_adr,r_g1,r_g2,r_frame,r_pos,r_parent,r_da,r_dn,r_root,r_gb,r_grav,r_cc,r_ec,work_acc,work_ext,work_int,r_decoded,r_status,r_b,r_v,r_ns,r_ng,true,diagnostics!=0,qacc_world_stride,bias_acc,bias_force,cache_ok,body_index,body_offsets,body_counts,level_offsets,level_bodies,nlevels,children_offsets,children_indices,event_ping,event_pong,contact_index_mode,radix_min_contacts,cached_event_body,cached_contact_meta,topology_cache);
 // All lanes remain live through publication. Only sensor math and state decisions
 // are leader-owned; their result is visible before the collective copies.
 if(threadIdx.x==0){
  sensor_shared(w,work_int,s_com,s_pos,s_mat,s_active,s_type,s_site,s_adr,s_sb,s_br,work_sensor,s_status,s_b,s_ns,true,r_status[w]==0,true,work_planes);
  if(r_active[w]==1&&r_status[w]==0){if(s_status[w]!=0)r_status[w]=4;else if(!post_finite(work_sensor,6)){r_status[w]=5;libero_vec::fill_serial(work_sensor,6,REAL_C(0.0));}}
 }
 __syncthreads();
 if(diagnostics){
  post_export(work_acc,r_acc+w*6*r_b,6*r_b,r_b,work_planes);
  post_export(work_ext,r_ext+w*6*r_b,6*r_b,r_b,work_planes);
  post_export(work_int,r_intr+w*6*r_b,6*r_b,r_b,work_planes);
  libero_vec::copy_block(work_sensor,s_result+6*w,6);
 }
 __syncthreads();
 if(threadIdx.x==0){
  c_valid[w]=0;
  if(c_gate[w]==1&&c_first[w]==0&&(r_status[w]!=0||s_status[w]!=0))c_first[w]=215;
 }
 __syncthreads();
 // These owner/world predicates are uniform. Blocked worlds retain all public
 // payload and written epochs; diagnostics retain their original private writes.
 if(c_gate[w]!=1||c_first[w]!=0)return;
 post_export(work_ext,c_pe+w*c_bw,c_bw,r_b,work_planes);
 post_export(work_int,c_pi+w*c_bw,c_bw,r_b,work_planes);
 libero_vec::copy_block(work_sensor,c_ps+w*c_sw,c_sw);
 __syncthreads();
 if(threadIdx.x==0){c_written[w]=c_epoch[w];c_valid[w]=1;}

}

__device__ __forceinline__ void run_root(int w,const int* g_active,const int* g_first,const int* g_dv,const int* g_nv,const int* g_ne,const int64_t* g_nw,const int64_t* g_na,const int64_t* g_expected,const int64_t* g_state,const int64_t* g_consumed,const int64_t* g_forward,const int64_t* g_cf,int* g_g,const int* r_active,const real_t* r_qv,const real_t* r_qa,const real_t* r_ci,const real_t* r_cv,const real_t* r_cd,const real_t* r_cdd,const real_t* r_xi,const real_t* r_com,const real_t* r_sp,const real_t* r_sm,const real_t* r_xf,const real_t* r_ef,const int* r_nc,const int* r_ne,const int* r_neq,const int* r_dim,const int* r_adr,const int* r_g1,const int* r_g2,const real_t* r_frame,const real_t* r_pos,const int* r_parent,const int* r_da,const int* r_dn,const int* r_root,const int* r_gb,const real_t* r_grav,int r_cc,int r_ec,real_t* r_acc,real_t* r_ext,real_t* r_intr,real_t* r_decoded,int* r_status,int r_b,int r_v,int r_ns,int r_ng,const real_t* s_fr,const real_t* s_com,const real_t* s_pos,const real_t* s_mat,const int* s_active,const int* s_type,const int* s_site,const int* s_adr,const int* s_sb,const int* s_br,real_t* s_result,int* s_status,int s_b,int s_ns,const int* c_gate,const int* c_status,const int* c_ss,int* c_first,const int64_t* c_epoch,const real_t* c_ext,const real_t* c_intr,const real_t* c_sensor,real_t* c_pe,real_t* c_pi,real_t* c_ps,int* c_valid,int64_t* c_written,int c_bw,int c_sw,int diagnostics,const real_t* cache_acc=nullptr,const real_t* cache_force=nullptr,const int* cache_status=nullptr,const int* cache_valid=nullptr,const int64_t* cache_state=nullptr,const int64_t* cache_forward=nullptr,int cache_gravity_disabled=0,int* body_index=nullptr,int* body_offsets=nullptr,int* body_counts=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int nlevels=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0){
 extern __shared__ real_t post_work[];
 run_workspace(w,g_active,g_first,g_dv,g_nv,g_ne,g_nw,g_na,g_expected,g_state,g_consumed,g_forward,g_cf,g_g,r_active,r_qv,r_qa,r_ci,r_cv,r_cd,r_cdd,r_xi,r_com,r_sp,r_sm,r_xf,r_ef,r_nc,r_ne,r_neq,r_dim,r_adr,r_g1,r_g2,r_frame,r_pos,r_parent,r_da,r_dn,r_root,r_gb,r_grav,r_cc,r_ec,r_acc,r_ext,r_intr,r_decoded,r_status,r_b,r_v,r_ns,r_ng,s_fr,s_com,s_pos,s_mat,s_active,s_type,s_site,s_adr,s_sb,s_br,s_result,s_status,s_b,s_ns,c_gate,c_status,c_ss,c_first,c_epoch,c_ext,c_intr,c_sensor,c_pe,c_pi,c_ps,c_valid,c_written,c_bw,c_sw,diagnostics,post_work,true,cache_acc,cache_force,cache_valid&&cache_status&&cache_state&&cache_forward&&cache_acc&&cache_force&&cache_gravity_disabled==0&&cache_valid[w]==1&&cache_status[w]==0&&cache_state[w]==g_state[w]&&cache_forward[w]==g_forward[w],body_index,body_offsets,body_counts,level_offsets,level_bodies,nlevels,children_offsets,children_indices,event_ping,event_pong,contact_index_mode,radix_min_contacts,cached_event_body,cached_contact_meta,topology_cache);
}
}

namespace libero_post { using post_root38_original::post_export; __device__ __forceinline__ void run(int w,const int* g_active,const int* g_first,const int* g_dv,const int* g_nv,const int* g_ne,const int64_t* g_nw,const int64_t* g_na,const int64_t* g_expected,const int64_t* g_state,const int64_t* g_consumed,const int64_t* g_forward,const int64_t* g_cf,int* g_g,const int* r_active,const real_t* r_qv,const real_t* r_qa,const real_t* r_ci,const real_t* r_cv,const real_t* r_cd,const real_t* r_cdd,const real_t* r_xi,const real_t* r_com,const real_t* r_sp,const real_t* r_sm,const real_t* r_xf,const real_t* r_ef,const int* r_nc,const int* r_ne,const int* r_neq,const int* r_dim,const int* r_adr,const int* r_g1,const int* r_g2,const real_t* r_frame,const real_t* r_pos,const int* r_parent,const int* r_da,const int* r_dn,const int* r_root,const int* r_gb,const real_t* r_grav,int r_cc,int r_ec,real_t* r_acc,real_t* r_ext,real_t* r_intr,real_t* r_decoded,int* r_status,int r_b,int r_v,int r_ns,int r_ng,const real_t* s_fr,const real_t* s_com,const real_t* s_pos,const real_t* s_mat,const int* s_active,const int* s_type,const int* s_site,const int* s_adr,const int* s_sb,const int* s_br,real_t* s_result,int* s_status,int s_b,int s_ns,const int* c_gate,const int* c_status,const int* c_ss,int* c_first,const int64_t* c_epoch,const real_t* c_ext,const real_t* c_intr,const real_t* c_sensor,real_t* c_pe,real_t* c_pi,real_t* c_ps,int* c_valid,int64_t* c_written,int c_bw,int c_sw,int diagnostics,const real_t* cache_acc=nullptr,const real_t* cache_force=nullptr,const int* cache_status=nullptr,const int* cache_valid=nullptr,const int64_t* cache_state=nullptr,const int64_t* cache_forward=nullptr,int cache_gravity_disabled=0,int* body_index=nullptr,int* body_offsets=nullptr,int* body_counts=nullptr,const int* level_offsets=nullptr,const int* level_bodies=nullptr,int nlevels=0,const int* children_offsets=nullptr,const int* children_indices=nullptr,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0,const int* chain_offsets=nullptr,const int* chain_bodies=nullptr,const int* chain_level_offsets=nullptr,const int* chain_level_indices=nullptr,int nchain_levels=0) {
 if(!diagnostics&&r_nc[w]==512)post_root38_fast::run_root(w,g_active,g_first,g_dv,g_nv,g_ne,g_nw,g_na,g_expected,g_state,g_consumed,g_forward,g_cf,g_g,r_active,r_qv,r_qa,r_ci,r_cv,r_cd,r_cdd,r_xi,r_com,r_sp,r_sm,r_xf,r_ef,r_nc,r_ne,r_neq,r_dim,r_adr,r_g1,r_g2,r_frame,r_pos,r_parent,r_da,r_dn,r_root,r_gb,r_grav,r_cc,r_ec,r_acc,r_ext,r_intr,r_decoded,r_status,r_b,r_v,r_ns,r_ng,s_fr,s_com,s_pos,s_mat,s_active,s_type,s_site,s_adr,s_sb,s_br,s_result,s_status,s_b,s_ns,c_gate,c_status,c_ss,c_first,c_epoch,c_ext,c_intr,c_sensor,c_pe,c_pi,c_ps,c_valid,c_written,c_bw,c_sw,diagnostics,cache_acc,cache_force,cache_status,cache_valid,cache_state,cache_forward,cache_gravity_disabled,body_index,body_offsets,body_counts,level_offsets,level_bodies,nlevels,children_offsets,children_indices,event_ping,event_pong,contact_index_mode,radix_min_contacts,cached_event_body,cached_contact_meta,topology_cache);
 else post_root38_original::run_original(w,g_active,g_first,g_dv,g_nv,g_ne,g_nw,g_na,g_expected,g_state,g_consumed,g_forward,g_cf,g_g,r_active,r_qv,r_qa,r_ci,r_cv,r_cd,r_cdd,r_xi,r_com,r_sp,r_sm,r_xf,r_ef,r_nc,r_ne,r_neq,r_dim,r_adr,r_g1,r_g2,r_frame,r_pos,r_parent,r_da,r_dn,r_root,r_gb,r_grav,r_cc,r_ec,r_acc,r_ext,r_intr,r_decoded,r_status,r_b,r_v,r_ns,r_ng,s_fr,s_com,s_pos,s_mat,s_active,s_type,s_site,s_adr,s_sb,s_br,s_result,s_status,s_b,s_ns,c_gate,c_status,c_ss,c_first,c_epoch,c_ext,c_intr,c_sensor,c_pe,c_pi,c_ps,c_valid,c_written,c_bw,c_sw,diagnostics,cache_acc,cache_force,cache_status,cache_valid,cache_state,cache_forward,cache_gravity_disabled,body_index,body_offsets,body_counts,level_offsets,level_bodies,nlevels,children_offsets,children_indices,event_ping,event_pong,contact_index_mode,radix_min_contacts,cached_event_body,cached_contact_meta,topology_cache);
}}
