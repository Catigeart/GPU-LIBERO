#include "precision.cuh"
#pragma once
#include "post_dataflow.cuh"
// P09 prototype ABI post_sensor_only_v1.
// Input contract: the ORIGINAL full Post validation has succeeded; parent,
// level/children/chain tables and masks have cold owner/extent seals. Workspace
// is zero-filled and belongs solely to this ABI. Only selected sensor output
// may be published. No full-force/cacc/contact-diagnostic valid flag is set.
// Current contact forces and qacc are always read; Newton runs unchanged.
namespace libero_post {
__device__ inline void rne_sensor_only_validated(int w,const real_t* qv,const real_t* qa,const real_t* ci,const real_t* cv,
 const real_t* cd,const real_t* cdd,const real_t* xi,const real_t* com,const real_t* xf,const real_t* ef,
 const int* nc,const int* dim,const int* adr,const int* g1,const int* g2,const real_t* frame,const real_t* pos,
 const int* parent,const int* da,const int* dn,const int* root,const int* gb,const real_t* grav,int cc,
 real_t* acc,real_t* ext,real_t* intr,real_t* decoded,int b,int v,bool diagnostics,bool qacc_world_stride,
 const real_t* bias_acc,const real_t* bias_force,bool cache_ok,int* body_index,int* body_offsets,int* body_counts,
 const int* level_offsets,const int* level_bodies,int nlevels,const int* children_offsets,const int* children_indices,const int* force_mask,const int* acc_mask,int* event_ping=nullptr,int* event_pong=nullptr,int contact_index_mode=0,int radix_min_contacts=256,
 int* cached_event_body=nullptr,int* cached_contact_meta=nullptr,int topology_cache=0,const int* chain_offsets=nullptr,const int* chain_bodies=nullptr,const int* chain_level_offsets=nullptr,const int* chain_level_indices=nullptr,int nchain_levels=0){
 qv+=w*v;if(qacc_world_stride)qa+=w*v;ci+=w*b*10;cv+=w*b*6;cd+=w*v*6;cdd+=w*v*6;
 xi+=w*b*3;com+=w*b*3;xf+=w*b*6;dim+=w*cc;adr+=w*cc;g1+=w*cc;g2+=w*cc;
 frame+=w*cc*9;pos+=w*cc*3;decoded+=w*cc*6;
 body_index+=w*2*cc;body_offsets+=w*(b+1);body_counts+=w*b;
 post_contact_index(nc[w],cc,b,adr,g1,g2,gb,body_index,body_offsets,body_counts,
  event_ping?event_ping+w*2*cc:nullptr,event_pong?event_pong+w*2*cc:nullptr,contact_index_mode,radix_min_contacts,
  cached_event_body?cached_event_body+w*2*cc:nullptr,cached_contact_meta?cached_contact_meta+2*w:nullptr,topology_cache);
 // Bounded contact tile replaces repeated per-incidence wrench decoding.
 // Every body keeps its stable CSR cursor; tile progression cannot reorder sides.
 __shared__ real_t contact_tile[9*LIBERO_POST_CONTACT_TILE];
 // Supported body levels fit one block in the qualified path. For larger body
 // sets the body owner loops below use the existing per-body path instead.
 if(b-1<=int(blockDim.x)){
  const int body=1+int(threadIdx.x);
  real_t local[6]={0,0,0,0,0,0};
  int cursor=body<b&&force_mask[body]?body_offsets[body]:0;
  if(body<b&&force_mask[body]){bool nz=false;for(int k=0;k<6;k++)nz|=xf[6*body+k]!=0;
   if(nz)wrench(local,xf+6*body,xf+6*body+3,com+3*root[body],xi+3*body,1);}
  for(int base=0;base<nc[w];base+=LIBERO_POST_CONTACT_TILE){
   const int end=min(base+LIBERO_POST_CONTACT_TILE,nc[w]);
   for(int j=base+int(threadIdx.x);j<end;j+=blockDim.x)if(adr[j]>=0){
    const int slot=j-base;real_t d[6]={0,0,0,0,0,0};
    post_decode(ef+adr[j],dim[j],d);real_t fm[9],point[3];libero_vec::copy_serial(frame+9*j,fm,9);libero_vec::copy_serial(pos+3*j,point,3);
    if(diagnostics)libero_vec::copy_serial(d,decoded+6*j,6);
    for(int k=0;k<3;k++){
     contact_tile[k*LIBERO_POST_CONTACT_TILE+slot]=fm[k]*d[0]+fm[3+k]*d[1]+fm[6+k]*d[2];
     contact_tile[(3+k)*LIBERO_POST_CONTACT_TILE+slot]=fm[k]*d[3]+fm[3+k]*d[4]+fm[6+k]*d[5];
     contact_tile[(6+k)*LIBERO_POST_CONTACT_TILE+slot]=point[k];
    }
   }
   __syncthreads();
   if(body<b&&force_mask[body])while(cursor<body_offsets[body+1]){
    const int event=body_index[cursor],j=event>>1,side=event&1;
    if(j>=end)break;
    const int slot=j-base;real_t f[3],t[3],point[3];
    for(int k=0;k<3;k++){f[k]=contact_tile[k*LIBERO_POST_CONTACT_TILE+slot];t[k]=contact_tile[(3+k)*LIBERO_POST_CONTACT_TILE+slot];point[k]=contact_tile[(6+k)*LIBERO_POST_CONTACT_TILE+slot];}
    wrench(local,f,t,com+3*root[body],point,side?1:-1);++cursor;
   }
   __syncthreads();
  }
  if(body<b&&force_mask[body])for(int k=0;k<6;k++)ext[k*b+body]=local[k];
 }else{
  // General body-count fallback keeps the existing stable incidence algorithm.
  if(diagnostics)for(int j=threadIdx.x;j<nc[w];j+=blockDim.x)if(adr[j]>=0)
   libero_vec::copy_serial(ef+adr[j],decoded+6*j,dim[j]);
  for(int body=1+threadIdx.x;body<b;body+=blockDim.x){
   if(!force_mask[body])continue;
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
 const bool use_chains=LIBERO_POST_CHAINS&&chain_offsets&&chain_bodies&&chain_level_offsets&&chain_level_indices;
 if(use_chains){
  for(int level=0;level<nchain_levels;level++){
   for(int c=chain_level_offsets[level]+threadIdx.x;c<chain_level_offsets[level+1];c+=blockDim.x){
    const int chain=chain_level_indices[c];
    for(int i=chain_offsets[chain];i<chain_offsets[chain+1];i++){
     const int j=chain_bodies[i];if(!acc_mask[j])continue;
     for(int k=0;k<6;k++)acc[k*b+j]=acc[k*b+parent[j]]+dof(cache_ok?cd:cdd,cache_ok?qa:qv,da[j],dn[j],k);
     if(!cache_ok)for(int k=0;k<6;k++)acc[k*b+j]+=dof(cd,qa,da[j],dn[j],k);
    }
   }
   __syncthreads();
  }
 }else{
 for(int level=0;level<nlevels;level++){
  for(int i=level_offsets[level]+threadIdx.x;i<level_offsets[level+1];i+=blockDim.x){
   const int j=level_bodies[i];if(!acc_mask[j])continue;
   for(int k=0;k<6;k++)acc[k*b+j]=acc[k*b+parent[j]]+dof(cache_ok?cd:cdd,cache_ok?qa:qv,da[j],dn[j],k);
   if(!cache_ok)for(int k=0;k<6;k++)acc[k*b+j]+=dof(cd,qa,da[j],dn[j],k);
  }
  __syncthreads();
 }
 }
 if(cache_ok){bias_acc+=w*6*b;bias_force+=w*6*b;}
 for(int j=1+threadIdx.x;j<b;j+=blockDim.x){
  if(!force_mask[j])continue;
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
 if(use_chains){
  for(int level=nchain_levels-1;level>=0;level--){
   for(int c=chain_level_offsets[level]+threadIdx.x;c<chain_level_offsets[level+1];c+=blockDim.x){
    const int chain=chain_level_indices[c];
    for(int i=chain_offsets[chain+1]-1;i>=chain_offsets[chain];i--){
     const int j=chain_bodies[i];if(!force_mask[j])continue;
     for(int c=children_offsets[j];c<children_offsets[j+1];c++){
      const int child=children_indices[c];for(int k=0;k<6;k++)intr[k*b+j]+=intr[k*b+child];
     }
    }
   }
   __syncthreads();
  }
 }else{
 for(int level=nlevels-1;level>=0;level--){
  for(int i=level_offsets[level]+threadIdx.x;i<level_offsets[level+1];i+=blockDim.x){
   const int j=level_bodies[i];if(!force_mask[j])continue;
   for(int c=children_offsets[j];c<children_offsets[j+1];c++){
    const int child=children_indices[c];
    for(int k=0;k<6;k++)intr[k*b+j]+=intr[k*b+child];
   }
  }
  __syncthreads();
 }
 }
 if(threadIdx.x==0&&force_mask[0])for(int c=children_offsets[0];c<children_offsets[1];c++){
  const int child=children_indices[c];for(int k=0;k<6;k++)intr[k*b]+=intr[k*b+child];
 }
 __syncthreads();
}
} // namespace libero_post
