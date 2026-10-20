#include "precision.cuh"
#pragma once
#include <math.h>
namespace libero_observe {
__device__ __forceinline__ void rotation32(const real_t* q,real_t* r) {
 float v[4]={(float)q[3],(float)q[0],(float)q[1],(float)q[2]};
 float n=0;for(int i=0;i<4;++i)n+=v[i]*v[i];
 for(int i=0;i<9;++i)r[i]=(i%4==0)?REAL_C(1.):REAL_C(0.);
 if(n>=float(REAL_C(4.)*real_t(REAL_EPSILON))){
  real_t scale=sqrt(REAL_C(2.)/real_t(n));for(int i=0;i<4;++i)v[i]=float(real_t(v[i])*scale);
  r[0]=float(REAL_C(1.)-real_t(v[2]*v[2])-real_t(v[3]*v[3]));
  r[1]=v[1]*v[2]-v[3]*v[0];r[2]=v[1]*v[3]+v[2]*v[0];
  r[3]=v[1]*v[2]+v[3]*v[0];r[4]=float(REAL_C(1.)-real_t(v[1]*v[1])-real_t(v[3]*v[3]));
  r[5]=v[2]*v[3]-v[1]*v[0];r[6]=v[1]*v[3]-v[2]*v[0];r[7]=v[2]*v[3]+v[1]*v[0];
  r[8]=float(REAL_C(1.)-real_t(v[1]*v[1])-real_t(v[2]*v[2]));
 }
}
__device__ __forceinline__ void matrix_quaternion32(const float* m,real_t* out){
 float k[16]={
 m[0]-m[4]-m[8],m[1]+m[3],m[2]+m[6],m[7]-m[5],
 m[1]+m[3],m[4]-m[0]-m[8],m[5]+m[7],m[2]-m[6],
 m[2]+m[6],m[5]+m[7],m[8]-m[0]-m[4],m[3]-m[1],
 m[7]-m[5],m[2]-m[6],m[3]-m[1],m[0]+m[4]+m[8]};
 for(int i=0;i<16;++i)k[i]/=3.f;
 int largest=0;for(int i=1;i<4;++i)if(k[5*i]>k[5*largest])largest=i;
 float q[4]={0,0,0,0};q[largest]=1;
 for(int it=0;it<24;++it){float v[4]={0,0,0,0};
  for(int i=0;i<4;++i)for(int j=0;j<4;++j)v[i]+=k[4*i+j]*q[j];
  float n=0;for(int i=0;i<4;++i)n+=v[i]*v[i];n=sqrtf(n);
  for(int i=0;i<4;++i)q[i]=v[i]/n;
 }
 for(int i=0;i<4;++i)out[i]=q[3]<0?-real_t(q[i]):real_t(q[i]);
}
__device__ __forceinline__ real_t raw_read(int field,int j,int w,
 const int* strides,const real_t* qp,const real_t* qv,const real_t* bp,
 const real_t* bq,const real_t* sp){
 const real_t* p=field==0?qp:field==1?qv:field==2?bp:field==3?bq:sp;
 return p[w*strides[field]+j];
}
__device__ __forceinline__ void lowdim_sensor(int s,int w,int ns,int width,int source_stride,
 const int* op,const int* offsets,const int* sizes,const int* index_offsets,
 const int* indices,const int* dep0,const int* dep1,const int* dep2,
 const int* presence_deps,const int* enabled,const real_t* periods,
 const int* strides,const int* packoffsets,
 const real_t* qp,const real_t* qv,const real_t* bp,const real_t* bq,const real_t* sp,
 real_t* packed,int materialize,real_t* cache,real_t* observed,real_t* sidequat,
 real_t* elapsed,int* sampled,int* present,const int* mask,real_t dt,int force,int* events){
 int b=w*width,sb=w*ns;
  events[sb+s]=0;if(!enabled[s])return;
  real_t e=elapsed[sb+s]+dt;elapsed[sb+s]=e;
  bool take=(!sampled[sb+s]&&periods[s]>=e)||force;
  bool fallback=e>=periods[s]&&!sampled[sb+s];
  if(take||fallback){
   int at=b+offsets[s],d0=b+dep0[s],d1=b+dep1[s],d2=b+dep2[s];
   for(int j=0;j<sizes[s];++j){real_t value=0;
    if(op[s]==0){int absolute=indices[index_offsets[s]+j],field=0;
     while(field<4&&absolute>=packoffsets[field+1])++field;
     value=raw_read(field,absolute-packoffsets[field],w,strides,qp,qv,bp,bq,sp);}
    else if(op[s]==1&&present[sb+presence_deps[3*s]])value=cos(cache[d0+j]);
    else if(op[s]==2&&present[sb+presence_deps[3*s]])value=sin(cache[d0+j]);
    cache[at+j]=value;
   }
   if(op[s]==3){
    if(!present[sb+presence_deps[3*s+1]]||!present[sb+presence_deps[3*s]]){
     for(int row=0;row<4;++row)cache[at+5*row]=REAL_C(1.);
    }else{real_t r[9];rotation32(cache+d1,r);
     for(int row=0;row<3;++row){real_t t=0;
      for(int col=0;col<3;++col){cache[at+row*4+col]=r[col*3+row];t-=r[col*3+row]*real_t(float(cache[d0+col]));}
      cache[at+row*4+3]=t;
     }cache[at+15]=REAL_C(1.);
    }
   }else if(op[s]==4){
    if(present[sb+presence_deps[3*s]]&&present[sb+presence_deps[3*s+1]]&&present[sb+presence_deps[3*s+2]]){
     real_t ro[9];float rr[9];rotation32(cache+d1,ro);
     for(int row=0;row<3;++row){real_t t=cache[d2+row*4+3];
      for(int col=0;col<3;++col){t+=cache[d2+row*4+col]*real_t(float(cache[d0+col]));
       real_t v=0;for(int k=0;k<3;++k)v+=cache[d2+row*4+k]*ro[k*3+col];rr[row*3+col]=float(v);
      }cache[at+row]=t;
     }matrix_quaternion32(rr,sidequat+b+offsets[s]);
    }
   }else if(op[s]==5&&present[sb+presence_deps[3*s]]){
    for(int j=0;j<4;++j)cache[at+j]=sidequat[d0+j];
   }
   if(sizes[s]>0)libero_vec::copy_serial(cache+at,observed+at,sizes[s]);
   present[sb+s]=1;sampled[sb+s]=1;events[sb+s]=1;
  }
  if(e>=periods[s]){elapsed[sb+s]=fmod(e,periods[s]);sampled[sb+s]=0;}

}
__device__ void lowdim(int w,int ns,int width,int source_stride,
 const int* op,const int* offsets,const int* sizes,const int* index_offsets,
 const int* indices,const int* dep0,const int* dep1,const int* dep2,
 const int* presence_deps,const int* enabled,const real_t* periods,
 const int* strides,const int* packoffsets,
 const real_t* qp,const real_t* qv,const real_t* bp,const real_t* bq,const real_t* sp,
 real_t* packed,int materialize,real_t* cache,real_t* observed,real_t* sidequat,
 real_t* elapsed,int* sampled,int* present,const int* mask,real_t dt,int force,int* events){
 // Compatibility diagnostics remain materialized even for masked worlds.
 if(materialize)for(int f=0;f<5;++f){
  const real_t* p=f==0?qp:f==1?qv:f==2?bp:f==3?bq:sp;
  if(strides[f]>0)libero_vec::copy_serial(p+w*strides[f],packed+w*source_stride+packoffsets[f],strides[f]);
 }
 if(!mask[w])return;
 for(int s=0;s<ns;++s)lowdim_sensor(s,w,ns,width,source_stride,op,offsets,sizes,index_offsets,indices,dep0,dep1,dep2,presence_deps,enabled,periods,strides,packoffsets,qp,qv,bp,bq,sp,packed,materialize,cache,observed,sidequat,elapsed,sampled,present,mask,dt,force,events);
}
// Cold schedule preserves all original read/write/presence hazards. Integer
// descriptors are checked against an exact frozen snapshot every invocation;
// mutable descriptors transparently retain the accepted ordered path.
__device__ void lowdim_dag(int w,int ns,int width,int source_stride,
 const int* op,const int* offsets,const int* sizes,const int* index_offsets,
 const int* indices,const int* dep0,const int* dep1,const int* dep2,
 const int* presence_deps,const int* enabled,const real_t* periods,
 const int* strides,const int* packoffsets,
 const real_t* qp,const real_t* qv,const real_t* bp,const real_t* bq,const real_t* sp,
 real_t* packed,int materialize,real_t* cache,real_t* observed,real_t* sidequat,
 real_t* elapsed,int* sampled,int* present,const int* mask,real_t dt,int force,int* events,int nlayers,int nindex,
 const int* layer_offsets,const int* sensor_order,const int* schema,
 const int* run_offsets,const int* element_offsets,const int* runs,const int* layout_schema){
 __shared__ int schema_bad;
 if(threadIdx.x==0)schema_bad=nlayers<=0;
 __syncthreads();
 const int* arrays[9]={op,offsets,sizes,index_offsets,indices,dep0,dep1,dep2,presence_deps};
 int cursor=0;
 for(int a=0;a<9;++a){
  const int count=a==4?nindex:(a==8?3*ns:ns);
  for(int z=threadIdx.x;z<count;z+=blockDim.x)
   if(arrays[a][z]!=schema[cursor+z])atomicExch(&schema_bad,1);
  cursor+=count;
 }
 for(int z=threadIdx.x;z<5;z+=blockDim.x)
  if(strides[z]!=layout_schema[z]||packoffsets[z]!=layout_schema[5+z])atomicExch(&schema_bad,1);
 __syncthreads();
 if(schema_bad){if(threadIdx.x==0)lowdim(w,ns,width,source_stride,op,offsets,sizes,index_offsets,indices,dep0,dep1,dep2,presence_deps,enabled,periods,strides,packoffsets,qp,qv,bp,bq,sp,packed,materialize,cache,observed,sidequat,elapsed,sampled,present,mask,dt,force,events);__syncthreads();return;}
 if(materialize)for(int f=0;f<5;++f){
  const real_t* p=f==0?qp:f==1?qv:f==2?bp:f==3?bq:sp;
  if(strides[f]>0)libero_vec::copy_block(p+w*strides[f],packed+w*source_stride+packoffsets[f],strides[f]);
 }
 __syncthreads();
 if(!mask[w])return;
 for(int level=0;level<nlayers;++level){
  if(element_offsets[level]==element_offsets[level+1]){
   for(int at=layer_offsets[level]+threadIdx.x;at<layer_offsets[level+1];at+=blockDim.x)
    lowdim_sensor(sensor_order[at],w,ns,width,source_stride,op,offsets,sizes,index_offsets,indices,dep0,dep1,dep2,presence_deps,enabled,periods,strides,packoffsets,qp,qv,bp,bq,sp,packed,materialize,cache,observed,sidequat,elapsed,sampled,present,mask,dt,force,events);
   __syncthreads();continue;
  }
  // Clock ownership remains one lane per sensor; data lanes own independent
  // elements. Complex pose math retains its original order and precision.
  for(int at=layer_offsets[level]+threadIdx.x;at<layer_offsets[level+1];at+=blockDim.x){
   int s=sensor_order[at],sb=w*ns+s;
   if(op[s]>2){lowdim_sensor(s,w,ns,width,source_stride,op,offsets,sizes,index_offsets,indices,dep0,dep1,dep2,presence_deps,enabled,periods,strides,packoffsets,qp,qv,bp,bq,sp,packed,materialize,cache,observed,sidequat,elapsed,sampled,present,mask,dt,force,events);continue;}
   events[sb]=0;if(!enabled[s])continue;
   real_t e=elapsed[sb]+dt;elapsed[sb]=e;
   bool take=(!sampled[sb]&&periods[s]>=e)||force;
   bool fallback=e>=periods[s]&&!sampled[sb];
   if(take||fallback){events[sb]=1;sampled[sb]=1;}
   if(e>=periods[s]){elapsed[sb]=fmod(e,periods[s]);sampled[sb]=0;}
  }
  __syncthreads();
  for(int run=run_offsets[level];run<run_offsets[level+1];++run){
   const int* r=runs+6*run;int s=r[1];
   if(!events[w*ns+s])continue;
   int count=r[5];
   for(int j=2*threadIdx.x;j<count;j+=2*blockDim.x){
    int out=w*width+r[4]+j;real_t x=0,y=0;bool pair=j+1<count;
    if(r[2]>=0){
     const real_t* p=r[2]==0?qp:r[2]==1?qv:r[2]==2?bp:r[2]==3?bq:sp;
     p+=w*strides[r[2]]+r[3]+j;
     if(pair){real2_t v=libero_vec::load_pair(p);x=v.x;y=v.y;}else x=*p;
    }else if(present[w*ns+presence_deps[3*s]]){
     const real_t* p=cache+w*width+r[3]+j;
     if(pair){real2_t v=libero_vec::load_pair(p);x=r[2]==-1?cos(v.x):sin(v.x);y=r[2]==-1?cos(v.y):sin(v.y);}
     else x=r[2]==-1?cos(*p):sin(*p);
    }
    if(pair){libero_vec::store_pair(cache+out,x,y);libero_vec::store_pair(observed+out,x,y);}
    else {cache[out]=x;observed[out]=x;}
   }
  }
  __syncthreads();
  for(int at=layer_offsets[level]+threadIdx.x;at<layer_offsets[level+1];at+=blockDim.x){
   int s=sensor_order[at];if(op[s]<=2&&events[w*ns+s])present[w*ns+s]=1;
  }
  __syncthreads();
 }
}
__device__ __forceinline__ const real_t* scene_field(int i,
 const real_t* p0,const real_t* p1,const real_t* p2,const real_t* p3,
 const real_t* p4,const real_t* p5,const real_t* p6,const real_t* p7,
 const real_t* p8,const real_t* p9,const real_t* p10,const real_t* p11,const real_t* p12){
 switch(i){case 0:return p0;case 1:return p1;case 2:return p2;case 3:return p3;
 case 4:return p4;case 5:return p5;case 6:return p6;case 7:return p7;
 case 8:return p8;case 9:return p9;case 10:return p10;case 11:return p11;default:return p12;}
}
__device__ void scenes(int w,int nc,int stride,const int* offsets,const int* lengths,
 const real_t* p0,const real_t* p1,const real_t* p2,const real_t* p3,
 const real_t* p4,const real_t* p5,const real_t* p6,const real_t* p7,
 const real_t* p8,const real_t* p9,const real_t* p10,const real_t* p11,const real_t* p12,
 real_t* packed,int materialize,real_t* out,const real_t* periods,const int* enabled,
 real_t* elapsed,int* sampled,int* generation,const int* mask,real_t dt,int force){
 if(materialize)for(int f=0;f<13;++f){
  const real_t* p=scene_field(f,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12);
  if(lengths[f]>0)libero_vec::copy_block(p+w*lengths[f],packed+w*stride+offsets[f],lengths[f]);
 }
 if(!mask[w])return;
 __shared__ int capture;
 for(int c=0;c<nc;++c){
  int s=w*nc+c;
  if(threadIdx.x==0){capture=0;
   if(enabled[c]){real_t e=elapsed[s]+dt;
    capture=(!sampled[s]&&periods[c]>=e)||force||(e>=periods[c]&&!sampled[s]);
    if(capture){generation[s]+=1;sampled[s]=1;}
    if(e>=periods[c]){e=fmod(e,periods[c]);sampled[s]=0;}elapsed[s]=e;
   }
  }__syncthreads();
  if(capture)for(int f=0;f<13;++f){
   const real_t* p=scene_field(f,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12);
   if(lengths[f]>0)libero_vec::copy_block(p+w*lengths[f],out+s*stride+offsets[f],lengths[f]);
  }__syncthreads();
 }
}
}
