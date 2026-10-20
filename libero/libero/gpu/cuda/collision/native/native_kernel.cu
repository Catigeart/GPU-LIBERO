#include <precision.cuh>
// Private SoA-to-original-C adapter. Every query runs entirely on device.
#include "native_source.cuh"
static_assert(sizeof(real_t)==FLASH_REAL_BITS/8 && sizeof(float)==4 && sizeof(int)==4,"dtype ABI");
static_assert(sizeof(mjContact)==13*sizeof(real_t) && offsetof(mjContact,frame)==4*sizeof(real_t),"private contact layout");
// This private struct is an adapter, not the native MuJoCo mjContact ABI.
__global__ void native_kernel(
 const int* cg1,const int* cg2,const int* ct1,const int* ct2,const real_t* margin,
 const int* counts,const int* active,const real_t* xpos,const real_t* xmat,
 const real_t* size,const real_t* rbound,const int* dataid,
 const float* vert,const int* vertadr,const int* vertnum,const int* graphadr,const int* graph,
 int N,int P,int ngeom,int nmesh,int nvert,int ngraph,int maxiter,real_t tolerance,int multiccd,
 real_t* raw,int* out_count,int* status,int* diag,real_t* frame0,int* framewritten) {
 const size_t first=(size_t)blockIdx.x*blockDim.x;
 const size_t total=(size_t)N*P;
 const size_t slots=first<total?(total-first<blockDim.x?total-first:blockDim.x):0;
 // Initialize consecutive diagnostic words collectively before any query
 // owner can publish. Tail threads also enter this block barrier.
 for(size_t k=threadIdx.x;k<slots*16;k+=blockDim.x)diag[first*16+k]=0;
 __syncthreads();
 const size_t slot=first+threadIdx.x;
 if(slot>=total)return;
 int w=slot/P,j=slot%P;
 out_count[slot]=0; status[slot]=0;framewritten[slot]=0;
 if(active[w]==0)return;
 if(active[w]!=1){status[slot]=2;return;}
 if(counts[w]<0||counts[w]>P){status[slot]=3;return;}
 if(j>=counts[w])return;
 int a=cg1[slot],b=cg2[slot],ta=ct1[slot],tb=ct2[slot];
 if(ta<5 || tb>7 || ta>tb)return;
 if(a<0 || b<0 || a>=ngeom || b>=ngeom || a==b){status[slot]=4;return;}
 int types[2]={ta,tb},ids[2]={dataid[a],dataid[b]};
 real_t sz[6],rb[2]={rbound[a],rbound[b]},pos[6],mat[18];
 for(int k=0;k<3;k++){sz[k]=size[3*a+k];sz[3+k]=size[3*b+k];
   pos[k]=xpos[3*(w*ngeom+a)+k];pos[3+k]=xpos[3*(w*ngeom+b)+k];}
 for(int k=0;k<9;k++){mat[k]=xmat[9*(w*ngeom+a)+k];mat[9+k]=xmat[9*(w*ngeom+b)+k];}
 for(int k=0;k<6;k++)if(!isfinite(sz[k])||!isfinite(pos[k])){status[slot]=5;return;}
 for(int k=0;k<18;k++)if(!isfinite(mat[k])){status[slot]=5;return;}
 if(!isfinite(margin[slot])||maxiter<0||!isfinite(tolerance)||tolerance<0){status[slot]=5;return;}
 for(int k=0;k<2;k++)if(types[k]==7){
   int id=ids[k];
   if(id<0||id>=nmesh||vertadr[id]<0||vertnum[id]<=0||
      (long long)3*(vertadr[id]+(long long)vertnum[id])>nvert){status[slot]=7;return;}
   int ga=graphadr[id];
   if(ga>=0 && (ga>=ngraph||graph[ga]<=0||
       (long long)ga+2+2LL*graph[ga]>ngraph)){status[slot]=7;return;}
 }
 int error=0,fw=0,trace[16]={0};
 mjModel m={sz,rb,types,ids,(float*)vert,(int*)vertadr,(int*)vertnum,(int*)graphadr,(int*)graph,
   {maxiter,multiccd?1:0,tolerance},&error,trace,&fw,nvert,ngraph,nmesh,4096};
 mjData d={pos,mat};
 mjContact con[50];
 // Source preserves uninitialized fields; initialize scratch to explicit dirty output.
 for(int k=0;k<50;k++){con[k].dist=raw[slot*500+10*k];
  for(int h=0;h<3;h++)con[k].pos[h]=raw[slot*500+10*k+1+h];
  for(int h=0;h<6;h++)con[k].frame[h]=raw[slot*500+10*k+4+h];
  for(int h=6;h<9;h++)con[k].frame[h]=0;}
 int n=ta==6&&tb==6?mjc_BoxBox(&m,&d,con,0,1,margin[slot]):
                             mjc_Convex(&m,&d,con,0,1,margin[slot]);
 for(int k=0;k<16;k++)diag[16*slot+k]=trace[k];
 diag[16*slot+4]=n;diag[16*slot+5]=fw;
 if(error || n<0 || n>50){status[slot]=error?error:8;return;}
 for(int k=0;k<n;k++){
   if(!isfinite(con[k].dist)){status[slot]=5;return;}
   for(int h=0;h<3;h++)if(!isfinite(con[k].pos[h])){status[slot]=5;return;}
   for(int h=0;h<6;h++)if(!isfinite(con[k].frame[h])){status[slot]=5;return;}
 }
 if(fw){
  for(int h=0;h<6;h++)if(!isfinite(con[0].frame[h])){status[slot]=5;return;}
  framewritten[slot]=1;
  for(int h=0;h<6;h++)frame0[6*slot+h]=con[0].frame[h];
 }
 for(int k=0;k<n;k++){raw[slot*500+10*k]=con[k].dist;
  for(int h=0;h<3;h++)raw[slot*500+10*k+1+h]=con[k].pos[h];
  for(int h=0;h<6;h++)raw[slot*500+10*k+4+h]=con[k].frame[h];}
 out_count[slot]=n;
}
extern "C" int launch_native(
 const int* cg1,const int* cg2,const int* ct1,const int* ct2,const real_t* margin,
 const int* counts,const int* active,const real_t* xpos,const real_t* xmat,
 const real_t* size,const real_t* rbound,const int* dataid,
 const float* vert,const int* vertadr,const int* vertnum,const int* graphadr,const int* graph,
 int N,int P,int ngeom,int nmesh,int nvert,int ngraph,int maxiter,real_t tolerance,int multiccd,
 real_t* raw,int* out_count,int* status,int* diag,real_t* frame0,int* framewritten,cudaStream_t stream) {
 native_kernel<<<(N*P+63)/64,64,0,stream>>>(cg1,cg2,ct1,ct2,margin,counts,active,xpos,xmat,
   size,rbound,dataid,vert,vertadr,vertnum,graphadr,graph,N,P,ngeom,nmesh,nvert,ngraph,
   maxiter,tolerance,multiccd,raw,out_count,status,diag,frame0,framewritten);
 return (int)cudaGetLastError();
}
extern "C" int native_private_abi(int* sizes){
 sizes[0]=sizeof(real_t);sizes[1]=sizeof(float);sizes[2]=sizeof(int);
 sizes[3]=sizeof(mjContact);sizes[4]=offsetof(mjContact,dist);
 sizes[5]=offsetof(mjContact,pos);sizes[6]=offsetof(mjContact,frame);
 return 7;
}

// Host-only module identity. This prepares the registered production kernel
// before case execution; it reads no geom/contact/count buffers.
extern "C" int native_kernel_identity(unsigned long long* values, int* attrs) {
 const void* symbol = reinterpret_cast<const void*>(native_kernel);
 cudaFunction_t function = nullptr;
 cudaError_t error = cudaGetFuncBySymbol(&function, symbol);
 if (error != cudaSuccess) return (int)error;
 cudaFuncAttributes a;
 error = cudaFuncGetAttributes(&a, symbol);
 if (error != cudaSuccess) return (int)error;
 int runtime_version = 0, driver_version = 0;
 error = cudaRuntimeGetVersion(&runtime_version);
 if (error != cudaSuccess) return (int)error;
 error = cudaDriverGetVersion(&driver_version);
 if (error != cudaSuccess) return (int)error;
 values[0] = reinterpret_cast<unsigned long long>(symbol);
 values[1] = reinterpret_cast<unsigned long long>(function);
 values[2] = a.localSizeBytes;
 values[3] = a.sharedSizeBytes;
 values[4] = a.constSizeBytes;
 attrs[0] = a.binaryVersion; attrs[1] = a.ptxVersion;
 attrs[2] = a.numRegs; attrs[3] = a.maxThreadsPerBlock;
 attrs[4] = a.cacheModeCA; attrs[5] = runtime_version;
 attrs[6] = driver_version; attrs[7] = sizeof(cudaFuncAttributes);
 return 0;
}
