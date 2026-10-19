// Offline standalone test: independent body pairs, ordered requests, explicit
// suppression and the original capacity/error fallback, with two worlds.
#include "common/vector.cuh"
#include "collision/collision_tree.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

constexpr int W=2,B=4,G=32,Nodes=60,S=8,R=1024,V=4096,K=8192,L=1024,Stack=30;
struct Fixture {
  int stage[W*S*6],count[W],active[W],body_adr[B],body_num[B],geom_adr[B],geom_num[B];
  int child[Nodes*2],geomid[Nodes],gtype[G],pair1[3],pair2[3];
  double xpos[W*B*3],xmat[W*B*9],gpos[W*G*3],gmat[W*G*9];
  double bvh[Nodes*6],geom[G*6],bound[G],product[W*36],offset[W*12];
  int stack1[W*Stack],stack2[W*Stack],req[W*R*4],visit[W*V*12];
  int stack_tape[W*K*5],leaf[W*L*3],active_nodes[W*Nodes];
  int rc[W],vc[W],sc[W],lc[W],status[W],first[W],parallel[W];
};
static void check(cudaError_t error){
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
#define TREE_ARGS w,f->stage,f->count,f->active,f->xpos,f->xmat,f->gpos,f->gmat, \
 f->body_adr,f->body_num,f->geom_adr,f->geom_num,f->child,f->geomid,f->bvh,f->geom, \
 f->bound,f->gtype,f->pair1,f->pair2,f->stack1,f->stack2,f->product,f->offset, \
 f->req,f->visit,f->stack_tape,f->leaf,f->active_nodes,f->rc,f->vc,f->sc,f->lc, \
 f->status,f->first,S,req_capacity,visit_capacity,stack_capacity,leaf_capacity,B,G,Nodes,Stack,3
__global__ void traverse(Fixture* f,bool candidate,int req_capacity,int visit_capacity,
                        int stack_capacity,int leaf_capacity){
  const int w=blockIdx.y;
  if(candidate){
    const bool admitted=collision_cuda::tree_parallel_live(TREE_ARGS);
    if(threadIdx.x==0)f->parallel[w]=admitted;
    if(!admitted)collision_cuda::cf_tree_impl<false>(TREE_ARGS);
  }else collision_cuda::cf_tree_impl<false>(TREE_ARGS);
}
#undef TREE_ARGS
static void initialize(Fixture* f){
  std::memset(f,0,sizeof(*f));
  for(int b=0;b<B;++b){
    f->body_adr[b]=15*b;f->body_num[b]=15;f->geom_adr[b]=8*b;f->geom_num[b]=8;
    for(int i=0;i<15;++i){
      const int n=15*b+i;
      f->child[2*n]=i<7?2*i+1:-1;f->child[2*n+1]=i<7?2*i+2:-1;
      f->geomid[n]=i<7?-1:8*b+i-7;
      for(int k=3;k<6;++k)f->bvh[n*6+k]=1;
    }
  }
  f->pair1[0]=f->pair1[1]=0;f->pair2[0]=f->pair2[1]=8;
  f->pair1[2]=16;f->pair2[2]=24;
  for(int g=0;g<G;++g){f->bound[g]=2;f->gtype[g]=6;for(int k=3;k<6;++k)f->geom[6*g+k]=.5;}
  const int entries[24]={0,1,1,0,3,1, 0,2,0,0,0,1, 1,3,1,0,3,2, 2,3,1,0,3,1};
  for(int w=0;w<W;++w){
    f->active[w]=1;f->count[w]=4;
    std::memcpy(f->stage+w*S*6,entries,sizeof(entries));
    for(int b=0;b<B;++b){
      f->xpos[(w*B+b)*3]=.03*b+.01*w;
      const double angle=.2*b+.1*w,c=std::cos(angle),s=std::sin(angle);
      double* mat=f->xmat+(w*B+b)*9;
      mat[0]=mat[4]=c;mat[1]=-s;mat[3]=s;mat[8]=1;
    }
    for(int g=0;g<G;++g){
      f->gpos[(w*G+g)*3]=.01*g+.03*w;
      const double angle=.07*g+.1*w,c=std::cos(angle),s=std::sin(angle);
      double* mat=f->gmat+(w*G+g)*9;
      mat[0]=mat[4]=c;mat[1]=-s;mat[3]=s;mat[8]=1;
    }
    for(int i=0;i<36;++i)f->product[w*36+i]=.013*i;
    for(int i=0;i<12;++i)f->offset[w*12+i]=-.007*i;
  }
}
int main(){
 try{
  Fixture *a,*b;check(cudaMallocManaged(&a,sizeof(Fixture)));check(cudaMallocManaged(&b,sizeof(Fixture)));
  int tests=0,admitted=0,fallbacks=0;
  for(int threads:{32,64,128})for(int mode=0;mode<16;++mode){
    initialize(a);int rc=R,vc=V,sc=K,lc=L;
    if(mode==1)a->count[0]=a->count[1]=1;
    if(mode==2)rc=3;
    if(mode==3)vc=2;
    if(mode==4)sc=5;
    if(mode==5)lc=3;
    if(mode==6)a->stage[5]=0;
    if(mode==7)a->active[0]=a->active[1]=0;
    if(mode==8)a->stage[0]=B;
    if(mode==9)a->body_num[0]=0;
    if(mode==10)vc=100;
    if(mode==11){a->count[0]=a->count[1]=8;for(int w=0;w<W;++w)std::memcpy(a->stage+(w*S+4)*6,a->stage+w*S*6,24*sizeof(int));}
    // A fully infinite root postpones coefficient initialization to a
    // descendant: exercise that path as well as ordinary root initialization.
    if(mode==12)for(int body=0;body<B;++body)for(int k=3;k<6;++k)a->bvh[15*body*6+k]=1e12;
    if(mode==13)for(int body=0;body<B;++body)for(int k=3;k<5;++k)a->bvh[15*body*6+k]=1e12;
    if(mode==14)for(int w=0;w<W;++w){a->xpos[(w*B+1)*3]+=100;for(int g=8;g<16;++g)a->gpos[(w*G+g)*3]+=100;}
    if(mode==15)for(int w=0;w<W;++w)for(int body=0;body<B;++body){a->xpos[(w*B+body)*3]+=10*body;for(int g=8*body;g<8*(body+1);++g)a->gpos[(w*G+g)*3]+=10*body;}
    std::memcpy(b,a,sizeof(*a));
    traverse<<<dim3(1,W),threads>>>(a,false,rc,vc,sc,lc);
    traverse<<<dim3(1,W),threads>>>(b,true,rc,vc,sc,lc);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    for(int w=0;w<W;++w){
      if(a->rc[w]!=b->rc[w]||a->vc[w]!=b->vc[w]||a->sc[w]!=b->sc[w]||a->lc[w]!=b->lc[w]||a->status[w]!=b->status[w]||a->first[w]!=b->first[w])
        throw std::runtime_error("ordered counts or first error differ");
      if(std::memcmp(a->req+w*rc*4,b->req+w*rc*4,a->rc[w]*4*sizeof(int)))
        throw std::runtime_error("request order, geom or prefix differ");
      for(int i=0;i<36;++i)if(std::fabs(a->product[w*36+i]-b->product[w*36+i])>1e-12)
        throw std::runtime_error("retained OBB product differs");
      for(int i=0;i<12;++i)if(std::fabs(a->offset[w*12+i]-b->offset[w*12+i])>1e-12)
        throw std::runtime_error("retained OBB offset differs");
      if(mode==0 && !b->parallel[w])throw std::runtime_error("normal case missed parallel path");
      admitted+=b->parallel[w];fallbacks+=!b->parallel[w];
    }
    ++tests;
  }
  check(cudaFree(a));check(cudaFree(b));
  std::printf("PASS cases=%d parallel_worlds=%d fallback_worlds=%d\n",tests,admitted,fallbacks);
 }catch(const std::exception& error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
