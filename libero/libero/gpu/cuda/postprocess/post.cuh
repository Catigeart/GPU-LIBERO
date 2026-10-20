#include "precision.cuh"
#pragma once
namespace libero_post {
__device__ inline bool finite(const real_t* a,int n){for(int i=0;i<n;i++)if(!isfinite(a[i]))return false;return true;}
__device__ inline real_t dof(const real_t* a,const real_t* q,int adr,int n,int k){if(n==1)return a[6*adr+k]*q[adr];real_t v=0;for(int i=0;i<n;i++)if(q[adr+i]!=0)v+=a[6*(adr+i)+k]*q[adr+i];return v;}
__device__ inline void inert(const real_t* i,const real_t* v,real_t* f){
 f[0]=i[0]*v[0]+i[3]*v[1]+i[4]*v[2]-i[8]*v[4]+i[7]*v[5];
 f[1]=i[3]*v[0]+i[1]*v[1]+i[5]*v[2]+i[8]*v[3]-i[6]*v[5];
 f[2]=i[4]*v[0]+i[5]*v[1]+i[2]*v[2]-i[7]*v[3]+i[6]*v[4];
 f[3]=i[8]*v[1]-i[7]*v[2]+i[9]*v[3];f[4]=i[6]*v[2]-i[8]*v[0]+i[9]*v[4];f[5]=i[7]*v[0]-i[6]*v[1]+i[9]*v[5];}
__device__ inline void wrench(real_t* out,const real_t* f,const real_t* t,const real_t* com,const real_t* pos,real_t sign){real_t d[3],c[3];for(int k=0;k<3;k++)d[k]=com[k]-pos[k];libero_cuda::cross3(d,f,c);for(int k=0;k<3;k++){out[k]+=sign*(t[k]-c[k]);out[k+3]+=sign*f[k];}}
__device__ inline void gate(int w,const int* active,const int* first,const int* dv,const int* nv,const int* ne,const int64_t* nw,const int64_t* na,const int64_t* expected,const int64_t* state,const int64_t* consumed,const int64_t* forward,const int64_t* cf,int* g){g[w]=(active[w]==1&&first[w]==0&&dv[w]==1&&nv[w]==1&&ne[w]==0&&expected[w]>0&&nw[w]==expected[w]&&na[w]==expected[w]&&state[w]==consumed[w]&&forward[w]==cf[w]);}
// One world owns its tree: ordered recurrences and ordered contact accumulation.
__device__ inline void rne(int w,const int* active,const real_t* qv,const real_t* qa,const real_t* ci,const real_t* cv,const real_t* cd,const real_t* cdd,const real_t* xi,const real_t* com,const real_t* sp,const real_t* sm,const real_t* xf,const real_t* ef,const int* nc,const int* ne,const int* neq,const int* dim,const int* adr,const int* g1,const int* g2,const real_t* frame,const real_t* pos,const int* parent,const int* da,const int* dn,const int* root,const int* gb,const real_t* grav,int cc,int ec,real_t* acc,real_t* ext,real_t* intr,real_t* decoded,int* status,int b,int v,int ns,int ng){
 status[w]=0;qv+=w*v;qa+=w*v;ci+=w*b*10;cv+=w*b*6;cd+=w*v*6;cdd+=w*v*6;xi+=w*b*3;com+=w*b*3;sp+=w*ns*3;sm+=w*ns*9;xf+=w*b*6;ef+=w*ec;dim+=w*cc;adr+=w*cc;g1+=w*cc;g2+=w*cc;frame+=w*cc*9;pos+=w*cc*3;acc+=w*b*6;ext+=w*b*6;intr+=w*b*6;decoded+=w*cc*6;
 for(int k=0;k<b*6;k++)acc[k]=ext[k]=intr[k]=0;for(int k=0;k<cc*6;k++)decoded[k]=0;
 if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}int n=nc[w],e=ne[w];if(n<0||n>cc||e<0||e>ec){status[w]=2;return;}if(neq[w]!=0){status[w]=6;return;}
 if(!finite(qv,v)||!finite(qa,v)||!finite(cd,6*v)||!finite(cdd,6*v)||!finite(ci,10*b)||!finite(cv,6*b)||!finite(xf,6*b)||!finite(xi,3*b)||!finite(com,3*b)||!finite(ef,e)||!finite(sp,3*ns)||!finite(sm,9*ns)){status[w]=4;return;}
 for(int j=0;j<n;j++)if(adr[j]>=0){if(dim[j]<1||dim[j]>6||adr[j]>e-dim[j]||g1[j]<0||g1[j]>=ng||g2[j]<0||g2[j]>=ng){status[w]=3;return;}if(!finite(frame+9*j,9)||!finite(pos+3*j,3)){status[w]=4;return;}}
 for(int k=0;k<3;k++)acc[3+k]=-grav[k];
 for(int j=1;j<b;j++){bool nz=false;for(int k=0;k<6;k++)nz|=xf[6*j+k]!=0;if(nz)wrench(ext+6*j,xf+6*j,xf+6*j+3,com+3*root[j],xi+3*j,1);}
 for(int j=0;j<n;j++)if(adr[j]>=0){real_t* d=decoded+6*j;for(int k=0;k<dim[j];k++)d[k]=ef[adr[j]+k];real_t f[3],t[3];for(int k=0;k<3;k++){f[k]=frame[9*j+k]*d[0]+frame[9*j+3+k]*d[1]+frame[9*j+6+k]*d[2];t[k]=frame[9*j+k]*d[3]+frame[9*j+3+k]*d[4]+frame[9*j+6+k]*d[5];}for(int side=0;side<2;side++){int body=gb[side?g2[j]:g1[j]];if(body>0)wrench(ext+6*body,f,t,com+3*root[body],pos+3*j,side?1:-1);}}
 for(int j=1;j<b;j++){real_t* a=acc+6*j;const real_t* c=cv+6*j;for(int k=0;k<6;k++)a[k]=acc[6*parent[j]+k]+dof(cdd,qv,da[j],dn[j],k);for(int k=0;k<6;k++)a[k]+=dof(cd,qa,da[j],dn[j],k);real_t f[6],r[6],x[6];inert(ci+10*j,a,f);inert(ci+10*j,c,r);
 x[0]=-c[2]*r[1]+c[1]*r[2];x[1]=c[2]*r[0]-c[0]*r[2];x[2]=-c[1]*r[0]+c[0]*r[1];x[3]=-c[2]*r[4]+c[1]*r[5];x[4]=c[2]*r[3]-c[0]*r[5];x[5]=-c[1]*r[3]+c[0]*r[4];x[0]+=-c[5]*r[4]+c[4]*r[5];x[1]+=c[5]*r[3]-c[3]*r[5];x[2]+=-c[4]*r[3]+c[3]*r[4];for(int k=0;k<6;k++)intr[6*j+k]=(f[k]+x[k])-ext[6*j+k];}
 for(int j=b-1;j>0;j--)for(int k=0;k<6;k++)intr[6*parent[j]+k]+=intr[6*j+k];
 if(!finite(acc,6*b)||!finite(ext,6*b)||!finite(intr,6*b)){status[w]=5;for(int k=0;k<6*b;k++)acc[k]=ext[k]=intr[k]=0;}
}
__device__ inline void sensor(int w,const real_t* fr,const real_t* com,const real_t* pos,const real_t* mat,const int* active,const int* type,const int* site,const int* adr,const int* sb,const int* br,real_t* result,int* status,int b,int ns){
 status[w]=0;result+=6*w;for(int k=0;k<6;k++)result[k]=0;if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}fr+=w*b*6;com+=w*b*3;pos+=w*ns*3;mat+=w*ns*9;
 if(!finite(fr,6*b)||!finite(com,3*b)||!finite(pos,3*ns)||!finite(mat,9*ns)){status[w]=2;return;}
 for(int s=0;s<2;s++){int st=site[s],body=sb[st],root=br[body];real_t d[3],c[3],t[3];const real_t* f=fr+6*body+3;for(int k=0;k<3;k++)d[k]=pos[3*st+k]-com[3*root+k];libero_cuda::cross3(d,f,c);for(int k=0;k<3;k++)t[k]=fr[6*body+k]-c[k];const real_t* v=type[s]==4?f:t;for(int k=0;k<3;k++)result[adr[s]+k]=mat[9*st+k]*v[0]+mat[9*st+3+k]*v[1]+mat[9*st+6+k]*v[2];}
}
__device__ inline void finish(int w,const int* active,int* status,const int* ss,real_t* r){if(active[w]!=1||status[w]!=0)return;if(ss[w]!=0){status[w]=4;return;}if(!finite(r+6*w,6)){status[w]=5;for(int k=0;k<6;k++)r[6*w+k]=0;}}
__device__ inline void commit(int w,const int* gate,const int* status,const int* ss,int* first,const int64_t* epoch,const real_t* ext,const real_t* intr,const real_t* sensor,real_t* pe,real_t* pi,real_t* ps,int* valid,int64_t* written,int bw,int sw){valid[w]=0;if(gate[w]!=1||first[w]!=0)return;if(status[w]!=0||ss[w]!=0){first[w]=215;return;}for(int k=0;k<bw;k++){pe[w*bw+k]=ext[w*bw+k];pi[w*bw+k]=intr[w*bw+k];}for(int k=0;k<sw;k++)ps[w*sw+k]=sensor[w*sw+k];written[w]=epoch[w];valid[w]=1;}
}
