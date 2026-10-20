#include "precision.cuh"
#pragma once
#include "../common/packed_mass.cuh"
// Sparse MuJoCo tree LDL is not a dense LAPACK factorization.
namespace cf_euler {
__device__ void prepare(int w,const int* active,const real_t* qpos,const real_t* qvel,const real_t* time,const real_t* qm,const real_t* smooth,const real_t* constraint,const real_t* acc,const real_t* damping,const int* madr,real_t* mq,real_t* rhs,int* count,int* ma,int* damp,int* status,real_t dt,int nq,int nv,int nm){
 status[w]=0;damp[w]=0;ma[w]=0;count[w]=1;mq+=w*nm;rhs+=w*4*nv;
 for(int i=0;i<nm;i++)mq[i]=0;for(int i=0;i<4*nv;i++)rhs[i]=0;
 if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}
 if(!isfinite(time[w])){status[w]=2;return;}
 for(int i=0;i<nq;i++)if(!isfinite(qpos[w*nq+i])){status[w]=2;return;}
 for(int i=0;i<nv;i++){
  if(!isfinite(qvel[w*nv+i])||!isfinite(acc[w*nv+i])||!isfinite(smooth[w*nv+i])||!isfinite(constraint[w*nv+i])){status[w]=2;return;}
  if(damping[i]>0)damp[w]=1;
 }
 if(!damp[w])return;
 for(int i=0;i<nm;i++){if(!isfinite(qm[w*nm+i])){status[w]=2;return;}mq[i]=qm[w*nm+i];}
 for(int i=0;i<nv;i++){mq[madr[i]]+=dt*damping[i];rhs[i]=smooth[w*nv+i]+constraint[w*nv+i];}ma[w]=1;
}
// All block threads enter; warp 0 performs ordered validation and copying.
// The first invalid qM element preserves the scalar partial-copy boundary.
__device__ void prepare_block(int w,const int* active,const real_t* qpos,const real_t* qvel,const real_t* time,const real_t* qm,const real_t* smooth,const real_t* constraint,const real_t* acc,const real_t* damping,const int* madr,real_t* mq,real_t* rhs,int* count,int* ma,int* damp,int* status,real_t dt,int nq,int nv,int nm){
 const int t=threadIdx.x;
 if(t==0){status[w]=0;damp[w]=0;ma[w]=0;count[w]=1;}
 libero_vec::fill_block(mq+w*nm,nm,REAL_C(0.0));
 libero_vec::fill_block(rhs+w*4*nv,4*nv,REAL_C(0.0));
 __syncthreads();
 if(t<32){
  if(active[w]!=1){if(t==0&&active[w]!=0)status[w]=1;}
  else {
   bool bad=!isfinite(time[w]);
   bad|=libero_vec::any_nonfinite_lane(qpos+w*nq,nq,t,32);
   bad=libero_warp::any(bad);
   if(bad){if(t==0)status[w]=2;}
   else {
    int first=libero_warp::no_index;
    for(int i=2*t;i<nv;i+=64){
     if(i+1<nv){
      const real2_t v=libero_vec::load_pair(qvel+w*nv+i),a=libero_vec::load_pair(acc+w*nv+i);
      const real2_t sm=libero_vec::load_pair(smooth+w*nv+i),co=libero_vec::load_pair(constraint+w*nv+i);
      if(!isfinite(v.x)||!isfinite(a.x)||!isfinite(sm.x)||!isfinite(co.x))first=min(first,i);
      if(!isfinite(v.y)||!isfinite(a.y)||!isfinite(sm.y)||!isfinite(co.y))first=min(first,i+1);
     }else if(!isfinite(qvel[w*nv+i])||!isfinite(acc[w*nv+i])||!isfinite(smooth[w*nv+i])||!isfinite(constraint[w*nv+i]))first=min(first,i);
    }
    first=libero_warp::min(first);
    bool positive=false;
    for(int i=t;i<nv&&i<first;i+=32)positive|=damping[i]>0;
    positive=libero_warp::any(positive);
    if(t==0){damp[w]=positive; if(first!=libero_warp::no_index)status[w]=2;}
    if(first==libero_warp::no_index&&positive){
     int bad_qm=libero_warp::no_index;
     for(int i=2*t;i<nm;i+=64){
      if(i+1<nm){const real2_t q=libero_vec::load_pair(qm+w*nm+i);if(!isfinite(q.x))bad_qm=min(bad_qm,i);if(!isfinite(q.y))bad_qm=min(bad_qm,i+1);}
      else if(!isfinite(qm[w*nm+i]))bad_qm=min(bad_qm,i);
     }
     bad_qm=libero_warp::min(bad_qm);
     libero_vec::copy_lane(qm+w*nm,mq+w*nm,min(nm,bad_qm),t,32);
     __syncwarp(libero_warp::mask);
     if(bad_qm!=libero_warp::no_index){if(t==0)status[w]=2;}
     else {
      // Keep the original diagonal-add order if descriptors alias.
      if(t==0){for(int i=0;i<nv;i++)mq[w*nm+madr[i]]+=dt*damping[i];ma[w]=1;}
      for(int i=2*t;i<nv;i+=64){
       if(i+1<nv){const real2_t sm=libero_vec::load_pair(smooth+w*nv+i),co=libero_vec::load_pair(constraint+w*nv+i);libero_vec::store_pair(rhs+w*4*nv+i,sm.x+co.x,sm.y+co.y);}
       else rhs[w*4*nv+i]=smooth[w*nv+i]+constraint[w*nv+i];
      }
     }
    }
   }
  }
 }
 __syncthreads();
}
__device__ void mass_clear(real_t* ld,real_t* inv,real_t* si,real_t* solution,int* pivot,int* warnings,int* last,int nv,int nm,bool preserve){
 libero_vec::fill_serial(ld,nm,REAL_C(0.0));libero_vec::fill_serial(inv,nv,REAL_C(0.0));libero_vec::fill_serial(si,nv,REAL_C(0.0));
 if(!preserve)libero_vec::fill_serial(pivot,nv,0);
 libero_vec::fill_serial(solution,4*nv,REAL_C(0.0));if(!preserve){*warnings=0;*last=-1;}
}
__device__ void mass(int w,const real_t* qm,const real_t* rhs,const int* count,const int* active,const int* madr,const int* parent,real_t* ld,real_t* inv,real_t* si,real_t* solution,int* pivot,int* status,int* warnings,int* last,int nv,int nm){
 qm+=w*nm;rhs+=w*4*nv;ld+=w*nm;inv+=w*nv;si+=w*nv;solution+=w*4*nv;pivot+=w*nv;warnings+=w;last+=w;
 status[w]=0;mass_clear(ld,inv,si,solution,pivot,warnings,last,nv,nm,false);
 if(active[w]==0)return;if(active[w]!=1){status[w]=1;return;}int n=count[w];if(n<1||n>4){status[w]=2;return;}
 if(libero_vec::any_nonfinite_lane(qm,nm,0,1)){status[w]=3;return;}
 // Admission deliberately validates all four RHS slots, including unused rows.
 if(libero_vec::any_nonfinite_lane(rhs,4*nv,0,1)){status[w]=3;return;}
 bool bad=false;libero_vec::copy_serial(qm,ld,nm);
 for(int k=nv-1;k>=0;k--){int kk=madr[k];if(ld[kk]<REAL_C(1e-15)){pivot[k]=1;(*warnings)++;*last=k;ld[kk]=REAL_C(1e-15);}int ki=kk+1;
  for(int i=parent[k];i>=0;i=parent[i],ki++){real_t tmp=ld[ki]/ld[kk];if(!isfinite(tmp))bad=true;int cnt=(i<nv-1?madr[i+1]:nm)-madr[i];
   for(int a=0;a<cnt;a++){ld[madr[i]+a]+=ld[ki+a]*(-tmp);if(!isfinite(ld[madr[i]+a]))bad=true;}ld[ki]=tmp;}
 }
 for(int i=0;i<nv;i++){inv[i]=REAL_C(1.0)/ld[madr[i]];si[i]=REAL_C(1.0)/sqrt(ld[madr[i]]);if(!isfinite(inv[i])||!isfinite(si[i]))bad=true;}
 libero_vec::copy_serial(rhs,solution,n*nv);
 for(int i=nv-1;i>=0;i--){int a=madr[i]+1;for(int j=parent[i];j>=0;j=parent[j],a++)for(int r=0;r<n;r++){
  real_t tmp=solution[r*nv+i];if(tmp!=0){solution[r*nv+j]-=ld[a]*tmp;if(!isfinite(solution[r*nv+j]))bad=true;}}}
 for(int i=0;i<nv;i++)for(int r=0;r<n;r++){solution[r*nv+i]*=inv[i];if(!isfinite(solution[r*nv+i]))bad=true;}
 for(int i=0;i<nv;i++){int a=madr[i]+1;for(int j=parent[i];j>=0;j=parent[j],a++)for(int r=0;r<n;r++){
  solution[r*nv+i]-=ld[a]*solution[r*nv+j];if(!isfinite(solution[r*nv+i]))bad=true;}}
 if(bad){mass_clear(ld,inv,si,solution,pivot,warnings,last,nv,nm,true);status[w]=4;}else if(*warnings)status[w]=5;
}
// Collective sparse factor/solve. The scalar mass() remains the fallback
// reference for leader-only compatibility callers; this path owns the same
// warning events, validates all four RHS slots, and uses the same clear gate.
__device__ void mass_block(int w,const real_t* qm,const real_t* rhs,const int* count,
 const int* active,const int* madr,const int* parent,real_t* ld,real_t* inv,
 real_t* si,real_t* solution,int* pivot,int* status,int* warnings,int* last,int nv,int nm){
 qm+=w*nm;rhs+=w*4*nv;ld+=w*nm;inv+=w*nv;si+=w*nv;solution+=w*4*nv;
 pivot+=w*nv;warnings+=w;last+=w;
 if(threadIdx.x==0){status[w]=0;*warnings=0;*last=-1;}
 libero_vec::fill_block(ld,nm,REAL_C(0.0));libero_vec::fill_block(inv,nv,REAL_C(0.0));
 libero_vec::fill_block(si,nv,REAL_C(0.0));libero_vec::fill_block(solution,4*nv,REAL_C(0.0));
 libero_vec::fill_block(pivot,nv,0);__syncthreads();
 if(active[w]==0)return;
 if(active[w]!=1){if(threadIdx.x==0)status[w]=1;__syncthreads();return;}
 const int n=count[w];
 if(n<1||n>4){if(threadIdx.x==0)status[w]=2;__syncthreads();return;}
 const bool invalid=libero_vec::any_nonfinite_lane(qm,nm,threadIdx.x,blockDim.x)|
     libero_vec::any_nonfinite_lane(rhs,4*nv,threadIdx.x,blockDim.x);
 if(__syncthreads_or(invalid)){if(threadIdx.x==0)status[w]=3;__syncthreads();return;}
 libero_packed_mass::ForestPlan forest;
 const bool factor_ok=libero_packed_mass::factor_block(qm,ld,inv,si,pivot,*warnings,*last,nv,nm,madr,parent,forest);
 const bool solve_ok=libero_packed_mass::solve_rhs_block(rhs,solution,ld,inv,n,nv,madr,parent,forest);
 if(threadIdx.x==0){
  if(!factor_ok||!solve_ok){mass_clear(ld,inv,si,solution,pivot,warnings,last,nv,nm,true);status[w]=4;}
  else if(*warnings)status[w]=5;
 }
 __syncthreads();
}
__device__ void replay(int w,const int* active,const int* damp,const int* status,const real_t* rhs,const real_t* ld,const int* madr,const int* parent,const int* simple,real_t* work,int* skip,int nv,int nm){
 work+=w*nv;skip+=w*nv;rhs+=w*4*nv;ld+=w*nm;for(int i=0;i<nv;i++){work[i]=0;skip[i]=0;}
 if(active[w]!=1||!damp[w]||(status[w]!=0&&status[w]!=5))return;
 for(int i=0;i<nv;i++)work[i]=rhs[i];for(int i=nv-1;i>=0;i--){if(simple[i])continue;if(work[i]==0)skip[i]=1;else {int a=madr[i]+1;for(int j=parent[i];j>=0;j=parent[j],a++)work[j]-=ld[a]*work[i];}}
}
// Fixed Horner coefficients are the same binary64 Taylor coefficients as the reference.
__device__ real_t trig(real_t z,bool sine){
 const real_t sc[23]={REAL_C(1.),-REAL_C(.16666666666666666),REAL_C(.008333333333333333),-REAL_C(.0001984126984126984),REAL_C(2.7557319223985893e-6),-REAL_C(2.505210838544172e-8),REAL_C(1.6059043836821613e-10),-REAL_C(7.647163731819816e-13),REAL_C(2.8114572543455206e-15),-REAL_C(8.22063524662433e-18),REAL_C(1.9572941063391263e-20),-REAL_C(3.868170170630684e-23),REAL_C(6.446950284384474e-26),-REAL_C(9.183689863795546e-29),REAL_C(1.1309962886447716e-31),-REAL_C(1.216125041553518e-34),REAL_C(1.151633562077195e-37),-REAL_C(9.67759295863189e-41),REAL_C(7.265460179153071e-44),-REAL_C(4.902469756513544e-47),REAL_C(2.9893108271424046e-50),-REAL_C(1.6552108677421951e-53),REAL_C(8.359650847182804e-57)};
 const real_t cc[23]={REAL_C(1.),-REAL_C(.5),REAL_C(.041666666666666664),-REAL_C(.001388888888888889),REAL_C(2.48015873015873e-5),-REAL_C(2.755731922398589e-7),REAL_C(2.08767569878681e-9),-REAL_C(1.1470745597729725e-11),REAL_C(4.779477332387385e-14),-REAL_C(1.5619206968586225e-16),REAL_C(4.110317623312165e-19),-REAL_C(8.896791392450574e-22),REAL_C(1.6117375710961184e-24),-REAL_C(2.4795962632247976e-27),REAL_C(3.279889237069838e-30),-REAL_C(3.7699876288159054e-33),REAL_C(3.8003907548547434e-36),-REAL_C(3.387157535521162e-39),REAL_C(2.6882202662866363e-42),-REAL_C(1.911963205040282e-45),REAL_C(1.2256174391283858e-48),-REAL_C(7.117406731291439e-52),REAL_C(3.7618428812322616e-55)};
 const real_t* c=sine?sc:cc;real_t p=c[22],zz=z*z;for(int i=21;i>=0;i--)p=p*zz+c[i];return sine?z*p:p;
}
__device__ void integrate(int w,const int* active,const real_t* qp,const real_t* qv,const real_t* time,const real_t* acc,const real_t* solution,const int* ms,const int* damp,const int* jt,const int* padr,const int* vadr,real_t* sp,real_t* sv,real_t* used,real_t* st,real_t* axis,real_t* angle,real_t* increment,real_t* pre,int* af,int* za,int* qf,int* status,real_t dt,int nq,int nv,int nj,int nq4,bool check_final=true){
 if(active[w]!=1||status[w])return;if(damp[w]&&ms[w]!=0&&ms[w]!=5){status[w]=10+ms[w];return;}
 qp+=w*nq;qv+=w*nv;acc+=w*nv;solution+=w*4*nv;sp+=w*nq;sv+=w*nv;used+=w*nv;axis+=w*nq4*3;angle+=w*nq4;increment+=w*nq4*4;pre+=w*nq4*4;af+=w*nq4;za+=w*nq4;qf+=w*nq4;
 for(int i=0;i<nv;i++){used[i]=damp[w]?solution[i]:acc[i];sv[i]=qv[i]+dt*used[i];if(!isfinite(sv[i])){status[w]=3;return;}}
 for(int i=0;i<nq;i++)sp[i]=qp[i];int qi=0;
 for(int j=0;j<nj;j++){int pa=padr[j],va=vadr[j],kind=jt[j];if(kind==0){for(int k=0;k<3;k++)sp[pa+k]+=dt*sv[va+k];pa+=3;va+=3;}else if(kind==2||kind==3)sp[pa]+=dt*sv[va];
  if(kind!=0&&kind!=1)continue;
  real_t x=sv[va],y=sv[va+1],z=sv[va+2],xx=x*x,yy=y*y,zz=z*z,sxy=xx+yy,sum=sxy+zz;
  if(!isfinite(xx)||!isfinite(yy)||!isfinite(zz)||!isfinite(sxy)||!isfinite(sum)){status[w]=3;return;}real_t length=sqrt(sum);if(!isfinite(length)){status[w]=3;return;}
  af[qi]=length<REAL_C(1e-15);if(af[qi]){x=1;y=z=0;}else{real_t inv=1/length;x*=inv;y*=inv;z*=inv;}real_t a=dt*length;
  if(!isfinite(a)||!isfinite(x)||!isfinite(y)||!isfinite(z)){status[w]=3;return;}
  axis[qi*3]=x;axis[qi*3+1]=y;axis[qi*3+2]=z;angle[qi]=a;za[qi]=a==0;real_t c=1,s=0;
  if(a!=0){real_t h=REAL_C(.5)*a;if(!isfinite(h)){status[w]=3;return;}s=fabs(h)<=4?trig(h,true):sin(h);c=fabs(h)<=4?trig(h,false):cos(h);}
  real_t b[4]={c,x*s,y*s,z*s};if(a==0)b[1]=b[2]=b[3]=0;
  for(int k=0;k<4;k++){if(!isfinite(b[k])){status[w]=3;return;}increment[qi*4+k]=b[k];}
  real_t a0=sp[pa],a1=sp[pa+1],a2=sp[pa+2],a3=sp[pa+3];real_t t[4]={a0*b[0]-a1*b[1]-a2*b[2]-a3*b[3],a0*b[1]+a1*b[0]+a2*b[3]-a3*b[2],a0*b[2]-a1*b[3]+a2*b[0]+a3*b[1],a0*b[3]+a1*b[2]-a2*b[1]+a3*b[0]};
  for(int k=0;k<4;k++){if(!isfinite(t[k])){status[w]=3;return;}pre[qi*4+k]=t[k];}
  real_t sq[4];for(int k=0;k<4;k++){sq[k]=t[k]*t[k];if(!isfinite(sq[k])){status[w]=3;return;}}real_t s01=sq[0]+sq[1],s012=s01+sq[2],ss=s012+sq[3];
  if(!isfinite(s01)||!isfinite(s012)||!isfinite(ss)){status[w]=3;return;}real_t qlen=sqrt(ss);if(!isfinite(qlen)){status[w]=3;return;}qf[qi]=qlen<REAL_C(1e-15);
  if(qf[qi]){t[0]=1;t[1]=t[2]=t[3]=0;}else {real_t inv=1/qlen;for(int k=0;k<4;k++)t[k]*=inv;}
  for(int k=0;k<4;k++){if(!isfinite(t[k])){status[w]=3;return;}sp[pa+k]=t[k];}qi++;
 }
 st[w]=time[w]+dt;if(!isfinite(st[w])){status[w]=3;return;}if(check_final)for(int i=0;i<nq;i++)if(!isfinite(sp[i])){status[w]=3;return;}
}
// A pair is written only after its two ordered checks pass. On an invalid
// second element the first scalar store preserves the diagnostic prefix.
__device__ inline bool store_finite4(real_t* dst,const real_t* value){
 for(int k=0;k<4;k+=2){
  if(!isfinite(value[k]))return false;
  if(!isfinite(value[k+1])){dst[k]=value[k];return false;}
  libero_vec::store_pair(dst+k,value[k],value[k+1]);
 }
 return true;
}
// One independently addressed joint; arithmetic and quaternion order are unchanged.
__device__ inline bool integrate_joint(int j,int qi,const int* jt,const int* padr,const int* vadr,
 real_t* sp,const real_t* sv,real_t* axis,real_t* angle,real_t* increment,real_t* pre,
 int* af,int* za,int* qf,real_t dt){
int pa=padr[j],va=vadr[j],kind=jt[j];if(kind==0){
 const real2_t p=libero_vec::load_pair(sp+pa),v=libero_vec::load_pair(sv+va);
 libero_vec::store_pair(sp+pa,p.x+dt*v.x,p.y+dt*v.y);sp[pa+2]+=dt*sv[va+2];pa+=3;va+=3;
}else if(kind==2||kind==3)sp[pa]+=dt*sv[va];
  if(kind!=0&&kind!=1)return true;
  const real2_t velocity=libero_vec::load_pair(sv+va);
  real_t x=velocity.x,y=velocity.y,z=sv[va+2],xx=x*x,yy=y*y,zz=z*z,sxy=xx+yy,sum=sxy+zz;
  if(!isfinite(xx)||!isfinite(yy)||!isfinite(zz)||!isfinite(sxy)||!isfinite(sum)){return false;}real_t length=sqrt(sum);if(!isfinite(length)){return false;}
  af[qi]=length<REAL_C(1e-15);if(af[qi]){x=1;y=z=0;}else{real_t inv=1/length;x*=inv;y*=inv;z*=inv;}real_t a=dt*length;
  if(!isfinite(a)||!isfinite(x)||!isfinite(y)||!isfinite(z)){return false;}
  libero_vec::store_pair(axis+qi*3,x,y);axis[qi*3+2]=z;angle[qi]=a;za[qi]=a==0;real_t c=1,s=0;
  if(a!=0){real_t h=REAL_C(.5)*a;if(!isfinite(h)){return false;}s=fabs(h)<=4?trig(h,true):sin(h);c=fabs(h)<=4?trig(h,false):cos(h);}
  real_t b[4]={c,x*s,y*s,z*s};if(a==0)b[1]=b[2]=b[3]=0;
  if(!store_finite4(increment+qi*4,b))return false;
  const real2_t p01=libero_vec::load_pair(sp+pa),p23=libero_vec::load_pair(sp+pa+2);
  real_t a0=p01.x,a1=p01.y,a2=p23.x,a3=p23.y;real_t t[4]={a0*b[0]-a1*b[1]-a2*b[2]-a3*b[3],a0*b[1]+a1*b[0]+a2*b[3]-a3*b[2],a0*b[2]-a1*b[3]+a2*b[0]+a3*b[1],a0*b[3]+a1*b[2]-a2*b[1]+a3*b[0]};
  if(!store_finite4(pre+qi*4,t))return false;
  real_t sq[4];for(int k=0;k<4;k++){sq[k]=t[k]*t[k];if(!isfinite(sq[k])){return false;}}real_t s01=sq[0]+sq[1],s012=s01+sq[2],ss=s012+sq[3];
  if(!isfinite(s01)||!isfinite(s012)||!isfinite(ss)){return false;}real_t qlen=sqrt(ss);if(!isfinite(qlen)){return false;}qf[qi]=qlen<REAL_C(1e-15);
  if(qf[qi]){t[0]=1;t[1]=t[2]=t[3]=0;}else {real_t inv=1/qlen;for(int k=0;k<4;k++)t[k]*=inv;}
  if(!store_finite4(sp+pa,t))return false;
 return true;
}
// All lanes enter. Private failures are cleared by the existing transaction commit.
__device__ __forceinline__ bool integration_bad(bool bad,int* scratch){
 const int local=libero_warp::any(bad);
 if(libero_warp::lane()==0)scratch[libero_warp::warp()]=local;
 __syncthreads();
 bool result=false;
 if(libero_warp::warp()==0)result=libero_warp::any(libero_warp::lane()<(int(blockDim.x)>>5)&&scratch[libero_warp::lane()]!=0);
 if(threadIdx.x==0)scratch[0]=result;
 __syncthreads();
 const bool out=scratch[0]!=0;
 __syncthreads();
 return out;
}
__device__ void integrate_block(int w,const int* active,const real_t* qp,const real_t* qv,const real_t* time,const real_t* acc,const real_t* solution,const int* ms,const int* damp,const int* jt,const int* padr,const int* vadr,real_t* sp,real_t* sv,real_t* used,real_t* st,real_t* axis,real_t* angle,real_t* increment,real_t* pre,int* af,int* za,int* qf,int* status,real_t dt,int nq,int nv,int nj,int nq4,bool check_final=true,const int* joint_quat=nullptr,const int* joint_type_offsets=nullptr,const int* joint_type_indices=nullptr){
 __shared__ int integrate_vote[32];
 if(active[w]!=1||status[w])return;
 if(damp[w]&&ms[w]!=0&&ms[w]!=5){if(threadIdx.x==0)status[w]=10+ms[w];return;}
 qp+=w*nq;qv+=w*nv;acc+=w*nv;solution+=w*4*nv;sp+=w*nq;sv+=w*nv;used+=w*nv;
 axis+=w*nq4*3;angle+=w*nq4;increment+=w*nq4*4;pre+=w*nq4*4;af+=w*nq4;za+=w*nq4;qf+=w*nq4;
 bool bad=false;
 const real_t* selected=damp[w]?solution:acc;
 for(int i=2*threadIdx.x;i<nv;i+=2*blockDim.x){
  if(i+1<nv){
   const real2_t a=libero_vec::load_pair(selected+i),v=libero_vec::load_pair(qv+i);
   const real_t x=v.x+dt*a.x,y=v.y+dt*a.y;
   libero_vec::store_pair(used+i,a.x,a.y);libero_vec::store_pair(sv+i,x,y);
   bad|=!isfinite(x)||!isfinite(y);
  }else{used[i]=selected[i];sv[i]=qv[i]+dt*used[i];bad|=!isfinite(sv[i]);}
 }
 if(integration_bad(bad,integrate_vote)){if(threadIdx.x==0)status[w]=3;return;}
 libero_vec::copy_block(qp,sp,nq);
 __syncthreads();
 bad=false;
 if(joint_quat){
  // The binding supplies this map only after proving disjoint qpos ranges.
  if(joint_type_offsets&&joint_type_indices){
   for(int kind=0;kind<4;kind++)
    for(int i=joint_type_offsets[kind]+threadIdx.x;i<joint_type_offsets[kind+1];i+=blockDim.x){
     const int j=joint_type_indices[i];
     bad|=!integrate_joint(j,joint_quat[j],jt,padr,vadr,sp,sv,axis,angle,increment,pre,af,za,qf,dt);
    }
  }else for(int j=threadIdx.x;j<nj;j+=blockDim.x)
   bad|=!integrate_joint(j,joint_quat[j],jt,padr,vadr,sp,sv,axis,angle,increment,pre,af,za,qf,dt);
 }else if(threadIdx.x==0){
  int qi=0;
  for(int j=0;j<nj;j++){
   if(!integrate_joint(j,qi,jt,padr,vadr,sp,sv,axis,angle,increment,pre,af,za,qf,dt)){bad=true;break;}
   qi+=jt[j]==0||jt[j]==1;
  }
 }
 if(integration_bad(bad,integrate_vote)){if(threadIdx.x==0)status[w]=3;return;}
 if(threadIdx.x==0){st[w]=time[w]+dt;if(!isfinite(st[w]))status[w]=3;}
 __syncthreads();
 if(check_final){
  bad=false;for(int i=threadIdx.x;i<nq;i+=blockDim.x)bad|=!isfinite(sp[i]);
  if(integration_bad(bad,integrate_vote)){if(threadIdx.x==0)status[w]=3;}
 }
 __syncthreads();
}
__device__ void commit(int w,const int* active,real_t* qp,real_t* qv,real_t* time,real_t* qh,real_t* qhi,real_t* ld,real_t* inv,real_t* mq,real_t* rhs,real_t* si,real_t* solution,int* pivot,int* wc,int* wl,const int* damp,real_t* sp,real_t* sv,real_t* used,real_t* st,real_t* axis,real_t* angle,real_t* increment,real_t* pre,int* af,int* za,int* qf,real_t* work,int* skip,const int* status,int nq,int nv,int nm,int nq4){
 qp+=w*nq;qv+=w*nv;qh+=w*nm;qhi+=w*nv;ld+=w*nm;inv+=w*nv;mq+=w*nm;rhs+=w*4*nv;si+=w*nv;solution+=w*4*nv;pivot+=w*nv;sp+=w*nq;sv+=w*nv;used+=w*nv;axis+=w*nq4*3;angle+=w*nq4;increment+=w*nq4*4;pre+=w*nq4*4;af+=w*nq4;za+=w*nq4;qf+=w*nq4;work+=w*nv;skip+=w*nv;
 if(active[w]!=1||status[w]){
  for(int i=0;i<nm;i++){mq[i]=0;ld[i]=0;}for(int i=0;i<nv;i++){inv[i]=si[i]=work[i]=sv[i]=used[i]=0;pivot[i]=skip[i]=0;}for(int i=0;i<4*nv;i++)rhs[i]=solution[i]=0;wc[w]=0;wl[w]=-1;
  for(int i=0;i<nq;i++)sp[i]=0;st[w]=0;for(int i=0;i<nq4;i++){angle[i]=0;af[i]=za[i]=qf[i]=0;}for(int i=0;i<3*nq4;i++)axis[i]=0;for(int i=0;i<4*nq4;i++)increment[i]=pre[i]=0;return;
 }
 for(int i=0;i<nq;i++)qp[i]=sp[i];for(int i=0;i<nv;i++)qv[i]=sv[i];time[w]=st[w];if(damp[w]){for(int i=0;i<nm;i++)qh[i]=ld[i];for(int i=0;i<nv;i++)qhi[i]=inv[i];}
}
// Cooperative commit retains the scalar helper for leader-only callers.
__device__ void commit_block(int w,const int* active,real_t* qp,real_t* qv,real_t* time,real_t* qh,real_t* qhi,real_t* ld,real_t* inv,real_t* mq,real_t* rhs,real_t* si,real_t* solution,int* pivot,int* wc,int* wl,const int* damp,real_t* sp,real_t* sv,real_t* used,real_t* st,real_t* axis,real_t* angle,real_t* increment,real_t* pre,int* af,int* za,int* qf,real_t* work,int* skip,const int* status,int nq,int nv,int nm,int nq4){
 // Every lane enters: active/status are world-uniform and remain immutable.
 const int lane=threadIdx.x;

 qp+=w*nq;qv+=w*nv;qh+=w*nm;qhi+=w*nv;ld+=w*nm;inv+=w*nv;mq+=w*nm;rhs+=w*4*nv;si+=w*nv;solution+=w*4*nv;pivot+=w*nv;sp+=w*nq;sv+=w*nv;used+=w*nv;axis+=w*nq4*3;angle+=w*nq4;increment+=w*nq4*4;pre+=w*nq4*4;af+=w*nq4;za+=w*nq4;qf+=w*nq4;work+=w*nv;skip+=w*nv;
 const bool failed=active[w]!=1||status[w]!=0;
 if(failed){
  libero_vec::fill_block(mq,nm,REAL_C(0.0));libero_vec::fill_block(ld,nm,REAL_C(0.0));
  libero_vec::fill_block(inv,nv,REAL_C(0.0));libero_vec::fill_block(si,nv,REAL_C(0.0));
  libero_vec::fill_block(work,nv,REAL_C(0.0));libero_vec::fill_block(sv,nv,REAL_C(0.0));libero_vec::fill_block(used,nv,REAL_C(0.0));
  libero_vec::fill_block(pivot,nv,0);libero_vec::fill_block(skip,nv,0);
  libero_vec::fill_block(rhs,4*nv,REAL_C(0.0));libero_vec::fill_block(solution,4*nv,REAL_C(0.0));
  libero_vec::fill_block(sp,nq,REAL_C(0.0));libero_vec::fill_block(angle,nq4,REAL_C(0.0));
  libero_vec::fill_block(af,nq4,0);libero_vec::fill_block(za,nq4,0);libero_vec::fill_block(qf,nq4,0);
  libero_vec::fill_block(axis,3*nq4,REAL_C(0.0));
  libero_vec::fill_block(increment,4*nq4,REAL_C(0.0));libero_vec::fill_block(pre,4*nq4,REAL_C(0.0));
  if(lane==0){wc[w]=0;wl[w]=-1;st[w]=0;}
 }else{
  libero_vec::copy_block(sp,qp,nq);
 }
 __syncthreads();
 if(!failed)libero_vec::copy_block(sv,qv,nv);
 __syncthreads();
 // Preserve the scalar transaction's vectors -> time -> damped mass order.
 if(!failed&&lane==0)time[w]=st[w];
 __syncthreads();
 if(!failed&&damp[w]){
  libero_vec::copy_block(ld,qh,nm);
 }
 __syncthreads();
 if(!failed&&damp[w])libero_vec::copy_block(inv,qhi,nv);
 // Compatibility export and valid/epoch publication follow completed stores.
 __syncthreads();
}
__device__ void gate(int w,const int* intent,const int64_t* epoch,const int* fv,const int64_t* fw,const int* fe,int* first,const real_t* qp,const real_t* qv,const real_t* qa,int* active,int nq,int nv){
 active[w]=0;if(intent[w]==0||first[w])return;if(intent[w]!=1){first[w]=1;return;}if(epoch[w]==INT64_MAX){first[w]=9;return;}if(fe[w]){first[w]=fe[w];return;}if(fv[w]!=1||fw[w]!=epoch[w])return;
 for(int i=0;i<nq;i++)if(!isfinite(qp[w*nq+i])||fabs(qp[w*nq+i])>REAL_C(1e10)){first[w]=2;return;}
 for(int i=0;i<nv;i++){if(!isfinite(qv[w*nv+i])||fabs(qv[w*nv+i])>REAL_C(1e10)){first[w]=3;return;}if(!isfinite(qa[w*nv+i])||fabs(qa[w*nv+i])>REAL_C(1e10)){first[w]=4;return;}}active[w]=1;
}
// qpos errors precede velocity/acceleration errors. For the latter,
// index * 2 + field retains the scalar loop's velocity-before-acceleration rule.
__device__ void gate_block(int w,const int* intent,const int64_t* epoch,const int* fv,const int64_t* fw,const int* fe,int* first,const real_t* qp,const real_t* qv,const real_t* qa,int* active,int nq,int nv){
 if(threadIdx.x==0)active[w]=0;
 __syncthreads();
 if(threadIdx.x<32){
  const int lane=threadIdx.x;
  if(intent[w]==0||first[w]){}
  else if(intent[w]!=1){if(lane==0)first[w]=1;}
  else if(epoch[w]==INT64_MAX){if(lane==0)first[w]=9;}
  else if(fe[w]){if(lane==0)first[w]=fe[w];}
  else if(fv[w]==1&&fw[w]==epoch[w]){
   bool bad=false;
   for(int i=lane;i<nq;i+=32)bad|=!isfinite(qp[w*nq+i])||fabs(qp[w*nq+i])>REAL_C(1e10);
   bad=libero_warp::any(bad);
   if(bad){if(lane==0)first[w]=2;}
   else {
    int key=libero_warp::no_index;
    for(int i=lane;i<nv;i+=32){
     if(!isfinite(qv[w*nv+i])||fabs(qv[w*nv+i])>REAL_C(1e10))key=min(key,2*i);
     if(!isfinite(qa[w*nv+i])||fabs(qa[w*nv+i])>REAL_C(1e10))key=min(key,2*i+1);
    }
    key=libero_warp::min(key);
    if(lane==0){if(key!=libero_warp::no_index)first[w]=(key&1)?4:3;else active[w]=1;}
   }
  }
 }
 __syncthreads();
}
__device__ void publish(int w,const int* active,const int* status,int64_t* epoch,int64_t* written,int* valid,int* first,int* fv){
 valid[w]=0;if(active[w]!=1)return;if(status[w]){if(!first[w])first[w]=100+status[w];return;}epoch[w]++;written[w]=epoch[w];valid[w]=1;fv[w]=0;
}
__device__ void expected(int w,const int* intent,int64_t* expected){if(intent[w]==1&&expected[w]<INT64_MAX)expected[w]++;}
__device__ void count(int w,const int* valid,int64_t* counter){if(valid[w]==1)counter[w]++;}

// The numerical stages share a block-owned working set. Global scratch is
// exported once for the compatible diagnostic API, never used for handoff.
// qacc_stride == 0 also permits a caller to supply a block-local acceleration
// vector after a future N/post tail integration.
#if defined(CF_EULER_NV)
#ifndef CF_EULER_DIAGNOSTICS
#define CF_EULER_DIAGNOSTICS 1
#endif
struct SharedEulerWork {
 real_t qm[CF_EULER_NM],rhs[4*CF_EULER_NV],ld[CF_EULER_NM];
 real_t inv[CF_EULER_NV],sqrt_inv[CF_EULER_NV],solution[4*CF_EULER_NV];
 real_t pos[CF_EULER_NQ],vel[CF_EULER_NV],used[CF_EULER_NV],time[1];
 real_t zero_work[CF_EULER_NV];
 real_t axis[3*CF_EULER_QUAT_STORAGE],angle[CF_EULER_QUAT_STORAGE];
 real_t increment[4*CF_EULER_QUAT_STORAGE],pre[4*CF_EULER_QUAT_STORAGE];
 int active[1],count[1],mass_active[1],damped[1],status[1],mass_status[1];
 int warning_count[1],warning_last[1],pivot[CF_EULER_NV],zero_skip[CF_EULER_NV];
 int axis_fallback[CF_EULER_QUAT_STORAGE],zero_angle[CF_EULER_QUAT_STORAGE],quat_fallback[CF_EULER_QUAT_STORAGE];
};
__device__ void evaluate_shared(int w,const int* active,real_t* qp,real_t* qv,real_t* time,
 const real_t* qm,const real_t* smooth,const real_t* constraint,const real_t* acc,
 const real_t* damping,const int* madr,real_t* mq,real_t* rhs,int* count,int* ma,
 int* damp,int* status,real_t dt,int nq,int nv,int nm,const int* parent,real_t* ld,
 real_t* inv,real_t* si,real_t* solution,int* pivot,int* ms,int* wc,int* wl,
 const int* simple,real_t* work,int* skip,const int* jt,const int* padr,const int* vadr,
 real_t* sp,real_t* sv,real_t* used,real_t* st,real_t* axis,real_t* angle,
 real_t* increment,real_t* pre,int* af,int* za,int* qf,int nj,int nq4,real_t* qh,
 real_t* qhi,int qacc_stride,const int* joint_quat=nullptr,
 const int* joint_type_offsets=nullptr,const int* joint_type_indices=nullptr) {
 __shared__ SharedEulerWork scratch;
 if(threadIdx.x==0) {
  scratch.active[0]=active[w];
 }
 __syncthreads();
 prepare_block(0,scratch.active,qp+w*nq,qv+w*nv,time+w,qm+w*nm,
   smooth+w*nv,constraint+w*nv,acc+w*qacc_stride,damping,madr,
   scratch.qm,scratch.rhs,scratch.count,scratch.mass_active,scratch.damped,
   scratch.status,dt,nq,nv,nm);
 mass_block(0,scratch.qm,scratch.rhs,scratch.count,scratch.mass_active,madr,parent,
   scratch.ld,scratch.inv,scratch.sqrt_inv,scratch.solution,scratch.pivot,
   scratch.mass_status,scratch.warning_count,scratch.warning_last,nv,nm);
#if CF_EULER_DIAGNOSTICS
 // Replay produces only the compatible zero-skip diagnostic; integration
 // consumes mass_block's solution. Keep its barrier in both profiles.
 if(threadIdx.x==0) {
  replay(0,scratch.active,scratch.damped,scratch.mass_status,scratch.rhs,
   scratch.ld,madr,parent,simple,scratch.zero_work,scratch.zero_skip,nv,nm);
 }
#endif
 __syncthreads();
 integrate_block(0,scratch.active,qp+w*nq,qv+w*nv,time+w,acc+w*qacc_stride,
   scratch.solution,scratch.mass_status,scratch.damped,jt,padr,vadr,
   scratch.pos,scratch.vel,scratch.used,scratch.time,scratch.axis,scratch.angle,
   scratch.increment,scratch.pre,scratch.axis_fallback,scratch.zero_angle,
   scratch.quat_fallback,scratch.status,dt,nq,nv,nj,nq4,false,joint_quat,joint_type_offsets,joint_type_indices);
 __syncthreads();
 if(threadIdx.x<32&&scratch.active[0]==1&&scratch.status[0]==0){
  bool bad=false;
  for(int i=threadIdx.x;i<nq;i+=32)bad|=!isfinite(scratch.pos[i]);
  bad=libero_warp::any(bad);
  if(threadIdx.x==0&&bad)scratch.status[0]=3;
 }
 __syncthreads();
 commit_block(0,scratch.active,qp+w*nq,qv+w*nv,time+w,qh+w*nm,qhi+w*nv,
   scratch.ld,scratch.inv,scratch.qm,scratch.rhs,scratch.sqrt_inv,scratch.solution,
   scratch.pivot,scratch.warning_count,scratch.warning_last,scratch.damped,
   scratch.pos,scratch.vel,scratch.used,scratch.time,scratch.axis,scratch.angle,
   scratch.increment,scratch.pre,scratch.axis_fallback,scratch.zero_angle,
   scratch.quat_fallback,scratch.zero_work,scratch.zero_skip,scratch.status,nq,nv,nm,nq4);
 if(threadIdx.x==0){
  count[w]=scratch.count[0];ma[w]=scratch.mass_active[0];damp[w]=scratch.damped[0];
  status[w]=scratch.status[0];ms[w]=scratch.mass_status[0];
  wc[w]=scratch.warning_count[0];wl[w]=scratch.warning_last[0];
 }
 __syncthreads();
 // qacc_used is retained input-use evidence in both profiles. Runtime
 // diagnostic lengths exclude quaternion storage padding.
 libero_vec::copy_block(scratch.used,used+w*nv,nv);
#if CF_EULER_DIAGNOSTICS
 libero_vec::copy_block(scratch.qm,mq+w*nm,nm);libero_vec::copy_block(scratch.ld,ld+w*nm,nm);
 libero_vec::copy_block(scratch.rhs,rhs+w*4*nv,4*nv);libero_vec::copy_block(scratch.solution,solution+w*4*nv,4*nv);
 libero_vec::copy_block(scratch.inv,inv+w*nv,nv);libero_vec::copy_block(scratch.sqrt_inv,si+w*nv,nv);
 libero_vec::copy_block(scratch.pivot,pivot+w*nv,nv);libero_vec::copy_block(scratch.zero_work,work+w*nv,nv);
 libero_vec::copy_block(scratch.zero_skip,skip+w*nv,nv);libero_vec::copy_block(scratch.vel,sv+w*nv,nv);
 libero_vec::copy_block(scratch.pos,sp+w*nq,nq);libero_vec::copy_block(scratch.axis,axis+w*3*nq4,3*nq4);
 libero_vec::copy_block(scratch.increment,increment+w*4*nq4,4*nq4);libero_vec::copy_block(scratch.pre,pre+w*4*nq4,4*nq4);
 libero_vec::copy_block(scratch.angle,angle+w*nq4,nq4);libero_vec::copy_block(scratch.axis_fallback,af+w*nq4,nq4);
 libero_vec::copy_block(scratch.zero_angle,za+w*nq4,nq4);libero_vec::copy_block(scratch.quat_fallback,qf+w*nq4,nq4);
 if(threadIdx.x==0)st[w]=scratch.time[0];
#endif
}
#endif
}
