#include "precision.cuh"
#pragma once
__device__ real_t assembly_cell(int w,int i,int j,int rows,int* count,real_t* J,real_t* D,int* state,
 real_t* qm,int* madr,int* parent,int* simple){
  real_t s=0;for(int r=0;r<count[w];++r){int z=w*rows+r;
    if(state[z]==1&&D[z]!=0){real_t v=J[z*NV+i];if(v!=0)s+=J[z*NV+j]*(v*D[z]);}}
  int adr=mass_index(i,j,madr,parent,simple);if(adr>=0)s+=qm[w*NM+adr];return s;
}
__device__ void hessian_validate(int w,int* count,int* ncon,int* active,int* kind,int* state,int* oldstate,
 real_t* D,real_t* J,real_t* qm,real_t* H,int* id,int* dim,real_t* contact_H,int* ncone,int* nnz,int* status,int* route,int rows,int contacts,
 int* invariant_valid=nullptr){
  status[w]=0;route[w]=0;if(!active[w])return;
  int k=kind[w],nr=count[w],nc=ncon[w],rb=w*rows;
  if(active[w]!=1||(k!=1&&k!=2&&k!=3)){status[w]=1;return;}
  if(nr<0||nr>rows||nc<0||nc>contacts){status[w]=2;return;}
  if(k!=1&&nnz[w]!=NV*NV){status[w]=3;return;}
  if(k!=1)for(int i=0;i<NV;++i)for(int j=0;j<NV;++j)
    if((k==2||ncone[w]!=0||j<=i)&&!isfinite(H[w*NV*NV+i*NV+j])){status[w]=4;return;}
  int cones=0;
  for(int r=0;r<nr;++r){int s=state[rb+r],old=oldstate[rb+r];
    if(s<0||s>4||(k==3&&(old<0||old>4))){status[w]=3;return;}if(s==4)++cones;
    bool changed=k==3&&((old==1)!=(s==1));
    if(((s==1&&k==1)||changed)&&(D[rb+r]<0||(!(invariant_valid&&invariant_valid[0])&&!isfinite(D[rb+r])))){status[w]=4;return;}
    if(!(invariant_valid&&invariant_valid[0])&&(s==4||(k==1&&s==1&&D[rb+r]!=0)||changed))
      for(int j=0;j<NV;++j)if(!isfinite(J[(rb+r)*NV+j])){status[w]=4;return;}
  }
  if(cones!=ncone[w]){status[w]=3;return;}
  for(int r=0;r<nr;){if(state[rb+r]!=4){++r;continue;}int c=id[rb+r];
    if(c<0||c>=nc){status[w]=3;return;}int d=dim[w*contacts+c];
    if((d!=1&&d!=3&&d!=4&&d!=6)||r+d>nr){status[w]=3;return;}
    for(int j=0;j<d;++j)if(state[rb+r+j]!=4||id[rb+r+j]!=c){status[w]=3;return;}
    for(int j=0;j<d*d;++j)if(!isfinite(contact_H[(w*contacts+c)*36+j])){status[w]=4;return;}r+=d;
  }
  if(k==1)for(int j=0;j<NM;++j)if(!isfinite(qm[w*NM+j])){status[w]=4;return;}
}
// All block threads enter; only the first full warp performs validation.
// The caller must retain its stage barrier before consuming status/route.
__device__ void hessian_validate_block(int w,int* count,int* ncon,int* active,int* kind,int* state,int* oldstate,
 real_t* D,real_t* J,real_t* qm,real_t* H,int* id,int* dim,real_t* contact_H,int* ncone,int* nnz,int* status,int* route,int rows,int contacts,
 int* invariant_valid=nullptr){
  if(libero_warp::warp()!=0)return;
  const int lane=libero_warp::lane();
  int k=kind[w],nr=count[w],nc=ncon[w],rb=w*rows,code=0;
  if(active[w]){
    if(active[w]!=1||(k!=1&&k!=2&&k!=3))code=1;
    else if(nr<0||nr>rows||nc<0||nc>contacts)code=2;
    else if(k!=1&&nnz[w]!=NV*NV)code=3;
  }
  if(lane==0){status[w]=code;route[w]=0;}
  if(code||!active[w])return;
  bool bad=false;
  if(k!=1)for(int z=lane;z<NV*NV;z+=32)
    if((k==2||ncone[w]!=0||z%NV<=z/NV)&&!isfinite(H[w*NV*NV+z]))bad=true;
  if(libero_warp::any(bad)){if(lane==0)status[w]=4;return;}
  const bool invariant=invariant_valid&&invariant_valid[0];
  int first=libero_warp::no_index,first_code=0,cones=0;
  for(int r=lane;r<nr;r+=32){
    int st=state[rb+r],old=oldstate[rb+r],err=0;
    cones+=st==4;
    if(st<0||st>4||(k==3&&(old<0||old>4)))err=3;
    else{
      bool changed=k==3&&((old==1)!=(st==1));
      if(((st==1&&k==1)||changed)&&(D[rb+r]<0||(!invariant&&!isfinite(D[rb+r]))))err=4;
      if(!invariant&&(st==4||(k==1&&st==1&&D[rb+r]!=0)||changed))
        for(int j=0;j<NV;++j)if(!isfinite(J[(rb+r)*NV+j]))err=4;
    }
    // Row position dominates error code, exactly as the serial row loop.
    if(err&&r<first){first=r;first_code=err;}
  }
  first=libero_warp::min(first);cones=libero_warp::sum(cones);
  if(first!=libero_warp::no_index){
    first_code=__shfl_sync(libero_warp::mask,first_code,first&31);
    if(lane==0)status[w]=first_code;return;
  }
  if(cones!=ncone[w]){if(lane==0)status[w]=3;return;}
  // Variable-length cone schema is ordered and must precede qM validation.
  code=0;
  if(lane==0)for(int r=0;r<nr;){
    if(state[rb+r]!=4){++r;continue;}int c=id[rb+r];
    if(c<0||c>=nc){code=3;break;}int d=dim[w*contacts+c];
    if((d!=1&&d!=3&&d!=4&&d!=6)||r+d>nr){code=3;break;}
    for(int j=0;j<d;++j)if(state[rb+r+j]!=4||id[rb+r+j]!=c){code=3;break;}
    if(code)break;
    for(int j=0;j<d*d;++j)if(!isfinite(contact_H[(w*contacts+c)*36+j])){code=4;break;}
    if(code)break;r+=d;
  }
  code=__shfl_sync(libero_warp::mask,code,0);
  if(code){if(lane==0)status[w]=code;return;}
  bad=false;
  if(k==1)for(int j=lane;j<NM;j+=32)if(!isfinite(qm[w*NM+j]))bad=true;
  if(libero_warp::any(bad)&&lane==0)status[w]=4;
}
__device__ void hessian_assemble(int tid,int* count,int* active,int* kind,real_t* J,real_t* D,int* state,
 real_t* qm,int* madr,int* parent,int* simple,real_t* H,int* status,int rows){
  int w=tid/(NV*NV),i=(tid/NV)%NV,j=tid%NV;
  if(!active[w]||kind[w]!=1||status[w]||j>i)return;
  real_t v=assembly_cell(w,i,j,rows,count,J,D,state,qm,madr,parent,simple);
  H[w*NV*NV+i*NV+j]=v;if(i!=j)H[w*NV*NV+j*NV+i]=v;
}
__device__ void hessian_cone(int w,int rows,int contacts,int* count,int* state,int* id,int* dim,
 real_t* contact_H,real_t* J,real_t* H,real_t* HC,real_t* local,real_t* ltj,real_t* vec,int* nupdate){
  int rb=w*rows;H+=w*NV*NV;HC+=w*NV*NV;local+=w*36;ltj+=w*6*NV;vec+=w*NV;
  for(int z=0;z<NV*NV;++z)HC[z]=H[z];
  for(int i=0;i<count[w];){if(state[rb+i]!=4){++i;continue;}int c=w*contacts+id[rb+i],d=dim[c];
    for(int z=0;z<d*d;++z)local[z]=contact_H[c*36+z];chol_factor(local,d,d);
    for(int z=0;z<d*NV;++z)ltj[z]=REAL_C(0.0);
    for(int r=0;r<d;++r)for(int c0=0;c0<=r;++c0){real_t s=local[r*d+c0];
      for(int k=0;k<NV;++k)ltj[c0*NV+k]+=J[(rb+i+r)*NV+k]*s;}
    for(int r=0;r<d;++r){for(int k=0;k<NV;++k)vec[k]=ltj[r*NV+k];chol_update(HC,vec,1);}
    nupdate[w]+=d;i+=d;
  }
}
__device__ void hessian_factor_update(int w,int* count,int* ncon,int* active,int* kind,real_t* J,real_t* D,
 int* state,int* oldstate,real_t* qm,int* madr,int* parent,int* simple,int* id,int* dim,real_t* contact_H,
 int* ncone,real_t* H,real_t* HC,int* nnz,int* nupdate,real_t* local,real_t* ltj,real_t* vec,int* status,int* route,int rows,int contacts){
  if(!active[w]||status[w])return;int hb=w*NV*NV,vb=w*NV,k=kind[w];
  if(k==1){chol_factor(H+hb,NV,NV);nnz[w]=NV*NV;
    if(ncone[w]){nupdate[w]=0;hessian_cone(w,rows,contacts,count,state,id,dim,contact_H,J,H,HC,local,ltj,vec,nupdate);route[w]=2;}
    else route[w]=1;nupdate[w]=count[w];
  }else if(k==2){hessian_cone(w,rows,contacts,count,state,id,dim,contact_H,J,H,HC,local,ltj,vec,nupdate);route[w]=6;}
  else{nupdate[w]=0;bool lost=false;
    for(int r=0;r<count[w];++r){int z=w*rows+r;bool before=oldstate[z]==1,after=state[z]==1;
      if(before!=after){real_t s=sqrt(D[z]);for(int j=0;j<NV;++j)vec[vb+j]=J[z*NV+j]*s;
        int rank=chol_update(H+hb,vec+vb,after?1:0);++nupdate[w];if(rank<NV){lost=true;break;}}
    }
    if(lost){for(int z=0;z<NM;++z)if(!isfinite(qm[w*NM+z])){status[w]=4;return;}
      for(int r=0;r<count[w];++r){int z=w*rows+r;if(state[z]!=1)continue;
        if(D[z]<0||!isfinite(D[z])){status[w]=4;return;}
        if(D[z]!=0)for(int j=0;j<NV;++j)if(!isfinite(J[z*NV+j])){status[w]=4;return;}
      }
      for(int i=0;i<NV;++i)for(int j=0;j<=i;++j){real_t v=assembly_cell(w,i,j,rows,count,J,D,state,qm,madr,parent,simple);
        H[hb+i*NV+j]=v;if(i!=j)H[hb+j*NV+i]=v;}
      chol_factor(H+hb,NV,NV);nnz[w]=NV*NV;
      if(ncone[w]){nupdate[w]=0;hessian_cone(w,rows,contacts,count,state,id,dim,contact_H,J,H,HC,local,ltj,vec,nupdate);}
      nupdate[w]=count[w];route[w]=5;
    }else if(ncone[w]){hessian_cone(w,rows,contacts,count,state,id,dim,contact_H,J,H,HC,local,ltj,vec,nupdate);route[w]=4;}
    else route[w]=3;
  }
  for(int z=0;z<NV*NV;++z){bool check=k==1||route[w]==5||(k==3&&route[w]!=5&&z/NV>=z%NV);
    if(check&&!isfinite(H[hb+z])){status[w]=5;return;}
    if(ncone[w]&&!isfinite(HC[hb+z])){status[w]=5;return;}
  }
}
__device__ void direction(int w,real_t* Ma,real_t* smooth,real_t* constraint,real_t* H,real_t* HC,
 int* ncone,int* nefc,int* active,int* upstream,real_t* grad,real_t* Mgrad,real_t* candidate,int* status,
 real_t* scratch_grad,real_t* x,real_t* fwd){
  if(upstream[w]){status[w]=upstream[w];return;}status[w]=0;if(!active[w])return;
  if(active[w]!=1){status[w]=1;return;}
  if(nefc[w]<0||ncone[w]<0||ncone[w]>nefc[w]){status[w]=2;return;}
  real_t* L=(ncone[w]>0?HC:H)+w*NV*NV;int vb=w*NV;
  for(int i=0;i<NV;++i){if(!isfinite(Ma[vb+i])||!isfinite(smooth[vb+i])||!isfinite(constraint[vb+i])){status[w]=3;return;}
    for(int j=0;j<=i;++j)if(!isfinite(L[i*NV+j])){status[w]=3;return;}}
  for(int i=0;i<NV;++i)if(L[i*NV+i]==0){status[w]=4;return;}
  for(int i=0;i<NV;++i){real_t v=(Ma[vb+i]-smooth[vb+i])-constraint[vb+i];if(!isfinite(v)){status[w]=5;return;}
    scratch_grad[vb+i]=v;x[vb+i]=v;}
  for(int i=0;i<NV;++i){real_t v=x[vb+i];if(i)v-=dot4(L+i*NV,x+vb,i);v/=L[i*NV+i];
    if(!isfinite(v)){status[w]=5;return;}x[vb+i]=v;fwd[vb+i]=v;}
  for(int i=NV-1;i>=0;--i){real_t v=x[vb+i];for(int j=i+1;j<NV;++j)v-=L[j*NV+i]*x[vb+j];v/=L[i*NV+i];
    if(!isfinite(v)){status[w]=5;return;}x[vb+i]=v;}
  for(int i=0;i<NV;++i){grad[vb+i]=scratch_grad[vb+i];Mgrad[vb+i]=x[vb+i];candidate[vb+i]=copysign(x[vb+i],-copysign(REAL_C(1.0),x[vb+i]));}
}
