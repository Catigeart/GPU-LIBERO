// Maintained scalar-parameterized MuJoCo 2.3.7 line search.
#pragma once
#include "precision.cuh"
// MuJoCo 2.3.7 source-order Newton line search, fixed dense-J model boundary.
// Source: engine_solver.c:834-1267, engine_support.c:826-873,
// engine_util_blas.c:631-694. Apache 2.0, DeepMind Technologies Limited.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

struct S07Args {
  const real_t *search, *Ma, *Jaref, *quad0, *smooth, *qM, *J;
  const real_t *D, *R, *loss, *cone_mu, *cone_friction;
  const int32_t *efc_type, *efc_id, *cone_dim;
  const int32_t *madr, *parent, *simplenum;
  const int32_t *nefc, *ne, *nf, *active;
  const real_t *model_scalars;  // tolerance, meaninertia
  real_t *Mv, *Jv, *quadGauss, *quad, *alpha, *slope, *eval_trace;
  int32_t *LSiter, *LSresult, *eval_tag, *status;
};

struct P { real_t alpha, cost, d0, d1; };

// Mirrors engine_util_blas.c mju_dot, including four independent lanes and
// its 1/2/3-element tail grouping. CUDA build disables FMA contraction.
__device__ real_t dot_strided(const real_t* a, int astride,
                              const real_t* b, int bstride, int n) {
  real_t res = REAL_C(0.0);
  int i = 0;
  int n4 = n - 4;
  real_t r0 = REAL_C(0.0), r1 = REAL_C(0.0), r2 = REAL_C(0.0), r3 = REAL_C(0.0);
  if (n4 >= 0) {
    r0 = a[0]*b[0]; r1 = a[astride]*b[bstride];
    r2 = a[2*astride]*b[2*bstride]; r3 = a[3*astride]*b[3*bstride];
    i = 4;
  }
  for (; i <= n4; i += 4) {
    r0 += a[i*astride] * b[i*bstride];
    r1 += a[(i+1)*astride] * b[(i+1)*bstride];
    r2 += a[(i+2)*astride] * b[(i+2)*bstride];
    r3 += a[(i+3)*astride] * b[(i+3)*bstride];
  }
  res = (r0+r2)+(r1+r3);
  int tail = n-i;
  if (tail == 3) res += a[i*astride]*b[i*bstride] +
                         a[(i+1)*astride]*b[(i+1)*bstride] +
                         a[(i+2)*astride]*b[(i+2)*bstride];
  else if (tail == 2) res += a[i*astride]*b[i*bstride] +
                              a[(i+1)*astride]*b[(i+1)*bstride];
  else if (tail == 1) res += a[i*astride]*b[i*bstride];
  return res;
}

__device__ void add3(real_t* total, const real_t* q) {
  total[0] += q[0]; total[1] += q[1]; total[2] += q[2];
}

__device__ P eval_point(const S07Args& a, int w, int rows, int nefc,
                        int ne, int nf, real_t alpha, int tag, int* niter) {
  const int qbase = w*rows*3, base = w*rows;
  real_t total[3] = {a.quad0[w], a.quadGauss[w*3+1],
                     a.quadGauss[w*3+2]};
  real_t cost = REAL_C(0.0), deriv0 = REAL_C(0.0), deriv1 = REAL_C(0.0);
  for (int i=0; i<ne; i++) add3(total, a.quad+qbase+3*i);
  for (int i=ne; i<ne+nf; i++) {
    real_t start = a.Jaref[base+i], dir = a.Jv[base+i];
    real_t x = start + alpha*dir;
    real_t f = a.loss[base+i], Rf = a.R[base+i]*f;
    if (-Rf < x && x < Rf) add3(total, a.quad+qbase+3*i);
    else if (x <= -Rf) {
      real_t qf[3] = {f*(-REAL_C(0.5)*Rf-start), -f*dir, REAL_C(0.0)};
      add3(total, qf);
    } else {
      real_t qf[3] = {f*(-REAL_C(0.5)*Rf+start), f*dir, REAL_C(0.0)};
      add3(total, qf);
    }
  }
  for (int i=ne+nf; i<nefc; i++) {
    if (a.efc_type[base+i] == 7) {  // mjCNSTR_CONTACT_ELLIPTIC
      const real_t* q = a.quad+qbase+3*i;
      int dim = a.cone_dim[base+i];
      real_t mu = a.cone_mu[base+i];
      real_t U0=q[3], V0=q[4], UU=q[5], UV=q[6], VV=q[7], Dm=q[8];
      real_t N = U0 + alpha*V0;
      real_t Tsqr = UU + alpha*(REAL_C(2.0)*UV + alpha*VV);
      if (Tsqr <= REAL_C(0.0)) {
        if (N < REAL_C(0.0)) add3(total, q);
      } else {
        real_t T = real_sqrt_rn(Tsqr);
        if (N >= mu*T) {
          // top zone
        } else if (mu*N+T <= REAL_C(0.0)) {
          add3(total, q);
        } else {
          real_t N1=V0;
          real_t T1=real_div_rn(UV+alpha*VV,T);
          real_t T2=real_div_rn(VV,T) - real_div_rn((UV+alpha*VV)*T1,T*T);
          cost += REAL_C(0.5)*Dm*(N-mu*T)*(N-mu*T);
          deriv0 += Dm*(N-mu*T)*(N1-mu*T1);
          deriv1 += Dm*((N1-mu*T1)*(N1-mu*T1)+(N-mu*T)*(-mu*T2));
        }
      }
      i += dim-1;
    } else {
      real_t x = a.Jaref[base+i] + alpha*a.Jv[base+i];
      if (x < REAL_C(0.0)) add3(total, a.quad+qbase+3*i);
    }
  }
  cost += alpha*alpha*total[2] + alpha*total[1] + total[0];
  deriv0 += REAL_C(2.0)*alpha*total[2] + total[1];
  deriv1 += REAL_C(2.0)*total[2];
  if (deriv1 <= REAL_C(0.0)) deriv1 = REAL_C(1.0e-15);
  if (!isfinite(alpha) || !isfinite(cost) || !isfinite(deriv0) || !isfinite(deriv1))
    a.status[w] = 5;
  P p = {alpha, cost, deriv0, deriv1};
  int idx = *niter;
  a.eval_trace[(w*52+idx)*4] = alpha;
  a.eval_trace[(w*52+idx)*4+1] = cost;
  a.eval_trace[(w*52+idx)*4+2] = deriv0;
  a.eval_trace[(w*52+idx)*4+3] = deriv1;
  a.eval_tag[w*52+idx] = tag;
  *niter = idx+1;
  return p;
}

__device__ int update_bracket(const S07Args& a, int w, int rows, int nefc,
                              int ne, int nf, P* p, const P* candidates,
                              P* next, int* niter, int tag) {
  int flag=0;
  for (int i=0; i<3; i++) {
    if (p->d0 < REAL_C(0.0) && candidates[i].d0 < REAL_C(0.0) && p->d0 < candidates[i].d0) {
      *p = candidates[i]; flag=1;
    } else if (p->d0 > REAL_C(0.0) && candidates[i].d0 > REAL_C(0.0) && p->d0 > candidates[i].d0) {
      *p = candidates[i]; flag=2;
    }
  }
  if (flag) {
    real_t alpha = p->alpha - real_div_rn(p->d0,p->d1);
    *next = eval_point(a,w,rows,nefc,ne,nf,alpha,tag,niter);
  }
  return flag;
}

__device__ void finish(const S07Args& a, int w, real_t alpha,
                       int niter, int result, real_t slope) {
  a.alpha[w]=alpha; a.LSiter[w]=niter; a.LSresult[w]=result; a.slope[w]=slope;
}

__device__ void search_kernel(int w, S07Args a, int nworld, int nv, int nM,
                              int rows, int tracecap) {

  if (w >= nworld) return;
  a.status[w] = 0;
  if (a.active[w] == 0) return;
  if (a.active[w] != 1) { a.status[w]=1; return; }
  int nefc=a.nefc[w], ne=a.ne[w], nf=a.nf[w];
  if (nefc < 0 || nefc > rows || ne < 0 || nf < 0 || ne > nefc || nf > nefc-ne) {
    a.status[w]=2; return;
  }
  if (tracecap < 52) { a.status[w]=3; return; }
  int base=w*rows;
  for (int i=0; i<nefc; i++) {
    if (a.efc_type[base+i] == 7) {
      int dim=a.cone_dim[base+i];
      if (i < ne+nf || (dim != 3 && dim != 4 && dim != 6) || i+dim > nefc ||
          a.cone_mu[base+i] <= REAL_C(0.0)) { a.status[w]=4; return; }
      i += dim-1;
    }
  }
  const real_t* v=a.search+w*nv;
  real_t snorm=real_sqrt_rn(dot_strided(v,1,v,1,nv));
  if (!isfinite(snorm)) { a.status[w]=5; return; }
  // The CPU context carries quadGauss[0] on entry; the device ABI stores that
  // frozen input separately, so materialize its unchanged value in the output.
  a.quadGauss[w*3] = a.quad0[w];
  if (snorm < REAL_C(1.0e-15)) { finish(a,w,REAL_C(0.0),0,1,REAL_C(1.0)); return; }
  real_t mean=a.model_scalars[1];
  real_t gtol=real_linesearch_tolerance(a.model_scalars[0]*REAL_C(0.01)*snorm*mean*(nv>1 ? nv : 1));
  real_t slopescl=real_div_rn(REAL_C(1.0),snorm*mean*(nv>1 ? nv : 1));
  if (!isfinite(gtol) || !isfinite(slopescl)) { a.status[w]=5; return; }
  real_t* Mv=a.Mv+w*nv;
  const real_t* M=a.qM+w*nM;
  for (int i=0; i<nv; i++) Mv[i]=REAL_C(0.0);
  for (int i=0; i<nv; i++) {
    int adr=a.madr[i];
    if (a.simplenum[i]) {
      Mv[i]=M[adr]*v[i];
    } else {
      Mv[i] += M[adr]*v[i];
      int j=a.parent[i]; adr++;
      while (j>=0) {
        Mv[i] += M[adr]*v[j];
        Mv[j] += M[adr]*v[i];
        j=a.parent[j]; adr++;
      }
    }
  }
  for (int i=0; i<nv; i++) if (!isfinite(Mv[i])) { a.status[w]=5; return; }
  real_t* Jv=a.Jv+base;
  const real_t* J=a.J+w*rows*nv;
  for (int i=0; i<nefc; i++) Jv[i]=dot_strided(J+i*nv,1,v,1,nv);
  for (int i=0; i<nefc; i++) if (!isfinite(Jv[i])) { a.status[w]=5; return; }
  real_t* qG=a.quadGauss+w*3;
  qG[1]=dot_strided(v,1,a.Ma+w*nv,1,nv)-dot_strided(v,1,a.smooth+w*nv,1,nv);
  qG[2]=REAL_C(0.5)*dot_strided(v,1,Mv,1,nv);
  if (!isfinite(qG[1]) || !isfinite(qG[2])) { a.status[w]=5; return; }
  real_t* qbase=a.quad+w*rows*3;
  for (int i=0; i<nefc; i++) {
    real_t* q=qbase+3*i;
    bool elliptic=a.efc_type[base+i] == 7;
    real_t DJ0=a.D[base+i]*a.Jaref[base+i];
    q[0]=a.Jaref[base+i]*DJ0;
    q[1]=Jv[i]*DJ0;
    q[2]=Jv[i]*a.D[base+i]*Jv[i];
    if (elliptic) {
      int dim=a.cone_dim[base+i];
      real_t mu=a.cone_mu[base+i];
      for (int j=1; j<dim; j++) {
        real_t DJj=a.D[base+i+j]*a.Jaref[base+i+j];
        q[0] += a.Jaref[base+i+j]*DJj;
        q[1] += Jv[i+j]*DJj;
        q[2] += Jv[i+j]*a.D[base+i+j]*Jv[i+j];
      }
      real_t U0=a.Jaref[base+i]*mu, V0=Jv[i]*mu;
      real_t UU=REAL_C(0.0), UV=REAL_C(0.0), VV=REAL_C(0.0);
      for (int j=1; j<dim; j++) {
        real_t U=a.Jaref[base+i+j]*a.cone_friction[(base+i)*5+j-1];
        real_t V=Jv[i+j]*a.cone_friction[(base+i)*5+j-1];
        UU += U*U; UV += U*V; VV += V*V;
      }
      q[3]=U0; q[4]=V0; q[5]=UU; q[6]=UV; q[7]=VV;
      q[8]=real_div_rn(a.D[base+i],mu*mu*(REAL_C(1.0)+mu*mu));
      i += dim-1;
    }
    q[0] *= REAL_C(0.5); q[2] *= REAL_C(0.5);
    for (int k=0; k<3; k++) if (!isfinite(q[k])) { a.status[w]=5; return; }
    if (elliptic) {
      for (int k=3; k<9; k++) if (!isfinite(q[k])) { a.status[w]=5; return; }
    }
  }
  int niter=0;
  P p0=eval_point(a,w,rows,nefc,ne,nf,REAL_C(0.0),1,&niter);
  P p1=eval_point(a,w,rows,nefc,ne,nf,
                  p0.alpha-real_div_rn(p0.d0,p0.d1),2,&niter);
  if (p0.cost < p1.cost) p1=p0;
  if (fabs(p1.d0) < gtol) {
    finish(a,w,p1.alpha,niter,p1.alpha==REAL_C(0.0) ? 2 : 0,fabs(p1.d0)*slopescl);
    return;
  }
  int dir=p1.d0<REAL_C(0.0) ? 1 : -1;
  int p2update=0;
  P p2={};
  while (p1.d0*dir <= -gtol && niter < 50) {
    p2=p1; p2update=1;
    p1=eval_point(a,w,rows,nefc,ne,nf,
                  p1.alpha-real_div_rn(p1.d0,p1.d1),3,&niter);
    if (fabs(p1.d0)<gtol) {
      finish(a,w,p1.alpha,niter,0,fabs(p1.d0)*slopescl); return;
    }
  }
  if (niter>=50) {
    finish(a,w,p1.alpha,niter,3,fabs(p1.d0)*slopescl); return;
  }
  if (!p2update) {
    finish(a,w,p1.alpha,niter,6,fabs(p1.d0)*slopescl); return;
  }
  P p2next=p1;
  P p1next=eval_point(a,w,rows,nefc,ne,nf,
                      p1.alpha-real_div_rn(p1.d0,p1.d1),4,&niter);
  while (niter<50) {
    P pmid=eval_point(a,w,rows,nefc,ne,nf,REAL_C(0.5)*(p1.alpha+p2.alpha),5,&niter);
    P candidates[3]={p1next,p2next,pmid};
    real_t bestcost=REAL_C(0.0); int bestind=-1;
    for (int i=0; i<3; i++) {
      if (fabs(candidates[i].d0)<gtol &&
          (bestind==-1 || candidates[i].cost<bestcost)) {
        bestcost=candidates[i].cost; bestind=i;
      }
    }
    if (bestind>=0) {
      finish(a,w,candidates[bestind].alpha,niter,0,
             fabs(candidates[bestind].d0)*slopescl); return;
    }
    int b1=update_bracket(a,w,rows,nefc,ne,nf,&p1,candidates,&p1next,&niter,6);
    int b2=update_bracket(a,w,rows,nefc,ne,nf,&p2,candidates,&p2next,&niter,7);
    if (!b1 && !b2) {
      finish(a,w,pmid.alpha,niter,pmid.cost<p0.cost ? 0 : 7,
             fabs(pmid.d0)*slopescl); return;
    }
  }
  if (p1.cost<=p2.cost && p1.cost<p0.cost) {
    finish(a,w,p1.alpha,niter,4,fabs(p1.d0)*slopescl);
  } else if (p2.cost<=p1.cost && p2.cost<p0.cost) {
    finish(a,w,p2.alpha,niter,4,fabs(p2.d0)*slopescl);
  } else {
    finish(a,w,REAL_C(0.0),niter,5,REAL_C(1.0));
  }
}

