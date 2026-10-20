#include "precision.cuh"
#pragma once
namespace osc_cuda {
__device__ inline void pinv_observed(const real_t*a,int n,real_t*out,int&rank,int&ok,int* sweeps) {
    // Column-major private workspace makes each ordered column scan contiguous.
    // Only addresses change: reduction order, rotations, scale and cutoff stay fixed.
    real_t b[36]={},v[36]={},norm[6]={},scale=0;
    rank=0;ok=1;if(sweeps)*sweeps=0;
    for(int z=0;z<36;z++)out[z]=0;
    for(int i=0;i<n;i++)for(int j=0;j<n;j++)scale=fmax(scale,fabs(a[i*6+j]));
    if(!(scale>0))return;
    for(int i=0;i<n;i++){v[i*6+i]=1;for(int j=0;j<n;j++)b[j*6+i]=a[i*6+j]/scale;}
    ok=0;
    for(int sweep=0;sweep<64;sweep++){
        if(sweeps)*sweeps=sweep+1;
        int rotations=0;
        for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
            real_t alpha=0,beta=0,gamma=0;
            for(int i=0;i<n;i++){real_t bp=b[p*6+i],bq=b[q*6+i];alpha+=bp*bp;beta+=bq*bq;gamma+=bp*bq;}
            if(fabs(gamma)>real_t(REAL_EPSILON)*sqrt(alpha)*sqrt(beta)){
                real_t zeta=(beta-alpha)/(2*gamma),t=(zeta<0?-REAL_C(1.):REAL_C(1.))/(fabs(zeta)+sqrt(1+zeta*zeta));
                real_t c=1/sqrt(1+t*t),s=c*t;
                for(int i=0;i<n;i++){real_t bp=b[p*6+i],bq=b[q*6+i],vp=v[p*6+i],vq=v[q*6+i];
                    b[p*6+i]=c*bp-s*bq;b[q*6+i]=s*bp+c*bq;v[p*6+i]=c*vp-s*vq;v[q*6+i]=s*vp+c*vq;}
                rotations++;
            }
        }
        if(rotations==0){ok=1;break;}
    }
    real_t maxnorm=0,maxcorr=0,singular[6]={};bool retained[6]={};
    for(int p=0;p<n;p++){for(int i=0;i<n;i++)norm[p]+=b[p*6+i]*b[p*6+i];singular[p]=sqrt(norm[p]);maxnorm=fmax(maxnorm,singular[p]);}
    for(int p=0;p<n;p++)retained[p]=singular[p]>REAL_C(1e-15)*maxnorm;
    for(int p=0;p<n;p++)for(int q=p+1;q<n;q++)if(retained[p]&&retained[q]){
        real_t gamma=0;for(int i=0;i<n;i++)gamma+=b[p*6+i]*b[q*6+i];maxcorr=fmax(maxcorr,fabs(gamma)/(singular[p]*singular[q]));}
    if(maxcorr<REAL_C(2e-15))ok=1;
    for(int k=0;k<n;k++)if(retained[k]){rank++;real_t wt=(1/norm[k])/scale;
        for(int i=0;i<n;i++)for(int j=0;j<n;j++)out[i*6+j]+=v[k*6+i]*b[k*6+j]*wt;}
}

}
