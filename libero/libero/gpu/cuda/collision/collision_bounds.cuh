#include "precision.cuh"
#pragma once
#include <math.h>
#include <stdint.h>
#include "../common/warp.cuh"
#include "../common/vector.cuh"
#include "collision_sap_sort.cuh"

namespace collision_cuda {
// The original routines retain the owner-lane ABI. The *_block bounds
// entry points below require all threads of a >=32-thread block per world.
__device__ inline void cf_quat_matrix(real_t* r, const real_t* q) {
    const real_t w=q[0], x=q[1], y=q[2], z=q[3];
    if(w==REAL_C(1.0) && x==REAL_C(0.0) && y==REAL_C(0.0) && z==REAL_C(0.0)) {
        for(int k=0;k<9;++k) r[k]=(k%4==0)?REAL_C(1.0):REAL_C(0.0);
        return;
    }
    const real_t ww=w*w, wx=w*x, wy=w*y, wz=w*z;
    const real_t xx=x*x, xy=x*y, xz=x*z, yy=y*y, yz=y*z, zz=z*z;
    r[0]=ww+xx-yy-zz; r[4]=ww-xx+yy-zz; r[8]=ww-xx-yy+zz;
    r[1]=REAL_C(2.0)*(xy-wz); r[2]=REAL_C(2.0)*(xz+wy); r[3]=REAL_C(2.0)*(xy+wz);
    r[5]=REAL_C(2.0)*(yz-wx); r[6]=REAL_C(2.0)*(xz-wy); r[7]=REAL_C(2.0)*(yz+wx);
}
__device__ inline void cf_quat_mul(const real_t* a,const real_t* b,real_t* r) {
    // Snapshot permits any of the three arrays to alias.
    const real_t aw=a[0],ax=a[1],ay=a[2],az=a[3];
    const real_t bw=b[0],bx=b[1],by=b[2],bz=b[3];
    r[0]=aw*bw-ax*bx-ay*by-az*bz;
    r[1]=aw*bx+ax*bw+ay*bz-az*by;
    r[2]=aw*by-ax*bz+ay*bw+az*bx;
    r[3]=aw*bz+ax*by-ay*bx+az*bw;
}
__device__ inline bool cf_quat_normalize(real_t* q) {
    const real_t n=sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if(!(n==n) || n>REAL_NORM_LIMIT) return false;
    if(n<REAL_C(1e-15)) {q[0]=REAL_C(1.0);q[1]=q[2]=q[3]=REAL_C(0.0);}
    else {const real_t s=REAL_C(1.0)/n;for(int k=0;k<4;++k)q[k]*=s;}
    return true;
}
// Quaternion Jacobi preserves MuJoCo's pivot ties, stopping criterion and
// eigenvector orientation; a generic eigensolver has a different frame ABI.
__device__ inline int cf_eig3(const real_t* cov,real_t* eig,real_t* frame,
                              real_t* quat,real_t* tmp,real_t* work) {
    quat[0]=REAL_C(1.0);quat[1]=quat[2]=quat[3]=REAL_C(0.0);
    real_t off_tolerance=REAL_C(1e-12);
#if FLASH_REAL_BITS == 32
    real_t norm=REAL_C(0.0);
    for(int row=0;row<3;++row) {
        real_t sum=REAL_C(0.0);
        for(int col=0;col<3;++col)sum+=fabs(cov[3*row+col]);
        norm=fmax(norm,sum);
    }
    off_tolerance=fmax(off_tolerance,REAL_C(8.0)*REAL_EPSILON*norm);
#endif
    int iters=0;bool converged=false;
    for(int it=0;it<500;++it) {
        iters=it;cf_quat_matrix(frame,quat);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
            real_t sum=REAL_C(0.0);
            for(int k=0;k<3;++k)sum+=frame[3*k+i]*cov[3*k+j];
            work[3*i+j]=sum;
        }
        for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
            real_t sum=REAL_C(0.0);
            for(int k=0;k<3;++k)sum+=work[3*i+k]*frame[3*k+j];
            tmp[3*i+j]=sum;
        }
        eig[0]=tmp[0];eig[1]=tmp[4];eig[2]=tmp[8];
        const real_t d01=fabs(tmp[1]),d02=fabs(tmp[2]),d12=fabs(tmp[5]);
        int rk,ck,rotk;
        if(d01>d02 && d01>d12){rk=0;ck=1;rotk=2;}
        else if(d02>d12){rk=0;ck=2;rotk=1;}
        else {rk=1;ck=2;rotk=0;}
        const real_t off=tmp[3*rk+ck];
        if(fabs(off)<off_tolerance){converged=true;break;}
        const real_t tau=(tmp[4*ck]-tmp[4*rk])/(REAL_C(2.0)*off);
        const real_t t=tau>=REAL_C(0.0) ? REAL_C(1.0)/(tau+sqrt(REAL_C(1.0)+tau*tau))
                                         : -REAL_C(1.0)/(-tau+sqrt(REAL_C(1.0)+tau*tau));
        const real_t c=REAL_C(1.0)/sqrt(REAL_C(1.0)+t*t);
#if FLASH_REAL_BITS == 64
        if(c>REAL_C(1.0)-REAL_C(1e-12)){converged=true;break;}
#endif
        tmp[0]=tmp[1]=tmp[2]=tmp[3]=REAL_C(0.0);
#if FLASH_REAL_BITS == 32
        // sin(theta/2) = sin(theta)/sqrt(2*(1+cos(theta))). The original
        // subtraction loses the rotation when FP32 rounds c to exactly one.
        real_t s=fabs(t)*c/sqrt(REAL_C(2.0)*(REAL_C(1.0)+c));
#else
        real_t s=sqrt(REAL_C(0.5)-REAL_C(0.5)*c);
#endif
        if(tau>=REAL_C(0.0))s=-s;
        if(rotk==1)s=-s;
        tmp[rotk+1]=s;tmp[0]=sqrt(REAL_C(1.0)-s*s);
        if(!cf_quat_normalize(tmp))return -1;
        cf_quat_mul(quat,tmp,work);
        for(int k=0;k<4;++k)quat[k]=work[k];
        if(!cf_quat_normalize(quat))return -1;
    }
    for(int j=0;j<3;++j) {
        const int k=j%2;
        if(eig[k]<eig[k+1]) {
            const real_t v=eig[k];eig[k]=eig[k+1];eig[k+1]=v;
            tmp[0]=REAL_C(0.707106781186548);tmp[1]=tmp[2]=tmp[3]=REAL_C(0.0);
            tmp[(k+2)%3+1]=tmp[0];
            cf_quat_mul(quat,tmp,work);
            for(int l=0;l<4;++l)quat[l]=work[l];
            if(!cf_quat_normalize(quat))return -1;
        }
    }
    cf_quat_matrix(frame,quat);
    return converged ? iters : -2;
}

// Explicit world-local position view. Public/global positions retain their
// AoS ABI; F may replace its existing shared 3*ngeom tile with axis planes.
__device__ __forceinline__ real_t cf_geom_pos(const real_t* pos,int g,int axis,
        int ngeom,bool planes){return pos[planes?axis*ngeom+g:3*g+axis];}
__device__ inline void cf_bounds(int w,
    const real_t* xpos,const int* active,const int* gates,const int* collision_mode,
    const real_t* geom_rbound,const real_t* geom_margin,const int* body_geomadr,
    const int* body_geomnum,const int* geom_bodyid,real_t* aabb,real_t* frame,
    int* entered,int* count,int* status,real_t* scratch,int ngeom,int nbody,
    const real_t* world_geom_pos=nullptr,bool world_geom_pos_planes=false) {
    status[w]=0;entered[w]=0;count[w]=0;
    if(active[w]==0){status[w]=1;return;}
    if(active[w]!=1){status[w]=2;return;}
    if(gates[0]!=0 || gates[1]!=0 || gates[2]==0 || nbody<2 || collision_mode[0]==1)return;
    const real_t* pos=world_geom_pos?world_geom_pos:xpos+w*ngeom*3;
    const bool planes=world_geom_pos && world_geom_pos_planes;
    real_t center[3]={REAL_C(0.0),REAL_C(0.0),REAL_C(0.0)};int cnt=0;
    for(int g=0;g<ngeom;++g)if(geom_bodyid[g]!=0) {
        const real_t x=cf_geom_pos(pos,g,0,ngeom,planes),y=cf_geom_pos(pos,g,1,ngeom,planes),z=cf_geom_pos(pos,g,2,ngeom,planes);
        if(!(x==x && y==y && z==z)){status[w]=3;return;}
        center[0]+=x;center[1]+=y;center[2]+=z;++cnt;
    }
    if(cnt==0)return;
    for(int j=0;j<3;++j)center[j]/=real_t(cnt);
    // Jacobi's 34-double work state and 9-double frame are one producer /
    // consumer chain. Retain the public diagnostic state only at its exits.
    real_t local_s[34],local_f[9];
    for(int k=0;k<34;++k)local_s[k]=scratch[w*34+k];
    for(int k=0;k<9;++k)local_f[k]=frame[w*9+k];
    real_t* s=local_s;real_t* f=local_f;
    for(int k=0;k<9;++k)s[k]=REAL_C(0.0);
    for(int g=0;g<ngeom;++g)if(geom_bodyid[g]!=0) {
        const real_t dx=cf_geom_pos(pos,g,0,ngeom,planes)-center[0],dy=cf_geom_pos(pos,g,1,ngeom,planes)-center[1],dz=cf_geom_pos(pos,g,2,ngeom,planes)-center[2];
        s[0]+=dx*dx;s[1]+=dx*dy;s[2]+=dx*dz;s[3]+=dx*dy;s[4]+=dy*dy;
        s[5]+=dy*dz;s[6]+=dx*dz;s[7]+=dy*dz;s[8]+=dz*dz;
    }
    for(int k=0;k<9;++k)s[k]/=real_t(cnt);
    const int eigen_status=cf_eig3(s,s+27,f,s+30,s+9,s+18);
    for(int k=0;k<34;++k)scratch[w*34+k]=s[k];
    for(int k=0;k<9;++k)frame[w*9+k]=f[k];
    if(eigen_status<0){status[w]=5;return;}
    entered[w]=1;count[w]=cnt;
    real_t* bounds=aabb+w*nbody*6;
    for(int b=0;b<nbody;++b) {
        if(b==0 || body_geomnum[b]==0){for(int k=0;k<6;++k)bounds[6*b+k]=REAL_C(0.0);continue;}
        for(int i=0;i<body_geomnum[b];++i) {
            const int g=body_geomadr[b]+i;
            for(int j=0;j<3;++j) {
                const real_t c=cf_geom_pos(pos,g,0,ngeom,planes)*f[3*j]+cf_geom_pos(pos,g,1,ngeom,planes)*f[3*j+1]+cf_geom_pos(pos,g,2,ngeom,planes)*f[3*j+2];
                const real_t lo=c-geom_rbound[g]-geom_margin[g],hi=c+geom_rbound[g]+geom_margin[g];
                real_t* dst=bounds+6*b+2*j;
                if(i==0){dst[0]=lo;dst[1]=hi;}
                else {if(dst[0]>lo)dst[0]=lo;if(dst[1]<hi)dst[1]=hi;}
            }
        }
    }
}

// F calls this after its own successful position publication. The geom
// pointer is world-local shared storage; all other arenas use world strides.
// Seeding preserves the original bounds function's untouched branches.
__device__ inline void cf_bounds_prepare_cached(int w,const real_t* world_geom_pos,
    const int* position_valid,const int64_t* position_written,const int64_t* forward_epoch,const int* first_error,
    const int* gates,const int* collision_mode,const real_t* geom_rbound,const real_t* geom_margin,
    const int* body_geomadr,const int* body_geomnum,const int* geom_bodyid,
    const real_t* public_aabb,const real_t* public_frame,const real_t* public_scratch,
    real_t* cache_aabb,real_t* cache_frame,real_t* cache_scratch,
    int* cache_entered,int* cache_count,int* cache_status,int64_t* cache_epoch,int* cache_options,
    int64_t* prepare_count,
    int ngeom,int nbody,bool world_geom_pos_planes=false){
    if(position_valid[w]!=1||first_error[w]||position_written[w]!=forward_epoch[w])return;
    for(int k=0;k<nbody*6;++k)cache_aabb[w*nbody*6+k]=public_aabb[w*nbody*6+k];
    for(int k=0;k<9;++k)cache_frame[w*9+k]=public_frame[w*9+k];
    for(int k=0;k<34;++k)cache_scratch[w*34+k]=public_scratch[w*34+k];
    cf_bounds(w,world_geom_pos,position_valid,gates,collision_mode,geom_rbound,geom_margin,
        body_geomadr,body_geomnum,geom_bodyid,cache_aabb,cache_frame,cache_entered,cache_count,
        cache_status,cache_scratch,ngeom,nbody,world_geom_pos,world_geom_pos_planes);
    ++prepare_count[w];
    for(int k=0;k<3;++k)cache_options[w*4+k]=gates[k];
    cache_options[w*4+3]=collision_mode[0];cache_epoch[w]=position_written[w];
}

__device__ inline void cf_bounds_cached(int w,
    const real_t* xpos,const int* active,const int* gates,const int* collision_mode,
    const real_t* geom_rbound,const real_t* geom_margin,const int* body_geomadr,const int* body_geomnum,
    const int* geom_bodyid,real_t* aabb,real_t* frame,int* entered,int* count,int* status,real_t* scratch,
    int ngeom,int nbody,const real_t* cache_aabb,const real_t* cache_frame,const real_t* cache_scratch,
    const int* cache_entered,const int* cache_count,const int* cache_status,const int64_t* cache_epoch,
    const int* cache_options,const int* position_valid,const int64_t* position_written,
    const int64_t* forward_epoch,const int* first_error,int64_t* hit_count,int64_t* fallback_count){
    bool current=active[w]==1&&position_valid[w]==1&&!first_error[w]&&
                 position_written[w]==forward_epoch[w]&&cache_epoch[w]==position_written[w];
    for(int k=0;k<3;++k)current=current&&cache_options[w*4+k]==gates[k];
    current=current&&cache_options[w*4+3]==collision_mode[0];
    if(!current){
        ++fallback_count[w];
        cf_bounds(w,xpos,active,gates,collision_mode,geom_rbound,geom_margin,body_geomadr,body_geomnum,
            geom_bodyid,aabb,frame,entered,count,status,scratch,ngeom,nbody);return;
    }
    ++hit_count[w];
    for(int k=0;k<nbody*6;++k)aabb[w*nbody*6+k]=cache_aabb[w*nbody*6+k];
    for(int k=0;k<9;++k)frame[w*9+k]=cache_frame[w*9+k];
    for(int k=0;k<34;++k)scratch[w*34+k]=cache_scratch[w*34+k];
    entered[w]=cache_entered[w];count[w]=cache_count[w];status[w]=cache_status[w];
}

// Collective producer: only warp 0 reduces the geom cloud; the Jacobi
// dependency chain remains on its leader and body bounds are independent.
// The enclosing launch must assign one whole block to the same world.
__device__ inline void cf_bounds_block(int w,
    const real_t* xpos,const int* active,const int* gates,const int* collision_mode,
    const real_t* geom_rbound,const real_t* geom_margin,const int* body_geomadr,
    const int* body_geomnum,const int* geom_bodyid,real_t* aabb,real_t* frame,
    int* entered,int* count,int* status,real_t* scratch,int ngeom,int nbody,
    const real_t* world_geom_pos=nullptr,bool world_geom_pos_planes=false) {
    if(threadIdx.x==0){status[w]=0;entered[w]=0;count[w]=0;}
    __syncthreads();
    if(active[w]==0){if(threadIdx.x==0)status[w]=1;return;}
    if(active[w]!=1){if(threadIdx.x==0)status[w]=2;return;}
    if(gates[0]!=0 || gates[1]!=0 || gates[2]==0 || nbody<2 || collision_mode[0]==1)return;
    const real_t* pos=world_geom_pos?world_geom_pos:xpos+w*ngeom*3;
    const bool planes=world_geom_pos && world_geom_pos_planes;
    __shared__ int cloud_count, cloud_bad, eigen_result;
    __shared__ real_t center[3], covariance[9], axes[9];
    if(libero_warp::warp()==0){
        const int lane=libero_warp::lane();
        real_t xsum=REAL_C(0.0),ysum=REAL_C(0.0),zsum=REAL_C(0.0);int cnt=0;bool bad=false;
        for(int g=lane;g<ngeom;g+=32)if(geom_bodyid[g]!=0){
            const real_t x=cf_geom_pos(pos,g,0,ngeom,planes),y=cf_geom_pos(pos,g,1,ngeom,planes),z=cf_geom_pos(pos,g,2,ngeom,planes);
            bad=bad || !(x==x && y==y && z==z);
            xsum+=x;ysum+=y;zsum+=z;++cnt;
        }
        const bool any_bad=libero_warp::any(bad);
        const int total=libero_warp::sum(cnt);
        xsum=libero_warp::sum_to_leader(xsum);
        ysum=libero_warp::sum_to_leader(ysum);
        zsum=libero_warp::sum_to_leader(zsum);
        if(lane==0){
            cloud_bad=any_bad;cloud_count=total;
            if(total>0){center[0]=xsum/real_t(total);center[1]=ysum/real_t(total);center[2]=zsum/real_t(total);}
        }
    }
    __syncthreads();
    if(cloud_bad){if(threadIdx.x==0)status[w]=3;return;}
    if(cloud_count==0)return;
    if(libero_warp::warp()==0){
        real_t c[6]={REAL_C(0.0),REAL_C(0.0),REAL_C(0.0),REAL_C(0.0),REAL_C(0.0),REAL_C(0.0)};
        for(int g=libero_warp::lane();g<ngeom;g+=32)if(geom_bodyid[g]!=0){
            const real_t dx=cf_geom_pos(pos,g,0,ngeom,planes)-center[0],dy=cf_geom_pos(pos,g,1,ngeom,planes)-center[1],dz=cf_geom_pos(pos,g,2,ngeom,planes)-center[2];
            c[0]+=dx*dx;c[1]+=dx*dy;c[2]+=dx*dz;
            c[3]+=dy*dy;c[4]+=dy*dz;c[5]+=dz*dz;
        }
        for(int k=0;k<6;++k)c[k]=libero_warp::sum_to_leader(c[k]);
        if(libero_warp::lane()==0){
            const real_t denom=real_t(cloud_count);
            covariance[0]=c[0]/denom;covariance[1]=covariance[3]=c[1]/denom;
            covariance[2]=covariance[6]=c[2]/denom;covariance[4]=c[3]/denom;
            covariance[5]=covariance[7]=c[4]/denom;covariance[8]=c[5]/denom;
        }
    }
    __syncthreads();
    if(threadIdx.x==0){
        real_t local_s[34],local_f[9];
        for(int k=0;k<34;++k)local_s[k]=scratch[w*34+k];
        for(int k=0;k<9;++k)local_f[k]=frame[w*9+k];
        for(int k=0;k<9;++k)local_s[k]=covariance[k];
        eigen_result=cf_eig3(local_s,local_s+27,local_f,local_s+30,local_s+9,local_s+18);
        for(int k=0;k<34;++k)scratch[w*34+k]=local_s[k];
        for(int k=0;k<9;++k)frame[w*9+k]=axes[k]=local_f[k];
        if(eigen_result<0)status[w]=5;
        else {entered[w]=1;count[w]=cloud_count;}
    }
    __syncthreads();
    if(eigen_result<0)return;
    real_t* bounds=aabb+w*nbody*6;
    for(int task=threadIdx.x;task<nbody*3;task+=blockDim.x){
        const int b=task/3,j=task%3;
        real_t* dst=bounds+6*b+2*j;
        if(b==0 || body_geomnum[b]==0){dst[0]=dst[1]=REAL_C(0.0);continue;}
        // Three axis owners per body improve occupancy without changing the
        // ordered geom comparisons (NaN and equal bounds keep old semantics).
        for(int i=0;i<body_geomnum[b];++i){
            const int g=body_geomadr[b]+i;
            const real_t c=cf_geom_pos(pos,g,0,ngeom,planes)*axes[3*j]+cf_geom_pos(pos,g,1,ngeom,planes)*axes[3*j+1]+cf_geom_pos(pos,g,2,ngeom,planes)*axes[3*j+2];
            const real_t lo=c-geom_rbound[g]-geom_margin[g],hi=c+geom_rbound[g]+geom_margin[g];
            if(i==0){dst[0]=lo;dst[1]=hi;}
            else {if(dst[0]>lo)dst[0]=lo;if(dst[1]<hi)dst[1]=hi;}
        }
    }
    __syncthreads();
}

__device__ inline void cf_bounds_prepare_cached_block(int w,const real_t* world_geom_pos,
    const int* position_valid,const int64_t* position_written,const int64_t* forward_epoch,const int* first_error,
    const int* gates,const int* collision_mode,const real_t* geom_rbound,const real_t* geom_margin,
    const int* body_geomadr,const int* body_geomnum,const int* geom_bodyid,
    const real_t* public_aabb,const real_t* public_frame,const real_t* public_scratch,
    real_t* cache_aabb,real_t* cache_frame,real_t* cache_scratch,
    int* cache_entered,int* cache_count,int* cache_status,int64_t* cache_epoch,int* cache_options,
    int64_t* prepare_count,int ngeom,int nbody,bool world_geom_pos_planes=false){
    if(position_valid[w]!=1||first_error[w]||position_written[w]!=forward_epoch[w])return;
    if(nbody>0)libero_vec::copy_block(public_aabb+w*nbody*6,cache_aabb+w*nbody*6,(size_t)nbody*6);
    libero_vec::copy_block(public_frame+w*9,cache_frame+w*9,9);
    libero_vec::copy_block(public_scratch+w*34,cache_scratch+w*34,34);
    __syncthreads();
    cf_bounds_block(w,world_geom_pos,position_valid,gates,collision_mode,geom_rbound,geom_margin,
        body_geomadr,body_geomnum,geom_bodyid,cache_aabb,cache_frame,cache_entered,cache_count,
        cache_status,cache_scratch,ngeom,nbody,world_geom_pos,world_geom_pos_planes);
    // Even failed/skipped preparation publishes its diagnostic cache epoch.
    if(threadIdx.x==0){
        ++prepare_count[w];
        for(int k=0;k<3;++k)cache_options[w*4+k]=gates[k];
        cache_options[w*4+3]=collision_mode[0];cache_epoch[w]=position_written[w];
    }
    __syncthreads();
}

__device__ inline void cf_bounds_cached_block(int w,
    const real_t* xpos,const int* active,const int* gates,const int* collision_mode,
    const real_t* geom_rbound,const real_t* geom_margin,const int* body_geomadr,const int* body_geomnum,
    const int* geom_bodyid,real_t* aabb,real_t* frame,int* entered,int* count,int* status,real_t* scratch,
    int ngeom,int nbody,const real_t* cache_aabb,const real_t* cache_frame,const real_t* cache_scratch,
    const int* cache_entered,const int* cache_count,const int* cache_status,const int64_t* cache_epoch,
    const int* cache_options,const int* position_valid,const int64_t* position_written,
    const int64_t* forward_epoch,const int* first_error,int64_t* hit_count,int64_t* fallback_count){
    bool current=active[w]==1&&position_valid[w]==1&&!first_error[w]&&
                 position_written[w]==forward_epoch[w]&&cache_epoch[w]==position_written[w];
    for(int k=0;k<3;++k)current=current&&cache_options[w*4+k]==gates[k];
    current=current&&cache_options[w*4+3]==collision_mode[0];
    if(!current){
        if(threadIdx.x==0)++fallback_count[w];
        cf_bounds_block(w,xpos,active,gates,collision_mode,geom_rbound,geom_margin,body_geomadr,body_geomnum,
            geom_bodyid,aabb,frame,entered,count,status,scratch,ngeom,nbody);return;
    }
    if(threadIdx.x==0){
        ++hit_count[w];entered[w]=cache_entered[w];count[w]=cache_count[w];status[w]=cache_status[w];
    }
    if(nbody>0)libero_vec::copy_block(cache_aabb+w*nbody*6,aabb+w*nbody*6,(size_t)nbody*6);
    libero_vec::copy_block(cache_frame+w*9,frame+w*9,9);
    libero_vec::copy_block(cache_scratch+w*34,scratch+w*34,34);
    __syncthreads();
}

__device__ inline void cf_broadphase(int w,
    const real_t* aabb,const int* active,const int* gates,const int* collision_mode,
    const int* body_weldid,const int* body_parentid,const int* body_geomadr,
    const int* body_geomnum,const int* geom_bodyid,const int* geom_type,
    const int* geom_contype,const int* geom_conaffinity,const int* can,const int* plane,
    const int* filterparent_disabled,int* out_pair,int* out_npair,int* status,
    float* sortbuf,int* sorttag,int* activebuf,int nbody,int pair_capacity) {
    (void)geom_bodyid;(void)geom_type;
    status[w]=0;out_npair[w]=0;
    if(active[w]==0)return;
    if(active[w]!=1){status[w]=2;return;}
    if(gates[0]!=0 || gates[1]!=0 || gates[2]==0 || nbody<2 || collision_mode[0]==1)return;
    int npair=0;int* pairs=out_pair+w*pair_capacity;
    const real_t* ab=aabb+w*nbody*6;
    for(int b1=0;b1<nbody;++b1) {
        if(can[b1]==0)continue;
        if(!((b1==0 && body_geomnum[b1]>0) || (body_weldid[b1]==0 && plane[b1]!=0)))continue;
        for(int b2=0;b2<nbody;++b2) {
            if(body_weldid[b2]==0)continue;
            if(npair>=pair_capacity){status[w]=1;return;}
            pairs[npair++]=b1<b2 ? (b1<<16)+b2 : (b2<<16)+b1;
        }
    }
    int cnt=0;for(int b=1;b<nbody;++b)cnt+=body_geomnum[b];
    if(cnt==0){out_npair[w]=npair;return;}
    float* keys=sortbuf+w*2*nbody;int* tags=sorttag+w*2*nbody;
    int* live=activebuf+w*nbody;int ne=0;
    for(int b=1;b<nbody;++b)if(can[b]!=0) {
        keys[ne]=float(ab[6*b]);tags[ne++]=b;
        keys[ne]=float(ab[6*b+1]);tags[ne++]=b+65536;
    }
    for(int i=1;i<ne;++i) {
        const float key=keys[i];const int tag=tags[i];int j=i;
        while(j>0 && key<keys[j-1]){keys[j]=keys[j-1];tags[j]=tags[j-1];--j;}
        keys[j]=key;tags[j]=tag;
    }
    int nactive=0;
    for(int i=0;i<ne;++i) {
        const int tag=tags[i];
        if((tag&65536)==0) {
            const int b2=tag;
            for(int j=0;j<nactive;++j) {
                const int b1=live[j],w1=body_weldid[b1],w2=body_weldid[b2];
                const int p1=body_weldid[body_parentid[w1]],p2=body_weldid[body_parentid[w2]];
                if(w1==w2 || (filterparent_disabled[0]==0 && w1!=0 && w2!=0 && (w1==p2 || w2==p1)))continue;
                const real_t* a1=ab+6*b1;const real_t* a2=ab+6*b2;
                if(a1[2]>a2[3] || a1[3]<a2[2] || a1[4]>a2[5] || a1[5]<a2[4])continue;
                // Preserve source capacity check before the affinity filter.
                if(npair>=pair_capacity){status[w]=1;return;}
                if(body_geomnum[b1]==1 && body_geomnum[b2]==1) {
                    const int g1=body_geomadr[b1],g2=body_geomadr[b2];
                    if((geom_contype[g1]&geom_conaffinity[g2])==0 && (geom_contype[g2]&geom_conaffinity[g1])==0)continue;
                }
                pairs[npair++]=b1<b2 ? (b1<<16)+b2 : (b2<<16)+b1;
            }
            live[nactive++]=b2;
        } else {
            const int b=tag-65536;
            for(int j=0;j<nactive;++j)if(live[j]==b) {
                for(int k=j;k<nactive-1;++k)live[k]=live[k+1];--nactive;break;
            }
        }
    }
    for(int i=1;i<npair;++i) {
        const int key=pairs[i];int j=i;
        while(j>0 && key<pairs[j-1]){pairs[j]=pairs[j-1];--j;}pairs[j]=key;
    }
    out_npair[w]=npair;
}

// Warp-cooperative candidate filtering with ordered SAP control. Every lane
// of warp 0 enters; other block warps wait in the wrapper. Bounded register
// networks preserve stable equal keys. NaN events and larger lists retain the
// original insertion sort and its comparison barriers.
__device__ inline void cf_broadphase_warp(int w,
    const real_t* aabb,const int* active,const int* gates,const int* collision_mode,
    const int* body_weldid,const int* body_parentid,const int* body_geomadr,
    const int* body_geomnum,const int* geom_bodyid,const int* geom_type,
    const int* geom_contype,const int* geom_conaffinity,const int* can,const int* plane,
    const int* filterparent_disabled,int* out_pair,int* out_npair,int* status,
    float* sortbuf,int* sorttag,int* activebuf,int nbody,int pair_capacity) {
    (void)geom_bodyid;(void)geom_type;
    const int lane=libero_warp::lane();
    if(lane==0){status[w]=0;out_npair[w]=0;}
    if(active[w]==0)return;
    if(active[w]!=1){if(lane==0)status[w]=2;return;}
    if(gates[0]!=0 || gates[1]!=0 || gates[2]==0 || nbody<2 || collision_mode[0]==1)return;
    int npair=0;int* pairs=out_pair+w*pair_capacity;
    const real_t* ab=aabb+w*nbody*6;
    for(int b1=0;b1<nbody;++b1){
        if(can[b1]==0)continue;
        if(!((b1==0 && body_geomnum[b1]>0) || (body_weldid[b1]==0 && plane[b1]!=0)))continue;
        for(int base=0;base<nbody;base+=32){
            const int b2=base+lane;
            const bool take=b2<nbody && body_weldid[b2]!=0;
            const int rank=libero_warp::compact_rank(take);
            const int total=libero_warp::compact_count(take);
            const int bad=libero_warp::first_index(take && npair+rank>=pair_capacity,lane);
            if(take && lane<bad)pairs[npair+rank]=b1<b2 ? (b1<<16)+b2 : (b2<<16)+b1;
            __syncwarp(libero_warp::mask);
            if(bad!=libero_warp::no_index){if(lane==0)status[w]=1;return;}
            npair+=total;
        }
    }
    int local_count=0;
    for(int b=1+lane;b<nbody;b+=32)local_count+=body_geomnum[b];
    if(libero_warp::sum(local_count)==0){if(lane==0)out_npair[w]=npair;return;}
    float* keys=sortbuf+w*2*nbody;int* tags=sorttag+w*2*nbody;
    int* live=activebuf+w*nbody;int ne=0;
    for(int base=1;base<nbody;base+=32){
        const int b=base+lane;
        const bool take=b<nbody && can[b]!=0;
        const int rank=libero_warp::compact_rank(take);
        const int total=libero_warp::compact_count(take);
        if(take){
            const int id=ne+2*rank;
            keys[id]=float(ab[6*b]);tags[id]=b;
            keys[id+1]=float(ab[6*b+1]);tags[id+1]=b+65536;
        }
        ne+=2*total;
    }
    __syncwarp(libero_warp::mask);
    const bool events_sorted=sap_events64(keys,tags,ne);
    if(!events_sorted&&lane==0)for(int i=1;i<ne;++i){
        const float key=keys[i];const int tag=tags[i];int j=i;
        while(j>0 && key<keys[j-1]){keys[j]=keys[j-1];tags[j]=tags[j-1];--j;}
        keys[j]=key;tags[j]=tag;
    }
    __syncwarp(libero_warp::mask);
    int nactive=0;
    for(int i=0;i<ne;++i){
        const int tag=tags[i];
        if((tag&65536)==0){
            const int b2=tag,w2=body_weldid[b2];
            // The current sweep body is common to every live candidate.
            // Keep its four tested fields in registers across the active tiles.
            const real_t b2_ylo=ab[6*b2+2],b2_yhi=ab[6*b2+3];
            const real_t b2_zlo=ab[6*b2+4],b2_zhi=ab[6*b2+5];
            // Tiny live sets do not justify several warp scans. Preserve
            // the cheap ordered owner path and broadcast its result.
            if(nactive<8){
                int failed=0;
                if(lane==0)for(int j=0;j<nactive;++j){
                    const int b1=live[j],w1=body_weldid[b1];
                    const int p1=body_weldid[body_parentid[w1]],p2=body_weldid[body_parentid[w2]];
                    if(w1==w2 || (filterparent_disabled[0]==0 && w1!=0 && w2!=0 && (w1==p2 || w2==p1)))continue;
                    const real_t* a1=ab+6*b1;
                    if(a1[2]>b2_yhi || a1[3]<b2_ylo || a1[4]>b2_zhi || a1[5]<b2_zlo)continue;
                    if(npair>=pair_capacity){status[w]=1;failed=1;break;}
                    if(body_geomnum[b1]==1 && body_geomnum[b2]==1){
                        const int g1=body_geomadr[b1],g2=body_geomadr[b2];
                        if((geom_contype[g1]&geom_conaffinity[g2])==0 && (geom_contype[g2]&geom_conaffinity[g1])==0)continue;
                    }
                    pairs[npair++]=b1<b2 ? (b1<<16)+b2 : (b2<<16)+b1;
                }
                npair=__shfl_sync(libero_warp::mask,npair,0);
                failed=__shfl_sync(libero_warp::mask,failed,0);
                if(failed)return;
            }else for(int base=0;base<nactive;base+=32){
                const int j=base+lane;
                int b1=0;bool overlap=false,emit=false;
                if(j<nactive){
                    b1=live[j];
                    const int w1=body_weldid[b1];
                    const int p1=body_weldid[body_parentid[w1]],p2=body_weldid[body_parentid[w2]];
                    if(!(w1==w2 || (filterparent_disabled[0]==0 && w1!=0 && w2!=0 && (w1==p2 || w2==p1)))){
                        const real_t* a1=ab+6*b1;
                        overlap=!(a1[2]>b2_yhi || a1[3]<b2_ylo || a1[4]>b2_zhi || a1[5]<b2_zlo);
                        emit=overlap;
                        if(overlap && body_geomnum[b1]==1 && body_geomnum[b2]==1){
                            const int g1=body_geomadr[b1],g2=body_geomadr[b2];
                            emit=!((geom_contype[g1]&geom_conaffinity[g2])==0 &&
                                   (geom_contype[g2]&geom_conaffinity[g1])==0);
                        }
                    }
                }
                const int rank=libero_warp::compact_rank(emit);
                const int total=libero_warp::compact_count(emit);
                // Test capacity using the output prefix *before* this lane.
                // An overlap rejected by affinity still triggers a full-cap
                // error, exactly as the ordered source does.
                const int bad=libero_warp::first_index(overlap && npair+rank>=pair_capacity,lane);
                if(emit && lane<bad)pairs[npair+rank]=b1<b2 ? (b1<<16)+b2 : (b2<<16)+b1;
                __syncwarp(libero_warp::mask);
                if(bad!=libero_warp::no_index){if(lane==0)status[w]=1;return;}
                npair+=total;
            }
            if(lane==0)live[nactive]=b2;
            ++nactive;
        }else{
            if(lane==0){
                const int b=tag-65536;
                for(int j=0;j<nactive;++j)if(live[j]==b){
                    for(int k=j;k<nactive-1;++k)live[k]=live[k+1];
                    --nactive;break;
                }
            }
            nactive=__shfl_sync(libero_warp::mask,nactive,0);
        }
        __syncwarp(libero_warp::mask);
    }
    const bool pairs_sorted=sap_pairs128(pairs,npair);
    if(!pairs_sorted&&lane==0){
        for(int i=1;i<npair;++i){
            const int key=pairs[i];int j=i;
            while(j>0 && key<pairs[j-1]){pairs[j]=pairs[j-1];--j;}
            pairs[j]=key;
        }
    }
    if(lane==0)out_npair[w]=npair;
}

__device__ inline void cf_broadphase_block(int w,
    const real_t* aabb,const int* active,const int* gates,const int* collision_mode,
    const int* body_weldid,const int* body_parentid,const int* body_geomadr,
    const int* body_geomnum,const int* geom_bodyid,const int* geom_type,
    const int* geom_contype,const int* geom_conaffinity,const int* can,const int* plane,
    const int* filterparent_disabled,int* out_pair,int* out_npair,int* status,
    float* sortbuf,int* sorttag,int* activebuf,int nbody,int pair_capacity) {
    if(libero_warp::warp()==0)cf_broadphase_warp(w,aabb,active,gates,collision_mode,body_weldid,
        body_parentid,body_geomadr,body_geomnum,geom_bodyid,geom_type,geom_contype,geom_conaffinity,
        can,plane,filterparent_disabled,out_pair,out_npair,status,sortbuf,sorttag,activebuf,nbody,pair_capacity);
    __syncthreads();
}

__device__ inline void cf_stagebuild(int w,
    const int* pair,const int* npair_arr,const int* active,const int* body_geomnum,
    const int* pair_geom1,const int* pair_geom2,const int* pair_signature,
    const int* exclude_signature,const int* midphase_disabled,const int* gates,
    const int* collision_mode,int* out_stage,int* out_stage_count,int* out_stage_route,
    int* out_merged,int* out_seg_count,int* out_seg_pair_start,int* out_seg_pair_end,
    int* out_seg_geom_stage,int* out_trailing_start,int* out_trailing_end,
    int* out_pairadr,int* status,int* first_item,int nbody,int pair_capacity,
    int stage_slots,int npair_static,int nexclude) {
    (void)pair_geom1;(void)pair_geom2;
    status[w]=0;out_stage_count[w]=0;out_seg_count[w]=0;out_pairadr[w]=0;first_item[w]=-1;
    if(active[w]==0)return;
    if(active[w]!=1){status[w]=1;return;}
    if(gates[0]!=0 || gates[1]!=0 || gates[2]==0 || nbody<2) {
        out_trailing_start[w]=0;out_trailing_end[w]=0;return;
    }
    if(collision_mode[0]==1){out_trailing_start[w]=0;out_trailing_end[w]=npair_static;return;}
    const int n=npair_arr[w];
    if(n<0 || n>pair_capacity){status[w]=2;return;}
    int scount=0,segcount=0,last_signature=-1,pairadr=0,exadr=0;
    const int mode=collision_mode[0];
    for(int i=0;i<n;++i) {
        const int packed=pair[w*pair_capacity+i];
        const int b1=(packed>>16)&65535,b2=packed&65535;
        const int signature=((b1+1)<<16)+(b2+1);
        if(signature==last_signature)continue;
        last_signature=signature;int merged=0;const int start=pairadr;
        if(npair_static!=0 && mode==0)while(pairadr<npair_static && pair_signature[pairadr]<=signature) {
            if(pair_signature[pairadr]==signature)merged=1;++pairadr;
        }
        while(exadr<nexclude && exclude_signature[exadr]<signature)++exadr;
        const bool skip=exadr<nexclude && exclude_signature[exadr]==signature;
        const bool geom=!skip && body_geomnum[b1]!=0 && body_geomnum[b2]!=0;
        if(pairadr>start || geom) {
            if(segcount>=stage_slots){status[w]=3;first_item[w]=segcount;return;}
            const int seg=w*stage_slots+segcount;
            out_seg_pair_start[seg]=start;out_seg_pair_end[seg]=pairadr;
            if(geom) {
                const int route=midphase_disabled[0]==0 && body_geomnum[b1]*body_geomnum[b2]>1 ? 1 : 2;
                const int id=w*stage_slots+scount;int* s=out_stage+6*id;
                s[0]=b1;s[1]=b2;s[2]=merged;s[3]=start;s[4]=pairadr;s[5]=route;
                out_stage_route[id]=route;out_merged[id]=merged;
                out_seg_geom_stage[seg]=scount++;
            }else out_seg_geom_stage[seg]=-1;
            ++segcount;
        }
    }
    out_trailing_start[w]=pairadr;
    out_trailing_end[w]=(npair_static!=0 && mode==0)?npair_static:pairadr;
    out_pairadr[w]=pairadr;out_stage_count[w]=scount;out_seg_count[w]=segcount;
}
__device__ inline int cf_bound_upper_index(const int* keys,int n,int key){
    int lo=0,hi=n;
    while(lo<hi){const int mid=lo+(hi-lo)/2;if(keys[mid]<=key)lo=mid+1;else hi=mid;}
    return lo;
}
__device__ inline int cf_bound_lower_index(const int* keys,int n,int key){
    int lo=0,hi=n;
    while(lo<hi){const int mid=lo+(hi-lo)/2;if(keys[mid]<key)lo=mid+1;else hi=mid;}
    return lo;
}

// Large ordered lists use independent static-pair/exclusion index queries
// and stable scans. Short lists and any nonmonotonic standalone inputs retain
// the original cursor algorithm, including its exact partial-write behavior.
__device__ inline void cf_stagebuild_warp(int w,
    const int* pair,const int* npair_arr,const int* active,const int* body_geomnum,
    const int* pair_geom1,const int* pair_geom2,const int* pair_signature,
    const int* exclude_signature,const int* midphase_disabled,const int* gates,
    const int* collision_mode,int* out_stage,int* out_stage_count,int* out_stage_route,
    int* out_merged,int* out_seg_count,int* out_seg_pair_start,int* out_seg_pair_end,
    int* out_seg_geom_stage,int* out_trailing_start,int* out_trailing_end,
    int* out_pairadr,int* status,int* first_item,int nbody,int pair_capacity,
    int stage_slots,int npair_static,int nexclude){
    const int lane=libero_warp::lane(),n=npair_arr[w];
    bool fallback=active[w]!=1 || gates[0]!=0 || gates[1]!=0 || gates[2]==0 ||
                  nbody<2 || collision_mode[0]==1 || n<32 || n>pair_capacity;
    if(!fallback){
        bool unordered=false;
        const int* p=pair+w*pair_capacity;
        for(int i=1+lane;i<n;i+=32)unordered=unordered || p[i]<p[i-1];
        if(collision_mode[0]==0)for(int i=1+lane;i<npair_static;i+=32)
            unordered=unordered || pair_signature[i]<pair_signature[i-1];
        for(int i=1+lane;i<nexclude;i+=32)
            unordered=unordered || exclude_signature[i]<exclude_signature[i-1];
        fallback=libero_warp::any(unordered);
    }
    if(fallback){
        if(lane==0)cf_stagebuild(w,pair,npair_arr,active,body_geomnum,pair_geom1,pair_geom2,
            pair_signature,exclude_signature,midphase_disabled,gates,collision_mode,out_stage,
            out_stage_count,out_stage_route,out_merged,out_seg_count,out_seg_pair_start,
            out_seg_pair_end,out_seg_geom_stage,out_trailing_start,out_trailing_end,out_pairadr,
            status,first_item,nbody,pair_capacity,stage_slots,npair_static,nexclude);
        __syncwarp(libero_warp::mask);return;
    }
    if(lane==0){status[w]=0;out_stage_count[w]=0;out_seg_count[w]=0;out_pairadr[w]=0;first_item[w]=-1;}
    const int* p=pair+w*pair_capacity;
    const bool explicit_pairs=npair_static!=0 && collision_mode[0]==0;
    int scount=0,segcount=0;
    for(int base=0;base<n;base+=32){
        const int i=base+lane;
        int b1=0,b2=0,start=0,end=0,merged=0;
        bool geom=false,take=false;
        if(i<n && (i==0 || p[i]!=p[i-1])){
            const int packed=p[i];
            b1=(packed>>16)&65535;b2=packed&65535;
            const int signature=((b1+1)<<16)+(b2+1);
            if(explicit_pairs){
                end=cf_bound_upper_index(pair_signature,npair_static,signature);
                if(i>0){
                    const int prev=p[i-1];
                    const int prevsig=((((prev>>16)&65535)+1)<<16)+((prev&65535)+1);
                    start=cf_bound_upper_index(pair_signature,npair_static,prevsig);
                }
                merged=end>start && pair_signature[end-1]==signature;
            }
            const int ex=cf_bound_lower_index(exclude_signature,nexclude,signature);
            const bool skip=ex<nexclude && exclude_signature[ex]==signature;
            geom=!skip && body_geomnum[b1]!=0 && body_geomnum[b2]!=0;
            take=end>start || geom;
        }
        const int seg_rank=libero_warp::compact_rank(take);
        const int stage_rank=libero_warp::compact_rank(take && geom);
        const int nseg=libero_warp::compact_count(take);
        const int nstage=libero_warp::compact_count(take && geom);
        const int bad=libero_warp::first_index(take && segcount+seg_rank>=stage_slots,lane);
        if(take && lane<bad){
            const int seg=w*stage_slots+segcount+seg_rank;
            out_seg_pair_start[seg]=start;out_seg_pair_end[seg]=end;
            if(geom){
                const int route=midphase_disabled[0]==0 && body_geomnum[b1]*body_geomnum[b2]>1 ? 1 : 2;
                const int stage=scount+stage_rank,id=w*stage_slots+stage;
                int* s=out_stage+6*id;
                s[0]=b1;s[1]=b2;s[2]=merged;s[3]=start;s[4]=end;s[5]=route;
                out_stage_route[id]=route;out_merged[id]=merged;out_seg_geom_stage[seg]=stage;
            }else out_seg_geom_stage[seg]=-1;
        }
        __syncwarp(libero_warp::mask);
        if(bad!=libero_warp::no_index){
            const int failed=segcount+__shfl_sync(libero_warp::mask,seg_rank,bad);
            if(lane==0){status[w]=3;first_item[w]=failed;}
            return;
        }
        scount+=nstage;segcount+=nseg;
    }
    if(lane==0){
        const int packed=p[n-1];
        const int signature=((((packed>>16)&65535)+1)<<16)+((packed&65535)+1);
        const int pairadr=explicit_pairs?cf_bound_upper_index(pair_signature,npair_static,signature):0;
        out_trailing_start[w]=pairadr;out_trailing_end[w]=explicit_pairs?npair_static:pairadr;
        out_pairadr[w]=pairadr;out_stage_count[w]=scount;out_seg_count[w]=segcount;
    }
}

__device__ inline void cf_stagebuild_block(int w,
    const int* pair,const int* npair_arr,const int* active,const int* body_geomnum,
    const int* pair_geom1,const int* pair_geom2,const int* pair_signature,
    const int* exclude_signature,const int* midphase_disabled,const int* gates,
    const int* collision_mode,int* out_stage,int* out_stage_count,int* out_stage_route,
    int* out_merged,int* out_seg_count,int* out_seg_pair_start,int* out_seg_pair_end,
    int* out_seg_geom_stage,int* out_trailing_start,int* out_trailing_end,
    int* out_pairadr,int* status,int* first_item,int nbody,int pair_capacity,
    int stage_slots,int npair_static,int nexclude){
    if(libero_warp::warp()==0)cf_stagebuild_warp(w,pair,npair_arr,active,body_geomnum,pair_geom1,
        pair_geom2,pair_signature,exclude_signature,midphase_disabled,gates,collision_mode,
        out_stage,out_stage_count,out_stage_route,out_merged,out_seg_count,out_seg_pair_start,
        out_seg_pair_end,out_seg_geom_stage,out_trailing_start,out_trailing_end,out_pairadr,
        status,first_item,nbody,pair_capacity,stage_slots,npair_static,nexclude);
    __syncthreads();
}
} // namespace collision_cuda
