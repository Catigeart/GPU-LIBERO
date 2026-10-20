#include "precision.cuh"
#pragma once
namespace collision_cuda {
__device__ void cf_commit_contacts(int w,int* active,int* error,int* ncon,
        real_t* candidate_f,int* candidate_i,real_t* public_f,int* public_i,int capacity){
    if(active[w]!=1||error[w])return;
    const int n=ncon[w];if(n<0||n>capacity)return;
    const long long base=1LL*w*capacity;
    libero_vec::copy_block(candidate_f+base*65,public_f+base*65,size_t(n)*65);
    libero_vec::copy_block(candidate_i+base*5,public_i+base*5,size_t(n)*5);
}

__device__ inline void cf_scatter_f(int j,int t,real_t value,real_t* dist,real_t* pos,
        real_t* frame,real_t* margin,real_t* friction,real_t* solref,
        real_t* solreffriction,real_t* solimp){
    if(j==0)dist[t]=value;
    else if(j<4)pos[t*3+j-1]=value;
    else if(j<13)frame[t*9+j-4]=value;
    else if(j==13)margin[t]=value;
    else if(j<19)friction[t*5+j-14]=value;
    else if(j<21)solref[t*2+j-19]=value;
    else if(j<23)solreffriction[t*2+j-21]=value;
    else if(j<28)solimp[t*5+j-23]=value;
}
__device__ inline void cf_scatter_i(int j,int t,int value,int* dim,int* geom1,int* geom2,int* exclude){
    if(j==0)dim[t]=value;else if(j==1)geom1[t]=value;
    else if(j==2)geom2[t]=value;else if(j==3)exclude[t]=value;
}

// Candidate contacts stay private until all C stages have succeeded. The
// final scatter publishes C's AoS and S's SoA from the same guarded wide loads.
// S admission remains a separate decision and never rolls back a valid C.
__device__ void cf_commit_contacts_transport(int w,int* active,int* valid,int* error,
        int* cerror,int* cactive,int* ncon,real_t* candidate_f,int* candidate_i,
        real_t* public_f,int* public_i,int* target_active,int* target_ncon,
        real_t* dist,real_t* pos,real_t* frame,real_t* margin,real_t* friction,
        real_t* solref,real_t* solreffriction,real_t* solimp,
        int* dim,int* geom1,int* geom2,int* exclude,int source_capacity,int target_capacity){
    if(threadIdx.x==0)target_active[w]=0;
    __syncthreads();
    const int n=ncon[w];
    const bool c_ok=cactive[w]==1&&!cerror[w]&&n>=0&&n<=source_capacity;
    const bool target_ok=c_ok&&active[w]==1&&valid[w]==1&&!error[w]&&n<=target_capacity;
    if(!c_ok)return;
    // Consecutive lanes own consecutive fields of the retained AoS. The
    // target scatter reuses these loads without a second full-contact mirror.
    const long long base=1LL*w*source_capacity;
    // Complete C spans can pair across record boundaries: both records are
    // committed. Each loaded component scatters through its own field mapping.
    const long long nf=1LL*n*65;
    for(long long unit=threadIdx.x;unit<(nf+1)/2;unit+=blockDim.x){
        const long long field=unit*2;
        real2_t v;
        if(field+1<nf){v=libero_vec::load_pair(candidate_f+base*65+field);
            libero_vec::store_pair(public_f+base*65+field,v.x,v.y);}
        else{v.x=candidate_f[base*65+field];public_f[base*65+field]=v.x;}
        if(target_ok)for(int q=0;q<2&&field+q<nf;++q){
            const long long f=field+q;
            cf_scatter_f(f%65,w*target_capacity+f/65,q?v.y:v.x,
                dist,pos,frame,margin,friction,solref,solreffriction,solimp);
        }
    }
    const long long ni=1LL*n*5;
    for(long long unit=threadIdx.x;unit<(ni+3)/4;unit+=blockDim.x){
        const long long field=unit*4;
        const int* src=candidate_i+base*5+field;int* dst=public_i+base*5+field;
        if(field+3<ni){
            int4 v;
            if(libero_vec::aligned16(src))v=*reinterpret_cast<const int4*>(src);
            else{v.x=src[0];v.y=src[1];v.z=src[2];v.w=src[3];}
            if(libero_vec::aligned16(dst))*reinterpret_cast<int4*>(dst)=v;
            else{dst[0]=v.x;dst[1]=v.y;dst[2]=v.z;dst[3]=v.w;}
            if(target_ok){const int values[4]={v.x,v.y,v.z,v.w};
                for(int q=0;q<4;++q)cf_scatter_i((field+q)%5,w*target_capacity+(field+q)/5,
                    values[q],dim,geom1,geom2,exclude);}
        }else for(int q=0;field+q<ni;++q){const int value=src[q];dst[q]=value;
            if(target_ok)cf_scatter_i((field+q)%5,w*target_capacity+(field+q)/5,
                value,dim,geom1,geom2,exclude);}
    }
    __syncthreads();
    if(threadIdx.x==0&&target_ok){target_ncon[w]=n;target_active[w]=1;}
}

// Live candidates contain only the 28 initialized solver fields. Preserve
// all 65 public doubles, including mu/H zeros, at transactional commit.
__device__ void cf_commit_contacts_live(int w,int* active,int* error,int* ncon,
        real_t* candidate_f,int* candidate_i,real_t* public_f,int* public_i,int capacity){
    if(active[w]!=1||error[w])return;
    const int n=ncon[w];if(n<0||n>capacity)return;
    const long long base=1LL*w*capacity;
    for(long long field=threadIdx.x;field<1LL*n*65;field+=blockDim.x){
        const int k=field%65;const long long row=base+field/65;
        public_f[row*65+k]=k<28?candidate_f[row*28+k]:REAL_C(0.0);
    }
    libero_vec::copy_block(candidate_i+base*5,public_i+base*5,size_t(n)*5);
}

__device__ void cf_commit_contacts_transport_live(int w,int* active,int* valid,int* error,
        int* cerror,int* cactive,int* ncon,real_t* candidate_f,int* candidate_i,
        real_t* public_f,int* public_i,int* target_active,int* target_ncon,
        real_t* dist,real_t* pos,real_t* frame,real_t* margin,real_t* friction,
        real_t* solref,real_t* solreffriction,real_t* solimp,
        int* dim,int* geom1,int* geom2,int* exclude,int source_capacity,int target_capacity){
    if(threadIdx.x==0)target_active[w]=0;
    __syncthreads();
    const int n=ncon[w];
    const bool c_ok=cactive[w]==1&&!cerror[w]&&n>=0&&n<=source_capacity;
    const bool target_ok=c_ok&&active[w]==1&&valid[w]==1&&!error[w]&&n<=target_capacity;
    if(!c_ok)return;
    const long long base=1LL*w*source_capacity;
    for(long long field=threadIdx.x;field<1LL*n*65;field+=blockDim.x){
        const int k=field%65;const long long row=base+field/65;
        const real_t value=k<28?candidate_f[row*28+k]:REAL_C(0.0);
        public_f[row*65+k]=value;
        if(target_ok&&k<28)cf_scatter_f(k,w*target_capacity+field/65,value,
            dist,pos,frame,margin,friction,solref,solreffriction,solimp);
    }
    for(long long field=threadIdx.x;field<1LL*n*5;field+=blockDim.x){
        const int value=candidate_i[base*5+field];public_i[base*5+field]=value;
        if(target_ok)cf_scatter_i(field%5,w*target_capacity+field/5,value,dim,geom1,geom2,exclude);
    }
    __syncthreads();
    if(threadIdx.x==0&&target_ok){target_ncon[w]=n;target_active[w]=1;}
}

__device__ void cf_transport(int w,int* active,int* valid,int* error,int* cerror,int* cactive,int* ncon,
        real_t* cf,int* ci,int* target_active,int* target_ncon,real_t* dist,real_t* pos,real_t* frame,
        real_t* margin,real_t* friction,real_t* solref,real_t* solreffriction,real_t* solimp,
        int* dim,int* geom1,int* geom2,int* exclude,int source_capacity,int target_capacity){
    if(threadIdx.x==0)target_active[w]=0;
    __syncthreads();
    int n=ncon[w];
    if(active[w]!=1||valid[w]!=1||error[w]||cactive[w]!=1||cerror[w]||n<0||n>source_capacity||n>target_capacity)return;
    for(int k=threadIdx.x;k<n;k+=blockDim.x){
        int a=(w*source_capacity+k)*65,b=(w*source_capacity+k)*5,t=w*target_capacity+k;
        for(int j=0;j<28;j+=2){const real2_t v=libero_vec::load_pair(cf+a+j);
            cf_scatter_f(j,t,v.x,dist,pos,frame,margin,friction,solref,solreffriction,solimp);
            cf_scatter_f(j+1,t,v.y,dist,pos,frame,margin,friction,solref,solreffriction,solimp);}
        const int* src=ci+b;
        if(libero_vec::aligned16(src)){const int4 v=*reinterpret_cast<const int4*>(src);
            dim[t]=v.x;geom1[t]=v.y;geom2[t]=v.z;exclude[t]=v.w;}
        else{dim[t]=src[0];geom1[t]=src[1];geom2[t]=src[2];exclude[t]=src[3];}
    }
    __syncthreads();
    if(threadIdx.x==0){target_ncon[w]=n;target_active[w]=1;}
}
}
