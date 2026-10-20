#include "precision.cuh"
#include <cublasdx.hpp>
#include "../common/vector.cuh"

#if D_NU > 0
// Compile the descriptor from cold model dimensions. moment is NU x NV;
// A = moment^T, B = force, C = generalized actuator force.
using DActuationBLAS = decltype(cublasdx::Size<D_NV,1,D_NU>() +
    cublasdx::Precision<real_t>() + cublasdx::Type<cublasdx::type::real>() +
    cublasdx::Function<cublasdx::function::MM>() +
    cublasdx::Arrangement<cublasdx::col_major,cublasdx::col_major,cublasdx::col_major>() +
    cublasdx::Alignment<sizeof(real_t),sizeof(real_t),sizeof(real_t)>() + cublasdx::Block() +
    cublasdx::BlockDim<D_THREADS>() + cublasdx::SM<D_SM>());

__device__ void d_actuation_gemv_dx(const real_t* moment,const real_t* force,real_t* output) {
    using B=DActuationBLAS;
    __shared__ __align__(16) real_t a[B::a_size];
    __shared__ __align__(16) real_t b[B::b_size];
    __shared__ __align__(16) real_t c[B::c_size];
    auto at=cublasdx::make_tensor(a,B::get_layout_smem_a());
    auto bt=cublasdx::make_tensor(b,B::get_layout_smem_b());
    auto ct=cublasdx::make_tensor(c,B::get_layout_smem_c());
    // Admission mask preserves zero-skip semantics: no access to an ignored
    // moment row, including rows containing NaN or infinity.
    // A packet stays within one admitted actuator row, including odd NV.
    constexpr int row_packets=(D_NV+1)/2;
    for(int z=threadIdx.x;z<row_packets*D_NU;z+=blockDim.x) {
        const int i=z/row_packets,j=2*(z%row_packets);
        real2_t values=make_real2(REAL_C(0.0),REAL_C(0.0));
        if(force[i]!=REAL_C(0.0)) {
            if(j+1<D_NV)values=libero_vec::load_pair(moment+i*D_NV+j);
            else values.x=moment[i*D_NV+j];
        }
        // Tensor layout is library-owned; only the compact source is paired.
        at(j,i)=values.x;if(j+1<D_NV)at(j+1,i)=values.y;
    }
    for(int i=2*threadIdx.x;i<D_NU;i+=2*blockDim.x) {
        if(i+1<D_NU) {
            const real2_t values=libero_vec::load_pair(force+i);
            bt(i,0)=values.x;bt(i+1,0)=values.y;
        } else bt(i,0)=force[i];
    }
    for(int j=threadIdx.x;j<D_NV;j+=blockDim.x) ct(j,0)=REAL_C(0.0);
    __syncthreads();
    B().execute(REAL_C(1.0),at,bt,REAL_C(0.0),ct);
    __syncthreads();
    for(int j=2*threadIdx.x;j<D_NV;j+=2*blockDim.x) {
        if(j+1<D_NV)libero_vec::store_pair(output+j,ct(j,0),ct(j+1,0));
        else output[j]=ct(j,0);
    }
    __syncthreads();
}
#else
// A model without actuators has the empty contraction, not a zero-size GEMM.
__device__ void d_actuation_gemv_dx(const real_t*,const real_t*,real_t* output) {
    libero_vec::fill_block(output,D_NV,REAL_C(0.0));
    __syncthreads();
}
#endif

#ifdef D_SCALAR_SUPPORT
__device__ void d_actuation_support_or_dx(const real_t*moment,const real_t*force,real_t*output) {
    const int support[D_NU>0?D_NU:1]=D_SCALAR_SUPPORT;
    bool bad=false;
    for(int a=threadIdx.x;a<D_NU;a+=blockDim.x) {
        bad|=!isfinite(force[a]) || fabs(force[a])>REAL_SAFE_MAGNITUDE;
        if(force[a]!=REAL_C(0.0)) {
            const real_t value=moment[a*D_NV+support[a]];
            // A different dense reduction schedule can change overflow/NaN
            // diagnostics. Keep those extreme admitted rows on original Dx.
            bad|=!isfinite(value) || fabs(value)>REAL_SAFE_MAGNITUDE;
        }
    }
    for(int z=threadIdx.x;z<D_NU*D_NV;z+=blockDim.x) {
        const int a=z/D_NV,j=z%D_NV;
        if(force[a]!=REAL_C(0.0) && j!=support[a])bad|=real_bits(moment[z])!=0;
    }
    if(__syncthreads_or(bad)) {d_actuation_gemv_dx(moment,force,output);return;}
    for(int j=threadIdx.x;j<D_NV;j+=blockDim.x) {
        real_t value=REAL_C(0.0);
        for(int a=0;a<D_NU;++a)if(force[a]!=REAL_C(0.0) && support[a]==j)
            value+=moment[a*D_NV+j]*force[a];
        output[j]=value+REAL_C(0.0);
    }
    __syncthreads();
}
#endif
