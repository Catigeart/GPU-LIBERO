#include "precision.cuh"
#pragma once

// MuJoCo 2.3.7 contact parameter completion. One block owns one world;
// each lane independently completes contacts. Input tables are immutable.
namespace collision_cuda {

// All contacts emitted by a call share these parameters. Geometry-dependent
// mixing belongs to the call, while distance, position and frame belong to
// each contact. The compatible path still materializes every public row.
struct PairParams {
    real_t f[26];
    int flags[5];
    bool route_valid;
};

__device__ __forceinline__ real_t cf_param_max(real_t a, real_t b) {
    return a >= b ? a : b;
}
__device__ __forceinline__ real_t cf_param_min(real_t a, real_t b) {
    return a <= b ? a : b;
}

__device__ __forceinline__ PairParams cf_pair_params(
    int g1,int g2,int pair,int user,real_t user_margin,
    const int* priority,const int* condim,const real_t* geom_margin,const real_t* geom_gap,
    const real_t* geom_friction,const real_t* geom_solmix,const real_t* geom_solref,const real_t* geom_solimp,
    const real_t* pair_margin,const real_t* pair_gap,const int* pair_dim,const real_t* pair_friction,
    const real_t* pair_solref,const real_t* pair_solreffriction,const real_t* pair_solimp,
    const real_t* o_margin,const real_t* o_solref,const real_t* o_solimp,const int* override_enabled,
    int ngeom,int npair) {
    PairParams p;
    libero_vec::fill_serial(p.f,26,REAL_C(0.0));
    libero_vec::fill_serial(p.flags,4,-1);
    p.flags[4]=0;p.route_valid=false;
    if(g1<0||g2<0||g1>=ngeom||g2>=ngeom){p.flags[4]=2;return p;}
    if(pair< -1||pair>=npair||(user!=0&&user!=1)){p.flags[4]=3;return p;}
    p.route_valid=true;
    const bool paired=pair>=0,same_priority=priority[g1]==priority[g2];
    const int dominant=priority[g1]>priority[g2]?g1:g2;
    real_t margin=paired?pair_margin[pair]:cf_param_max(geom_margin[g1],geom_margin[g2]);
    const real_t gap=paired?pair_gap[pair]:cf_param_max(geom_gap[g1],geom_gap[g2]);
    const int dim=paired?pair_dim[pair]:(same_priority?(condim[g1]>condim[g2]?condim[g1]:condim[g2]):condim[dominant]);
    if(user)margin=user_margin;else if(override_enabled[0])margin=o_margin[0];
    p.f[10]=margin;p.f[11]=margin-gap;p.flags[0]=dim;p.flags[1]=g1;p.flags[2]=g2;
    if(dim<1||dim>6){p.flags[4]=4;return p;}
    if(paired){
        {const real_t* src=pair_friction+pair*5;real_t* dst=p.f+12;for(int k=0;k<4;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
        {const real_t* src=pair_solimp+pair*5;real_t* dst=p.f+21;for(int k=0;k<4;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
        {const real_t* src=pair_solref+pair*2;real_t* dst=p.f+17;for(int k=0;k<2;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
        if(pair_solreffriction[pair*2]!=REAL_C(0.0)||pair_solreffriction[pair*2+1]!=REAL_C(0.0))
            {const real_t* src=pair_solreffriction+pair*2;real_t* dst=p.f+19;for(int k=0;k<2;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
    }else{
        if(!same_priority){
            {const real2_t v=libero_vec::load_pair(geom_friction+dominant*3);p.f[12]=v.x;p.f[14]=v.y;p.f[16]=geom_friction[dominant*3+2];}
            {const real_t* src=geom_solref+dominant*2;real_t* dst=p.f+17;for(int k=0;k<2;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
            {const real_t* src=geom_solimp+dominant*5;real_t* dst=p.f+21;for(int k=0;k<4;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
        }else{
            for(int k=0;k<3;++k)p.f[12+2*k]=cf_param_max(geom_friction[g1*3+k],geom_friction[g2*3+k]);
            const real_t s1=geom_solmix[g1],s2=geom_solmix[g2];real_t mix;
            if(s1>=REAL_C(1e-15)&&s2>=REAL_C(1e-15))mix=s1/(s1+s2);
            else if(s1<REAL_C(1e-15)&&s2<REAL_C(1e-15))mix=REAL_C(0.5);
            else if(s1<REAL_C(1e-15))mix=REAL_C(0.0);else mix=REAL_C(1.0);
            if(geom_solref[g1*2]>REAL_C(0.0)&&geom_solref[g2*2]>REAL_C(0.0))
                for(int k=0;k<2;++k)p.f[17+k]=mix*geom_solref[g1*2+k]+(REAL_C(1.0)-mix)*geom_solref[g2*2+k];
            else for(int k=0;k<2;++k)p.f[17+k]=cf_param_min(geom_solref[g1*2+k],geom_solref[g2*2+k]);
            for(int k=0;k<5;++k)p.f[21+k]=geom_solimp[g1*5+k]*mix+geom_solimp[g2*5+k]*(REAL_C(1.0)-mix);
        }
        p.f[13]=p.f[12];p.f[15]=p.f[16];
    }
    for(int k=0;k<5;++k)p.f[12+k]=cf_param_max(REAL_C(1e-5),p.f[12+k]);
    if(override_enabled[0]){
        {const real2_t v=libero_vec::load_pair(o_solref);p.f[17]=p.f[19]=v.x;p.f[18]=p.f[20]=v.y;}
        {const real_t* src=o_solimp;real_t* dst=p.f+21;for(int k=0;k<4;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
    }
    return p;
}

// Cold model-owned table, shared by every world. Use exactly the device
// arithmetic of the uncached path, including override and direct solref.
__device__ void cf_pair_params_table(int w,
    const int* priority,const int* condim,const real_t* geom_margin,const real_t* geom_gap,
    const real_t* geom_friction,const real_t* geom_solmix,const real_t* geom_solref,const real_t* geom_solimp,
    const real_t* pair_margin,const real_t* pair_gap,const int* pair_dim,const real_t* pair_friction,
    const real_t* pair_solref,const real_t* pair_solreffriction,const real_t* pair_solimp,
    const real_t* o_margin,const real_t* o_solref,const real_t* o_solimp,const int* override_enabled,
    int ngeom,int npair,real_t* table_f,int* table_i){
    for(int entry=threadIdx.x;entry<ngeom*ngeom;entry+=blockDim.x){
        const PairParams p=cf_pair_params(entry/ngeom,entry%ngeom,-1,0,REAL_C(0.0),
            priority,condim,geom_margin,geom_gap,geom_friction,geom_solmix,geom_solref,geom_solimp,
            pair_margin,pair_gap,pair_dim,pair_friction,pair_solref,pair_solreffriction,pair_solimp,
            o_margin,o_solref,o_solimp,override_enabled,ngeom,npair);
        libero_vec::copy_serial(p.f+10,table_f+entry*16,16);
        table_i[entry*2]=p.flags[0];table_i[entry*2+1]=p.flags[4];
    }
}

__device__ void cf_params(
    int w, const int* route, const real_t* raw, const real_t* usermargin,
    const int* active, const int* counts, const int* priority, const int* condim,
    const real_t* geom_margin, const real_t* geom_gap,
    const real_t* geom_friction, const real_t* geom_solmix,
    const real_t* geom_solref, const real_t* geom_solimp,
    const real_t* pair_margin, const real_t* pair_gap, const int* pair_dim,
    const real_t* pair_friction, const real_t* pair_solref,
    const real_t* pair_solreffriction, const real_t* pair_solimp,
    const real_t* o_margin, const real_t* o_solref, const real_t* o_solimp,
    const int* override_enabled, real_t* out_f, int* out_i,
    int slots, int ngeom, int npair) {
    for (int q = threadIdx.x; q < slots; q += blockDim.x) {
        const int row = w * slots + q;
        real_t* f = out_f + row * 26;
        int* flags = out_i + row * 5;
        libero_vec::fill_serial(f,26,REAL_C(0.0));
        libero_vec::fill_serial(flags,4,-1);
        flags[4] = 0;
        // Completion resets padding even on inactive/error worlds. The caller
        // owns public-output rollback; these are intermediate contact records.
        if (!active[w]) continue;
        if (counts[w] < 0 || counts[w] > slots) {
            flags[4] = 1;
            continue;
        }
        if (q >= counts[w]) continue;
        const int* r = route + row * 4;
        const int g1 = r[0], g2 = r[1], pair = r[2], user = r[3];
        if (g1 < 0 || g2 < 0 || g1 >= ngeom || g2 >= ngeom) {
            flags[4] = 2;
            continue;
        }
        if (pair < -1 || pair >= npair || (user != 0 && user != 1)) {
            flags[4] = 3;
            continue;
        }
        libero_vec::copy_serial(raw+row*10,f,10);
        const bool paired = pair >= 0;
        const bool same_priority = priority[g1] == priority[g2];
        const int dominant = priority[g1] > priority[g2] ? g1 : g2;
        real_t margin = paired ? pair_margin[pair]
                               : cf_param_max(geom_margin[g1], geom_margin[g2]);
        const real_t gap = paired ? pair_gap[pair]
                                  : cf_param_max(geom_gap[g1], geom_gap[g2]);
        const int dim = paired ? pair_dim[pair]
                               : (same_priority ? (condim[g1] > condim[g2] ? condim[g1] : condim[g2])
                                                : condim[dominant]);
        if (user) margin = usermargin[row];
        else if (override_enabled[0]) margin = o_margin[0];
        f[10] = margin;
        f[11] = margin - gap;
        flags[0] = dim;
        flags[1] = g1;
        flags[2] = g2;
        if (dim < 1 || dim > 6) {
            flags[4] = 4;
            continue;
        }
        if (paired) {
            {const real_t* src=pair_friction+pair*5;real_t* dst=f+12;for(int k=0;k<4;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
            {const real_t* src=pair_solimp+pair*5;real_t* dst=f+21;for(int k=0;k<4;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
            {const real_t* src=pair_solref+pair*2;real_t* dst=f+17;for(int k=0;k<2;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
            if (pair_solreffriction[pair * 2] != REAL_C(0.0) || pair_solreffriction[pair * 2 + 1] != REAL_C(0.0))
                {const real_t* src=pair_solreffriction+pair*2;real_t* dst=f+19;for(int k=0;k<2;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
        } else {
            if (!same_priority) {
                {const real2_t v=libero_vec::load_pair(geom_friction+dominant*3);f[12]=v.x;f[14]=v.y;f[16]=geom_friction[dominant*3+2];}
                {const real_t* src=geom_solref+dominant*2;real_t* dst=f+17;for(int k=0;k<2;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
                {const real_t* src=geom_solimp+dominant*5;real_t* dst=f+21;for(int k=0;k<4;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
            } else {
                for (int k = 0; k < 3; ++k)
                    f[12 + 2 * k] = cf_param_max(geom_friction[g1 * 3 + k], geom_friction[g2 * 3 + k]);
                const real_t s1 = geom_solmix[g1], s2 = geom_solmix[g2];
                real_t mix;
                if (s1 >= REAL_C(1e-15) && s2 >= REAL_C(1e-15)) mix = s1 / (s1 + s2);
                else if (s1 < REAL_C(1e-15) && s2 < REAL_C(1e-15)) mix = REAL_C(0.5);
                else if (s1 < REAL_C(1e-15)) mix = REAL_C(0.0);
                else mix = REAL_C(1.0);
                if (geom_solref[g1 * 2] > REAL_C(0.0) && geom_solref[g2 * 2] > REAL_C(0.0)) {
                    for (int k = 0; k < 2; ++k)
                        f[17 + k] = mix * geom_solref[g1 * 2 + k] + (REAL_C(1.0) - mix) * geom_solref[g2 * 2 + k];
                } else {
                    for (int k = 0; k < 2; ++k)
                        f[17 + k] = cf_param_min(geom_solref[g1 * 2 + k], geom_solref[g2 * 2 + k]);
                }
                for (int k = 0; k < 5; ++k)
                    f[21 + k] = geom_solimp[g1 * 5 + k] * mix + geom_solimp[g2 * 5 + k] * (REAL_C(1.0) - mix);
            }
            f[13] = f[12];
            f[15] = f[16];
        }
        for (int k = 0; k < 5; ++k) f[12 + k] = cf_param_max(REAL_C(1e-5), f[12 + k]);
        if (override_enabled[0]) {
            {const real2_t v=libero_vec::load_pair(o_solref);f[17]=f[19]=v.x;f[18]=f[20]=v.y;}
            {const real_t* src=o_solimp;real_t* dst=f+21;for(int k=0;k<4;k+=2){const real2_t v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
        }
        flags[3] = raw[row * 10] < f[11] ? 0 : 1;
    }
}

} // namespace collision_cuda
