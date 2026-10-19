#pragma once

// MuJoCo 2.3.7 contact parameter completion. One block owns one world;
// each lane independently completes contacts. Input tables are immutable.
namespace collision_cuda {

// All contacts emitted by a call share these parameters. Geometry-dependent
// mixing belongs to the call, while distance, position and frame belong to
// each contact. The compatible path still materializes every public row.
struct PairParams {
    double f[26];
    int flags[5];
    bool route_valid;
};

__device__ __forceinline__ double cf_param_max(double a, double b) {
    return a >= b ? a : b;
}
__device__ __forceinline__ double cf_param_min(double a, double b) {
    return a <= b ? a : b;
}

__device__ __forceinline__ PairParams cf_pair_params(
    int g1,int g2,int pair,int user,double user_margin,
    const int* priority,const int* condim,const double* geom_margin,const double* geom_gap,
    const double* geom_friction,const double* geom_solmix,const double* geom_solref,const double* geom_solimp,
    const double* pair_margin,const double* pair_gap,const int* pair_dim,const double* pair_friction,
    const double* pair_solref,const double* pair_solreffriction,const double* pair_solimp,
    const double* o_margin,const double* o_solref,const double* o_solimp,const int* override_enabled,
    int ngeom,int npair) {
    PairParams p;
    libero_vec::fill_serial(p.f,26,0.0);
    libero_vec::fill_serial(p.flags,4,-1);
    p.flags[4]=0;p.route_valid=false;
    if(g1<0||g2<0||g1>=ngeom||g2>=ngeom){p.flags[4]=2;return p;}
    if(pair< -1||pair>=npair||(user!=0&&user!=1)){p.flags[4]=3;return p;}
    p.route_valid=true;
    const bool paired=pair>=0,same_priority=priority[g1]==priority[g2];
    const int dominant=priority[g1]>priority[g2]?g1:g2;
    double margin=paired?pair_margin[pair]:cf_param_max(geom_margin[g1],geom_margin[g2]);
    const double gap=paired?pair_gap[pair]:cf_param_max(geom_gap[g1],geom_gap[g2]);
    const int dim=paired?pair_dim[pair]:(same_priority?(condim[g1]>condim[g2]?condim[g1]:condim[g2]):condim[dominant]);
    if(user)margin=user_margin;else if(override_enabled[0])margin=o_margin[0];
    p.f[10]=margin;p.f[11]=margin-gap;p.flags[0]=dim;p.flags[1]=g1;p.flags[2]=g2;
    if(dim<1||dim>6){p.flags[4]=4;return p;}
    if(paired){
        {const double* src=pair_friction+pair*5;double* dst=p.f+12;for(int k=0;k<4;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
        {const double* src=pair_solimp+pair*5;double* dst=p.f+21;for(int k=0;k<4;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
        {const double* src=pair_solref+pair*2;double* dst=p.f+17;for(int k=0;k<2;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
        if(pair_solreffriction[pair*2]!=0.0||pair_solreffriction[pair*2+1]!=0.0)
            {const double* src=pair_solreffriction+pair*2;double* dst=p.f+19;for(int k=0;k<2;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
    }else{
        if(!same_priority){
            {const double2 v=libero_vec::load_pair(geom_friction+dominant*3);p.f[12]=v.x;p.f[14]=v.y;p.f[16]=geom_friction[dominant*3+2];}
            {const double* src=geom_solref+dominant*2;double* dst=p.f+17;for(int k=0;k<2;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
            {const double* src=geom_solimp+dominant*5;double* dst=p.f+21;for(int k=0;k<4;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
        }else{
            for(int k=0;k<3;++k)p.f[12+2*k]=cf_param_max(geom_friction[g1*3+k],geom_friction[g2*3+k]);
            const double s1=geom_solmix[g1],s2=geom_solmix[g2];double mix;
            if(s1>=1e-15&&s2>=1e-15)mix=s1/(s1+s2);
            else if(s1<1e-15&&s2<1e-15)mix=0.5;
            else if(s1<1e-15)mix=0.0;else mix=1.0;
            if(geom_solref[g1*2]>0.0&&geom_solref[g2*2]>0.0)
                for(int k=0;k<2;++k)p.f[17+k]=mix*geom_solref[g1*2+k]+(1.0-mix)*geom_solref[g2*2+k];
            else for(int k=0;k<2;++k)p.f[17+k]=cf_param_min(geom_solref[g1*2+k],geom_solref[g2*2+k]);
            for(int k=0;k<5;++k)p.f[21+k]=geom_solimp[g1*5+k]*mix+geom_solimp[g2*5+k]*(1.0-mix);
        }
        p.f[13]=p.f[12];p.f[15]=p.f[16];
    }
    for(int k=0;k<5;++k)p.f[12+k]=cf_param_max(1e-5,p.f[12+k]);
    if(override_enabled[0]){
        {const double2 v=libero_vec::load_pair(o_solref);p.f[17]=p.f[19]=v.x;p.f[18]=p.f[20]=v.y;}
        {const double* src=o_solimp;double* dst=p.f+21;for(int k=0;k<4;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
    }
    return p;
}

// Cold model-owned table, shared by every world. Use exactly the device
// arithmetic of the uncached path, including override and direct solref.
__device__ void cf_pair_params_table(int w,
    const int* priority,const int* condim,const double* geom_margin,const double* geom_gap,
    const double* geom_friction,const double* geom_solmix,const double* geom_solref,const double* geom_solimp,
    const double* pair_margin,const double* pair_gap,const int* pair_dim,const double* pair_friction,
    const double* pair_solref,const double* pair_solreffriction,const double* pair_solimp,
    const double* o_margin,const double* o_solref,const double* o_solimp,const int* override_enabled,
    int ngeom,int npair,double* table_f,int* table_i){
    for(int entry=threadIdx.x;entry<ngeom*ngeom;entry+=blockDim.x){
        const PairParams p=cf_pair_params(entry/ngeom,entry%ngeom,-1,0,0.0,
            priority,condim,geom_margin,geom_gap,geom_friction,geom_solmix,geom_solref,geom_solimp,
            pair_margin,pair_gap,pair_dim,pair_friction,pair_solref,pair_solreffriction,pair_solimp,
            o_margin,o_solref,o_solimp,override_enabled,ngeom,npair);
        libero_vec::copy_serial(p.f+10,table_f+entry*16,16);
        table_i[entry*2]=p.flags[0];table_i[entry*2+1]=p.flags[4];
    }
}

__device__ void cf_params(
    int w, const int* route, const double* raw, const double* usermargin,
    const int* active, const int* counts, const int* priority, const int* condim,
    const double* geom_margin, const double* geom_gap,
    const double* geom_friction, const double* geom_solmix,
    const double* geom_solref, const double* geom_solimp,
    const double* pair_margin, const double* pair_gap, const int* pair_dim,
    const double* pair_friction, const double* pair_solref,
    const double* pair_solreffriction, const double* pair_solimp,
    const double* o_margin, const double* o_solref, const double* o_solimp,
    const int* override_enabled, double* out_f, int* out_i,
    int slots, int ngeom, int npair) {
    for (int q = threadIdx.x; q < slots; q += blockDim.x) {
        const int row = w * slots + q;
        double* f = out_f + row * 26;
        int* flags = out_i + row * 5;
        libero_vec::fill_serial(f,26,0.0);
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
        double margin = paired ? pair_margin[pair]
                               : cf_param_max(geom_margin[g1], geom_margin[g2]);
        const double gap = paired ? pair_gap[pair]
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
            {const double* src=pair_friction+pair*5;double* dst=f+12;for(int k=0;k<4;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
            {const double* src=pair_solimp+pair*5;double* dst=f+21;for(int k=0;k<4;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
            {const double* src=pair_solref+pair*2;double* dst=f+17;for(int k=0;k<2;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
            if (pair_solreffriction[pair * 2] != 0.0 || pair_solreffriction[pair * 2 + 1] != 0.0)
                {const double* src=pair_solreffriction+pair*2;double* dst=f+19;for(int k=0;k<2;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
        } else {
            if (!same_priority) {
                {const double2 v=libero_vec::load_pair(geom_friction+dominant*3);f[12]=v.x;f[14]=v.y;f[16]=geom_friction[dominant*3+2];}
                {const double* src=geom_solref+dominant*2;double* dst=f+17;for(int k=0;k<2;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}}
                {const double* src=geom_solimp+dominant*5;double* dst=f+21;for(int k=0;k<4;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
            } else {
                for (int k = 0; k < 3; ++k)
                    f[12 + 2 * k] = cf_param_max(geom_friction[g1 * 3 + k], geom_friction[g2 * 3 + k]);
                const double s1 = geom_solmix[g1], s2 = geom_solmix[g2];
                double mix;
                if (s1 >= 1e-15 && s2 >= 1e-15) mix = s1 / (s1 + s2);
                else if (s1 < 1e-15 && s2 < 1e-15) mix = 0.5;
                else if (s1 < 1e-15) mix = 0.0;
                else mix = 1.0;
                if (geom_solref[g1 * 2] > 0.0 && geom_solref[g2 * 2] > 0.0) {
                    for (int k = 0; k < 2; ++k)
                        f[17 + k] = mix * geom_solref[g1 * 2 + k] + (1.0 - mix) * geom_solref[g2 * 2 + k];
                } else {
                    for (int k = 0; k < 2; ++k)
                        f[17 + k] = cf_param_min(geom_solref[g1 * 2 + k], geom_solref[g2 * 2 + k]);
                }
                for (int k = 0; k < 5; ++k)
                    f[21 + k] = geom_solimp[g1 * 5 + k] * mix + geom_solimp[g2 * 5 + k] * (1.0 - mix);
            }
            f[13] = f[12];
            f[15] = f[16];
        }
        for (int k = 0; k < 5; ++k) f[12 + k] = cf_param_max(1e-5, f[12 + k]);
        if (override_enabled[0]) {
            {const double2 v=libero_vec::load_pair(o_solref);f[17]=f[19]=v.x;f[18]=f[20]=v.y;}
            {const double* src=o_solimp;double* dst=f+21;for(int k=0;k<4;k+=2){const double2 v=libero_vec::load_pair(src+k);libero_vec::store_pair(dst+k,v.x,v.y);}dst[4]=src[4];}
        }
        flags[3] = raw[row * 10] < f[11] ? 0 : 1;
    }
}

} // namespace collision_cuda
