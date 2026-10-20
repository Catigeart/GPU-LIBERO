#include "precision.cuh"
#pragma once
#include <math.h>

// Ordered collision routing from engine_collision_driver.c. One lane owns
// one world's request sequence; untouched output slots retain their contents.
namespace collision_cuda {

__device__ __forceinline__ real_t cd_max(real_t a, real_t b) {
    return a >= b ? a : b;
}
__device__ __forceinline__ bool cd_admitted(real_t a) {
    return a == a && fabs(a) <= REAL_SAFE_MAGNITUDE;
}
__device__ __forceinline__ bool cd_positions_ok(
    const real_t* xpos, int p1, int p2) {
    for (int k = 0; k < 3; ++k)
        if (!cd_admitted(xpos[p1+k]) || !cd_admitted(xpos[p2+k]))
            return false;
    return true;
}
__device__ __forceinline__ bool cd_plane_ok(const real_t* xmat, int m) {
    for (int k = 0; k < 3; ++k)
        if (!cd_admitted(xmat[m+2+3*k])) return false;
    return true;
}
__device__ __forceinline__ real_t cd_plane_distance(
    const real_t* xpos, const real_t* xmat, int p1, int p2, int m) {
    const real_t dx = xpos[p2]-xpos[p1];
    const real_t dy = xpos[p2+1]-xpos[p1+1];
    const real_t dz = xpos[p2+2]-xpos[p1+2];
    return dx*xmat[m+2] + dy*xmat[m+5] + dz*xmat[m+8];
}
__device__ __forceinline__ bool cd_route_changed_lane(int lane,int ngeom,
    const int* geom_type,const int* geom_contype,const int* geom_conaffinity,
    const int* geom_priority,const int* geom_condim,const int* present,
    const int* route_shadow){
    if(!route_shadow)return true;
    bool changed=false;int offset=0;
    for(int field=0;field<6;++field){
        const int* current=field==0?geom_type:field==1?geom_contype:
            field==2?geom_conaffinity:field==3?geom_priority:
            field==4?geom_condim:present;
        const int width=field==5?64:ngeom;
        for(int i=lane;i<width;i+=32)changed|=current[i]!=route_shadow[offset+i];
        offset+=width;
    }
    return changed;
}

__device__ void cf_dispatch(
    int w, const int* raw, const real_t* usermargin,
    const real_t* xpos, const real_t* xmat, const int* active,
    const int* counts, const int* geom_type, const int* geom_contype,
    const int* geom_conaffinity, const int* geom_priority,
    const int* geom_condim, const real_t* geom_margin,
    const real_t* geom_gap, const real_t* geom_rbound,
    const int* pair_geom1, const int* pair_geom2, const int* pair_dim,
    const real_t* pair_margin, const real_t* pair_gap,
    const int* override_enabled, const real_t* o_margin, const int* present,
    int* out_i, real_t* out_f, int* call_index, int* call_count,
    int* mid_count, int* status, int* error_index,
    int slots, int capacity, int ngeom, int npair) {
    if (threadIdx.x != 0) return;
    const int n = counts[w];
    status[w] = 0;
    error_index[w] = -1;
    call_count[w] = 0;
    mid_count[w] = 0;
    if (n < 0 || n > slots) { status[w] = 1; return; }
    if (active[w] != 0 && active[w] != 1) { status[w] = 6; return; }
    if (!active[w]) return;
    for (int q = 0; q < n; ++q) {
        const int t = w*slots+q;
        int g1 = raw[t*3], g2 = raw[t*3+1];
        const int user = raw[t*3+2];
        int ipair = -1;
        if (g2 < 0) {
            ipair = g1;
            if (ipair < 0 || ipair >= npair) {
                status[w] = 2; error_index[w] = q; return;
            }
            g1 = pair_geom1[ipair]; g2 = pair_geom2[ipair];
        }
        if (g1 < 0 || g1 >= ngeom || g2 < 0 || g2 >= ngeom) {
            status[w] = 2; error_index[w] = q; return;
        }
        if (user != 0 && user != 1) {
            status[w] = 3; error_index[w] = q; return;
        }
        if (user && !cd_admitted(usermargin[t])) {
            status[w] = 5; error_index[w] = q; return;
        }
        int type1 = geom_type[g1], type2 = geom_type[g2];
        if (type1 > type2) {
            int old = g1; g1 = g2; g2 = old;
            old = type1; type1 = type2; type2 = old;
        }
        const int b = t*12;
        out_i[b] = 0; out_i[b+1] = 0;
        out_i[b+2] = g1; out_i[b+3] = g2;
        out_i[b+4] = type1; out_i[b+5] = type2;
        out_i[b+7] = ipair;
        out_i[b+8] = mid_count[w]; out_i[b+9] = mid_count[w];
        out_i[b+10] = 0; out_i[b+11] = -1;
        if (!present[type1*8+type2]) { out_i[b] = 1; continue; }
        if (ipair < 0 && !user &&
            !(geom_contype[g1]&geom_conaffinity[g2]) &&
            !(geom_contype[g2]&geom_conaffinity[g1])) {
            out_i[b] = 2; continue;
        }
        real_t margin = REAL_C(0.0), gap = REAL_C(0.0);
        int dim = 0;
        if (ipair < 0) {
            margin = cd_max(geom_margin[g1], geom_margin[g2]);
            gap = cd_max(geom_gap[g1], geom_gap[g2]);
            if (geom_priority[g1] != geom_priority[g2]) {
                const int gp = geom_priority[g1] > geom_priority[g2] ? g1 : g2;
                dim = geom_condim[gp];
            } else {
                dim = geom_condim[g1] > geom_condim[g2] ?
                      geom_condim[g1] : geom_condim[g2];
            }
        } else {
            margin = pair_margin[ipair]; gap = pair_gap[ipair];
            dim = pair_dim[ipair];
        }
        if (user) margin = usermargin[t];
        else if (override_enabled[0]) margin = o_margin[0];
        out_i[b+6] = dim; out_i[b+10] = 1;
        out_f[t*2] = margin; out_f[t*2+1] = gap;
        const real_t r1 = geom_rbound[g1], r2 = geom_rbound[g2];
        bool rejected = false;
        if (r1 > REAL_C(0.0) && r2 > REAL_C(0.0)) {
            const real_t bound = r1+r2+margin;
            const int p1 = (w*ngeom+g1)*3, p2 = (w*ngeom+g2)*3;
            if (!cd_positions_ok(xpos, p1, p2)) {
                status[w] = 5; error_index[w] = q; return;
            }
            const real_t dx = xpos[p1]-xpos[p2];
            const real_t dy = xpos[p1+1]-xpos[p2+1];
            const real_t dz = xpos[p1+2]-xpos[p2+2];
            rejected = dx*dx+dy*dy+dz*dz > bound*bound;
        }
        if (!rejected && type1 == 0 && r2 > REAL_C(0.0)) {
            const int p1 = (w*ngeom+g1)*3, p2 = (w*ngeom+g2)*3;
            const int m = (w*ngeom+g1)*9;
            if (!cd_positions_ok(xpos,p1,p2) || !cd_plane_ok(xmat,m)) {
                status[w] = 5; error_index[w] = q; return;
            }
            rejected = cd_plane_distance(xpos,xmat,p1,p2,m) > margin+r2;
        }
        if (!rejected && type2 == 0 && r1 > REAL_C(0.0)) {
            const int p1 = (w*ngeom+g2)*3, p2 = (w*ngeom+g1)*3;
            const int m = (w*ngeom+g2)*9;
            if (!cd_positions_ok(xpos,p1,p2) || !cd_plane_ok(xmat,m)) {
                status[w] = 5; error_index[w] = q; return;
            }
            rejected = cd_plane_distance(xpos,xmat,p1,p2,m) > margin+r1;
        }
        if (rejected) { out_i[b] = 3; continue; }
        const int index = call_count[w];
        if (index >= capacity) {
            status[w] = 4; error_index[w] = q;
            call_count[w] = -1; mid_count[w] = -1; return;
        }
        call_index[w*capacity+index] = q;
        call_count[w] = index+1; ++mid_count[w];
        out_i[b+1] = 1; out_i[b+9] = mid_count[w];
        out_i[b+11] = index;
    }
}


// Classification stays private until the warp chooses the first ordered stop.
// A numeric error has a partially initialized row; a metadata error has none.
struct cd_dispatch_row {
    int i[12]; unsigned written; int error; bool callable, floats;
    real_t margin, gap;
    __device__ void put(int k, int v) { i[k]=v; written |= 1u << k; }
};
__device__ __forceinline__ void cd_classify_request(
    int w, const int* raw, const real_t* usermargin,
    const real_t* xpos, const real_t* xmat, const int* geom_type, const int* geom_contype,
    const int* geom_conaffinity, const int* geom_priority,
    const int* geom_condim, const real_t* geom_margin,
    const real_t* geom_gap, const real_t* geom_rbound,
    const int* pair_geom1, const int* pair_geom2, const int* pair_dim,
    const real_t* pair_margin, const real_t* pair_gap,
    const int* override_enabled, const real_t* o_margin, const int* present,
    int slots, int ngeom, int npair, int q, cd_dispatch_row& r,
    const unsigned* geom_pair_route=nullptr) {
    r.written=0; r.error=0; r.callable=false; r.floats=false;
        const int t = w*slots+q;
        int g1 = raw[t*3], g2 = raw[t*3+1];
        const int user = raw[t*3+2];
        int ipair = -1;
        if (g2 < 0) {
            ipair = g1;
            if (ipair < 0 || ipair >= npair) {
                r.error = 2; return;
            }
            g1 = pair_geom1[ipair]; g2 = pair_geom2[ipair];
        }
        if (g1 < 0 || g1 >= ngeom || g2 < 0 || g2 >= ngeom) {
            r.error = 2; return;
        }
        if (user != 0 && user != 1) {
            r.error = 3; return;
        }
        if (user && !cd_admitted(usermargin[t])) {
            r.error = 5; return;
        }
        // Immutable model route: bits 0..2/3..5 ordered types, 6 swap,
        // 7 detector-present, 8 bilateral affinity, 16..31 resolved condim.
        // A null pointer preserves the mutable/standalone classification.
        const unsigned route=geom_pair_route?geom_pair_route[g1*ngeom+g2]:0;
        int type1, type2;
        if(geom_pair_route){
            type1=route&7u;type2=(route>>3)&7u;
            if(route&64u){const int old=g1;g1=g2;g2=old;}
        }else{
            type1=geom_type[g1];type2=geom_type[g2];
            if(type1>type2){const int old=g1;g1=g2;g2=old;
                const int t=type1;type1=type2;type2=t;}
        }
        const int b = t*12;
        r.put(0, 0); r.put(1, 0);
        r.put(2, g1); r.put(3, g2);
        r.put(4, type1); r.put(5, type2);
        r.put(7, ipair);
        r.put(8, 0); r.put(9, 0);
        r.put(10, 0); r.put(11, -1);
        if (!(geom_pair_route?bool(route&128u):bool(present[type1*8+type2]))) { r.put(0, 1); return; }
        if (ipair < 0 && !user && !(geom_pair_route?bool(route&256u):
            bool((geom_contype[g1]&geom_conaffinity[g2]) ||
                 (geom_contype[g2]&geom_conaffinity[g1])))) {
            r.put(0, 2); return;
        }
        real_t margin = REAL_C(0.0), gap = REAL_C(0.0);
        int dim = 0;
        if (ipair < 0) {
            margin = cd_max(geom_margin[g1], geom_margin[g2]);
            gap = cd_max(geom_gap[g1], geom_gap[g2]);
            if(geom_pair_route)dim=int(route>>16);
            else if (geom_priority[g1] != geom_priority[g2]) {
                const int gp = geom_priority[g1] > geom_priority[g2] ? g1 : g2;
                dim = geom_condim[gp];
            } else {
                dim = geom_condim[g1] > geom_condim[g2] ?
                      geom_condim[g1] : geom_condim[g2];
            }
        } else {
            margin = pair_margin[ipair]; gap = pair_gap[ipair];
            dim = pair_dim[ipair];
        }
        if (user) margin = usermargin[t];
        else if (override_enabled[0]) margin = o_margin[0];
        r.put(6, dim); r.put(10, 1);
        r.margin = margin; r.gap = gap; r.floats = true;
        const real_t r1 = geom_rbound[g1], r2 = geom_rbound[g2];
        bool rejected = false;
        if (r1 > REAL_C(0.0) && r2 > REAL_C(0.0)) {
            const real_t bound = r1+r2+margin;
            const int p1 = (w*ngeom+g1)*3, p2 = (w*ngeom+g2)*3;
            if (!cd_positions_ok(xpos, p1, p2)) {
                r.error = 5; return;
            }
            const real_t dx = xpos[p1]-xpos[p2];
            const real_t dy = xpos[p1+1]-xpos[p2+1];
            const real_t dz = xpos[p1+2]-xpos[p2+2];
            rejected = dx*dx+dy*dy+dz*dz > bound*bound;
        }
        if (!rejected && type1 == 0 && r2 > REAL_C(0.0)) {
            const int p1 = (w*ngeom+g1)*3, p2 = (w*ngeom+g2)*3;
            const int m = (w*ngeom+g1)*9;
            if (!cd_positions_ok(xpos,p1,p2) || !cd_plane_ok(xmat,m)) {
                r.error = 5; return;
            }
            rejected = cd_plane_distance(xpos,xmat,p1,p2,m) > margin+r2;
        }
        if (!rejected && type2 == 0 && r1 > REAL_C(0.0)) {
            const int p1 = (w*ngeom+g2)*3, p2 = (w*ngeom+g1)*3;
            const int m = (w*ngeom+g2)*9;
            if (!cd_positions_ok(xpos,p1,p2) || !cd_plane_ok(xmat,m)) {
                r.error = 5; return;
            }
            rejected = cd_plane_distance(xpos,xmat,p1,p2,m) > margin+r1;
        }
        if (rejected) { r.put(0, 3); return; }
    r.callable=true;
}
__device__ void cf_dispatch_block(
    int w, const int* raw, const real_t* usermargin,
    const real_t* xpos, const real_t* xmat, const int* active,
    const int* counts, const int* geom_type, const int* geom_contype,
    const int* geom_conaffinity, const int* geom_priority,
    const int* geom_condim, const real_t* geom_margin,
    const real_t* geom_gap, const real_t* geom_rbound,
    const int* pair_geom1, const int* pair_geom2, const int* pair_dim,
    const real_t* pair_margin, const real_t* pair_gap,
    const int* override_enabled, const real_t* o_margin, const int* present,
    int* out_i, real_t* out_f, int* call_index, int* call_count,
    int* mid_count, int* status, int* error_index,
    int slots, int capacity, int ngeom, int npair,
    const unsigned* geom_pair_route=nullptr,const int* route_shadow=nullptr) {

    // All block threads enter. Only the first complete warp owns routing;
    // the caller supplies the block barrier before downstream stages.
    if (blockDim.x < 32) {
        cf_dispatch(w, raw, usermargin, xpos, xmat, active, counts, geom_type, geom_contype, geom_conaffinity, geom_priority, geom_condim, geom_margin, geom_gap, geom_rbound, pair_geom1, pair_geom2, pair_dim, pair_margin, pair_gap, override_enabled, o_margin, present, out_i, out_f, call_index, call_count, mid_count, status, error_index, slots, capacity, ngeom, npair);
        return;
    }
    if (threadIdx.x >= 32) return;
    const unsigned full=0xffffffffu;
    const int lane=threadIdx.x;
    const int n=counts[w];
    int header=0;
    if (n<0 || n>slots) header=1;
    else if (active[w]!=0 && active[w]!=1) header=6;
    if (lane==0) {
        status[w]=header; error_index[w]=-1;
        call_count[w]=0; mid_count[w]=0;
    }
    if (header || !active[w]) return;
    // A cold route table is guarded once per invocation, including mutable
    // diagnostic statics. No stale model classification reaches a request.
    if(geom_pair_route){
        const bool changed=cd_route_changed_lane(lane,ngeom,geom_type,geom_contype,
            geom_conaffinity,geom_priority,geom_condim,present,route_shadow);
        if(__any_sync(full,changed))geom_pair_route=nullptr;
    }
    int count=0;
    for (int base=0; base<n; base+=32) {
        const int q=base+lane;
        cd_dispatch_row r;
        r.written=0; r.error=0; r.callable=false; r.floats=false;
        if (q<n) cd_classify_request(w, raw, usermargin, xpos, xmat, geom_type, geom_contype, geom_conaffinity, geom_priority, geom_condim, geom_margin, geom_gap, geom_rbound, pair_geom1, pair_geom2, pair_dim, pair_margin, pair_gap, override_enabled, o_margin, present, slots, ngeom, npair, q, r, geom_pair_route);
        const unsigned calls=__ballot_sync(full, r.callable);
        const unsigned lower=(1u << lane)-1u;
        const int rank=__popc(calls & lower);
        const bool overflow=r.callable && count+rank>=capacity;
        const unsigned errors=__ballot_sync(full, r.error!=0 || overflow);
        const int stop=errors ? __ffs(errors)-1 : 32;
        const unsigned before=stop==32 ? full : (1u << stop)-1u;
        const int completed=__popc(calls & before);
        // Rows before the stop, plus the stopping row's partial output, are
        // the only public writes. Later speculative records remain private.
        if (q<n && lane<=stop) {
            const int b=(w*slots+q)*12;
            if (r.written) {
                r.i[8]=count+rank; r.i[9]=count+rank;
                if (r.callable && !overflow) {
                    r.i[1]=1; r.i[9]=count+rank+1; r.i[11]=count+rank;
                    call_index[w*capacity+count+rank]=q;
                }
                for (int k=0; k<12; ++k)
                    if (r.written & (1u<<k)) out_i[b+k]=r.i[k];
                if (r.floats) {
                    out_f[(w*slots+q)*2]=r.margin;
                    out_f[(w*slots+q)*2+1]=r.gap;
                }
            }
        }
        const int stop_error=__shfl_sync(full, r.error, stop==32 ? 0 : stop);
        count+=completed;
        if (errors) {
            if (lane==0) {
                status[w]=stop_error ? stop_error : 4;
                error_index[w]=base+stop;
                call_count[w]=stop_error ? count : -1;
                mid_count[w]=stop_error ? count : -1;
            }
            return;
        }
        if (lane==0) { call_count[w]=count; mid_count[w]=count; }
    }
}

} // namespace collision_cuda
