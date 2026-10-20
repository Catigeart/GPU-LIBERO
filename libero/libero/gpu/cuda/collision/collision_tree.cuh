#include "precision.cuh"
#pragma once
#include <math.h>
#include <stdint.h>

// Ordered body BVH traversal. The first warp compacts direct-route requests;
// lane 0 retains the bounded DFS stack and diagnostic tapes. All block threads
// must enter cf_tree; its caller retains the normal inter-stage block barrier.
namespace collision_cuda {
__device__ inline real_t tree_dot(real_t a,real_t b,real_t c,real_t x,real_t y,real_t z) {
    return a*x+b*y+c*z;
}
__device__ inline real_t tree_normal(const real_t* m,int base,int axis,int coord) {
    return m[base+3*coord+axis];
}
__device__ inline real_t tree_center(const real_t* aa,int ab,const real_t* p,int pb,
                                    const real_t* m,int mb,int axis) {
    return tree_dot(aa[ab],aa[ab+1],aa[ab+2],m[mb+3*axis],m[mb+3*axis+1],m[mb+3*axis+2])+p[pb+axis];
}
// Return overlap in bit 0, cache initialization in bit 1, reject axis+1 above.
__device__ inline int tree_obb(const real_t* aa,int ab0,int ab1,const real_t* pos,
                              int pb0,int pb1,const real_t* mat,int mb0,int mb1,
                              int cached,real_t* product,real_t* offset,int sb,int initialize) {
    bool inf1[3],inf2[3];
    for(int k=0;k<3;++k){inf1[k]=aa[ab0+3+k]>=REAL_C(1.e10);inf2[k]=aa[ab1+3+k]>=REAL_C(1.e10);}
    if((inf1[0]&&inf1[1]&&inf1[2])||(inf2[0]&&inf2[1]&&inf2[2]))return 1+2*initialize;
    real_t center[2][3]={{0,0,0},{0,0,0}};
    if(!cached)for(int k=0;k<3;++k){
        center[0][k]=tree_center(aa,ab0,pos,pb0,mat,mb0,k);
        center[1][k]=tree_center(aa,ab1,pos,pb1,mat,mb1,k);
    }
    if(cached&&initialize){
        for(int i=0;i<2;++i){int mi=i?mb1:mb0,pi=i?pb1:pb0;
            for(int j=0;j<2;++j){int mj=j?mb1:mb0;
                for(int k=0;k<3;++k){
                    for(int l=0;l<3;++l)product[sb+18*i+9*j+3*k+l]=tree_dot(
                        tree_normal(mat,mi,l,0),tree_normal(mat,mi,l,1),tree_normal(mat,mi,l,2),
                        tree_normal(mat,mj,k,0),tree_normal(mat,mj,k,1),tree_normal(mat,mj,k,2));
                    offset[(sb/36)*12+6*i+3*j+k]=tree_dot(pos[pi],pos[pi+1],pos[pi+2],
                        tree_normal(mat,mj,k,0),tree_normal(mat,mj,k,1),tree_normal(mat,mj,k,2));
                }
            }
        }
        initialize=0;
    }
    for(int j=0;j<2;++j){
        if((j==0&&(inf2[0]||inf2[1]||inf2[2]))||(j==1&&(inf1[0]||inf1[1]||inf1[2])))continue;
        int mj=j?mb1:mb0;
        for(int k=0;k<3;++k){
            real_t n[3]={tree_normal(mat,mj,k,0),tree_normal(mat,mj,k,1),tree_normal(mat,mj,k,2)};
            real_t p[2]={0,0},r[2]={0,0};
            for(int i=0;i<2;++i){int ab=i?ab1:ab0,mi=i?mb1:mb0;
                if(!cached){
                    p[i]=tree_dot(center[i][0],center[i][1],center[i][2],n[0],n[1],n[2]);
                    real_t d0=tree_dot(tree_normal(mat,mi,0,0),tree_normal(mat,mi,0,1),tree_normal(mat,mi,0,2),n[0],n[1],n[2]);
                    real_t d1=tree_dot(tree_normal(mat,mi,1,0),tree_normal(mat,mi,1,1),tree_normal(mat,mi,1,2),n[0],n[1],n[2]);
                    real_t d2=tree_dot(tree_normal(mat,mi,2,0),tree_normal(mat,mi,2,1),tree_normal(mat,mi,2,2),n[0],n[1],n[2]);
                    r[i]=fabs(aa[ab+3]*d0)+fabs(aa[ab+4]*d1)+fabs(aa[ab+5]*d2);
                }else{
                    int adr=sb+18*i+9*j+3*k;
                    p[i]=aa[ab]*product[adr]+aa[ab+1]*product[adr+1]+aa[ab+2]*product[adr+2]+offset[(sb/36)*12+6*i+3*j+k];
                    r[i]=fabs(aa[ab+3]*product[adr])+fabs(aa[ab+4]*product[adr+1])+fabs(aa[ab+5]*product[adr+2]);
                }
            }
            if(r[0]+r[1]<fabs(p[1]-p[0]))return 2*initialize+4*(3*j+k+1);
        }
    }
    return 1+2*initialize;
}
__device__ inline int tree_sphere(int g1,int g2,const real_t* bound,const int* type,
                                 const real_t* pos,const real_t* mat,int pb,int mb){
    int p1=pb+3*g1,p2=pb+3*g2;
    if(bound[g1]>0&&bound[g2]>0){
        real_t radius=bound[g1]+bound[g2];
        real_t x=pos[p1]-pos[p2],y=pos[p1+1]-pos[p2+1],z=pos[p1+2]-pos[p2+2];
        if(x*x+y*y+z*z>radius*radius)return 0;
    }
    if(type[g1]==0&&bound[g2]>0){int m=mb+9*g1;
        if(tree_dot(pos[p2]-pos[p1],pos[p2+1]-pos[p1+1],pos[p2+2]-pos[p1+2],mat[m+2],mat[m+5],mat[m+8])>bound[g2])return 0;
    }
    if(type[g2]==0&&bound[g1]>0){int m=mb+9*g2;
        if(tree_dot(pos[p1]-pos[p2],pos[p1+1]-pos[p2+1],pos[p1+2]-pos[p2+2],mat[m+2],mat[m+5],mat[m+8])>bound[g1])return 0;
    }
    return 1;
}
// Optional cold-model CSR lists every original explicit-pair index for both
// directed geom keys, sorted by index. Duplicates and arbitrary stage slices
// remain exact; a lone first-pair index cannot represent this predicate.
// The host may provide this cache only for the immutable pair1/pair2 owner.
__device__ inline bool tree_suppressed(int g1,int g2,int merged,int start,int end,const int* p1,const int* p2,
    const int* pair_key_offsets=nullptr,const int* pair_key_indices=nullptr,int ngeom=0){
    if(!merged)return false;
    if(pair_key_offsets && pair_key_indices && g1>=0 && g1<ngeom && g2>=0 && g2<ngeom){
        const int key=g1*ngeom+g2;
        int lo=pair_key_offsets[key],hi=pair_key_offsets[key+1];
        const int limit=hi;
        while(lo<hi){const int mid=lo+(hi-lo)/2;
            if(pair_key_indices[mid]<start)lo=mid+1;else hi=mid;}
        return lo<limit && pair_key_indices[lo]<end;
    }
    for(int k=start;k<end;++k)if((p1[k]==g1&&p2[k]==g2)||(p1[k]==g2&&p2[k]==g1))return true;
    return false;
}
__device__ inline real_t tree_surface(const real_t* aa,int b){
    real_t x=aa[b+3]-aa[b],y=aa[b+4]-aa[b+1],z=aa[b+5]-aa[b+2];return x*y+y*z+z*x;
}
__device__ inline bool tree_pairs_changed_lane(int lane,int npair,
    const int* pair1,const int* pair2,const int* pair_shadow){
    if(!pair_shadow)return true;
    bool changed=false;
    for(int i=lane;i<npair;i+=32)
        changed|=pair1[i]!=pair_shadow[i] || pair2[i]!=pair_shadow[npair+i];
    return changed;
}
template<bool Diagnostics>
__device__ inline void cf_tree_impl(int w,const int* stage,const int* stage_count,const int* active_in,
    const real_t* xipos,const real_t* ximat,const real_t* geom_pos,const real_t* geom_mat,
    const int* body_adr,const int* body_num,const int* geom_adr,const int* geom_num,
    const int* child,const int* geomid,const real_t* bvh_aa,const real_t* geom_aa,
    const real_t* rbound,const int* gtype,const int* pair1,const int* pair2,
    int* stack1,int* stack2,real_t* product,real_t* offset,
    int* out_req,int* out_visit,int* out_stack,int* out_leaf,int* out_active,
    int* req_count,int* visit_count,int* stack_count,int* leaf_count,int* status,int* first_stage,
    int stage_slots,int req_slots,int visit_slots,int stack_slots,int leaf_slots,int nbody,int ngeom,
    int body_nodes,int max_stack,int npair,
    const int* pair_key_offsets=nullptr,const int* pair_key_indices=nullptr,
    const int* pair_shadow=nullptr){
    // Only the first full warp participates; no block-wide barriers occur here.
    // Keeping the DFS leader avoids changing traversal/tape/scratch ordering.
    if(threadIdx.x >= 32)return;
    const unsigned mask=0xffffffffu;
    const int lane=threadIdx.x;
    if(lane==0){
        req_count[w]=visit_count[w]=stack_count[w]=leaf_count[w]=0;
        status[w]=0;first_stage[w]=-1;
    }
    __syncwarp(mask);
    int count=stage_count[w];
    if(count<0||count>stage_slots){if(lane==0){status[w]=1;first_stage[w]=0;}return;}
    if(!active_in[w])return;
    // Guard the complete explicit-pair source once, rather than repeating its
    // linear scan for every query. Mutable statics use the original fallback.
    if(pair_key_offsets && pair_key_indices){
        const bool changed=tree_pairs_changed_lane(lane,npair,pair1,pair2,pair_shadow);
        if(__any_sync(mask,changed)){
            pair_key_offsets=nullptr;pair_key_indices=nullptr;
        }
    }
    // Every OBB predicate in this world traversal consumes the same body's
    // 36 products and 12 offsets. Keep that state local across visits, then
    // preserve its retained scratch owner once at the ordered exit.
    real_t local_product[36],local_offset[12];
    if(lane==0){
        libero_vec::copy_serial(product+w*36,local_product,36);
        libero_vec::copy_serial(offset+w*12,local_offset,12);
    }
    if constexpr(Diagnostics)for(int i=lane;i<body_nodes;i+=32)out_active[w*body_nodes+i]=0;
    __syncwarp(mask);
    int rc=0,vc=0,sc=0,lc=0;
    for(int s=0;s<count;++s){
        int sr=(w*stage_slots+s)*6,b1=stage[sr],b2=stage[sr+1],merged=stage[sr+2];
        int start=stage[sr+3],end=stage[sr+4],route=stage[sr+5];
        if(b1<0||b1>=nbody||b2<0||b2>=nbody||(merged!=0&&merged!=1)||start<0||end<start||end>npair||(route!=1&&route!=2)){
            if(lane==0){status[w]=2;first_stage[w]=s;}break;
        }
        if(route==2){
            // Ordered Cartesian index: g1-major, g2-minor. Predicate scans
            // remain lane-local because each lane has a different geom pair.
            const int n1=geom_num[b1],n2=geom_num[b2];
            const long long total=(n1>0&&n2>0)?(long long)n1*n2:0;
            for(long long base=0;base<total;base+=32){
                const long long index=base+lane;
                int g1=0,g2=0;
                bool keep=false;
                if(index<total){
                    g1=geom_adr[b1]+int(index/n2);
                    g2=geom_adr[b2]+int(index%n2);
                    keep=!tree_suppressed(g1,g2,merged,start,end,pair1,pair2,pair_key_offsets,pair_key_indices,ngeom);
                }
                const unsigned selected=__ballot_sync(mask,keep);
                const int rank=__popc(selected&((1u<<lane)-1u));
                const int added=__popc(selected);
                const int remaining=rc<req_slots?req_slots-rc:0;
                if(keep&&rank<remaining){
                    const int rr=(w*req_slots+rc+rank)*4;
                    out_req[rr]=s;out_req[rr+1]=g1;out_req[rr+2]=g2;out_req[rr+3]=0;
                }
                rc+=added<remaining?added:remaining;
                if(added>remaining){
                    if(lane==0){status[w]=6;first_stage[w]=s;}
                    break;
                }
            }
            __syncwarp(mask);
            if(status[w])break;
            continue;
        }
        if(lane==0){do{
        int adr1=body_adr[b1],adr2=body_adr[b2],num1=body_num[b1],num2=body_num[b2];
        if(adr1<0||adr2<0||num1<=0||num2<=0||adr1+num1>body_nodes||adr2+num2>body_nodes){status[w]=3;first_stage[w]=s;break;}
        int limit=num1+num2;
        if(limit>max_stack){status[w]=4;first_stage[w]=s;break;}
        int sb=w*max_stack,nstack=1;
        stack1[sb]=stack2[sb]=0;
        if(sc>=stack_slots){status[w]=5;first_stage[w]=s;break;}
        int tr=(w*stack_slots+sc)*5;
        if constexpr(Diagnostics){out_stack[tr]=s;out_stack[tr+1]=1;out_stack[tr+2]=0;out_stack[tr+3]=0;out_stack[tr+4]=1;}++sc;
        int initialize=1;
        while(nstack>0){
            --nstack;int n1=stack1[sb+nstack],n2=stack2[sb+nstack];
            if(n1<0||n1>=num1||n2<0||n2>=num2){status[w]=7;first_stage[w]=s;break;}
            if(sc>=stack_slots){status[w]=5;first_stage[w]=s;break;}
            tr=(w*stack_slots+sc)*5;
            if constexpr(Diagnostics){out_stack[tr]=s;out_stack[tr+1]=0;out_stack[tr+2]=n1;out_stack[tr+3]=n2;out_stack[tr+4]=nstack;}++sc;
            int a1=adr1+n1,a2=adr2+n2;
            bool leaf1=child[2*a1]==-1&&child[2*a1+1]==-1,leaf2=child[2*a2]==-1&&child[2*a2+1]==-1;
            int id1=geomid[a1],id2=geomid[a2];
            if((leaf1&&(id1< -1||id1>=ngeom))||(leaf2&&(id2< -1||id2>=ngeom))){status[w]=7;first_stage[w]=s;break;}
            if(vc>=visit_slots){status[w]=8;first_stage[w]=s;break;}
            int vr=(w*visit_slots+vc)*12;
            if constexpr(Diagnostics){out_visit[vr]=s;out_visit[vr+1]=n1;out_visit[vr+2]=n2;out_visit[vr+3]=leaf1;out_visit[vr+4]=leaf2;
            out_visit[vr+5]=id1;out_visit[vr+6]=id2;out_visit[vr+7]=nstack;
            for(int k=8;k<12;++k)out_visit[vr+k]=-1;}
            ++vc;
            if(leaf1&&leaf2&&id1!=-1&&id2!=-1){
                int sphere=tree_sphere(id1,id2,rbound,gtype,geom_pos,geom_mat,w*ngeom*3,w*ngeom*9);
                if constexpr(Diagnostics)out_visit[vr+8]=sphere;
                if(sphere){
                    int obb=tree_obb(geom_aa,6*id1,6*id2,geom_pos,w*ngeom*3+3*id1,w*ngeom*3+3*id2,
                        geom_mat,w*ngeom*9+9*id1,w*ngeom*9+9*id2,0,local_product,local_offset,0,initialize);
                    if constexpr(Diagnostics){out_visit[vr+9]=obb&1;out_visit[vr+11]=(obb>>2)-1;}
                    if(obb&1){
                        if(lc>=leaf_slots){status[w]=9;first_stage[w]=s;break;}
                        int lr=(w*leaf_slots+lc)*3;
                        if constexpr(Diagnostics){out_leaf[lr]=s;out_leaf[lr+1]=id1;out_leaf[lr+2]=id2;}++lc;
                        if(!tree_suppressed(id1,id2,merged,start,end,pair1,pair2,pair_key_offsets,pair_key_indices,ngeom)){
                            if(rc>=req_slots){status[w]=6;first_stage[w]=s;break;}
                            int rr=(w*req_slots+rc)*4;
                            out_req[rr]=s;out_req[rr+1]=id1;out_req[rr+2]=id2;out_req[rr+3]=0;++rc;
                        }
                        if constexpr(Diagnostics)out_active[w*body_nodes+a1]=out_active[w*body_nodes+a2]=1;
                    }
                }
                continue;
            }
            int obb=tree_obb(bvh_aa,6*a1,6*a2,xipos,w*nbody*3+3*b1,w*nbody*3+3*b2,
                ximat,w*nbody*9+9*b1,w*nbody*9+9*b2,1,local_product,local_offset,0,initialize);
            initialize=(obb>>1)&1;
            if constexpr(Diagnostics){out_visit[vr+10]=obb&1;out_visit[vr+11]=(obb>>2)-1;}
            if(!(obb&1))continue;
            if constexpr(Diagnostics)out_active[w*body_nodes+a1]=out_active[w*body_nodes+a2]=1;
            int split1=0;
            if(!leaf1&&leaf2)split1=1;
            else if(!(leaf1&&!leaf2)&&tree_surface(bvh_aa,6*a1)>tree_surface(bvh_aa,6*a2))split1=1;
            for(int i=0;i<2;++i){
                int c=split1?child[2*a1+i]:child[2*a2+i],p1=split1?c:n1,p2=split1?n2:c;
                if(c!=-1){
                    if(nstack>=limit||nstack>=max_stack){status[w]=10;first_stage[w]=s;break;}
                    stack1[sb+nstack]=p1;stack2[sb+nstack]=p2;++nstack;
                    if(sc>=stack_slots){status[w]=5;first_stage[w]=s;break;}
                    tr=(w*stack_slots+sc)*5;
                    if constexpr(Diagnostics){out_stack[tr]=s;out_stack[tr+1]=1;out_stack[tr+2]=p1;out_stack[tr+3]=p2;out_stack[tr+4]=nstack;}++sc;
                }
            }
            if(status[w])break;
        }
        }while(false);}
        __syncwarp(mask);
        rc=__shfl_sync(mask,rc,0);
        vc=__shfl_sync(mask,vc,0);
        sc=__shfl_sync(mask,sc,0);
        lc=__shfl_sync(mask,lc,0);
        if(status[w])break;
    }
    if(lane==0){
        req_count[w]=rc;visit_count[w]=vc;stack_count[w]=sc;leaf_count[w]=lc;
        libero_vec::copy_serial(local_product,product+w*36,36);
        libero_vec::copy_serial(local_offset,offset+w*12,12);
    }
}
} // namespace collision_cuda
#include "collision_tree_parallel.cuh"
namespace collision_cuda {
// General component/debug callers retain complete diagnostic tapes.
template<class... Args>
__device__ __forceinline__ void cf_tree(Args... args){cf_tree_impl<true>(args...);}
// Production reuses the unused visit tape as bounded private global scratch.
// The caller supplies the usual stage barrier after either collective path.
template<class... Args>
__device__ __forceinline__ void cf_tree_live(Args... args){
    if(!tree_parallel_live(args...))cf_tree_impl<false>(args...);
}
} // namespace collision_cuda
