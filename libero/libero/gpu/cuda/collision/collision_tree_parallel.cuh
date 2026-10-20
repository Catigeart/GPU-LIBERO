#include "precision.cuh"
#pragma once

namespace collision_cuda {
// The live path does not publish visit tapes. Reuse that retained global arena
// for private per-stage requests, OBB coefficients and per-worker DFS stacks.
// No allocation or shared-memory extent grows with the number of worlds.
// Admission and ordered commit fall back to the original traversal whenever
// the arena or any logical output budget is insufficient. Thus even overflow
// preserves the original first error, stage and successful request prefix.
__device__ inline bool tree_parallel_live(int w,
    const int* stage,const int* stage_count,const int* active_in,
    const real_t* xipos,const real_t* ximat,const real_t* geom_pos,const real_t* geom_mat,
    const int* body_adr,const int* body_num,const int* geom_adr,const int* geom_num,
    const int* child,const int* geomid,const real_t* bvh_aa,const real_t* geom_aa,
    const real_t* rbound,const int* gtype,const int* pair1,const int* pair2,
    int* stack1,int* stack2,real_t* product,real_t* offset,
    int* out_req,int* out_visit,int* out_stack,int* out_leaf,int* out_active,
    int* req_count,int* visit_count,int* stack_count,int* leaf_count,int* status,int* first_stage,
    int stage_slots,int req_slots,int visit_slots,int stack_slots,int leaf_slots,int nbody,int ngeom,
    int body_nodes,int max_stack,int npair,
    const int* pair_key_offsets=nullptr,const int* pair_key_indices=nullptr,const int* pair_shadow=nullptr) {
    constexpr int fields=10;
    enum { UpperBase, UpperCount, Requests, Visits, StackEvents, Leaves,
           Initialized, OutputBase, Failed, Reserved };
    __shared__ int admitted, pairs_changed, last_initialized;
    const int tid=threadIdx.x,count=stage_count[w];
    if(tid==0){admitted=1;pairs_changed=0;last_initialized=-1;}
    __syncthreads();
    // Tiny/invalid stage lists use the low-overhead, fully ordered owner path.
    if(!active_in[w] || count<2 || count>stage_slots || max_stack<1 || visit_slots<1)
        return false;
    int* arena=out_visit+size_t(w)*visit_slots*12;
    const size_t cells=size_t(visit_slots)*12;
    const size_t coefficient_start=(size_t(count)*fields+1)&~size_t(1);
    const size_t stack_start=coefficient_start+size_t(count)*96;
    const size_t request_start=stack_start+size_t(blockDim.x)*max_stack*2;
    if(request_start>cells)return false;
    int* metadata=arena;
    real_t* coefficients=reinterpret_cast<real_t*>(arena+coefficient_start);
    int* stacks=arena+stack_start;
    int* private_requests=arena+request_start;
    if(tid==0){
        size_t upper=0;
        for(int s=0;s<count;++s){
            const int* entry=stage+(size_t(w)*stage_slots+s)*6;
            const int b1=entry[0],b2=entry[1],route=entry[5];
            if(b1<0||b1>=nbody||b2<0||b2>=nbody||
               (entry[2]!=0&&entry[2]!=1)||entry[3]<0||entry[4]<entry[3]||entry[4]>npair||
               (route!=1&&route!=2)){admitted=0;break;}
            if(route==1 && (body_adr[b1]<0||body_adr[b2]<0||body_num[b1]<=0||body_num[b2]<=0||
               body_adr[b1]+body_num[b1]>body_nodes||body_adr[b2]+body_num[b2]>body_nodes||
               body_num[b1]+body_num[b2]>max_stack)){admitted=0;break;}
            const int n1=geom_num[b1],n2=geom_num[b2];
            const size_t bound=n1>0&&n2>0?size_t(n1)*n2:0;
            // Geom lists are disjoint within a valid body BVH. At most one
            // request is emitted for any pair of leaves, before suppression.
            if(upper+bound>(cells-request_start)/2 || upper+bound>size_t(INT32_MAX)){
                admitted=0;break;
            }
            metadata[s*fields+UpperBase]=int(upper);
            metadata[s*fields+UpperCount]=int(bound);upper+=bound;
        }
    }
    if(tid<32 && pair_key_offsets && pair_key_indices &&
       tree_pairs_changed_lane(tid,npair,pair1,pair2,pair_shadow))atomicExch(&pairs_changed,1);
    __syncthreads();
    if(!admitted)return false;
    if(pairs_changed){pair_key_offsets=nullptr;pair_key_indices=nullptr;}
    for(int s=tid;s<count;s+=blockDim.x){
        int* meta=metadata+s*fields;
        const int* entry=stage+(size_t(w)*stage_slots+s)*6;
        const int b1=entry[0],b2=entry[1],merged=entry[2],start=entry[3],end=entry[4],route=entry[5];
        int* requests=private_requests+size_t(meta[UpperBase])*2;
        const int capacity=meta[UpperCount];
        real_t* products=coefficients+size_t(s)*48;
        real_t* offsets=products+36;
        int rc=0,vc=0,sc=0,lc=0,initialize=1;
        bool failed=false;
        if(route==2){
            const int n1=geom_num[b1],n2=geom_num[b2];
            for(int i=0;i<n1;++i)for(int j=0;j<n2;++j){
                const int g1=geom_adr[b1]+i,g2=geom_adr[b2]+j;
                if(!tree_suppressed(g1,g2,merged,start,end,pair1,pair2,pair_key_offsets,pair_key_indices,ngeom)){
                    if(rc>=capacity){failed=true;break;}
                    requests[2*rc]=g1;requests[2*rc+1]=g2;++rc;
                }
            }
        }else{
            const int adr1=body_adr[b1],adr2=body_adr[b2],num1=body_num[b1],num2=body_num[b2];
            const int limit=num1+num2;
            int* first=stacks+size_t(tid)*max_stack*2;
            int* second=first+max_stack;
            int nstack=1;first[0]=second[0]=0;++sc;
            while(nstack && !failed){
                --nstack;const int n1=first[nstack],n2=second[nstack];++sc;
                if(n1<0||n1>=num1||n2<0||n2>=num2){failed=true;break;}
                const int a1=adr1+n1,a2=adr2+n2;
                const bool leaf1=child[2*a1]==-1&&child[2*a1+1]==-1;
                const bool leaf2=child[2*a2]==-1&&child[2*a2+1]==-1;
                const int id1=geomid[a1],id2=geomid[a2];++vc;
                if((leaf1&&(id1< -1||id1>=ngeom))||(leaf2&&(id2< -1||id2>=ngeom))){failed=true;break;}
                if(leaf1&&leaf2&&id1!=-1&&id2!=-1){
                    if(tree_sphere(id1,id2,rbound,gtype,geom_pos,geom_mat,w*ngeom*3,w*ngeom*9) &&
                       (tree_obb(geom_aa,6*id1,6*id2,geom_pos,w*ngeom*3+3*id1,w*ngeom*3+3*id2,
                        geom_mat,w*ngeom*9+9*id1,w*ngeom*9+9*id2,0,products,offsets,0,initialize)&1)){
                        ++lc;
                        if(!tree_suppressed(id1,id2,merged,start,end,pair1,pair2,pair_key_offsets,pair_key_indices,ngeom)){
                            if(rc>=capacity){failed=true;break;}
                            requests[2*rc]=id1;requests[2*rc+1]=id2;++rc;
                        }
                    }
                    continue;
                }
                const int obb=tree_obb(bvh_aa,6*a1,6*a2,xipos,w*nbody*3+3*b1,w*nbody*3+3*b2,
                    ximat,w*nbody*9+9*b1,w*nbody*9+9*b2,1,products,offsets,0,initialize);
                initialize=(obb>>1)&1;
                if(!(obb&1))continue;
                int split1=0;
                if(!leaf1&&leaf2)split1=1;
                else if(!(leaf1&&!leaf2)&&tree_surface(bvh_aa,6*a1)>tree_surface(bvh_aa,6*a2))split1=1;
                for(int i=0;i<2;++i){
                    const int c=split1?child[2*a1+i]:child[2*a2+i];
                    if(c!=-1){
                        if(nstack>=limit||nstack>=max_stack){failed=true;break;}
                        first[nstack]=split1?c:n1;second[nstack]=split1?n2:c;++nstack;++sc;
                    }
                }
                // Preserve termination for malformed/cyclic or oversized data;
                // the ordered fallback reports its original logical error.
                if(vc>visit_slots||sc>stack_slots||lc>leaf_slots||rc>req_slots)failed=true;
            }
        }
        meta[Requests]=rc;meta[Visits]=vc;meta[StackEvents]=sc;meta[Leaves]=lc;
        meta[Initialized]=!initialize;meta[Failed]=failed;
    }
    __syncthreads();
    if(tid==0){
        int rc=0,vc=0,sc=0,lc=0;
        for(int s=0;s<count;++s){
            int* meta=metadata+s*fields;
            meta[OutputBase]=rc;
            if(meta[Failed] || meta[Requests]>req_slots-rc || meta[Visits]>visit_slots-vc ||
               meta[StackEvents]>stack_slots-sc || meta[Leaves]>leaf_slots-lc){admitted=0;break;}
            rc+=meta[Requests];vc+=meta[Visits];sc+=meta[StackEvents];lc+=meta[Leaves];
            if(meta[Initialized])last_initialized=s;
        }
        if(admitted){req_count[w]=rc;visit_count[w]=vc;stack_count[w]=sc;leaf_count[w]=lc;status[w]=0;first_stage[w]=-1;}
    }
    __syncthreads();
    if(!admitted)return false;
    for(int s=tid;s<count;s+=blockDim.x){
        const int* meta=metadata+s*fields;
        const int* requests=private_requests+size_t(meta[UpperBase])*2;
        for(int r=0;r<meta[Requests];++r){
            int* output=out_req+(size_t(w)*req_slots+meta[OutputBase]+r)*4;
            output[0]=s;output[1]=requests[2*r];output[2]=requests[2*r+1];output[3]=0;
        }
    }
    if(last_initialized>=0){
        const real_t* values=coefficients+size_t(last_initialized)*48;
        for(int i=tid;i<48;i+=blockDim.x){
            if(i<36)product[w*36+i]=values[i];else offset[w*12+i-36]=values[i];
        }
    }
    return true;
}
} // namespace collision_cuda
