#pragma once
#include <stdint.h>
namespace position_reuse {
// State writers (Euler, reset, set_init_state) advance state_epoch. Model and
// mocap holders are immutable in this explicitly admitted native-batch owner.
__device__ void check(int w,const int* active,const int* first,
 const int64_t* state,const int64_t* saved,const int* cached,const int* status,
 const int* committed,const int* collision_error,const int* mass_warning,
 const int* collision_warning,int* work,int* hit,
 const int64_t* forward,int* valid,int64_t* written,int* collision_mask,
 int* transport_active,uint64_t* consumed) {
 const bool admitted=active[w]==1 && !first[w];
 const bool reuse=admitted && cached[w] && state[w]==saved[w] &&
                  !status[w] && committed[w]==1 && !collision_error[w] &&
                  !mass_warning[w] && !collision_warning[w];
 hit[w]=reuse; work[w]=admitted && !reuse;
 if(!admitted)collision_mask[w]=transport_active[w]=0;
 if(reuse){
  valid[w]=1;written[w]=forward[w];
  collision_mask[w]=transport_active[w]=1;
  ++consumed[w];
 }
}
__device__ void predicate(int w,const int* work,int worlds,int* any) {
 if(w)return;any[0]=0;
 for(int i=0;i<worlds;i++)if(work[i]){any[0]=1;return;}
}
__device__ void store(int w,const int* work,const int* first,
 const int* valid,const int* status,const int* committed,const int* collision_error,
 const int* mass_warning,const int* collision_warning,
 const int64_t* state,int64_t* saved,int* cached,uint64_t* math) {
 if(!work[w])return;
 cached[w]=!first[w] && valid[w]==1 && !status[w] && committed[w]==1 &&
           !collision_error[w] && !mass_warning[w] && !collision_warning[w];
 if(cached[w]){saved[w]=state[w];++math[w];}
}
}
