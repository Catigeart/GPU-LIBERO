// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned forward.py::_next_time_builder. Warmstart copy remains stream ordered.
#include "mjwarp_publication.cuh"
#include <cstdio>
namespace flash::mjwarp {
__global__ void publish_state(PublicationInputs a) {
  for (long long index=blockIdx.x*blockDim.x+threadIdx.x;index<a.worlds;index+=gridDim.x*blockDim.x) {
    int world=int(index),count=a.count[world],bits=a.overflow[world];
    a.time[world]+=a.timestep[world%a.timestep_rows];
    if (count>a.rows) {
      if (a.warnings&1) printf("nefc overflow - please increase njmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NEFC (or = 0 for all)\n",a.rows);
      bits|=1;
    } else if (count>0 && a.sparse) {
      int last=world*a.rows+count-1;
      if (a.rowadr[last]+a.rownnz[last]>a.nonzeros) {
        if (a.warnings&2) printf("njmax_nnz overflow - please increase njmax_nnz beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NJMAX_NNZ (or = 0 for all)\n",a.nonzeros);
        bits|=2;
      }
    }
    if (a.candidate_count[0]>a.contacts) {
      if (world==0 && (a.warnings&4)) printf("broadphase overflow - please increase nconmax beyond %u or naconmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.BROADPHASE (or = 0 for all)\n",a.contacts/a.worlds,a.contacts);
      bits|=4;
    }
    if (a.contact_count[0]>a.contacts) {
      if (world==0 && (a.warnings&8)) printf("narrowphase overflow - please increase nconmax beyond %u or naconmax beyond %u\nTo disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n",a.contacts/a.worlds,a.contacts);
      bits|=8;
    }
    a.overflow[world]=bits;
  }
}
}
extern "C" {
const char* flash_mjwarp_publication_source_id() {return FLASH_MJW_PUBLICATION_SOURCE_ID;}
int flash_mjwarp_publication_arch() {return FLASH_MJW_PUBLICATION_ARCH;}
std::size_t flash_mjwarp_publication_arguments_bytes() {return sizeof(flash::mjwarp::PublicationInputs);}
int flash_mjwarp_publication_launch(const flash::mjwarp::PublicationInputs* a,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->timestep_rows<1 || a->rows<0 || a->contacts<0 || a->nonzeros<0) return int(cudaErrorInvalidValue);
  flash::mjwarp::publish_state<<<min((a->worlds+127)/128,4096),128,0,stream>>>(*a);
  return int(cudaGetLastError());
}
}
