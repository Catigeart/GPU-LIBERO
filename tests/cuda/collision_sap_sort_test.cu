#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include "../../libero/libero/gpu/cuda/collision/collision_sap_sort.cuh"

constexpr int stride = 256, cases = 4096;

__global__ void events(float* keys, int* tags) {
    const int row=blockIdx.x, count=row%97;
    keys+=row*stride;tags+=row*stride;
    if(!collision_cuda::sap_events64(keys,tags,count)&&threadIdx.x==0)
        for(int i=1;i<count;++i){
            const float key=keys[i];const int tag=tags[i];int j=i;
            while(j>0&&key<keys[j-1]){keys[j]=keys[j-1];tags[j]=tags[j-1];--j;}
            keys[j]=key;tags[j]=tag;
        }
}

__global__ void pairs(int* values) {
    const int row=blockIdx.x,count=row%193;
    values+=row*stride;
    if(!collision_cuda::sap_pairs128(values,count)&&threadIdx.x==0)
        for(int i=1;i<count;++i){const int value=values[i];int j=i;
            while(j>0&&value<values[j-1]){values[j]=values[j-1];--j;}
            values[j]=value;
        }
}

int main() {
    std::mt19937 random(7192);
    std::vector<float> keys(cases*stride);
    std::vector<int> tags(cases*stride),pair(cases*stride);
    for(int row=0;row<cases;++row)for(int k=0;k<stride;++k){
        const int at=row*stride+k;
        keys[at]=float(int(random()%17)-8);
        if(k%11==0)keys[at]=-0.0f;
        if(k%13==0)keys[at]=std::numeric_limits<float>::infinity();
        if(k%19==0)keys[at]=-std::numeric_limits<float>::infinity();
        if(row%7==0&&k%23==0)keys[at]=std::numeric_limits<float>::quiet_NaN();
        tags[at]=int(random()%100000);
        pair[at]=int(random()%113)-56;
        if(k%17==0)pair[at]=INT_MAX;
        if(k%21==0)pair[at]=INT_MIN;
    }
    auto expected_keys=keys;
    auto expected_tags=tags,expected_pairs=pair;
    for(int row=0;row<cases;++row){
        const int base=row*stride;
        for(int i=1;i<row%97;++i){
            const float key=expected_keys[base+i];const int tag=expected_tags[base+i];int j=i;
            while(j>0&&key<expected_keys[base+j-1]){
                expected_keys[base+j]=expected_keys[base+j-1];
                expected_tags[base+j]=expected_tags[base+j-1];--j;
            }
            expected_keys[base+j]=key;expected_tags[base+j]=tag;
        }
        for(int i=1;i<row%193;++i){const int v=expected_pairs[base+i];int j=i;
            while(j>0&&v<expected_pairs[base+j-1]){
                expected_pairs[base+j]=expected_pairs[base+j-1];--j;
            }
            expected_pairs[base+j]=v;
        }
    }
    float* device_keys=nullptr;int *device_tags=nullptr,*device_pairs=nullptr;
    const size_t bytes=keys.size()*sizeof(float);
    if(cudaMalloc(&device_keys,bytes)!=cudaSuccess||cudaMalloc(&device_tags,bytes)!=cudaSuccess||
       cudaMalloc(&device_pairs,bytes)!=cudaSuccess)return 3;
    cudaMemcpy(device_keys,keys.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(device_tags,tags.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(device_pairs,pair.data(),bytes,cudaMemcpyHostToDevice);
    events<<<cases,32>>>(device_keys,device_tags);
    pairs<<<cases,32>>>(device_pairs);
    if(cudaDeviceSynchronize()!=cudaSuccess)return 4;
    cudaMemcpy(keys.data(),device_keys,bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(tags.data(),device_tags,bytes,cudaMemcpyDeviceToHost);
    cudaMemcpy(pair.data(),device_pairs,bytes,cudaMemcpyDeviceToHost);
    const bool passed=std::memcmp(keys.data(),expected_keys.data(),bytes)==0&&
        tags==expected_tags&&pair==expected_pairs;
    cudaFree(device_keys);cudaFree(device_tags);cudaFree(device_pairs);
    std::printf("{\"passed\":%s,\"event_cases\":4096,\"pair_cases\":4096,"
                "\"checks\":\"stable ties, signed zero, infinity, NaN fallback, size fallback, tails\"}\n",
                passed?"true":"false");
    return passed?0:2;
}
