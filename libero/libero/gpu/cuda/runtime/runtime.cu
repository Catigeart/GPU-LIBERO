// Copyright 2026 Flash-LIBERO contributors
// Independent CUDA resource and execution ownership. No Warp dependency.
#include <cuda_runtime.h>
#include <cstddef>
#include <new>

namespace {
struct WhileCapture {
  cudaGraph_t parent;
  cudaGraphNode_t node;
  cudaGraphConditionalHandle handle;
  const int* counter;
};
__global__ void condition_from_counter(cudaGraphConditionalHandle handle,const int* counter) {
  cudaGraphSetConditional(handle,*counter!=0);
}
__global__ void store_integer(int* pointer,int value) { *pointer=value; }
__global__ void decrement_integer(int* pointer) { --*pointer; }
}

extern "C" {
const char* flash_runtime_source_id() { return FLASH_RUNTIME_SOURCE_ID; }
const char* flash_runtime_error(int error) { return cudaGetErrorString(cudaError_t(error)); }
int flash_runtime_device(int ordinal, int* arch, std::size_t* free_bytes,
                         std::size_t* total_bytes) {
  cudaError_t error=cudaSetDevice(ordinal);
  if (error!=cudaSuccess) return error;
  cudaDeviceProp properties;
  error=cudaGetDeviceProperties(&properties,ordinal);
  if (error!=cudaSuccess) return error;
  *arch=10*properties.major+properties.minor;
  return cudaMemGetInfo(free_bytes,total_bytes);
}
int flash_runtime_attribute(int ordinal,int attribute,int* value) {
  return cudaDeviceGetAttribute(value,cudaDeviceAttr(attribute),ordinal);
}
int flash_runtime_allocate(void** pointer,std::size_t bytes) {
  if (!bytes) { *pointer=nullptr; return cudaSuccess; }
  return cudaMalloc(pointer,bytes);
}
int flash_runtime_free(void* pointer) { return cudaFree(pointer); }
int flash_runtime_stream_create(cudaStream_t* stream) {
  return cudaStreamCreateWithFlags(stream,cudaStreamNonBlocking);
}
int flash_runtime_stream_destroy(cudaStream_t stream) { return cudaStreamDestroy(stream); }
int flash_runtime_synchronize(cudaStream_t stream) { return cudaStreamSynchronize(stream); }
int flash_runtime_copy(void* target,const void* source,std::size_t bytes,int direction,cudaStream_t stream) {
  return bytes ? cudaMemcpyAsync(target,source,bytes,cudaMemcpyKind(direction),stream) : cudaSuccess;
}
int flash_runtime_zero(void* target,std::size_t bytes,cudaStream_t stream) {
  return bytes ? cudaMemsetAsync(target,0,bytes,stream) : cudaSuccess;
}
int flash_runtime_capture_begin(cudaStream_t stream) {
  return cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal);
}
int flash_runtime_capture_end(cudaStream_t stream,cudaGraph_t* graph) {
  return cudaStreamEndCapture(stream,graph);
}
int flash_runtime_graph_instantiate(cudaGraph_t graph,cudaGraphExec_t* executable) {
  return cudaGraphInstantiate(executable,graph,0);
}
int flash_runtime_graph_launch(cudaGraphExec_t executable,cudaStream_t stream) {
  return cudaGraphLaunch(executable,stream);
}
int flash_runtime_graph_destroy(cudaGraph_t graph,cudaGraphExec_t executable) {
  cudaError_t first=executable ? cudaGraphExecDestroy(executable) : cudaSuccess;
  cudaError_t second=graph ? cudaGraphDestroy(graph) : cudaSuccess;
  return first!=cudaSuccess ? first : second;
}
int flash_runtime_event_create(cudaEvent_t* event) { return cudaEventCreate(event); }
int flash_runtime_event_record(cudaEvent_t event,cudaStream_t stream) { return cudaEventRecord(event,stream); }
int flash_runtime_event_elapsed(cudaEvent_t first,cudaEvent_t last,float* milliseconds) {
  cudaError_t error=cudaEventSynchronize(last);
  return error!=cudaSuccess ? error : cudaEventElapsedTime(milliseconds,first,last);
}
int flash_runtime_event_destroy(cudaEvent_t event) { return cudaEventDestroy(event); }
int flash_runtime_store_integer(int* pointer,int value,cudaStream_t stream) {
  store_integer<<<1,1,0,stream>>>(pointer,value);return cudaGetLastError();
}
int flash_runtime_decrement_integer(int* pointer,cudaStream_t stream) {
  decrement_integer<<<1,1,0,stream>>>(pointer);return cudaGetLastError();
}
// Populate a conditional body on the same stream by ending and resuming the
// parent capture. This is CUDA's graph ownership API, with no host polling.
int flash_runtime_while_begin(cudaStream_t stream,const int* counter,void** output) {
  auto* scope=new(std::nothrow) WhileCapture{};
  if (!scope) return cudaErrorMemoryAllocation;
  cudaStreamCaptureStatus status;const cudaGraphNode_t* dependencies=nullptr;
  std::size_t count=0;
  cudaError_t error=cudaStreamGetCaptureInfo(stream,&status,nullptr,&scope->parent,&dependencies,nullptr,&count);
  if (error!=cudaSuccess || status!=cudaStreamCaptureStatusActive) {
    delete scope;return error!=cudaSuccess ? error : cudaErrorStreamCaptureUnsupported;
  }
  scope->counter=counter;
  error=cudaGraphConditionalHandleCreate(&scope->handle,scope->parent);
  if (error!=cudaSuccess) { delete scope;return error; }
  condition_from_counter<<<1,1,0,stream>>>(scope->handle,counter);
  error=cudaGetLastError();if (error!=cudaSuccess) { delete scope;return error; }
  error=cudaStreamGetCaptureInfo(stream,&status,nullptr,nullptr,&dependencies,nullptr,&count);
  if (error!=cudaSuccess) { delete scope;return error; }
  cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
  parameters.conditional.handle=scope->handle;parameters.conditional.type=cudaGraphCondTypeWhile;
  parameters.conditional.size=1;
  error=cudaGraphAddNode(&scope->node,scope->parent,dependencies,nullptr,count,&parameters);
  if (error!=cudaSuccess) { delete scope;return error; }
  error=cudaStreamUpdateCaptureDependencies(stream,&scope->node,nullptr,1,cudaStreamSetCaptureDependencies);
  if (error!=cudaSuccess) { delete scope;return error; }
  cudaGraph_t paused=nullptr;
  error=cudaStreamEndCapture(stream,&paused);
  if (error==cudaSuccess)
    error=cudaStreamBeginCaptureToGraph(stream,parameters.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);
  if (error!=cudaSuccess) { delete scope;return error; }
  *output=scope;return cudaSuccess;
}
int flash_runtime_while_end(cudaStream_t stream,void* pointer) {
  auto* scope=static_cast<WhileCapture*>(pointer);
  condition_from_counter<<<1,1,0,stream>>>(scope->handle,scope->counter);
  cudaError_t error=cudaGetLastError();cudaGraph_t child=nullptr;
  if (error==cudaSuccess) error=cudaStreamEndCapture(stream,&child);
  if (error==cudaSuccess)
    error=cudaStreamBeginCaptureToGraph(stream,scope->parent,&scope->node,nullptr,1,cudaStreamCaptureModeThreadLocal);
  delete scope;return error;
}
}
