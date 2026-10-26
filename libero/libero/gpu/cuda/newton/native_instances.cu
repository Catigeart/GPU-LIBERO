#include "native_kernel.cuh"

// CMake compiles this explicit instantiation for each LiberoModel in
// libero_models.def. All instances are built by flash_native_newton_all.
template __global__ void
libero_native::native_newton<NativeShape>(const libero_native::Parameters *, const int *,
                                          const int *, const int *, const int64_t *, int);

extern "C" int launch_persistent(const void *, int, void *) { return int(cudaErrorInvalidValue); }
extern "C" int launch_entry(const void *p, int worlds, const int *upstream, const int *enabled,
                            const int *publication, const int64_t *plan, int supplied,
                            void *stream) {
  libero_native::native_newton<NativeShape>
      <<<dim3(1, worlds), 128, NativeLayout::shared_bytes, (cudaStream_t)stream>>>(
          static_cast<const libero_native::Parameters *>(p), upstream, enabled, publication, plan,
          supplied);
  return int(cudaGetLastError());
}
extern "C" int native_newton_configuration(int field) {
  const int values[] = {NV, NM, ROWS, 5000, 100, 128, 64, DX_SM / 10, 1, 1, 1, 0};
  return field >= 0 && field < int(sizeof(values) / sizeof(values[0])) ? values[field] : -1;
}
extern "C" int newton_kernel_info(size_t *out, size_t count) {
  if (!out || count != 11)
    return int(cudaErrorInvalidValue);
  cudaFuncAttributes attr{};
  auto kernel = libero_native::native_newton<NativeShape>;
  cudaError_t err = cudaFuncGetAttributes(&attr, kernel);
  if (err != cudaSuccess)
    return int(err);
  int device = 0, shared_limit = 0, thread_limit = 0;
  err = cudaGetDevice(&device);
  if (err != cudaSuccess)
    return int(err);
  err = cudaDeviceGetAttribute(&shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device);
  if (err != cudaSuccess)
    return int(err);
  err = cudaDeviceGetAttribute(&thread_limit, cudaDevAttrMaxThreadsPerBlock, device);
  if (err != cudaSuccess)
    return int(err);
  out[0] = 1;
  out[1] = sizeof(libero_native::Parameters);
  out[2] = 128;
  out[3] = NativeLayout::shared_bytes;
  out[4] = DX_SM / 10;
  out[5] = attr.numRegs;
  out[6] = attr.localSizeBytes;
  out[7] = attr.sharedSizeBytes;
  out[8] = attr.maxThreadsPerBlock;
  out[9] = shared_limit;
  out[10] = thread_limit;
  return int(cudaSuccess);
}
