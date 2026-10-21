set(cuda_runtime_source "${FLASH_GPU}/cuda/runtime/runtime.cu")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${cuda_runtime_source}")
file(SHA256 "${cuda_runtime_source}" cuda_runtime_identity)
add_library(flash_cuda_runtime SHARED EXCLUDE_FROM_ALL "${cuda_runtime_source}")
target_compile_definitions(flash_cuda_runtime PRIVATE
  FLASH_RUNTIME_SOURCE_ID="${cuda_runtime_identity}")
set_target_properties(flash_cuda_runtime PROPERTIES PREFIX "" OUTPUT_NAME "cuda_runtime"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
