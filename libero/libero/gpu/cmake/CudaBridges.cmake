set(cuda_bridge_source "${FLASH_GPU}/cuda/runtime/bridges.cu")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${cuda_bridge_source}")
file(SHA256 "${cuda_bridge_source}" cuda_bridge_identity)
add_library(flash_cuda_bridges_fp32 SHARED EXCLUDE_FROM_ALL "${cuda_bridge_source}")
target_compile_definitions(flash_cuda_bridges_fp32 PRIVATE FLASH_BRIDGE_SOURCE_ID="${cuda_bridge_identity}")
target_compile_options(flash_cuda_bridges_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_cuda_bridges_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "cuda_bridges_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
