# Optional FP32 mass backsubstitution; independently compiled and device-linked.
set(fp32_actuation_solve_sources
  "${FLASH_GPU}/cuda/fp32/fused_actuation_solve.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_actuation.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_smooth_force.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_solve.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_actuation_solve_sources})
set(actuation_solve_hashes)
foreach(source IN LISTS fp32_actuation_solve_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND actuation_solve_hashes "${source_hash}")
endforeach()
list(JOIN actuation_solve_hashes ":" actuation_solve_hash_input)
string(SHA256 actuation_solve_source_id "${actuation_solve_hash_input}")
add_library(flash_fp32_fused_actuation_solve SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/fp32/fused_actuation_solve.cu")
target_include_directories(flash_fp32_fused_actuation_solve PRIVATE "${FLASH_MATHDX_ROOT}/include"
  "${FLASH_MATHDX_ROOT}/external/cutlass/include")
target_compile_definitions(flash_fp32_fused_actuation_solve PRIVATE FLASH_FP32_ACTUATION_SOLVE_SOURCE_ID="${actuation_solve_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_fp32_fused_actuation_solve PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-arch=lto_${FLASH_CUDA_ARCH};-dlto;-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
target_link_options(flash_fp32_fused_actuation_solve PRIVATE "$<DEVICE_LINK:-dlto;-arch=sm_${FLASH_CUDA_ARCH}>")
target_link_libraries(flash_fp32_fused_actuation_solve PRIVATE "${FLASH_SOLVERDX}")
set_target_properties(flash_fp32_fused_actuation_solve PROPERTIES
  PREFIX "" OUTPUT_NAME "fp32_fused_actuation_solve" CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_ARCHITECTURES OFF
  UNITY_BUILD OFF JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
