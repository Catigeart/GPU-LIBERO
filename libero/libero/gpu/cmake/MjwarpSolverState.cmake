# Optional dense FP32 Newton initialization and stopping, independently compiled.
set(mjwarp_solver_state_sources
  "${FLASH_GPU}/cuda/newton/mjwarp_solver_state.cu"
  "${FLASH_GPU}/cuda/newton/mjwarp_solver_state.cuh"
  "${FLASH_GPU}/cuda/newton/mjwarp_products.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_solver_state_sources})
set(solver_state_hashes)
foreach(source IN LISTS mjwarp_solver_state_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND solver_state_hashes "${source_hash}")
endforeach()
list(JOIN solver_state_hashes ":" solver_state_hash_input)
string(SHA256 solver_state_source_id "${solver_state_hash_input}")
add_library(flash_mjwarp_solver_state_fp32 SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/newton/mjwarp_solver_state.cu")
target_compile_definitions(flash_mjwarp_solver_state_fp32 PRIVATE
  FLASH_MJW_SOLVER_STATE_SOURCE_ID="${solver_state_source_id}" FLASH_MJW_SOLVER_STATE_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_solver_state_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_solver_state_fp32 PROPERTIES
  PREFIX "" OUTPUT_NAME "mjwarp_solver_state_fp32" CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
