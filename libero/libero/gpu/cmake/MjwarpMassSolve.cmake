# Optional FP32 mass backsubstitution; independently compiled and device-linked.
set(mjwarp_mass_solve_sources
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_solve.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_solve.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_mass_solve_sources})
set(mass_solve_hashes)
foreach(source IN LISTS mjwarp_mass_solve_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND mass_solve_hashes "${source_hash}")
endforeach()
list(JOIN mass_solve_hashes ":" mass_solve_hash_input)
string(SHA256 mass_solve_source_id "${mass_solve_hash_input}")
add_library(flash_mjwarp_mass_solve_fp32 SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_solve.cu")
target_include_directories(flash_mjwarp_mass_solve_fp32 PRIVATE "${FLASH_MATHDX_ROOT}/include"
  "${FLASH_MATHDX_ROOT}/external/cutlass/include")
target_compile_definitions(flash_mjwarp_mass_solve_fp32 PRIVATE FLASH_MJW_MASS_SOLVE_SOURCE_ID="${mass_solve_source_id}" FLASH_MJW_MASS_SOLVE_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_mass_solve_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-arch=lto_${FLASH_CUDA_ARCH};-dlto;-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
target_link_options(flash_mjwarp_mass_solve_fp32 PRIVATE "$<DEVICE_LINK:-dlto;-arch=sm_${FLASH_CUDA_ARCH}>")
target_link_libraries(flash_mjwarp_mass_solve_fp32 PRIVATE "${FLASH_SOLVERDX}")
set_target_properties(flash_mjwarp_mass_solve_fp32 PROPERTIES
  PREFIX "" OUTPUT_NAME "mjwarp_mass_solve_fp32" CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_ARCHITECTURES OFF
  UNITY_BUILD OFF JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
