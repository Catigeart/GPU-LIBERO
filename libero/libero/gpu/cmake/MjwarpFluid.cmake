# Independent inertia-box fluid force and generalized-force projection.
set(mjwarp_fluid_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_fluid.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_fluid.cuh" "${FLASH_GPU}/cuda/dynamics/mjwarp_passive.cuh" "${FLASH_GPU}/cuda/dynamics/mjwarp_passive_math.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_fluid_sources})
set(fluid_hashes)
foreach(source IN LISTS mjwarp_fluid_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND fluid_hashes "${source_hash}")
endforeach()
list(JOIN fluid_hashes ":" fluid_hash_input)
string(SHA256 fluid_source_id "${fluid_hash_input}")
add_library(flash_mjwarp_fluid_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_fluid.cu")
target_compile_definitions(flash_mjwarp_fluid_fp32 PRIVATE
  FLASH_MJW_FLUID_SOURCE_ID="${fluid_source_id}" FLASH_MJW_FLUID_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_fluid_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_fluid_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_fluid_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
