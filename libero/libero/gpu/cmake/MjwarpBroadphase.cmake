# Independent NXN candidate generation and collision bounding filters.
set(mjwarp_broadphase_sources "${FLASH_GPU}/cuda/collision/mjwarp_broadphase.cu"
  "${FLASH_GPU}/cuda/collision/mjwarp_broadphase.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_broadphase_sources})
set(broadphase_hashes)
foreach(source IN LISTS mjwarp_broadphase_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND broadphase_hashes "${source_hash}")
endforeach()
list(JOIN broadphase_hashes ":" broadphase_hash_input)
string(SHA256 broadphase_source_id "${broadphase_hash_input}")
add_library(flash_mjwarp_broadphase_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/collision/mjwarp_broadphase.cu")
target_compile_definitions(flash_mjwarp_broadphase_fp32 PRIVATE
  FLASH_MJW_BROADPHASE_SOURCE_ID="${broadphase_source_id}" FLASH_MJW_BROADPHASE_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_broadphase_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_broadphase_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_broadphase_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
