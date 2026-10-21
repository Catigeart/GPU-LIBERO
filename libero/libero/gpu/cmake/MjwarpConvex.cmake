# Convex support building blocks; no production collision selection yet.
set(mjwarp_convex_sources "${FLASH_GPU}/cuda/collision/mjwarp_convex.cu"
  "${FLASH_GPU}/cuda/collision/mjwarp_convex.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_convex_geometry.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_convex_simplex.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_convex_gjk.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_convex_epa.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_box_manifold.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_convex_collision.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_contact.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_convex_sources})
set(convex_hashes)
foreach(source IN LISTS mjwarp_convex_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND convex_hashes "${source_hash}")
endforeach()
list(JOIN convex_hashes ":" convex_hash_input)
string(SHA256 convex_source_id "${convex_hash_input}")
add_library(flash_mjwarp_convex_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/collision/mjwarp_convex.cu")
target_compile_definitions(flash_mjwarp_convex_fp32 PRIVATE
  FLASH_MJW_CONVEX_SOURCE_ID="${convex_source_id}" FLASH_MJW_CONVEX_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_convex_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_convex_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_convex_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
