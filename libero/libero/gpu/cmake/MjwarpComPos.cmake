# Independent subtree CoM, centered spatial inertia and joint motion axes.
set(mjwarp_com_pos_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_com_pos.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_com_pos.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_com_pos_sources})
set(com_pos_hashes)
foreach(source IN LISTS mjwarp_com_pos_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND com_pos_hashes "${source_hash}")
endforeach()
list(JOIN com_pos_hashes ":" com_pos_hash_input)
string(SHA256 com_pos_source_id "${com_pos_hash_input}")
add_library(flash_mjwarp_com_pos_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_com_pos.cu")
target_compile_definitions(flash_mjwarp_com_pos_fp32 PRIVATE
  FLASH_MJW_COM_POS_SOURCE_ID="${com_pos_source_id}" FLASH_MJW_COM_POS_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_com_pos_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_com_pos_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_com_pos_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
