# Independent sphere/capsule-box and primitive box-box contacts.
set(mjwarp_box_sources "${FLASH_GPU}/cuda/collision/mjwarp_box.cu"
  "${FLASH_GPU}/cuda/collision/mjwarp_box.cuh" "${FLASH_GPU}/cuda/collision/mjwarp_contact.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_box_sources})
set(box_hashes)
foreach(source IN LISTS mjwarp_box_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND box_hashes "${source_hash}")
endforeach()
list(JOIN box_hashes ":" box_hash_input)
string(SHA256 box_source_id "${box_hash_input}")
add_library(flash_mjwarp_box_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/collision/mjwarp_box.cu")
target_compile_definitions(flash_mjwarp_box_fp32 PRIVATE
  FLASH_MJW_BOX_SOURCE_ID="${box_source_id}" FLASH_MJW_BOX_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_box_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_box_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_box_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
