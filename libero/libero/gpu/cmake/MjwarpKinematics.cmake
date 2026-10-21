# Independent optional rigid-body, inertial, geom and site kinematics.
set(mjwarp_kinematics_sources "${FLASH_GPU}/cuda/dynamics/mjwarp_kinematics.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_kinematics.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_camlight.cuh" "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_kinematics_sources})
set(kinematics_hashes)
foreach(source IN LISTS mjwarp_kinematics_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND kinematics_hashes "${source_hash}")
endforeach()
list(JOIN kinematics_hashes ":" kinematics_hash_input)
string(SHA256 kinematics_source_id "${kinematics_hash_input}")
add_library(flash_mjwarp_kinematics_fp32 SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/dynamics/mjwarp_kinematics.cu")
target_compile_definitions(flash_mjwarp_kinematics_fp32 PRIVATE
  FLASH_MJW_KINEMATICS_SOURCE_ID="${kinematics_source_id}" FLASH_MJW_KINEMATICS_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_kinematics_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_kinematics_fp32 PROPERTIES PREFIX "" OUTPUT_NAME "mjwarp_kinematics_fp32"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
