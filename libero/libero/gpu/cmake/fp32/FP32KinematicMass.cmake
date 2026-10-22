set(fp32_kinematic_mass_sources
  "${FLASH_GPU}/cuda/fp32/fused_kinematic_mass.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_kinematics.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_camlight.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_com_pos.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_matrix.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_kinematic_mass_sources})
set(kinematic_mass_hashes)
foreach(source IN LISTS fp32_kinematic_mass_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND kinematic_mass_hashes "${source_hash}")
endforeach()
list(JOIN kinematic_mass_hashes ":" kinematic_mass_hash_input)
string(SHA256 kinematic_mass_source_id "${kinematic_mass_hash_input}")
add_library(flash_fp32_fused_kinematic_mass SHARED EXCLUDE_FROM_ALL "${FLASH_GPU}/cuda/fp32/fused_kinematic_mass.cu")
target_compile_definitions(flash_fp32_fused_kinematic_mass PRIVATE
  FLASH_FP32_KINEMATIC_MASS_SOURCE_ID="${kinematic_mass_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_fp32_fused_kinematic_mass PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_fp32_fused_kinematic_mass PROPERTIES PREFIX "" OUTPUT_NAME "fp32_fused_kinematic_mass"
  CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH} UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
