# Optional FP32 CRB / CSR mass assembly. No dependency on native physics TUs.
set(mjwarp_mass_matrix_sources
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_matrix.cu"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_matrix.cuh"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${mjwarp_mass_matrix_sources})
set(mass_matrix_hashes)
foreach(source IN LISTS mjwarp_mass_matrix_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND mass_matrix_hashes "${source_hash}")
endforeach()
list(JOIN mass_matrix_hashes ":" mass_matrix_hash_input)
string(SHA256 mass_matrix_source_id "${mass_matrix_hash_input}")
add_library(flash_mjwarp_mass_matrix_fp32 SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_matrix.cu")
target_compile_definitions(flash_mjwarp_mass_matrix_fp32 PRIVATE
  FLASH_MJW_MASS_MATRIX_SOURCE_ID="${mass_matrix_source_id}" FLASH_MJW_MASS_MATRIX_ARCH=${FLASH_CUDA_ARCH})
target_compile_options(flash_mjwarp_mass_matrix_fp32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
set_target_properties(flash_mjwarp_mass_matrix_fp32 PROPERTIES
  PREFIX "" OUTPUT_NAME "mjwarp_mass_matrix_fp32" CUDA_ARCHITECTURES ${FLASH_CUDA_ARCH}
  UNITY_BUILD OFF JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
