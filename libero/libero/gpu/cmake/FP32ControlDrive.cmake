if(NOT TARGET flash_osc_solver_fp32)
  return()
endif()
set(fp32_control_drive_sources
  "${FLASH_GPU}/cuda/fp32/fused_control_drive.cu"
  "${FLASH_GPU}/cuda/fp32/fused_control_prepare.cuh"
  "${FLASH_GPU}/cuda/runtime/bridge_types.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_queries.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_actuation.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_smooth_force.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_mass_solve.cuh"
  "${FLASH_GPU}/cuda/control/osc_solver.cu"
  "${FLASH_GPU}/cuda/control/osc_solver.cuh"
  "${FLASH_GPU}/cuda/control/osc_profile.cuh"
  "${FLASH_GPU}/cuda/step/substep_shared_layout.cuh"
  "${FLASH_GPU}/cuda/step/substep_shared.cuh"
  "${FLASH_GPU}/cuda/step/forward_workspace.cuh"
  "${FLASH_GPU}/cuda/step/substep_shape.cuh"
  "${FLASH_GPU}/cuda/step/substep_parameters.cuh"
  "${FLASH_GPU}/cuda/step/substep_types.cuh"
  "${FLASH_GPU}/cuda/common/precision.cuh"
  "${FLASH_GPU}/cuda/common/vector.cuh"
  "${FLASH_GPU}/cuda/step/instantiations/osc_mathdx.cu"
  "${CMAKE_CURRENT_LIST_FILE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_control_drive_sources})
set(control_drive_hashes)
foreach(source IN LISTS fp32_control_drive_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND control_drive_hashes "${source_hash}")
endforeach()
list(JOIN control_drive_hashes ":" control_drive_hash_input)
string(SHA256 control_drive_source_id "${control_drive_hash_input}")
if(FLASH_GLOBAL_DEVICE_LTO)
  set(control_drive_vendor "$<TARGET_OBJECTS:flash_osc_mathdx_fp32>")
  set(control_drive_vendor_target flash_osc_mathdx_fp32)
else()
  set(control_drive_vendor "${CMAKE_CURRENT_BINARY_DIR}/device_modules/osc_mathdx_fp32.o")
  set(control_drive_vendor_target flash_osc_mathdx_fp32_device)
endif()
add_library(flash_fp32_fused_control_drive SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/fp32/fused_control_drive.cu"
  $<TARGET_OBJECTS:flash_osc_solver_fp32> ${control_drive_vendor})
target_link_libraries(flash_fp32_fused_control_drive PRIVATE flash_cuda_options_fp32)
target_include_directories(flash_fp32_fused_control_drive PRIVATE "${FLASH_MATHDX_ROOT}/include"
  "${FLASH_MATHDX_ROOT}/external/cutlass/include")
target_compile_options(flash_fp32_fused_control_drive PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-arch=lto_${FLASH_CUDA_ARCH};-dlto>")
target_link_options(flash_fp32_fused_control_drive PRIVATE "$<DEVICE_LINK:-dlto;-arch=sm_${FLASH_CUDA_ARCH}>")
target_link_libraries(flash_fp32_fused_control_drive PRIVATE "${FLASH_SOLVERDX}")
target_compile_definitions(flash_fp32_fused_control_drive PRIVATE
  FLASH_FP32_CONTROL_DRIVE_SOURCE_ID="${control_drive_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
add_dependencies(flash_fp32_fused_control_drive ${control_drive_vendor_target})
set_target_properties(flash_fp32_fused_control_drive PROPERTIES PREFIX "" OUTPUT_NAME "fp32_fused_control_drive"
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES OFF UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
if(FLASH_GLOBAL_DEVICE_LTO)
  target_link_options(flash_fp32_fused_control_drive PRIVATE "$<DEVICE_LINK:-dlto;-arch=sm_${FLASH_CUDA_ARCH}>")
  target_link_libraries(flash_fp32_fused_control_drive PRIVATE "${FLASH_SOLVERDX}")
endif()
