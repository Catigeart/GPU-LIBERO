if(NOT TARGET flash_osc_solver_fp32)
  return()
endif()
set(fp32_controller_sources
  "${FLASH_GPU}/cuda/fp32/fused_controller.cu"
  "${FLASH_GPU}/cuda/fp32/fused_control_prepare.cuh"
  "${FLASH_GPU}/cuda/runtime/bridge_types.cuh"
  "${FLASH_GPU}/cuda/dynamics/mjwarp_queries.cuh"
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
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fp32_controller_sources})
set(controller_hashes)
foreach(source IN LISTS fp32_controller_sources)
  file(SHA256 "${source}" source_hash)
  list(APPEND controller_hashes "${source_hash}")
endforeach()
list(JOIN controller_hashes ":" controller_hash_input)
string(SHA256 controller_source_id "${controller_hash_input}")
if(FLASH_GLOBAL_DEVICE_LTO OR NOT FLASH_BUILD_NATIVE)
  set(controller_vendor "$<TARGET_OBJECTS:flash_osc_mathdx_fp32>")
  set(controller_vendor_target flash_osc_mathdx_fp32)
else()
  set(controller_vendor "${CMAKE_CURRENT_BINARY_DIR}/device_modules/osc_mathdx_fp32.o")
  set(controller_vendor_target flash_osc_mathdx_fp32_device)
endif()
add_library(flash_fp32_fused_controller SHARED EXCLUDE_FROM_ALL
  "${FLASH_GPU}/cuda/fp32/fused_controller.cu"
  $<TARGET_OBJECTS:flash_osc_solver_fp32> ${controller_vendor})
target_link_libraries(flash_fp32_fused_controller PRIVATE flash_cuda_options_fp32)
target_compile_definitions(flash_fp32_fused_controller PRIVATE
  FLASH_FP32_CONTROLLER_SOURCE_ID="${controller_source_id}" FLASH_CUDA_ARCH=${FLASH_CUDA_ARCH})
add_dependencies(flash_fp32_fused_controller ${controller_vendor_target})
set_target_properties(flash_fp32_fused_controller PROPERTIES PREFIX "" OUTPUT_NAME "fp32_fused_controller"
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" UNITY_BUILD OFF
  JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
if(FLASH_GLOBAL_DEVICE_LTO OR NOT FLASH_BUILD_NATIVE)
  target_link_options(flash_fp32_fused_controller PRIVATE "$<DEVICE_LINK:-dlto>")
  target_link_libraries(flash_fp32_fused_controller PRIVATE "${FLASH_SOLVERDX}")
endif()
