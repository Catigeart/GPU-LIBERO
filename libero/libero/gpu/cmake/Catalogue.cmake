# The default project build contains production targets only. Compiler profile
# selection is a CMake build-directory setting, not another task enumeration.
set(catalogue_targets)
if(TARGET flash_native_batch_all)
  list(APPEND catalogue_targets flash_native_batch_all)
endif()
if(TARGET flash_mjwarp_controller_fp32)
  set(fp32_components controller hessian direction constraint linesearch solver_state
    mass_solve smooth_force mass_factor mass_matrix velocity bias postconstraint
    passive actuation euler transmission kinematics com_pos camlight fluid external_wrench
    constraint_build broadphase plane box convex queries publication)
  set(fp32_targets flash_cuda_runtime flash_cuda_bridges_fp32)
  foreach(component IN LISTS fp32_components)
    list(APPEND fp32_targets flash_mjwarp_${component}_fp32)
  endforeach()
  foreach(component tail observations position kinematic_mass motion motion_passive
      control_prepare controller actuation_solve euler_step control_drive)
    list(APPEND fp32_targets flash_fp32_fused_${component})
  endforeach()
  add_custom_target(flash_cuda_fp32_all DEPENDS ${fp32_targets})
  list(APPEND catalogue_targets flash_cuda_fp32_all)
endif()
add_custom_target(flash_gpu_all ALL DEPENDS ${catalogue_targets})
