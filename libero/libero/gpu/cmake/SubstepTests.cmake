# Optional repository fixtures; installed GPU packages configure without them.
function(flash_add_newton_cone_fixture suffix precision_bits mathdx_input mathdx_target)
  set(source "${FLASH_GPU}/../../../tests/cuda/newton_cone_factor_test.cu")
  if(NOT EXISTS "${source}")
    return()
  endif()
  flash_precision_suffix(${precision_bits} precision_suffix)
  set(target flash_newton_cone_factor_test${suffix}${precision_suffix})
  add_executable(${target} EXCLUDE_FROM_ALL "${source}" ${mathdx_input}
    $<TARGET_OBJECTS:flash_newton_atomic${precision_suffix}>
    $<TARGET_OBJECTS:flash_newton_transpose${precision_suffix}>
    $<TARGET_OBJECTS:flash_newton_jacobian_cache${precision_suffix}>)
  target_include_directories(${target} PRIVATE "${FLASH_STEP}")
  target_compile_definitions(${target} PRIVATE ${ARGN})
  target_compile_options(${target} PRIVATE --split-compile=3)
  target_link_libraries(${target} PRIVATE flash_cuda_options${precision_suffix})
  add_dependencies(${target} ${mathdx_target})
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
  if(FLASH_GLOBAL_DEVICE_LTO)
    target_link_options(${target} PRIVATE "$<DEVICE_LINK:-dlto>")
    target_link_libraries(${target} PRIVATE "${FLASH_SOLVERDX}")
  endif()
endfunction()

function(flash_add_newton_line_fixture precision_bits mathdx_input mathdx_target)
  set(source "${FLASH_GPU}/../../../tests/cuda/newton_linesearch_products_test.cu")
  if(NOT precision_bits STREQUAL "64" OR NOT EXISTS "${source}")
    return()
  endif()
  set(target flash_newton_linesearch_products_test)
  add_executable(${target} EXCLUDE_FROM_ALL "${source}" ${mathdx_input}
    $<TARGET_OBJECTS:flash_newton_atomic> $<TARGET_OBJECTS:flash_newton_transpose>
    $<TARGET_OBJECTS:flash_newton_jacobian_cache>)
  target_include_directories(${target} PRIVATE "${FLASH_STEP}")
  target_compile_definitions(${target} PRIVATE ${ARGN})
  target_compile_options(${target} PRIVATE --split-compile=3)
  target_link_libraries(${target} PRIVATE flash_cuda_options)
  add_dependencies(${target} ${mathdx_target})
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
  if(FLASH_GLOBAL_DEVICE_LTO)
    target_link_options(${target} PRIVATE "$<DEVICE_LINK:-dlto>")
    target_link_libraries(${target} PRIVATE "${FLASH_SOLVERDX}")
  endif()
endfunction()

# Optional fixed-input diagnostics; normal products never depend on this target.
function(flash_add_newton_replay_probe suffix precision_bits mathdx_input mathdx_target osc_input osc_target)
  cmake_parse_arguments(PROBE "PROFILE" "" "" ${ARGN})
  set(source "${FLASH_GPU}/../../../tests/cuda/newton_replay_probe.cu")
  if(NOT EXISTS "${source}")
    return()
  endif()
  flash_precision_suffix(${precision_bits} precision_suffix)
  set(profile_suffix)
  set(solver flash_newton_solver${precision_suffix}${suffix})
  if(PROBE_PROFILE)
    set(profile_suffix _profile)
    set(solver flash_newton_solver_profile${precision_suffix}${suffix})
  endif()
  set(target flash_newton_replay_probe${profile_suffix}${suffix}${precision_suffix})
  set(forward_inputs)
  if(suffix STREQUAL "" AND NOT PROBE_PROFILE)
    list(APPEND forward_inputs ${osc_input}
      $<TARGET_OBJECTS:flash_post_solver${precision_suffix}>
      $<TARGET_OBJECTS:flash_osc_solver${precision_suffix}>
      $<TARGET_OBJECTS:flash_controller_pipeline${precision_suffix}>)
  endif()
  add_library(${target} SHARED EXCLUDE_FROM_ALL "${source}" ${mathdx_input} ${forward_inputs}
    $<TARGET_OBJECTS:${solver}>
    $<TARGET_OBJECTS:flash_newton_atomic${precision_suffix}>
    $<TARGET_OBJECTS:flash_newton_transpose${precision_suffix}>
    $<TARGET_OBJECTS:flash_newton_jacobian_cache${precision_suffix}>)
  target_include_directories(${target} PRIVATE "${FLASH_STEP}"
    "${FLASH_MATHDX_ROOT}/include" "${FLASH_MATHDX_ROOT}/external/cutlass/include")
  target_compile_definitions(${target} PRIVATE ${PROBE_UNPARSED_ARGUMENTS})
  if(precision_bits STREQUAL "64" AND NOT PROBE_PROFILE)
    target_compile_definitions(${target} PRIVATE FLASH_NEWTON_SPLIT_PROBE=1)
  endif()
  if(PROBE_PROFILE OR NOT suffix STREQUAL "")
    target_compile_definitions(${target} PRIVATE FLASH_NEWTON_ONLY_PROBE=1)
  endif()
  target_link_libraries(${target} PRIVATE flash_cuda_options${precision_suffix})
  add_dependencies(${target} ${mathdx_target})
  if(suffix STREQUAL "" AND NOT PROBE_PROFILE)
    add_dependencies(${target} ${osc_target})
  endif()
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
    JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
  if(FLASH_GLOBAL_DEVICE_LTO)
    target_link_options(${target} PRIVATE "$<DEVICE_LINK:-dlto>")
    target_link_libraries(${target} PRIVATE "${FLASH_SOLVERDX}")
  endif()
endfunction()

function(flash_add_substep_tests precision_bits osc_input osc_target)
    flash_precision_suffix(${precision_bits} precision_suffix)
  # These historical fixtures own double buffers and retain FP64-only bounds.
  # Precision-generic tests below provide the separate FP32 numerical coverage.
  if(precision_bits STREQUAL "64" AND EXISTS "${FLASH_GPU}/../../../tests/cuda/osc_reference_test.cu")
    add_executable(flash_osc_reference_test${precision_suffix} EXCLUDE_FROM_ALL
      "${FLASH_GPU}/../../../tests/cuda/osc_reference_test.cu"
      $<TARGET_OBJECTS:flash_osc_solver${precision_suffix}> ${osc_input})
    target_link_libraries(flash_osc_reference_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
    add_dependencies(flash_osc_reference_test${precision_suffix} ${osc_target})
    set_target_properties(flash_osc_reference_test${precision_suffix} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" JOB_POOL_LINK flash_cuda_link)
    if(FLASH_GLOBAL_DEVICE_LTO)
      target_link_options(flash_osc_reference_test${precision_suffix} PRIVATE "$<DEVICE_LINK:-dlto>")
      target_link_libraries(flash_osc_reference_test${precision_suffix} PRIVATE "${FLASH_SOLVERDX}")
    endif()
  endif()

  if(precision_bits STREQUAL "64" AND EXISTS "${FLASH_GPU}/../../../tests/cuda/controller_pipeline_test.cu")
    add_executable(flash_controller_pipeline_test${precision_suffix} EXCLUDE_FROM_ALL
      "${FLASH_GPU}/../../../tests/cuda/controller_pipeline_test.cu"
      $<TARGET_OBJECTS:flash_controller_pipeline${precision_suffix}> $<TARGET_OBJECTS:flash_osc_solver${precision_suffix}> ${osc_input})
    target_link_libraries(flash_controller_pipeline_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
    add_dependencies(flash_controller_pipeline_test${precision_suffix} ${osc_target})
    set_target_properties(flash_controller_pipeline_test${precision_suffix} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" JOB_POOL_LINK flash_cuda_link)
    if(FLASH_GLOBAL_DEVICE_LTO)
      target_link_options(flash_controller_pipeline_test${precision_suffix} PRIVATE "$<DEVICE_LINK:-dlto>")
      target_link_libraries(flash_controller_pipeline_test${precision_suffix} PRIVATE "${FLASH_SOLVERDX}")
    endif()
  endif()

  if(precision_bits STREQUAL "64" AND EXISTS "${FLASH_GPU}/../../../tests/cuda/post_demand_test.cu")
    add_executable(flash_post_demand_test${precision_suffix} EXCLUDE_FROM_ALL
      "${FLASH_GPU}/../../../tests/cuda/post_demand_test.cu" $<TARGET_OBJECTS:flash_post_solver${precision_suffix}>)
    target_link_libraries(flash_post_demand_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
    set_target_properties(flash_post_demand_test${precision_suffix} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real")
  endif()

  # Optional diagnostics: actual production Post and Euler mathematical bodies.
  # Normal products do not depend on these targets. No runtime code generation.
  if(precision_bits STREQUAL "64" AND EXISTS "${FLASH_GPU}/../../../tests/cuda/post_replay_probe.cu")
    add_library(flash_post_replay_probe SHARED EXCLUDE_FROM_ALL
      "${FLASH_GPU}/../../../tests/cuda/post_replay_probe.cu"
      $<TARGET_OBJECTS:flash_post_solver>)
    target_include_directories(flash_post_replay_probe PRIVATE "${FLASH_STEP}"
      "${FLASH_GPU}/cuda/postprocess")
    target_link_libraries(flash_post_replay_probe PRIVATE flash_cuda_options)
    set_target_properties(flash_post_replay_probe PROPERTIES
      CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real"
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
      JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
    if(FLASH_GLOBAL_DEVICE_LTO)
      target_link_options(flash_post_replay_probe PRIVATE "$<DEVICE_LINK:-dlto>")
    endif()
  endif()
  if(precision_bits STREQUAL "64" AND EXISTS "${FLASH_GPU}/../../../tests/cuda/euler_replay_probe.cu")
    add_library(flash_euler_replay_probe SHARED EXCLUDE_FROM_ALL
      "${FLASH_GPU}/../../../tests/cuda/euler_replay_probe.cu")
    target_include_directories(flash_euler_replay_probe PRIVATE "${FLASH_STEP}")
    # Runtime widths select actual model inputs; this is only the arena ceiling.
    target_compile_definitions(flash_euler_replay_probe PRIVATE FLASH_SUBSTEP_NV=57
      FLASH_SUBSTEP_NQ=65 FLASH_SUBSTEP_NM=212 FLASH_SUBSTEP_NJ=18
      FLASH_SUBSTEP_NU=9 FLASH_SUBSTEP_CONTACTS=5000)
    target_link_libraries(flash_euler_replay_probe PRIVATE flash_cuda_options)
    set_target_properties(flash_euler_replay_probe PROPERTIES
      CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real"
      JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
  endif()

  if(precision_bits STREQUAL "64" AND EXISTS "${FLASH_GPU}/../../../tests/cuda/body_replay_probe.cu")
    add_library(flash_body_replay_probe SHARED EXCLUDE_FROM_ALL
      "${FLASH_GPU}/../../../tests/cuda/body_replay_probe.cu")
    target_include_directories(flash_body_replay_probe PRIVATE "${FLASH_STEP}")
    target_link_libraries(flash_body_replay_probe PRIVATE flash_cuda_options)
    set_target_properties(flash_body_replay_probe PROPERTIES
      CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real"
      JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
  endif()

  if(precision_bits STREQUAL "64" AND EXISTS "${FLASH_GPU}/../../../tests/cuda/transmission_replay_probe.cu")
    add_library(flash_transmission_replay_probe SHARED EXCLUDE_FROM_ALL
      "${FLASH_GPU}/../../../tests/cuda/transmission_replay_probe.cu")
    target_include_directories(flash_transmission_replay_probe PRIVATE "${FLASH_STEP}")
    target_link_libraries(flash_transmission_replay_probe PRIVATE flash_cuda_options)
    set_target_properties(flash_transmission_replay_probe PROPERTIES
      CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real"
      JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
  endif()

  if(precision_bits STREQUAL "64" AND EXISTS "${FLASH_GPU}/../../../tests/cuda/position_mass_replay_probe.cu")
    add_library(flash_position_mass_replay_probe SHARED EXCLUDE_FROM_ALL
      "${FLASH_GPU}/../../../tests/cuda/position_mass_replay_probe.cu")
    target_include_directories(flash_position_mass_replay_probe PRIVATE "${FLASH_STEP}")
    target_link_libraries(flash_position_mass_replay_probe PRIVATE flash_cuda_options)
    set_target_properties(flash_position_mass_replay_probe PROPERTIES
      CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real"
      JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
  endif()

  if(IS_DIRECTORY "${FLASH_GPU}/../../../tests/cuda")
  # GEMV owns precision-generic inputs and independent rounding-depth bounds.
  if(EXISTS "${FLASH_GPU}/../../../tests/cuda/newton_gemv_test.cu")
    add_executable(flash_newton_gemv_test${precision_suffix} EXCLUDE_FROM_ALL "${FLASH_GPU}/../../../tests/cuda/newton_gemv_test.cu")
    target_include_directories(flash_newton_gemv_test${precision_suffix} PRIVATE "${FLASH_STEP}")
    target_link_libraries(flash_newton_gemv_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
    set_target_properties(flash_newton_gemv_test${precision_suffix} PROPERTIES CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real")
  endif()
  # Historical numerical tests retain their FP64 fixtures and bounds.
  if(precision_bits STREQUAL "64")
  if(EXISTS "${FLASH_GPU}/../../../tests/cuda/forward_mass_test.cu")
    add_executable(flash_forward_mass_test${precision_suffix} EXCLUDE_FROM_ALL "${FLASH_GPU}/../../../tests/cuda/forward_mass_test.cu")
    target_include_directories(flash_forward_mass_test${precision_suffix} PRIVATE "${FLASH_GPU}/cuda")
    target_link_libraries(flash_forward_mass_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
    set_target_properties(flash_forward_mass_test${precision_suffix} PROPERTIES CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real")
  endif()
  add_executable(flash_newton_atomic_test${precision_suffix} EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/newton_atomic_test.cu"
    $<TARGET_OBJECTS:flash_newton_atomic${precision_suffix}>)
  target_include_directories(flash_newton_atomic_test${precision_suffix} PRIVATE "${FLASH_STEP}")
  target_link_libraries(flash_newton_atomic_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  set_target_properties(flash_newton_atomic_test${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON JOB_POOL_LINK flash_cuda_link)
  add_executable(flash_newton_jacobian_cache_test${precision_suffix} EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/newton_jacobian_cache_test.cu"
    $<TARGET_OBJECTS:flash_newton_atomic${precision_suffix}> $<TARGET_OBJECTS:flash_newton_transpose${precision_suffix}>
    $<TARGET_OBJECTS:flash_newton_jacobian_cache${precision_suffix}>)
  target_include_directories(flash_newton_jacobian_cache_test${precision_suffix} PRIVATE "${FLASH_STEP}")
  target_link_libraries(flash_newton_jacobian_cache_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  set_target_properties(flash_newton_jacobian_cache_test${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON JOB_POOL_LINK flash_cuda_link)
  add_executable(flash_newton_transpose_test EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/newton_transpose_test.cu"
    $<TARGET_OBJECTS:flash_newton_transpose>)
  target_include_directories(flash_newton_transpose_test PRIVATE "${FLASH_STEP}")
  target_link_libraries(flash_newton_transpose_test PRIVATE flash_cuda_options)
  set_target_properties(flash_newton_transpose_test PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON JOB_POOL_COMPILE flash_cuda_compile JOB_POOL_LINK flash_cuda_link)
  add_executable(flash_collision_tree_test${precision_suffix} EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/collision_tree_parallel_test.cu")
  target_include_directories(flash_collision_tree_test${precision_suffix} PRIVATE "${FLASH_GPU}/cuda")
  target_link_libraries(flash_collision_tree_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  set_target_properties(flash_collision_tree_test${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real")
  add_executable(flash_collision_sap_sort_test${precision_suffix} EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/collision_sap_sort_test.cu")
  target_link_libraries(flash_collision_sap_sort_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  set_target_properties(flash_collision_sap_sort_test${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real")
  endif()

  add_executable(flash_precision_eig3_test${precision_suffix} EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/precision_eig3_test.cu")
  target_include_directories(flash_precision_eig3_test${precision_suffix} PRIVATE "${FLASH_GPU}/cuda")
  target_link_libraries(flash_precision_eig3_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  set_target_properties(flash_precision_eig3_test${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real")

  add_executable(flash_newton_rank_update_test${precision_suffix} EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/newton_rank_update_test.cu")
  target_include_directories(flash_newton_rank_update_test${precision_suffix} PRIVATE "${FLASH_STEP}")
  target_link_libraries(flash_newton_rank_update_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  set_target_properties(flash_newton_rank_update_test${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real")

  add_executable(flash_precision_boxbox_test${precision_suffix} EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/precision_boxbox_test.cu")
  target_link_libraries(flash_precision_boxbox_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  target_compile_definitions(flash_precision_boxbox_test${precision_suffix} PRIVATE
    FLASH_BOX_CORNER_FIXTURE="${FLASH_GPU}/../../../tests/fixtures/boxbox_first_face_vertex.csv"
    FLASH_BOX_SAT_FIXTURE="${FLASH_GPU}/../../../tests/fixtures/boxbox_sat_face_tie.csv")
  set_target_properties(flash_precision_boxbox_test${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" JOB_POOL_COMPILE flash_cuda_compile
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  add_library(flash_boxbox_replay_probe${precision_suffix} SHARED EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/boxbox_replay_probe.cu")
  target_link_libraries(flash_boxbox_replay_probe${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  set_target_properties(flash_boxbox_replay_probe${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" JOB_POOL_COMPILE flash_cuda_compile
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON JOB_POOL_LINK flash_cuda_link)

  add_executable(flash_precision_gram_test${precision_suffix} EXCLUDE_FROM_ALL
    "${FLASH_GPU}/../../../tests/cuda/precision_gram_test.cu"
    $<TARGET_OBJECTS:flash_newton_atomic${precision_suffix}>
    $<TARGET_OBJECTS:flash_newton_transpose${precision_suffix}>
    $<TARGET_OBJECTS:flash_newton_jacobian_cache${precision_suffix}>)
  target_include_directories(flash_precision_gram_test${precision_suffix} PRIVATE "${FLASH_STEP}")
  target_link_libraries(flash_precision_gram_test${precision_suffix} PRIVATE flash_cuda_options${precision_suffix})
  set_target_properties(flash_precision_gram_test${precision_suffix} PROPERTIES
    CUDA_ARCHITECTURES "${FLASH_CUDA_ARCH}-real" CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON JOB_POOL_LINK flash_cuda_link)
  endif() # Repository-only CUDA test sources.

endfunction()
