# Third-Party Notices

This file records upstream source origins, adaptation scope, and external dependencies. Project-original contributions use the MIT license in the root [LICENSE](../../../LICENSE). Upstream and third-party portions retain their own terms and copyright notices; the project MIT license does not replace them.

## Component Licensing

| Component / rights holder | Use and scope | Retained terms | Source or license |
| --- | --- | --- | --- |
| LIBERO / Lifelong Robot Learning | Base environment, tasks, learning code, and inherited controller code | MIT; original Copyright 2023 Lifelong Robot Learning | [Original LIBERO license](../../../licenses/LIBERO-MIT.txt) |
| MuJoCo 2.3.7 / DeepMind Technologies Limited | Collision and mathematical helpers adapted for CUDA and this project's layouts | Apache-2.0 | [License](../../../licenses/Apache-2.0.txt), [convex collision](host/runtime/collision/native_originals/engine_collision_convex.c.txt), [spatial helpers](host/runtime/collision/native_originals/engine_util_spatial.c.txt) |
| MuJoCo box collision / Svetoslav Kolev | Native box collision and selected independent SAT translations | Apache-2.0; original Copyright 2016 Svetoslav Kolev | [Original source](host/runtime/collision/native_originals/engine_collision_box.c.txt), [SAT translation](reference/collision/box_cross/kernel.py) |
| libccd / Daniel Fiser and the attributed Czech Technical University research group | Vector, support, simplex, and MPR convex collision code | BSD-3-Clause | [License](../../../licenses/libccd-BSD-3-Clause.txt), [original license copy](host/runtime/collision/native_originals/BSD-LICENSE.txt), [MPR source](host/runtime/collision/native_originals/ccd_mpr.c.txt) |
| NVIDIA cuBLASDx / cuSOLVERDx | External device-side matrix operations, POTRF, and solves through bridges | Terms supplied with the installed MathDx SDK | [POTRF bridge](cuda/newton/potrf_dx_bridge.cu), [Hessian bridge](cuda/newton/hessian_blas_dx_bridge.cu), [OSC bridge](cuda/control/osc_bridge.cu) |
| NVIDIA Warp | External framework used by reference operators and host construction | Terms supplied with the installed warp-lang package | `reference/`, `host/assembly/`, and runtime templates |
| MuJoCo Warp / The Newton Developers | Optional FP32 physics backend and isolated CUDA Hessian adaptation; installed upstream source remains unmodified | Apache-2.0, Copyright 2025 The Newton Developers | [Pinned source](https://github.com/google-deepmind/mujoco_warp/tree/c384ac51df67eb92ab7657cc472ff7494dc036bf), [backend integration](host/backends/mjwarp/README.md), [Hessian adaptation](cuda/newton/mjwarp_hessian.cu) |
| robosuite / MuJoCo CPU implementation | Environment dependencies and historical CPU comparison references | Their respective installed-package licenses; inherited code is identified separately | OSC adaptation below and upstream source notices |

## Adapted Device Code

`cuda/` includes both project implementations and adapted upstream source. Handwritten CUDA describes the implementation language, not exclusive authorship of every algorithm or source fragment.

- [native_source.cuh](cuda/collision/native/native_source.cuh) contains CUDA device adaptations of retained MuJoCo/libccd C source. The historical generator named in its header is archived outside this distribution. The maintained source is compiled directly. Its geometric source derivation is retained in the repository test helper `tests/support/box_contact_source.py`. Tests check the corrected edge tail and first-vertex clipping; numerical fixtures cover the SAT precision branch. This helper is not part of production code generation.
- [collision_detect_primitive.cuh](cuda/collision/collision_detect_primitive.cuh) directly adapts MuJoCo scalar collision mathematics.
- CUDA fusion, layouts, thread organization, host bindings, caching, and restoration add project implementation work while retaining upstream attribution for inherited portions.
- MathDx bridges call external libraries; they do not include or relicense the SDK's internal BLAS/LAPACK implementation.
- [mjwarp_hessian.cu](cuda/newton/mjwarp_hessian.cu) adapts dense Hessian assembly and elliptic-cone curvature from the pinned MuJoCo Warp `solver.py` (Apache-2.0, Copyright 2025 The Newton Developers), and register sub-tile selection from Warp 1.18.0 `native/tile_matmul.h` (Apache-2.0, Copyright 2026 NVIDIA CORPORATION & AFFILIATES). It uses explicit CUDA threads and plain device pointers. The independently compiled FP32 leaf is locally qualified before any production solver-dispatch replacement; the external upstream package remains unchanged.
- [newton.py](host/backends/mjwarp/newton.py) adapts the pinned MuJoCo Warp `_solve`, `_solver_iteration` and dense `_update_gradient` host orchestration (Apache-2.0, Copyright 2025 The Newton Developers). The opt-in path explicitly substitutes the CUDA Hessian leaf and retains upstream numerical kernels and stopping/warmstart semantics. The external package is unchanged.
- [mjwarp_direction.cu](cuda/newton/mjwarp_direction.cu) adapts gradient kernels and small/blocked Cholesky direction dispatch from pinned MuJoCo Warp `solver.py` and the augmented-RHS algorithm from `block_cholesky.py` (Apache-2.0, Copyright 2025 The Newton Developers). Register subtiles and aligned tile transfers follow Warp 1.18 `native/tile_matmul.h` and tile storage (Apache-2.0, Copyright 2026 NVIDIA CORPORATION & AFFILIATES). cuSolverDx remains an external SDK dependency under its supplied license. The CUDA file explicitly implements the control/data flow and calls device POTRF/POTRS/TRSM; it does not load upstream numerical kernels.

Original source and notices are retained in [native_originals/](host/runtime/collision/native_originals/). Combined native files may repeat notices because they incorporate multiple upstream files. These source-level notices remain applicable.

## FP32 Newton Constraint Adaptation

The optional [mjwarp_constraint.cu](cuda/newton/mjwarp_constraint.cu) adapts the
fixed MuJoCo Warp `solver.py` constraint projection and dense force-transpose
kernels (Apache-2.0, Copyright 2025 The Newton Developers). It retains their
per-row/per-DOF work assignment and full-rebuild semantics. CPU MuJoCo 2.3.7
`mj_constraintUpdate` is the independent mathematical reference for the local
FP64 validation oracle; no upstream package is modified.

## OSC Adaptation

The fixed-size LU inverse, one-sided Jacobi SVD, FP32 quaternion conversion, OSC equations, and selected reset logic in `reference/control/osc/controller.py` were adapted from a historical LIBERO GPU controller at SHA256 `98711c69d24a4d6a1c5d96d1a0018c6f8cb897cbc5b3370db245c2ea654a099c`.

That historical source was MIT licensed under the LIBERO repository license, Copyright 2023 Lifelong Robot Learning. The original notice is preserved in [LIBERO-MIT.txt](../../../licenses/LIBERO-MIT.txt). No historical repository is imported at runtime.

The adapted implementation adds physical-input and controller-cache buffers, device-side cache selection, per-world policy/configuration, intermediate outputs, and capacity checks. The historical CPU comparison reference was robosuite 1.4.0; reference results do not supply production GPU state.

## Included License Texts

- [Project MIT license and component scope](../../../LICENSE): project-original Flash-LIBERO contributions, Copyright 2026 Flash-LIBERO contributors.
- [Original LIBERO MIT license](../../../licenses/LIBERO-MIT.txt): unchanged upstream text and attribution.
- [Apache License 2.0](../../../licenses/Apache-2.0.txt): retained from the installed MuJoCo 2.3.7 distribution.
- [libccd BSD-3-Clause license](../../../licenses/libccd-BSD-3-Clause.txt): unchanged original text, conditions, and disclaimer.

The original Apache source notices identify DeepMind Technologies Limited and, for box collision, Svetoslav Kolev. Source-level copyright and modification notices are retained. SDK and framework licenses are supplied by their external distributions rather than this repository's MIT license.

## Maintenance

When adding or adapting upstream source, record its origin, affected code, and applicable terms. Preserve required copyright, license, disclaimer, and modification notices in source and generated output. Keep project-original additions distinguishable from upstream portions. This inventory documents retained terms and does not replace file-level notices or external licenses.

The optional `cuda/newton/mjwarp_linesearch.cu` and its resident dispatch are
handwritten adaptations of pinned MJWarp `support.py::mul_m_kernel` and
`solver.py::_linesearch_jv_fused_kernel`, `_linesearch_iterative_kernel` and
its elliptic cost/derivative helpers at commit
`c384ac51df67eb92ab7657cc472ff7494dc036bf` (Copyright 2025 The Newton
Developers, Apache-2.0). The reduction ordering follows Warp 1.18
`native/tile_reduce.h`. No generated Warp CUDA or Warp native header is linked.

`cuda/newton/mjwarp_solver_state.cu` and `mjwarp_products.cuh` adapt the pinned
MJWarp `solver.py::_solve_init_dof`, `_solve_init_efc`,
`_solve_init_jaref_kernel`, `_solve_done` and `support.py::mul_m_kernel`
(Copyright 2025 The Newton Developers, Apache-2.0). They use readable CUDA and
borrow existing device arrays; no generated Warp source is included.

`cuda/dynamics/mjwarp_mass_solve.cu` adapts the pinned MJWarp
`smooth.py::_small_cholesky_solve_block`, `_small_cholesky_solve` and
`_tile_cholesky_solve_block` (Copyright 2025 The Newton Developers,
Apache-2.0). Its cooperative solve calls external cuSolverDx POTRS, following
Warp 1.18's tile-Cholesky operator dispatch (Copyright 2026 NVIDIA CORPORATION
& AFFILIATES, Apache-2.0); it includes no generated Warp source or native header.
`host/backends/mjwarp/acceleration.py` adapts only the non-discrete, non-sleeping
`forward.py::fwd_acceleration` host schedule, retaining upstream force assembly
and projection and explicitly replacing backsubstitution. The installed upstream
package remains unchanged. cuSolverDx retains the SDK's supplied license.

`cuda/dynamics/mjwarp_smooth_force.cu` adapts pinned MJWarp
`forward.py::_qfrc_smooth(False)` and `support.py::_apply_ft(flg_add=True)`
(Copyright 2025 The Newton Developers, Apache-2.0). The combined launch retains
per-DOF ownership and subtree traversal; the split launch remains a local
comparison path. The external upstream package is unmodified. CPU MuJoCo 2.3.7
`mj_applyFT` supplies independent frozen-input Jacobians for validation, not
production state or code generation.

`cuda/dynamics/mjwarp_mass_factor.cu` adapts pinned MJWarp `smooth.py` scalar/tile
Cholesky factorization and fused factorize-solve blocks (Copyright 2025 The
Newton Developers, Apache-2.0). Opposite-triangle clearing follows Warp 1.18
`native/tile_cholesky.h` (Copyright 2026 NVIDIA CORPORATION & AFFILIATES,
Apache-2.0). It calls external cuSolverDx POTRF/POTRS and includes no generated
Warp source or native header. `host/backends/mjwarp/mass_factor.py` and
`host/backends/mjwarp/position.py` adapt only the ordinary non-sleeping position
and Euler host schedules; unrelated physics
and `_advance` remain installed upstream implementations. The installed package
is unmodified, and cuSolverDx retains its SDK license.

`cuda/dynamics/mjwarp_mass_matrix.cu` adapts pinned MJWarp
`smooth.py::_crb_accumulate`, `_M`, and `math.py::inert_vec` (Copyright 2025
The Newton Developers, Apache-2.0). It preserves layered atomic accumulation
and the sparse ancestor traversal. The installed package is unmodified; no
generated Warp source is included in the production CUDA translation unit.
CPU MuJoCo 2.3.7 `mj_crb` provides a separate frozen-input validation oracle.

`cuda/dynamics/mjwarp_velocity.cu`, `mjwarp_bias.cu`, `mjwarp_spatial.cuh` and
`host/backends/mjwarp/velocity.py` adapt pinned MJWarp `smooth.py::com_vel`,
`rne(flg_acc=False)`, their numerical kernels, `math.py` spatial operators and
the ordinary `forward.py::fwd_velocity` schedule (Copyright 2025 The Newton
Developers, Apache-2.0). The handwritten branches use private prefixes and
canonical output ownership. The installed upstream package is unmodified.
CPU MuJoCo 2.3.7 `mj_comVel` and `mj_rne` are independent validation oracles.

`cuda/dynamics/mjwarp_postconstraint.cu` adapts pinned MJWarp
`smooth.py::rne_postconstraint`'s acceleration/force suffix (Copyright 2025
The Newton Developers, Apache-2.0). The literal external-wrench host prefix
is retained in `host/backends/mjwarp/postconstraint_external.py`, with kernel
names qualified to the unmodified installed upstream package. No generated
Warp source is used. The installed CPU MuJoCo 2.3.7 `mj_rnePostConstraint`
provides an independent frozen-input validation oracle.

`cuda/dynamics/mjwarp_passive.cu` adapts pinned MJWarp
`passive.py::_spring_damper_dof_passive` and `_qfrc_passive_kernel`,
`util_misc.py::_poly_force`, `math.py::quat_sub`/`quat_to_vel`/`mul_quat`
(Copyright 2025 The Newton Developers, Apache-2.0). Quaternion normalization
retains Warp1.18 `native/quat.h` semantics (NVIDIA CORPORATION & AFFILIATES,
Apache-2.0). The rigid gravity/fluid/adhesion host schedule is retained in
`host/backends/mjwarp/passive_other.py`; the installed packages are unmodified.
CPU MuJoCo2.3.7 `mj_passive` and `mju_subQuat` independently validate the linear
joint-force and quaternion mathematical references.

`cuda/dynamics/mjwarp_actuation.cu` adapts pinned MJWarp `forward.py`'s stateless
fixed/affine scalar-joint `_actuator_force`, `_qfrc_actuator` and
`_qfrc_actuator_gravcomp_limits` (Copyright 2025 The Newton Developers,
Apache-2.0). The host schedule in `host/backends/mjwarp/actuation.py` preserves
upstream history reading, disable clearing and callback ordering. Floating
clamping follows Warp1.18 native fmin/fmax semantics; no generated Warp CUDA is
included and the installed packages are unmodified. Installed CPU MuJoCo2.3.7
`mj_transmission`, `mj_fwdActuation` and `mju_mulMatTVec` supply independent
frozen-input validation oracles.
