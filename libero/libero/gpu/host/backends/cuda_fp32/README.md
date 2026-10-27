# Independent CUDA FP32

This backend owns FP32 CUDA buffers, streams, graphs, model preprocessing and
LIBERO batch lifecycle. Neither `warp` nor `mujoco_warp` is required to prepare
its modern CPU models or execute physics. Numerical leaves are shared with the
handwritten MJWarp migration path; readable world-local fusion explicitly
combines accepted phases in `cuda/fp32/`.
It is the default `GpuBatchEnv` backend when precision is omitted and
`LIBERO_GPU_PRECISION` is unset. Select `precision="fp64"` explicitly to use
the native FP64 backend in its separate environment.
The development environment pins CPU MuJoCo 3.16.0 for model compilation and OpenGL rendering.
Production intake uses the actual model API; each prepared MJB is bound to its Python/native engine version.
CPU MuJoCo 2.3.7 remains the independent reference for overlapping semantics.

The algorithm reference is MJWarp commit
`c384ac51df67eb92ab7657cc472ff7494dc036bf`. The deliberate collision corrections
are documented separately in [NUMERICAL_DEVIATIONS.md](../mjwarp/NUMERICAL_DEVIATIONS.md).
Removing Warp does not revert those corrections or introduce new solver math.
See [qualification results and retained exceptions](QUALIFICATION.md) for the
130-task run, long trajectories, package checks and shared-device timings.

## Installation and explicit preparation

Use an isolated Python environment:

```sh
python -m pip install '.[cuda-fp32]'
python -m libero.gpu.build --precision fp32 \
  --build-root /path/to/common-project \
  --mathdx-root /path/to/mathdx --jobs 2
```

CPU scene preparation is separate from this CMake build. To create a new
common catalogue, use the CPU-only command in the separate CPU MuJoCo 2.3.7
environment (with the `cpu-catalog` extra installed):

```sh
python -m libero.gpu.host.backends.cuda_fp32.catalog \
  --build-root /path/to/new-common-project --tasks all --seed 0
```

Then import its fixed scenes in the independent FP32 environment:

```sh
python -m libero.gpu.host.backends.cuda_fp32.prepare \
  --build-root /path/to/common-project --tasks all --skip-build
```

The common project contains the CPU scene catalogue
`TASK_INSTANCES.json`, scene XML, fixed fixture parameters and initial states.
Those are CPU model inputs, not serialized Warp arrays. The existing catalogue
preparation uses CPU MuJoCo 2.3.7; run it in its separate CPU environment.
The scene command imports these inputs using CPU MuJoCo 3.16.0. CUDA compilation
uses the fixed `flash_cuda_fp32_all` CMake target; it does not load scenes or
construct GPU environments. The default compiler profile is `dev`;
`--build-mode release` selects the other profile, and `--build-mode all`
explicitly requests both. Existing compatible CMake products are reused.
The older preparation command retains task selectors and `--release`, with
`--skip-build` selecting only CPU scene preparation. After upgrading from the legacy scene cache,
run this once for the task profiles you use; it validates and imports compatible
old MJBs into the direct-load format without compiling CUDA. Ordinary environment
construction never scans old cache directories or hashes asset/binary contents.
Offline preparation validates those contents; runtime retains format, model
layout and task association checks. Preparation and execution must be serialized
on shared GPU machines. Runtime construction defaults to `allow_compile=False`.
Camera observation choices, image dimensions and episode lifecycle options
reuse the same fixed CPU model: they configure each fresh host shell, rather
than selecting another MJB. Physics XML, fixed scene inputs and non-host model
options remain part of model selection. Preparing all tasks therefore covers
the ordinary no-camera and default camera observation configurations without
creating separate compiled scene copies for each observation size.

```python
from libero.libero.envs import GpuBatchEnv

env = GpuBatchEnv(bddl_path, 2, precision="fp32", physics_engine="cuda_fp32",
                  build_root=project_root, allow_compile=False,
                  use_camera_obs=False, ignore_done=True)
```

Use `cuda_options={"fast_compile": True, "nconmax": ..., "njmax": ...}` for
explicit allocation/build choices. The older `mjwarp_options` spelling is
accepted only for compatible CUDA-only settings. There is no upstream numerical
fallback. Unsupported domains fail before simulation.

## Structure and call chain

```text
GpuBatchEnv -> CudaFP32BatchEnv                         environment.py
  scene inputs -> modern CPU model -> device data      scene.py, model.py, data.py
  Runtime.enqueue_action                              runtime.py
    CUDA Graph: 25 ordinary physical substeps
      Physics.step1                                   physics.py
        kinematics / derived frames / centers / mass
        pooled collision -> constraint construction
        transmission + entry clears
        position-only gravity and motion / bias / passive forces
      FusedControlDrive                               fused_control_drive.py
        policy / full Jacobian / gather -> maintained OSC
        actuator publication -> actuation / smooth force / mass solve
      Physics.solve(unconstrained_ready=...)           physics.py
        Newton initialization
        constraint / gradient -> Hessian -> direction
        device WHILE: line search -> gradient / Hessian / direction + stopping
        pooled external wrench -> postconstraint / force sensors
      FusedEulerStep                                  fused_euler_step.py
        damping / private factor + solve -> qvel / qpos
        Newton warmstart / time / overflow / substep accounting
      current low-dimensional observation sampling     observations.py
  public boundary: predicates, dictionaries, GL        shared adapters
```

This is one Newton solve per physical substep. Public `forward()` retains its
factor/solve path; it does not reuse a stale public factor. Unsupported fusion
layouts use accepted CUDA components. Disabled actuation, undamped integration,
warning paths and custom observation callbacks keep their existing schedules.
The archived 25-substep mega-kernel is not part of this execution path.

`device.py` owns contiguous allocations and views, rather than compiling
numerical Python. Views retain allocation owners, uploads retain private host
storage until completion, and Graphs retain launch owners. Allocation is
forbidden during capture. Newton initializes its reusable workspace per solve;
the conditional loop uses a current device counter without host iteration
polling. `Controller.bootstrap` and standalone gather remain explicit entries.

The runtime selects `sm<actual-device-architecture>-<mode>` after a driver-only
device query. Pass `--arch` to the offline build for each target deployment;
MathDx operators use that target, without an SM80 fallback. Independent FP32
allocations, streams, graph capture, execution and release restore the calling
thread's previous CUDA device.

`scene_seed=None` uses the fixed prepared fixture scene. `seed` controls each
world's episode sequence and can change without preparing a new scene or CUDA
product. N is also a runtime argument, with memory and int32 indexing limits.
Observation layout choices do not select another physics binary. OSC action
input/output ranges, Panda formatter speed and the existing `uncouple_pos_ori`
setting are passed to the solver; the 7-DOF Panda/9-control layout and unported
physics families remain actual implementation limits.

Prepared data uses root-relative internal paths and configured asset roots.
A runtime deployment can contain scene data, model caches and the selected
`.so` bank without CMake files, a compiler or the MathDx development SDK.
Independent FP32 does not select/freeze the legacy Warp scalar policy. The
legacy native and MJWarp dependency environments retain their own real type
and engine compatibility requirements.

### Numerical source map and selection

All paths below read current device physical inputs. Only immutable topology
and layout indices are retained between calls. Links point to actual CUDA
bodies; fusion is continuous source, with small mathematical helpers and
maintained SDK/OSC operations where appropriate.

| Region | CUDA source / function | Organization and retained path |
| --- | --- | --- |
| Position prefix | [fused_kinematic_mass.cu](../../../cuda/fp32/fused_kinematic_mass.cu), `kinematic_mass` | One 512-thread world CTA when at most 16 canonical branches each have a resident warp, bounded preorder intervals and joint scratch fit. The fixed 2560-byte arena reuses joint/weighted-center storage as CRB storage. Wider trees use the two accepted regions below. |
| Branch frames | [mjwarp_kinematics.cu](../../../cuda/dynamics/mjwarp_kinematics.cu), `branch_kinematics<4,true>` | Four independent branch warps per CTA publish body/inertial, geom/site and admitted FIXED camera/light frames. Heavy frame queues retain separate tiles. |
| Centers and mass | [fused_position.cu](../../../cuda/fp32/fused_position.cu), `centered_mass` | Bounded direct subtree sums on small preorder trees; staged/global-storage fallback for other trees. Fixed 2560-byte shared storage. |
| Transmission | [mjwarp_transmission.cu](../../../cuda/dynamics/mjwarp_transmission.cu), `world_transmission` | Counter reset, scalar rows and optional actuator velocity share a world warp or 128-thread CTA. Step1 also clears sensors/energy. Ordinary calls preserve those buffers; generic CSR velocity remains callable. |
| Motion and passive forces | [fused_motion_passive.cu](../../../cuda/fp32/fused_motion_passive.cu), `motion_passive` | Literal joint spring/damper, spatial velocity/bias, fluid and passive composition in 512 threads. Three explicit block barriers and a reused 3072-byte arena. Packed fluid scratch through 42 bodies, global fluid storage for larger admitted trees; accepted components outside the tree domain. |
| No-fluid / disabled passive | [fused_motion.cu](../../../cuda/fp32/fused_motion.cu), `motion_bias` | Accepted motion/bias region plus the caller's corresponding passive/clear paths. Position-only gravity projection retains its independent entry. |
| Control through unconstrained acceleration | [fused_control_drive.cu](../../../cuda/fp32/fused_control_drive.cu), `control_drive` | The 9-control Panda domain joins actual preparation/publication and force/solve code in 64 threads. Warp-local control publication and the existing force barrier order reuse of the finished OSC arena; shared allocation does not grow. |
| Control components | [fused_controller.cu](../../../cuda/fp32/fused_controller.cu), `controller_region`; [fused_control_prepare.cu](../../../cuda/fp32/fused_control_prepare.cu) | Preparation, maintained OSC and publication remain independently callable. Six 8-lane groups compute full-DOF EE velocity. NV above 64 uses the accepted preparation/publication components. |
| Actuation and mass-solve component | [fused_actuation_solve.cu](../../../cuda/fp32/fused_actuation_solve.cu), `actuation_solve` | Force evaluation, CSR atomics, limits, smooth RHS and scalar/SDK backsubstitution. DOF producers stage the cooperative RHS directly. Four explicit barriers, at most 376 shared bytes; unsupported groups use accepted components. |
| Independent mass groups | [mjwarp_mass_factor.cu](../../../cuda/dynamics/mjwarp_mass_factor.cu), `world_factor`; [mjwarp_mass_solve.cu](../../../cuda/dynamics/mjwarp_mass_solve.cu), `world_solve` | Literal scalar 1/6 blocks and one cooperative 9-DOF SDK block per world. Cold selection bounds scalar block count to 32; other layouts retain component launches. Exact in-place solves and padding contracts are retained. |
| Newton initialization | [mjwarp_solver_state.cu](../../../cuda/newton/mjwarp_solver_state.cu), `initialize_world` | Warm/cold selection, solve clears, Ma and Jaref in a 1024-thread world CTA. Bounded 512-byte shared acceleration through NV128, global input above it. |
| Constraint and gradient | [mjwarp_constraint.cu](../../../cuda/newton/mjwarp_constraint.cu), `constraint_gradient` | Row force/state, transpose force and gradient in 512 threads with one producer barrier. Four-KiB row staging through 1024 rows; larger sets read global publication. |
| Newton matrix and direction | [mjwarp_hessian.cu](../../../cuda/newton/mjwarp_hessian.cu); [mjwarp_direction.cu](../../../cuda/newton/mjwarp_direction.cu) | World quadratic assembly, pooled contact-cone contributions and SDK direction retain completion boundaries. Iterative direction publication includes stopping without another barrier; initial direction omits stopping and warning mode retains its original path. |
| Line search | [mjwarp_linesearch.cu](../../../cuda/newton/mjwarp_linesearch.cu), `iterative_search`, `world_products` | Small NV prepends Mv to the one-warp search; larger NV joins world products before the maintained search. Search iterations and root decisions are not unrolled. |
| Postconstraint and sensors | [mjwarp_postconstraint.cu](../../../cuda/dynamics/mjwarp_postconstraint.cu), `world_postconstraint` | 256 threads, two explicit barriers, 1536 bytes reused from acceleration to raw force. Admitted force/torque sensors consume the shared result; large sensor/tree domains retain accepted queries/layers. Pooled external-contact accumulation remains separate. |
| Damped integration | [fused_euler_step.cu](../../../cuda/fp32/fused_euler_step.cu), `euler_step` | Derivative/private factor/solve, integration and publication in 64 threads with 372 shared bytes. The normal cooperative path uses five explicit barriers, apart from SDK internals. Optional world-local accounting adds no barrier. Warning/layout/output fallbacks use prepared factor plus advance and separate accounting. |
| Integration component | [fused_tail.cu](../../../cuda/fp32/fused_tail.cu), `advance` | Velocity/position advancement, Newton warmstart and time/overflow publication; also used for undamped integration. Private damped acceleration never replaces public Newton qacc or warmstart. |
| Observations | [fused_observations.cu](../../../cuda/fp32/fused_observations.cu), `pack_and_sample` | Default five-field packing or direct current-field gathering with ordered dependent sampling. Plans through 128 specs use a fixed 16-byte parallel-clock bitmap; larger plans retain scalar clocks. Camera/public completion remains outside this region. |

The pooled collision reset in
[bridges.cu](../../../cuda/runtime/bridges.cu), `collision_reset`, clears both
pool counters in one small launch. It remains ahead of all producer CTAs;
placing reset in a world producer would introduce a cross-CTA race. Disabled
contact/constraint clears only the contact count and retains the pair pool.
Broadphase, narrowphase families and constraint construction retain their
pooled publication boundaries.

### Build and binding boundaries

[FP32Fusion.cmake](../../../cmake/FP32Fusion.cmake) includes separate recipes
from [cmake/fp32/](../../../cmake/fp32/) for
tail, observations, centered mass, motion, preparation, motion/passive,
controller, kinematic mass, actuation/solve, Euler step and control/force/solve.
Editing one source rebuilds its own target and required device link, rather than
all scene variants. Controller products reuse the separately compiled OSC and
MathDx objects. OSC remains in
[osc_solver.cu](../../../cuda/control/osc_solver.cu); its matrix implementation
is not copied into a giant forward/Newton source expansion.

OSC keeps FP32 resident inputs, goals, gripper state and control outputs, while
its mass inverse, operational-space matrices, Jacobi pseudoinverses, nullspace
projection and torque accumulation use FP64. Conversion occurs while loading
the existing per-world operands; final limited torques are converted to FP32.
Derived-matrix caches also retain FP64 values. The 64-sweep limit, relative
`1e-15` singular-value cutoff and error reporting remain in force; no damping
is added. Higher arithmetic precision cannot recover information already lost
in FP32 inputs or establish the appropriate numerical rank by itself.

After updating these sources, rebuild `flash_mjwarp_controller_fp32`,
`flash_fp32_fused_controller` and `flash_fp32_fused_control_drive` offline in the
existing target-architecture build directory. Their OSC/MathDx objects and
shared-memory layout must be updated together. Physics kernels retain FP32.
Historical trajectory and timing qualifications below predate this change.
For independent small-matrix SVD and controller-torque checks, build the optional
`flash_osc_mixed_replay_probe_fp32` target and run
`LIBERO_OSC_REPLAY_LIBRARY=/path/to/libflash_osc_mixed_replay_probe_fp32.so python -m unittest discover -s tests -p test_osc_mixed_precision.py`.
The sibling `cuda_runtime.so` and `mjwarp_controller_fp32.so` must be prepared;
the test never compiles code during execution.

`cuda/runtime/runtime.cu` and `cuda/runtime/bridges.cu` have independent resource
and transport targets. Shared mathematical leaves live in
`cuda/{dynamics,newton,collision}/mjwarp*`; the name records algorithm provenance,
not a dependency on the Warp compiler or runtime. Source identities include
transitive headers, architecture and C ABI sizes. Pointer/index/owner/alias
admission occurs before capture. Captured replay borrows live arrays and does
not compile, allocate numerical scratch or cache physical factor values.
FP32-only sources do not invalidate native FP64 numerical source identities.

The CPU helpers `../mjwarp/scene_cache.py`, `scene_inputs.py` and model
compatibility helper remain shared. Their directory does not make them Warp
runtime dependencies. Scene/asset/model hashes and source admission remain in
force. Larger input domains use bounded CUDA fallbacks rather than increasing
shared memory with world count.

## Supported scope and validation

The admitted shipped task domain is rigid, dense Newton, elliptic contacts,
Euler integration, the default Panda OSC, FORCE/TORQUE sensors and FIXED
cameras/lights. Flex, tendons, equalities, activation/history, sleep, noslip and
plugins are rejected. This implements the LIBERO batch domain rather than all
MuJoCo features. Native FP64 and the explicit upstream MJWarp reference retain
separate backends and environments.

Stable independent validation commands in `tools/validate_gpu.py` are:

- `cuda-runtime`: buffers, streams, Graphs, conditional loops and ownership.
- `cuda-model`: all arrays/scalars/layouts against CPU-exported pinned data.
- `cuda-functions`: common-input prefix, OSC, Newton and Euler comparisons.
- `cuda-lifecycle`: partial reset, continuation, mutations and callbacks;
  `--cameras --prepare` additionally checks explicit scene preparation and RGB/depth.
- `cuda-complete`: source-bound N=2 trajectories, optional single timing pass.

Reference export is explicit and uses the separate reference environment.
Independent checks prohibit Warp imports; isolated installation checks also
establish that neither package can be found. Existing strict checks, accepted
FP32 trajectory envelopes and geometry diagnostics remain distinct outcomes.

The later world-local fusion campaign uses **regional N=2 endpoints**, normally
1 and 20 replays with current inputs changed after capture, plus independent
FP64 function/residual/sensitivity bounds where applicable. It does not require
bitwise numerical equality. Indexing, exact-copy, padding, owner, counter and
public hold contracts remain exact. No full-task rollout or 130-task timing
was added for these changes. The earlier full-suite results in
[QUALIFICATION.md](QUALIFICATION.md) and the repository throughput chart retain
their original revision and measurement scopes.

Local timing compares each candidate with its immediate accepted regional
reference in one interleaved paired round. Other GPU jobs were resident.
Controller-region measurements use a warmed controller cache; Euler-region
measurements advance warmed local state with fixed matrix/RHS. Those are not
fresh-state or complete-task measurements. Position fusion deliberately excludes
a wider tree layout that regressed. Some larger direction/control timings were
mixed under contention; they are not stable gain claims. Regional ratios must
not be multiplied into an alleged whole-task speedup.

World/pool completion boundaries, the Newton conditional loop and SDK block
contracts remain explicit. Further broad expansion that removes those useful
work mappings requires a new algorithmic organization, rather than simply
concatenating more source or unrolling solver calls.
