# MJWarp physics backend

This explicit reference backend defaults to upstream MJWarp physics with the
maintained GPU OSC, GPU observation sampler, and original LIBERO task predicates.
Select it with `physics_engine="mjwarp"`. Automatic FP32 batch routing now selects
the [independent CUDA owner](../cuda_fp32/README.md), which requires neither Warp
package. FP32 is the default precision; explicit FP64 uses native CUDA. The older
native FP32 comparison path remains available as `physics_engine="native"`.
Physics and OSC here use FP32. No CPU controller or per-substep device download
is used.

The explicit `mjwarp_options={"preset": "cuda_libero", "fast_compile": True}`
selects every migrated numerical leaf required by the130 shipped tasks,
including controller Jacobian, FORCE/TORQUE sensors, gravity projection,
stateless publication and reset/public forward. It remains opt-in while the
full comparison is qualified. Each unsupported domain retains a visible
fallback; the complete-path qualifier rejects such a fallback for shipped
inputs. See [the migration map](MIGRATION.md) for sources and scope.
Intentional changes to the pinned numerical behavior are recorded separately
in [Numerical deviations](NUMERICAL_DEVIATIONS.md), including their domains,
independent evidence and remaining trajectory qualification limits.

Prepare source-bound scenes/operators offline, then run the separate load-only
comparison with `tools/validate_gpu.py mjwarp-complete --build-root ROOT
--output OUTPUT --tasks all --prepare` followed by the same command without
`--prepare`. It uses N=2, five warmup steps and20 measured steps once, no
cameras, with shared imported model and independent original/CUDA state,
controller and observation owners. Initialization/reset/Graph preparation are
recorded separately. Per-task reports retain final differences, capacity,
force-balance diagnostics, actual dispatch and GPU competition. No runtime
compilation is permitted in the measured run.

An opt-in `mjwarp_options={"newton": "cuda_hessian"}` selects project-owned
Newton host orchestration and independently compiled CUDA Hessian assembly.
Initialization, constraint updates, Cholesky/solve, line search, stopping and
noslip remain pinned upstream operations. `"newton": "upstream"` is the default.
The selector visibly falls back to the full upstream solver for sparse,
pyramidal, discrete, sleeping/compact or unqualified dimensions. Requested and
selected paths, fallback reason and the selected Hessian binary identity are
recorded in environment/build metadata. Reset/public refresh defaults to upstream forward; the independent
`public_forward=cuda` selector uses the migrated stages in original public order.

`"newton": "cuda_direction"` additionally replaces gradient initialization,
gradient accumulation, Cholesky factorization, direction solve, direction norm
and Newton decrement with [readable CUDA](../../../cuda/newton/mjwarp_direction.cu).
The small-NV route uses cuSolverDx POTRF/POTRS with 64 threads; the larger route
uses 16-row blocked augmented-RHS Cholesky and 32 threads, following the pinned
upstream block algorithm. It admits the catalogue's 13 NV families and the
qualified upstream block dimensions. Other configurations visibly fall back.
The call chain adds `DenseDirection.enqueue(1)` before assembly and
`DenseDirection.enqueue(2)` afterward. Initialization/warmstart, constraints,
line search, stopping and noslip still execute upstream numerical kernels.
This is a partial handwritten port, not a fully replaced Newton solver.
Both bindings and their context remain owned by the captured graph.

Build `flash_mjwarp_direction_fp32` independently, then prepare the profile with
`--newton cuda_direction`. Its receipt binds both leaf sources and libraries.
The default remains `upstream`; local parity is checked with
`tools/validate_gpu.py mjwarp-direction`, and cumulative Newton equivalence with
`mjwarp-newton --direction-library /path/to/mjwarp_direction_fp32.so`.

`"newton": "cuda_constraint"` adds the independently compiled
[constraint update](../../../cuda/newton/mjwarp_constraint.cu) to that cumulative
port. It computes force/state from existing equality, friction, unilateral and
elliptic contact rows, then accumulates `J.T @ force` with the pinned per-DOF
sequential loop. Both solver initialization and each Newton iteration call the
same selected leaf. The adapted initialization host prefix still calls upstream
reset, `J @ qacc - aref` and `M @ qacc` kernels. Line search, stopping and noslip
remain upstream. No contact/constraint-row generation is replaced by this leaf.
Build `flash_mjwarp_constraint_fp32` offline and prepare with
`--newton cuda_constraint`; receipts bind all three selected leaf libraries.
Use `mjwarp-constraint` for local replay and `mjwarp-newton --constraint-library
... --direction-library ...` for cumulative solver equivalence.

The call chain is `Runtime -> PhysicsStages.newton -> Newton.solve -> _gradient
-> DenseHessian.enqueue -> mass_quadratic / cone_curvature`. `HessianPlan` loads
the library and prepares CSR addresses before graph capture. Each solve creates
fresh upstream scratch; the captured graph retains its own context/binding
owner list. No per-iteration host reads or new device buffers are introduced.
Newton has its own maintained [host source](newton.py); the readable
[CUDA assembly source](../../../cuda/newton/mjwarp_hessian.cu) is built as
`flash_mjwarp_hessian_fp32`, independently of the old fused-substep solver.

Build that target offline, then prepare the selected profile with
`python -m libero.gpu.host.backends.mjwarp.build ... --newton cuda_hessian`.
Its receipt additionally binds the CUDA source, host binding, CMake recipe and
compiled Hessian library. Changing the selector requires a matching offline
profile and a new graph. The local `tools/validate_gpu.py mjwarp-newton` protocol
qualifies the solver stage without rerunning physical trajectories.

Adapter correctness is checked against the pinned upstream engine with the
same model, roots, warmstart and recorded actuator commands. CPU MuJoCo 2.3.7
trajectory differences are separate cross-engine observations. The earlier
three-forward adapter did not satisfy all old CPU error gates and reproduced
a drawer force-balance anomaly in raw upstream execution. Matching upstream
does not establish the upstream solver's own residual quality; force-balance
diagnostics remain separate. No upstream solver code is changed here.

The pinned baseline is `mujoco-warp` commit
`c384ac51df67eb92ab7657cc472ff7494dc036bf`, MuJoCo 3.16.0 and Warp 1.18.0.
Use an isolated environment: the native backend and independent CPU semantic
reference require MuJoCo 2.3.7. Do not upgrade their environment in place.
The integration consumes exported scene XML, not incompatible 2.3.7 MJB files.
Install the pinned package in the isolated environment; runtime does not need
the upstream source checkout.
This initial integration is used from the Flash-LIBERO source tree. The normal
wheel metadata still pins MuJoCo 2.3.7 for native execution; installing that
wheel with dependency resolution in the MJWarp environment would conflict.
Keep the common LIBERO dependencies available separately and import this
source tree in the isolated backend environment. The workspace launcher uses
this arrangement without upgrading the native environment.

Build the `flash_mjwarp_controller_fp32` CMake target once in the existing
unified build directory. It links the existing OSC object and MathDx module;
it does not compile task-specific controllers. Then explicitly prepare Warp
modules and the captured task profile offline:

```sh
python -m libero.gpu.host.backends.mjwarp.build \
  --build-root /path/to/project --tasks libero_10:4 --worlds 2 --fast-compile
```

Ordinary construction is load-only and requires a matching profile. Receipts
bind the scene, versions, capacities, project/upstream sources and controller
binary. Missing/stale profiles and Warp cache misses fail without compilation.
The profile filename also includes source and binary hashes, preserving older
admissions when the optional backend changes.
Preparation captures both policy branches over two substeps without replay;
it asserts zero integrations. Runtime captures the complete control graph.
`--audit-models --tasks all` imports all 130 models without advancing physics;
this checks model support, not trajectory correctness or capacity sufficiency.

```python
env = GpuBatchEnv(bddl, 2, precision="fp32",
                  build_root=project, allow_compile=False,
                  mjwarp_options={"fast_compile": True},
                  use_camera_obs=False, ignore_done=True)
obs = env.reset(seed=0, init_states=initial_states[:2])
obs, rewards, successes, infos = env.step(actions)
```

The initial supported controller profile is the default seven-joint Panda
OSC_POSE with fixed gains and Panda gripper. Other profiles fail explicitly.
`nconmax`, `njmax`, and `collision` (`default` or `primitive`) can be set in
`mjwarp_options`; they require their own offline profile. Defaults scale with
NV and contact dimension. The contact budget is at least 256; row budgets
include contact, joint and equality rows and round up to a power of two
(up to 2048 with the current default contact share). These are allocation budgets, not a
guarantee for every trajectory. Capacity overflows fail explicitly; legal
Newton iteration/line-search limits remain visible as diagnostic warning bits.
The baseline uses dense Newton, matching CPU 2.3.7 AUTO for the catalogue's
NV21..57, and the existing FP32 tolerance policy. Upstream numerical code is
unmodified. NativeCCD is left enabled unless primitive collision is requested.
MULTICCD is explicitly mapped from the original CPU model's enable bit:
MuJoCo 3.16 enables it by default, unlike 2.3.7. Existing nonzero contact
margins are preserved. Instance-local joint address adapters bridge the newer
MuJoCo enum representation without patching robosuite globally.

`GpuBatchEnv.step` captures all 25 physical substeps in one control graph:

```text
upstream step1 -> device gather / GPU OSC / scatter ctrl -> upstream step2
              -> work counters -> GPU observation sampling
```

Each control step therefore calls the position/collision prefix, solver and
integrator 25 times. Newton iterations within each solve remain variable.
The adapter does not perform controller-refresh solves or publish intermediate
warmstarts: upstream integration maintains warmstart. OSC runs on every
physical substep, and its target receives the new policy action once per
control step. Reset initializes the controller from the actual reset state,
and the first invocation consumes the current step1 inputs. It does not replay
robosuite's initial constructor-cache exception.

Current LIBERO task feedback, rewards, success predicates and default policy
observations do not read pre-control constraint accelerations or contact forces.
Robosuite's internal pre-control force/torque history is not reproduced by this
fast path. Public post-solve force and sensor arrays remain available.
Selected reset restores unselected worlds' public derived arrays after the
upstream all-world forward. Their pooled contact read views are held separately
until the next integration; these views are never physical solver inputs.

Files: `model.py` imports scenes; `controller.py` binds resident arrays to
`cuda/control/mjwarp_controller.cu`; `kernels.py` gathers/scatters device data;
`runtime.py` captures physical/control/observation stages; `environment.py`
binds lifecycle and task predicates; `build.py` owns offline preparation.

`stages.py:PhysicsStages` exposes position, velocity, actuation, smooth
acceleration, Newton and integration as separate enqueue methods. `Runtime`
uses their step1/step2 orchestration around OSC. Every stage borrows the same
upstream Model/Data; there is no intermediate host copy or independent model
compilation. The default methods still call the pinned upstream mathematics.
This is the replacement boundary for future handwritten FP32 modules, not a
claim that those modules have already been ported. Raw upstream step1/step2
replay remains independent of this orchestration.

`tools/validate_gpu.py mjwarp-stages --freeze-newton` records GPU-event timings
and one Newton input/output boundary from that exact Model/Data. Its output
binds sources, the offline profile and frozen arrays. The snapshots are
validation-only device copies, outside the Newton event interval. Position
and velocity are children of step1; actuation, acceleration, Newton and
integration are children of step2. Do not add inclusive parents to children
or treat instrumented timings as ordinary throughput.

Validation uses `tools/validate_gpu.py --physics-engine mjwarp --precision fp32`.
`tools/validation/mjwarp_reference.py` compares independent NumPy bridge values,
initial/warm substeps and a complete 5-warmup/20-control-step recorded-control
replay through raw upstream `step1`/`step2`. Use `--prepare` offline for the
validation-only trace operators, then run ordinary checks with compilation
forbidden. Partial reset and public data holds are checked exactly.
The stable entry is `tools/validate_gpu.py mjwarp-reference`. Its
`--raw-repeat` mode compares long-trajectory errors against two independent
upstream references, while retaining the original roundoff-only screen and
separate force-balance diagnostics. Upstream variability never changes the
native/CPU regression policy.
Before cross-engine validation, run `tools/export_mjwarp_mass.py` with CPU
MuJoCo 2.3.7 to export exact packed-mass index maps. This operation advances no
physics. The adapter uses these maps only to compare saved endpoint matrices;
normal environment execution does not depend on the CPU reference process.
Camera rendering, arbitrary model mutation and custom controller profiles
are outside the initial qualification scope. Matching upstream MJWarp and
passing the older CPU numerical gates are separate validation results.

`"newton": "cuda_linesearch"` additionally replaces `M @ search`, `J @ search`,
elliptic line-cost/derivative evaluation, bracketing and accepted-step commits
with [readable CUDA](../../../cuda/newton/mjwarp_linesearch.cu). The two NV routes
(fused Jv through NV=50; 20-DOF partial sums above it) retain the pinned algorithm.
It admits the upstream 32-thread line-search block and otherwise visibly falls
back to the full upstream solver. Build `flash_mjwarp_linesearch_fp32` offline;
prepare the selected profile with `--newton cuda_linesearch`. All four selected
CUDA sources/libraries are bound to the receipt. Default remains `upstream`.
Initialization, stopping and optional noslip remain upstream. Model/Data,
per-solve scratch allocation and conditional Graph control also remain upstream.
No per-iteration device allocation, transfer or shared state cache is introduced.

`"newton": "cuda_newton"` completes the admitted Newton numerical path with
[initialization and stopping](../../../cuda/newton/mjwarp_solver_state.cu).
Warmstart selection, world reset, Jaref and Ma initialization retain the pinned
semantics, including the full Jaref clear and first-part aref subtraction above
NV=50. Dense initialization leaves qfrc_constraint for the projection stage.
Stopping keeps the same tolerance/decrement/iteration-limit rules and the
once-only done/nsolving transition. Optional noslip remains upstream; all 130
catalog tasks disable it. Model/Data, per-solve workspace allocation and
conditional Graph control remain Warp infrastructure, not numerical migration.

Build `flash_mjwarp_solver_state_fp32` independently and prepare with
`--newton cuda_newton`. The small `cuda/newton/mjwarp_products.cuh` shares only
the dot-product code with line search; each caller keeps its own scheduling,
skip and commit rules. Both leaf fingerprints bind that header. The full path
retains all five binary sources and their hashes, and defaults remain upstream.

`"mass_solve": "cuda"` independently selects readable
[mass backsubstitution](../../../cuda/dynamics/mjwarp_mass_solve.cu). The 130-scene
mass-layout inventory contains 1x1 compact diagonal blocks, 6x6 scalar Cholesky
blocks and 9x9 cooperative Cholesky blocks; these are the admitted families.
The scalar path retains the triangular loop order. The 9x9 path uses 64 threads
and cuSolverDx POTRS on the existing upper factor. No matrix factorization,
force assembly, force projection or Euler algorithm is replaced by this option.
Other block sizes, sparse LDL regions, sleeping/compacted and discrete callers
visibly retain the complete upstream acceleration caller.

Build `flash_mjwarp_mass_solve_fp32` offline and prepare with `--mass-solve cuda`.
The binding borrows current Data factors and RHS; it caches only immutable model
layout metadata. Exact in-place solve is supported. Captured graphs retain the
plan, arrays and launch bindings. No per-call allocation or host read is added.
The call chain is `Runtime -> PhysicsStages.acceleration -> Acceleration.enqueue
-> upstream force assembly/projection -> MassSolve.enqueue -> scalar_solve /
tile_solve9`. Reset/public refresh remains upstream. This selector can be combined
with `"newton": "cuda_newton"`; both default to upstream and require their own
source/binary-bound offline profile. `tools/validate_gpu.py mjwarp-mass-solve`
qualifies frozen function inputs; its timings are not full-task speedups.

`"acceleration": "cuda"` additionally selects
[smooth force assembly and external-wrench projection](../../../cuda/dynamics/mjwarp_smooth_force.cu)
and implies `"mass_solve": "cuda"`. Explicitly requesting upstream mass solve
with CUDA acceleration is an error. The existing mass-only selector remains
available for comparison. The complete path retains the pinned FP32 addition
order, world-space force/torque convention, zero-wrench skip and depth-first
subtree traversal. One thread owns one (world, DOF) output; there are no atomics,
block barriers or scratch arrays scaling with model size. It supports the
catalogue's topology through resident indices, not a task-4-specific template.

`SmoothForce.enqueue()` combines ordinary force assembly and external projection
in one launch, then the existing `MassSolve.enqueue()` computes qacc_smooth.
The published qfrc_smooth array remains available. `split=True` is a local
validation comparator using two launches, not another public environment mode.
Force inputs, geometry, factors and outputs are borrowed from the same Data;
there is no host-side zero-force assumption or numerical cache. Discrete,
sleeping and unqualified mass domains retain the full upstream caller.

Build the independent `flash_mjwarp_smooth_force_fp32` target and the existing
mass target offline, then prepare with `--acceleration cuda`. Both binary/source
identities are bound to the receipt. This can be combined with `--newton
cuda_newton`; defaults and reset/public forward remain upstream. Position's
matrix assembly/factorization and integration's factor_solve_i are unchanged.
The stable local qualifier is `tools/validate_gpu.py mjwarp-acceleration`.

`"mass_factor": "cuda"` independently selects
[mass Cholesky factorization and fused factor/solve](../../../cuda/dynamics/mjwarp_mass_factor.cu).
The position schedule keeps factorization after CRB/tendon armature and before
collision/constraint construction. It retains upstream kinematics, mass assembly,
collision, constraints and transmission. Euler retains upstream damping
derivatives, diagonal modification and `_advance`, replacing only the block
factor/solve. Its RHS is the current `efc.Ma`; its private acceleration never
overwrites public Newton `qacc`, and upstream still publishes warmstart.

Compact size-1 blocks use reciprocal diagonals, full scalar size-6 blocks keep
the pinned factor/forward-solve interleaving, and size-9 tiles use 32-thread
cuSolverDx POTRF/POTRS. Opposite-triangle clearing follows Warp; scalar unwritten
regions remain held. The static CSR/block/gather maps are checked before capture.
Out-of-range gather indices represent structural zero, as in upstream. Other
sizes, sparse LDL, sleeping or non-Euler callers visibly use the whole upstream
caller. All shipped scenes use Euler. The mathematical leaf is FP32 and retains
the pinned MJWarp Cholesky semantics rather than CPU 2.3.7's LDL storage.

`host/backends/mjwarp/mass_factor.py::MassFactors` owns the position binding and
one private Euler workspace per environment. Substeps on its stream execute in
order, so that workspace can be reused: each call overwrites the full matrix,
damping derivative and every factor/solution element it consumes. It does not
reuse numerical factors between substeps. Graphs retain this owner and the same
Model/Data. No per-call host read or workspace allocation is introduced.

Build the independent `flash_mjwarp_mass_factor_fp32` target offline, then prepare
with `--mass-factor cuda`. It can be combined with `--acceleration cuda --newton
cuda_newton`. Both position and integration selections are bound to the same
leaf source/library identity. Defaults and reset/public forward remain upstream.
The stable local qualifier is `tools/validate_gpu.py mjwarp-mass-factor`.

`"mass_matrix": "cuda"` independently selects
[CRB accumulation and CSR mass assembly](../../../cuda/dynamics/mjwarp_mass_matrix.cu).
It retains the pinned copy of `cinert` to `crb`, reverse body-depth launches with
componentwise FP32 atomic additions, matrix clearing, and one thread per
(world, DOF) walking its ancestor chain. Spatial inertia uses the pinned vec10
layout. Armature reads use the model's broadcast row convention. Tree/CSR
ownership is checked once before capture; no numerical values are cached and
no additional device workspace or per-call host read is introduced.

The shared `position.py::Position` schedule owns kinematics -> selected CRB ->
upstream tendon armature -> selected factorization -> collision/constraints ->
transmission. Matrix generation and factorization are independent selectors;
neither is run twice. `MassFactors` still owns the Euler factor/solve workspace.
Non-Euler or sleeping matrix callers visibly retain upstream execution.
Reset/public forward, tendon armature, velocity, forces, and integration retain
their existing implementations. The matrix leaf has dynamic model dimensions
and tree layers, without per-task template instantiation or shared-memory
growth with world count.

Build `flash_mjwarp_mass_matrix_fp32` offline and prepare profiles with
`--mass-matrix cuda`; runtime uses completed binaries only. Combine it with
`--mass-factor cuda --acceleration cuda --newton cuda_newton` for the cumulative
candidate. Default remains upstream. The local qualifier is
`tools/validate_gpu.py mjwarp-mass-matrix`; independent FP64 formulas are checked
against installed CPU MuJoCo 2.3.7 on the same frozen inputs. Local correctness
does not establish whole-trajectory equivalence or a performance improvement.

`"com_vel": "cuda"` and `"bias": "cuda"` independently select the readable
[spatial velocity/DOF derivative](../../../cuda/dynamics/mjwarp_velocity.cu) and
[bias-only recursive Newton-Euler](../../../cuda/dynamics/mjwarp_bias.cu) leaves.
The shared `velocity.py::Velocity` caller retains actuator/tendon velocities,
passive forces and the final tendon-bias contribution upstream, in pinned order.
If neither leaf is selected, it delegates the complete upstream `fwd_velocity`.
Non-Euler or sleeping callers visibly fall back before optional binary loading.
The bias RNE leaf implements only `flg_acc=False`; postconstraint RNE is a
separate selector described below.
Public `cvel`, `cdof_dot`, `cacc`, `cfrc_int` and `qfrc_bias` remain available.

Pinned MJWarp branches are complete world-to-leaf paths with repeated common
ancestors. Each CUDA thread computes its branch prefix in registers; one
canonical branch publishes each body and its DOFs. This retains the joint
arithmetic order while eliminating duplicate output writes and cross-thread
prefix reads. A single immutable body-owner array is shared by both bindings
(4 bytes/body, independent of world count). It stores no physical values.
Velocity retains FREE/BALL derivative grouping. Bias RNE retains gravity
broadcasting/disable behavior, layered FP32 force atomics, and force publication
into world; that last convention differs from the CRB accumulator.

Build `flash_mjwarp_velocity_fp32` and `flash_mjwarp_bias_fp32` independently and
prepare profiles with `--com-vel cuda --bias cuda`. Either flag can be used
alone, or combined with the mass, acceleration and Newton selectors. Defaults
and reset/public forward remain upstream. Ordinary execution is load-only.
The local `mjwarp-motion` qualifier uses nonzero frozen velocities and the CPU
2.3.7 `mj_comVel`/`mj_rne` oracle, then checks each leaf and their composition.
Local qualification does not establish performance or full-task equivalence.


`"postconstraint": "cuda"` selects the readable
[postconstraint RNE suffix](../../../cuda/dynamics/mjwarp_postconstraint.cu).
The original predicate in `stages.py::PhysicsStages.step2` is preserved:
explicit postconstraint requests or enabled sensors that require RNE run it
after Newton, before acceleration sensors and integration. The selected caller
executes the literal pinned external-wrench schedule in
`postconstraint_external.py::assemble_external`, then the CUDA suffix. Contact,
equality, applied-force and spatial-tendon wrench builders remain installed
upstream kernels, with their original temporary allocations and ordering.
The default delegates the complete upstream `rne_postconstraint`; sleeping,
non-Euler and empty callers visibly fall back before optional binary loading.

The suffix consumes current `qacc` and `cfrc_ext`, adds each DOF's velocity and
acceleration terms in pinned order, forms inertial/Coriolis forces minus
external wrenches, and accumulates body forces by reversed depth layers.
It updates public `cacc` and `cfrc_int` without writing `qfrc_bias`, `qacc`,
`qacc_warmstart` or `cfrc_ext`. It shares the existing immutable body-owner map
when velocity CUDA leaves are selected; otherwise it owns one such map. There
is no numerical workspace/cache or shared-memory allocation. The bias binary
is independent and unchanged.

Build `flash_mjwarp_postconstraint_fp32` offline, then prepare the desired
profile with `--postconstraint cuda`. It works alone or with the earlier CUDA
selectors. Runtime stays load-only, and reset/public forward remains upstream.
The stable `tools/validate_gpu.py mjwarp-postconstraint` command checks only
this suffix on frozen inputs, including nonzero accelerations and external
wrenches. Local correctness and captured Graph wiring do not establish
whole-application equivalence or performance parity.


`"passive": "cuda"` selects joint spring/damper and final passive-force
combination in [mjwarp_passive.cu](../../../cuda/dynamics/mjwarp_passive.cu).
The independent `flash_mjwarp_passive_fp32` target has no dependency on the
large native, Newton or MathDx translation units. Prepare the selected profile
offline with `--passive cuda`; ordinary execution remains load-only. It can be
used without com_vel/bias CUDA, or combined with all previous selectors.

One thread per (world, joint) owns its complete spring/damper DOF span. The
pinned four joint types, quaternion normalization/difference, shortest-arc
branch, polynomial coefficients and each parameter's broadcast row count are
retained. One thread per (world, DOF) combines the current contributions in
pinned order. Public spring, damper, gravity-compensation, fluid, adhesion and
passive arrays keep their owners. There is no numerical cache, device workspace,
shared-memory array or topology owner allocation for these two leaves.

The [passive.py::Passive](passive.py) caller preserves the original gravity,
fluid and adhesion generation in [passive_other.py](passive_other.py), then
executes the selected final sum and original passive callback. Tendon/flex,
non-Euler, sleeping and empty callers visibly use the whole upstream caller.
Both spring and damper disabled retain the original five-array clear and early
return, including holding adhesion and skipping the callback. This wrapper does
not silently drop fluid or change gravity-compensation actuator routing.

The stable local qualifier is `tools/validate_gpu.py mjwarp-passive`. Existing
130 cached CPU models contain no nonzero joint springs or body gravcomp and no
tendons; all DOFs have nonzero damping and all models enable inertia-box fluid
with density1.2/viscosity2e-5. Nonzero spring/BALL/nonlinear/broadcast/flag cases
therefore require manufactured qualification in addition to real task inputs.
See [the complete migration map](MIGRATION.md) for migrated numerical leaves,
retained upstream mathematics, and infrastructure ownership. Local qualification
does not establish full-trajectory or performance parity.


`"actuation": "cuda"` selects readable stateless scalar-joint actuator force,
current CSR moment projection and joint-force limits in
[mjwarp_actuation.cu](../../../cuda/dynamics/mjwarp_actuation.cu). Build the
independent `flash_mjwarp_actuation_fp32` target offline, then prepare a profile
with `--actuation cuda`. It can run alone or with all earlier selectors; default
and reset/public forward remain upstream. Runtime is load-only.

The admitted domain has no activation state, DynType.NONE, fixed/affine gain,
none/affine bias, scalar SLIDE/HINGE joint transmissions, and zero/one control
slot per actuator. Actuator and control counts are independent, not hardcoded9.
Tendon, stateful, muscle/DC-motor/user types, BALL/site setpoint wrapping,
sleeping, non-Euler and empty domains visibly use the full upstream caller.
Cold admission checks enums/indices once and is shared by product selection and
binding; it never caches physical coefficients or sparse moment values/indices.

One thread per (world, actuator) computes the output force and each CSR row's
projection, retaining pinned global FP32 atomics after clearing qfrc_actuator.
One thread per (world, DOF) adds actuator-side gravity compensation then applies
joint limits. The three act_dyn/act_gain/act_bias callbacks remain in upstream
order after output force and before projection. When ACTUATION is disabled,
act_dot, actuator_force and qfrc_actuator are cleared and callbacks are skipped.
Delayed-control interpolation remains upstream; when history exists, a stable
per-environment control workspace is retained. Direct-control shipped models
allocate no extra workspace. Public force arrays and Newton qacc/warmstart keep
their owners, with no numerical cache or shared-memory allocation.

All130 cached scene XMLs have no defaults and share seven motors plus two
kp1000 position grippers. Representative compiled CPU/MJWarp models validate
that shorthand's enum/parameter expansion. The stable local command is
`tools/validate_gpu.py mjwarp-actuation`, covering force, projection, limits and
composition with independent FP64 rounding/atomic-order bounds and CPU2.3.7
oracles. See [the migration map](MIGRATION.md); local checks do not claim
whole-task or performance parity.


`"euler": "cuda"` independently selects damping-derivative preparation, the
private integration-matrix diagonal update, semi-implicit velocity advancement,
and FREE/BALL/SLIDE/HINGE position integration in
[mjwarp_euler.cu](../../../cuda/dynamics/mjwarp_euler.cu). Build only
`flash_mjwarp_euler_fp32` offline, then prepare the desired profile with
`--euler cuda`. Default selection stays upstream; reset/public forward stays
upstream. Non-Euler, sleeping, stateful-activation and empty domains visibly
delegate to the existing selected mass-factor integrator before loading this
optional binary.

[euler.py::Euler.integrate](euler.py) executes derivative, matrix copy, diagonal
update, existing factor/solve, retained activation, CUDA velocity/position, then
the literal pinned publication tail in [euler_other.py](euler_other.py). With
`mass_factor=cuda`, it borrows that driver's existing private matrix, factor,
diagonal, acceleration and derivative arrays: zero additional workspace. With
upstream mass factors, it owns one equivalent per-environment workspace and
uses upstream `factor_solve_i` there. It does not port sparse LDL mathematics.
EULERDAMP or DAMPER disabled bypasses preparation/factorization and advances
using public `qacc`, without allocating Euler workspace.

Damping-adjusted acceleration remains private; public mass/factors and Newton
`qacc` keep their original owners and contents. The upstream tail inserts control
history before time/overflow handling and copies public `qacc` to warmstart,
not the private Euler acceleration. Parameters are read from their current device
arrays on every replay. This path has no numerical cache, topology owner array,
shared-memory allocation or new physical algorithm.

The stable `tools/validate_gpu.py mjwarp-euler` qualifier covers the four leaves
and their composition on frozen N=2 inputs, including polynomial damping,
independent broadcast counts, all joint types, quaternion normalization and
zero-rotation/zero-quaternion conventions. Independent FP64 roundoff/sensitivity
bounds and CPU2.3.7 quaternion integration qualify local outputs. Caller checks
separately exercise existing factor/solve and publication on real inputs. These
checks do not establish full-task trajectories or performance parity.


`"transmission": "cuda"` and `"actuator_velocity": "cuda"` independently
select [mjwarp_transmission.cu](../../../cuda/dynamics/mjwarp_transmission.cu).
The independent offline target is `flash_mjwarp_transmission_fp32`; prepare
profiles with `--transmission cuda` and/or `--actuator-velocity cuda`.
Both default to upstream, work alone or together with prior selectors, and
leave reset/public forward on upstream. Non-Euler, sleeping and empty callers
fall back before optional library loading. The selectors and fallback reasons
are recorded separately in the environment/runtime transmission identity.

[transmission.py::Transmission](transmission.py) is shared by position and
velocity dispatch. Position replaces only the final `smooth.transmission`
boundary, preserving preceding kinematics, mass, collision and constraint order.
Velocity replaces only `_actuator_velocity`, preserving tendon velocity,
spatial velocity, passive, bias and tendon-bias order. The two numerical stages
remain separate; the original position/velocity stage boundaries are retained.

The transmission leaf admits JOINT/JOINTINPARENT actuators attached to scalar
SLIDE/HINGE joints. It reads current gear and qpos, writes public actuator
length and CSR metadata/moments, and retains pinned per-world atomic row
allocation even for zero gear. Duplicate actuator targets and model-driven
counts are supported. FREE/BALL, tendon, site, slider-crank and body routes
use the complete upstream transmission. Generic CUDA velocity can remain
selected with those upstream transmission routes: it reads current CSR rows,
columns, coefficients and qvel in row order, including empty/long rows.

Only a private integer counter is retained for transmission, four bytes per
world, zeroed before each enqueue. Velocity alone allocates no counter. There
is no physical-value cache, shared-memory array, topology owner allocation or
per-task specialization. Public arrays keep their existing owners. Current
parameter and valid CSR index changes are observed by existing Graphs.

The stable `tools/validate_gpu.py mjwarp-transmission` qualifier compares the
two leaves, their composition and composition into the existing CUDA actuation
against raw pinned upstream operators and independent FP64 arithmetic bounds.
CPU2.3.7 `mj_transmission`/`mju_mulMatVec` provide additional local oracles.
The upstream atomic allocator does not guarantee raw row-offset order;
qualification therefore checks exact row ownership/columns/copied moments,
capacity and tail holds in their logical actuator rows. Floating arithmetic uses
roundoff bounds rather than bitwise equality. See [the migration map](MIGRATION.md).
Local checks and Graph capture do not establish full-task or performance parity.


`"kinematics": "cuda"` selects the readable rigid kinematics stages in
[mjwarp_kinematics.cu](../../../cuda/dynamics/mjwarp_kinematics.cu), independently
built by `flash_mjwarp_kinematics_fp32`. Prepare the chosen offline profile with
`--kinematics cuda`. Defaults and reset/public forward stay upstream. Ordinary
runtime remains compilation-forbidden; non-Euler, sleeping and empty domains
fall back before optional library loading.

[position.py](position.py) calls [Kinematics.forward](kinematics.py), which runs
body poses, body matrices, inertial frames, geom frames and site frames, then
retains original `com_pos`, `camlight`, `flex` and `tendon` ordering (`com_pos`
has its own independent selector below). Mass,
collision, constraints and transmission follow at their existing boundaries.
The selector migrates rigid kinematics, not subtree CoM/inertia/DOF-axis
construction, camera/light updates or flexible/tendon mathematics.

Every thread traverses one validated complete root-to-leaf path, keeping its
parent pose privately and retaining per-body joint order. The existing
first-full-leaf-path owner publishes each body and its joint anchors/axes once;
branches do not read another thread's evolving parent output. All four joint
types, off-center joint transforms, model parameter broadcasting and mocap
inputs retain pinned semantics. Quaternion normalization is local: qpos and
mocap_quat inputs are not rewritten. World pose is held, and static geoms retain
the exact `body_weldid/body_rootid/body_mocapid` early-return rule from upstream.

The immutable body-owner array is shared with CUDA velocity/RNE when present;
otherwise kinematics owns four bytes per body. No numerical workspace, physical
cache or shared-memory array is allocated. Model dimensions are dynamic, with
no per-task specialization. Current transforms/qpos/mocap inputs are read on
replay; static-geom outputs retain their make_data ownership even if local
parameter values are subsequently changed.

The stable qualifier is `tools/validate_gpu.py mjwarp-kinematics`. Independent
FP64 frame formulas propagate normwise quaternion-product, normalization and
rotation sensitivities to componentwise FP32 allowances with fixed8x safety.
Raw pinned operators qualify actual FP32 inputs, including zero quaternions.
CPU2.3.7 `mj_kinematics` additionally checks the common mathematical domain on
normalized FP64 model quaternions/axes: its unit-quaternion rotation shortcut
is not algebraically identical to the pinned homogeneous polynomial for
non-unit FP32 parameters. This oracle normalization changes neither GPU inputs
nor production semantics. The CPU input-normalization side effects and zero
fallback are not imported into this backend. See [the migration map](MIGRATION.md).

## Optional subtree CoM, centered inertia and motion axes

`"com_pos": "cuda"` independently selects
[mjwarp_com_pos.cu](../../../cuda/dynamics/mjwarp_com_pos.cu), built offline by
`flash_mjwarp_com_pos_fp32`. Prepare the selected profile with `--com-pos cuda`.
It supports either upstream or CUDA rigid kinematics. Default selection and
reset/public forward remain upstream; non-Euler, sleeping and empty domains
fall back visibly before optional loading.

`Position.position -> Kinematics.forward(com_pos=...) -> ComPos.enqueue` retains
kinematics, CoM, camera/light, flex and tendon order before CRB/factor/collision.
With no selected position leaf, the original whole `fwd_position` delegation is
retained. The standalone CoM driver also supports the original `smooth.com_pos`
boundary. Five CUDA kernel types implement weighted-position initialization,
reverse-depth FP32 atomic accumulation, division by supplied model subtree
mass, centered spatial inertia and per-joint motion axes. Accumulation includes
the world node. Inertia and axes use each body's root subtree CoM, and the
public vec10/spatial-vector layouts remain unchanged. FREE translations are in
the world frame; FREE/BALL rotational axes are body-matrix columns.

All numerical arrays and immutable tree layers are borrowed. There is no
additional device allocation, shared-memory requirement, numerical workspace or
physical cache. Current mass, supplied subtree mass, principal inertia and
frame arrays are read on every enqueue; broadcast periods are independent.
Cold admission checks dependencies, root references, disjoint DOF spans,
layouts, overlaps and source/ABI/architecture. Runtime compilation is forbidden.

CPU2.3.7 computes subtree mass dynamically and substitutes xipos below mjMINVAL;
pinned MJWarp divides by supplied `body_subtreemass` when nonzero and otherwise
holds the weighted sum. It also evaluates world inertia instead of explicitly
zeroing it. These differences are retained. `mjwarp-com-pos` qualifies normal
mathematics against CPU2.3.7 and raw pinned GPU behavior against independent
FP64 tensor/spatial formulas with input-derived FP32 error bounds, including
zero/tiny mass, world contributions, depth, fan-in and broadcast cases. Optional
mass-chain checks include the existing CRB/CSR assembly. Local qualification
does not establish complete-trajectory correctness or performance parity.

## Optional FIXED camera/light world frames

`"camlight": "cuda"` selects two readable kernels in
[mjwarp_camlight.cu](../../../cuda/dynamics/mjwarp_camlight.cu), independently
built by `flash_mjwarp_camlight_fp32`. Prepare the chosen profile offline with
`--camlight cuda`. Its default is upstream, independent of `kinematics` and
`com_pos`; reset/public forward remains upstream. Mode qualification happens
before optional product loading. A non-FIXED camera or light retains the entire
original camlight boundary upstream, with a visible fallback reason. Non-Euler,
sleeping and entirely empty domains also fall back. An empty camera or light
half alone is a supported no-op.

`Position.position -> Kinematics.forward -> CamLight.enqueue` retains rigid
kinematics -> CoM -> camera/light -> flex/tendon order. Existing CRB, factor,
collision and constraints follow. With no selected position leaf, the original
whole `fwd_position` delegation is preserved. This leaf updates geometric
frames, independently of image rendering. FIXED includes world-mounted and
moving-body-mounted attachments; it means a fixed local transform, not a held
world pose. Each thread owns one attachment in one world. There is no reduction,
barrier, shared-memory array, numerical workspace or additional device allocation.

The binding borrows every input/output. Current local transforms, light
directions and body poses are reread on each replay; the four parameter
broadcast periods are independent. Cold admission checks FIXED modes, body
indices, layout, device, overlap and source/ABI/architecture. Modes and attachment
structure are immutable for a binding/profile; structural edits require
requalification and Graph recapture. Numerical model/frame updates do not.

The pinned quaternion rotation is homogeneous in its wxyz components and does
not normalize either quaternion. The light direction uses Warp1.18's vec3
normalize: sqrt(dot), divide when length>0, otherwise zero. CPU2.3.7 assumes unit
quaternions in parts of its rotation code and substitutes a default axis below
mjMINVAL. The CUDA path preserves raw pinned behavior, including zero vectors.
`mjwarp-camlight` qualifies raw FP32 inputs against independent FP64 frame
formulas and arithmetic/sensitivity bounds, and separately checks actual CPU
mj_camlight on the common unit-quaternion/nondegenerate domain. No rendering,
complete-trajectory or performance claim follows from these local checks.

## Optional inertia-box fluid forces

`"fluid": "cuda"` selects the readable two-stage
[mjwarp_fluid.cu](../../../cuda/dynamics/mjwarp_fluid.cu), independently built
by `flash_mjwarp_fluid_fp32`. Prepare the profile with `--fluid cuda`. Default
selection is upstream, independently of the joint `passive` selector. Ellipsoid,
tendon/flex, sleeping, non-Euler, empty and import-disabled fluid callers visibly
retain upstream before optional product loading. Structural model flags and
topology require requalification/recapture after edits.

`Velocity.enqueue -> Passive.enqueue -> other_forces -> Fluid.enqueue` retains
joint spring/damping -> gravity -> fluid -> adhesion -> final passive sum ->
callback order. Fluid-only selection uses original upstream joint and sum
kernels and prevents whole `fwd_velocity`/`passive` delegation from skipping the
selected leaf. If both spring and damping are disabled, the pinned clear and
early return still skip fluid/adhesion/callback and hold adhesion/workspace.
Reset/public forward remains upstream.

Each force thread owns one world/body, translates root-CoM spatial velocity to
the body inertial origin, rotates into local axes, subtracts wind, derives box
sizes from current mass/principal inertia and computes viscous/quadratic drag.
World and below-MJ_MINVAL bodies publish zero. Principal-inertia combinations
retain the MJ_MINVAL clamp. The second kernel assigns one thread per world/DOF
and projects the current body wrenches in original DFS subtree order. Wrench
storage is (force,torque), while cvel/cdof are (angular,linear). There is no
inter-thread reduction or shared-memory barrier.

One retained device array `(N,nbody,6)` costs `24*N*nbody` bytes, is allocated
once and reused across all Graph substeps. Model/frame/public arrays are
borrowed. Current mass, inertia, density, viscosity, wind and frames are read
after numerical updates without cached physical coefficients. Cold admission
checks exact DFS/root/subtree metadata, DOF bodies, layout/device/overlap and
source/ABI/architecture. All dimensions and broadcast periods are dynamic.

`mjwarp-fluid` checks force, projection and their composition separately; an
optional passive product also checks composition into the existing joint/sum
leaves. Actual CPU2.3.7 mj_passive with other coefficients zero isolates fluid;
mj_applyFT separately checks projection. FP64 reference algebra propagates
FP32 matrix/cross-product, clamp, sqrt and powf allowances with fixed8x safety.
The FP32 conversion of the 1e-15 clamp literal is included analytically, since
near-threshold masses can amplify it. Local correctness/capture does not claim
complete trajectories or performance parity.

## Optional applied/contact external wrenches

`"external_wrench": "cuda"` selects the readable two-stage
[mjwarp_external_wrench.cu](../../../cuda/dynamics/mjwarp_external_wrench.cu),
built offline by `flash_mjwarp_external_wrench_fp32`. Prepare the profile with
`--external-wrench cuda`. It is independent of `postconstraint`, which selects
the RNE suffix. Both default to upstream. The original whole upstream boundary
is retained when neither leaf selects CUDA; either mixed combination executes
one external prefix and one RNE suffix. Reset/public forward remains upstream.

The applied kernel owns one world/body and initializes `cfrc_ext`, clearing the
world body. Applied storage is (force,torque); RNE storage is (torque,force), in
world axes about each body's root subtree CoM. The contact kernel owns one
entry of the flat multi-world list, decodes solver forces, subtracts adhesive
pull only for included contacts, rotates by the full contact frame, translates
torque to each body's root CoM and atomically subtracts/adds the two wrenches.
Both PYRAMIDAL and ELLIPTIC cones and dimensions1/3/4/6 are supported. Elliptic
component addresses may be scattered; pinned row-capacity clipping is retained.
A negative first address contributes no force or adhesion. The first contact
body takes the negative wrench; body0 receives no contact contribution.

All arrays are borrowed from the existing Model/Data, including the current
solver force and device contact count. There is no new numerical workspace,
physical cache, device allocation, block barrier or shared-memory requirement.
The two kernels retain stream ordering. FP32 component atomics retain the pinned
accumulation semantics. Model cone, topology, array shapes and storage are
structural: edits require requalification/rebinding and Graph recapture. Changes
to force, contact count, addresses, world ids and physical frames are read from
the same arrays on replay. Active contact metadata follows the pinned producer
contract: valid geom/world/dimension, nonnegative included component addresses,
and in-range first row for a frictionless pyramidal contact. Padded entries are
never read past the current count or allocated contact capacity.

Cold admission rejects equality, tendon, flex, sleep, non-Euler and unsupported
cone/dimension domains before optional product loading, retaining the complete
upstream external prefix with a visible reason. Parent/root/geom mappings,
array dtype/layout/device/overlap, index size and source/ABI/architecture are
checked once. This domain includes all130 shipped scene configurations; it is
not a claim of130-task numerical or trajectory testing.

`mjwarp-external-wrench` checks applied, contact-only and composed paths against
independent FP64 spatial algebra with componentwise FP32 error propagation and
an absolute-contribution bound for arbitrary atomic ordering, using fixed8x
safety. CPU2.3.7 actual mj_contactForce/mj_rnePostConstraint qualify the common
zero-adhesion mathematics. Its contiguous force rows are repacked from equivalent
MJWarp component addresses; clipped rows become zero. Adhesion is a pinned
MJWarp extension and is qualified directly against the independent formula.
Local correctness and Graph capture do not establish trajectory or performance
parity.

## Optional dense rigid constraint construction

`"constraint_build": "cuda"` independently selects
[mjwarp_constraint_build.cu](../../../cuda/dynamics/mjwarp_constraint_build.cu),
built offline by `flash_mjwarp_constraint_build_fp32`. Prepare with
`--constraint-build cuda`. This is row/Jacobian/impedance/reference construction,
separate from Newton's `constraint` update on already-built rows. Default is
upstream. `Position.position` keeps collision -> selected construction ->
transmission order; construction-only selection prevents whole upstream
position delegation from bypassing the selected leaf. Reset/public forward
currently remains upstream.

Six readable CUDA stages reset counts, generate DOF-friction rows, generate
scalar-joint limit rows, allocate contact row blocks, calculate dense contact
Jacobians/Jqvel, and publish contact reference/regularization data. PYRAMIDAL
and ELLIPTIC cones, dimensions1/3/4/6, frictional/direct/standard solref, refsafe,
separate tangential damping, adhesion, independent broadcasts and disable flags
retain pinned behavior. A CTA owns one contact row, lanes own DOFs, and eight
shared floats bound reduction storage to32 bytes independently of NV/worlds.
All numerical/topological arrays are borrowed: no additional device workspace,
per-call allocation or physical cache. Current device counts and model/contact
numerical arrays are reread. Structure/flags/storage require rebind/recapture.

Counters include attempted rows beyond capacity; only allocated rows are
written. Active contact blocks are contiguous and disjoint, with clipped
components addressed by-1. Inactive contact addresses and unrelated row tails
hold their inputs. Dense scalar rows write only NV columns, retaining padded
columns; contact rows publish zero padding through NV_pad. Contact-enabled
construction clears the complete Jqvel array; disabled constraints reset only
counters. Sparse-only row metadata remains untouched. Atomic row order may vary;
row identity/ownership/category order remains exact.

Cold admission visibly retains the whole upstream builder for equality,
tendon/flex, limited BALL, moving surfaces, sparse, non-Euler/sleeping and empty
unsupported domains before loading. Exact parent/root/geom/ancestor/DOF/joint
metadata, layouts, overlap, integer index bounds and source/ABI/architecture are
checked. The shipped130 models have no equality/tendon/BALL limits and use
dense-auto NV21..57, nine friction-loss DOFs, scalar limits and elliptic cones.
Their geom condim includes3/4. The CUDA implementation is dynamically sized;
qualification also includes NV276/padded288 DOF striding beyond one CTA.

`mjwarp-constraint-build` compares pinned numerical kernels and CUDA with an
independent FP64 row/Jacobian/impedance/reference oracle and propagated FP32
roundoff/sensitivity bounds with fixed8x safety. Integer counters, category and
contact-block ownership, copied values and unwritten bytes are exact contracts.
Actual CPU2.3.7 mj_makeConstraint/mj_referenceConstraint qualify equivalent
zero-adhesion raw inputs with corresponding flags and impratio. CPU2.3.7 stores
zero elliptic tangential pos/margin; pinned MJWarp stores include-margin there.
The pinned storage contract is retained and separately qualified on the GPU.
No performance or complete-trajectory claim follows from local qualification.

`"broadphase": "cuda"` independently selects maintained NXN candidate generation
and plane/sphere/corner-AABB/six-axis OBB bounding filters in
[readable CUDA](../../../cuda/collision/mjwarp_broadphase.cu). All130 shipped
tasks choose NXN under the pinned importer threshold (at most270 geoms and
36,315 unfiltered pairs, below250,000). SAP, sleeping and flex callers visibly
retain complete upstream collision. The selected caller retains one candidate
context across Graph substeps, resets counters in the original order and calls
the original narrowphase/contact filter. Numerical workspace/shared memory are
zero; borrowed model bounds are reread on every invocation. Pooled allocation
retains attempted counts beyond capacity. Build `flash_mjwarp_broadphase_fp32`
and prepare with `--broadphase cuda`; `mjwarp-broadphase` qualifies the local
mathematics and pooled ownership. Narrowphase is not migrated by this option.

`"plane_collision": "cuda"` independently replaces all six plane contact
families in [mjwarp_plane.cu](../../../cuda/collision/mjwarp_plane.cu). Its
[shared contact code](../../../cuda/collision/mjwarp_contact.cuh) implements
current material mixing, explicit pairs, margins/gaps, adhesion, collision
sensors and pooled contact publication. Both candidate contexts and numeric
arrays remain owned/borrowed by the environment. No numerical workspace or
shared memory is allocated. The caller retains convex CCD and non-plane
primitive families, preserving the box-box margin/NATIVECCD route. It can be
combined with `broadphase=cuda` or original broadphase. Flex/sleep/unsupported
callers visibly retain whole original collision. Build
`flash_mjwarp_plane_fp32`, prepare with `--plane-collision cuda`, then use
`mjwarp-plane` for frozen local qualification. This option is not a complete
narrowphase migration.

`"box_collision": "cuda"` independently replaces sphere-box, capsule-box and
primitive box-box geometry in [mjwarp_box.cu](../../../cuda/collision/mjwarp_box.cu).
Box-box is processed only when the original margin/NATIVECCD table routes it
to PRIMITIVE; ordinary convex box-box stays in CCD. Sphere/capsule/box families
are removed from the original primitive residue when selected. Build
`flash_mjwarp_box_fp32`, prepare with `--box-collision cuda`, then qualify with
`mjwarp-box`. It shares the existing borrowed contact binding/material code,
adds no numerical workspace/shared memory, and skips its launch when no
handled family is present. The original capsule-box face-crossing limitation
is preserved and documented in [the migration map](MIGRATION.md).


`"convex_collision": "cuda"` selects the seven rigid convex pair families used
by all130 tasks. Build `flash_mjwarp_convex_fp32`, prepare with
`--convex-collision cuda`, and qualify the pooled entry using
`mjwarp-convex-collision`. The readable implementation is in
[mjwarp_convex_collision.cuh](../../../cuda/collision/mjwarp_convex_collision.cuh),
with separate geometry, simplex, GJK, EPA and box-manifold headers. One retained
global EPA workspace is reused across serial type kernels and Graph substeps;
there is no shared-memory requirement. Original per-type capacity and overflow,
current materials/margins, collision sensors and local manifold metadata are
preserved. `broadphase`, `plane_collision`, `box_collision` and `convex_collision`
can be selected independently. Extra unqualified pairs, including
cylinder-cylinder, and general mesh/cylinder MULTICCD visibly retain upstream
convex collision. See [the migration map](MIGRATION.md) for the exact limits.

`queries=cuda` selects the independently compiled `flash_mjwarp_queries_fp32`
for controller point Jacobians, rigid gravity compensation and FORCE/TORQUE
sensors. All inputs remain borrowed and current. Sensor history/delay and
other sensor families retain the complete original sensor boundary. Callbacks
retain stage order; disabled energy is cleared without evaluating energy math.
The local `mjwarp-queries` qualifier checks frozen N=2 original/updated inputs
against actual CPU2.3.7 public queries and independent FP64 operation bounds.
This function qualification does not establish complete trajectory parity.


Before modern scene preparation, export the original prepared fixed poses in
the CPU2.3.7 environment with `python tools/validate_gpu.py mjwarp-scene-inputs
--build-root ROOT --tasks all`. These small portable inputs preserve original
fixture placement and joint indexing; the modern engine never reads an old MJB.
Runtime missing/stale input fails with offline-preparation guidance.

For one environment/model/Graph lifetime per task, `mjwarp-complete
--prepare-and-run --resume` completes explicit offline preparation first,
then freezes compilation for the unchanged single measurement. CPU pose-oracle
checks precede the run. Function replays and snapshots happen outside timing.
