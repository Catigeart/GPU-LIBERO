# GPU Source Documentation

The source tree keeps the GPU implementation under `libero/libero/gpu/`. Both source checkouts and installed packages expose it through the canonical `libero.gpu` namespace.

## Directory Layout

```text
gpu/
  cuda/              Handwritten CUDA and adapted upstream device code
    position/        Kinematics, coordinate transforms, and mass matrices
    collision/       Collision detection, contact processing, native geometry
    constraints/     Constraint rows, impedance, reference values, projection
    dynamics/        Velocity, forces, and smooth acceleration
    newton/          Newton iterations, Hessians, line search, MathDx bridges
    control/         Joint and OSC controllers
    integration/     Euler integration
    postprocess/     RNE, sensors, and Newton finalization
    observations/    Low-dimensional observations and scene sampling
    runtime/         Publication, restoration, copies, and version checks
    common/          Shared mathematical helpers
  host/              Python host code
    factory.py       Physics preparation and environment assembly entry points
    bindings/        CUDA argument bindings and component installation
    build/           Compilation, parameter ABI, and kernel construction
    runtime/         Owners, workspaces, scheduling, and lifecycle protocols
    assembly/        Composition, pipelines, and MathDx integration
    adapters/        LIBERO environment and observation adapters
    paths.py         Canonical source locations
  model/             Static model construction and metadata
  reference/         Reference operators and runtime construction templates
```

## Entry Points

LIBERO users normally create environments through the original public API:

```python
from libero.libero.envs import OffScreenRenderEnv
```

Low-level callers can use:

```python
from libero.gpu.host.factory import prepare_cuda_physics
```

This entry supports `nworld` and `dataflow_profile="fast"` or `"compatible"`. The physics preparation default is `"fast"`; existing internal component profiles remain independently configured. The same module exposes `create` for environment assembly.

Sequential batch environments may share an explicitly qualified build session:

```python
from libero.libero.envs import GpuBatchEnv
from libero.gpu.host.build.aot import qualify_ready_session

with qualify_ready_session(build_root, num_envs=8, seed=0) as ready:
    for task_file in task_files:
        with GpuBatchEnv(task_file, num_envs=8, seed=0, build_root=build_root,
                         ready_session=ready, allow_compile=False) as env:
            ...
```

Qualification uses CPU filesystem checks to validate the complete current
manifest, all task bindings and all artifact SHA256 values once. Keep the source
and build tree immutable throughout the session. Subsequent constructors check
the full source file set and file metadata (device, inode, mode, size, mtime and
ctime); an ordinary change permanently invalidates the session. Requalification
performs the full content checks again. This is a process-local optimization,
not a filesystem lock or protection against forged metadata. Sessions reject
serialization and reuse after a fork or close. Prepared scene hashes and
product-specific loading checks still run independently. Omitting
`ready_session` retains strict full-bank hash validation for each construction.
Changed source requires a matching offline-qualified manifest; the session API
never refreshes a manifest or grants compilation authority. `ready.info` exposes
small qualification metadata; `ready.record` returns a detached manifest copy.

A batch freezes one detached model snapshot shared by all world facades.
Per-world writable model and visual views remain private and are allocated on
first access. Fixed fixture poses are applied before computing the final model
identity. The native fast Newton path selects its diagnostic profile before
constructing the solver, so unused witness history is never allocated.

The fast combined Newton entry seeds force/state only through the bounded live
row prefix and omits input copies of solver statistics, whose published records
are fully overwritten by this entry. Unwritten public row/stat tails stay held;
the compatible path retains full seeds. Ordered cone admission also records the
last middle-cone Hessian writer, avoiding a second classification pass without
changing error ordering, solve count, or warmstart behavior.

The fast Newton path cooperatively builds the line-search tile-head index only
through `ceil(nefc / 16)`, including its final sentinel. Unused index capacity
is private scratch; the compatible path retains its full-capacity records.
For development, `HandwrittenCudaCore(..., phase_profiling=True,
phase_trace_capacity=150)` can retain bounded per-world invocation timings and
work counters. Trace overflow increments the invocation count without writing
past capacity. Parent and child timings overlap, and instrumented timings must
be compared separately from normal control-step measurements. This profiling
is disabled by default.

`GpuBatchEnv(..., initialize=False)` prepares owners without capturing a control
graph. Reset before stepping; the first step captures the graph if needed.
Ordinary resets retain the graph. The offline builder explicitly records the
two controller branches for module preparation. CUDA capture uses Warp's
driver-dependent module-loading default and retains the compilation guard.
Keep the same batch open and reset it between episodes to amortize construction.
Pass `profile_startup=True` to collect inclusive cold-phase wall times in
`env.startup_timings`; disabled profiling performs no clock reads and adds no
step instrumentation. Nested phases must not be summed with their parents.

Start with `cuda/` to inspect device mathematics and `host/` to inspect memory, launches, caching, bindings, and restoration. Some cold construction paths read operators and workspace templates from `reference/`, so that directory is part of the runtime source dependency set.

## Configuration and Lifecycle

[Algorithm Configuration](ALGORITHM_NOTES.md) describes algorithm choices, defaults, and capacity boundaries. Batched cone/rebuild selection and Post radix/topology reuse retain their configured conditions. Newton islands and collision temporal caching are explicit experimental choices with their original admission checks.

`GpuBatchEnv(..., algorithm_options=...)` also accepts `collision_schedule`
(`auto`, `fused`, `compact`, `parallel`), `collision_workers` (positive integer,
default 128), and `forward_reuse` (`auto`, `off`, `epoch`, `exact`). The parallel
schedule compacts live query tiles across worlds and uses separate primitive,
convex, box-box and capsule-box workers, followed by the original ordered commit.
`auto` selects parallel collision for up to 32 worlds and fused collision above
that size; this is a cold policy, not runtime autotuning.

For an optimized native batch with immutable model parameters, automatic reuse
retains position kinematics, the mass matrix/factor, collision and position
constraints until the world's state epoch changes. Euler, reset, state upload
and recovery advance that epoch. Every prefix kernel skips cached worlds before
writing scratch or public data, so mixed batches recompute only missed worlds.
In the component schedule, velocity, reference, bias, control forces, Newton and
warmstart still execute on every logical forward. The fused-substep entry also
reuses velocity/reference/bias within one physical substep, as described below.
Position/mass or collision warnings prevent position-cache hits.
Direct writes to resident arrays without the owner's state-writer protocol are
outside this mode's contract; general component owners keep the uncached path.
Debug mode retains the exact-input diagnostic cache. For independent comparison,
use `{"collision_schedule": "fused", "forward_reuse": "off"}`. Prepare each selected
variant offline; load-only runtime continues to reject missing compiled products.

- Device mathematics uses CUDA; standard matrix operations use cuBLASDx and cuSOLVERDx.
- FMA is enabled; precise division, square root, and non-FTZ behavior are retained.
- Model, source, toolchain, policy, and owner changes require the corresponding cold admission or explicit invalidation.
- Capacity checks, stable ordering, errors, fallback paths, and lifecycle ownership remain part of each implementation's contract.

With collision diagnostics disabled, the fast path writes one route descriptor
per live detector call and schedules candidate rows by their actual counts.
Private candidates contain 28 doubles; transactional commit expands the public
65-double contact record, including the required zero `mu` and Hessian fields.
Exact BoxBox deduplication keeps the last coincident position in original order.
Immutable model owners precompute ordinary geom-pair material parameters once
on the GPU, shared across worlds. Native model replacement rebuilds this table;
explicit pairs, caller margins, mutable workspaces, and models whose optional
dense table would exceed 17 MiB use direct mixing. Compatible/debug paths retain
the full diagnostic records.

The fast collision path omits BVH visit, stack, leaf and visualization tapes;
their contents are unspecified. Physical requests, traversal scratch, live counts
and errors remain available. Unused detector and append-counter tails are also
unspecified. BoxBox queries use one lane per query in this profile, retaining the
scalar clipping algorithm and avoiding replicated per-warp private contact state.
Compatible detector calls keep the collective mapping and full capacity records.

Independent-function measurements do not establish whole-application speedups or coverage of every model and platform.

## Single-physical-substep CUDA fusion

`GpuBatchEnv.prepare_substep_fusion(build_root, allow_compile=False,
fast_compile=False, device_lto=False)` explicitly selects the single-substep
implementation. A CUDA Graph organizes the physical substeps in a control step;
the CUDA kernel itself advances one physical substep and has no control-step
loop. The first invocation receives `policy_step=True`.

The authoritative source files are:

| Source | Responsibility |
| --- | --- |
| `cuda/step/fused_substep.cu` | Single-substep orchestration, forward dynamics, controller entry and integration. |
| `cuda/control/controller_pipeline.cu` | Dimension-general gather, overlapping seed copies, solve scheduling and publication; ordered alias fallback. |
| `cuda/control/osc_solver.cu` | Independently compiled fixed 7/6 OSC numerical body; collective success return. |
| `host/runtime/step/controller_copy_plan.py` | Cold packet scheduling and all-world overlap/wiring proof. |
| `cuda/step/newton_solver.cu` | Independently compiled FP64 Newton numerical call tree. |
| `cuda/postprocess/post_solver.cu` | Dimension-general force/acceleration Post, sensor math, publication and internal output demand. |
| `cuda/postprocess/post_solver.cuh` | Collective Post API; receives the completed qacc and reusable scratch views. |
| `cuda/postprocess/post_profile.cuh` | Post-only timestamps in the optional instrumented product. |
| `cuda/step/newton_jacobian_cache.cu` | Position-generation row/column/product cache and sparse reductions. |
| `cuda/step/newton_jacobian_layout.cuh` | Bounded persistent global cache offsets. |
| `cuda/step/newton_transpose.cu` | Dense column ownership and shared-partial candidates. |
| `cuda/step/newton_solver.cuh` | Small collective device API and Newton execution options. |
| `cuda/step/substep_shared.cuh` | Shared base and byte-offset address helpers; no numerical bodies. |
| `cuda/step/newton_types.cuh` | Preparation-time global model/world/Post descriptors; no private workspace. |
| `cuda/step/newton_workspace.cuh` | Private shared scalar aliases and their layout assertions. |
| `cuda/step/newton_shared_layout.cuh` | Private Newton offsets, bounded coefficient cache and host launch capacity. |
| `cuda/step/substep_shape.cuh` | Canonical fixed dimensions shared by the fusion and independent MathDx units. |
| `cuda/step/substep_shared_layout.cuh` | Shared address conventions and Smooth/OSC/Euler offsets and lifetimes. |
| `cuda/step/forward_workspace.cuh` | Per-world global forward-cache metadata and fluid temporary-storage contract. |
| `cuda/step/substep_types.cuh` | Named physical argument groups and explicit continuation points. |
| `cuda/step/substep_launch.cu` | Ordinary CUDA launch and configuration API. |
| `cuda/step/instantiations/*.cu` | Fixed-size MathDx numerical instantiations. |
| `host/runtime/step/fused_substep.py` | Initial parallel prefix, capture-time substep schedule, device conditions, and resume calls. |
| `host/runtime/step/substep_binding.py` | Explicit retained-owner bindings; no object-graph discovery or source generation. |
| `host/runtime/step/newton_binding.py` | Preparation-only adaptation of retained owners into fixed Newton views; world slicing and topology qualification. |
| `host/runtime/step/substep_types.py` | Maintained ctypes mirror of the CUDA argument types. |
| `cmake/Substep.cmake` | Incremental compilation and linking of `substep.so`. |

The default product links separate CUDA units for the substep, Newton,
Post, launcher, MathDx instances and sparse reduction/cache operators.
CMake compiles maintained project sources directly. Newton and Post are
ordinary collective device calls inside the parent kernel; neither entry is
force-inlined. Their implementation bodies are absent from public headers.
With default device LTO disabled, editing Newton numerical code rebuilds its
object and final links while the outer substep and MathDx objects remain cached.
Post is one dimension-general object shared by all catalogue specializations;
editing its body rebuilds that object and dependent links. The excluded
`flash_substep_post_profile` target instruments only Post and reuses ordinary
Newton/substep objects. Editing a shared ABI/layout header rebuilds its actual
consumers. The outer substep reads qacc/scratch offsets to hand completed Newton
storage to Post; it does not include Newton's private control implementation.

Within one forward, a successful smooth phase passes its shared force and
acceleration directly into Newton. Binding first qualifies that these are the
same owners as the published Newton inputs. Skipped forwards use the canonical
global inputs. Durable outputs, every Newton request and warmstart publication
retain their original order.

Within one physical substep, successful velocity/passive, constraint-reference
and bias results remain in their existing global owners. Later forwards reuse
them only for the same state epoch and position-prefix generation. Every new
substep clears readiness; any prefix-resume path computes these stages again,
including a failed rebuild that did not advance the generation. Each forward
still republishes the current stage and RNE-cache epochs. Actuation, smooth
acceleration, Newton and warmstart continue on every admitted call.
Binding qualifies that controller/Newton/Post writes cannot overlap retained
numerical results or their successful status flags.

The fixed native substep retains complete Post output on both intermediate
policy-step forwards and the final forward of every physical substep. Other
intermediate forwards defer unconsumed force/sensor work after the original
Newton/D/state/forward admission succeeds. A deferred call still publishes
working qacc, clears Post validity, and leaves its output stamp unchanged;
intermediate forward completion uses the current Newton result. No stale Post
payload is marked current. Diagnostics and failed admissions use complete Post.
For the normal 25-substep schedule with three active forwards, this evaluates
Post 27 times per control step instead of 75. Final full-body force and sensor
outputs, observation sampling, reset forwards and all 75 Newton solves remain.

With a valid velocity-bias cache and at most one block's worth of bodies,
Post assigns ancestor-chain acceleration sums to body/component lanes. The
velocity/gravity part remains cached; the qacc contribution is recomputed for
each solve. This changes addition association and removes the parent-level
barriers. Larger trees and missing caches retain the level recurrence. Reverse
force propagation assigns the six independent spatial components to lanes,
retaining each component's original child accumulation order and level barriers.

The additional device allocation is 48 metadata bytes and
`8 * (9*nbody + nv)` fluid-workspace bytes per world, sized from the actual model.
It does not extend the shared arena. Fluid body loads are computed independently;
each DOF then accumulates bodies in their original order, with two collective
votes and no per-body barrier or atomic sum. Diagnostic and exceptional inputs
use the original ordered fluid path before any candidate force is committed.

Newton prepares its immutable constraint grouping once per invocation. Its
line-search preparation keeps only the direction norm and two scales in a
24-byte shared region; each thread makes the same local ready decision after
the required numerical publication. Per-direction schema/alias scans and
redundant publication barriers are absent from the owned fused path.

A fixed cache stores at most 128 rows of `sqrt(D)`, friction thresholds and
elliptic-cone `Dm` coefficients, using at most 3 KiB per world block. Rows beyond
this prefix use the original equations. Equality rows remain quadratic and do
not need rank-update square roots. The existing entry barrier publishes the
cache together with metadata; no cache-specific barrier or capacity-sized
matrix/global cache is introduced. Coefficients are refreshed for every Newton
call, including recovery and single-world prefix misses. Constraint activity,
cone Hessians and search directions continue to change within the iteration.

Warmstart comparison computes the warm and smooth constraint costs in one
cooperative pass, without producing candidate forces/states or their transpose
products. The warm Gauss term shares the cost reduction's publication. The
strict cost comparison selects `qacc`; the selected residual then receives a
complete constraint update and `J^T force` before the initial Hessian/direction.
Each solve, CPU stopping rule and final warmstart publication retain their
semantics. Entry admission disables a malformed world before any consumer can
read a partially constructed constraint layout.

The kernel has three logical calls to one `forward_dynamics` definition. That
function calls the velocity/passive, reference, bias/actuation, smooth and
constraint-projection functions, then one `solve_newton` definition. Controller
refresh admission determines whether the second forward performs work.
`solve_newton` consumes prepared model/world views and performs initialization, warmstart selection,
initial Hessian/direction construction, Newton iteration loop, and final
publication. `forward_dynamics` then calls `physics_post::run` with explicit
output demand before publishing the forward. Complete calls preserve the
RNE/sensor/cache order; deferred calls retain working qacc and invalidate Post.
Constraint update, Hessian assembly/factor/update,
direction, line search and convergence statistics remain separate CUDA
functions with their original thread scopes and synchronization contracts.
The local factorization chain is `local_potrf` -> `block_potrf<N>` -> MathDx,
with `chol_factor` as the regularized fallback. The world block owns one dynamic
shared allocation. Numerical functions form local aliases from its declared
byte offsets; component-local flags, reductions and some SDK/contact staging
buffers retain static shared storage. Newton helpers live in `newton_solver.cu`
and Post helpers in `post_solver.cu`; common primitives and SDK code use their
normal headers. Device
function calls stay within the same kernel launch; this organization adds no
kernel launches. Controller execution, state checks, warmstart publication and
the explicit continuation entry points retain their order.

The fusion path has its own Newton ABI: eight model pointers and 69 per-world
pointers, with Post in a separate descriptor. The old generic generated
`Parameters` layout is consumed only by the preparation adapter. Per-world
pointer offsets, fixed dimensions and immutable mass-plan topology are qualified
once before capture; a prepared instance requires unchanged model topology.
Internal iteration flags, costs and counters live in the `NewtonControl` scalar
region of the dynamic shared allocation. Warmstart, constraint-force and
warm-mass vectors have explicit byte offsets in `NewtonLayout`.
DOF lanes load their own vector cells and cooperatively zero control storage.
`substep_shared_layout.cuh` declares the complete 41568-byte dynamic allocation.
Smooth force and acceleration occupy the protected first 640 bytes. Newton
starts at byte 640, preserving the relative matrix offsets and compact MathDx
leading dimensions of its previous 39712-byte numerical region; uniform control
and warm vectors follow it. Smooth temporary storage overlaps the future Newton
region after its last reader. OSC (including all three simultaneous pseudoinverse
workspaces) and Euler overlay the allocation only after the previous phase has
published and finished reading its outputs. `substep_launch.cu` uses this capacity
directly, and Python queries the compiled capacity instead of duplicating it.

The constraint-update, Hessian-update and direction interfaces receive the shared
base and prepared global owners. Line-search helpers access these same owners
directly; the former 34-pointer `S07Args` packing/unpacking has been removed.
`NewtonControl` is only a typed alias for uniform scalar cells, with compile-time
size/offset checks; it does not own or copy numerical arrays. Large constraint
arrays and canonical state remain in global memory. Each world also owns a
30053-entry global integer buffer for
ordered changed-row compaction; this buffer is retained by the prepared binding.
The entry parser builds the constraint head lists once per solve. Warp 0 checks
contact metadata after that publication, while later constraint updates reuse
the lists. Smooth and warmstart Jacobian products share a two-column GEMM, and
the direction uses one FP64 POTRS with the existing Cholesky factor. The warm
mass product and its Gauss term share output ownership; short-vector publication
uses DOF ownership instead of the general packet-copy policy.
Final publication remains at each solve, and Post reuses dead
MathDx scratch. Inner ABI sizes are checked before any kernel launch, so an old
library requires an explicit offline rebuild. FP64 equations, line-search,
regularization, incremental-update order and stopping rules are retained.
Matrix primitives and reductions may reassociate arithmetic; correctness is
qualified by end-to-end accumulated error against the accepted CPU/GPU results.

The call sequence is:

```text
GpuBatchEnv.step(action)
  -> replay control-step CUDA Graph
     -> for each captured physical substep:
        -> existing forward admission and parallel position/mass/collision prefix
        -> fused_substep_kernel(Start, policy_step)
           -> application forward dynamics and Newton
           -> controller refresh forward and Newton; OSC
           -> mj_checkPos; mj_checkVel
           -> physical forward dynamics and Newton
           -> mj_checkAcc; Euler
        -> selected-world prefix/reset continuations, if required
        -> existing device observation update
```

One 64-thread block owns one world. The normal path invokes one fusion kernel
for the world-local suffix. Position/mass/collision remains in the existing
multi-CTA prefix. Three small continuation-selection kernels execute per substep;
CUDA conditional nodes run additional prefix/resume work only for suspended
worlds. Observation sampling remains outside the fusion kernel. Thus this entry
does not claim that the entire physical substep is a single CUDA launch.

The continuation points are named `ControllerPosition`, `PhysicsPosition`, and
`AccelerationReset`. The first two resume after the requested parallel position
prefix. The last performs the original reset-only full forward and resumes at
Euler. Controller updates and completed Newton/warmstart writes are not repeated.
The prepared nonreset-world preservation owner protects the other worlds around
these conditional branches. There is no host predicate readback.

Execution order follows CPU MuJoCo 2.3.7 and the CPU reference's robosuite:
application `sim.forward`, controller update/conditional `sim.forward`, then
`mj_step`'s checkPos/checkVel/forward/checkAcc/Euler sequence. Position reuse never
substitutes for a requested Newton solve. Aliased controller publication copies
retain the original lane-0 order; independent fields use the whole block.

Scalar admission chains execute in order on lane 0 and synchronize when other
lanes consume their results. A clean checkPos/checkVel/checkAcc returns after its
block-wide decision, without traversing reset fields. Actual resets retain their
ordered field barriers. Decision votes and no-op exits also close flag-read
lifetimes before a subsequent stage can overwrite those flags.
Cold controller binding checks all float/int source and
destination spans, including cross-field and cross-world aliases, before marking
a field eligible for parallel copying. Actuator velocity dots run independently
with their original accumulation order; exceptional/aliased inputs retain scalar
behavior. Newton copies live state rows and direction vectors cooperatively,
preserving ordered shifted-alias behavior. The Jacobian-vector and transpose
helpers own their exit synchronization; callers do not repeat it. Matrix,
reduction, and shared-storage lifetime barriers remain block-wide.

Smooth-force assembly keeps its world-local operands in shared memory. General-
path factor scans, result export, and `d_commit_smooth_shared_block` use the whole block;
only routing, warnings, and final validity publication have a single writer.
Overlapping public commit views retain their ordered scalar fallback.

On a current position-factor hit, smooth reads the completed packed LDL factor
directly. `cuda/step/forward_mass.cuh` keeps one solution value per lane and
cooperates in 8/16/32-lane groups, preserving the single-RHS MuJoCo update order.
`host/runtime/step/forward_plan.py` builds immutable component bounds, depths and
ancestor masks once; large or interleaved components retain the general solver.
Existing smooth work-array exports, warnings and canonical commit remain valid.
Velocity and the bias RNE forward pass share one body-level traversal and one
barrier per level. Exceptional inputs retain the separate stage routing.
Small actuator sets use one lane per actuator and one lane per generalized-force
output, without the MathDx matrix staging; larger sets retain the library route.
The three scalar Post epoch snapshots share their leader with the epoch producer
and are published together with contact addresses before Newton/Post.

Newton line-search preparation reuses the invocation's validated constraint
groups and ordered packed-mass product descriptor. Each output row has one
owner, including every assignment event in the original scatter order. Small
dot-product statements are maintained in `newton_solver.cu`: four lanes compute the four
ordered accumulator streams and another computes the original grouped tail.
`numerics/detail/newton_policy.cuh` retains `LIBERO_NEWTON_TRACE=0` for this path;
line-search evaluation counts and stopping decisions remain active, while
evaluation-trace initialization and writes are disabled. A diagnostic build can
enable the policy and uses a cooperative trace fill.

OSC state and derived-matrix cache payloads are copied cooperatively. On a
derived-cache miss, `cuda/control/osc_solver.cu::pinv_three_block` computes the
6-by-6 pseudoinverse in warp 0 and the two 3-by-3 pseudoinverses in the two
half-warps of warp 1. Each group owns matrix rows, retains ordered Jacobi column
pairs, and synchronizes before reconstructing cells that read other lanes'
rows. Scaling, the relative singular-value cutoff, and the orthogonality check
follow the retained scalar algorithm. Numerical acceptance uses accumulated
end-to-end error against the CPU reference, rather than byte equality. The
padded 4/8-row shuffle tree changes summation association; ordered column-pair
rotations, the 64-sweep bound and rank/convergence thresholds are retained.

`pinv_group` keeps each lane's column and right-singular-vector row in
`PseudoinverseRow<N>` registers. Each sweep rotates fixed slots 0/1; cyclic
register moves bring the next logical column into slot 1. Columns and vectors
follow the same permutation, retaining the ordered logical p/q sequence in one
compact rotation body. Canonical column order is restored before the next
sweep and every convergence exit. Runtime selectors remain only in the
post-sweep norm/correlation tail.
The completed rows are written once to the existing shared staging matrices
before the cell-owned reconstruction and its original warp rendezvous.
The 3/6 dimensions, participating groups, reduction tree and numerical
thresholds are unchanged. This trades shared accesses for register moves
and additional registers; correctness and compiled resources do
not by themselves establish a performance improvement. The norm/weight tail
and device-call stack may still use local memory; only the iteration matrices
are intended to remain in registers.

`fused_substep.cu::run_controller` calls the independently compiled
`cuda/control/controller_pipeline.cu::physics_controller::run_controller`.
The pipeline calls `osc_solver.cu::physics_controller::solve`; both entries
are `noinline`. Each common object serves all catalogue scenes with the fixed
7-joint/6-task controller. The solver accepts `ControllerSolveData` by reference
and the existing shared arena, returning the same success decision on every
lane after its final rendezvous.

Binding constructs a retained device packet table once, after proving all-world
float/int owners, gather accesses and gate wiring compatible. Warp zero gathers
while the remaining warps seed independent channels. Six velocity dots keep
full-NV accumulation order but use six lanes. Pose, velocity and bias copies
are also cooperative. Packet descriptors preserve each field's world stride
and odd tail; no packet crosses a field. Two pipeline block barriers publish
inputs and commit payloads. The caller's existing recovery-admission rendezvous
closes final metadata publication. Commit eligibility/event count stay in
registers, preserving first-error precedence and counter-overflow behavior.
Unknown wiring or overlapping owners use the retained ordered path, with its
original publication order. Independent component controllers are unchanged.

On a matrix-cache miss, OSC computes `X = M^-1 J^T` once and reuses it for
`A = J X` and the nullspace map. This removes the separate `J M^-1` product.
The five final small matrix-vector products use two warp-register phases:
task wrench and posture force are independent producers; torque lanes consume
their values through full-warp shuffles. They require no block barriers or
materialized projector transpose. The final block publication barrier remains.

The fused Newton path expands `constraint_update_core_block`,
`hessian_prepare_core_block`, `direction_core_blas_dx`, and
`commit_step_core_block`. It retains the actual solve/warmstart order and factor
pivot/rank-loss handling, while omitting repeated component admission scans and
transactional AXPY preflights. Constraint classification supplies both the
last-middle-cone writer map and force/cost computation. Tile offsets cover only
the live row range. The temporary cone norms reuse Jv after its direction has
been consumed; the next line search regenerates Jv. No new shared row array is
allocated. The checked component functions remain separate from this core path.

`NewtonOptions` defaults to MuJoCo 2.3.7's improvement-or-gradient stopping rule.
The previous componentwise roundoff-residual policy is an explicit comparison
option. Optional per-world counters/first-event scalars are null by default;
normal execution performs no residual audit or audit-buffer writes. The opaque
Newton owner ABI remains 2040 bytes; the maintained substep argument ABI is
6288 bytes and is checked against ctypes when the library is loaded.

Newton's H, Ma and other invocation-local vectors are no longer imported from
or exported to persistent diagnostic scratch on every solve. The direction
producer writes its next search vector directly. Canonical acceleration,
warmstart, constraint force/state, solver stats and Post sensor outputs retain
their publication. These private scratch arrays are not a fused diagnostic
snapshot; component diagnostics use their existing checked entry.

Euler's normal path is the `evaluate_core` function in `fused_substep.cu`: one RHS,
`M + dt*diag(damping)`, velocity/position integration and canonical state
publication. It preserves pivot clamps/warnings and quaternion normalization,
without repeated whole-state finite scans or quaternion trace stores. The
unused inverse square-root diagonal is not computed. The outer substep still
runs the CPU-equivalent checkPos/checkVel/checkAcc recovery sequence.

OSC's private workspace is qualified once at binding. The fused gather and
control use this qualification to skip repeated input/alias scans. Thirteen
fully overwritten outputs need no old-value seed; their successful publication
is retained. Cache/history, targets, gripper and partially written quaternion
or nullspace diagnostics retain their seeds. Exact-input derived-cache
validation and pseudoinverse convergence/cutoffs remain active.

Build from the repository root, with an external build directory:

```sh
cmake -S . -B "$FLASH_BUILD_ROOT/sm80-dev" -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DFLASH_CUDA_ARCH=80 \
  -DFLASH_MATHDX_ROOT="$FLASH_MATHDX_ROOT" -DFLASH_FAST_COMPILE=ON
cmake --build "$FLASH_BUILD_ROOT/sm80-dev" --target flash_substeps_all --parallel 4
```

Pass the same `FLASH_BUILD_ROOT` and `fast_compile=True` to Python. For the
standard assembler policy use `sm80-release` and `FLASH_FAST_COMPILE=OFF`.
The installed GPU package has its own standalone CMake entry and produces
`substep.so` independently of the remaining physics modules. Its CUDA
translation units include the outer fusion body, Newton, dimension-general Post
and OSC/controller scheduling, host launcher, sparse cache/reductions, and two fixed MathDx instantiations.
N is a runtime argument; N=1 and N=2 use the same library.
Editing a maintained numerical source rebuilds its own object and the
two final link steps; cached MathDx objects remain unchanged. Source edits
reuse the stable CMake build tree.

`flash_substep_osc_profile` produces the optional `substep_osc_profile.so`.
It uses OSC solver/pipeline objects with `FLASH_OSC_PROFILE=1`,
while reusing ordinary outer fusion, Newton, Post and MathDx objects. Its
`flash_libero_osc_profile_*` exports configure a bounded per-world uint64 buffer
and describe the record layout. The buffer starts with 8 header cells (count,
overflow, then reserved cells); each record has 12 metadata cells followed by
nanoseconds/count pairs for each reported phase. Metadata includes cache reuse
and the three Jacobi sweep/rotation counts. Gather/seed/solve/commit and the
outer total contain nested timings; do not add parent and child phases. On the
cooperative pipeline, Gather includes the concurrent seed interval and Seed has
no separate sample; the ordered fallback retains separate Gather/Seed samples.
Normal products compile all markers out. The profile product is excluded from
the ordinary catalogue build.

Whole-program device LTO is off by default. CMake resolves only the two fixed-size
MathDx units with the vendor LTO library using NVCC 13.2 or newer's
`--relocatable-link`. Each reusable object contains both relocatable device code
and its matching host fatbin registration, and is passed to both final linkers.
Newton/OSC algorithm logic and the final library use ordinary device linking.
`FLASH_GLOBAL_DEVICE_LTO=ON` is an explicit separate build configuration.
FP64, FMA, division, square-root, and FTZ options remain explicit.
The final host link rejects unresolved symbols, including missing CUDA fatbin
registration symbols.

Ordinary runtime construction is load-only. It checks the completed CMake product
and Ninja's dry-run result and rejects missing/stale products. Offline callers may
explicitly permit a CMake build. Build inputs are maintained project files; there
is no execution recorder, numbered-stage dispatch, pointer dispatch table, or
runtime CUDA source generator in this fusion entry.

The complete fused path currently targets SM80. The substep CMake build reads
`data/libero_task_parameters.json` and compiles
all 13 dimension families covering the 130 shipped tasks. `flash_substeps_all`
is one offline build; the NV39 development target remains `flash_substep` and
produces `substep.so`. Other dimensions use `flash_substep_nv21`, etc., producing
`substep_nv21.so`, etc., in the same build tree. Runtime selects from actual
model dimensions and verifies the library's exported dimensions before launch.
World count is a runtime argument, not a template instance. Models with the
same dimensions share numerical code while keeping their own model arrays.

`substep_parameters.cuh` defines the dimension template. `substep_shape.cuh`
selects an offline instance through compiler definitions. Shared vector pitches,
Hessian/MathDx offsets and global coefficient pitches are derived from those
dimensions; ctypes world pointers use the actual retained array extents. The
fusion binder still rejects unsupported optional features and unknown catalogue
shapes. Source/build coverage is distinct from numerical task qualification.

Newton layout metadata has its own valid bit, position generation and count
header in the existing global workspace. Successful position versions reuse
head/type/tile metadata across forwards; misses rebuild and publish only after
validation. Velocity-dependent state, residuals and search directions are not
cached across solves. The native FP64 GEMV policy uses 8-lane row groups for
NV<=32 and 16 lanes otherwise, with two candidate accumulators per J load.
Transpose GEMV partitions rows across warps for NV<=32 or at least 256 live
rows. These are initial tuning policies; the complete SDK implementations remain
available through the offline `FLASH_NEWTON_NATIVE_GEMV=0` definition.

Line-search Jv tiles immediately feed their quadratic-coefficient consumer;
straddling contact groups wait for their last row. Jv still has a global mirror
for the accepted-step residual update. Coefficients are reused across candidate
step lengths of one search direction, never across different directions.
Newton and its descendants remain in the independently compiled
`newton_solver.cu`; the substep still performs all three logical forwards.

The offline `flash_newton_gemv_test` executable checks all catalogue NV values,
two-world mixed row counts, empty/odd tails, dispatch boundaries and zero-force
NaN masking against long-double matrix products.
`flash_forward_mass_test` directly checks the packed smooth solver against an
independent long-double dense solve for all catalogue NV values, 8/16/32-lane
groups and large/interleaved-component fallback. Both are correctness tools.

### FP64 Newton reductions and position-generation cache

`FLASH_NEWTON_ATOMIC_REDUCTIONS` remains a CMake option, default `ON`. The
optimized product retains the global FP64 atomic Gram implementation as a
fallback and an explicitly selectable atomic `J^T force` candidate. Transpose
and Gram select their algorithms independently; atomic accumulation is not
forced when direct column ownership or sparse gathers do less work.
`FLASH_NEWTON_TRANSPOSE_METHOD` chooses `0` (automatic, the default), `1`
(global atomic), `2` (direct columns), `3/4/5` (32/16/8-lane row partitions), or
`6/7/8` (1/4/8-lane sparse column gathers). An unavailable sparse cache falls
back to the dense path. Automatic dense selection uses direct columns for
64-thread blocks and 16-lane partitions for blocks of at least 128 threads and
at least 128 live rows. Explicit sparse column gathers can use 1/4/8 lanes per column. Automatic
selection omits CSC construction for the three-forward workload, retaining
direct dense column ownership to include the preparation cost in the choice.

`newton_transpose.cu` and `newton_jacobian_cache.cu` are small, independently
compiled device objects shared by all catalogue shapes. The numerical entries
remain `__noinline__`; they allocate no shared arrays. Tiny load/store primitives
only specify global/shared address spaces across the separate compilation
boundary. `newton_atomic.cu` retains the earlier two-private-warp sparse Gram,
using the existing MathDx arena and a final global atomic merge. No shared FP64
atomic/CAS is required.

`FLASH_NEWTON_JACOBIAN_CACHE`, default `ON`, adds bounded per-world storage to the
top-level global device workspace. It retains exact-nonzero J/D*J row records,
two independently sorted row-parity segments per column, and optional grouped
row Gram products. The cache includes **all live constraint rows**. It does not
cache force, active states, assembled Hessians, factors, accelerations, or
warmstart results. CPU MuJoCo 2.3.7 produces J and D in `mj_fwdPosition`; constraint
updates change force/state while reusing those position outputs. The three
forwards before integration can therefore share this cache. Reset, integration,
layout/count changes, invalid entries, and forced prefix misses rebuild it using
the existing position generation and validated four-count layout contract.

The builder gives two warps disjoint row pools and column-list ranges. Column
counts and write cursors are private registers; neither column-list pass uses
atomics. For sparse input, the first row pass also precomputes `(D*J_i)*J_j`
contributions grouped by packed H cell. A borrowed MathDx arena holds native
32-bit shared integer slot counters and two compact row records; FP64 products
and their row ids go directly into global slots. It adds no shared allocation. Each cell can retain up to
128 row contributions. Every solve applies its **fresh quadratic-state mask**,
then its sole output owner sums these contributions: no matrix clear, private
shared matrix, row warp barrier, or atomic merge is needed on this path.
Precomputing a rounded product changes FP64 summation/contraction order; physical
endpoint error scale, rather than bit equality, is the correctness contract.
Dense/exceptional rows disable the optional product cache and retain the
general row Gram. Product-slot overflow also uses the cached row Gram; row-entry overflow or NV>64 uses the original Gram/dense transpose.
Exceptional rows retain dense nonfinite propagation; nonfinite forces select
dense transpose so implicit zeros retain their NaN/Inf behavior.

`newton_jacobian_layout.cuh` specifies row records, SoA value/index arrays,
column lists, and fixed product slots by byte offsets. Its row cache has at most
65,536 entries, split into two pools; product capacity is bounded by NV and 128
slots per packed cell. It never scales shared allocation with world count and
never allocates during graph replay. `newton_coefficient_layout.cuh` appends
this cache after the coefficient planes and aliased reduction scratch.
`host/runtime/step/newton_binding.py` allocates the complete world pitch once;
the loader rejects a stale library with a different pitch before launching.

For matched diagnostics, the optimized library exports
`flash_libero_newton_configure_reductions(cache_enabled, transpose_method)`.
Call it only with the owning device current and no graph/kernel in flight.
`(0,1)` restores the previous uncached atomic path; `(1,0)` selects the defaults.
Changing this policy does not compile, allocate, recapture, or alter the solver
ABI. Normal applications use the CMake defaults.

Ordered comparison products continue to coexist in the same build tree:

```bash
cmake --build "$FLASH_BUILD_ROOT/sm80-dev" --target flash_substep flash_substep_ordered --parallel 4
```

`env.prepare_substep_fusion(..., atomic_reductions=None)` follows the configured
CMake option. Passing `False` selects `substep_ordered.so`; passing `True`
requires the optimized product. Both paths load completed products only.
`flash_substeps_ordered_all` builds all ordered comparison shapes. Setting the
master CMake option `OFF` preserves the same host/device workspace contract.

The optional `flash_newton_jacobian_cache_test` executable calls the actual
separate numerical functions against independent long-double products. It
covers all 13 catalogue NV values plus NV65 fallback, mixed two-world counts,
64/128-thread blocks, fresh/hit/refreshed generations, changing force/state,
dense/sparse/nonfinite rows, row/product-budget fallbacks, and global/shared
bounds. `flash_newton_atomic_test` retains the original reduction tests.

### Newton phase profiling

The optional `flash_substep_newton_profile` CMake target produces
`substep_newton_profile.so` in the same build tree. It reuses MathDx objects
and compiles separate Newton and outer fusion objects with
`FLASH_NEWTON_PROFILE=1` and `FLASH_SUBSTEP_PROFILE=1`, respectively:

```bash
cmake --build "$FLASH_BUILD_ROOT/sm80-dev" --target flash_substep_newton_profile --parallel 4
```

Select it with `env.prepare_substep_fusion(..., profile_newton=True)`. Runtime
loading remains compile-free. `newton_profile.cuh` defines opt-in per-world,
per-solve device timing records, and the profiling library exports their layout
and buffer configuration. A null configured buffer disables recording during
warmup. Normal `substep.so` compiles the markers out entirely.

`substep_profile.cuh` additionally records the complete world-local substep:
velocity/reference/bias cache misses and hits, actuation, smooth acceleration,
Newton input preparation, Newton, Post, controller, checks and integration.
Its `flash_libero_substep_profile_*` exports use the same bounded per-world
buffer convention. `forward` and `total` are inclusive parents. Substep records
do not cover the multi-CTA position/collision prefix or host environment work;
use a CUDA graph node trace for those parts of a complete control step.
No profiling marker introduces a numerical barrier or changes shared storage.

The outer phase records are exclusive; line-search and Hessian subphase records
are inclusive subsets and must not be added to their parents. Device timer
durations include waiting at existing synchronization boundaries. Summing two
worlds' durations measures accumulated work time, not GPU wall time.

## Licensing and Sources

Project-original contributions use the project MIT license, subject to the component boundaries in the root [LICENSE](../../../LICENSE). Upstream code keeps its own notices and terms. [Third-Party Notices](THIRD_PARTY_NOTICES.md) records adapted code, external dependencies, and retained license texts.
