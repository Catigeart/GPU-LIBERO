# Runtime paths and source ownership

This index describes maintained entries and retained alternatives. Version
suffixes do not imply obsolescence. Cold assembly uses imports, callbacks and
AST-derived templates: an import search alone cannot establish unused code.
`tools/runtime_inventory.py --output <external-file.json>` lists every GPU Python
module, its functions/classes, imports, static reachability and source composition
without executing the GPU runtime.

## Layout

| Directory | Responsibility |
| --- | --- |
| `model/` | Native intake, static topology, observation descriptions and collision model extraction. |
| `host/build/` | Offline builds, source/artifact admission, child environments and AOT ready sessions. |
| `host/backends/cuda_fp32/` | Default independent FP32 model preprocessing, CUDA resources, Graphs, physics scheduling and public batch lifecycle. No Warp dependency. |
| `host/backends/mjwarp/` | Explicit upstream reference, shared CPU model helpers and numerical migration selectors. |
| `host/assembly/` | Cold composition and explicit replacement before live binding. |
| `host/adapters/` | Environment facade, same-model batching, observations and rendering boundaries. |
| `host/bindings/` | Handwritten CUDA ABI/layout adaptation and launch plans. |
| `host/runtime/` | Retained owners, scheduling, recovery and configured algorithms. |
| `cuda/` | Maintained handwritten device implementations and separate numerical compilation units. |
| `reference/` | Retained Warp/reference implementations consumed by assembly, comparisons or fallbacks. Not an archive. |

Product GPU code does not import `scripts/`. That directory retains original
LIBERO user commands. Historical GPU experiment drivers live in the external
workspace archive, together with migration checksums. Results, generated fixtures,
profiles and build caches also belong outside the source checkout.

## Public entry and assembly

| Entry | Selection / scope | Implementation |
| --- | --- | --- |
| `libero.libero.envs.GpuBatchEnv` | Worlds of one task and one exact static model; defaults to independent CUDA FP32. Explicit FP64 selects native. Equal dimensions do not permit mixing models. | `host/adapters/environment/batch_env.py::GpuBatchEnv.__new__` → `host/backends/selection.py::resolve_engine` |
| Independent FP32 owner | Preprocess/upload one CPU model, allocate world-major device state, enforce capacities and own reset/observation/Graph lifetime. | `host/backends/cuda_fp32/environment.py::CudaFP32BatchEnv` |
| Original environment / vector APIs | Preserve upstream signatures, routing and worker lifecycle. | Parent `envs/env_wrapper.py`, `envs/venv.py` |
| Batch physical arena | Stack reset states and construct one shared model with world-major dynamic owners. | `host/adapters/environment/batched_physics.py::prepare_native_batch` |
| CUDA composition | Batch factory selects optimized fast dataflow with cold source/owner/capacity admission. | `host/factory.py::prepare_cuda_physics` → `host/assembly/components.py::prepare_cuda_physics` → `host/assembly/mathdx/factory.py` and pipeline/compose templates |
| Facade components | Typed cleanup, final events and configured recovery; `r02` remains active. | `host/factory.py::create` → `host/runtime/step/recovery_cleanup_factory_r02.py::create` |
| Collision model intake | Extract live MuJoCo arrays without private test XML or CPU contact outputs. | `model/collision_static.py::static_from_model` |
| Capsule extension | Cold source closure and route extension before binding/launch, with independent source identity. | `host/runtime/collision/capsule_extension.py::prepare_workspace_extension` |

## Environment ownership and offline products

See [Environment Lifecycle](ENVIRONMENT_LIFECYCLE.md) for runtime filesystem
behavior, cleanup guarantees and the differences from CPU LIBERO reset semantics.

| Responsibility | Implementation |
| --- | --- |
| Read-only FP32 scene inputs | `host/backends/cuda_fp32/scene.py::_scene_files` |
| FP32 Graph, renderer and stream release | `host/backends/cuda_fp32/environment.py::CudaFP32BatchEnv.close` |
| Native compatibility preparation and load-only scope | `host/build/compatibility.py`; `host/adapters/environment/product_factory.py::compatibility_scope` |
| Signed stateless native code and per-caller parameters | `host/build/native.py`; `host/build/product_cache.py` |
| Temporary native loader ownership and retry | `host/runtime/core/build_lifetime.py::BuildLifetime` |
| In-memory native model intake and replacement | `host/runtime/core/model_bytes.py`; `host/runtime/core/native_model_update.py` |
| Private library release after borrowed functions expire | `host/build/library_lifetime.py` |
| Retired storage maintenance, outside environment execution | `host/build/legacy_storage.py` |

## Control-step execution

The default FP32 path is:

```text
GpuBatchEnv.step(actions)
  -> CudaFP32BatchEnv / shared public batch boundary
  -> host/backends/cuda_fp32/runtime.py::Runtime.enqueue_action
     -> action upload and control Graph
        -> 25 ordinary physical substeps
           -> physics.py::Physics.step1 (position, collision, velocity, forces)
           -> controller.py::Controller.enqueue (gather -> shared CUDA OSC -> scatter)
           -> physics.py::Physics.step2
              -> actuation / acceleration / physics.py::Newton.solve
              -> postconstraint / sensors / Euler / warmstart publication
           -> GPU observation sampling at each physical-substep boundary
  -> host observation packaging, task predicates and optional rendering
```

`cuda/runtime/runtime.cu` owns CUDA buffers/streams/Graphs through its C ABI;
`cuda/runtime/bridges.cu` provides controller transport and observations. Shared
handwritten physics is in `cuda/{collision,dynamics,newton}/mjwarp*`; those files
require neither Warp headers nor the Warp compiler. The conditional Newton loop
runs on the device. See the [FP32 guide](host/backends/cuda_fp32/README.md) and
[intentional numerical corrections](host/backends/mjwarp/NUMERICAL_DEVIATIONS.md).

The following retained path and optimization details apply to native FP64 and
the explicit legacy native FP32 comparison engine:

```text
GpuBatchEnv.step(actions)
  -> NativeBatchedPhysics.enqueue_action
     -> stage action
     -> control graph, when observer/debug policy permits
        -> 25 physical substeps for the standard 20 Hz / 2 ms schedule
           -> position / collision / constraints prefix when needed
           -> fused_substep_kernel IF prepare_substep_fusion was selected
              -> forward_dynamics three times -> solve_newton each time
              -> controller, physical checks/reset continuation, Euler
     -> component substep loop when graph capture is ineligible
  -> typed field downloads -> task predicates and observation packaging
```

The ordinary component graph and explicitly selected single-substep fused graph
are distinct. Constructing a batch does not select fusion;
`GpuBatchEnv.prepare_substep_fusion(...)` does. The retired 25-substep kernel is
not part of this project. The outer graph owns the substep sequence.

| Path | Trigger / contract | Source |
| --- | --- | --- |
| Component graph | Non-debug graph-eligible observer, no explicit substep fusion. | `NativeBatchedPhysics.prepare_control_graph` |
| Single-substep CUDA graph | Explicit fusion API; only completed CMake products are loaded during normal execution. | `host/runtime/step/fused_substep.py::FusedSubstep`; `cuda/step/fused_substep.cu::fused_substep_kernel` |
| Regional FP64 staged graph | Explicit `prepare_substep_fusion(..., staged=True)`; optional branch motion plus native preparation/Newton/Post. Newton and OSC retain 64-thread SDK objects. | `host/runtime/step/staged_substep.py::StagedSubstep`; `cuda/step/staged_{motion,forward,newton,post}.cu` |
| Host/debug loop | Debug or observer needing host callbacks; captured borrows are discarded. | `NativeBatchedPhysics.enqueue_action`; `host/runtime/step/schedule.py` |
| Position refresh | State/generation invalidation or suspended prefix; rebuild before resuming. | `FusedSubstep.enqueue_substep`; `cuda/runtime/position_reuse.cuh` |
| Velocity/reference/bias reuse | Same physical substep, state epoch and position generation. New substep or prefix reentry invalidates readiness. | `cuda/step/fused_substep.cu::forward_dynamics` |
| Smooth reuse | Successful first→second forward, qualified owners and no writer; third forward computes again. | `forward_dynamics`; `host/runtime/step/substep_binding.py::_validate_smooth_reuse` |
| Deferred Post | Exact built-in graph-eligible observer and `defer_post=True`; policy samples and last-substep pre-Euler output remain. Unknown/subclass observers retain the complete publication path. | `NativeBatchedPhysics.prepare_fused_substep_graph`; `forward_dynamics` |
| Newton | Each admitted forward, warmstart and final publication; separate compilation unit. | `cuda/step/newton_solver.cu::solve_newton` |
| Physical reset | Checks, continuation, root restoration and full-forward publication. | `host/runtime/step/recovery_final_event_factory.py`; `recovery_cleanup_environment_r02.py`; `cuda/runtime/recovery.cuh` |

Normal qualified native fused execution performs 75 Newton solves, 50 factored mass
solves and 3 complete Post evaluations per control step. Recovery and alternative
observer/debug policies can do additional work.

### FP64 motion replacement

The opt-in `prepare_substep_fusion(..., motion_passive=True)` path replaces the
native fused body's cold motion/bias/passive mathematics with a standalone
`cuda/step/fp64_motion_passive.cu::fp64_motion_passive_kernel`. The remaining
body is `fused_substep.cu::motion_residual_kernel`, compiled from the maintained
substep body with `FLASH_FP64_MOTION_PASSIVE`. It retains all three forward call
sites, Newton/warmstart semantics, control, Post, checks and Euler integration.
This path does not use the A/B/C decomposition below.

`prepare_substep_fusion(..., actuation_solve=True, force_threads=256)` also
selects the entry force producer in `cuda/step/fp64_actuation_solve.cu`.
The ordinary chain becomes position prefix -> motion/passive ->
actuation/smooth/mass solve -> `fused_substep.cu::force_residual_kernel`.
Actuators and DOFs own independent calculations, external wrenches use a
world-local activity mask, and the existing CPU2.3.7 packed LDL factors remain
the mass owner. Affine terms with zero velocity coefficient need no private
duplicate transmission dot; original transmission admission still runs.

The first forward admits the private force outputs at the original point.
The next forward may reuse the successful producer while ctrl is unchanged;
the existing public smooth cache retains precedence and factor-reuse counts.
After the controller changes ctrl and position/velocity checks complete,
`force_residual_kernel` calls `prepare_fp64_force` from that same CUDA source.
The expired OSC shared arena holds its scalar forces, smooth RHS and acceleration;
the next forward consumes those cells directly on a valid prefix reuse. This
adds no global launch. A cold preparation can instead read the complete private
global work view. Handoff states 0/1/2 mean fallback/global/resident, and the
loader checks version 2 of this handoff contract. Every Newton call and warmstart
update remains. BADCTRL and other
exceptions use the native implementation with its per-invocation warnings.
`force_threads` accepts 64/128/256 or 0 to disable force production; disabling
motion also disables this dependent producer. Unsupported extents and unusual
output aliases retain native calculation. The additional handoff is four
bytes/world, with no persistent force mirror.

`host/runtime/step/peeled_force.py::PeeledForceSubstep` owns this load-only
selection. `flash_substep_force[_nvN]` builds individual residual families;
`flash_force_substeps_all` builds the complete catalogue in the existing tree.
Both producers are dimension-general objects shared across NV families.

The new producer follows `cuda/fp32/fused_motion_passive.cu`: one world block,
one warp per branch (eight spatial lanes), independent body force production,
contiguous-subtree bias reduction and DOF projection. Two fixed 384-double
shared arenas overlay velocity and composite force; small bodies also fit the
fluid wrench arena. Larger admitted bodies borrow the existing device fluid
workspace. `motion_threads` may be 128, 256 or 512; 0 exercises the retained
native producer in the residual without an extra motion launch.

Native CPU2.3.7 spring/damping/fluid mathematics and input owners remain in use.
The new kernel computes the complete private bias before constraint reference;
the residual admits and commits it at the original reference-dependent point.
One private ready integer per world qualifies only the immediately adjacent
producer/residual pair. It does not authorize full forward reuse. Later forwards
retain the state/position-generation cache and epoch publication. Position
continuations run a fresh producer; acceleration-reset forward remains with
its existing owner. Unsupported topology, diagnostics and exceptional inputs
retain component calculation and error behavior.

`host/runtime/step/peeled_motion.py::PeeledMotionSubstep` owns the descriptor,
ready flags and load-only launch. The descriptor checks contiguous subtrees,
64 bodies and 32 branches, matching the source algorithm's admitted domain;
other topologies select the native fallback. `cmake/PeeledMotion.cmake` builds
one dimension-general producer and one residual per catalogue NV, reusing the
existing separately compiled Newton, Post, controller and vendor objects.
The offline aggregate target is `flash_motion_substeps_all`; build it in the
existing CMake tree to prepare every catalogue NV together with the canonical
native products. Loading a partial development tree requires that selected NV's
motion and canonical products are already current. Runtime never builds them.
Acceptance and timing remain specific to the measured inputs and GPU load.

### Regional FP64 decomposition

The optional staged entry retains the native three-forward schedule. Within
each forward it launches motion/passive production, preparation, complete
Newton, then Post/publication with the adjacent controller/check/Euler tail.
An ordinary substep has ten region nodes, or nine on the component fallback.
Prefix refresh and acceleration-reset continuations retain the
existing conditional Graph callbacks, including mixed-world preservation.
The outer Graph still sequences the physical substeps; this is not a fused
25-substep kernel.

`staged_motion.cu::staged_motion_kernel` owns velocity and passive production.
Eight lanes walk a branch in registers, maintaining parent spatial velocity
and bias acceleration without a block barrier at every tree level. Shared
prefixes may be recomputed; one canonical branch publishes each body. The
immutable host descriptor contains paths and owners, not cached physics.
The normal path has no dynamically sized shared spatial arrays. A topology
whose duplicated paths exceed four times its non-world body count retains the
original level traversal. Exceptional inputs retain the native producer's
stage-specific status and clearing behavior.

`staged_forward.cu::staged_prepare_kernel` owns constraint reference, reverse
bias accumulation/publication, actuation, smooth acceleration and the Newton
input bridge. On the component fallback it also produces velocity/passive.
Only the first phase of each entry launches a separate motion producer. Later
forward reuse/publication stays in preparation; a prefix continuation admits
a new first producer. An unexpected later miss retains the original producer
inside preparation, so this scheduling choice is not a cache-validity assumption.
Velocity and bias share the original cache-validity publication after reference
admission; computing the private RNE prefix does not publish a complete bias.
`staged_newton.cu::staged_newton_kernel` calls the unchanged separately compiled
`newton_solver.cu::solve_newton`. Its Hessian, factors, reductions, line search
and warmstart stay within that kernel and keep the CPU2.3.7 mathematics.
`staged_post.cu::staged_post_kernel` publishes the forward before executing its
adjacent tail. There is no device-wide synchronization inside these kernels;
stream/Graph dependencies order the regions.

The original fused entry and staged entry share maintained numerical bodies
in `substep_forward_math.cuh`, `substep_euler_math.cuh` and the
`substep_forward_{prepare,newton,post}.cuh` helpers. `substep_tail.cuh` owns
their common controller/check/reset/integration logic. None includes another
kernel `.cu`, generates CUDA text at runtime or duplicates the numerical math.

Each region requests its own dynamic shared-memory budget. Preparation uses
`SmoothLayout`, Newton uses `NewtonLayout`, and reset/integration uses
`EulerLayout`. Post has a compact qacc vector followed by three spatial body
arrays and six sensor outputs. Once Post completes, the controller and Euler
reuse that allocation. `staged_workspace.cuh` defines this lifetime; it does
not reserve a Hessian for Post or apply a task-specific body limit.
The motion launch uses 128 or 256 threads based on branch count; preparation,
Newton and Post retain 64 threads because their vendor objects have that ABI.

The private handoff uses three integers and NV FP64 values per world, allocated
before capture and retained by `StagedSubstep`. Existing smooth force and
acceleration owners bridge preparation to Newton. The private Newton qacc
view is explicitly carried to Post because rejected/held solves need not
publish canonical qacc. No Hessian or iterative solver state is spilled.
The existing substep/Newton argument ABIs remain unchanged.

All catalogue shapes have explicit CMake targets. The optional products combine
one shared, runtime-dimensioned motion object with shape-specific
preparation/Newton/Post objects. No per-task motion specialization is generated.
Build them offline in the existing build tree, together with their canonical fused binding
and selector products:

```bash
python -m libero.gpu.host.build.substep_native --precision fp64 --staged \
  --build-root /path/to/project/cmake-control --mathdx-root /path/to/mathdx \
  --arch 80 --fast-compile
```

Then select `env.prepare_substep_fusion(build_root, fast_compile=True,
staged=True, allow_compile=False)`. It requires native FP64, 64-thread SDK
objects and the three-solve schedule. FP32, single-solve and fused profiling
requests are rejected. The default fused entry is still available with
`staged=False`; this option alone does not claim a performance improvement.
The kernel-catalogue build does not qualify scene inputs; prepare native scenes
through the normal `libero.gpu.build` entry as usual.

## Retained alternatives

| Family | Selection and purpose | Decision |
| --- | --- | --- |
| `generic_prepare`, `shape_prepare`, `generic_*`, `shape_*` | Generic/shape templates used directly or by cold recovery/assembly AST composition. | Retain; direct call searches cannot establish that they are unused. |
| `recovery_*factory*`, `recovery_*environment*`, `environment_cleanup_r02` | Recovery inventories, final-event and cleanup providers; entry names also appear in admission records. | Retain names and guards rather than renaming the family wholesale. |
| `quality_*`, `jar_recovery_*`, `s04_jar_specialization_r01` | Explicit quality-recording and jar/domain paths, with diagnostic/admitted distinctions. | Retain; availability is not universal qualification. |
| `raw_copy_factory`, `copy_factory`, `prepare_copy_opt`, `prepare_copy_tape_opt` | Explicit `P06_STEP_COLD_COPY=cn` choice or retained fixed preparation. | Retain consumed constructors. Historical private-fixture `fixed_profile_factory` drivers moved to the workspace archive. |
| `host/runtime/production/*`, `host/adapters/environment/production_factory_*` | Domain admission, delegation, lifecycle and publication checks. | Retain owner/selection records and distinct diagnostic paths. |
| `reference/*`, original collision/Post variants | Mathematical comparisons and configured fallbacks. | Retain until complete owner/configuration/call analysis establishes otherwise. |

[ALGORITHM_NOTES.md](ALGORITHM_NOTES.md) lists selectors and defaults. CMake
options select compiled products; runtime algorithm and diagnostic profiles do
not implicitly change that configuration. [README.md](README.md) provides ABI
and numerical details.

## Build and validation

The checkout root and GPU package's `CMakeLists.txt` both include the maintained
`cmake/Substep.cmake`. Newton, Post, OSC, controller pipeline and MathDx keep
separate objects in a stable external build directory. Moving Python helpers
does not require recompiling unchanged CUDA objects.

AOT readiness also authenticates Python source identity. Import refactors can
invalidate readiness despite unchanged CUDA bytes. Use the normal offline build
and qualification entry to refresh admission; never rewrite an old receipt to
claim that new sources were qualified. Source identity and compiler work are
separate concerns.

Use the [stable validation tools](../../../tools/validation/README.md) for
same-input endpoints and frozen-old budgets. Source, import and packaging changes
use CPU integration checks rather than an automatic full physics sweep.

### Independent FP32 world-local fusions

The independent backend retains the component graph. Readable CUDA translation
units in `cuda/fp32/` fuse only world-owned producers and consumers. Their
CMake recipes live in [cmake/fp32/](cmake/fp32/), included by
[FP32Fusion.cmake](cmake/FP32Fusion.cmake). Each remains an independent target;
its loader verifies source, ABI and GPU
architecture before capture; normal execution never compiles them.

`Physics.integrate` uses `FusedAdvance` in `host/backends/cuda_fp32/fused_tail.py`
to join velocity/position advancement, warmstart copying and publication in
`cuda/fp32/fused_tail.cu`. Damping preparation and the mass solve remain upstream.
The copied warmstart is Newton's acceleration, not the damping-corrected Euler
acceleration. The outer graph still organizes 25 physical substeps.

`ObservationAdapter._sample_lowdim` uses `FusedRawPack.sample` in
`host/backends/cuda_fp32/fused_observations.py`. Its CUDA body in
`cuda/fp32/fused_observations.cu` gathers the current five raw fields directly,
advances independent sampling clocks cooperatively, then evaluates dependent
observations in plan order. The clock bitmap occupies a fixed 16 bytes per CTA;
plans above 128 items retain scalar clocks. Standalone `DeviceFieldPacker`
packing retains its five-copy fusion; other layouts keep the original copies.
Masks, disabled observations, events, forced sampling, camera publication and
post-sample host hooks retain their existing contracts. Pose epochs are unchanged:
post-integration qpos/qvel coexist with the previously computed body/site frames.
These changes have local endpoint and region-timing evidence; they do not add a
claim about complete-task throughput to the published benchmark table.

`Physics.position` now calls `KinematicFrames.enqueue_fused` in
`host/runtime/dynamics/mjwarp_kinematics.py`, loading the existing independent
kinematics target. `cuda/dynamics/mjwarp_kinematics.cu::branch_kinematics<4>` joins
body poses, body matrices, inertial frames, geom transforms and site transforms
in one readable CUDA body. Four branch warps share a CTA within one world.
One producer lane traverses each complete private prefix; canonical ownership
makes its downstream frame jobs disjoint from other warps. A warp barrier orders
those consumers, with zero shared numerical storage and no world-wide barrier.
The cold attachment CSR contains only immutable canonical-branch indices, never
physical values. Static geom and world-pose holds remain exact. The original
five component leaves remain available; superseded intermediate fused bodies
were removed rather than kept as runtime choices.

`Physics.position -> CenteredFrames.enqueue_fused -> centered_frames<64>` in
`cuda/dynamics/mjwarp_com_pos.cu` joins weighted subtree construction, reverse
child gathering, normalization, centered inertia and motion axes. A parent has
one writer; layer barriers order child reads, replacing separate depth launches
and sibling atomic additions. Division remains parallel across bodies. The
common model path uses 768 bytes of shared storage; larger trees use global
storage without a larger shared allocation. All public subtree/inertia/axis
outputs, including the accepted MJWarp zero-mass and world-body conventions,
remain current. `MotionTopology.prepare_layer_tree` owns reusable immutable
layer/child indices through `ResidentTree`; CoM borrows that owner, so downstream
world-local tree fusions can reuse the indices without a physical-value cache.

`Physics.position -> MassMatrix.enqueue_fused -> world_mass<128>` in
`cuda/dynamics/mjwarp_mass_matrix.cu` joins composite inertia construction and CSR
mass assembly. Independent tensor components gather children at each completed
depth; every stored matrix element is then assigned once by its row owner, so
there is no matrix-clear pass. World-body composite inertia retains its input.
The common path uses a fixed 2560-byte shared tensor buffer; larger trees use
global storage. `ResidentTree` is shared with CoM. Root-only admitted matrices
have an explicit initialization barrier. The source retains the original
component leaves. Flat-entry product staging was correct, but matched-width
controls did not establish a general gain; its extra metadata and barriers were
removed from the maintained path.

The independent default now joins those two accepted stages through
`Physics.position -> FusedCenteredMass.enqueue -> centered_mass<128>` in
`cuda/fp32/fused_position.cu`. Its continuous CUDA body constructs weighted
centers, centered tensors and motion axes, composite tensors, and CSR mass rows.
A cold topology proof admits direct interval sums only for preorder trees with
at most 64 bodies and at most 1024 total descendant visits. All 130 shipped task
trees satisfy this domain (maximum 38 bodies and 211 visits). This trades a small,
bounded amount of redundant arithmetic for fewer dependent tree barriers.
Non-preorder, deeper or larger trees retain staged child gathering. The same
fixed 2560-byte shared buffer holds weighted centers and composite tensors in
disjoint lifetimes; larger trees use global storage. Only immutable subtree
lengths are retained on device; every invocation reads current physical inputs.
Camera/light frames depend on body poses only in the admitted FIXED domain and
follow the combined mass stage. The separately callable centered/mass kernels
remain available for component users and comparison. This qualification covers
local endpoints and regional timing, without a new full-task throughput claim.

`CamLightFrames.enqueue_fused -> fixed_frames` in
`cuda/dynamics/mjwarp_camlight.cu` joins FIXED camera/light transforms in one
independent attachment queue. Small counts use a world-owned warp. Larger
counts use multiple world-local CTA tiles so the queue does not serialize a
large attachment domain. The body/quaternion inputs remain live, zero direction
and homogeneous quaternion behavior retain the accepted FP32 convention, and
both original component phases remain callable. There is no shared storage or
barrier in this leaf.

For the default independent position stage,
`KinematicFrames.prepare_frames` groups geom, site, FIXED camera and light jobs
by their canonical branch owner. `Physics.position -> enqueue_frames ->
branch_kinematics<4,true>` appends their actual transform code after the existing
pose publication barrier and shares one attachment transform body. No new
barrier or shared numerical storage is needed; register usage remains 56.
Each job reads a pose published by its own warp, including the held world pose.
If any branch owns more than 32 camera/light jobs, cold selection retains the
separate kinematics and tiled fixed-frame kernels to preserve attachment
parallelism. Both paths publish the same public frame arrays and retain the
same live/broadcast and static-geom hold contracts. This does not join the
motion-prefix kernel to the world-wide mass stage or cross a pooled collision
boundary.

The standalone `ComVelocity.enqueue_fused -> velocity_components` entry
uses a readable six-component spatial prefix in
`cuda/dynamics/mjwarp_velocity.cu`. Eight active lanes carry six components and
two shuffle participants. Canonical ownership still publishes every body/DOF
once; FREE translations precede rotations, and all BALL/FREE rotational axes
consume the same pre-rotation velocity. Root zero publication shares the kernel
without a barrier because private prefixes never read root output. Deep trees
with at most 32 branches use one branch per warp; broader/shallow domains pack
four independent branches into a warp. Both layouts tile only within a world
and allocate no shared storage. Existing overlap admission justifies restricted
input/output aliases, allowing read-only loads without a physical-value cache.
The original two component phases remain callable. Qualification covers local
N=2 endpoints plus an N=128 regional layout comparison, not task trajectories.

The standalone `Bias.enqueue_fused -> world_bias` entry in
`cuda/dynamics/mjwarp_bias.cu` joins ordinary bias RNE: gravity/private
acceleration prefixes, spatial body forces, subtree force gathering and DOF
projection. Small admitted preorder trees use bounded direct subtree sums.
Deep small branch sets use 512 threads: a warp produces six acceleration
coordinates, synchronizes locally, then distributes its canonical bodies'
force jobs over all 32 lanes. Shared rows hold acceleration until the unique
body-thread consumer has finished every read, then hold raw force. This removes
one block barrier while retaining public cacc/cfrc_int/qfrc_bias. Deferred jobs
retain body parallelism; computing all force algebra serially at each prefix
node is avoided. The selected path uses two block barriers, fixed 1536-byte
shared storage, 40 registers and no stack/local allocation. Other models keep
the 128-thread body queue; ineligible interval trees retain staged child gathering. A broad
singleton parent uses warp-cooperative child summation. Larger trees use global
storage with no shared allocation. World-force publication and disabled-gravity
semantics retain the accepted FP32 behavior. Existing component entries remain
available. N=2 regional endpoints include gravity-off and large/deep fallbacks;
N=128 local comparisons additionally qualify the branch-warp organization.

The independent default combines these accepted velocity/bias leaves through
`Physics.velocity_stage -> FusedMotion.enqueue -> motion_bias` in
`cuda/fp32/fused_motion.cu`, built by the independent
[FP32Motion.cmake](cmake/fp32/FP32Motion.cmake) recipe.
Canonical branch warps compute spatial velocity, every public motion derivative
and bias acceleration together, retaining each DOF's accumulation and the
BALL/FREE pre-rotation convention. Private noncanonical prefixes recompute
needed derivatives without reading another warp's output. Bulk body-force jobs
follow one block barrier. A fixed 3072-byte arena holds acceleration then raw
force in one half, velocity then composite force in the other. Public fields
remain current; DOF projection reads the reused shared composite arena.

Cold common-owner/pointer checks and the validated interval domain admit at
most 64 bodies and 32 branches; other inputs retain the accepted component
kernels. The selected kernel uses 512 threads, 40 registers, no stack/local
storage and three block barriers. Reusing per-axis input registers improves
local work; attempts to remove another block barrier, defer root publication
or use 256 threads did not establish a better path and were removed.

Bias now precedes passive forces inside the independent velocity stage. The
actual fluid binding allocates a private body-force workspace, and passive/
gravity outputs are separate qfrc fields. Fluid still consumes current cvel
after its producer; these stages do not consume bias outputs or overwrite its
inputs/outputs. No application
callback is inserted between these internal operations. Existing phase APIs
remain callable. Evidence is regional N=2/N=128 endpoints and paired local
timing under other GPU work, without a new complete-task result.

`Controller.enqueue -> flash_mjwarp_controller_publish -> control<true>` in
`cuda/control/mjwarp_controller.cu` appends the actual nine-actuator scatter to
its existing OSC kernel. The maintained separately compiled OSC mathematics is
unchanged. Normal torque publication already ends at its existing block
barrier; inactive/error exits retain ctrl without a new write, so the scatter
needs no additional barrier. Live actuator IDs, permutation, controller ctrl
buffers and final Data.ctrl retain their contracts. Original launch/scatter
entries remain available. Local OSC-plus-publication timing is neutral under
shared GPU load; an address-prefetch trial did not establish a useful gain and
was removed. No complete-task result is implied.
