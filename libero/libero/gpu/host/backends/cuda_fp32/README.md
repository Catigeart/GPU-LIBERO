# Independent CUDA FP32

This backend owns FP32 CUDA buffers, streams, graphs, model preprocessing and
LIBERO batch lifecycle. Neither `warp` nor `mujoco_warp` is required to prepare
its modern CPU models or execute physics. Numerical leaves are shared with the
handwritten MJWarp migration path; readable world-local fusion explicitly
combines accepted phases in `cuda/fp32/`.
It is the default `GpuBatchEnv` backend when precision is omitted and
`LIBERO_GPU_PRECISION` is unset. Select `precision="fp64"` explicitly to use
the native FP64 backend in its separate environment.
CPU MuJoCo 3.16.0 remains a dependency for model compilation and OpenGL rendering.
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
python -m libero.gpu.host.backends.cuda_fp32.prepare \
  --build-root /path/to/common-project --tasks all \
  --mathdx-root /path/to/mathdx --jobs 2
```

To create a common catalogue from scratch, invoke the following in the separate
CPU MuJoCo 2.3.7 environment, with this checkout on `PYTHONPATH`. No Warp package
is needed in that environment either:

```sh
python -m pip install '.[cpu-catalog]'
python -m libero.gpu.host.backends.cuda_fp32.catalog \
  --build-root /path/to/new-common-project --tasks all --seed 0
```

The common project contains the CPU scene catalogue
`TASK_INSTANCES.json`, scene XML, fixed fixture parameters and initial states.
Those are CPU model inputs, not serialized Warp arrays. The existing catalogue
preparation uses CPU MuJoCo 2.3.7; run it in its separate CPU environment.
The command above imports the inputs using CPU MuJoCo 3.16.0 and builds only
the independent runtime/transport, 29 shared handwritten CUDA targets and
the world-local FP32 fusion target.
It advances zero physics substeps. Existing compatible CMake products are reused.
`--skip-build` prepares scenes only. Preparation and execution must be serialized
on shared GPU machines. Runtime construction defaults to `allow_compile=False`.

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
GpuBatchEnv -> CudaFP32BatchEnv                       environment.py
  build_scene -> prepare -> upload -> make_data      scene.py, model.py, data.py
  Runtime.enqueue_action                             runtime.py
    control Graph, 25 ordinary physical substeps
      Physics.step1                                  physics.py
        kinematics/COM/cameras/mass/collision/constraints/transmission
        velocity/bias -> passive/gravity/fluid
      Controller.enqueue                             controller.py, bridges.py
        gather -> shared CUDA OSC -> scatter
      Physics.step2                                  physics.py
        actuation -> smooth force -> mass solve
        Newton.solve -> device conditional WHILE
        postconstraint/sensors -> Euler -> warmstart publication
      GPU observable/scene sampling                  observations.py
  public boundary: predicates, dictionaries, GL      shared adapters
```

`device.py` defines contiguous allocations and views, not a numerical compiler.
Logical vector/matrix shapes omit fixed components; byte capacities include
them. Views retain their allocation owner. Uploads retain private host storage
until stream completion. Graphs retain launch owners. Allocation is forbidden
inside capture. Newton owns reusable device workspace; each solve explicitly
initializes the same state that upstream initializes on allocation. Device
conditionals use the current solve counter, without host iteration polling.

`cuda/runtime/runtime.cu` implements the small C ABI resource owner;
`cuda/runtime/bridges.cu` implements controller transport and observable sampling.
They have independent CMake targets `flash_cuda_runtime` and
`flash_cuda_bridges_fp32`. Physics remains in `cuda/{dynamics,newton,collision}/mjwarp*`
and OSC in its existing controller target. The `mjwarp` prefix records algorithm
provenance; those `.cu/.cuh` files do not depend on the Warp compiler or runtime.
Shared ctypes pointer bindings select storage metadata via `host/runtime/cuda_api.py`.

Readable world-local fusion lives in `cuda/fp32/`, independently compiled by
`cmake/FP32Fusion.cmake`. `fused_tail.cu::advance` combines velocity and position
advancement in one 64-thread CTA per world; its block barrier replaces the
original kernel-completion dependency. The same kernel copies Newton `qacc`
to warmstart and publishes time/overflow state. Warp 1 handles this auxiliary
work while warp 0 integrates joint poses. `fused_tail.py::FusedAdvance` borrows
validated Euler and publication bindings. Damping preparation and factor/solve
still precede advancement; observation sampling follows the completed kernel.
Local frozen-input qualification does not imply a new full-task numerical or
performance result.

`fused_observations.cu::pack_and_sample` combines the five low-dimensional field copies
in one 128-thread CTA per world. `DeviceFieldPacker` selects it for the default
five-field layout; scene and other layouts retain their existing copy path.
The target `flash_fp32_fused_observations` and the tail target have separate
CMake recipes and source identities, so editing one region does not recompile
the other. Standalone packing retains the original observation buffer and sampling epoch.
Default low-dimensional sampling directly gathers the current raw fields in the
same kernel. Independent clocks advance through a fixed 16-byte bitmap before
ordered dependent evaluation; plans above 128 specs use scalar clocks. The
shared adapter completion hook and camera publication remain outside this leaf.

The pure CPU helpers `../mjwarp/scene_cache.py`, `scene_inputs.py` and the model
compatibility helper in `facade.py` remain shared. Their location does not make
them external Warp dependencies. Source/asset/model hashes still admit every
scene; each CUDA loader verifies the corresponding source identity and ABI.

## Supported scope and validation

The admitted shipped task domain is rigid, dense Newton, elliptic contacts,
Euler integration, the default Panda OSC, FORCE/TORQUE sensors and FIXED
cameras/lights. Flex, tendons, equalities, activation/history, sleep, noslip and
plugins are rejected. This is the LIBERO batch implementation, not a general
replacement for every MuJoCo feature. FP64/native and the explicit upstream
MJWarp reference remain separate backends and environments.

Stable independent validation commands in `tools/validate_gpu.py` are:

- `cuda-runtime`: buffer/stream/Graph/conditional-loop ownership and replay.
- `cuda-model`: all model arrays/scalars/layouts against CPU-exported pinned data.
- `cuda-functions`: common-input prefix, OSC, Newton and Euler comparisons.
- `cuda-lifecycle`: partial reset, continuation, observation mutation and callbacks;
  `--cameras --prepare` additionally checks explicit CPU scene preparation and RGB/depth.
- `cuda-complete`: source-bound N=2 trajectories, optional single timing pass.

Reference export is explicit and uses the separate reference environment.
Independent checks prohibit Warp imports; an isolated installation additionally
establishes that neither package can be found. Existing strict checks, accepted
FP32 trajectory envelopes and geometry diagnostics retain separate outcomes.
Shared-device timing is an observation, not an isolated performance guarantee.

### Fused rigid kinematics

`Physics.position -> KinematicFrames.enqueue_fused -> branch_kinematics<4>` joins
the five kinematics leaves in `cuda/dynamics/mjwarp_kinematics.cu`. Each CTA groups
four independent branch warps from one world. One lane computes a branch's
private pose prefix; after a warp barrier its lanes build canonical body and
attached frames cooperatively. This preserves holds and broadcast/live physical
parameters without shared numerical storage or a world-wide barrier. The cold
CSR groups immutable attachment indices by canonical branch owner. Component
leaves remain callable; intermediate trial fusions are retained in Git history,
not the runtime. Evidence is region-only N=2 endpoints and local graph timings.

### Fused centered frames and mass

`Physics.position` calls `FusedCenteredMass` from `fused_position.py`. The
independent `flash_fp32_fused_position` target compiles
`cuda/fp32/fused_position.cu::centered_mass`: centers -> inertia/motion axes ->
composite inertia -> CSR mass assembly. Validated small preorder trees use
bounded direct subtree sums to remove depth barriers; other domains keep staged
child gathering. Shared storage is capped at 2560 bytes and reused by lifetime,
with a global-storage fallback for larger trees. Immutable intervals share the
existing topology owner; no physical values persist as a cache. The original
CoM/mass component entries remain callable.

FIXED camera/light frames use `CamLightFrames.enqueue_fused` and the readable
`mjwarp_camlight.cu::fixed_frames` queue. Small domains use one warp per world;
large domains split independent attachments across world-local tiles. The
component camera/light phases remain available. No shared storage or barrier
is introduced, and live broadcast parameters retain their existing semantics.

The default `KinematicFrames.enqueue_frames` additionally joins admitted FIXED
camera/light jobs to the canonical kinematic branch warp. The readable
`branch_kinematics<4,true>` body groups all attachment transforms behind its
existing pose barrier. Heavy camera/light branches retain the separate tiled
path; cold immutable indices select the path. This adds no shared storage or
barrier, and the original independent entries remain callable. The following
CoM/mass kernel retains its own completion boundary.

### Fused spatial velocity

`ComVelocity.enqueue_fused` in `host/runtime/dynamics/mjwarp_velocity.py` selects
`velocity_components<8/32>` in `cuda/dynamics/mjwarp_velocity.cu`. Six lanes carry
the spatial vector and two lanes complete the shuffle group. Cold topology
selects a branch warp for deeper small branch sets or four groups per warp for
broader shallow trees. Root zero publication shares this kernel; no branch
reads it, so there is no new barrier. Live qvel/cdof and canonical publication
remain unchanged. The original root/branch component phases remain available.

### Fused ordinary bias RNE

`Bias.prepare_fusion` shares the admitted immutable tree owner;
`Physics.velocity_stage -> Bias.enqueue_fused -> mjwarp_bias.cu::world_bias`
joins acceleration prefixes, body forces, subtree gathering and bias projection.
Bounded small preorder trees use direct sums. Deep small branch sets produce
and consume acceleration rows within canonical warps, then synchronize the world
only for subtree forces and projection. The fixed 1536-byte shared row is reused
as acceleration then raw force; public outputs remain current. Trees outside
the direct interval domain retain staged gathering, including cooperative broad-parent sums and a
global-storage fallback. Component phases remain callable. Qualification is
local N=2 endpoints and N=128 layout comparisons, without a full-task claim.

### Fused spatial velocity and bias

The default velocity stage calls `FusedMotion` in `fused_motion.py`, loading the
independent `flash_fp32_fused_motion` product. Its continuous CUDA function
`cuda/fp32/fused_motion.cu::motion_bias` computes velocity, derivatives and bias
acceleration in canonical branch warps, then body forces, subtree forces and
DOF bias. The 3072-byte shared allocation has two reused lifetimes: acceleration
becomes raw force; velocity becomes composite force. No physical cache persists
between calls. Cold interval/owner/pointer admission keeps larger or unsupported
trees on the accepted component kernels.

The call chain is transmission velocity -> fused velocity/bias -> passive,
gravity and fluid forces. The fluid workspace is private, and those later stages
write distinct qfrc outputs, preserving public state. Selected configuration is
512 threads, 40 registers and three block barriers; original component APIs
remain available. Tests and timing cover only this region at N=2 and N=128.
