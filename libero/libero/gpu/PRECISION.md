# Floating-point products

FP64 remains the default precision and uses native CUDA with CPU MuJoCo 2.3.7
as its semantic reference. Automatic FP32 selection uses the
[upstream MJWarp backend](host/backends/mjwarp/README.md), with one official
`step1 -> GPU OSC -> step2` sequence per physical substep. Its physics oracle
is the pinned upstream engine; cross-engine CPU differences are reported
separately. The two paths differ in engine and execution schedule as well as
precision.

The remainder of this document describes the handwritten native scalar
products, including the retained **explicit** `physics_engine="native"` FP32
comparison backend. These products share the maintained CUDA/Warp algorithms,
controller schedule and warmstart semantics. Native preparation/build commands
below prepare those products, not the MJWarp backend.

## Select and build

Select precision before importing typed physics modules. Each process has one
precision; use separate processes to compare or run both. Changing precision
after typed modules or owners exist raises an error. Importing the lightweight
policy itself does not initialize CUDA.

```python
from libero.gpu import precision
precision.select("fp32")
from libero.libero.envs import GpuBatchEnv

with GpuBatchEnv(task_bddl, num_envs=2, precision="fp32",
                 physics_engine="native",
                 build_root=project_root, allow_compile=False,
                 use_camera_obs=False) as env:
    observation = env.reset()
```

`LIBERO_GPU_PRECISION=fp32` selects the same process policy. The default is
`fp64`. An explicit constructor selection must agree with any existing typed
imports. To use the default FP32 fast path, follow the isolated MJWarp
environment and offline preparation instructions linked above.

Prepare each precision offline for the intended world capacity:

```sh
python -m libero.gpu.build --precision fp64 --num-envs 2 --jobs 1 --build-root /cache/project
python -m libero.gpu.build --precision fp32 --num-envs 2 --jobs 1 --build-root /cache/project
```

FP32 cold products, Warp modules and readiness records live under
`/cache/project/fp32`; FP64 retains its existing locations. CPU model snapshots
and official initial-state banks stay FP64. They are converted once when
uploaded to the selected device owners. The two precisions have distinct source
identities and native compiler flags. A native scalar-size export rejects a
library/host ABI mismatch.

`host/build/project_paths.py` resolves the public base project, selected scalar
namespace and shared CMake directory without creating or copying files.
`prepare_precision_project()` performs explicit offline namespace/catalogue
preparation. Namespace metadata (or an existing precision-tagged AOT receipt)
identifies a selected root; a directory merely named `fp32` does not. Existing
cache locations are retained. A stale copied task catalogue is rejected.

The maintained CMake tree builds both precisions. `FLASH_REAL_PRECISIONS=64;32`
controls its product set. `flash_substeps_all` and `flash_substeps_all_fp32`
build the 13 catalogued shape specializations. Ordinary functions remain
readable CUDA functions; precision is selected by `real_t` in
`cuda/common/precision.cuh`. MathDx descriptors use `Precision<real_t>()`.
There is no runtime substitution of arithmetic source text to select precision.

The CMake entry `cmake/Substep.cmake` owns options and reads the catalogue once.
`SubstepTargets.cmake` registers production scalar/shape objects with explicit
precision arguments. `SubstepTests.cmake` registers optional repository fixtures;
installed packages do not require the repository test directory.

The commands above prepare the existing component products used by construction
and reset. The single-physical-substep libraries have their own
[CMake configuration](README.md#single-physical-substep-cuda-fusion). Once that
tree is configured, build both products offline:

```sh
cmake --build /cache/project/cmake-control/sm80-dev \
  --target flash_substeps_all flash_substeps_all_fp32 --parallel 4
```

Select the same fused path and assembler policy explicitly on the batch owner:

```python
env.prepare_substep_fusion("/cache/project/cmake-control",
                          allow_compile=False, fast_compile=True)
```

The FP32 binder selects the `_fp32` library from the shared CMake tree. All
products must be current before this load-only call. The precision comparison
uses this fused path for both products.

## Storage and numerical policy

Physical arrays, model coefficients, Newton/OSC workspace and simulation time
use the selected scalar. Integer indices, capacities, epochs and work counters
retain their original widths. Raw floating-point bit views use matching
`uint32` or `uint64` words; byte strides and native contact layouts follow the
selected scalar size. Graphics attachments retain their own formats.

FP32 uses at least `1e-6` for the main Newton tolerance, following the default
floor used by the inspected MuJoCo Warp implementation. Its line-search
derivative tolerance also has a `1e-6` floor. Finite-magnitude guards have a
representable FP32 bound (`1e10`) instead of overflowing the FP64 `1e100`
constant. The FP64-only extreme-value proof for cached fluid coefficients
retains the original-arithmetic fallback in FP32. The integration step,
iteration cap, three forward solves per physical substep and 25 physical
substeps per control step remain unchanged. FP32 time accumulates rounding
error; exact integer integration counts establish equal work. Small constants
such as MuJoCo's `1e-15` denominator guard are not globally replaced by machine
epsilon. FP64 reference/oracle CUDA files and historical qualified binaries keep
their original source and qualification scope.

The native fused API also provides `single_solve=True`, an explicit FP32
scheduling candidate in a separate CUDA kernel. It prepares position/velocity
before OSC and solves once afterward. It is not enabled by default: the initial
combined scheduling/controller-initialization experiment failed the existing
CPU regression envelope. No native tolerance is enlarged to admit it.
The ordinary native kernel keeps the three-forward schedule in both precisions.
The MJWarp FP32 fast path uses one solve per physical substep as documented
above and exposes same-Model/Data stage boundaries for gradual replacement.

The FP32 fused Newton path also checks the ordinary normalized force-balance
gradient at the selected initial acceleration, after warmstart selection and
constraint-force evaluation. An already converged initial point proceeds
through normal result/warmstart publication without Hessian assembly, factoring,
or a line-search correction. It uses the existing main tolerance; it does not
adopt MJWarp's additional Newton-decrement stopping criterion. The internal
`NewtonOptions.initial_gradient_stop` diagnostic switch permits frozen-input
comparisons with the previous entry behavior. FP64 defaults to the previous
entry behavior. This path preserves every solver invocation and physical step.

BoxBox face clipping also checks the first incident vertex against the reference
rectangle. Rounded transforms can select either face at a SAT tie; the original
assumption that vertex zero is already inside need not hold. The shared fix
retains that vertex on the boundary and clips its exterior. Both scalar products
have geometric regressions for the captured case and exact touching boundaries.
FP32 SAT selection also uses a projection-roundoff allowance, proportional to
box sizes and local displacement and inversely proportional to cross-axis
length. It retains the face when an edge improvement is unresolved at that
precision. Separating-axis rejection is unchanged. FP64 retains the original
`1-1e-12` face preference.

The inspected MJWarp precision defaults are pinned to commit
`c384ac51df67eb92ab7657cc472ff7494dc036bf`: see its
[model tolerance conversion](https://github.com/google-deepmind/mujoco_warp/blob/c384ac51df67eb92ab7657cc472ff7494dc036bf/mujoco_warp/_src/io.py)
and [line-search tolerance](https://github.com/google-deepmind/mujoco_warp/blob/c384ac51df67eb92ab7657cc472ff7494dc036bf/mujoco_warp/_src/solver.py#L1046).

## Newton execution and workspace

Newton remains a separately compiled device function. For FP32 matrices with
`32 < NV <= 64`, warp zero applies ordered Cholesky update batches with fixed
lower-row ownership and register-held vector components. The pivot recurrence,
update order, diagonal floor and first-rank-loss rebuild are preserved. Small
matrices retain the serial update path; FP64 and larger dimensions retain their
existing collective fallback. The implementation is in
`cuda/step/newton_rank_update.cuh`, selected by `newton_solver.cu`.

The global Jacobian-product cache has a bounded 256 slots per packed matrix cell
in FP32 and 128 in FP64. Its host allocation and compiled byte count must agree.
Inputs exceeding a product or sparse-row capacity retain the complete existing
fallback. Increasing FP32 product capacity adds
`128 * NV * (NV + 1) / 2 * (sizeof(float) + sizeof(int))` bytes per world relative
to 128 slots, without increasing shared memory. Performance and trajectory
qualification are representative-domain results; full 130-task FP32 coverage is
not yet established for these changes.

`FLASH_SUBSTEP_THREADS` remains 64 by default. The CMake thread count is recorded
in the product manifest and checked against the configured cache. Changing it
requires matching offline rebuilds of the launch, Newton, controller and MathDx
objects. A 128-thread configuration is experimental and has only representative
FP32 qualification; it is not a generally qualified catalogue default.

## Validation and timing

`tools/validate_gpu.py run --precision fp32` selects prebuilt FP32 products.
It preserves the N=2-first policy and prohibits runtime compilation. FP64 uses
the existing frozen-reference error envelope. FP32 additionally permits an
explicit roundoff screening floor of
`1024 * float32_epsilon * max(1, frozen_CPU_scale, frozen_FP64_scale)` per field
and metric. The floor is fixed before execution and never derived from a new
candidate. This is an engineering regression screen, not a mathematical bound
on contact-sensitive long trajectories.

Acceleration and warmstart also use a local linear sensitivity allowance:
`2*kappa*u/(1-kappa*u) * norm(frozen_FP64_acceleration)`, where
`u=float32_epsilon/2` and `kappa` is the matching norm condition number of the
frozen FP64 mass matrix. A nonfinite condition number or `kappa*u >= 1` refuses
this qualification. This budget is computed before candidate execution and
cannot grow with a candidate's errors. The original unconditioned checks stay
visible in the receipt. It is a local numerical screen, not a proof of a
complete nonlinear rollout error bound.

An independent force-balance check additionally requires
`norm(M*a-f_smooth-f_constraint, inf) /
(norm(M,inf)*norm(a,inf)+norm(f_smooth,inf)+norm(f_constraint,inf)) <= 8*gamma_n`,
with `gamma_n=n*u/(1-n*u)`, for acceleration and warmstart. Evaluation uses FP64
and MuJoCo 2.3.7's algebraic packed-mass expansion; it performs no CPU physics
step. Reports retain all actual errors, contact counts, reset events and exact
work counters. State, control and other endpoint fields retain their existing
envelopes and FP32 roundoff floor.

`--performance` measures once after correctness succeeds, reusing the same
environment and graph. Setup, reset and graph preparation are separate.
Shared-device runs are permitted and include visible GPU load/process samples
outside timing; they do not establish idle-device throughput. Published FP64
historical benchmark charts are a separate dataset.

Live and offline validation share `tools/validation/policy.py`. Offline FP32
comparison requires the exact model and matching frozen input receipts; it
recomputes the policy from frozen inputs rather than trusting candidate limits.
The `summarize-precision` command compares existing paired measurements without
running physics. Integer work verification is independent of simulation-time
rounding. See [validation tools](../../../tools/validation/README.md).
