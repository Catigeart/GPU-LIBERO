# CPU/GPU endpoint and timing tools

Run `python tools/validate_gpu.py --help`. Importing these tools starts no work.
The `compare`, `summarize` and `summarize-precision` commands only read saved data. The `run` command
explicitly executes physics. Runtime paths never prepare missing CUDA/Warp
products. The separate `mjwarp-hessian --prepare` and `mjwarp-newton --prepare`
phases prepare local-reference kernels offline, without physics integration.
Historical dated scripts remain evidence and are not dependencies of this entry.

## Local handwritten constraint update

Build the independent `flash_mjwarp_constraint_fp32` target, then run
`mjwarp-constraint --input /path/to/newton.npz --build-root /path/to/project
--library /path/to/mjwarp_constraint_fp32.so --output /path/to/results --prepare`.
The first preparation freezes task-4 initialization from the exact Newton input
and task-5 initialization after one raw upstream forward. This reference
collection executes algebraic kernels, including that forward's solver, but
does not integrate physics. Later preparation reuses the immutable boundaries
and only captures local graphs. Run without `--prepare` for load-only checks.

The N=2 cases cover 13 NV families, equality/friction/unilateral constraints,
condim 3/4/6 cone zones and boundaries, interleaved pooled contacts, empty and
done worlds, unused rows and reference early-return guards. Synthetic tangent
regularization follows the CPU reference's relation to normal D and friction.
Independent FP64 grouped cone projection checks force and state; an absolute
product majorant checks each implementation's `J.T @ its_own_force` separately.
Near switching surfaces adjacent valid state choices are distinguished from
continuous force error. Input and skipped/unused output holds remain exact.

`--performance` measures only the two real-input cases after all checks pass:
32 identical calls per graph, five warmups and twenty samples, reporting
per-call N=2 time. The cumulative `mjwarp-newton --constraint-library ...
--direction-library ...` check replaces the initial and iterative updates;
the other controls and existing nonlinear residual policy remain unchanged.

## Local handwritten gradient and Newton direction

`mjwarp-direction` compares the independently compiled CUDA gradient/direction
stage with the pinned upstream kernels on identical resident inputs. Build
`flash_mjwarp_direction_fp32` offline. Pass the frozen Newton boundary as
`--input`, the compiled direction `--library`, `--build-root`, and `--output`.
Run once with `--prepare`, then without it for guarded, load-only validation.
An optional `--small-input` supplies a real task-5 NV21 boundary and its adjacent
receipt. Preparation captures graphs without replay or physics integration.

The cases cover all 13 catalogue NV families, poison in the unused lower H
triangle, padding, skipped worlds and zero gradients. The large route follows
upstream's 16-row augmented-RHS Cholesky; the small route uses POTRF/POTRS.
Independent FP64 solves use the symmetric matrix defined by the input upper
triangle. Search error is bounded by frozen-input condition sensitivity;
linear-solve backward error and output sums are checked separately. These
checks do not turn an upstream nonlinear stopping diagnostic into a new gate.

`--performance` runs only after all cases pass. Each local graph contains 32
identical stage calls, amortizing event resolution; five warmups and twenty
timed graph samples report per-call time. Only real-input cases are timed.
This is local algebra, not 32 physical steps or a task trajectory.
Use `mjwarp-newton --direction-library ...` for the cumulative solver check;
it retains the raw upstream, host-only control and Hessian-only controls.

## Local Newton replay with CUDA Hessian

`mjwarp-newton` consumes a saved `newton.npz` plus its adjacent `summary.json`.
It imports the same pinned scene/model and verifies the compiled model identity,
CSR indices and solver options. It constructs no batch environment and performs
no physical integration. Actual Newton solves are executed only in the run phase.

```sh
python tools/validate_gpu.py mjwarp-newton --prepare \
  --input /path/to/newton.npz --build-root /path/to/project \
  --library /path/to/cmake-tree/mjwarp_hessian_fp32.so --output /path/to/results
```

Run the same command without `--prepare`; add `--performance` for one local
Newton timing comparison after correctness passes. Runtime is compilation-guarded.
Preparation captures the graphs without replay and retains their owners.
The graphs restore identical resident inputs before every solve. The baseline
is upstream `_solve`, the staged-upstream control changes only host dispatch,
and `cuda_hessian` additionally replaces assembly. Two raw correctness outputs
define the existing FP32 reference-only envelope; no candidate expands it.

The N=2 cases cover the original task-4 input, disabled warmstart, a close
warmstart, one world without constraints, and manufactured QPs in the real
NV21/NV57 model layouts. Manufactured QPs include equality, friction, limits
and condim 3/4/6; they are not claimed as physical task trajectories. Public
acceleration/forces, states, iterations, overflow and exact input/warmstart holds
are recorded. Legal stopping/line-search choices need not have equal counts.

Independent FP64 evaluations of published-force and `J.T @ force` balance are
reported separately. The tight roundoff residual screen is not an upstream
Newton stopping rule: decrement/improvement stopping can pass before it does.
Same-engine equivalence does not establish absolute residual quality, and an
existing upstream residual failure must not be described as fixed by this port.

Performance uses the original real input, 5 warmups and 20 continuous timing
samples for each implementation, one comparison. Each sample restores outside
the event interval and measures one complete N=2 Newton solve. It records
iterations and inferred Hessian launch counts, with shared GPU load. These
are solver-stage timings, not control-step throughput or 130-task coverage.

## Local handwritten FP32 Hessian replay

`mjwarp-hessian` validates an independent CUDA leaf before changing production
Newton dispatch. Its call chain is `validate_gpu.py -> validation/mjwarp_hessian.py
-> DenseHessian.enqueue -> mass_quadratic -> cone_curvature`. The maintained
sources are [the CUDA translation unit](../../libero/libero/gpu/cuda/newton/mjwarp_hessian.cu),
[host binding](../../libero/libero/gpu/host/runtime/newton/mjwarp_hessian.py), and
[CMake target](../../libero/libero/gpu/cmake/MjwarpHessian.cmake).

Build only `flash_mjwarp_hessian_fp32` in the existing CMake tree. It has no
dependency on fused-substep, the old Newton solver, Warp headers, or MathDx.
The binding verifies its compiled source fingerprint, argument ABI, architecture,
array layouts and selected algorithm domain; it never compiles on load.
The leaf currently admits dense, elliptic, non-discrete, full-DOF Newton with
the catalogue's padded widths 24/28/32/48/64. Other domains fail explicitly in
this opt-in binding. The normal upstream FP32 dispatcher is unchanged.

In the isolated MJWarp Python environment, prepare the reference kernels once:

```sh
python tools/validate_gpu.py mjwarp-hessian --prepare \
  --input /path/to/saved-newton.npz --build-root /path/to/project \
  --library /path/to/cmake-tree/mjwarp_hessian_fp32.so --output /path/to/results
```

Repeat the command without `--prepare`, adding `--performance` for local timing.
Both phases bind the input, tool and upstream solver hashes. Runtime cache misses
fail under the compilation guard. No environment is constructed and no physical
substep is integrated. The saved Newton boundary predates an assembly-specific
snapshot: this replay explicitly derives Jaref from saved final acceleration and
uses saved final constraint states. The derived assembly inputs are preserved.
They are identical for both implementations, not a claimed replay of a particular
internal Newton iteration.

Checks include the real NV39 input, manufactured N=2 inputs for all 13 NV
families, condim 1/3/4/6, mixed world ownership, unused row tails, inactive/absent
contacts, empty constraints, zero friction, and exact skipped-world output holds.
Both implementations are compared with FP64 construction of explicit local cone
matrices, using a fixed input-scaled forward-error screen. This is local numerical
qualification, not 130-task trajectory coverage or a Newton convergence result.
The upstream replay also checks the small/large NV endpoint cases.

Performance uses 5 warmups and 20 timing samples of one reused graph; each
sample contains 64 calls by default to amortize event overhead. Reported units
are microseconds per assembly across both worlds. Full assembly and the two
component phases are measured separately. Isolated cone timing accumulates into
an initialized output; full assembly initializes its output on every call. Shared
GPU load is recorded. These timings are neither control-step nor complete-Newton
speedups.

## Protocol

- Default: N=2, no cameras, FP64, 5 warmup control steps and 20 continuous steps.
- `--precision fp32` selects the separate prebuilt GPU product. Its endpoint
  screen adds a fixed floor of `1024 * float32_epsilon *
  max(1, frozen_CPU_scale, frozen_FP64_scale)` to the existing per-field max/RMS
  envelope. This is an explicit engineering regression screen, not a proven
  bound for long contact-sensitive trajectories. Actual errors remain visible.
- FP32 acceleration/warmstart also use a sensitivity allowance from the frozen
  FP64 mass matrix, and must pass an independent force-balance backward-error
  check. Unconditioned endpoint results remain visible. See the
  [precision policy](../../libero/libero/gpu/PRECISION.md#validation-and-timing)
  for the formulas and limitations. These checks expand the packed mass matrix
  algebraically with MuJoCo 2.3.7; they execute no CPU physics step.
- `--execution components` checks the public component graph instead of the
  default single-physical-substep fused graph. It uses the same physical input
  protocol and remains load-only. Put different execution paths in separate
  output directories; forced cache-miss injection currently requires fusion.
- Tasks: suite names, zero-based `suite:ID`, or `all` (the 130 distinct tasks,
  without double-counting LIBERO-100). Choose representatives for the changed
  module rather than always selecting task 4.
- Use the existing offline project's `TASK_INSTANCES.json`, exact task model,
  first N official initial states, seed and established sinusoidal actions.
  Reference receipts bind model/state/action hashes and the control protocol.
- GPU construction uses `allow_compile=False` under `compilation_scope(False)`;
  fusion preparation is also load-only. Missing products/admission fail as
  infrastructure errors. These tools never update readiness records.
- One environment and fused graph per task. If requested, timing follows a
  passing endpoint check, resets the same environment, and reuses that graph.
  One task per worker bounds context/allocator lifetime.
- Endpoints compare per-world max/RMS error against the same-input CPU endpoint:
  `new_error <= max(3 * frozen_old_error, 1e-10)`. Timing uses the SAME frozen
  budget, not the first candidate's error. All shared reference fields are
  compared; missing fields and changed per-world widths are errors. Flattened
  versus structured representations of the same width are accepted.
- Additional OLD samples require explicit recorded references or `compare
  --old-sample`. The original old endpoint is always included. No failed
  candidate automatically causes more reference samples or a wider tolerance.
- N=1 is diagnostic after an N=2 numerical/work failure. `--diagnose-n1` runs it
  only when matching N=1 CPU and old GPU references already exist. A missing
  compiler product, process failure or other infrastructure error never triggers
  N=1; a passing N=1 does not turn the N=2 failure into a pass.
- Throughput is `N * measured_control_steps / elapsed_seconds`. Setup, reset,
  graph preparation and warmup are excluded. No extra factor of 25 is applied.
  Per-step mean, standard deviation, CV and p95 describe this single trajectory,
  not confidence intervals from independent repeats.

Integer work and floating simulation time are reported separately. `summarize`
gives raw totals, an integer-work-consistent subset and a simulation-time-consistent
subset. Integer work checks the measured integration count, zero resets and zero
error codes. Newton iteration statistics are not required to match. New GPU
receipts record substeps explicitly; `--substeps` supplies that metadata for
legacy receipts. Missing metadata gives an unknown work result, never a pass.
FP32 simulation time uses the documented roundoff floor; time agreement does not
prove equal integer work. An accepted optimization
regression does not establish absolute CPU/GPU trajectory equivalence.

## Record references and check a candidate

FP32 defaults to the upstream MJWarp fast path; FP64 defaults to native CUDA.
Use `--physics-engine native --precision fp32` for the handwritten comparison
backend. MJWarp requires its isolated environment and separately prepared
offline profile. Its integration oracle is the pinned upstream engine, rather
than the CPU 2.3.7 trajectory. Keep cross-engine CPU results separate.

After preparing the environment profile, prepare validation-only tracing once
offline, then check the adapter without runtime compilation:

```sh
python tools/validate_gpu.py mjwarp-reference --prepare \
  --build-root "$GPU_PROJECT" --task libero_10:4 --worlds 2 \
  --output "$VALIDATION_ROOT/mjwarp-check-prepare"
python tools/validate_gpu.py mjwarp-reference --raw-repeat \
  --build-root "$GPU_PROJECT" --task libero_10:4 --worlds 2 \
  --output "$VALIDATION_ROOT/mjwarp-check"
```

This checks the device bridge, initial and warmed physical substeps, a full
5-warmup/20-control-step recorded-control replay, and selected reset holds.
Raw replay uses official `step1`/`step2` and its own Data instance; it has no
adapter controller, sampler or work-counter calls. Independent force-balance
residuals are reported separately for both engines and are not silently
classified as passing physics merely because the adapter matches upstream.
`--prepare` captures operators without replaying physics and does not claim
trajectory correctness. Ordinary checks reject compilation.

`--raw-repeat` freezes a second independent raw Data/Graph with identical roots
and the recorded controls. Long-trajectory comparisons use per-world max/RMS
limits from those references alone: the unchanged 1024-epsilon floor or one
decimal order (ten times) their spread, whichever is larger. This implements
the FP32 fast path's accumulated-error-order criterion. Commands and time remain exact.
The original roundoff-only screen is retained in the report, as are failed
diagnostics. This engine-equivalence envelope does not change native/CPU error
budgets and does not qualify upstream force-balance residuals.
The device bridge and one-substep positions, velocities, commands, time and
smooth forces remain strict checks. Single-substep constraint accelerations
and constraint forces retain their roundoff-only diagnostic results; the
whole-trajectory order check includes them. This avoids classifying a
cancellation-sensitive upstream force output as an adapter state update error.

Use external directories. `GPU_PROJECT` is the existing unified project root;
`VALIDATION_ROOT` is a results directory outside the source checkout.
Run CPU recording with the CPU-reference Python/environment so its `libero`
imports resolve to the independent CPU checkout, using MuJoCo 2.3.7.
The runner rejects this project's GPU-routing `ControlEnv` before construction;
selecting `--backend cpu` alone cannot silently turn the GPU checkout into an
independent CPU reference.

```sh
/path/to/cpu-python /path/to/Flash-LIBERO/tools/validate_gpu.py run \
  --backend cpu --tasks libero_spatial:0 libero_10:6 \
  --build-root "$GPU_PROJECT" --output "$VALIDATION_ROOT/cpu" --performance
```

Record OLD GPU endpoints before modifying mathematics. This is a reference
record, not a claim of numerical qualification:

```sh
python tools/validate_gpu.py run --backend gpu --record-reference \
  --tasks libero_spatial:0 libero_10:6 --build-root "$GPU_PROJECT" \
  --output "$VALIDATION_ROOT/old-gpu" --fast-compile
```

After the normal offline qualification for the changed sources, check the new
version. Omit `--performance` for correctness only; add it to measure once after
correctness succeeds. `--fast-compile` selects already-built `sm*-dev` products;
it does not compile. Omit it for the standard assembler `sm*-release` products.

```sh
python tools/validate_gpu.py run --backend gpu \
  --tasks libero_spatial:0 libero_10:6 --build-root "$GPU_PROJECT" \
  --cpu-reference "$VALIDATION_ROOT/cpu" \
  --baseline-reference "$VALIDATION_ROOT/old-gpu" \
  --output "$VALIDATION_ROOT/candidate" --fast-compile --diagnose-n1
```

`--case varying_cache_miss` forces the last world through position-prefix
refresh; the others keep ordinary reuse. It changes a retained position-cache
flag, not physical state. It is useful for owner/lifetime/reuse changes and is
not mandatory for every numerical edit. CPU physical inputs match `varying`, so
its CPU reference can be reused. Old GPU references remain case-specific.
`--case zero` is available when a zero-action trajectory is relevant.

Reference endpoints and receipts are written once; an existing per-task result
is never overwritten. Recorded references have `status=recorded_cpu` or
`recorded_gpu`; checked candidates use `qualified` or `regression`. Infrastructure
errors are distinct. A source-only maintenance check does not require recording
new trajectories.

The `run` command reports public control-step timing. For MJWarp stage profiles
and an optional frozen Newton boundary, use its isolated Python environment:

```sh
python tools/validate_gpu.py mjwarp-stages --freeze-newton \
  --build-root "$GPU_PROJECT" --task libero_10:4 --worlds 2 \
  --output "$VALIDATION_ROOT/mjwarp-stages"
```

This command uses prebuilt products, one graph and 5 warmup/20 measured control
steps. It times resident stages with GPU events; parent step1/step2 times are
inclusive. Frozen inputs and outputs belong to the same compiled upstream
Model/Data. Event and snapshot overhead make these diagnostic timings distinct
from ordinary throughput. Numerical qualification remains the separate raw
upstream reference check. `run --single-solve` explicitly selects the native
FP32 scheduling candidate; the ordinary native schedule is unchanged.

Performance receipts include host load averages and CPU availability, plus visible
GPU utilization, memory and process samples
before and after timing. Shared-device timing is allowed and explicitly marked;
the command does not wait for idle conditions, stop processes or repeat timings.
Samples outside timing cannot establish the load throughout the measurement.

## Reuse legacy saved endpoints without physics

The caller must establish matching inputs and frozen-old provenance; bare legacy
NPZ files do not carry the new input signature. Prefixes allow reading a saved
combined `baseline.*` / `candidate.*` archive:

```sh
python tools/validate_gpu.py compare \
  --candidate "$SAVED_GPU_ENDPOINT" --candidate-prefix candidate. \
  --baseline "$SAVED_GPU_ENDPOINT" --baseline-prefix baseline. \
  --cpu "$SAVED_CPU_ENDPOINT" --worlds 2 \
  --output "$VALIDATION_ROOT/endpoint-check.json"
```

For an independently verified old-reference envelope, pass each extra NPZ with
`--old-sample`. The original `--baseline` remains in the envelope. This endpoint
command accepts partial auxiliary-field samples: a missing sample field retains
the original baseline's budget rather than inventing a new reference. The
command does not prove compiler admission, work counters or equal reset events.

Live and offline comparisons share `policy.py::prepare_policy` and
`ComparisonPolicy.evaluate`. `common.py` compares fields, while `precision.py`
owns mass expansion, frozen sensitivity and force-balance mathematics. FP64
bare/prefixed legacy NPZ comparison remains available with unverified input
provenance explicitly reported. FP32 comparison requires adjacent JSON input
receipts and the exact serialized model; model hashes, input signatures, world
counts and frozen-reference status are checked before policy construction:

```sh
python tools/validate_gpu.py compare --precision fp32 \
  --candidate "$VALIDATION_ROOT/fp32/libero_10__004_n2_varying.npz" \
  --baseline "$VALIDATION_ROOT/old-gpu/libero_10__004_n2_varying.npz" \
  --cpu "$VALIDATION_ROOT/cpu/libero_10__004_n2_varying.npz" \
  --model "$TASK_MODEL_DIRECTORY/model.mjb" --worlds 2 \
  --output "$VALIDATION_ROOT/fp32-check.json"
```

The budget is computed from frozen references before the candidate is evaluated.
Candidate receipt thresholds are not accepted as authority. Comparing a saved
performance endpoint finds its adjacent receipt by removing `-performance`.

Summarize the new runner's matching, qualified and timed receipts:

```sh
python tools/validate_gpu.py summarize --cpu "$VALIDATION_ROOT/cpu" \
  --gpu "$VALIDATION_ROOT/candidate" --output "$VALIDATION_ROOT/comparison"
```

Compare two already measured GPU precision products with the same input
signatures and build policies:

```sh
python tools/validate_gpu.py summarize-precision \
  --fp64 "$VALIDATION_ROOT/fp64" --fp32 "$VALIDATION_ROOT/fp32" \
  --output "$VALIDATION_ROOT/precision-comparison"
```

The comparison retains missing/failed cells, original unconditioned acceleration
checks, residuals, setup stages, integer work and shared-device load observations.
It performs no new qualification, compilation, timing or physics. Aggregate
throughput is total world control steps divided by summed elapsed time over
qualified matched pairs. This is a current FP64/FP32 comparison; it does not
silently mix historical CPU timing into that aggregate.

The existing `benchmark_scripts/benchmark_gpu_batch.py` remains the public long
batch benchmark (600 measured steps by default). It uses the ordinary component
graph and its existing random-action protocol. Do not mix its results with this
tool's explicitly fused sinusoidal protocol without stating that difference.

Local FP32 line-search qualification: build `flash_mjwarp_linesearch_fp32`, then
run `validate_gpu.py mjwarp-linesearch --input /path/to/newton.npz --build-root
/path/to/project --library /path/to/mjwarp_linesearch_fp32.so --output /path/to/results
--prepare`. This explicitly freezes the task 4/NV39 and task 0/NV57 upstream
pre-line-search boundaries without integration. Preparation may compile upstream
references; ordinary replay must have zero compilation events. Use
`--products-only --prepare`, then `--products-only` for the first product stage;
prepare/run again without that flag for the complete line search. Both passes
use N=2 with FP64 product/physical-cone-energy and commit checks. `--performance`
is opt-in, uses one function invocation per sample, restores outside timing, and
records concurrent device load. Cumulative `mjwarp-newton` accepts
`--linesearch-library` together with the three earlier libraries. No full task
trajectory is required for this local port.

Local Newton initialization/stopping qualification uses `mjwarp-solver-state`
with `--input /path/to/newton.npz --build-root /path/to/project --library
/path/to/mjwarp_solver_state_fp32.so --output /path/to/results`. Prepare once
with `--prepare`, then run without it. Preparation freezes the real task 4 and
task 0 initialization inputs and a complete task-0 Newton boundary, executing
one raw forward without integration. Local tests cover all 13 NV families,
warmstart, zero rows, untouched data, stopping thresholds, overflow flags and
repeated stopping calls. Ordinary checks prohibit compilation.

Pass `--state-library` plus the four earlier libraries to `mjwarp-newton` for
`cuda_newton`. Optional `--task0-input /path/to/frozen/task0-newton.npz` and its
same-stem JSON add a real NV57 fixture without recreating the boundary. With
`--performance`, task 4 and the optional real task 0 compare raw upstream,
previous cumulative cuda_linesearch and complete cuda_newton once each, using
5 warmups and 20 samples by default. Restore copies are outside timing. No
full physical trajectory or N=1 run is needed after N=2 passes.

## Local mass backsubstitution

Build `flash_mjwarp_mass_solve_fp32` offline. Use `mjwarp-mass-solve --build-root
/path/to/project --library /path/to/mjwarp_mass_solve_fp32.so --catalogue
/path/to/mass-layout-catalogue.json --output /path/to/results`, once with
`--prepare` and then without it. The catalogue records all 130 original scene
hashes and the pinned `io.m_block_layout` classification. Preparation imports
LIBERO-10 tasks 0/NV57 and 3/NV24, verifies their XML and exact block metadata
against that inventory, and freezes current factors and force RHS after raw
upstream position, velocity, actuation and acceleration. It does not invoke
Newton or integrate; existing frozen fixtures are reused, never overwritten.

Checks use N=2, the three actual block families (compact 1, scalar 6, tile 9),
supported scalar branch edges, exact aliasing, padding, zero RHS and deliberately
poisoned unused factor entries. The same captured bindings are replayed again
after replacing factor/RHS values to exclude stale numerical caches. Independent
FP64 residuals use the existing 8-gamma-block backward bound and its normwise
forward sensitivity bound; floating-point output equality is not required.
Input/hold contracts remain exact. Ordinary execution forbids compilation.

`--performance` measures only the two real LIBERO-10 function boundaries, once
per implementation, after correctness passes. Default counts are 5 warmups and
20 samples. Each sample has 32 idempotent solves to resolve microsecond timing;
reported times divide by 32. No input restore or host transfer is timed. GPU
competitors are recorded. These are local `solve_m` results, not trajectory or
130-task performance qualification. No N=1 check is needed after N=2 passes.

## Local complete acceleration

Build `flash_mjwarp_smooth_force_fp32` and the existing mass-solve target offline.
Use `mjwarp-acceleration --build-root /path/to/project --mass-library
/path/to/mjwarp_mass_solve_fp32.so --force-library /path/to/mjwarp_smooth_force_fp32.so
--catalogue /path/to/topology-catalogue.json --output /path/to/results`. First
prepare with `--prepare --cpu-python /path/to/cpu-reference-launcher`, then run
without preparation. The topology inventory reads the existing 130 CPU 2.3.7
MJBs, derives and proves contiguous subtree ranges from parent chains, and
records model hashes. The two GPU imports must match that inventory exactly.

Preparation freezes task 0/NV57 and task 3/NV24 after raw upstream position,
velocity and actuation. No Newton solve or physical integration is performed.
The actual CPU MuJoCo 2.3.7 `mj_applyFT` computes force/torque basis Jacobians
using exactly the frozen cdof/subtree_com/xipos arrays. Ordinary checks validate
that oracle against its receipt and cannot compile. Fixtures are reused.

N=2 cases cover real forces, a single loaded body, multiple loaded bodies,
world-asymmetric loads, force-only, torque-only, cancellation and padding. Each
is checked again with new applied forces using the same binding/Graph, including
zero-to-nonzero external force. Read-only inputs and output padding are held
exactly. Force errors use an a-priori FP32 operation-magnitude bound (3-operation
assembly, 8*gamma_(32+subtree-size) projection and final addition); no tolerance
is fitted to candidate output. Backsubstitution keeps the existing 8-gamma-block
residual and normwise sensitivity bounds, including the independent RHS error.

`--performance` measures only the two real LIBERO-10 boundaries, once for each
of upstream, previous mass-only CUDA, split CUDA force and fused CUDA force.
Times include the entire acceleration stage. Each uses 5 warmups and 20 samples
of 32 idempotent calls, divided by 32. No copies, preparation or host validation
are timed. Keep all spikes and record competing GPU processes; these results
do not establish full-task throughput or nonzero-wrench performance.

## Local mass factorization and factor/solve

Build `flash_mjwarp_mass_factor_fp32` independently, then use `mjwarp-mass-factor
--build-root /path/to/project --library /path/to/mjwarp_mass_factor_fp32.so
--catalogue /path/to/mass-layout-catalogue.json --output /path/to/results`.
Run once with `--prepare`, then qualify `--phase factor`, and finally qualify
`--phase solve --performance`. Solve qualification requires a passing factor
receipt for identical sources and inputs. Ordinary runs prohibit compilation.

Preparation freezes task 0/NV57 and task 3/NV24 original mass matrices and actual
Euler damping-adjusted matrices. It executes one raw initial-state Newton solve
per task to obtain the real `efc.Ma` RHS, then only prepares the effective matrix.
It never advances state/time or integrates. Static blocks are compared with the
existing 130-task inventory; CSR/gather consistency is independently admitted.
Existing frozen fixtures are reused rather than regenerated.

N=2 manufactured cases cover compact, scalar and tile blocks, repeated groups,
structural zero with negative sentinels, unequal world scales, exact x/y alias,
padding, zero RHS and an SPD condition-number case. Every case reuses its Graph
after replacing matrix/RHS values. Only effective triangles are checked
numerically; scalar untouched storage, tile lower-triangle zeros, readonly
inputs, indexing and padding keep exact contracts. Independent FP64 CSR
reconstruction uses the existing 8-gamma-block factor/solve residual budget and
normwise solution sensitivity. Equality of floating-point outputs is not a gate.

The local receipt binds the leaf source/header/CMake/binding, pinned upstream,
model collector, frozen inputs and binary. Unrelated backend scheduling edits
do not invalidate this frozen mathematical boundary; environment profiles still
bind all backend sources and require fresh offline admission after those edits.

Performance is limited to two real LIBERO-10 inputs: `factor_m` on original M
and `factor_solve_i` on the Euler matrix, each upstream/CUDA once, after both
correctness phases pass. Use 5 warmups and 20 samples of 32 idempotent calls,
dividing by 32. Copies and initialization stay outside timing. Retain every
sample and GPU-load snapshot; no full trajectory or N=1 run follows N=2 success.

## Local CRB and CSR mass assembly

Build `flash_mjwarp_mass_matrix_fp32` independently. Run `mjwarp-mass-matrix
--build-root /path/to/project --library /path/to/mjwarp_mass_matrix_fp32.so
--catalogue /path/to/topology-catalogue.json --output /path/to/results --prepare`
in the isolated MJWarp environment. Preparation freezes N=2 kinematic inputs
for LIBERO-10 tasks 0 and 3 and spatial task 0 (the inventory's maximum body
count), using kinematics/CRB only, with no Newton solves or integration.

Run the same command with `--cpu-reference` instead of `--prepare` in the CPU
2.3.7 environment. Its `mj_crb` oracle consumes those exact cinert/cdof/armature
values and converts CPU packed rows to the frozen CSR coordinates. The CPU
simple-DOF shortcut is disabled because it would otherwise consume cached M0
instead of the supplied leaf inputs. The independent FP64 spatial-matrix
formulas must agree with this oracle before candidate qualification.

Finally run without either flag in the MJWarp environment; compilation is
forbidden. Manufactured N=2 cases add wide sibling fan-in, deep branches,
independent trees, zero inertia, unequal world scales and armature broadcasting.
Each Graph is replayed again after replacing resident inertia/motion/armature
values. CRB and matrix values use componentwise bounds propagated from tree
addition majorants, spatial matrix-vector products and six-term dot products.
Readonly inputs, index maps, complete output overwrite and world separation
retain exact contracts. No trajectory or N=1 replay is required after this
local N=2 qualification. This command does not measure performance.

## Local spatial velocity and bias RNE

Build `flash_mjwarp_velocity_fp32`, then run `mjwarp-motion --phase com_vel
--build-root /path/to/project --kinematics /path/to/frozen-mass-matrix-inputs
--output /path/to/results --velocity-library /path/to/mjwarp_velocity_fp32.so
--prepare`. Preparation reuses the CRB qualifier's three real kinematic inputs
and creates distinct, nonzero velocities for both worlds. No Newton solve,
collision computation, integration or trajectory is needed. Run the same
command with `--cpu-reference` in CPU MuJoCo 2.3.7, then without either flag
in the isolated MJWarp environment with compilation forbidden.

After com_vel passes, build `flash_mjwarp_bias_fp32` and add `--bias-library
/path/to/mjwarp_bias_fp32.so --phase bias`, preparing offline and qualifying
load-only. Finally use `--phase composed` on the three real inputs; this requires
passing receipts for both exact current leaves. Ordinary qualification requires
the matching input/tool/reference/source/binary receipts. Existing inputs are
reused, not refrozen for each phase.

N=2 manufactured trees cover all four joint types, multiple joints/body, fixed
bodies, shared ancestor paths, wide fan-in, depth13 and independent trees. Zero
and nonzero velocities, gravity disabled/enabled and per-world gravity are
covered. The same Graph is replayed after resident input values change. Bias
tests use NaN qacc and nonzero external-force sentinels to check that the
flg_acc=False/no-external-force path ignores and preserves those inputs.

Independent FP64 dense spatial algebra propagates componentwise primitive
roundoff and tree-addition majorants; an 8x safety factor is applied to the
predicted bounds, not fitted to outputs. The composed check propagates velocity
error into RNE. CPU2.3.7 validates the FP64 formulas on exactly the frozen inputs;
its packed temporary/world-force conventions are not imposed on MJWarp public
intermediates. Floating-point equality is not a gate. Static maps, readonly
inputs, complete output ownership and model/index representation are exact
contracts. These checks measure no performance and skip N=1 after N=2 passes.


## Local postconstraint RNE suffix

Build `flash_mjwarp_postconstraint_fp32` offline, then prepare with:

```sh
python tools/validate_gpu.py mjwarp-postconstraint \
  --build-root /path/to/project \
  --motion-inputs /path/to/accepted-mjwarp-motion-inputs \
  --output /path/to/results \
  --library /path/to/mjwarp_postconstraint_fp32.so --prepare
```

Run the same command with `--cpu-reference` instead of `--prepare` in the
CPU MuJoCo 2.3.7 environment. Then run without either flag in the isolated
MJWarp environment with compilation forbidden. Receipts bind frozen inputs,
source, tool, reference and binary. No position/collision recomputation,
Newton solve, integration, trajectory or performance timing is needed.

N=2 cases cover three real task topologies and manufactured shared-prefix,
wide, deep, independent, mixed-joint and fixed-body trees. Distinct nonzero
accelerations and external wrenches, disabled/per-world gravity, pure
acceleration, pure external force and all-zero cases are included. Each
captured Graph is checked before and after changing resident physical inputs.
Both upstream and CUDA outputs are judged by independent FP64 spatial algebra
and componentwise input-derived FP32 roundoff bounds with fixed 8x safety.
Readonly inputs, public bias/warmstart holds and output ownership remain exact.
The CPU oracle calls actual `mj_rnePostConstraint` on the same inputs; coincident
application/subtree points make its external-force builder reproduce the frozen
wrenches exactly. This tests the suffix, not new external-wrench mathematics.
N=1 is skipped after N=2 passes; this command has no performance mode.


## Local joint passive forces and final combination

Build `flash_mjwarp_passive_fp32` offline. The `mjwarp-passive` command requires
`--build-root /path/to/project --catalogue /path/to/130-cached-model-inventory
--output /path/to/results --library /path/to/mjwarp_passive_fp32.so`.
Run first with `--cpu-reference` under CPU MuJoCo2.3.7 to freeze the three real
joint/parameter/state inputs and qualify the FP64 reference. Then run with
`--prepare` in the isolated MJWarp environment to prepare all upstream flag
variants offline, and finally with neither flag to qualify compilation-forbidden.
The catalogue must contain suite/task, model path and model SHA256 for the
three representative tasks. Existing accepted scene binaries are reused.

The local command checks joint forces, final sum and their leaf composition,
in that order. N=2 cases cover real NV57/24/43 models, all four joint types,
nonzero/polynomial/broadcast spring and damping, tiny and near-halfturn rotations,
Warp's zero-quaternion normalization convention, disable bits, gravity-compensation
actuator routing, fluid/adhesion selection and ignored NaN inputs. Other force
contributions are frozen manufactured vectors at this boundary, not recomputed
fluid forces. Each Graph is checked before and after resident inputs change.

Independent FP64 algebra uses primitive roundoff and quaternion sensitivity
bounds, with a fixed8x safety factor. The actual CPU `mj_passive` checks linear
real inputs with other contributions disabled; actual `mju_subQuat` checks
nonzero rotation inputs absent from the real spring-free models. Quaternion
branch sensitivity is explicitly included rather than imposing byte equality.
Readonly inputs, DOF ownership, output holds and source/binary receipts remain
exact. Separate caller tests retain the pinned both-disabled early return:
that branch clears the full passive result instead of calling the two-leaf
composition. No Newton solve, integration, N=1 replay or timing is performed.


## Local stateless actuation

Build `flash_mjwarp_actuation_fp32` offline. `mjwarp-actuation` requires
`--build-root /path/to/project --catalogue /path/to/cached-scene-inventory
--output /path/to/results --library /path/to/mjwarp_actuation_fp32.so`.
The catalogue contains suite/task/model path and SHA256 for LIBERO10 tasks0/3
and spatial task0. First use `--cpu-reference` under CPU MuJoCo2.3.7 to freeze
actual model parameters and transmission moments/lengths on cached states,
and qualify independent FP64 formulas. Then use `--prepare` in the isolated
MJWarp environment for upstream comparator preparation; ordinary qualification
runs with neither flag and compilation forbidden.

The check order is output force, sparse projection, joint limits, then composed
leaves. N=2 inputs cover the three real models and fixed/affine gains/biases,
control-slot permutations/shared/zero-input slots, distinct actuator/control
counts, multiple actuators per DOF and cancellation, per-world broadcast,
control/force/joint clamp boundaries, disabled clamps/gravity, zero rows and
ignored NaN length/velocity/moment storage. The same Graph runs original and
updated physical coefficients, controls, CSR values and valid column indices.
No Newton, position/collision recomputation, integration, N=1 or timing is needed.

Independent FP64 arithmetic propagates componentwise FP32 roundoff; clamp is
1-Lipschitz with exact endpoints and CSR atomic sums use absolute-contribution
majorants. A fixed8x safety applies, not floating output equality. CPU2.3.7
mj_fwdActuation checks both public real-model outputs; actual mju_mulMatTVec
checks isolated projection across all input families. Readonly arrays,
qacc/warmstart, output ownership and source/library receipts remain exact.
Host/capture checks separately preserve callbacks, disabled clearing, upstream
delay interpolation, whole-caller fallbacks and Graph owners. Stateful and
rotational-wrap mathematics is retained upstream and is not claimed migrated.
