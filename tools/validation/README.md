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

- The legacy `run` validator defaults to N=2, no cameras, FP64, 5 warmup
  control steps and 20 continuous steps. Public `GpuBatchEnv` defaults to
  independent CUDA FP32; use `cuda-complete` for its qualification protocol.
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


## Local Euler preparation and advancement

Build `flash_mjwarp_euler_fp32` offline. `mjwarp-euler` requires
`--build-root /path/to/project --inputs /path/to/frozen-input-directory
--output /path/to/results --library /path/to/mjwarp_euler_fp32.so`.
Inputs are NPZ cases with nq/nv/njnt/nC, joint_type/qposadr/dofadr/dof_joint,
CSR row/nnz/column, current qpos/qvel, damping/dampingpoly, timestep, matrix,
and frozen acceleration/derivative arrays. All states/matrices contain two
worlds; each physical parameter has its independent broadcast row count.
The directory's complete files are bound by SHA256 in the preparation receipt.
Use `--cpu-reference` under CPU MuJoCo2.3.7 first, `--prepare` in the isolated
MJWarp environment next, and neither flag for load-only qualification.

The order is damping derivative, private diagonal update, velocity, position,
then their leaf composition on a frozen solved-acceleration boundary. It does
not run collision, Newton or factorization. Original and updated parameters
replay the same captured Graph. Both pinned upstream and CUDA outputs use
independent FP64 arithmetic/norm/division/quaternion sensitivity allowances
with fixed8x safety. Readonly inputs, public mass/qacc/warmstart holds and
source/library receipts remain exact. Actual CPU quaternion integration is
an additional oracle; pinned Warp's different all-zero-quaternion fallback is
qualified against the raw pinned comparator and independent explicit formula.
There is no performance mode; N=1 is skipped after N=2 passes.

Production-caller checks can separately replay `Euler.integrate` on frozen
real model inputs: qualify preparation, mass-solve residual, then advancement
from its actual private acceleration boundary. Full control Graph capture
checks dispatch/owners without replaying a trajectory. Warmstart publication
must copy public Newton qacc, which is distinct from the private damping solve.


## Local transmission and actuator velocity

Build `flash_mjwarp_transmission_fp32` offline. `mjwarp-transmission` takes
`--build-root /path/to/project --inputs /path/to/frozen-input-directory
--output /path/to/results --library /path/to/mjwarp_transmission_fp32.so
--actuation-library /path/to/mjwarp_actuation_fp32.so`.
Use `--cpu-reference` in CPU MuJoCo2.3.7, then `--prepare` in the isolated
MJWarp environment, then neither flag for compilation-forbidden N=2 checks.
The command has no performance mode. Full collision/Newton/trajectories are
outside this frozen-function scope; N=1 is skipped after N=2 passes.

Each NPZ supplies the existing actuation fixture fields, nq/qpos/qvel,
qposadr/dofadr, per-world-or-broadcast gear, and a velocity_only boolean.
For real cases, matching JSON receipts bind model_path/model_sha256 so the
CPU oracle calls actual mj_transmission. Generic manufactured cases use a
small CPU model; velocity-only cases can have arbitrary valid CSR rows.
All input-directory files are source-bound in the preparation receipt.

The sequence is scalar transmission, generic ordered CSR velocity, composed
leaves, then composition into stateless actuation. Raw pinned upstream
transmission/velocity/fwd_actuation provide GPU comparators. Independent
FP64 products/dot roundoff and the previously qualified actuation bounds
check each stage at its actual input boundary, with fixed8x safety. Actual
CPU mj_transmission checks length/moments; mju_mulMatVec checks velocities.

Cases should include representative real layouts, negative/zero/broadcast
gear, repeated target DOFs, differing actuator/control counts, and multiple
CTAs sharing a world's row counter. Generic CSR cases include empty/long rows,
cancellation, padding with unused NaNs/invalid indices and updated valid
columns/rows. The same Graph replays original and updated current inputs.

CSR row allocation is atomic and raw row offsets are unspecified upstream.
Checks unpack logical actuator rows, then require exact row count/ownership,
column identity and copied gear payloads, valid capacity, complete allocation
and unchanged unused tails. Readonly physical state, qacc/warmstart and public
holds remain exact. No floating output byte equality is imposed.


## Local rigid kinematics

Build `flash_mjwarp_kinematics_fp32` offline. `mjwarp-kinematics` takes
`--build-root /path/to/project --inputs /path/to/frozen-input-directory
--output /path/to/results --library /path/to/mjwarp_kinematics_fp32.so`.
Run `--cpu-reference` with CPU2.3.7, then `--prepare` with isolated MJWarp,
then neither flag for load-only N=2 checks. There is no performance mode.

NPZ inputs contain dimensions; parent/joint/DOF/branch/depth metadata; root,
weld, mocap and geom/site attachment indices; independently broadcast local
transforms and qpos0; qpos/mocap inputs; and frozen public output frames. JSON
receipts bind each NPZ and supply either a cached CPU model path/hash or a
small manufactured XML. All input files and numerical/reference helpers are
bound by the preparation receipt.

The sequence qualifies body matrices, inertial frames, geom frames and site
frames independently on frozen body poses, then body/anchor/axis propagation,
then all five stages composed. Original/updated inputs reuse each Graph.
Cases should cover real topology families, all joint types, off-center and
multiple joints, deep chains, shared ancestors, branches spanning CTAs, mocap,
parameter broadcasts and zero/near-halfturn quaternions. Inputs and unselected
outputs are held exactly, as are world poses and skipped static-geom outputs.

FP64 formulas use L2 error radii internally to retain rotational correlations:
quaternion multiplication is norm multiplicative; rotation has operator norm
||q|| squared; normalization removes radial error. Output radii bound each
component with fixed8x safety. This avoids artificially exponential interval
bounds on deep trees. Numeric output byte equality is not a gate.

CPU2.3.7's optimized rotVecQuat assumes unit quaternions; MJWarp uses a
homogeneous polynomial, including FP32 model quaternion norm deviations.
The CPU oracle therefore supplies the same normalized FP64 local quaternions
and axes to both formulas and actual mj_kinematics, disabling CPU sameframe
shortcuts when local transforms change. The raw FP32 GPU/formula check retains
all original parameters. Degenerate zero-quaternion cases are excluded only
from the CPU oracle and remain required GPU/formula cases. CPU hardcoded world
inertial frames and recomputed static geoms are outside the common comparison.
The new kernel retains pinned local normalization and exact public holds.

## Local subtree CoM / inertia / motion axes

Build `flash_mjwarp_com_pos_fp32` offline. `mjwarp-com-pos` takes
`--build-root /path/to/project --inputs /path/to/frozen-inputs
--output /path/to/results --library /path/to/mjwarp_com_pos_fp32.so`.
Optionally pass `--matrix-library /path/to/mjwarp_mass_matrix_fp32.so` to check
CoM -> CRB/CSR composition on the three representative real fixtures.

Run `--cpu-reference` in the CPU2.3.7 reference environment; run `--prepare` in
the pinned isolated MJWarp environment; then run without either flag for
load-only N=2 checks. Receipts bind inputs, independent formulas, helper source,
upstream source and CUDA products. Offline preparation captures operators but
does not replay physics. This qualifier has no performance mode.

Frozen inputs contain current mass/subtree-mass/principal-inertia and world
frames, immutable topology and root/joint/CSR metadata. Subtree, inertia, axes
and composition are checked separately; original and updated parameters/frames
replay the same Graph. Real NV24/43/57 fixtures and manufactured four-joint,
deep/wide, broadcast, mass-imbalance, zero/tiny mass and nonzero-world families
exercise dimensions and boundary semantics. Positive-mass common mathematics
is compared to actual CPU mj_comPos with its dynamically summed FP64 subtree
mass; zero-mass leaves, tiny masses and world inertia follow raw pinned MJWarp
instead of importing CPU's different fallback. All fields remain subject to
the independent raw-input FP64 oracle in GPU checks.

Acceptance uses absolute-contribution bounds for arbitrary atomic sibling
order, propagated tensor/cross-product sensitivity and fixed8x safety. It does
not use candidate-measured tolerances or require floating-point byte equality.
Readonly inputs/unselected outputs are exact holds. No integration, Newton
solve, N=1 replay or full-task timing is needed for this local migration.

## Local FIXED camera/light frames

Build `flash_mjwarp_camlight_fp32` offline. `mjwarp-camlight` takes
`--build-root /path/to/project --inputs /path/to/frozen-inputs
--output /path/to/results --library /path/to/mjwarp_camlight_fp32.so`.
Run `--cpu-reference` in CPU2.3.7, then `--prepare` in the pinned MJWarp
environment, then without either flag for compilation-forbidden N=2 checks.
Receipts bind inputs, CUDA product, helpers, upstream and actual Warp native
vec3 normalization headers. Preparation captures operators without physics.

The camera kernel, light kernel and their composition are checked separately.
Original and updated local camera transforms, light positions/directions and
body frames replay the same Graph. Representative real fixtures cover7/2,
8/3 and8/2 camera/light counts; manufactured cases cover moving/world mounts,
independent broadcast periods, non-unit quaternions, zero/tiny directions,
zero body quaternions, an empty camera/light half and multi-CTA attachment counts.
Readonly inputs and unselected outputs are exact holds; numerical acceptance
uses independent FP64 quaternion/frame formulas with propagated FP32 normwise
error bounds and fixed8x safety. There is no performance mode.

CPU mj_camlight uses d.xmat for local-to-world translation and d.xquat for
orientation/light direction. The common-domain oracle supplies coherent
normalized FP64 quaternions and their matrices to the actual CPU implementation
and the independent formulas. Raw GPU inputs stay unchanged. CPU's zero/tiny
normalization fallback and Warp's zero result differ; those cases remain
required raw pinned GPU/formula checks and are excluded only from the CPU
common-domain comparison. Non-FIXED mode fallback and all selector combinations
are covered by host tests. Local integration/capture checks require neither
N=1 nor long dynamics trajectories or image rendering.

## Local inertia-box fluid force and projection

Build `flash_mjwarp_fluid_fp32` offline. `mjwarp-fluid` takes
`--build-root /path/to/project --inputs /path/to/frozen-inputs
--output /path/to/results --library /path/to/mjwarp_fluid_fp32.so`.
Optional `--passive-library /path/to/mjwarp_passive_fp32.so` composes the real
fixtures into already-qualified CUDA joint passive forces and final sum.
Run `--cpu-reference` in CPU2.3.7, `--prepare` in isolated pinned MJWarp, then
without either flag for compilation-forbidden N=2 correctness. Receipts bind
inputs, independent formulas/helpers, upstream sources and selected products.

Force, projection and composition use independent Graphs over frozen inputs;
original/updated mass, inertia, wind, density, viscosity, frames and spatial
vectors reuse them. Real NV24/43/57 fixtures and manufactured mixed joints,
independent broadcasts, viscosity/density-only, zero coefficients/velocity,
mass threshold, inertia clamp, nonorthogonal matrices and multi-CTA domains
exercise actual work. Projection-only inputs deliberately use asymmetric
force/torque components. Readonly inputs and unselected outputs are exact holds.

The independent FP64 reference propagates componentwise FP32 arithmetic,
clamp/sqrt/power and spatial sensitivity errors, with fixed8x safety. Real
1e-15 and its rounded FP32 literal bracket the clamp-floor uncertainty; this
conversion matters at masses of the same scale. CPU mj_passive with joint and
gravcomp coefficients zero checks actual fluid math/projection, and mj_applyFT
checks separately supplied wrenches. CPU2.3.7 lacks body_subtreenum; fixtures
compute exact descendant counts from parents and CUDA admission proves DFS
contiguity. No physics integration, Newton solve, performance mode or N=1 is
needed for these local checks. Source accuracy reference for powf:
[NVIDIA mathematical functions](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/05-appendices/mathematical-functions.html).

## Local applied/contact external wrenches

Build `flash_mjwarp_external_wrench_fp32` offline. `mjwarp-external-wrench`
takes `--build-root /path/to/project --inputs /path/to/frozen-inputs
--output /path/to/results --library /path/to/mjwarp_external_wrench_fp32.so`.
Run `--cpu-reference` in CPU2.3.7, `--prepare` in isolated pinned MJWarp, then
without either flag for compilation-forbidden N=2 qualification. Receipts bind
inputs, tools, helpers, upstream sources and selected product. No N=1, timing,
solver iteration or physics integration is needed for these local checks.

Applied initialization, contact-only accumulation and their composition use
separate captured Graphs. Updated forces, frames, friction, adhesion, world ids
and device active counts reuse them. Both cones, dimensions1/3/4/6, scattered
elliptic addresses, clipped rows, excluded contacts, world-body signs, zero
active/capacity and multi-CTA fan-in are covered. Inputs and unselected output
parts are exact holds. Padded metadata is deliberately invalid and never read.

Independent FP64 decode/rotation/translation propagates componentwise FP32
arithmetic error; arbitrary atomic orders use absolute contribution sums with
fixed8x safety. Actual CPU2.3.7 mj_contactForce/mj_rnePostConstraint verify the
common zero-adhesion formulas. Equivalent MJWarp scattered/clipped rows are
repacked into contiguous CPU storage, with absent components zero. CPU2.3.7
has no adhesion: pinned net force subtraction is separately qualified against
the independent formula rather than changing the older reference. The existing
postconstraint FP64 reference accepts optional cfrc_ext_error for composition,
propagating external assembly errors through the RNE tree and sensor formulas.

## Local rigid dense constraint construction

Build `flash_mjwarp_constraint_build_fp32` offline, then use
`mjwarp-constraint-build --build-root /path/to/project --inputs /path/to/inputs
--output /path/to/results --library /path/to/mjwarp_constraint_build_fp32.so`.
Run `--cpu-reference` in CPU2.3.7, `--prepare` in pinned MJWarp, and then neither
flag for compilation-forbidden N=2 checks. Counters, category boundaries,
contiguous/disjoint contact ownership, copied bits and unowned bytes are exact.
Rows are matched by type/id/contact component rather than nondeterministic
atomic row offsets. Independent FP64 formulas propagate FP32 arithmetic,
polynomial/powf/regularization and Jacobian/reduction bounds with fixed8x safety.
CPU actual makeConstraint/referenceConstraint check the common zero-adhesion
mathematics; its differing elliptic tangential pos/margin storage is recorded.
Original/updated device counts, coefficients, contact frames/world ids and
velocities reuse captured Graphs. Empty/disabled paths, capacity truncation,
all contact dimensions/cones, direct/standard references and NV276 DOF striding
have dedicated fixtures. There is no local performance mode or dynamics replay.

## Local NXN broadphase

Build `flash_mjwarp_broadphase_fp32` offline, then use `mjwarp-broadphase
--build-root /path/to/project --inputs /path/to/inputs --output /path/to/results
--library /path/to/mjwarp_broadphase_fp32.so`. Run `--cpu-reference` in CPU2.3.7,
`--prepare` in pinned MJWarp, then neither flag for compilation-forbidden N=2
checks. Independent FP64 plane/sphere/eight-corner AABB/six-face OBB
inequalities propagate FP32 arithmetic bounds with fixed8x safety. Predicate
differences are permitted only when those intervals cross the decision boundary.
Candidate metadata, uniqueness, overflow attempts and existing/unwritten storage
holds are exact; atomic enumeration order is not compared. CPU's actual exported
`mj_collideOBB` checks finite non-plane zero-margin predicates. Independent
broadcast periods, sensor overrides, missing meshes, capacity truncation,
append/empty inputs and multiple CTAs are covered. This is a local correctness
protocol without a performance mode or physics integration.

## Local plane contacts and material/publication

Build `flash_mjwarp_plane_fp32` offline. Use `mjwarp-plane --build-root
/path/to/project --inputs /path/to/inputs --output /path/to/results
--library /path/to/mjwarp_plane_fp32.so`, with `--cpu-reference` in actual
CPU2.3.7, `--prepare` in pinned MJWarp, then neither for compilation-forbidden
N=2 checks. Geometry/material fields use independent FP64 formulas and fixed
input-scaled FP32 bounds; counts/keys/categories/copied metadata, pooled
ownership and prefix/tail holds stay exact outside explicitly bounded activation
boundaries. Mesh support uses independent exhaustive vertices and FP64 incident
polygon construction; discrete choices are also compared through the pinned
raw operator. Fixtures cover real mesh/graph/polygon data, all six plane types,
updated parameters, priorities/direct references, special solmix, sensors,
adhesion, independent broadcasts, degeneracy and pool overflow/append/empty.
CPU's real collision driver checks common primitive distance/position equations;
its different mesh and degenerate-cylinder algorithms are explicitly excluded
from the common-domain oracle. No local performance or physics integration.

## Local sphere/capsule/box contacts

Build `flash_mjwarp_box_fp32` offline. `mjwarp-box` uses the same
`--build-root`, `--inputs`, `--output`, `--library`, `--cpu-reference` and
`--prepare` protocol as the plane tool. It checks original/candidate N=2
manifolds, signed point-box distance, exact piecewise-quadratic
segment-to-AABB distance outside the box,15-axis SAT, unit normals and surface
witness residuals; materials/metadata/count/hold contracts are also checked.
Geometry tolerances are fixed input-scaled FP32 budgets. Actual CPU2.3.7
mj_collision checks common outside-box sphere/capsule witnesses and definite
box-box overlap. CPU's older inside distance is excluded. The inherited
capsule face-interior crossing miss is reported as a separate geometry
diagnostic for both baseline and candidate; it is not called a passed physical
completeness guarantee. Fixtures include all six shipped margin-route models,
near-parallel/interior/zero-size shapes, sensors, independent broadcasts and
capacity/append/empty cases. There is no performance or integration mode.

## Local rigid convex support building block

Build `flash_mjwarp_convex_fp32` offline and use `mjwarp-convex-support` with
the usual required paths, `--cpu-reference`, `--prepare`, then a load-only
N=2 check. This currently probes `support`, without a production convex
selector. Independent FP64 closed forms/exhaustive mesh max-dot, indexed
point/cache relations, closed shape surfaces and fixed propagated FP32
projection bounds are checked. Nonunique exposed points may choose distinct
legal vertices within those bounds; no bitwise float ranking is required.
Actual CPU2.3.7 public plane collision minimum projection verifies the common
primitive domain; mesh, zero direction, negative margin and singular axial
cylinder conventions are explicitly excluded from that runtime oracle and
covered by independent/pinned checks. All arrays are frozen/read-only except
the declared outputs. Graphs replay original/updated inputs without runtime
compilation; no full collision, integration or performance is claimed.

## Reachable ordered GJK simplex projection

`mjwarp-convex-simplex` uses the convex target and the same offline preparation,
required paths, `--cpu-reference` and load-only protocol. Its input simplexes
come from recorded original GJK executions, preserving vertex ordering. The
updated case permutes coordinate axes and applies positive uniform scaling; it
does not arbitrarily reorder vertices. Independent FP64 affine-face QP
enumeration checks closest distance, barycentric feasibility, weighted point,
norm and KKT residuals; each candidate's nonzero mask is exact. Degenerate
active-set choices are not compared bytewise. CPU2.3.7 uses MPR, so this
mathematical oracle is distinguished from an actual CPU GJK engine call.
The current CUDA entry is still a function probe without production collision
integration or performance measurements.

`mjwarp-convex-gjk` qualifies the GJK loop in the same convex library with a
separate entry ABI. Frozen N=2 requests explicitly supply pair poses, initial
witnesses, caches, iteration/tolerance/cutoff controls and margins. Both captured
graphs are replayed after current parameter updates; preparation is explicit and
checking forbids compilation. The candidate must pass independent FP64 support
bounds, active-simplex convex-QP and witness feasibility, exact range/hold and
cutoff-return contracts. Zero-iteration finite witnesses differ from empty
cutoff returns. Inactive simplex tails and legal active-set choices are not
compared bytewise. CPU2.3.7 uses MPR rather than this GJK algorithm.

Pinned upstream's expanded triangle determinants can themselves violate
barycentric feasibility. These failed mathematical checks remain visible as
baseline defects; they do not relax candidate bounds or silently become passes.
Candidate mathematics, upstream public/range contracts and distance conformance
remain separate mandatory decisions. This function probe does not select
production convex collision or establish full-trajectory/performance parity.

`mjwarp-convex-ccd` extends the convex library with the rigid shape-aware GJK
phase and EPA. It accepts the same frozen query arrays, plus one global scratch
slice per request and explicit EPA/face/horizon capacities. The independent CCD
entry ABI also admits a read-only frozen original GJK simplex, for isolating EPA
from preceding floating-point active-set choices. Source admission and offline
preparation remain mandatory. Normal and forced capacity paths are separate
cases; no production collision selector is implied by this probe.

The qualifier checks signed witness norms, packed face ranges, affine plane and
witness residuals, independent rounded-box/SAT distances where applicable,
current-input Graph replay, public counts/overflow, read-only owners and raw
reference conformance. The pinned coincident-initial-witness immediate return
is retained and explicitly excluded from a complete penetration-depth promise.
Actual CPU2.3.7 collision outputs are retained as version/algorithm observations:
its interior sphere-box clamp, half-depth box face contacts and MPR are not an
EPA depth oracle. CPU common support projections were independently checked by
the preceding support qualifier.

General cylinder-cylinder EPA remains unqualified: legal almost perpendicular
support directions choose different nonunique caps, and the candidate can reach
face/horizon capacity while the pinned path converges. This pair is absent from
the shipped130-task prefiltered catalogue. Its failed full-corpus and same-frozen-
simplex prefix experiments must remain visible; a future production selector
must fall back for additional unqualified pair families. It must not claim a
generic six-shape EPA migration from the six-shape support qualification.

`mjwarp-box-manifold` freezes original EPA faces for the convex box-box
multicontact function. It checks box feature IDs, face/edge matching, clipping,
farthest-edge pruning and maximum-area four-point selection separately from
GJK/EPA. The candidate borrows immutable original polytope storage; input and
unused query outputs retain exact public holds. Explicit offline preparation and
load-only checking use the existing convex library and independent entry ABI.

Witness norms and the second box's surface use the FP32 residual budget. The
pinned face-face path treats normals with dot(n1,n2) < -cos(.0889) as aligned,
then projects along n2. Relative to exact projection along -n1, its first witness
can move by at most abs(depth)*sqrt(2*(1-cos(.0889))). This geometric approximation
is reported separately from floating-point error; it does not change candidate
arithmetic or enlarge the floating-point tolerance. Original strict surface
failures remain useful algorithm observations, not migration defects. Raw and
CUDA point sets/counts are compared independently of contact order. This probe
qualifies only box-box multicontact, which remains enabled when general MULTICCD
is disabled; it does not qualify generic mesh/cylinder multicontact.


`mjwarp-convex-collision` checks the actual production contact ABI on frozen
N=2 candidate pools. It covers every shipped convex pair family, per-world
broadcasts and parameter updates, direct reference materials, margins/gaps,
adhesion, sensor-only/suppressed constraints, append prefixes, pooled contact
overflow, zero/small CCD capacity and the primitive box-box route. It checks
independent materials/frames, public range/owner/hold contracts, attempted
counts/overflow, and unordered per-pair raw contact conformance. Local manifold
IDs remain complete0..n-1; contact ordering may rotate with a legal EPA feature
choice. Raw operators are explicitly prepared offline; checking forbids
compilation and retains the same workspace pointers. This command qualifies the
pooled convex leaf; complete collision/Graph ownership and full trajectories
are separate caller/final-suite qualifications.

## Retained complete-path endpoint analysis

`mjwarp-endpoint-report --build-root /path/to/project --input /path/to/complete-results
--output /path/to/endpoint-report.json` reads completed `mjwarp-complete` reports
on CPU. It checks task/scene bindings, recorded endpoint hashes and common
executed physics/reference signatures before using the existing FP32
`compare_replays` policy. Historical original-repeat arrays did not have a
pre-execution hash receipt; the reporter matches every stored variation
observation and records the current file digest without claiming an older hash.

The output preserves three separate results: original strict checks, all
post-Euler qpos/qvel plus closed-loop OSC output/time, and the task boundary
(robot9DOF, all public body poses, OSC output/time). Body poses are the public
pre-final-Euler frames. The task boundary excludes free-object velocities,
accelerations and internal forces. No alternate tolerance is fitted. If no
untimed original repeat was needed, the fixed existing FP32 floor applies.
Closed-loop OSC output is distinguished from identical recorded actuator
commands, whose equality remains exact in function replays.

This reporting command launches no physics, does not repeat timing, and never
rewrites the source summaries or readmits a failed strict comparison. Exit zero
means the evidence was analyzed successfully; numerical conclusions are the
explicitly scoped fields in the output. Missing/changed outputs, unlike executed
physics or inconsistent reference-repeat observations stop the report.

`mjwarp-complete --correctness-only` executes the same5 warmup and20 continuous
control actions without creating timing events or entering the measurement
helper. It records completed control counts in `execution.json`, empty timing
fields and no speedup. The mode and action helper are bound into source/protocol
receipts and propagated to children/resume admission. Explicit offline Graph
preparation is still separate; execution forbids compilation. The CPU endpoint
reporter accepts a completed untimed run and rejects partial action sequences.

For frozen CCD inputs, optional `verify_box_overlap` checks zero-margin BOX/BOX
completeness against independent FP64 SAT on transformed corners, outside the
existing input CCD tolerance. It checks separated controls as well as genuine
penetration and binds the SAT helper into the preparation signature. Apply it
to complete candidate requests, not only already emitted contacts. The pinned
implementation can itself fail this check; preserve its failures and raw
conformance separately when evaluating a geometric correction. Such a check
does not replace public ownership, capacity, witness, residual or trajectory
checks, and does not enlarge their accepted bounds.

## Maintained convex regression and archived diagnostics

`mjwarp-convex-regression --library CONVEX --build-root BUILD --output OUTPUT`
checks the retained tuned-direction GJK counterexample and deeper, separated and
translated controls. Run once with `--prepare` in the MJWarp reference
environment, then load-only without it. The fixture is
`tests/data/gjk_tuned_direction.json`; the function probe and production convex
leaf share their CUDA headers. Original failures remain visible separately from
the corrected candidate. This local regression does not run a task trajectory
or performance measurement.

One-off `cuda-attribution*` and `mjwarp-residual-*` capture, policy and causal
analysis tools were archived outside the source repository in the development
workspace's `archive/fp32_development_20261021/`. The archive retains the complete
original validator package, CLI and checkpoint-schema test with a SHA256 move
manifest. Historical evidence and exact executed tool snapshots remain in their
original results directories. These experiments are not production validation
commands. Reusable mathematical qualifiers, independent oracles and the GJK
regression fixture remain in this repository.

## FP64 motion replacement

`fp64-motion --build-root ROOT --baseline FROZEN_SUBSTEP_SO --output RESULTS`
uses one load-only N=2 native environment. `--task` selects a LIBERO-10 task;
`--state` optionally supplies a saved FP64 endpoint. First it compares the
motion/bias/passive region against the admitted native component functions,
without Newton or integration. It checks 512/256/128-thread candidates, then
short physical-substep composition, policy Post and task4 mixed cache misses.
Floating-point checks use the accepted input-scaled FP64 error envelope;
physical flow and invocation counters remain exact. It is not a 130-task gate.

`--performance` selects the locally fastest thread candidate and compares one
complete native control-step Graph against the frozen fused product. Each
version has five warmups and twenty continuing control steps. Versions are
interleaved with separate resident state snapshots; restore/save copies are
outside CUDA events. Construction, reset, capture and transfers are excluded.
GPU co-tenant snapshots and individual samples are retained. No runtime build,
clock change, renderer or other-process intervention is permitted.

`--host-state` keeps the two continuing-version snapshots in pinned host memory
when device memory is occupied by another job. Asynchronous copies use the same
stream as the Graph and remain outside the CUDA event interval; both versions
use the same storage mode and the report records it. Numerical input and
work-counter checks are unchanged.

`--producer force --baseline-motion` instead checks the entry actuation/smooth
producer and compares the complete chain against a frozen motion-replacement
product. Local checks include object/robot external wrenches, control limits
and BADCTRL fallback; composition checks include persistent control warnings.
Mass-factor reuse counters must match, and the final healthy candidate must
have produced both worlds' force handoffs. `--threads` limits variants when
producer mathematics is unchanged. `--prior-force SO` optionally compares a
frozen producer on the same held inputs for local tuning. These options retain
the same no-runtime-compilation and N=2-first policy.

`--baseline-force --baseline-force-threads N` instead binds an explicitly frozen
accepted force product after ABI/NV checks, allowing an incremental comparison.
The current healthy path must finish with handoff 2: after-controller force
outputs are resident in the residual kernel's shared arena. A version-1 frozen
baseline retains its original handoff 1 and launch implementation. Frozen
products are diagnostic inputs and are never used for ordinary product loading.

`--producer force --newton-split --baseline-force` evaluates the actual
Init256/Resume64 composition against that frozen force product. This direct
regional experiment deliberately permits every compiled NV shape; the public
`staged=True, actuation_solve=True` selector currently admits only measured
NV39 and otherwise retains force fusion. The optional final Post/Euler input
export is outside all numerical/timing gates and reports export failures
separately. Its diagnostic Full-Newton entry is configured before capture,
including the wide shape's dynamic shared-memory opt-in.

`--baseline-aligned` binds a frozen accepted init/resume product for incremental
comparison instead. `--verify-native-workspace` additionally checks the current
Full-Newton force path in the same composition cases, reusing the environment;
that third implementation is never included in the performance comparison.
Use this when a shared-arena change affects both the split and fused callers.

Optional CMake targets `flash_post_replay_probe` and `flash_euler_replay_probe`
expose the production mathematical regions to fixed-input diagnostic drivers.
Post accepts the Newton-private acceleration and an optional qualified subtree
descriptor; its scratch includes the shared contact-tile/raw-force lifetime.
Euler accepts runtime widths within the catalogue ceiling and executes the
same damped integration body as the native substep. These targets are excluded
from ordinary builds, contain no alternate reference mathematics, and do not
construct environments or compile at runtime. Independent reference inputs and
experiment reports belong outside the source repository.

Optional CMake targets `flash_newton_replay_probe_profile[_nv21|_nv57]` call
the actual independently compiled instrumented Newton with a fixed QP. They
reuse vendor and reduction objects, omit the forward body, and export the
existing profile table configuration/schema/name functions. The ordinary replay
and production targets contain no profile recorder. Record the initial cone
state as well as the final state: a converged frictionless final solution does
not establish that warmstart candidates needed no cone factorization. Nested
profile scopes overlap their parent phases and must not be added as independent
percentages. Fixed-QP timing is a regional measurement, not task throughput.
