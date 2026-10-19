# CPU/GPU endpoint and timing tools

Run `python tools/validate_gpu.py --help`. Importing these tools starts no work.
The `compare` and `summarize` commands only read saved data. The `run` command
explicitly executes physics; no tool here prepares missing CUDA/Warp products.
Historical dated scripts remain evidence and are not dependencies of this entry.

## Protocol

- Default: N=2, no cameras, FP64, 5 warmup control steps and 20 continuous steps.
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

Physical reset/time differences remain visible. `summarize` gives raw totals and
an explicitly separate time-event-consistent subset. An accepted optimization
regression does not establish absolute CPU/GPU trajectory equivalence.

## Record references and check a candidate

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

These tools report public control-step timing, not kernel phase profiles.
They do not monitor other GPU processes. On a shared device, establish idle
conditions and retain external competition samples before accepting performance
results; the command does not stop other processes or silently repeat timings.

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

Summarize the new runner's matching, qualified and timed receipts:

```sh
python tools/validate_gpu.py summarize --cpu "$VALIDATION_ROOT/cpu" \
  --gpu "$VALIDATION_ROOT/candidate" --output "$VALIDATION_ROOT/comparison"
```

The existing `benchmark_scripts/benchmark_gpu_batch.py` remains the public long
batch benchmark (600 measured steps by default). It uses the ordinary component
graph and its existing random-action protocol. Do not mix its results with this
tool's explicitly fused sinusoidal protocol without stating that difference.
