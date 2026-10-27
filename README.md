<div align="center">

# GPU-LIBERO

### GPU Simulation for Lifelong Robot Learning

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [Project Website](https://libero-project.github.io/) · [LIBERO Paper](https://arxiv.org/abs/2306.03310) · [LIBERO Documentation](https://lifelong-robot-learning.github.io/LIBERO/)

</div>

**Default batch backend: independent CUDA FP32, with no Warp dependency.**
Native CUDA FP64 and the original MJWarp FP32 reference remain available.
See [recorded throughput](#performance) for current FP64 and historical
four-backend results with their timing scopes.

Both `physics_engine="mjwarp"` and `physics_engine="cuda_fp32"` use the same
**CUDA FP64 OSC**, with the control algorithm ported from **robosuite's
operational space controller**. Physics and resident controller buffers remain
FP32. The controller promotes mass matrices and Jacobians to FP64 for matrix
solves, pseudoinverses, nullspace projection and torque accumulation, then
publishes FP32 controls. The implementation is
[osc_solver.cu](libero/libero/gpu/cuda/control/osc_solver.cu).

**GPU-LIBERO** adds GPU physics to [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), a benchmark for knowledge transfer in multitask and lifelong robot learning. Install this distribution and keep your existing LIBERO environment code: simulation uses the GPU by default.

- Original LIBERO imports, task definitions, environment interfaces, and benchmark suites.
- 130 manipulation tasks across **LIBERO-Spatial**, **LIBERO-Object**, **LIBERO-Goal**, and **LIBERO-100**, including the **LIBERO-90 / LIBERO-10** split.
- Native FP64 CUDA physics with cuBLASDx / cuSOLVERDx, and independent CUDA FP32 physics following MJWarp algorithms.
- Existing observation, rendering, controller, state, and vector-environment interfaces.
- LIBERO policy architectures, lifelong learning methods, and dataset tools.

The [independent CUDA FP32 backend](libero/libero/gpu/host/backends/cuda_fp32/README.md)
owns model preprocessing, device memory, Graphs, physics scheduling, OSC transport
and observations without installing Warp or MJWarp. It shares the maintained
handwritten CUDA mathematics with the MJWarp migration path. Its isolated
environment uses CPU MuJoCo 3.16.0 for model compilation and OpenGL rendering;
native FP64 uses MuJoCo 2.3.7. The explicit
[MJWarp reference backend](libero/libero/gpu/host/backends/mjwarp/README.md)
remains available. **FP32 is the default precision.** Select `precision="fp64"`
explicitly for native FP64. The performance section separates the current
native GPU FP64 result from historical CPU FP64, GPU FP64, original MJWarp and
independent GPU FP32 data.
For `GpuBatchEnv`, automatic routing selects independent CUDA for FP32 and
native CUDA for FP64. Use `physics_engine="mjwarp"` for the optional reference.
The [FP32 qualification report](libero/libero/gpu/host/backends/cuda_fp32/QUALIFICATION.md)
records 130-task execution, 127/130 upstream trajectory checks, retained
exceptions and recorded device throughput on a shared GPU. Its measurement
scope differs from the historical FP64/CPU public-step measurements.
Those full-suite numbers precede the subsequent ND-05 GJK correction; the report
separately records its targeted geometry and three-task correctness follow-up.

## Contents

- [Performance](#performance)
- [FP32 numerical corrections and scope](#fp32-numerical-corrections-and-scope)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Tests](#tests)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Performance

The GPU FP64 column uses the **2026-10-25 to 2026-10-26 public-step wall-clock run covering all 130 tasks**. CPU FP64, MJWarp FP32 and independent GPU FP32 retain their existing measurements; they were **not retested**. All use NVIDIA A100 SXM4 40GB where applicable, **N=2**, no cameras, five warmup and **20 continuous timed controls, once per task**. Suite throughput divides total environment control steps by total elapsed time, rather than averaging task rates.

**Dates, action protocols, timing scopes and GPU load differ. This is an absolute-throughput summary, not a controlled four-backend ranking or a same-workload speedup claim.**

![Absolute recorded throughput for four backends, with refreshed public-step GPU FP64 and retained historical measurements](media/benchmarks/libero-suite-throughput.svg)

| Task suite | Tasks | CPU FP64 | GPU FP64 | MJWarp FP32 | GPU FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LIBERO-Spatial | 10 | 31.99 | 20.59 | 49.25 | 58.45 |
| LIBERO-Object | 10 | 44.38 | 35.35 | 53.08 | 93.03 |
| LIBERO-Goal | 10 | 44.02 | 32.98 | 39.40 | 83.90 |
| LIBERO-90 | 90 | 46.75 | 38.27 | 58.90 | 83.04 |
| LIBERO-10 | 10 | 46.94 | 37.95 | 56.29 | 78.24 |
| All distinct tasks | 130 | 44.78 | 35.26 | 55.30 | 80.78 |

Values are **environment control steps per second summed over both worlds**.

| Backend | Date | Measured interval | Competition |
| --- | --- | --- | --- |
| CPU FP64 | 2026-10-19, retained | Public `step()` wall time; two CPU MuJoCo 2.3.7 subprocess workers on distinct physical cores, one BLAS thread each | Historical campaign conditions |
| GPU FP64 | 2026-10-25 to 2026-10-26, refreshed | Complete public `step()` wall time, including host observations and task predicates; existing optimized native kernels | Only attempts without sampled other GPU processes retained |
| MJWarp FP32 / independent GPU FP32 | 2026-10-21, retained | CUDA-event intervals including upload, physics, OSC and device observations; excludes host packaging and predicates | Shared GPU; per-task samples retained |

GPU FP64 timing wraps actual `env.step(action)` calls, including action submission, OSC, physics, required stream synchronization, host observation packaging and task predicates. Construction, reset, graph preparation and warmup are timed separately; actions are generated beforehand, and logging, diagnostics and file IO stay outside the measured loop. All 130 retained tasks completed **500 timed physical integrations per world**, with no runtime errors or compiler commands. Attempts sampling another GPU process were excluded; valid completed tasks were retained on resume.

The FP64 run reuses the baseline initial-state banks, seed 0 and fixed action `[0.02, -0.015, 0.01, 0.005, -0.003, 0.002, 0.1]` in both worlds. Existing optimized options are `staged=True`, `motion_passive=True`, `actuation_solve=True`, `motion_threads=512`, `force_threads=256`, `defer_post=True` and `fast_compile=True`: `ControllerAlignedSubstep` for NV39, `PeeledForceSubstep` for other catalogue shapes. Solver semantics and physics algorithms are unchanged. The full timed block totals **147.487 s**, yielding **35.26 environment control steps/s** across 130 tasks. [FP64 per-task timings and setup costs](media/benchmarks/fp64-public-step-tasks-20261025.csv) · [FP64 suite summary](media/benchmarks/fp64-public-step-summary-20261025.csv) · [FP64 protocol and provenance](media/benchmarks/fp64-public-step-provenance-20261025.json).

Creation, reset, graph preparation, compilation and rendering are excluded from throughput. Each control contains 25 physical substeps; throughput is not multiplied by 25. LIBERO-100 is the union of LIBERO-90 and LIBERO-10, not another set of measurements. The old LIBERO-90 task 85 CPU/GPU time-event comparison flag remains explicitly historical in the CSV.

Historical GPU FP64 **device-only CUDA Graph** results (October 23–25, 2026; **37.13 environment control steps/s** across 130 tasks) and the earlier paired experiment remain available as [suite summary](media/benchmarks/fp64-native-summary-20261025.csv), [standalone per-task data](media/benchmarks/fp64-current-standalone-20261025.csv), [paired per-task data](media/benchmarks/fp64-paired-20261022.csv) and [provenance](media/benchmarks/fp64-native-provenance-20261025.json). These exclude action staging and host packaging/predicates and are not mixed into the public-step column or figure.

**FP32 timings predate the ND-05 GJK correction and subsequent world-local fusion.** These later changes did not repeat the full-suite timings. See [FP32 regional checks and timings](libero/libero/gpu/host/backends/cuda_fp32/README.md#supported-scope-and-validation). Original MJWarp denotes pinned upstream physics with the shared project OSC/observation integration; GPU FP32 denotes the independent Warp-free CUDA backend.

[Per-task CSV](media/benchmarks/task-throughput.csv) · [Aggregate CSV](media/benchmarks/suite-throughput.csv) · [Measurement provenance](media/benchmarks/provenance.json).
Regenerate the aggregate and SVG with `python tools/plot_suite_throughput.py`.

## FP32 numerical corrections and scope

The independent FP32 implementation follows MJWarp commit
`c384ac51df67eb92ab7657cc472ff7494dc036bf` (MuJoCo 3.16.0, Warp 1.18.0 as the
reference). Its CUDA mathematics, device memory, Graphs, model preprocessing,
controller transport and observations execute without installing either Warp
package. The `mjwarp` prefix in shared CUDA filenames records algorithm
provenance; it is not a runtime dependency.

We intentionally retain these corrections beyond a literal port:

| ID | Correction | Scope / purpose |
| --- | --- | --- |
| ND-01 | Translate triangle determinants and affine-weight expressions to edge vectors | Reduce cancellation while preserving the real-arithmetic formula |
| ND-02 | Match zero-direction BOX support points to their corner feature IDs | Keep the support point and recovered feature consistent |
| ND-03 | Refine inconclusive GJK results and attempt EPA only with a certified enclosing hull | Avoid treating a positive simplex-distance upper bound as proof of separation in the qualified zero-margin BOX/BOX domain |
| ND-04 | Retain nondegenerate shallow EPA faces when the complete initial hull encloses the origin | Avoid discarding valid shallow contacts solely because squared face distance is below `1e-10` |
| ND-05 | Verify tuned GJK cutoff exits in the direction actually used for the support query | Repair an inherited invalid separating-plane certificate |

The [numerical-deviation register](libero/libero/gpu/host/backends/mjwarp/NUMERICAL_DEVIATIONS.md)
records exact domains, source functions, independent CPU/FP64 geometric evidence
and retained limits. ND-04's exceptional hull certificate uses FP64 predicates
on rounded FP32 vertices; state and solver arithmetic remain FP32. No blanket
claim of greater whole-trajectory accuracy is made. These corrections can
change contact sets and trajectories; matching unmodified MJWarp and satisfying
independent geometry are separate checks. Solver stopping rules, warmstart,
physical parameters and trajectory tolerances were not changed by them.

The completed 130-task campaign passes all public observation/predicate
boundaries and the existing upstream trajectory policy in **127/130** tasks.
The retained exceptions are Spatial:5 and LIBERO-90:73/75. The latest ND-05
follow-up still leaves these three outside that original endpoint policy:
Spatial:5 has a substantial impact-phase-sensitive angular-velocity difference,
while the two mugs differ on the order of `1e-4 rad/s`. Local collision and
same-QP checks distinguish these cases; they do not prove every whole
trajectory identical or globally more accurate. The full suite has not been
rerun after ND-05. See the
[qualification report](libero/libero/gpu/host/backends/cuda_fp32/QUALIFICATION.md)
for versioned results, the separate stricter checks and 600-control follow-ups.

The supported shipped domain is rigid LIBERO, dense Newton, elliptic contacts,
Euler, Panda OSC, FORCE/TORQUE sensors and FIXED cameras/lights. Unsupported
features fail explicitly; this backend is not a general replacement for every
MuJoCo model. Original MJWarp remains an explicit reference; native FP64 remains
available through `precision="fp64"` in its separate dependency environment.

## Installation

### Requirements

- Python **3.10 or newer**.
- An NVIDIA GPU, NVIDIA driver, and CUDA Toolkit with `nvcc` available.
- CMake **3.24 or newer**, Ninja, and a C/C++ compiler for native dependencies.
- NVIDIA MathDx. If no installation is configured, the runtime downloads its pinned **25.12.1 CUDA12** package on first use.

The single-substep build's default local MathDx prelink requires **NVCC 13.2 or
newer**. Older toolchains require the explicit whole-program device-LTO build
configuration; see the [GPU build guide](libero/libero/gpu/README.md).

For the default independent CUDA FP32 batch, install from the repository root
in an isolated environment and follow its
[offline preparation guide](libero/libero/gpu/host/backends/cuda_fp32/README.md):

```bash
python -m pip install '.[cuda-fp32]'
python -m libero.gpu.build --precision fp32 \
  --build-root /path/to/common-project \
  --mathdx-root /path/to/mathdx --jobs 2
```

For native FP64 and the original single-environment API, use a separate
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'cmake>=3.24,<4' ninja
python -m pip install '.[native]'
```

Use `-e` for an editable installation of the chosen extra. CUDA compilation
and CPU scene preparation are separate: the build command needs no task
environment or visible GPU. The linked FP32 guide describes its CPU scene
catalogue and preparation commands.
The extras select incompatible CPU MuJoCo versions and must not be combined.

This distribution installs the `libero` package. Use an environment containing this installation when running your LIBERO code.

### Task resources

Task BDDL files and scene/object XML definitions are included. Meshes, textures, initial states, and demonstration datasets are stored separately. Existing LIBERO resource configurations continue to work; missing default assets and initial states are fetched into the resource cache.

To prepare assets and initial states before creating an environment:

```bash
python -m libero.libero.resource_cli assets init_states
```

Compile all declared native FP64 CUDA instances in one project:

```bash
python -m libero.gpu.build --precision fp64 --jobs 3
```

The checked-in [model declarations](libero/libero/gpu/cuda/step/instantiations/libero_models.def)
list the 13 unique dimension tuples covering all 130 tasks. CMake compiles every
instance directly, including ordinary Newton, fused substeps and the adopted
motion/force/controller-aligned paths. The build command does not instantiate
environments, reset tasks, capture graphs or run physics. World count is a
runtime argument rather than a numerical template parameter.

The default `dev` compiler profile matches the accepted fast-compile configuration.
Use `--build-mode release` or explicitly `--build-mode all` for another compiler
profile; `--ordered-reductions` includes ordered comparison products. These
choices do not discover task instances at runtime. Repeating a matching build
uses normal Ninja dependency tracking. Native Newton loads the resulting
`native_newton_nv*.so` files; it does not generate a CUDA source file or invoke
a compiler in the shipped fast FP64 path. Set `LIBERO_GPU_BUILD_MODE=release`
when loading its release profile. Research configurations retain their explicit
legacy preparation tools.

CMake is also a direct entry, without Python or MuJoCo:

```bash
cmake -S . -B /path/to/project/cmake-control/sm80-dev -G Ninja \
  -DFLASH_MATHDX_ROOT=/path/to/mathdx -DFLASH_FAST_COMPILE=ON
cmake --build /path/to/project/cmake-control/sm80-dev --parallel 3
```

The default `flash_gpu_all` target includes native FP64 and independent FP32
production products. `flash_native_batch_all` and `flash_cuda_fp32_all` select
one backend. CPU scene/resource preparation and the legacy native transport
catalogue remain separate from numerical CUDA compilation; the build command
never performs a 130-environment bootstrap. The
[task parameter table](libero/libero/gpu/data/libero_task_parameters.json)
retains the full task-to-model mapping. Initialization and ordinary runtime
loading still check their actual model and ABI requirements.

Preparation has three boundaries: CPU resources/scenes, offline CUDA products,
and load-only environment construction. The common FP32 scene catalogue is
prepared in CPU MuJoCo 2.3.7 and imported by the modern FP32 dependency
environment as described in the [FP32 guide](libero/libero/gpu/host/backends/cuda_fp32/README.md).
Native CPU scenes use `python -m libero.gpu.host.build.task_catalog --build-root ROOT`.
Native transport/observer preparation remains the existing `host.build.aot.compile_catalog`
API; it records discarded captures without advancing physics. Numerical instances
are declared by the complete model tuple and built through CMake, independently
of that transport preparation.

Select the offline target with `--arch CC` (`-DFLASH_CUDA_ARCH=CC` in CMake).
Runtime queries the requested device before selecting `smCC-dev` or `smCC-release`;
it never substitutes another architecture's product. MathDx uses the same target,
subject to the installed SDK's actual operator support. For FP32-only CMake
configuration, pass `-DFLASH_BUILD_NATIVE=OFF` to omit native-specific prelink
requirements. The native local prelink still requires NVCC 13.2; the FP32 CUDA
Graph and MathDx requirements still apply.

`num_envs` is a runtime count. Available memory, int32 array spans and actual CUDA
launch/resource limits determine capacity. `seed` controls episode RNGs;
`scene_seed=None` selects the catalogue's fixed fixture scene, and an explicit
`scene_seed` must match that prepared scene. A different episode seed or N does
not select a new numerical binary. Camera names, sizes, depth, segmentation and
object-observation choices are observation data; native preparation includes the
no-camera, camera and depth sampler implementations rather than a benchmark
camera whitelist.

New catalogues store internal paths relative to the preparation root and XML
assets relative to the configured LIBERO/robosuite asset roots. Move the complete
prepared runtime data/product set together and update the resource configuration
for external assets. Load-only execution needs CUDA runtime/driver and compatible
MuJoCo models, but no MathDx SDK, NVCC or CMake. Development pins reproduce the
reference environment; MJB engine identity and actual array/enum/model support
checks remain separate runtime requirements. Rebuild older products offline after
this update: CUDA resource exports, controller argument layout and native Newton
shape metadata changed.

## Getting Started

### Default FP32 batches

By default, `GpuBatchEnv` selects independent CUDA FP32. Explicit
`precision="fp64"` selects native FP64; `LIBERO_GPU_PRECISION=fp64` can also set
the process policy before import. The independent
`precision="fp32", physics_engine="cuda_fp32"` batch uses the
MJWarp-derived handwritten CUDA algorithms and requires neither Warp package.
The older FP32 implementation sharing native FP64 algorithm sources remains
available through `physics_engine="native"`. Select precision before importing
typed physics modules and prepare products offline. Precision is fixed per
process; products can coexist in the same CMake build. See the
[precision guide](libero/libero/gpu/PRECISION.md) for selection, storage,
validation and current scope. The performance section identifies each backend and timing scope.

Use `GpuBatchEnv` to advance multiple worlds of **one task and one shared static
model** in a GPU batch. This is separate from the compatible
`DummyVectorEnv` / `SubprocVectorEnv` interfaces.

```python
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import GpuBatchEnv

suite = benchmark.get_benchmark("libero_10")()
with GpuBatchEnv(
    suite.get_task_bddl_file_path(0),
    num_envs=8,
    init_states=suite.get_task_init_states(0),
    seed=0,
    use_camera_obs=False,
) as env:
    obs = env.reset()
    obs, rewards, successes, infos = env.step(np.zeros((8, env.action_dim)))
    env.reset(id=[0, 3])
    print(env.diagnostics())
```

Observations are dictionaries with a leading environment axis. `step` advances
all worlds; `reset` and `set_init_state` accept selected IDs in caller order.
There is no automatic reset. `set_init_state` preserves controller history,
whereas `reset` clears episode/controller/observation state. Fixture poses are
sampled once at construction and shared; reset selects movable/robot states
from the supplied bank. If no bank is supplied, the corresponding official
`.pruned_init` file is loaded. Different fixture layouts or tasks require
separate batches. Model replacement and interactive rendering are not exposed
through this entry. Set `use_camera_obs=True` for offscreen images; image rendering
and task goal evaluation are still performed at the host boundary.
### Original single-environment API

Use the same task and environment imports as upstream LIBERO. No GPU backend argument or factory registration is required.
The following original single-environment example uses the native installation
and explicitly selects FP64 before importing typed physics. For default FP32,
use the batch example below with the independent CUDA installation.

```python
import os

from libero.gpu import precision
precision.select("fp64")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task_id = 0
task = task_suite.get_task(task_id)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)
print(task.language)

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=128,
    camera_widths=128,
)
try:
    env.seed(0)
    obs = env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[0])

    for _ in range(10):
        obs, reward, done, info = env.step([0.0] * 7)
finally:
    env.close()
```

Available suite names include `libero_spatial`, `libero_object`, `libero_goal`, `libero_90`, and `libero_10`. LIBERO uses sparse task-completion rewards.

### Native FP64 implementation details

The following options and implementation details apply to explicit native
FP64 batches (`precision="fp64"`) and the legacy native FP32 comparison path.
Default independent FP32 has its own
[configuration and preparation guide](libero/libero/gpu/host/backends/cuda_fp32/README.md).

Native batches retain the low-level physics factory's diagnostic construction mode.
Full-capacity Newton input snapshots (`entry_witness`) are disabled by default:
normal execution allocates no witness buffers and launches no witness copies.
`GpuBatchEnv(..., debug=False)` also disables recovery event recording and the
diagnostic prefix comparisons and cache snapshots. The immutable-model fast path
retains its qualified epoch-based position reuse. It captures
one complete control step when needed and replays that CUDA graph, avoiding
Python source/binding validation on every physical substep. Host boundary status
packets and per-step numerical verification are disabled. `debug=True` enables
these diagnostics and uses the original Python substep path. Cold construction
validation and the physical solver's control flow, convergence and warning/reset
behavior still apply. The dedicated quality recorder also enables input snapshots.
In the default fast profile, collision processing also skips inactive-output
backup copies and computes contact parameters only for live queries and kept
contacts. `debug=True` retains the full-capacity collision diagnostics. Contact
ordering, configured capacities and overflow/error handling are unchanged.
Offline native builds use the ordinary product cache and record compiler/library
dependencies. The legacy `strict_build_receipts` argument remains accepted; it
no longer selects a separate builder or observer mode.

For source ownership, ordinary versus explicitly fused graphs, and retained
fallback paths, see the [runtime architecture index](libero/libero/gpu/ARCHITECTURE.md).
For N=2 endpoint checks and single-measurement CPU/GPU comparisons, use the
[stable validation tools](tools/validation/README.md). Historical experiment
drivers and their evidence belong in the external development workspace.

Native batch execution loads a completed offline build, by default from
`~/.cache/libero-gpu/project` (or `$LIBERO_GPU_CACHE_DIR/project`). Set `build_root`
to choose another directory, using the same value for `libero.gpu.build`.

The original `OffScreenRenderEnv` entry uses signed native products. Its offline
compatibility profile is prepared separately from the default batch catalogue:

```bash
LIBERO_GPU_PRECISION=fp64 python -m libero.gpu.build --compatibility \
  --precision fp64 --suite libero_10 --task-ids 4
```

This prepares the selected tasks without cameras and advances one control step
to load lazy modules. It shares the project's constraint sources and Warp cache,
with separate runtime owners for mutable buffers and parameters. Ordinary compatibility
construction and subsequent step/reset/observation/render calls are load-only
and report a preparation command when a signed product is missing. Explicit
offline compilation authority applies only while its scope is active; it does
not grant the constructed environment permanent permission to compile. An
existing unsigned batch build is not a signed receipt;
source or profile changes can require new offline products. Other tasks or
camera configurations require their corresponding offline preparation.
Offline preparation requires every signed product to be persisted. A full
signed cache raises `ENOSPC`; `compatibility/BUILD.json` records `preparing`,
`failed`, or `complete`, and cannot report a private, unpersisted build as ready.

FP32 training reads fixed offline scenes and CUDA products. Creating, resetting
and closing an environment does not register scenes, update an LRU or maintain
disk-cache metadata. Scene preparation takes an offline writer lock; cleanup of
prepared inputs is a separate maintenance operation outside training.
Native compatibility environments also keep model intake and replacement in
memory. Stateless native code loads from fixed signed products; the few loaders
that still need private files use an environment-owned temporary directory,
removed after a successful close. Runtime construction does not maintain a
storage catalog, model-file leases, or LRU budgets. See
[environment and offline cache lifecycle](libero/libero/gpu/ENVIRONMENT_LIFECYCLE.md)
for configuration and inspection. Historical caches and the offline compiler
project are retained separately; environment close does not delete them.
Prepared native scene models and generated constraint/Newton sources are loaded
from the instance catalog. Runtime buffers, graphs and owner bindings are created
for each environment. Missing native or Warp products raise an error before
compilation; normal batch execution does not run NVCC or NVRTC builds.
Changes to source or the requested build profile require an offline build.
`allow_compile=True` is an explicit developer option used by the offline builder.
The default profile uses Panda/OSC_POSE, seed 0, N=8 and the standard camera setup.

Device memory is budgeted from the selected GPU's **current free memory** during
construction, after existing physics allocations. The planner reserves an
estimate for subsequent solver/controller/observation allocations plus the
larger of 512 MiB or 10% of the available budget as headroom. An optional `memory_budget`
dictionary accepts `limit_bytes`, `reserve_bytes`, and `safety_fraction`; the
limit applies to allocations remaining at the planning boundary. The snapshot
is available in `env.memory_budget`. These are estimates, so subsequent CUDA
allocation failures remain possible. Index, model, contact, and kernel resource
checks remain enforced. Insufficient memory raises an error without changing N
or truncating contacts.

After the unified build, verify a prepared task without resetting or advancing it:

```bash
python -m benchmark_scripts.benchmark_gpu_batch --suite libero_10 --num-envs 8 --task-ids 0 --build-only --output results/build.json
```

The report records the project directory and actual compiler calls. This
mode still creates CUDA owners and records graphs, but executes no control steps.
`GpuBatchEnv(..., initialize=False)` provides the same preparation boundary;
call `reset()` before stepping it.

To time the complete batch interface, excluding setup, reset, and policy inference:

```bash
python -m benchmark_scripts.benchmark_gpu_batch --suite libero_10 --num-envs 8 --warmup 20 --steps 600 --repeats 1
# Add --camera for two 128x128 RGB cameras, or --task-ids 0 for one task.
```

Use the same `--build-root` for preparation and timing if specifying one.
The JSON report includes the build cache and compiler calls, setup time, milliseconds per batch control
step, environment steps per second, actual integration counts, and the budget
snapshot. These measurements include host goal/observation work and are not
device-only kernel timings. Tasks rejected by the memory preflight are recorded
with `status="insufficient_memory"`; the runner continues with the other tasks
at the requested N.

For task generation and benchmark usage, see the [upstream documentation](https://lifelong-robot-learning.github.io/LIBERO/).

## Datasets

Download the original human demonstration datasets:

```bash
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

To download a single suite:

```bash
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_spatial --use-huggingface
```

Dataset choices are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_100`. The download location follows your LIBERO dataset configuration. Datasets are also available on [Hugging Face](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).

## Training

The original LIBERO training interface is retained. For example:

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python -m libero.lifelong.main \
    seed=0 \
    benchmark_name=LIBERO_SPATIAL \
    policy=bc_transformer_policy \
    lifelong=base
```

Choose a benchmark from `LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, `LIBERO_90`, or `LIBERO_10`; a policy from `bc_rnn_policy`, `bc_transformer_policy`, or `bc_vilt_policy`; and a method from `base`, `er`, `ewc`, `packnet`, or `multitask`.

## Evaluation

Evaluate a saved checkpoint using the original evaluation command:

```bash
python -m libero.lifelong.evaluate \
    --benchmark libero_spatial \
    --task_id 0 \
    --algo base \
    --policy bc_transformer_policy \
    --seed 0 \
    --ep 50 \
    --load_task 0 \
    --device_id 0
```

Set the suite, task, seed, epoch, and training-task index to match your checkpoint. Use `--experiment_dir` to select its experiment directory.

## Configuration

| Variable | Purpose |
| --- | --- |
| `LIBERO_CONFIG_PATH` | Directory containing your LIBERO resource-path configuration. |
| `LIBERO_RESOURCE_PATH` | Cache location for automatically fetched task assets and initial states. |
| `LIBERO_GPU_CACHE_DIR` | GPU build and runtime cache directory. |
| `LIBERO_GPU_BUILD_MODE` | Native Newton compiler profile to load: `dev` (default) or `release`; must match the offline build. |
| `LIBERO_NVCC` | Path to the CUDA compiler. |
| `LIBERO_MATHDX_ROOT` | MathDx installation root containing `include/`, `external/cutlass/include/`, and `lib/libcusolverdx.a`. |

For CUDA source layout and low-level entry points, see [GPU source documentation](libero/libero/gpu/README.md).

## Tests

A small regression suite covers installation layout, resource and toolchain paths, source dependencies, vector protocols, and observation/graphics ownership. See [tests/README.md](tests/README.md) for commands and scope.

## Contributing

Development and releases use this Git repository. See [CONTRIBUTING.md](CONTRIBUTING.md) for editable installation, focused branches, CPU checks, and tagged releases. The release tool builds checked wheel/source artifacts directly from the current working tree.

## Acknowledgments

This project builds on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), developed by the Lifelong Robot Learning team. We thank the LIBERO authors and the developers of MuJoCo, libccd, robosuite, NVIDIA Warp, and NVIDIA MathDx for their work.

## License

Project-original GPU-LIBERO contributions are licensed under the **MIT License**, with copyright attributed to the GPU-LIBERO contributors. See the root [LICENSE](LICENSE) for the project license and component scope.

Upstream LIBERO-derived code retains its original [MIT license and copyright notice](licenses/LIBERO-MIT.txt). Adapted MuJoCo code retains **Apache-2.0**, and libccd code retains **BSD-3-Clause**. These third-party terms are preserved and are not replaced by the project MIT license.

See [Third-Party Notices](libero/libero/gpu/THIRD_PARTY_NOTICES.md) for source attribution and license texts. External SDKs and dependencies, including NVIDIA MathDx, retain their own licenses.
