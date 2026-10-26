# Tests

This source checkout includes a small regression suite for installation and public interface maintenance.

| Test | Scope |
| --- | --- |
| `test_installation_layout.py` | Original public packages, canonical GPU namespace, compiler sources, catalogs, and templates. |
| `test_resource_paths.py` | Configuration compatibility, resource caching, mocked downloads, and the resource CLI. |
| `test_product_tool_paths.py` | Compiler and SDK path selection, private child configuration, and build-directory handling. |
| `test_native_capsule_sources.py` | Native source/header dependency closure, source generation, drift, and path confinement. |
| `test_native_box_contacts.py` | Host replay of the maintained BoxBox body: task-5 failure pose, projection determinant, margin contacts, transforms, and random geometric witnesses. Requires MuJoCo and g++. |
| `test_gpu_vector_protocol.py` | Vector protocol, cleanup, shared buffers, grouped workers, and per-world RNG using fake environments. |
| `test_gpu_memory_budget.py` | Adaptive free-memory snapshots, explicit limits, pressure failures, budgets above 6 GiB, and int32 bounds. |
| `test_gpu_batch_env.py` | Single batch dispatch, ordered outputs, failure quarantine, and selected-world reset using injected physics. |
| `test_gpu_collision_dataflow.py` | Formal collision uses live routes/rows while preserving physical copies; debug retains diagnostics. Immutable pair tables initialize once per model; mutable and large models use direct mixing. |
| `test_deferred_native_build.py` | Deferred reference compilation, preserved source/PTX rejection, and lazy reset graph capture. |
| `test_gpu_build_cache.py` | Persistent build reuse, stable generated namespaces, changed source/flags, tampered artifacts, and verified migration of prior builds. |
| `test_gpu_build_lifetime.py` | Current temporary runtime ownership, constructor failure, close retries, and independent replacement generations; does not use the retired storage manager. |
| `test_gpu_compatibility_build.py` | Offline/load-only boundaries, fixed build identity, private artifact cleanup, library/function ownership, and launcher retirement. |
| `test_gpu_cache_hardening.py` | Ordinary product files, legacy read-only reuse, budgets, offline dependency invalidation and runtime ABI checks. |
| `test_fp64_model_intake.py` | Direct model/metadata handoff without repeated source reads or MJB serialization; array, dimension, domain and external-file checks remain. |
| `test_portable_intake_management.py` | MuJoCo 2.3.7 model/package binding and updates without decoder/source certificates or producer authentication; actual model changes, callbacks, parameter permissions, owner replacement and commit/invalidation contracts remain checked. Skips on other MuJoCo versions. |
| `test_collision_load_only.py` | Native collision loading without compiler/image/SDK fingerprint checks or identity files; ABI, CUDA query/launch errors, cache lifetime and legacy recipe selection remain. Uses simulated CUDA exports; no physics execution. |
| `test_newton_load_only.py` | Offline Newton build versus runtime selection, loaded-kernel ABI/resource query, no runtime SDK/compiler/source checks, missing or mismatched products and failed publication. Uses simulated compiler/CUDA exports. |
| `test_collision_model_binding.py` | Resident collision static/context checks without source scans or file leases; loaded export, parameter, buffer, stream and close contracts, plus descriptive decoder intake. Host-only fixtures. |
| `test_fp32_load_only.py` | CUDA runtime/fused bindings without source certificates; portable scene data, host camera/lifecycle option sharing, and read-only legacy scene reuse. |
| `test_gpu_capability_registry.py` | In-memory domain/capacity/rejection checks without development files; explicit offline qualification still checks provenance. |
| `test_gpu_environment_lifecycle.py` | Read-only scene/project loads, partial close retry, shared stateless code, and in-memory model/Newton intake. |
| `test_gpu_substep_build.py` | Stable CMake build directories across single-substep source edits and load-only runtime boundaries. |
| `test_cuda_fp32_bridges.py` | Stream-owned library reuse, independent streams and products, ABI failures and CUDA error propagation. |
| `test_mjwarp_products.py` | Reference component selection without Ninja; ABI, architecture and symbol checks remain at binding; source files are not read. |
| `test_gpu_goal_read_plan.py` | Built-in field hints, changed callbacks, lazy same-boundary reads, contacts, and visual side effects. |
| `test_gpu_shape_assembly.py`, `test_gpu_startup.py` | Explicit FP64 constructor selection, startup behavior, failure cleanup and retained compatibility entries. |
| `test_gpu_controller_copy_plan.py` | Cold all-world copy packets, odd tails, private gate wiring and snapshot/field alias fallback. |
| `test_gpu_aot_build.py` | Complete task/shape instance coverage; runtime template loading checks model/presence without file hashes, while offline indexing rejects stale sources; cache misses and Warp worker compilation guards. |
| `test_gpu_batch_build.py` | Checked-in CUDA instances cover all 130 task shapes; CMake-only builds need no physics imports or environments; backend/profile routing, compiler failure propagation and scoped offline compilation permission. Host-only fixtures. |
| `test_gpu_ready_session.py` | Task-local readiness and compatibility sessions without source or whole-bank hashing. |
| `test_gpu_observer_graph_mutations.py`, `test_gpu_batch_direct_observations.py` | Public observable changes, callback fallback, cold Graph binding checks and explicit lazy rendering. |
| `gpu_product_compatibility/test_camera_contract.py` | Camera keys, shapes, segmentation mapping, and observation clocks using an injected bridge. |
| `gpu_product_compatibility/test_observable_extensions.py` | Original Observable callbacks, filters, clocks, and owner transfer using a fake facade. |
| `gpu_product_compatibility/test_render_model_owner.py` | Isolation of graphics model storage using an injected renderer. |

## Running

`flash_post_demand_test` is an excluded CMake CUDA target. It checks two-world
Post deferral with current/stale Newton stamps, held Post publication stamps,
working-qacc publication and inactive full calls at 32/64/128 threads and four
vector widths. Build and run it from an existing configured GPU build directory.

`flash_osc_reference_test` is another excluded CUDA target, sharing the ordinary
OSC and MathDx objects. Generate its fixture with `cuda/osc_reference_fixture.py`
in the CPU reference Python environment, then pass that file to the executable.
The generator calls the installed robosuite operational-space and nullspace
functions. Four two-world cases cover full rank, rank deficiency, scaled/zero
Jacobians, exact-input cache hits, and a mass change in only one world. The test
compares matrices, ranks, goals and torques with the CPU results; it performs no
physics rollout or performance measurement. Keep generated fixtures outside
the source checkout.

`flash_controller_pipeline_test` compares the cooperative controller pipeline
with its retained ordered path, using two worlds at NV=21/39/57. Fourteen
scenarios run twice each: normal and held cache, mixed inactive worlds, stale
producer/parent stamps, invalid Jacobian, existing errors, invalid cached input,
counter boundary/overflow and singular mass. It compares payloads, diagnostics,
first errors and publication stamps. Build/run this excluded CUDA target
separately; it performs no performance measurement.

Run from the repository root after installing the project and its dependencies:

```bash
python tools/check_source.py --cpu-tests
python -m unittest discover -s tests/gpu_product_compatibility -p 'test_*.py'
python tests/test_gpu_vector_protocol.py
```

The supported CPU check command also runs the vector protocol file. Its standalone
command is useful for focused work; those test functions are not collected by
unittest discovery. Individual optional suites retain their dependency requirements.

CI installs `tools/requirements-ci.txt`, including robosuite's original host
interfaces, MuJoCo 2.3.7 for native model intake, and Warp host types. The workflow
sets `MUJOCO_GL=glx` and hides CUDA devices. robosuite 1.4.0 overrides `disable`
with `egl`; `glx` instead selects its GLFW import path without loading EGL.
These tests create no GL context and need no display server or GPU.

These tests use temporary fixtures and mock or injected runtime components. They do not compile CUDA, execute GPU physics, render through a real OpenGL context, or benchmark performance. They do not establish full-application numerical qualification.

The BoxBox tests compile the maintained contact function for host execution and use MuJoCo's double-precision vector helpers. The captured regression fixture contains only the two boxes' sizes and transforms.

Historical algorithm experiments, evidence capture, and private data fixtures remain in the development workspace and are not part of this source release.

## FP32 regression scope

`test_precision_policy.py` checks the FP32 process default, explicit FP64 and
environment overrides, scalar ABI layouts and process precision freezing.
`test_mjwarp_frontend.py` checks default independent-CUDA routing and explicit
native/reference selection without allocating a GPU world.

The `test_mjwarp_*.py` files retain host contracts and independent mathematical
oracles for the shared handwritten CUDA leaves. Some require the isolated
MJWarp reference environment and skip when its optional dependencies are
absent; these do not become independent FP32 runtime dependencies. Run these
pytest-style tests with `python -m pytest tests/test_mjwarp_*.py`.

`data/gjk_tuned_direction.json` is a minimal real geometric counterexample,
not a trajectory snapshot. `test_mjwarp_convex_gjk.py` independently proves the
old separating-plane claim invalid. The maintained `mjwarp-convex-regression`
command additionally exercises the compiled production headers on this input
and deeper, separated and translated controls; see the validation protocol.
One-off multi-owner attribution and policy-counterfactual tools live in the
external development archive, with their original dependency closure.

## Maintained validation entry

`python tools/validate_gpu.py` provides explicit `run`, `compare` and `summarize`
commands. See [its protocol and examples](../tools/validation/README.md).
It shares action generation, task selection, frozen-old error budgets and timing
statistics across modules. N=2 is the default; optional N=1 diagnosis is limited
to N=2 numerical/work failures with existing matching references. It never
compiles products or uses dated experiment scripts.

`test_release_inputs.py` checks the independent CMake staging closure;
`test_gpu_substep_build.py` covers the installed CMake entry and stable load-only
build directory; `test_validation_tools.py` covers frozen-reference envelopes,
input selection and world-control-step throughput. These are CPU-only checks.

`python tools/runtime_inventory.py --output <external-file.json>` writes a
static module/import/source-composition inventory. Its unreached modules must
not be treated as proven unused: cold assembly also uses callbacks and derived
source templates. The human-maintained path index is
[ARCHITECTURE.md](../libero/libero/gpu/ARCHITECTURE.md).

The optional `flash_collision_tree_test` CMake target runs CUDA component tests
for ordered parallel BVH requests, explicit-pair suppression, tiny and inactive
inputs, and capacity/error fallbacks. It uses two worlds and 32, 64 and 128
threads. Build that target in an existing configured GPU build directory and
run its executable separately from the Python tests.
The cases include rotated frames, fully and partially infinite bounds, and
separated bodies, in addition to ordered overflow and explicit-pair handling.

Precision regressions build from the production CUDA bodies in both scalar configurations.
`precision_gram_test.cu` checks two-world dense/sparse reductions, the cache and
fallback, and shared-loan redzones. `tests/support/box_contact_source.py` retains
the upstream source derivation solely for tests; it does not generate production
code. The derivation test checks the edge tail and first-vertex clip statements;
BoxBox geometric fixtures cover touching, swapped boxes, SAT ties and scales.

The optional `flash_newton_replay_probe` and `flash_newton_replay_probe_fp32`
CMake shared-library targets expose diagnostic entries for the NV=39 normal
atomic-reduction configuration. They link the production, separately compiled
Newton object. A caller supplies retained device arguments and buffers; the
Newton-only entry holds geometry, constraints and smooth dynamics fixed, and
the forward entry returns before control or integration. The forward fixture
includes the maintained physical-substep source in its own translation unit to
access the actual forward function. Neither target is a normal product
dependency, and neither adds diagnostics to the production execution path.
The FP64 `flash_newton_replay_probe_nv21` and `_nv57` targets expose the same
Newton-only entries for small and large catalogue shapes; they omit the
forward source and its controller/Post link inputs.
The entries are declared in `cuda/newton_replay_probe.cu`; their return codes
report CUDA launch errors, while the output records contain solver status.
`flash_newton_replay_launch_entry_policy` selects the initial-gradient test
explicitly, so both policies can be compared using one numerical device object.

Two optional FP64 CUDA fixtures cover native Newton product organization:

- `flash_newton_transpose_test` compares the actual production reductions and
  their immediate gradient consumer with a long-double reference. It covers
  all 13 catalogue NV values, two worlds, row/CTA boundaries, masked NaNs,
  inactive holds and output redzones.
- `flash_newton_linesearch_products_test` compares the actual full-row and
  retained tile preparations in two worlds. It covers three-, four- and
  six-dimensional cone records, cross-tile groups, head-count fallback, Jv
  NaNs and coefficient overflow. The exceptional output prefixes are compared
  with the retained path. This NV39 fixture includes the maintained solver
  source solely to reach its private preparation functions; normal products
  keep their separately compiled Newton object.

Build these targets in the existing CMake tree and execute the resulting
binaries. They neither create an environment nor integrate a physical step.
The line-search fixture accepts `--timing` for its optional regional comparison;
the default invocation checks correctness only.
Any reported timing is regional; it is not full-task throughput.

The optional `flash_boxbox_replay_probe` and `_fp32` shared-library targets
accept batches of frozen sizes, positions, rotation matrices and margins.
They call the maintained BoxBox body using scalar or 8/16/32-lane cooperation
and export contact geometry and the selected axis. Their ABI is declared in
`cuda/boxbox_replay_probe.cu`; neither target is a normal product dependency.
