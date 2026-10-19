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
| `test_gpu_substep_build.py` | Stable CMake build directories across single-substep source edits and load-only runtime boundaries. |
| `test_gpu_controller_copy_plan.py` | Cold all-world copy packets, odd tails, private gate wiring and snapshot/field alias fallback. |
| `test_gpu_aot_build.py` | Complete task/shape instance coverage, runtime template loading, rejected cache misses, and compilation guards in Warp worker threads. |
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
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s tests/gpu_product_compatibility -p 'test_*.py'
python tests/test_gpu_vector_protocol.py
```

The vector protocol file exposes standalone test functions, so run its final command separately from unittest discovery.

These tests use temporary fixtures and mock or injected runtime components. They do not compile CUDA, execute GPU physics, render through a real OpenGL context, or benchmark performance. They do not establish full-application numerical qualification.

The BoxBox tests compile the maintained contact function for host execution and use MuJoCo's double-precision vector helpers. The captured regression fixture contains only the two boxes' sizes and transforms.

Historical algorithm experiments, evidence capture, and private data fixtures remain in the development workspace and are not part of this source release.

The optional `flash_collision_tree_test` CMake target runs CUDA component tests
for ordered parallel BVH requests, explicit-pair suppression, tiny and inactive
inputs, and capacity/error fallbacks. It uses two worlds and 32, 64 and 128
threads. Build that target in an existing configured GPU build directory and
run its executable separately from the Python tests.
The cases include rotated frames, fully and partially infinite bounds, and
separated bodies, in addition to ordered overflow and explicit-pair handling.
