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
| `libero.libero.envs.GpuBatchEnv` | Native worlds of one task and one exact static model; equal dimensions do not permit mixing models. | `host/adapters/environment/batch_env.py::GpuBatchEnv` |
| Original environment / vector APIs | Preserve upstream signatures, routing and worker lifecycle. | Parent `envs/env_wrapper.py`, `envs/venv.py` |
| Batch physical arena | Stack reset states and construct one shared model with world-major dynamic owners. | `host/adapters/environment/batched_physics.py::prepare_native_batch` |
| CUDA composition | Batch factory selects optimized fast dataflow with cold source/owner/capacity admission. | `host/factory.py::prepare_cuda_physics` → `host/assembly/components.py::prepare_cuda_physics` → `host/assembly/mathdx/factory.py` and pipeline/compose templates |
| Facade components | Typed cleanup, final events and configured recovery; `r02` remains active. | `host/factory.py::create` → `host/runtime/step/recovery_cleanup_factory_r02.py::create` |
| Collision model intake | Extract live MuJoCo arrays without private test XML or CPU contact outputs. | `model/collision_static.py::static_from_model` |
| Capsule extension | Cold source closure and route extension before binding/launch, with independent source identity. | `host/runtime/collision/capsule_extension.py::prepare_workspace_extension` |

## Control-step execution

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
| Host/debug loop | Debug or observer needing host callbacks; captured borrows are discarded. | `NativeBatchedPhysics.enqueue_action`; `host/runtime/step/schedule.py` |
| Position refresh | State/generation invalidation or suspended prefix; rebuild before resuming. | `FusedSubstep.enqueue_substep`; `cuda/runtime/position_reuse.cuh` |
| Velocity/reference/bias reuse | Same physical substep, state epoch and position generation. New substep or prefix reentry invalidates readiness. | `cuda/step/fused_substep.cu::forward_dynamics` |
| Smooth reuse | Successful first→second forward, qualified owners and no writer; third forward computes again. | `forward_dynamics`; `host/runtime/step/substep_binding.py::_validate_smooth_reuse` |
| Deferred Post | Exact built-in graph-eligible observer and `defer_post=True`; policy samples and last-substep pre-Euler output remain. Unknown/subclass observers retain the complete publication path. | `NativeBatchedPhysics.prepare_fused_substep_graph`; `forward_dynamics` |
| Newton | Each admitted forward, warmstart and final publication; separate compilation unit. | `cuda/step/newton_solver.cu::solve_newton` |
| Physical reset | Checks, continuation, root restoration and full-forward publication. | `host/runtime/step/recovery_final_event_factory.py`; `recovery_cleanup_environment_r02.py`; `cuda/runtime/recovery.cuh` |

Normal qualified fused execution performs 75 Newton solves, 50 factored mass
solves and 3 complete Post evaluations per control step. Recovery and alternative
observer/debug policies can do additional work.

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
