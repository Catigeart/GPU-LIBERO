# Algorithm Configuration and Scope

Device mathematics uses handwritten CUDA. Standard matrix multiplication, factorization, and solves use cuBLASDx/cuSOLVERDx. FMA is enabled, with precise division and square root and non-FTZ behavior retained.

## Configuration API

`host.factory.prepare_cuda_physics` accepts a frozen configuration object or a nested dictionary with the same structure:

```python
from libero.gpu.host.factory import prepare_cuda_physics
from libero.gpu.host.runtime.core.algorithm_config import AlgorithmOptions

policy = AlgorithmOptions()
bundle = prepare_cuda_physics(*physics_inputs, algorithm_options=policy)
```

`AlgorithmOptions.baseline()` selects the configurable Newton, collision, and Post comparison paths; it does not restore an earlier source snapshot. `AlgorithmOptions.experimental()` requests exact Newton islands and collision temporal caching, subject to their respective admission checks. A changed algorithm or owner requires cold reassembly. Unknown fields, invalid types, and simultaneous centralized/legacy Newton configuration are rejected.

| Option | Default | Behavior |
| --- | --- | --- |
| `newton.cone_hessian_strategy` | `auto` | Rank updates for few cones; batched U/Gram accumulation and complete POTRF for larger sets. |
| `newton.cone_batch_threshold` | `None`, resolved to NV | Select using effective cone rank; explicitly configurable. |
| `newton.rank_rebuild_threshold` | `None`, resolved to max(8,NV//2) | Rebuild the base Hessian after larger active-set changes; 0 disables this choice. |
| `newton.newton_islands` | `off` | `exact` partitions the exact nonzero graph of the current lower H; no cross-iteration physical-graph cache. |
| `collision.temporal` | `False` | Request conservative fat-AABB/candidate caching. |
| `collision.absolute_padding` / `relative_padding` | 0.005 / 0.01 | Spatial padding, subject to validity and capacity admission. |
| `collision.max_age` / `max_cache_bytes` | 32 / 8 MiB | Cache age and private memory limits. |
| `post.contact_index_mode` | `auto` | Scan, radix, or automatic selection. |
| `post.radix_min_contacts` | 256 | Automatic radix also requires nbody >= 32. |
| `post.contact_topology_cache` | `True` | Reuse CSR only when ncon and every event body key match exactly. |

Thresholds remain configurable. Historical measurements include synthetic independent-function inputs and do not establish universal optimal thresholds.

## Newton

Batched cones retain the local pivot floor: U = Lc^T J and the base matrix is Lbase Lbase^T from the input factor. The base H remains available for later rank updates. Final POTRF runs after all effective cones are processed. Public upper-triangle and blocked outputs retain their separate contracts.

Exact islands rebuild the graph from the current represented matrix, retaining every nonzero coupling without epsilon pruning. Global cost, shared alpha, line search, and stopping rules are preserved. Islands remain off by default because small-capacity timings did not improve. Full-NV workspace remains allocated.

`host.runtime.newton.blocked_factor.BlockedFactorPlan` is an independently callable dense factor/direction plan: tile16, private global factor/vector storage, one CTA per world, and runtime n/ld. POTRF, right-side TRSM, trailing GEMM, and blocked vector solving run in one kernel. Its independent capacity128 scope does not establish full persistent Newton NV128 support; the complete solver still requires its Hessian, arena, and lifecycle capacity admission.

Standalone Newton defaults to 128 threads. The single-substep fused kernel uses
`FLASH_SUBSTEP_THREADS` (64 by default); these are separate launch contracts.
Neither changes the general collision/constraint configuration or forces a
vector-copy choice. NV21/43 use scalar I/O; NV57 retains its selected mathematics
and packet I/O. Small-capacity complete Cholesky remains on cuSOLVERDx.

The native FP64 fused solver retains three forwards, CPU 2.3.7 stopping,
incremental Hessian updates and all warmstart submissions. Its 64-thread
transpose auto selector uses four lanes per DOF for NV<=48 and 128..512 live
rows. Other domains retain direct ownership; larger CTAs keep their existing
shared-partial selector. Explicit direct/atomic/CSC choices remain available.
The selected body and selector are in `cuda/step/newton_transpose.cu/.cuh`.

Native FP64 line-search preparation can publish the whole Jv and then prepare
independent head-owned quadratic records with three block rendezvous. The
admitted region has 64 threads, at least 64 live rows and at most 64 heads.
It borrows the completed smooth-candidate residual plane (`efc_b`), whose next
reader belongs to the next solve; no device allocation or larger shared arena
is introduced. Nonfinite speculative products or coefficients select the
original tile path before any published Jv/quad write. Wider head sets and other
CTA sizes also retain that path. `newton_solver.cu::s07_full_products_block`
changes preparation only; candidate evaluation, alpha selection, stopping and
result publication retain the CPU rules. Regional tests and shared-device
timing do not establish a new whole-task or 130-task speedup.

## Compiled Product Selection

The maintained options live in `cmake/Substep.cmake`; both the source checkout
and installed GPU package have a CMake entry. They select products in the same
stable build tree, not a separate project per task. The 130 tasks currently use
13 dimension families; exact scene identity remains separate from those sizes.

| CMake option | Default | Meaning |
| --- | --- | --- |
| `FLASH_CUDA_ARCH` | `80` | Target SM, selected explicitly for another GPU. |
| `FLASH_MATHDX_ROOT` | empty, required | SDK containing `lib/libcusolverdx.a` and headers. |
| `FLASH_SUBSTEP_THREADS` | `64` | Threads per fused world block. |
| `FLASH_FAST_COMPILE` | `OFF` | `ON` uses minimal assembler optimization in major units; independent reduction/cache units retain full optimization. This is not a different CMake build type: both profiles use Release. |
| `FLASH_GLOBAL_DEVICE_LTO` | `OFF` | `OFF` keeps local MathDx prelink and ordinary device linking; requires NVCC >= 13.2. `ON` is a distinct whole-program LTO configuration. |
| `FLASH_NEWTON_ATOMIC_REDUCTIONS` | `ON` | Default FP64 reduction policy; explicit ordered comparison targets coexist. |
| `FLASH_NEWTON_JACOBIAN_CACHE` | `ON` | Position-generation sparse Jacobian/product/column cache. |
| `FLASH_NEWTON_TRANSPOSE_METHOD` | `0` | Auto; 1 atomic, 2 direct, 3/4/5 shared 32/16/8, 6/7/8 CSC 1/4/8. |

Python `fast_compile`, `device_lto`, `profile_newton` and `atomic_reductions`
select matching prepared products. Load-only execution rejects missing/stale
products instead of configuring, compiling or linking. Profile and ordered
comparison targets are excluded from the normal build unless requested.

## Model-Dependent Fast Paths

Fast position computation uses 128 threads for NG >= 128, otherwise 64. It uses
scalar-register vectors and omits unconsumed diagnostic stores. NV21 and NV27
retain the original component-wise quaternion divisions to control observed
error accumulation; other qualified fast dimensions use reciprocal multiplication.
The compatible profile keeps its original normalization arithmetic.

SAP uses stable register sorting up to 64 float events or 128 integer pairs.
NaNs and larger inputs retain the original insertion path. Convex private
contact storage is bounded by 5 from the retained MuJoCo MPR/MultiCCD algorithm;
public capacity remains 50, and other detector families retain their own bounds.

The qualified fused path reuses first-forward smooth outputs only until the
second forward, before controller writes. Complete Post is deferred to its
consumer boundaries (3 evaluations in a normal 25-substep control step), with
explicit `defer_post=False` and observer fallbacks. See the
[runtime index](ARCHITECTURE.md) for entry points and invalidation conditions.

## Collision and Constraints

Temporal caching requires nonnegative radii/margins, a complete static shadow, a valid position epoch, worst-case query/stage/tree-tape capacity, and a private memory budget. Matrix-norm envelopes cover mildly nonorthogonal transforms and MPR RR^T margins. The explicit-pair tail is retained each step. Overrides, model changes, and invalid input fall back to the original path. Only successful cold admission can omit the repeated PCA-bounds producer; the `temporal_front` manifest reports the actual admission reason.

Exact BoxBox deduplication uses numerical xyz equality and retains the last original ordinal. Signed zeros share a group; NaN/inf are rejected first. Active-prefix computation is separate from fixed-capacity tail writes. Sorting and first-error publication rules are preserved.

MPR portal reuse is on. `LIBERO_NATIVE_SUPPORT_REUSE` is off by default; support reuse is valid only within the same query, not across poses. Historical actual MPR samples had no support-cache hits.

S caches ordered ancestor support for body pairs. Budget overruns and structural mismatches use the original merge. Common-column algebraic cancellation is limited by finite-input guards; larger inputs preserve overflow/NaN rejection. Pair caching is on, while common-zero skipping is off. `LIBERO_CUDA_S_COMMON_ZERO=1` remains an explicit experimental choice.

Mass forests run qualifying contiguous independent roots in parallel. Within-root arithmetic order, pivot floors, warnings, and failure cleanup are preserved. Single-root and interleaved layouts use the original path.

## Post and Memory Helpers

Post caching reuses body incidence only. Forces, efc addresses, frames, and positions always come from current input. The same producer owns private CSR workspace. External CSR overrides must restore matching key/metadata or call `cuda_post_contact_cache_invalidate`. Newton finalization uses the same Post algorithm and owner.

Copy and finite helpers retain scalar defaults. Selected 342-double copies require 64 threads and both actual addresses aligned to 16 bytes. Other tested sizes and unaligned views retain scalar paths. Integer copies are unchanged.

The F position chain binds model counts during cold construction. Its 14 finite checks use `finite_lane_known<Count>`; only compile-time Count342 selects the vector route. Unconfigured dynamics/Euler callers remain scalar. Existing safe helpers handle unaligned views. No wider threshold is inferred from the limited samples.

## Scope and Default Policy

- Forest, Post auto radix/exact topology caching, and adaptive Newton Hessian selection retain their production conditions and fallbacks.
- C portal reuse remains on; support caching and temporal caching remain off by default.
- Standalone tile16 capacity128 remains independent of the complete solver.
- Negative or mixed performance domains retain the original implementation.
- Runtime binding preserves model policy, buffer ownership, explicit invalidation and error behavior. Offline builds retain dependency checks; runtime source certificates and unused provenance frameworks are being removed as described in `ENVIRONMENT_LIFECYCLE.md`.

Historical independent-function checks used FP64 error accumulation, backward residuals, and sensitivity. Indices, stable order, capacity, copy bit patterns, and protected writes retain exact contracts. Cold compilation/resource checks do not substitute for numerical execution, and local results do not establish complete application correctness or whole-step speedups. No new GPU qualification is performed by this documentation update.
