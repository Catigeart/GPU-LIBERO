# Third-Party Notices

This file records upstream source origins, adaptation scope, and external dependencies. Project-original contributions use the MIT license in the root [LICENSE](../../../LICENSE). Upstream and third-party portions retain their own terms and copyright notices; the project MIT license does not replace them.

## Component Licensing

| Component / rights holder | Use and scope | Retained terms | Source or license |
| --- | --- | --- | --- |
| LIBERO / Lifelong Robot Learning | Base environment, tasks, learning code, and inherited controller code | MIT; original Copyright 2023 Lifelong Robot Learning | [Original LIBERO license](../../../licenses/LIBERO-MIT.txt) |
| MuJoCo 2.3.7 / DeepMind Technologies Limited | Collision and mathematical helpers adapted for CUDA and this project's layouts | Apache-2.0 | [License](../../../licenses/Apache-2.0.txt), [convex collision](host/runtime/collision/native_originals/engine_collision_convex.c.txt), [spatial helpers](host/runtime/collision/native_originals/engine_util_spatial.c.txt) |
| MuJoCo box collision / Svetoslav Kolev | Native box collision and selected independent SAT translations | Apache-2.0; original Copyright 2016 Svetoslav Kolev | [Original source](host/runtime/collision/native_originals/engine_collision_box.c.txt), [SAT translation](reference/collision/box_cross/kernel.py) |
| libccd / Daniel Fiser and the attributed Czech Technical University research group | Vector, support, simplex, and MPR convex collision code | BSD-3-Clause | [License](../../../licenses/libccd-BSD-3-Clause.txt), [original license copy](host/runtime/collision/native_originals/BSD-LICENSE.txt), [MPR source](host/runtime/collision/native_originals/ccd_mpr.c.txt) |
| NVIDIA cuBLASDx / cuSOLVERDx | External device-side matrix operations, POTRF, and solves through bridges | Terms supplied with the installed MathDx SDK | [POTRF bridge](cuda/newton/potrf_dx_bridge.cu), [Hessian bridge](cuda/newton/hessian_blas_dx_bridge.cu), [OSC bridge](cuda/control/osc_bridge.cu) |
| NVIDIA Warp | External framework used by reference operators and host construction | Terms supplied with the installed warp-lang package | `reference/`, `host/assembly/`, and runtime templates |
| robosuite / MuJoCo CPU implementation | Environment dependencies and historical CPU comparison references | Their respective installed-package licenses; inherited code is identified separately | OSC adaptation below and upstream source notices |

## Adapted Device Code

`cuda/` includes both project implementations and adapted upstream source. Handwritten CUDA describes the implementation language, not exclusive authorship of every algorithm or source fragment.

- [native_source.cuh](cuda/collision/native/native_source.cuh) contains CUDA device adaptations of retained MuJoCo/libccd C source. The historical generator named in its header is archived outside this distribution. The maintained source is compiled directly. Its geometric source correction is retained in [box_contact_source.py](host/build/box_contact_source.py) and checked against the CUDA body; this helper does not regenerate production sources.
- [collision_detect_primitive.cuh](cuda/collision/collision_detect_primitive.cuh) directly adapts MuJoCo scalar collision mathematics.
- CUDA fusion, layouts, thread organization, host bindings, caching, and restoration add project implementation work while retaining upstream attribution for inherited portions.
- MathDx bridges call external libraries; they do not include or relicense the SDK's internal BLAS/LAPACK implementation.

Original source and notices are retained in [native_originals/](host/runtime/collision/native_originals/). Combined native files may repeat notices because they incorporate multiple upstream files. These source-level notices remain applicable.

## OSC Adaptation

The fixed-size LU inverse, one-sided Jacobi SVD, FP32 quaternion conversion, OSC equations, and selected reset logic in `reference/control/osc/controller.py` were adapted from a historical LIBERO GPU controller at SHA256 `98711c69d24a4d6a1c5d96d1a0018c6f8cb897cbc5b3370db245c2ea654a099c`.

That historical source was MIT licensed under the LIBERO repository license, Copyright 2023 Lifelong Robot Learning. The original notice is preserved in [LIBERO-MIT.txt](../../../licenses/LIBERO-MIT.txt). No historical repository is imported at runtime.

The adapted implementation adds physical-input and controller-cache buffers, device-side cache selection, per-world policy/configuration, intermediate outputs, and capacity checks. The historical CPU comparison reference was robosuite 1.4.0; reference results do not supply production GPU state.

## Included License Texts

- [Project MIT license and component scope](../../../LICENSE): project-original Flash-LIBERO contributions, Copyright 2026 Flash-LIBERO contributors.
- [Original LIBERO MIT license](../../../licenses/LIBERO-MIT.txt): unchanged upstream text and attribution.
- [Apache License 2.0](../../../licenses/Apache-2.0.txt): retained from the installed MuJoCo 2.3.7 distribution.
- [libccd BSD-3-Clause license](../../../licenses/libccd-BSD-3-Clause.txt): unchanged original text, conditions, and disclaimer.

The original Apache source notices identify DeepMind Technologies Limited and, for box collision, Svetoslav Kolev. Source-level copyright and modification notices are retained. SDK and framework licenses are supplied by their external distributions rather than this repository's MIT license.

## Maintenance

When adding or adapting upstream source, record its origin, affected code, and applicable terms. Preserve required copyright, license, disclaimer, and modification notices in source and generated output. Keep project-original additions distinguishable from upstream portions. This inventory documents retained terms and does not replace file-level notices or external licenses.
