# Independent FP32 qualification

The 2026-10-21 campaign qualifies the independent resource owner and execution
path for the shipped rigid LIBERO batch domain. It is not a claim that every
trajectory is identical to unmodified MJWarp or CPU MuJoCo 2.3.7.

**Revision scope:** the full-suite numbers and installed-package evidence below
precede the ND-05 follow-up described at the end. The current convex library
contains that later correction; no new 130-task numerical or timing total is
inferred from its targeted checks. The subsequent world-local fusion has
[regional N=2 endpoint and local timing coverage](README.md#supported-scope-and-validation),
without another full-task/130-task run. The historical counts below describe
the qualified revision, not a rerun of every later fusion target.

## Dependency and integration results

- A genuinely isolated Python environment cannot find either `warp` or
  `mujoco_warp`. It prepares all 130 modern CPU scenes and executes all 130 tasks.
- CPU MuJoCo 2.3.7 catalogue preparation also has a Warp-free entry; a cold
  scene was prepared from an empty project, then imported with CPU MuJoCo 3.16.
- All 411 model arrays, scalar/options and mass layouts agree for all 130 tasks.
  A second comparison checks the separately compiled isolated-environment MJBs,
  rather than reusing the reference MJB; all 130 also agree.
- Native buffer/stream/Graph/conditional-loop checks pass. No-camera lifecycle
  checks pass 21 contracts; camera lifecycle checks pass 24, including partial
  reset, continuation, RGB/depth, public `sim.render` and custom observables.
- An installed wheel runs the public N=2 API under isolated Python, importing
  LIBERO code exclusively from site-packages. Both Warp packages are absent.
- Explicit offline preparation builds the 31 admitted CMake targets with
  `ninja: no work to do`, advancing zero physics substeps. The 29 existing
  mathematical CUDA libraries are reused. Runtime compilation is forbidden.

## Numerical results and retained exceptions

Four representative N=2 frozen-input comparisons cover position/velocity
prefixes, OSC transport, the same Newton QP and Euler. Five additional cases
freeze the standard 25-control endpoint. Their largest isolated-stage error is
below 2e-7. The additional tests rebuild collision and constraints: contact/row
counts agree, permutation-independent constraint matrices differ by at most
7.11e-15, and a complete substep using that rebuilt prefix differs by at most
4.99e-10 in the compared state/controller outputs.

All 130 tasks execute N=2, five warmup controls followed by 20 continuous
controls, once, without cameras or runtime compilation. Every task passes the
public observation/predicate boundary and commits 625 physical substeps per
world. The original fixed-roundoff comparison against the corrected CUDA
reference passes 100/130; the existing upstream FP32 trajectory policy passes
127/130. Neither policy was loosened.

The three upstream-policy exceptions also fail for the previous corrected
CUDA reference in this campaign:

| Task | Dominant free-body angular-velocity difference from primary upstream | Previous corrected CUDA difference |
| --- | ---: | ---: |
| LIBERO-Spatial 5 | 0.10968 rad/s | 0.10570 rad/s |
| LIBERO-90:73 | 1.648e-4 rad/s | 1.410e-4 rad/s |
| LIBERO-90:75 | 1.301e-4 rad/s | 1.914e-4 rad/s |

These are not newly introduced upstream-policy exceptions. The prior corrected
reference passes 125/130; its additional exceptions 90:63 and 10:6 pass in the
independent run. This is not evidence of a uniform accuracy improvement: the
same mathematical CUDA leaves are used, and contact/active-set sensitivity can
change endpoints. The raw failures remain failures. Deliberate departures from
upstream are documented in [NUMERICAL_DEVIATIONS.md](../mjwarp/NUMERICAL_DEVIATIONS.md);
this campaign does not attribute every residual difference to those corrections.

An additional untimed corrected-reference repeat passes the unchanged
reference-spread comparison in 4/5 targeted cases. In 90:75, the final maximum
qvel difference is 3.215e-4 rad/s against a 1.221e-4 floor; that check remains
failed. It concerns the resting mug's angular velocity; its qpos-coordinate
difference from upstream is below 2e-6. Frozen collision/constraint invariants
and the coupled substep agree in this case. These observations support local
equivalence and the retained FP32 error scale, not a proof about every earlier
substep or every future trajectory.

The sensitive 90:79 and 90:84 additionally run N=2, five warmup controls and 600
continuous controls each. Both pass the existing corrected-reference and
upstream endpoint policies, committing 15125 physical substeps per world.
No extra performance repeats are taken.

## Performance observations

On the shared A100 40 GB, N=2, no cameras, one 5+20 measurement per task:

| Suite | Tasks | Independent GPU FP32 env steps/s | Original MJWarp FP32 env steps/s |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 10 | 58.45 | 49.25 |
| LIBERO-Object | 10 | 93.03 | 53.08 |
| LIBERO-Goal | 10 | 83.90 | 39.40 |
| LIBERO-90 | 90 | 83.04 | 58.90 |
| LIBERO-10 | 10 | 78.24 | 56.29 |
| All distinct tasks | 130 | 80.78 | 55.30 |

Throughput uses summed elapsed times for equal work and counts control steps
summed over both worlds; each control contains 25 physical substeps. The
interval includes action upload and the device control Graph, including GPU
observation sampling. Creation, reset, capture, host packaging, predicates and
rendering are excluded. The root README shows these saved results alongside
older FP64/CPU public-step wall times, with the timing scopes explicitly
separated. Other GPU work was resident; these are shared-device observations,
not isolated performance certifications or causal measurements of an
individual change. These full-suite measurements precede ND-05.

Full per-task records, source/product hashes, preserved failures, interrupted
disk-full evidence and detailed reports are kept in the development workspace's
`results/warpfree_fp32_20261021/`, outside the project tree. The final factory
docstring update has a separate executable-AST equivalence proof; it does not
rewrite the pre-documentation scientific source bindings.

## Three-case causal follow-up and ND-05

The follow-up records every physical substep of Spatial:5,90:73,90:75, N=2,
five warmup and20 continuous controls, across raw MJWarp, the prior corrected
CUDA owner and the independent owner. Complete control-boundary checkpoints
recover selected first-growth and endpoint inputs; recovery drift is retained.
Frozen contact/current-QP/downstream/Euler substitutions isolate contact
sensitivity. For the mug cases, common-QP angular-velocity differences are
below5e-10rad/s; changing collision alone produces1e-4-scale differences.
Independent contact-surface witness residuals are approximately3.2e-8m for
both engines. Original-MJWarp one-ULP target-height perturbations produce
comparable or larger local responses. These support comparable FP32 numerical
accuracy, not a finding that the independent trajectory is globally better.

A real Spatial:5 BOX overlap exposes an inherited invalid GJK separation
certificate: a point queried in a tuned direction was tested against its
preceding direction. The current CUDA leaf repairs that mismatch as
[ND-05](../mjwarp/NUMERICAL_DEVIATIONS.md#nd-05-use-the-actual-support-plane-for-a-tuned-gjk-cutoff).
The real input, deeper-overlap, separated and translated probes pass. Actual
CPU2.3.7 and independent FP64 SAT agree on all1288 candidate occurrences in
the selected states. Corrected CUDA has zero beyond-tolerance detection misses;
raw MJWarp has seven, all in Spatial:5. This is local detection evidence,
not a proof of globally superior trajectories.

Relevant224-query independent geometry,12GJK,14EPA,6manifold and22production
ABI/capacity regressions pass; original mathematical/conformance failures are
retained separately. Only `flash_mjwarp_convex_fp32` was rebuilt. Three additional
independent N=2 5+20 rollouts complete with no runtime compilation; all three
still fail the unchanged raw endpoint policy. No performance run, full-suite
rerun or package rebuild was performed for this follow-up.

Source-bound reports, baseline/source/product archives and preserved failed
statuses are in the development workspace's `results/fp32_attribution_20261021/`.

## Stable source checkpoint

The process default is now FP32, with automatic `GpuBatchEnv` selection of this
independent owner. Explicit FP64 and `LIBERO_GPU_PRECISION=fp64` retain native
selection. Source cleanup archives one-off causal experiments outside the
repository, retaining reusable qualifiers and the tuned-direction geometry
fixture as `mjwarp-convex-regression`. The CUDA mathematics remains the ND-05
follow-up revision described above. Packaging and host regression checks do
not imply a new full-suite numerical or performance measurement. The README's
four-backend figure is regenerated exclusively from saved measurements, with
their timing boundaries, source hashes and pre-ND-05 scope preserved.
