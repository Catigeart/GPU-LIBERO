# Intentional numerical deviations from pinned MJWarp

This register records maintained changes to the reference mathematics or its
numerical decisions, beyond translating the implementation to CUDA. The
reference is MJWarp commit `c384ac51df67eb92ab7657cc472ff7494dc036bf`, MuJoCo
3.16.0 and Warp 1.18.0, bound by [the source pin](../../../data/mjwarp_source_pin.json).
Statements about inherited behavior refer to that version, not every MJWarp
release. The installed upstream package and its default path remain unchanged.

The corrections below are active inside the selected CUDA convex leaf,
`convex_collision="cuda"`, including the explicit `preset="cuda_libero"`.
They are not independently selectable public options. The independent
`cuda_fp32` backend uses these same shared leaves; the legacy `native`
FP32/FP64 algorithms remain separate. The [migration map](MIGRATION.md) records the
remaining leaves, fallback domains and complete-path qualification limits.

## Register

| ID | Original behavior | Maintained change | Affected domain |
|---|---|---|---|
| ND-01 | Expanded absolute-coordinate triangle minors and areas suffer cancellation | Compute algebraically equivalent determinants from translated edge vectors | Ordered simplex projection and EPA affine weights using the shared helpers |
| ND-02 | Zero box support components return an interior coordinate but a corner feature ID | Choose the negative extreme at an exactly zero component, matching the ID | BOX support, including downstream GJK/EPA and feature recovery |
| ND-03 | Positive closest-simplex distance can end penetration search without proof of separation | One witness-seeded GJK refinement, followed by a certified initial-hull EPA attempt | Inconclusive zero-margin discrete BOX/BOX with distance above tolerance, simplex dimension at least two and no separating certificate |
| ND-04 | Triangle EPA initialization rejects squared face distances below `1e-10` | Retain positive nondegenerate shallow faces only with complete initial-hull origin enclosure | Zero-margin BOX/BOX triangle initialization, including ordinary below-tolerance GJK results and ND-03 retries |
| ND-05 | Discrete GJK can test a tuned support point against its preceding, different search direction | Certify a proposed tuned cutoff exit in the actual support-query direction | Discrete GJK, norm below `1e-4`, segment/triangle simplex, proposed early separation |

These changes can alter contacts, manifold counts and trajectories relative to
the pinned implementation. Raw agreement and independent mathematical validity
are separate results. An upstream failure is retained as a failure, rather than
being used to weaken the independent oracle.

## ND-01: translated triangle determinants

**Original.** `collision_gjk.py::_S2D` and the affine-weight calculation expand
triangle determinants into absolute-coordinate products. Cancellation can
produce inconsistent barycentric weights and witnesses outside the support
hull. A retained original GJK case reconstructed a weight sum near 0.98778;
the translated CUDA expression gave approximately 0.99999996.

**Change.** [mjwarp_convex_simplex.cuh](../../../cuda/collision/mjwarp_convex_simplex.cuh)
implements `simplex_minor` using `u=b-a`, `v=c-a` and
`u[i]*v[j]-u[j]*v[i]`; `simplex_area` translates about the query point.
`simplex_triangle` and
[triangle_affine](../../../cuda/collision/mjwarp_convex_epa.cuh) share these
helpers. This preserves the real-arithmetic formula and vertex-order contract,
but intentionally changes floating-point evaluation.

**Evidence and limits.** The `mjwarp-convex-simplex` qualifier checks independent
FP64 convex-QP/KKT and witness residuals. GJK qualification retains the original
infeasible witnesses separately. This addresses cancellation in these helpers;
it is not a guarantee for every degenerate simplex or every collision pair.

Workspace evidence: `results/mjwarp_convex_20261021/gjk_loop/REPORT.txt`,
`upstream-mathematical-observations.json` and `upstream-triangle-failure/`.

## ND-02: box support point and feature ID consistency

**Original.** `collision_gjk.py::support` uses `sign(local_direction)*size` for
BOX. At an exactly zero direction component it returns coordinate zero, while
the corresponding vertex bit denotes the negative extreme. The support
projection is valid, but the point does not equal its indexed corner. EPA
feature recovery then interprets that ID as an actual corner.

**Change.** [mjwarp_convex_geometry.cuh](../../../cuda/collision/mjwarp_convex_geometry.cuh)
`convex_support` chooses `+size` for a positive component and `-size` otherwise.
At zero, the chosen extreme has the same maximum support projection and agrees
with the feature ID. Nonzero components retain their original choice; no
direction tolerance or approximate-zero test is introduced.

**Evidence and limits.** The support qualifier checks the independent support
maximum and additionally requires the returned BOX point to equal its indexed
corner. [The support tests](../../../../../../tests/test_mjwarp_convex_support.py)
include an interior-point counterexample that passes the support maximum but
fails the indexed-corner contract. A different legal support point can change
later GJK/EPA choices; equality to the original contact set is not promised.

## ND-03: inconclusive GJK results continue to penetration search

**Original.** After GJK, `collision_gjk.py::gjk_phase` can end the search when
the closest-simplex distance exceeds CCD tolerance, even when the final
support-plane test does not establish separation. A point on the current
simplex gives an upper distance bound. Its positive distance alone does not
prove that the complete shapes are disjoint. Actual shallow overlapping boxes
were missed in this path.

**Change.** [mjwarp_convex_epa.cuh](../../../cuda/collision/mjwarp_convex_epa.cuh)
`gjk_phase` marks only the ND-03 domain listed in the register for retry. One
GJK refinement starts from the current witness pair. A separating certificate,
or a valid smaller simplex-distance bound, replaces the preceding result.
The inconclusive result can then seed EPA, subject to the initial-hull
certificate in `epa_encloses_origin`. Conclusive separation remains an exit.
Other shape families, margins and the iteration limit retain their paths.

**Evidence and limits.** On retained inputs, actual CPU MuJoCo 2.3.7 collision
and independent FP64 SAT on transformed box corners confirm overlap. Frozen
phase probes isolate the failed transition and the recovered negative depth.
The correction does not assert that every inconclusive GJK result can produce
a suitable EPA hull. Degenerate or non-enclosing hulls remain rejected.

Workspace evidence: `results/mjwarp_velocity_20261021/goal-phase/`,
`goal-forced-epa/`, `cpu-box-oracle.json` and `certified-local/certified/`.

## ND-04: shallow initial faces are not geometric degeneracy

**Original.** `collision_gjk.py::_polytope3` compares `_attach_face`'s squared
distance with `MIN_DIST3=1e-10`. This rejects initial faces closer than
`sqrt(1e-10) = 1e-5 m`, or 10 micrometres. Retained valid enclosing hulls
with approximately 7–8 micrometre face distances were rejected. This is a
fixed geometric cutoff, not FP32's representable absolute resolution.

**Change.** For zero-margin BOX/BOX, `polytope_triangle` retains faces with
positive squared distance instead of applying that cutoff. If any face uses
this exception, the initializer sets `require_enclosing` for its caller;
`epa_encloses_origin` then checks every initial face half-space in FP64 on the
actual rounded FP32 hull vertices before EPA expansion. ND-03 retries already
require the same certificate. Degenerate and non-enclosing initial hulls are
rejected. This certificate is an algorithm prerequisite, not a debug-only
check. Ordinary hulls above the cutoff avoid the additional predicate. Other
shape families retain their inherited initializer rules.

The first revision applied this exception only to ND-03 retries. Exact
substep replay later exposed the same rejection in ordinary below-tolerance
GJK results. The maintained revision covers both entrances; extending the
domain does not change the GJK retry rule or the segment/tetrahedron cutoffs.

The existing CCD tolerance, contact margins/gaps and final-state tolerances
are unchanged. The implementation does not clamp a positive distance into
penetration. The certificate uses the existing initial hull and does not add
shared-memory storage or a new allocation.

**Evidence and limits.** The three isolated initializer failures have enclosing
half-spaces and face distances below the inherited cutoff. The correction is
qualified with deeper-overlap, shallow-overlap, separated and updated-input
queries. A subsequent same-binary isolated extension recovered 27 emitted
pair occurrences among 233 complete frozen BOX/BOX candidates. All additions
overlap according to both actual CPU 2.3.7 and independent FP64 SAT; no emitted
pair was removed and no separated pair was added. The overlap depths were
approximately 1.9e-9 to 2.7e-7 m. Some frozen angular-velocity pulses decreased
from about 0.09 rad/s to 1e-4 rad/s, while another retained pulse remained.
This is not a general replacement of every EPA degeneracy criterion or a
guarantee that all contact-induced jitter disappears.

Workspace evidence: `results/mjwarp_velocity_20261021/residual-phase/`,
`residual-hull-certificates.json` and `certified-local/certified/`.
The ordinary-path extension and source-bound counterfactuals are in
`results/mjwarp_residual_20261021/policies-expanded/`, `geometry-expanded/`
and `REPORT.md`.

## ND-05: use the actual support plane for a tuned GJK cutoff

**Original.** Pinned `collision_gjk.py::_gjk_support` can rotate the support
direction for a small discrete simplex. `gjk` subsequently uses
`dot(x_k, support_point)` as a separating lower bound. That point is extremal
in the tuned direction, which need not be parallel to `x_k`; the claimed
support-plane bound is then invalid. The CUDA translation inherited this
control flow. Different floating-point simplex histories can expose the
problem in either implementation.

**Change.** [mjwarp_convex_gjk.cuh](../../../cuda/collision/mjwarp_convex_gjk.cuh)
retains the actual query direction. Only when the original test proposes an
early separation after discrete tuning does it check the support plane in
that direction, normalizing the cutoff by its norm. A degenerate direction
cannot certify separation. A failed certificate continues the existing GJK
iteration. Untuned tests, support selection, iteration limits, Frank-Wolfe
stopping, final ordinary support test and EPA rules keep their paths. No
additional support query, contact-distance clamp or tolerance change is added.

**Evidence.** A real Spatial:5 physical-step235 BOX pair has independent FP64
SAT gap `-3.93257e-5 m`; actual CPU MuJoCo2.3.7 also detects overlap. The
pre-correction CUDA GJK claimed a positive lower bound `1.98736e-5 m` in a
direction whose actual FP64 support minimum is `-6.67993e-4 m`. Corrected
GJK/EPA returns penetration `-3.93260e-5 m`, matching the original on this
input. Deeper-overlap, separated and translated controls pass. The maintained
fixture is [gjk_tuned_direction.json](../../../../../../tests/data/gjk_tuned_direction.json),
with the stable `mjwarp-convex-regression` qualifier. The independent host test
checks that the historical claimed plane cannot separate the actual boxes.

This proves a specific inherited invalid-certificate path and its repair;
it does not establish that every Spatial:5 trajectory difference was caused
by that pair or that the corrected full trajectory is uniformly more accurate.
Workspace evidence is `results/fp32_attribution_20261021/`, including immutable
baseline product/source copies, original phase outputs and corrected probes.

## Integration and reference interpretation

The call path is [Collision.narrowphase](collision.py) ->
[ConvexContacts.enqueue](../../runtime/collision/mjwarp_convex_collision.py) ->
[convex_contacts](../../../cuda/collision/mjwarp_convex_collision.cuh) -> GJK/EPA
-> existing box manifold and contact publication. The load-only function
probes in [mjwarp_convex.cu](../../../cuda/collision/mjwarp_convex.cu) use the
same maintained headers. The library is independently built as
`flash_mjwarp_convex_fp32`; source/product admission and offline preparation
remain mandatory.

GPU work distribution, MathDx calls and reordered floating-point reductions
are ordinary implementation differences covered by the migration qualifiers;
they are distinct from this register's deliberate numerical corrections.
Newton stopping, warmstart semantics and the model's physical parameters were
not changed by these corrections. CPU 2.3.7 supplies independent geometry
and overlapping physical semantics; its older collision algorithm is not an
exact oracle for the newer EPA manifold or every contact depth.

Raw-vs-CUDA contact differences therefore have three possible explanations to
investigate: an intended correction, floating-point sensitivity, or a port
defect. A corrected trajectory need not match an uncorrected reference, but a
small SAT gap alone does not prove the ensuing force or trajectory error safe.

## Retained qualification and open issues

The 2026-10-21 correction follow-up records 224 independent frozen box queries
and 632 production box candidates across 22 retained states. No candidate
detection failure was found outside the existing CCD tolerance in those
production states. Legacy GJK, shipped-domain EPA, box-manifold and production
ABI/capacity regressions also pass. See the workspace's
`results/mjwarp_velocity_20261021/REPORT.md` and source-bound reports for the
individual scopes and preserved original failures.

The targeted 13-task N=2 run uses five warmup and 20 continuous controls, once
per implementation, without cameras, runtime compilation or performance
measurement. Original strict checks pass 5/13, the existing FP32 full-state
policy passes 11/13, and the task-boundary policy passes 13/13. These are
different checks; none replaces the remaining failures or establishes a new
130-task qualification. Earlier 130-task timings belong to older physics.

In that earlier 13-task run, `libero_90:79` and `libero_90:84` failed the
full-state velocity comparison.
At their retained endpoint poses, same-input Newton checks pass with matching
iteration counts. All 15 raw/CUDA emitting-pair differences among their 79
box candidates have SAT gaps at most `6.868e-7 m`, below the existing
`1e-6 m` CCD tolerance. This supports local contact-selection sensitivity,
but CCD tolerance is not a floating-point error bound or a bound on impulse
amplification. Endpoint replays do not establish what happened at earlier
trajectory substeps. Those historical failures remain recorded.

The follow-up used complete pre-solve substep checkpoints to isolate actual
angular-velocity growth. With identical contacts/QPs, raw/CUDA Newton angular
velocity differs by at most 2.98e-8 rad/s, the rebuilt downstream chain by
2.72e-7 rad/s, and Euler by zero in the sampled states. Replacing only collision
changes angular velocity by up to 0.0944 rad/s. The ordinary-path ND-04 scope
gap above is one demonstrated contributor. Disabling all four corrections
does not restore raw arithmetic, so not all differences can be attributed
to these corrections; compilation and contact-active-set sensitivity remain.

After extending ND-04, one N=2, five-warmup/20-control correctness run per
task passes 2/2 existing FP32 full-state and 2/2 task-boundary comparisons;
strict checks still pass 0/2. Task 79's maximum qvel discrepancy against either
original endpoint is 0.01328, against an observed raw-repeat spread of 0.01497;
task 84's is 0.000230, against 0.000163 (the direct primary-reference error is
0.0000669). The existing policy, including its reference-spread factor,
is unchanged. A passing scoped result is not proof that jitter is eliminated
or that endpoint errors uniformly decrease. The rebuilt convex library also
passes the retained 224-query and legacy local qualifications. No performance
measurement, new 13/130-task aggregate, or general force equivalence is claimed.

Future intentional changes should add an ID here with the pinned original
behavior, maintained behavior, exact domain, source/caller, independent
evidence, trajectory implications and qualification limits. Preserve the
original baseline and failed checks; do not fit tolerances to clear them.
