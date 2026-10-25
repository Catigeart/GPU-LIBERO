# FP32 MJWarp to project CUDA migration map

This map describes the explicit FP32 MJWarp-backed path, not the older native
FP32/FP64 CUDA implementation. A migrated leaf is selected only by its option,
in its admitted domain, with source-bound offline products. All physics
selectors default to `upstream`; the explicit `preset="cuda_libero"` selects
all migrated leaves, including public forward. Local mathematical qualification does not
establish full-application performance parity. The maintained controller and
observation sampler are project implementations reused by this backend.

The pinned reference is MJWarp `c384ac51df67eb92ab7657cc472ff7494dc036bf`,
MuJoCo3.16.0 and Warp1.18.0; see [the source pin](../../../data/mjwarp_source_pin.json).
CPU MuJoCo2.3.7 is the independent mathematical/semantic reference where the
algorithms overlap. Its older disable flags and absent polynomial features do
not replace the pinned FP32 engine's behavior.

Intentional numerical corrections beyond translation are maintained in the
separate [numerical deviation register](NUMERICAL_DEVIATIONS.md). That register
distinguishes inherited reference issues, deliberate corrections and open
trajectory differences; ordinary migration coverage remains in this map.

## Migrated numerical leaves

| Stage | Migrated work | CUDA source | Host selection / binding |
|---|---|---|---|
| Rigid kinematics | Four-joint branch body poses/anchors/axes, body matrices, inertial frames and geom/site world frames | [mjwarp_kinematics.cu](../../../cuda/dynamics/mjwarp_kinematics.cu) | [kinematics.py](kinematics.py), [runtime binding](../../runtime/dynamics/mjwarp_kinematics.py); `kinematics=cuda` |
| CoM position frames | Weighted subtree positions including world, reverse-depth FP32 atomic accumulation, supplied-mass division, centered spatial inertia and four-joint motion axes | [mjwarp_com_pos.cu](../../../cuda/dynamics/mjwarp_com_pos.cu) | [com_pos.py](com_pos.py), [runtime binding](../../runtime/dynamics/mjwarp_com_pos.py); independent `com_pos=cuda` |
| FIXED camera/light frames | Camera world position/matrix, light world position/normalized direction; all shipped130-task modes | [mjwarp_camlight.cu](../../../cuda/dynamics/mjwarp_camlight.cu) | [camlight.py](camlight.py), [runtime binding](../../runtime/dynamics/mjwarp_camlight.py); independent `camlight=cuda` |
| Rigid queries | Point Jacobian, FORCE/TORQUE sensors and current gravity compensation | [mjwarp_queries.cu](../../../cuda/dynamics/mjwarp_queries.cu) | [queries.py](queries.py), [runtime binding](../../runtime/dynamics/mjwarp_queries.py); `queries=cuda`; sensor history remains upstream |
| Stateless publication | Time, dense/sparse overflow flags and exact public acceleration warmstart copy; empty activation/history admission | [mjwarp_publication.cu](../../../cuda/dynamics/mjwarp_publication.cu) | [publication.py](publication.py), [runtime binding](../../runtime/dynamics/mjwarp_publication.py); `publication=cuda`, requires CUDA Euler |
| Public/reset forward | Original later factor/solve order using the migrated stages; retained public Graph and one Newton context | Existing numerical leaves | [stages.py::PhysicsStages.forward](stages.py), [runtime.py::Runtime.forward](runtime.py); `public_forward=cuda` |
| Position mass matrix | CRB copy/accumulation, DOF armature and CSR ancestor assembly | [mjwarp_mass_matrix.cu](../../../cuda/dynamics/mjwarp_mass_matrix.cu) | [mass_matrix.py](mass_matrix.py), [runtime binding](../../runtime/dynamics/mjwarp_mass_matrix.py); `mass_matrix=cuda` |
| Mass factorization | Compact1/scalar6/tile9 Cholesky, diagonal reciprocals | [mjwarp_mass_factor.cu](../../../cuda/dynamics/mjwarp_mass_factor.cu) | [mass_factor.py](mass_factor.py), [runtime binding](../../runtime/dynamics/mjwarp_mass_factor.py); `mass_factor=cuda` |
| Spatial velocity | Body `cvel`, DOF `cdof_dot`, unique branch output publication | [mjwarp_velocity.cu](../../../cuda/dynamics/mjwarp_velocity.cu) | [velocity.py](velocity.py), [runtime binding](../../runtime/dynamics/mjwarp_velocity.py); `com_vel=cuda` |
| Joint passive forces | FREE/BALL/SLIDE/HINGE spring and damping, quaternion difference, polynomial coefficients | [mjwarp_passive.cu](../../../cuda/dynamics/mjwarp_passive.cu): `joint_forces` | [passive.py](passive.py), [runtime binding](../../runtime/dynamics/mjwarp_passive.py); `passive=cuda` |
| Passive combination | Ordered sum of spring, damping, eligible gravcomp, fluid and adhesion | [mjwarp_passive.cu](../../../cuda/dynamics/mjwarp_passive.cu): `combine_forces` | Same passive selector; each contribution remains publicly stored |
| Inertia-box fluid | Current body viscous/quadratic drag and torque; ordered DFS projection into qfrc_fluid | [mjwarp_fluid.cu](../../../cuda/dynamics/mjwarp_fluid.cu) | [fluid.py](fluid.py), [runtime binding](../../runtime/dynamics/mjwarp_fluid.py); independent `fluid=cuda`, upstream or CUDA joint passive leaves |
| Bias RNE | Gravity root, velocity acceleration, inertial/Coriolis forces, reverse body sum and DOF projection | [mjwarp_bias.cu](../../../cuda/dynamics/mjwarp_bias.cu) | [velocity.py](velocity.py), [runtime binding](../../runtime/dynamics/mjwarp_bias.py); `bias=cuda` |
| Scalar-joint transmission | Current gear/qpos actuator length, atomic CSR row allocation and one-DOF moment publication | [mjwarp_transmission.cu](../../../cuda/dynamics/mjwarp_transmission.cu): `scalar_transmission` | [transmission.py](transmission.py), [runtime binding](../../runtime/dynamics/mjwarp_transmission.py); `transmission=cuda` |
| Actuator velocity | Ordered dot on current generic CSR moments and qvel | [mjwarp_transmission.cu](../../../cuda/dynamics/mjwarp_transmission.cu): `actuator_velocity` | Same driver/binding; independent `actuator_velocity=cuda` |
| Stateless actuation | Fixed/affine actuator force, control/force clamps, current CSR moment projection, actuator gravcomp and joint-force limits | [mjwarp_actuation.cu](../../../cuda/dynamics/mjwarp_actuation.cu) | [actuation.py](actuation.py), [runtime binding](../../runtime/dynamics/mjwarp_actuation.py); `actuation=cuda` |
| Smooth force | Force sum and external-wrench projection into `qfrc_smooth` | [mjwarp_smooth_force.cu](../../../cuda/dynamics/mjwarp_smooth_force.cu) | [acceleration.py](acceleration.py), [runtime binding](../../runtime/dynamics/mjwarp_smooth_force.py); `acceleration=cuda` |
| Mass solve | Compact1/scalar6/tile9 backsubstitution into smooth acceleration | [mjwarp_mass_solve.cu](../../../cuda/dynamics/mjwarp_mass_solve.cu) | [acceleration.py](acceleration.py), [runtime binding](../../runtime/dynamics/mjwarp_mass_solve.py); `mass_solve=cuda`, implied by CUDA acceleration |
| Rigid convex CCD in all130 pair domains | Shape support, ordered simplex projection, GJK shrink/inflate, EPA, required box-box manifold and pooled material/contact publication | [mjwarp_convex.cu](../../../cuda/collision/mjwarp_convex.cu), [production kernel](../../../cuda/collision/mjwarp_convex_collision.cuh), readable geometry/simplex/GJK/EPA/manifold headers alongside | [collision.py](collision.py), [production binding](../../runtime/collision/mjwarp_convex_collision.py); independent `convex_collision=cuda`, explicit pair-domain fallback |
| Sphere/capsule-box and primitive box-box | Sphere-box signed penetration, capsule endpoint/edge search and second witness, box-box SAT/edge witnesses/incident-face clipping | [mjwarp_box.cu](../../../cuda/collision/mjwarp_box.cu), [shared contact math](../../../cuda/collision/mjwarp_contact.cuh) | [collision.py](collision.py), [runtime binding](../../runtime/collision/mjwarp_box.py); independent `box_collision=cuda`, box-box retains original CONVEX/PRIMITIVE split |
| Six plane contact families | Plane versus sphere/capsule/ellipsoid/cylinder/box/mesh; current priority/solmix materials and pooled contact commit | [mjwarp_plane.cu](../../../cuda/collision/mjwarp_plane.cu), [shared contact math](../../../cuda/collision/mjwarp_contact.cuh) | [collision.py](collision.py), [runtime binding](../../runtime/collision/mjwarp_plane.py); independent `plane_collision=cuda` |
| NXN broadphase | Imported prefiltered pairs, current plane/sphere/corner-AABB/six-axis OBB predicates, pooled candidate append and metadata | [mjwarp_broadphase.cu](../../../cuda/collision/mjwarp_broadphase.cu) | [collision.py](collision.py), [runtime binding](../../runtime/collision/mjwarp_broadphase.py); independent `broadphase=cuda` |
| Dense rigid constraint construction | Counts, DOF friction/scalar limits, atomic contact row blocks, dense contact Jacobian/Jqvel, impedance and reference rows | [mjwarp_constraint_build.cu](../../../cuda/dynamics/mjwarp_constraint_build.cu) | [constraint_build.py](constraint_build.py), [runtime binding](../../runtime/dynamics/mjwarp_constraint_build.py); independent `constraint_build=cuda` |
| Newton initialization/stopping | Warmstart selection, reset, Jaref/Ma setup, convergence and done transitions | [mjwarp_solver_state.cu](../../../cuda/newton/mjwarp_solver_state.cu) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_solver_state.py); cumulative `newton=cuda_newton` |
| Newton constraint update | Constraint force/state and generalized-force projection on existing rows | [mjwarp_constraint.cu](../../../cuda/newton/mjwarp_constraint.cu) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_constraint.py) |
| Newton Hessian | Mass/constraint curvature assembly in the admitted dense elliptic domain | [mjwarp_hessian.cu](../../../cuda/newton/mjwarp_hessian.cu) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_hessian.py) |
| Newton direction | Gradient, factor/solve, direction norm and decrement; cuSolverDx for qualified small blocks | [mjwarp_direction.cu](../../../cuda/newton/mjwarp_direction.cu) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_direction.py) |
| Newton line search | M/J products, cost/derivatives, bracketing and accepted-step commit | [mjwarp_linesearch.cu](../../../cuda/newton/mjwarp_linesearch.cu), [shared products](../../../cuda/newton/mjwarp_products.cuh) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_linesearch.py) |
| Applied/contact external wrenches | Applied force layout/origin conversion, contact cone decode, world rotation, root-CoM torque shift and component FP32 atomics | [mjwarp_external_wrench.cu](../../../cuda/dynamics/mjwarp_external_wrench.cu) | [external_wrench.py](external_wrench.py), [runtime binding](../../runtime/dynamics/mjwarp_external_wrench.py); independent `external_wrench=cuda` |
| Postconstraint RNE suffix | Solved acceleration propagation, inertial/Coriolis minus external wrench, reverse body sum | [mjwarp_postconstraint.cu](../../../cuda/dynamics/mjwarp_postconstraint.cu) | [postconstraint.py](postconstraint.py), [runtime binding](../../runtime/dynamics/mjwarp_postconstraint.py); `postconstraint=cuda` |
| Euler preparation and advancement | Polynomial damping derivative, integration-matrix diagonal, velocity and four-joint position/quaternion advancement | [mjwarp_euler.cu](../../../cuda/dynamics/mjwarp_euler.cu) | [euler.py](euler.py), [runtime binding](../../runtime/dynamics/mjwarp_euler.py); `euler=cuda` |
| Euler factor/solve | Factorization/backsubstitution of damping-adjusted mass matrix, private acceleration workspace | [mjwarp_mass_factor.cu](../../../cuda/dynamics/mjwarp_mass_factor.cu) | [mass_factor.py](mass_factor.py); part of `mass_factor=cuda`; reused by `euler=cuda`, which otherwise retains upstream factor/solve |

Newton's cumulative selector controls the five Newton libraries together;
partial selectors remain available. Constraint update here operates on rows
already generated by the independently selected construction stage; it does not
itself migrate row construction or collision.
cuSolverDx is an external numerical dependency, not a handwritten factor kernel.
The dense elliptic route follows upstream full Hessian rebuilds; no separate
migration of an inactive rank-update branch is claimed.

## Upstream defaults, fallbacks and optional features

The paths below are within the installed pinned `mujoco_warp/_src/` package.
The corresponding project dispatch files are linked so the retained boundary is
reviewable without assuming that a host wrapper means the mathematics migrated.

| Stage / remaining work | Upstream file and functions | Project call site |
|---|---|---|
| Non-FIXED camera/light tracking modes | Complete `smooth.py::camlight` when any mode is unqualified | [camlight.py](camlight.py), [kinematics.py](kinematics.py): visible whole-boundary fallback |
| Flex and tendon geometry | `smooth.py::flex`, `tendon`; absent in the shipped130 scene XMLs, calls retained | [position.py](position.py), [kinematics.py](kinematics.py) |
| SAP broadphase; unqualified convex/multicontact and unselected primitive families/contact generation | `collision_driver.py`, `bvh.py`, `collision_core.py`, `collision_primitive*.py`, `collision_convex.py`, `collision_sdf.py` | [collision.py](collision.py), [position.py](position.py): NXN, planes, boxes and seven actual130 convex families independently selected; flex/sleep whole-collision fallback; SAP and additional convex/MULTICCD domains remain upstream |
| Equality/tendon/flex, limited BALL, sparse and moving-surface constraint construction | Complete `constraint.py::make_constraint` for unqualified domains | [constraint_build.py](constraint_build.py), [position.py](position.py): visible whole-builder fallback |
| Tendon armature mass contribution | `smooth.py::tendon_armature` | [position.py](position.py) |
| FREE/BALL, site, tendon, slider-crank and body actuator transmission | Complete `smooth.py::transmission` for unqualified scalar-joint domains | [transmission.py](transmission.py): transmission fallback leaves generic CUDA velocity independently selectable |
| Tendon velocities | `forward.py::_tendon_velocity` | [velocity.py](velocity.py) |
| Tendon/flex passive forces | `passive.py` tendon/flex kernels | [passive.py](passive.py): whole-caller fallback for these domains |
| Unselected gravity compensation | `passive.py::_gravity_force` | [queries.py](queries.py), [passive_other.py](passive_other.py); current rigid projection available with `queries=cuda` |
| Ellipsoid fluid and unsupported passive callers | Complete `passive.py::_fluid` / `_fluid_force`; `support.py::apply_ft` | [fluid.py](fluid.py), [passive_other.py](passive_other.py): visible upstream fallback |
| Adhesion-force generation | `passive.py::_qfrc_adhesion` | [passive_other.py](passive_other.py) |
| User passive callback | `model.callback.passive` | [passive.py::Passive.enqueue](passive.py) after the selected sum, preserving upstream early-return semantics |
| Tendon bias contribution | `smooth.py::tendon_bias` | [velocity.py](velocity.py) |
| Stateful/custom/muscle/DC-motor actuation, tendon limits and rotational setpoint wrapping | Complete `forward.py::fwd_actuation` for unqualified domains | [actuation.py](actuation.py) visible whole-caller fallback |
| Delayed control interpolation and act_dyn/act_gain/act_bias callbacks | `history.py::read_ctrl_delayed`, `model.callback` | [actuation.py::Actuation.enqueue](actuation.py); callback order retained between force and projection |
| Equality/tendon/flex and unsupported external-wrench callers | Complete `smooth.py` external-wrench prefix | [external_wrench.py](external_wrench.py): visible whole-prefix fallback; literal [postconstraint_external.py::assemble_external](postconstraint_external.py) |
| Other sensor families, history/delay and enabled energy | `sensor.py`, `history.py`, enabled energy math | [queries.py](queries.py); shipped FORCE/TORQUE available in CUDA, position/velocity sensor stages empty, disabled energy cleared |
| Nonempty activation/history or active sleep | Original activation/history/sleep suffix | [publication.py](publication.py); stateless/no-history CUDA publication is available, whole suffix fallback otherwise |
| Optional noslip solver | `solver.py::noslip` | [newton.py](newton.py); all shipped130 tasks disable noslip |
| Sparse/pyramidal/sleeping/discrete, unsupported mass families and other unqualified domains | Complete pinned upstream callers | Each selector's visible fallback; no migration is claimed for these paths |
| Unselected or non-Euler/sleep public forward | Public `mujoco_warp.forward` | [runtime.py::Runtime.forward](runtime.py); independent CUDA public refresh preserves later factor/solve order |

## Retained infrastructure and project components

| Component | Current ownership / source |
|---|---|
| CUDA Graph loops, Warp Model/Data and per-solve scratch allocation | Upstream Warp/MJWarp infrastructure, retained deliberately; [newton.py](newton.py), [runtime.py](runtime.py) own context/Graph lifetimes |
| Scene import, capacity/profile admission, reset/public holds | Project integration: [model.py](model.py), [environment.py](environment.py), [build.py](build.py) |
| Panda OSC | Existing project CUDA, reused: [mjwarp_controller.cu](../../../cuda/control/mjwarp_controller.cu), [controller.py](controller.py); [queries.py](queries.py) selects original or CUDA Jacobian input |
| Controller gather/scatter, goals and work counters | Project Warp/CUDA bridge: [kernels.py](kernels.py); it is not a migrated physics stage |
| Policy observation sampling and LIBERO predicates | Existing project sampler: [device.py](../../adapters/observations/device.py), [environment.py](environment.py), [facade.py](facade.py); original task Python predicates retained |
| Camera image rendering | Outside the current MJWarp migration qualification; default tests disable image observations. FIXED camera poses can use the selected CUDA leaf above. |

Independent CMake leaves are in [cmake/](../../../cmake/); stable local qualifiers
are exposed by `tools/validate_gpu.py`. The passive qualifier covers joint forces, final combination and their leaf
composition. The actuation qualifier covers stateless force, CSR projection,
joint limits and composition on frozen N=2 inputs. The shipped130 tasks share
one XML actuator pattern: seven torque motors and two kp1000 position grippers.
CUDA dimensions remain dynamic and also admit qualified differing actuator/control
counts and fixed/affine parameters. The Euler qualifier covers damping/diagonal,
velocity/position and their composition, with current parameter reads and exact
public holds. The selected caller shares the existing mass-factor workspace or
owns one private equivalent when upstream factors are selected. Nonempty activation
remains upstream; stateless publication has its own selector. The transmission qualifier covers scalar length/moment,
generic CSR velocity and composition into stateless actuation, including a
289-actuator multi-CTA allocation case and current parameter/index changes.
Atomic row offsets may vary just as upstream; exact row ownership, column
identity, copied payloads and untouched tails are checked after unpacking.
The rigid kinematics qualifier covers five stages and composition on real and
manufactured N=2 inputs, including deep/wide branches, all joint types, mocap,
broadcast and static-geom holds. Its branch owner can be shared with velocity
and RNE; every branch keeps its own parent pose, with one global publisher.
CPU checks use the common unit-quaternion mathematical domain; raw FP32 inputs
and pinned Warp normalization/static rules are separately qualified.
The CoM qualifier checks subtree, centered inertia and motion axes independently,
their composition and real-model CRB/CSR composition. It retains pinned supplied
subtree masses, zero/tiny-mass behavior and world contributions, with no added
device allocation or physical cache. Depth and dimensions are dynamic; existing
immutable tree layers and public numerical arrays are borrowed. CPU2.3.7 checks
the common positive-mass mathematics; raw pinned behavior remains authoritative
for the differing zero/tiny-mass and world-inertia conventions. CoM and rigid
kinematics selectors are independent, including upstream kinematics with CUDA CoM.
The FIXED camera/light qualifier covers the two kernels and composition on the
three real7/2,8/3 and8/2 attachment scales and manufactured mounts, broadcasts,
zero/tiny directions, non-unit quaternions, empty halves and multi-CTA cases.
CPU comparisons use coherent unit-quaternion frames; raw pinned homogeneous
quaternions and Warp zero normalization remain separately qualified. No device
workspace or physical cache is added; tracking modes visibly retain upstream.
The inertia-box fluid qualifier separates world-body wrench generation,
asymmetric-wrench projection and composition into joint passive/sum leaves.
It covers real NV24/43/57 and manufactured parameter/threshold/wind/inertia
domains. One retained 24*N*nbody-byte workspace is reused across Graph substeps;
there is no shared-memory requirement or cached mass/inertia-derived coefficient.
Actual CPU mj_passive and mj_applyFT separately qualify fluid and projection;
the FP32 clamp-literal conversion is part of independent numerical allowances.
Fluid selection can independently retain upstream joint and sum kernels;
their gravity/fluid/adhesion/callback ordering and both-disabled early return
remain explicit. Ellipsoid fluid and unqualified callers retain upstream.
These local qualifiers do not establish complete130-task trajectories or measure performance.

The external-wrench qualifier covers applied, contact-only and composed paths,
with both cones, dimensions1/3/4/6, scattered/clipped row addresses, world-body
signs, excluded/empty contacts, flat multi-world indexing and multi-CTA atomic
fan-in. All Model/Data arrays are borrowed with no numerical workspace.
CPU2.3.7 checks equivalent contiguous zero-adhesion forces; pinned adhesion and
capacity conventions are retained and independently bounded. The external
prefix and RNE suffix can each select upstream or CUDA. Force/torque sensors
consume the selected RNE results and have their own `queries=cuda` selector. All130 scene XMLs
have one force and one torque sensor and no equality constraints; this census
is separate from local numerical qualification.

The dense rigid row qualifier covers all shipped constraint families (nine
friction-loss DOFs, HINGE/SLIDE limits, elliptic contact condim3/4) and both cones,
condim1/3/4/6, disable/empty/capacity cases and NV276 DOF striding. Every numerical
array is borrowed with zero workspace. Exact counts/category order/contiguous
contact ownership/copy/hold contracts permit only the pinned atomic allocation
order freedom. Fixed8x independent FP64 bounds include kbi/powf conditioning and
Jacobian reductions. CPU2.3.7 actual makeConstraint/referenceConstraint qualify
common zero-adhesion math; pinned tangential pos/margin storage is retained.
Local qualification does not establish complete trajectories or performance.

## Preserved primitive limitation

The pinned capsule-box algorithm inherits CPU2.3.7's endpoint/box-edge search.
It does not explicitly solve a centerline crossing a box face interior. A
controlled capsule passing through a thin box can therefore be missed by both
engines, including the faithfully ported CUDA path. Local box validation keeps
that exact-overlap completeness test as a separately reported geometry
diagnostic; port parity and emitted-contact residuals remain mandatory. The
observed source geom pair is excluded by its original scene's collision masks,
so this function-level example is not a demonstrated shipped task failure.
No broader physical completeness claim is made from engine conformance.

## Convex building blocks and production integration

The convex support function is implemented in
[mjwarp_convex_geometry.cuh](../../../cuda/collision/mjwarp_convex_geometry.cuh)
with a load-only [probe entry](../../../cuda/collision/mjwarp_convex.cu) and
[binding](../../runtime/collision/mjwarp_convex.py). It covers rigid
sphere/capsule/ellipsoid/cylinder/box/mesh support and the distinct absolute
vertex-address versus local-graph cache formats. Standalone probe qualification is distinct from the production selector
described below; support for a shape alone does not qualify every pair.

The ordered simplex projection subalgorithm is also available as a qualified
function probe in
[mjwarp_convex_simplex.cuh](../../../cuda/collision/mjwarp_convex_simplex.cuh).
It preserves GJK's vertex-order precondition and uses translated triangle
determinants to avoid cancellation in expanded absolute-coordinate areas.
Probe fixtures come from original GJK executions and use independent FP64
convex-QP/KKT residuals. The production selector below uses this corrected projection in GJK/EPA.

The production convex selector now joins the qualified building blocks:
`Collision.narrowphase -> ConvexContacts.enqueue -> convex_contacts<Type1,Type2>
-> gjk_phase/epa_phase -> box_multicontact when required -> contact_material /
publish_contact`. The seven actual130 families are capsule-cylinder,
capsule-mesh, cylinder-box, cylinder-mesh, convex box-box, box-mesh and mesh-mesh.
Each specialization is precompiled into the same convex library. Per-type
kernels are submitted serially on one stream, so one retained global EPA pool
can safely serve every type and all25 Graph substeps. The66 private counters
are reset per call; attempted CCD/contact counts, overflow masks and public
contact metadata retain the pinned contracts. Primitive margin/NATIVECCD
box-box is excluded from this convex route. Empty primitive residues are
skipped using the immutable type census.

All130 prefiltered catalogues pass this pair-domain admission. General
cylinder-cylinder remains explicitly unqualified after a capacity divergence
with nonunique cap support; additional pair families and general mesh/cylinder
MULTICCD retain the whole upstream convex boundary. Support qualification for
a shape does not qualify every collision pair. Box-box manifold remains
active independently of general MULTICCD. Its approximately aligned-face
projection is preserved, with its geometric error bound reported separately
from floating-point error in the manifold qualifier. Standalone GJK, EPA and
manifold probes remain available; their local results do not establish complete
130-task trajectory or performance parity.


## Complete shipped-domain schedule

`mjwarp_options={"preset": "cuda_libero", "fast_compile": True}` selects the
complete admitted numerical path. Partial per-leaf overrides remain explicit;
the final qualifier rejects any selected upstream fallback. Ordinary control
is `position/factor -> velocity/passive/bias -> OSC/Jacobian -> actuation ->
acceleration -> Newton -> postconstraint/sensors -> Euler -> publication`,
repeated25 times by the control Graph. Public refresh instead delays factor
and solve until acceleration, does not integrate and holds warmstart. Its
single Newton context is owned separately from the25 control contexts.

The shipped domain excludes activation/history, energy enablement, flex,
tendon/equality constraints, sleep, noslip, extra convex pairs and generic
multicontact. These are admission conditions or explicit original fallbacks,
not omitted work hidden behind the complete preset. The source-bound
`mjwarp-complete` qualifier records actual selected paths and every positive
Warp numerical launch during public/control captures. Warp allocation, copies,
zeroing, graphs, imported Model/Data and external math libraries remain
infrastructure. Local and public-forward qualification is distinct from the
final130-task trajectory/performance comparison.

Modern scenes are prepared by [scene_cache.py](scene_cache.py). Runtime uses
engine versions, input XML, options and scene-parameter metadata for direct
selection. Asset/library/model content validation and legacy-cache migration
run only during explicit offline preparation. Runtime cache misses fail;
explicit offline preparation creates modern MJBs. CPU2.3.7 MJBs are not
loaded by the modern engine. The existing task factory remains the owner of
shell construction and final XML processing.


Fixed-fixture input is part of the maintained bridge. The original scene XML
precedes LIBERO's fixed `body_pos`/`body_quat` placement writes; importing that
XML alone places cabinets/stoves at the origin. [scene_inputs.py](scene_inputs.py)
loads small source-bound arrays exported by `mjwarp-scene-inputs` in the actual
CPU2.3.7 environment, checks body/joint index mapping, and applies the declared
fixture roots through `build_task_model(static_model_setup=...)` before the
native snapshot. It uses the prepared parameters directly, with no resampling
or runtime CPU simulation. The modern model-cache binding includes this input.
The complete qualifier checks CPU body/site/primitive poses against the original
N=2 oracle; mesh local frames remain explicitly compiler-dependent.

Final trajectory reports retain cross-trajectory intermediate deltas. Spatial
velocity and unconstrained acceleration additionally receive mandatory
same-input function replays after timing; unconstrained mass equations also
receive independent FP64 backward-error checks. Root-state bounds, existing
frozen FP32 acceleration sensitivity, exact public-copy/time/input holds and
capacity contracts remain required. These postchecks add no integration or
measured Graph nodes. Numerical budgets are unchanged.

## Explicit numerical corrections and final diagnostics

The translated triangle-minor algebra above is an intentional correction to
expanded-area cancellation, rather than a claim of identical floating-point
instructions. A second correction keeps box support points and their discrete
feature IDs consistent. When a local support direction component is exactly
zero, any point between the two extrema maximizes that component. The pinned
`sign(0)` chooses zero, but its vertex bit denotes the negative corner. EPA
then recovers a different edge or face from that ID. The maintained box support
chooses the negative extreme for zero, making the returned point match its ID
while preserving the exact support maximum. The strict indexed-corner oracle
is additional to the independent support projection check. These corrections
can change legal GJK/EPA paths and manifold counts near a feature boundary.

The complete Graph audit uses both Warp module names and the original Python
function provenance. Generated `module="unique"` kernel names cannot hide an
upstream numerical launch. Allocation, copies and Graph infrastructure remain
permitted.

After a failed strict final endpoint screen, `mjwarp-complete` additionally
replays the original Graph once from retained identical root/controller buffers,
without timing. It saves the original-against-original endpoint variation and
compares its decimal order per field/world with the CUDA discrepancy. This is
a diagnostic only: strict failed checks and the report's `passed` flag remain
unchanged. No extra numerical work is inserted into the measured Graph.
Finite numerical disagreement at initial forward also remains a failed check,
but does not prevent one diagnostic performance measurement. Nonfinite arrays,
out-of-range storage counts or changed common initial inputs stop execution.
Saved endpoint reassessment cannot clear an initial numerical failure.

## Shipped-domain rewrite completion and qualification limits

The 2026-10-21 sweep completed all130 tasks on A100/SM80, FP32, N=2,
5 warmup controls and20 continuous measured controls per implementation,
once, without cameras. All admitted public/control Graphs selected the CUDA
numerical path, with no original positive numerical kernel launches and no
runtime compilation. The29 independent libraries include28 numerical CUDA
translation units plus the existing controller. Warp Model/Data, allocation,
copies, Graph scheduling, scene import and external MathDx remain dependencies.
No unused general-engine feature in the fallback table above is newly claimed.

The rewritten shipped-domain code is complete; full-state numerical parity is
partially qualified. These statements must remain distinct:

| Retained check | Tasks passing | Scope |
|---|---:|---|
| Original strict complete-path checks | 58/130 | Includes acceleration, forces, rotations, sensors and fixed endpoint screens |
| Existing FP32 reference-repeat policy on full state | 117/130 | All post-Euler qpos/qvel, closed-loop OSC outputs and time; excludes acceleration/internal forces |
| Same policy on task boundary | 130/130 | Robot9DOF qpos/qvel, all body positions/quaternions, OSC outputs and time; excludes free-object velocity/internal forces |
| Mandatory common-input endpoint leaves | 130/130 | Spatial velocity, mass solve and independent unconstrained backward error |

The130/130 boundary result does not replace the72 strict failures. The13 full
state exceptions involve free-body angular velocities of plates, trays or mugs;
the largest discrepancy is0.0769rad/s. Their body poses remain within the
existing reference-repeat policy. Public body frames precede the final Euler
advance; the full-state analysis separately checks post-Euler qpos. One task,
libero_90:5, also fails the initial strict numerical screen. Its frozen identical
QP solver and independent mass backward errors pass, but that does not clear
the composed forward failure. No tolerance was enlarged to admit these cases.

Frozen N=2 Newton replays at both retained endpoints of four representative
velocity-outlier tasks (libero_90:69/62/82/83, including the three largest) pass the existing function policy,
with matching iteration counts on each common QP. Their contact-count
differences are table/thin-box contacts at FP64 SAT gaps below8.44e-7m.
Independent emitted-contact checks pass, but raw contact-set conformance fails;
contact completeness and whole-trajectory velocity parity are not thereby
proved. Two earlier geometric corrections intentionally differ from pinned
arithmetic, as described above. The original implementation remains the default;
`cuda_libero` is explicit and must not be described as fully state-equivalent.

Across all130 retained comparisons, the device-control Graph geometric mean
speedup against pinned MJWarp is1.211x (range1.046–1.763x). Suite geometric means
are spatial1.299x, object1.204x, goal1.236x, libero_90 1.201x and libero_10 1.191x.
These include numerical-failure cases as diagnostic timings. They exclude
initialization, reset, Graph preparation, cameras and host observation/predicate
packaging. The device interval includes the small action upload and the Graph's
OSC/observation sampling. Shared-GPU execution was authorized; this was not an isolated-device
performance certification. Native FP64 and native FP32 products are unchanged.

The CPU-only `mjwarp-endpoint-report` command reproduces the scoped conclusions
from retained source/scene/output bindings. It neither runs physics nor changes
the original strict statuses. Per-task evidence and the final migration report
belong in the development workspace's `results/mjwarp_complete_20261021/`,
outside the clean project tree.

## Free-object velocity diagnosis and certified box retries

A subsequent same-input investigation of the13 full-state exceptions found
no Newton port defect in the retained endpoint QPs: all13 tasks passed the
function policy at both endpoint poses. An exact pre-step checkpoint for
libero_90:69 also reproduces the angular-velocity impulse with the original
solver, including when warmstart is zeroed. Replacing the collision leaf on
that checkpoint changes the response. These checks isolate those QPs and
checkpoints; they do not prove equivalence at every trajectory substep.

Complete candidate pools, checked with actual CPU MuJoCo2.3.7 collision and an
independent FP64 corner-based box SAT, exposed shallow BOX/BOX misses in both
the pinned and rewritten paths. The earlier emitted-contact-only checks could
not establish completeness. Two inherited decisions contribute: a positive
closest-simplex distance without a separating support plane is inconclusive;
and the triangle EPA initializer rejects face distances squared below1e-10,
including valid enclosing hulls with penetration below10 micrometres.

For inconclusive zero-margin discrete BOX/BOX queries only, `gjk_phase` now
makes one witness-seeded refinement and attempts EPA. The retry requires a
nondegenerate initial hull whose complete face half-spaces enclose the origin,
evaluated in FP64 on the actual FP32 hull vertices. Its triangle initializer
uses that certificate instead of the coarse face-distance rejection. No
collision tolerance, endpoint tolerance, margin or distance clamp is changed.
Other shape families and conclusive separation returns retain their paths.
The production convex caller and function probe use the same implementation.

This is a physical correction beyond exact reproduction of pinned arithmetic.
Raw contact-set differences and raw failures remain recorded separately from
independent geometry checks. Small free-body velocity differences can still
reflect FP32 contact selection, position quantization and impulse phase; they
must not automatically be dismissed or used to clear an existing failed check.
The130-task timings above belong to the pre-correction version. This follow-up
performs correctness checks only and makes no performance claim for this fix.
The original backend remains the default and `cuda_libero` remains explicit.

The corrected version's targeted13-task follow-up uses N=2,5 warmup and20
continuous controls per implementation, once, without timing or runtime
compilation. It passes5/13 original strict checks,11/13 existing FP32 full-state
checks and13/13 task-boundary checks. The remaining full-state exceptions are
libero_90:79 and84, both free-object angular velocity. These observations do
not update the historical130-task denominators or certify the other117 tasks
under the new collision arithmetic. Strict failures remain failed; original
repeat variation and corrected geometry are reported separately.
At both final endpoint poses of those two residual tasks, same-input Newton
replays pass and79 complete box candidates agree with CPU2.3.7/independent SAT
outside the existing CCD tolerance. Their15 raw/CUDA emitting-pair differences
have SAT gaps no larger than6.868e-7m, below the existing1e-6m CCD tolerance.
This supports contact-selection sensitivity at those poses; it does not clear
the two trajectory failures or establish agreement at every prior substep.

### Exact substep attribution and ordinary shallow-face correction

The next investigation retained complete pre-solve Data/contact/efc and OSC
checkpoints at actual angular-velocity growth in tasks79 and84. Cross replacement
isolates collision: fixed-QP Newton differences are at most2.98e-8rad/s,
fixed-contact downstream differences at most2.72e-7rad/s, and identical solved
states produce identical sampled angular velocity through either Euler path.
Changing only the complete narrowphase, on the same broadphase candidates,
changes one-step angular velocity by up to0.0944rad/s. Raw trajectories also
have angular pulses. Small geometric gaps are not an impulse-error bound.

The certified triangle initializer initially covered only inconclusive GJK
retries. Ordinary zero-margin BOX/BOX results below CCD tolerance retained the
same coarse squared-face-distance rejection. An isolated same-binary extension
recovers27 overlapping pair occurrences in233 frozen candidates; actual
CPU2.3.7 and independent FP64 SAT agree on all233 overlap signs. No pair is
removed or falsely added. The production initializer now propagates
`require_enclosing` when an ordinary box triangle uses the shallow-face
exception, then certifies every initial half-space before EPA. Ordinary hulls
above the cutoff avoid that extra predicate. Other shapes, margins, CCD
tolerance, Newton, warmstart and endpoint bounds are unchanged. See
[ND-04](NUMERICAL_DEVIATIONS.md#nd-04-shallow-initial-faces-are-not-geometric-degeneracy).

The updated convex product passes the retained224-query,12-GJK,14-EPA,
6-manifold and22-production-state local regressions (raw mathematical and
contact-conformance differences are retained separately). A final untimed
N=2,5-warmup/20-control run on79 and84 passes2/2 existing FP32 full-state and
2/2 task-boundary comparisons, while strict checks remain0/2. Maximum qvel
differences against either original endpoint are0.01328 and0.000230; observed
raw-repeat spreads are0.01497 and0.000163. The direct primary-reference error
on84 is0.0000669. The existing reference-spread policy was not changed. This does not
establish uniform velocity improvement or eliminate every retained angular
pulse. All-four-corrections-off CUDA also differs from raw Warp, so arithmetic
and contact-active-set sensitivity must remain distinct from correction effects.

Reports, checkpoints, counterfactual source/binary receipts and limitations are
in `results/mjwarp_residual_20261021/` in the development workspace. Only the
convex library changed; native FP32/FP64 source identities are unchanged. These
two tests do not update the historical13/130-task results or timings above.

## Independent CUDA owner

The maintained handwritten FP32 mathematical leaves are now also used by an
independent CUDA allocation/stream/Graph/model/lifecycle owner. Automatic FP32
batch routing selects `physics_engine="cuda_fp32"`; this path requires neither
Warp package during preparation or execution. Explicit `physics_engine="mjwarp"`
retains the upstream reference and the earlier migration selectors.

The subsequent all130-task qualification records all130 public boundaries,
127/130 existing upstream FP32 trajectory checks and100/130 fixed-roundoff
checks against the corrected CUDA reference. The corrected reference itself
passes125/130 upstream checks in that campaign. Original failures are retained;
these counts do not rewrite older benchmark results or prove universal raw
trajectory equivalence. Two additional600-control runs pass both endpoint
policies. See the independent owner's [qualification report](../cuda_fp32/QUALIFICATION.md)
for scope, unchanged bounds, residual cases and shared-device measurements.

### Subsequent three-case GJK follow-up

ND-05 repairs an inherited tuned-direction separation-certificate mismatch in
the shared CUDA convex leaf. Same-input probes and independent CPU2.3.7/FP64
geometry establish the specific defect and recovered contact. Targeted function
regressions and three N=2 rollouts pass execution checks; all three retained
raw endpoint exceptions remain failed. These local changes do not update the
historical130-task pass counts, timing results or previously built packages.
See the [deviation register](NUMERICAL_DEVIATIONS.md) and the independent owner's
[follow-up scope](../cuda_fp32/QUALIFICATION.md#three-case-causal-follow-up-and-nd-05).
