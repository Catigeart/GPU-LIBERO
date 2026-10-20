# FP32 MJWarp to project CUDA migration map

This map describes the explicit FP32 MJWarp-backed path, not the older native
FP32/FP64 CUDA implementation. A migrated leaf is selected only by its option,
in its admitted domain, with source-bound offline products. All physics
selectors default to `upstream`. Local mathematical qualification does not
establish full-application performance parity. The maintained controller and
observation sampler are project implementations reused by this backend.

The pinned reference is MJWarp `c384ac51df67eb92ab7657cc472ff7494dc036bf`,
MuJoCo3.16.0 and Warp1.18.0; see [the source pin](../../../data/mjwarp_source_pin.json).
CPU MuJoCo2.3.7 is the independent mathematical/semantic reference where the
algorithms overlap. Its older disable flags and absent polynomial features do
not replace the pinned FP32 engine's behavior.

## Migrated numerical leaves

| Stage | Migrated work | CUDA source | Host selection / binding |
|---|---|---|---|
| Position mass matrix | CRB copy/accumulation, DOF armature and CSR ancestor assembly | [mjwarp_mass_matrix.cu](../../../cuda/dynamics/mjwarp_mass_matrix.cu) | [mass_matrix.py](mass_matrix.py), [runtime binding](../../runtime/dynamics/mjwarp_mass_matrix.py); `mass_matrix=cuda` |
| Mass factorization | Compact1/scalar6/tile9 Cholesky, diagonal reciprocals | [mjwarp_mass_factor.cu](../../../cuda/dynamics/mjwarp_mass_factor.cu) | [mass_factor.py](mass_factor.py), [runtime binding](../../runtime/dynamics/mjwarp_mass_factor.py); `mass_factor=cuda` |
| Spatial velocity | Body `cvel`, DOF `cdof_dot`, unique branch output publication | [mjwarp_velocity.cu](../../../cuda/dynamics/mjwarp_velocity.cu) | [velocity.py](velocity.py), [runtime binding](../../runtime/dynamics/mjwarp_velocity.py); `com_vel=cuda` |
| Joint passive forces | FREE/BALL/SLIDE/HINGE spring and damping, quaternion difference, polynomial coefficients | [mjwarp_passive.cu](../../../cuda/dynamics/mjwarp_passive.cu): `joint_forces` | [passive.py](passive.py), [runtime binding](../../runtime/dynamics/mjwarp_passive.py); `passive=cuda` |
| Passive combination | Ordered sum of spring, damping, eligible gravcomp, fluid and adhesion | [mjwarp_passive.cu](../../../cuda/dynamics/mjwarp_passive.cu): `combine_forces` | Same passive selector; each contribution remains publicly stored |
| Bias RNE | Gravity root, velocity acceleration, inertial/Coriolis forces, reverse body sum and DOF projection | [mjwarp_bias.cu](../../../cuda/dynamics/mjwarp_bias.cu) | [velocity.py](velocity.py), [runtime binding](../../runtime/dynamics/mjwarp_bias.py); `bias=cuda` |
| Stateless actuation | Fixed/affine actuator force, control/force clamps, current CSR moment projection, actuator gravcomp and joint-force limits | [mjwarp_actuation.cu](../../../cuda/dynamics/mjwarp_actuation.cu) | [actuation.py](actuation.py), [runtime binding](../../runtime/dynamics/mjwarp_actuation.py); `actuation=cuda` |
| Smooth force | Force sum and external-wrench projection into `qfrc_smooth` | [mjwarp_smooth_force.cu](../../../cuda/dynamics/mjwarp_smooth_force.cu) | [acceleration.py](acceleration.py), [runtime binding](../../runtime/dynamics/mjwarp_smooth_force.py); `acceleration=cuda` |
| Mass solve | Compact1/scalar6/tile9 backsubstitution into smooth acceleration | [mjwarp_mass_solve.cu](../../../cuda/dynamics/mjwarp_mass_solve.cu) | [acceleration.py](acceleration.py), [runtime binding](../../runtime/dynamics/mjwarp_mass_solve.py); `mass_solve=cuda`, implied by CUDA acceleration |
| Newton initialization/stopping | Warmstart selection, reset, Jaref/Ma setup, convergence and done transitions | [mjwarp_solver_state.cu](../../../cuda/newton/mjwarp_solver_state.cu) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_solver_state.py); cumulative `newton=cuda_newton` |
| Newton constraint update | Constraint force/state and generalized-force projection on existing rows | [mjwarp_constraint.cu](../../../cuda/newton/mjwarp_constraint.cu) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_constraint.py) |
| Newton Hessian | Mass/constraint curvature assembly in the admitted dense elliptic domain | [mjwarp_hessian.cu](../../../cuda/newton/mjwarp_hessian.cu) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_hessian.py) |
| Newton direction | Gradient, factor/solve, direction norm and decrement; cuSolverDx for qualified small blocks | [mjwarp_direction.cu](../../../cuda/newton/mjwarp_direction.cu) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_direction.py) |
| Newton line search | M/J products, cost/derivatives, bracketing and accepted-step commit | [mjwarp_linesearch.cu](../../../cuda/newton/mjwarp_linesearch.cu), [shared products](../../../cuda/newton/mjwarp_products.cuh) | [newton.py](newton.py), [runtime binding](../../runtime/newton/mjwarp_linesearch.py) |
| Postconstraint RNE suffix | Solved acceleration propagation, inertial/Coriolis minus external wrench, reverse body sum | [mjwarp_postconstraint.cu](../../../cuda/dynamics/mjwarp_postconstraint.cu) | [postconstraint.py](postconstraint.py), [runtime binding](../../runtime/dynamics/mjwarp_postconstraint.py); `postconstraint=cuda` |
| Euler factor/solve | Factorization/backsubstitution of damping-adjusted mass matrix, private acceleration workspace | [mjwarp_mass_factor.cu](../../../cuda/dynamics/mjwarp_mass_factor.cu) | [mass_factor.py](mass_factor.py); part of `mass_factor=cuda`, other integration phases remain upstream |

Newton's cumulative selector controls the five Newton libraries together;
partial selectors remain available. Constraint update here operates on rows
already generated by upstream; it does not migrate row construction or collision.
cuSolverDx is an external numerical dependency, not a handwritten factor kernel.
The dense elliptic route follows upstream full Hessian rebuilds; no separate
migration of an inactive rank-update branch is claimed.

## Numerical work still executed upstream

The paths below are within the installed pinned `mujoco_warp/_src/` package.
The corresponding project dispatch files are linked so the retained boundary is
reviewable without assuming that a host wrapper means the mathematics migrated.

| Stage / remaining work | Upstream file and functions | Project call site |
|---|---|---|
| Body/geom/site kinematics; CoM, spatial inertia and DOF axes | `smooth.py::kinematics`, `com_pos` | [position.py::Position.position](position.py), via `forward.fwd_kinematics` |
| Camera/light poses; flex and tendon geometry | `smooth.py::camlight`, `flex`, `tendon` | [position.py](position.py) |
| Broad/narrow collision, contact filtering and generation | `collision_driver.py`, `bvh.py`, `collision_core.py`, `collision_primitive*.py`, `collision_convex.py`, `collision_sdf.py` | [position.py](position.py) |
| Constraint row/Jacobian/impedance/reference construction | `constraint.py::make_constraint` and its helpers | [position.py](position.py) |
| Tendon armature mass contribution | `smooth.py::tendon_armature` | [position.py](position.py) |
| Actuator transmission lengths/moments | `smooth.py::transmission` | [position.py](position.py) |
| Actuator/tendon velocities | `forward.py::_actuator_velocity`, `_tendon_velocity` | [velocity.py](velocity.py) |
| Tendon/flex passive forces | `passive.py` tendon/flex kernels | [passive.py](passive.py): whole-caller fallback for these domains |
| Body gravity-compensation generation | `passive.py::_gravity_force` | [passive_other.py::other_forces](passive_other.py) |
| Fluid force generation and body-to-DOF projection | `passive.py::_fluid`, `_fluid_force`; `support.py::apply_ft` | [passive_other.py](passive_other.py) |
| Adhesion-force generation | `passive.py::_qfrc_adhesion` | [passive_other.py](passive_other.py) |
| User passive callback | `model.callback.passive` | [passive.py::Passive.enqueue](passive.py) after the selected sum, preserving upstream early-return semantics |
| Tendon bias contribution | `smooth.py::tendon_bias` | [velocity.py](velocity.py) |
| Stateful/custom/muscle/DC-motor actuation, tendon limits and rotational setpoint wrapping | Complete `forward.py::fwd_actuation` for unqualified domains | [actuation.py](actuation.py) visible whole-caller fallback |
| Delayed control interpolation and act_dyn/act_gain/act_bias callbacks | `history.py::read_ctrl_delayed`, `model.callback` | [actuation.py::Actuation.enqueue](actuation.py); callback order retained between force and projection |
| Applied/contact/equality/tendon external-wrench assembly after solve | `smooth.py` external-wrench builders | [postconstraint_external.py::assemble_external](postconstraint_external.py) |
| Position/velocity/acceleration sensors; energy | `sensor.py`, `forward.py::_energy_pos`, `_energy_vel` | [stages.py](stages.py) |
| Euler damping derivative and mass diagonal modification | `forward.py::_compute_damping_deriv`, `_euler_damp_qfrc` | [mass_factor.py::MassFactors.integrate](mass_factor.py) |
| State/activation/time integration, quaternion position integration, warmstart publication and history | `forward.py::_advance`, `_next_activation`, `_next_velocity`, `_next_position`, `_next_time_builder`; `history.py` | [mass_factor.py](mass_factor.py), [stages.py](stages.py) |
| Optional noslip solver | `solver.py::noslip` | [newton.py](newton.py); all shipped130 tasks disable noslip |
| Sparse/pyramidal/sleeping/discrete, unsupported mass families and other unqualified domains | Complete pinned upstream callers | Each selector's visible fallback; no migration is claimed for these paths |
| Reset and public forward refresh | Public `mujoco_warp.forward` | [runtime.py::Runtime.forward](runtime.py) |

## Retained infrastructure and project components

| Component | Current ownership / source |
|---|---|
| CUDA Graph loops, Warp Model/Data and per-solve scratch allocation | Upstream Warp/MJWarp infrastructure, retained deliberately; [newton.py](newton.py), [runtime.py](runtime.py) own context/Graph lifetimes |
| Scene import, capacity/profile admission, reset/public holds | Project integration: [model.py](model.py), [environment.py](environment.py), [build.py](build.py) |
| Panda OSC | Existing project CUDA, reused: [mjwarp_controller.cu](../../../cuda/control/mjwarp_controller.cu), [controller.py](controller.py); `mjw.jac`/upstream support still supplies its Jacobian input |
| Controller gather/scatter, goals and work counters | Project Warp/CUDA bridge: [kernels.py](kernels.py); it is not a migrated physics stage |
| Policy observation sampling and LIBERO predicates | Existing project sampler: [device.py](../../adapters/observations/device.py), [environment.py](environment.py), [facade.py](facade.py); original task Python predicates retained |
| Camera image rendering | Outside the current MJWarp migration qualification; default tests disable cameras. Camera poses above still come from upstream. |

Independent CMake leaves are in [cmake/](../../../cmake/); stable local qualifiers
are exposed by `tools/validate_gpu.py`. The passive qualifier covers joint forces, final combination and their leaf
composition. The actuation qualifier covers stateless force, CSR projection,
joint limits and composition on frozen N=2 inputs. The shipped130 tasks share
one XML actuator pattern: seven torque motors and two kp1000 position grippers.
CUDA dimensions remain dynamic and also admit qualified differing actuator/control
counts and fixed/affine parameters. These local qualifiers do not establish
complete130-task trajectories or measure performance.
