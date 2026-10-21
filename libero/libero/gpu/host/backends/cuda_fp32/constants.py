# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Adapted from MJWarp c384ac51df67eb92ab7657cc472ff7494dc036bf types.py.
# Plain enums/metadata only; no Warp package or code generator is used.


import enum
import mujoco
from . import device as storage

if mujoco.__version__ != '3.16.0':
    raise ValueError('Independent CUDA FP32 requires an isolated MuJoCo3.16.0 environment; install Flash-LIBERO[cuda-fp32]')

MJ_MINVAL = mujoco.mjMINVAL

MJ_MAXVAL = mujoco.mjMAXVAL

MJ_MINIMP = mujoco.mjMINIMP

MJ_MAXIMP = mujoco.mjMAXIMP

MJ_MAXCONPAIR = mujoco.mjMAXCONPAIR

MJ_MINMU = mujoco.mjMINMU

MJ_MINAWAKE = mujoco.mjMINAWAKE

MJ_MAX_EPAHORIZON = 24

MJ_MAX_EPAFACES = 5

TILE_SIZE_JTDAJ_SPARSE = 16

TILE_SIZE_JTDAJ_DENSE = 16

M_BLOCK_DENSE_MAX = 64

M_BLOCK_SCALAR_MAX = 6

Q_LD_BLOCK_COMPACT = -2

Q_LD_BLOCK_SPARSE = -1

class BroadphaseType(enum.IntEnum):
    """Type of broadphase algorithm.

  Attributes:
     NXN: Broad phase checking all pairs
     SAP_TILE: Sweep and prune broad phase using tile sort
     SAP_SEGMENTED: Sweep and prune broad phase using segment sort
  """
    NXN = 0
    SAP_TILE = 1
    SAP_SEGMENTED = 2

class BroadphaseFilter(enum.IntFlag):
    """Bitmask specifying which collision functions to run during broadphase.

  Attributes:
    PLANE: collision between bounding sphere and plane
    SPHERE: collision between bounding spheres
    AABB: collision between axis-aligned bounding boxes
    OBB: collision between oriented bounding boxes
  """
    PLANE = 1 << 0
    SPHERE = 1 << 1
    AABB = 1 << 2
    OBB = 1 << 3

class OverflowType(enum.IntFlag):
    """Bitmask for physics and collision overflows.

  Attributes:
    NONE: no overflow
    NEFC: nefc > njmax
    NJMAX_NNZ: njmax_nnz overflow
    BROADPHASE: broadphase overflow / flex broadphase overflow
    NARROWPHASE: narrowphase overflow / flex narrowphase / contact overflow
    CCD: CCD overflow / flex CCD overflow
    HFIELD: height field collision overflow
    CONTACT_MATCH: contact match sensor overflow
    NVMAX: nvmax overflow (islands)
    EPA_HORIZON: EPA horizon buffer overflow
    ITERATIONS: solver iteration limit reached
    LS_ITERATIONS: linesearch iteration limit reached
    TACTILE: tactile sensor collision pair overflow
    ALL: all overflows
  """
    NONE = 0
    NEFC = 1 << 0
    NJMAX_NNZ = 1 << 1
    BROADPHASE = 1 << 2
    NARROWPHASE = 1 << 3
    CCD = 1 << 4
    HFIELD = 1 << 5
    CONTACT_MATCH = 1 << 6
    NVMAX = 1 << 7
    EPA_HORIZON = 1 << 8
    ITERATIONS = 1 << 9
    LS_ITERATIONS = 1 << 10
    TACTILE = 1 << 11
    ALL = NEFC | NJMAX_NNZ | BROADPHASE | NARROWPHASE | CCD | HFIELD | CONTACT_MATCH | NVMAX | EPA_HORIZON | ITERATIONS | LS_ITERATIONS | TACTILE

class CamLightType(enum.IntEnum):
    """Type of camera light.

  Attributes:
    FIXED: pos and rot fixed in body
    TRACK: pos tracks body, rot fixed in global
    TRACKCOM: pos tracks subtree com, rot fixed in body
    TARGETBODY: pos fixed in body, rot tracks target body
    TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
  """
    FIXED = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    TRACK = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
    TRACKCOM = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    TARGETBODY = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    TARGETBODYCOM = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM

class ProjectionType(enum.IntEnum):
    """Type of camera projection.

  Attributes:
    PERSPECTIVE: perspective projection
    ORTHOGRAPHIC: orthographic projection
  """
    PERSPECTIVE = mujoco.mjtProjection.mjPROJ_PERSPECTIVE
    ORTHOGRAPHIC = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC

class Stage(enum.IntEnum):
    """Computation stage.

  Attributes:
    POS: position-dependent
    VEL: velocity-dependent
    ACC: acceleration/force-dependent
  """
    POS = mujoco.mjtStage.mjSTAGE_POS
    VEL = mujoco.mjtStage.mjSTAGE_VEL
    ACC = mujoco.mjtStage.mjSTAGE_ACC

class DataType(enum.IntFlag):
    """Sensor data types.

  Attributes:
    REAL: real values, no constraints
    POSITIVE: positive values, 0 or negative: inactive
  """
    REAL = mujoco.mjtDataType.mjDATATYPE_REAL
    POSITIVE = mujoco.mjtDataType.mjDATATYPE_POSITIVE

class DisableBit(enum.IntFlag):
    """Disable default feature bitflags.

  Attributes:
    CONSTRAINT:   entire constraint solver
    EQUALITY:     equality constraints
    FRICTIONLOSS: joint and tendon frictionloss constraints
    LIMIT:        joint and tendon limit constraints
    CONTACT:      contact constraints
    SPRING:       passive spring forces
    DAMPER:       passive damper forces
    GRAVITY:      gravitational forces
    CLAMPCTRL:    clamp control to specified range
    WARMSTART:    warmstart constraint solver
    FILTERPARENT: disable collisions between parent and child bodies
    ACTUATION:    apply actuation forces
    REFSAFE:      integrator safety: make ref[0]>=2*timestep
    SENSOR:       sensors
    EULERDAMP:    implicit damping for Euler integration
    NATIVECCD:    native convex collision detection (ignored in MJWarp)
    ISLAND:       constraint islands
    MULTICCD:     disable multiple contacts with CCD
  """
    CONSTRAINT = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY
    FRICTIONLOSS = mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
    LIMIT = mujoco.mjtDisableBit.mjDSBL_LIMIT
    CONTACT = mujoco.mjtDisableBit.mjDSBL_CONTACT
    SPRING = mujoco.mjtDisableBit.mjDSBL_SPRING
    DAMPER = mujoco.mjtDisableBit.mjDSBL_DAMPER
    GRAVITY = mujoco.mjtDisableBit.mjDSBL_GRAVITY
    CLAMPCTRL = mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
    WARMSTART = mujoco.mjtDisableBit.mjDSBL_WARMSTART
    FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION
    REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE
    SENSOR = mujoco.mjtDisableBit.mjDSBL_SENSOR
    EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    NATIVECCD = mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    ISLAND = mujoco.mjtDisableBit.mjDSBL_ISLAND
    MULTICCD = mujoco.mjtDisableBit.mjDSBL_MULTICCD

class EnableBit(enum.IntFlag):
    """Enable optional feature bitflags.

  Attributes:
    ENERGY: energy computation
    INVDISCRETE: discrete-time inverse dynamics
    SLEEP: sleeping
  """
    ENERGY = mujoco.mjtEnableBit.mjENBL_ENERGY
    INVDISCRETE = mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    SLEEP = mujoco.mjtEnableBit.mjENBL_SLEEP

class SleepPolicy(enum.IntEnum):
    """Per-tree sleep policy.

  Attributes:
    AUTO: compiler chooses sleep policy
    AUTO_NEVER: compiler sleep policy: never
    AUTO_ALLOWED: compiler sleep policy: allowed
  """
    AUTO = mujoco.mjtSleepPolicy.mjSLEEP_AUTO
    AUTO_NEVER = mujoco.mjtSleepPolicy.mjSLEEP_AUTO_NEVER
    AUTO_ALLOWED = mujoco.mjtSleepPolicy.mjSLEEP_AUTO_ALLOWED

class SleepState(enum.IntEnum):
    """Sleep state for bodies.

  Attributes:
    STATIC: body is static (world body or mocap)
    ASLEEP: body is asleep
    AWAKE: body is awake
  """
    STATIC = mujoco.mjtSleepState.mjS_STATIC
    ASLEEP = mujoco.mjtSleepState.mjS_ASLEEP
    AWAKE = mujoco.mjtSleepState.mjS_AWAKE

class TrnType(enum.IntEnum):
    """Type of actuator transmission.

  Attributes:
    JOINT: force on joint
    JOINTINPARENT: force on joint, expressed in parent frame
    SLIDERCRANK: force via slider-crank linkage
    TENDON: force on tendon
    BODY: adhesion force on body's geoms
    SITE: force on site
  """
    JOINT = mujoco.mjtTrn.mjTRN_JOINT
    JOINTINPARENT = mujoco.mjtTrn.mjTRN_JOINTINPARENT
    SLIDERCRANK = mujoco.mjtTrn.mjTRN_SLIDERCRANK
    TENDON = mujoco.mjtTrn.mjTRN_TENDON
    BODY = mujoco.mjtTrn.mjTRN_BODY
    SITE = mujoco.mjtTrn.mjTRN_SITE

class DynType(enum.IntEnum):
    """Type of actuator dynamics.

  Attributes:
    NONE: no internal dynamics; ctrl specifies force
    INTEGRATOR: integrator: da/dt = u
    FILTER: linear filter: da/dt = (u-a) / tau
    FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
    MUSCLE: piece-wise linear filter with two time constants
    USER: user-defined dynamics via act_dyn_callback
    DCMOTOR: DC motor dynamics
  """
    NONE = mujoco.mjtDyn.mjDYN_NONE
    INTEGRATOR = mujoco.mjtDyn.mjDYN_INTEGRATOR
    FILTER = mujoco.mjtDyn.mjDYN_FILTER
    FILTEREXACT = mujoco.mjtDyn.mjDYN_FILTEREXACT
    MUSCLE = mujoco.mjtDyn.mjDYN_MUSCLE
    USER = mujoco.mjtDyn.mjDYN_USER
    DCMOTOR = mujoco.mjtDyn.mjDYN_DCMOTOR

class GainType(enum.IntEnum):
    """Type of actuator gain.

  Attributes:
    FIXED: fixed gain
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle FLV curve computed by muscle_gain
    USER: user-defined gain via act_gain_callback
    DCMOTOR: DC motor gain
  """
    FIXED = mujoco.mjtGain.mjGAIN_FIXED
    AFFINE = mujoco.mjtGain.mjGAIN_AFFINE
    MUSCLE = mujoco.mjtGain.mjGAIN_MUSCLE
    USER = mujoco.mjtGain.mjGAIN_USER
    DCMOTOR = mujoco.mjtGain.mjGAIN_DCMOTOR
    SO3 = mujoco.mjtGain.mjGAIN_SO3

class BiasType(enum.IntEnum):
    """Type of actuator bias.

  Attributes:
    NONE: no bias
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle passive force computed by muscle_bias
    USER: user-defined bias via act_bias_callback
    DCMOTOR: DC motor back-EMF bias
  """
    NONE = mujoco.mjtBias.mjBIAS_NONE
    AFFINE = mujoco.mjtBias.mjBIAS_AFFINE
    MUSCLE = mujoco.mjtBias.mjBIAS_MUSCLE
    USER = mujoco.mjtBias.mjBIAS_USER
    DCMOTOR = mujoco.mjtBias.mjBIAS_DCMOTOR

class CtrlInput(enum.IntFlag):
    """Control input signature bitflags for actuators.

  Attributes:
    POS: position setpoint input
    VEL: velocity setpoint input
    FF: feedforward input
    VOLTAGE: raw terminal voltage input
    NONE: explicitly no inputs (purely passive)
  """
    POS = mujoco.mjtCtrlInput.mjINPUT_POS
    VEL = mujoco.mjtCtrlInput.mjINPUT_VEL
    FF = mujoco.mjtCtrlInput.mjINPUT_FF
    VOLTAGE = mujoco.mjtCtrlInput.mjINPUT_VOLTAGE
    NONE = mujoco.mjtCtrlInput.mjINPUT_NONE

class CtrlChart(enum.IntEnum):
    """Orientation chart type for SO(3) actuators.

  Attributes:
    EXPMAP: exponential-map orientation target (3 controls)
    QUAT: quaternion orientation target (4 controls)
  """
    EXPMAP = mujoco.mjtCtrlChart.mjCHART_EXPMAP
    QUAT = mujoco.mjtCtrlChart.mjCHART_QUAT

class JointType(enum.IntEnum):
    """Type of degree of freedom.

  Attributes:
    FREE:  global position and orientation (quat)       (7,)
    BALL:  orientation (quat) relative to parent        (4,)
    SLIDE: sliding distance along body-fixed axis       (1,)
    HINGE: rotation angle (rad) around body-fixed axis  (1,)
  """
    FREE = mujoco.mjtJoint.mjJNT_FREE
    BALL = mujoco.mjtJoint.mjJNT_BALL
    SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
    HINGE = mujoco.mjtJoint.mjJNT_HINGE

class ConeType(enum.IntEnum):
    """Type of friction cone.

  Attributes:
    PYRAMIDAL: pyramidal
    ELLIPTIC: elliptic
  """
    PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL
    ELLIPTIC = mujoco.mjtCone.mjCONE_ELLIPTIC

class IntegratorType(enum.IntEnum):
    """Integrator mode.

  Attributes:
    EULER: semi-implicit Euler
    RK4: 4th-order Runge Kutta
    IMPLICITFAST: implicit in velocity, no rne derivative
    IMPLICIT: implicit in velocity, with rne derivative
    DISCRETE: discrete step map: constraint solve in the effective metric
  """
    EULER = mujoco.mjtIntegrator.mjINT_EULER
    RK4 = mujoco.mjtIntegrator.mjINT_RK4
    IMPLICITFAST = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    IMPLICIT = mujoco.mjtIntegrator.mjINT_IMPLICIT
    DISCRETE = mujoco.mjtIntegrator.mjINT_DISCRETE

class GeomType(enum.IntEnum):
    """Type of geometry.

  Attributes:
    PLANE: plane
    HFIELD: heightfield
    SPHERE: sphere
    CAPSULE: capsule
    ELLIPSOID: ellipsoid
    CYLINDER: cylinder
    BOX: box
    MESH: mesh
    SDF: sdf
    FLEX: flex
  """
    PLANE = mujoco.mjtGeom.mjGEOM_PLANE
    HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD
    SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
    CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
    ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    MESH = mujoco.mjtGeom.mjGEOM_MESH
    SDF = mujoco.mjtGeom.mjGEOM_SDF
    FLEX = mujoco.mjtGeom.mjGEOM_FLEX
    TRIANGLE = 999

class CollisionType(enum.IntEnum):
    """Type of narrowphase collision.

  Attributes:
    PRIMITIVE: primitive collision
    CONVEX: convex collision (CCD)
    SDF: sdf collision
  """
    PRIMITIVE = 0
    CONVEX = 1
    SDF = 2

class SolverType(enum.IntEnum):
    """Constraint solver algorithm.

  Attributes:
    CG: Conjugate gradient (primal)
    NEWTON: Newton (primal)
  """
    CG = mujoco.mjtSolver.mjSOL_CG
    NEWTON = mujoco.mjtSolver.mjSOL_NEWTON

class ConstraintState(enum.IntEnum):
    """State of constraint.

  Attributes:
    SATISFIED: constraint satisfied, zero cost (limit, contact)
    QUADRATIC: quadratic cost (equality, friction, limit, contact)
    LINEARNEG: linear cost, negative side (friction)
    LINEARPOS: linear cost, positive side (friction)
    CONE: square distance to cone cost (elliptic contact)
  """
    SATISFIED = mujoco.mjtConstraintState.mjCNSTRSTATE_SATISFIED
    QUADRATIC = mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC
    LINEARNEG = mujoco.mjtConstraintState.mjCNSTRSTATE_LINEARNEG
    LINEARPOS = mujoco.mjtConstraintState.mjCNSTRSTATE_LINEARPOS
    CONE = mujoco.mjtConstraintState.mjCNSTRSTATE_CONE

class ConstraintType(enum.IntEnum):
    """Type of constraint.

  Attributes:
    EQUALITY: equality constraint
    FRICTION_DOF: dof friction
    FRICTION_TENDON: tendon friction
    LIMIT_JOINT: joint limit
    LIMIT_TENDON: tendon limit
    CONTACT_FRICTIONLESS: frictionless contact
    CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
    CONTACT_ELLIPTIC: frictional contact, elliptic friction cone
  """
    EQUALITY = mujoco.mjtConstraint.mjCNSTR_EQUALITY
    FRICTION_DOF = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    FRICTION_TENDON = mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON
    LIMIT_JOINT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
    LIMIT_TENDON = mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
    CONTACT_FRICTIONLESS = mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS
    CONTACT_PYRAMIDAL = mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
    CONTACT_ELLIPTIC = mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC

class SensorType(enum.IntEnum):
    """Type of sensor.

  Attributes:
    MAGNETOMETER: magnetometer
    CAMPROJECTION: camera projection
    RANGEFINDER: scalar distance to nearest geom or site along z-axis
    JOINTPOS: joint position
    TENDONPOS: scalar tendon position
    ACTUATORPOS: actuator position
    BALLQUAT: ball joint orientation
    JOINTLIMITPOS: joint limit distance-margin
    TENDONLIMITPOS: tendon limit distance-margin
    FRAMEPOS: frame position
    FRAMEXAXIS: frame x-axis
    FRAMEYAXIS: frame y-axis
    FRAMEZAXIS: frame z-axis
    FRAMEQUAT: frame orientation, represented as quaternion
    SUBTREECOM: subtree center of mass
    GEOMDIST: signed distance between two geoms
    GEOMNORMAL: normal direction between two geoms
    GEOMFROMTO: segment between two geoms
    INSIDESITE: 1 if object is inside site, 0 otherwise
    E_POTENTIAL: potential energy
    E_KINETIC: kinetic energy
    CLOCK: simulation time
    VELOCIMETER: 3D linear velocity, in local frame
    GYRO: 3D angular velocity, in local frame
    JOINTVEL: joint velocity
    TENDONVEL: scalar tendon velocity
    ACTUATORVEL: actuator velocity
    BALLANGVEL: ball joint angular velocity
    JOINTLIMITVEL: joint limit velocity
    TENDONLIMITVEL: tendon limit velocity
    FRAMELINVEL: 3D linear velocity
    FRAMEANGVEL: 3D angular velocity
    SUBTREELINVEL: subtree linear velocity
    SUBTREEANGMOM: subtree angular momentum
    TOUCH: scalar contact normal forces summed over sensor zone
    CONTACT: contacts which occurred during the simulation
    ACCELEROMETER: accelerometer
    FORCE: force
    TORQUE: torque
    ACTUATORFRC: scalar actuator force, measured at the joint
    TENDONACTFRC: scalar actuator force, measured at the tendon
    JOINTACTFRC: scalar actuator force, measured at the joint
    JOINTLIMITFRC: joint limit force
    TENDONLIMITFRC: tendon limit force
    FRAMELINACC: 3D linear acceleration
    FRAMEANGACC: 3D angular acceleration
    TACTILE: tactile sensor
    USER: user-defined sensor via sensor_callback
  """
    MAGNETOMETER = mujoco.mjtSensor.mjSENS_MAGNETOMETER
    CAMPROJECTION = mujoco.mjtSensor.mjSENS_CAMPROJECTION
    RANGEFINDER = mujoco.mjtSensor.mjSENS_RANGEFINDER
    JOINTPOS = mujoco.mjtSensor.mjSENS_JOINTPOS
    TENDONPOS = mujoco.mjtSensor.mjSENS_TENDONPOS
    ACTUATORPOS = mujoco.mjtSensor.mjSENS_ACTUATORPOS
    BALLQUAT = mujoco.mjtSensor.mjSENS_BALLQUAT
    JOINTLIMITPOS = mujoco.mjtSensor.mjSENS_JOINTLIMITPOS
    TENDONLIMITPOS = mujoco.mjtSensor.mjSENS_TENDONLIMITPOS
    FRAMEPOS = mujoco.mjtSensor.mjSENS_FRAMEPOS
    FRAMEXAXIS = mujoco.mjtSensor.mjSENS_FRAMEXAXIS
    FRAMEYAXIS = mujoco.mjtSensor.mjSENS_FRAMEYAXIS
    FRAMEZAXIS = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
    FRAMEQUAT = mujoco.mjtSensor.mjSENS_FRAMEQUAT
    SUBTREECOM = mujoco.mjtSensor.mjSENS_SUBTREECOM
    GEOMDIST = mujoco.mjtSensor.mjSENS_GEOMDIST
    GEOMNORMAL = mujoco.mjtSensor.mjSENS_GEOMNORMAL
    GEOMFROMTO = mujoco.mjtSensor.mjSENS_GEOMFROMTO
    INSIDESITE = mujoco.mjtSensor.mjSENS_INSIDESITE
    E_POTENTIAL = mujoco.mjtSensor.mjSENS_E_POTENTIAL
    E_KINETIC = mujoco.mjtSensor.mjSENS_E_KINETIC
    CLOCK = mujoco.mjtSensor.mjSENS_CLOCK
    VELOCIMETER = mujoco.mjtSensor.mjSENS_VELOCIMETER
    GYRO = mujoco.mjtSensor.mjSENS_GYRO
    JOINTVEL = mujoco.mjtSensor.mjSENS_JOINTVEL
    TENDONVEL = mujoco.mjtSensor.mjSENS_TENDONVEL
    ACTUATORVEL = mujoco.mjtSensor.mjSENS_ACTUATORVEL
    BALLANGVEL = mujoco.mjtSensor.mjSENS_BALLANGVEL
    JOINTLIMITVEL = mujoco.mjtSensor.mjSENS_JOINTLIMITVEL
    TENDONLIMITVEL = mujoco.mjtSensor.mjSENS_TENDONLIMITVEL
    FRAMELINVEL = mujoco.mjtSensor.mjSENS_FRAMELINVEL
    FRAMEANGVEL = mujoco.mjtSensor.mjSENS_FRAMEANGVEL
    SUBTREELINVEL = mujoco.mjtSensor.mjSENS_SUBTREELINVEL
    SUBTREEANGMOM = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
    TOUCH = mujoco.mjtSensor.mjSENS_TOUCH
    CONTACT = mujoco.mjtSensor.mjSENS_CONTACT
    ACCELEROMETER = mujoco.mjtSensor.mjSENS_ACCELEROMETER
    FORCE = mujoco.mjtSensor.mjSENS_FORCE
    TORQUE = mujoco.mjtSensor.mjSENS_TORQUE
    ACTUATORFRC = mujoco.mjtSensor.mjSENS_ACTUATORFRC
    TENDONACTFRC = mujoco.mjtSensor.mjSENS_TENDONACTFRC
    JOINTACTFRC = mujoco.mjtSensor.mjSENS_JOINTACTFRC
    JOINTLIMITFRC = mujoco.mjtSensor.mjSENS_JOINTLIMITFRC
    TENDONLIMITFRC = mujoco.mjtSensor.mjSENS_TENDONLIMITFRC
    FRAMELINACC = mujoco.mjtSensor.mjSENS_FRAMELINACC
    FRAMEANGACC = mujoco.mjtSensor.mjSENS_FRAMEANGACC
    TACTILE = mujoco.mjtSensor.mjSENS_TACTILE
    USER = mujoco.mjtSensor.mjSENS_USER

class ObjType(enum.IntEnum):
    """Type of object.

  Attributes:
    UNKNOWN: unknown object type
    BODY: body
    XBODY: body, used to access regular frame instead of i-frame
    GEOM: geom
    FLEX: flex
    SITE: site
    CAMERA: camera
  """
    UNKNOWN = mujoco.mjtObj.mjOBJ_UNKNOWN
    BODY = mujoco.mjtObj.mjOBJ_BODY
    XBODY = mujoco.mjtObj.mjOBJ_XBODY
    GEOM = mujoco.mjtObj.mjOBJ_GEOM
    FLEX = mujoco.mjtObj.mjOBJ_FLEX
    SITE = mujoco.mjtObj.mjOBJ_SITE
    CAMERA = mujoco.mjtObj.mjOBJ_CAMERA

class EqType(enum.IntEnum):
    """Type of equality constraint.

  Attributes:
    CONNECT: connect two bodies at a point (ball joint)
    JOINT: couple the values of two scalar joints with cubic
    WELD: fix relative position and orientation of two bodies
    TENDON: couple the lengths of two tendons with cubic
    FLEX: couple the edge lengths of a flex
    FLEXSTRAIN: strain constraint for interpolated flex
  """
    CONNECT = mujoco.mjtEq.mjEQ_CONNECT
    WELD = mujoco.mjtEq.mjEQ_WELD
    JOINT = mujoco.mjtEq.mjEQ_JOINT
    TENDON = mujoco.mjtEq.mjEQ_TENDON
    FLEX = mujoco.mjtEq.mjEQ_FLEX
    FLEXSTRAIN = mujoco.mjtEq.mjEQ_FLEXSTRAIN

class WrapType(enum.IntEnum):
    """Type of tendon wrapping object.

  Attributes:
    JOINT: constant moment arm
    PULLEY: pulley used to split tendon
    SITE: pass through site
    SPHERE: wrap around sphere
    CYLINDER: wrap around (infinite) cylinder
  """
    JOINT = mujoco.mjtWrap.mjWRAP_JOINT
    PULLEY = mujoco.mjtWrap.mjWRAP_PULLEY
    SITE = mujoco.mjtWrap.mjWRAP_SITE
    SPHERE = mujoco.mjtWrap.mjWRAP_SPHERE
    CYLINDER = mujoco.mjtWrap.mjWRAP_CYLINDER

class State(enum.IntEnum):
    """State component elements as integer bitflags.

  Includes several convenient combinations of these flags.

  Attributes:
    TIME: time
    QPOS: position
    QVEL: velocity
    ACT: actuator activation
    HISTORY: delay/interval history buffers
    WARMSTART: acceleration used for warmstart
    CTRL: control
    QFRC_APPLIED: applied generalized force
    XFRC_APPLIED: applied Cartesian force/torque
    EQ_ACTIVE: enable/disable constraints
    MOCAP_POS: positions of mocap bodies
    MOCAP_QUAT: orientations of mocap bodies
    NSTATE: number of state elements
    PHYSICS: TIME | QPOS | QVEL | ACT | HISTORY
    FULLPHYSICS: TIME | PHYSICS | PLUGIN
    USER: CTRL | QFRC_APPLIED | XFRC_APPLIED | EQ_ACTIVE | MOCAP_POS | MOCAP_QUAT | USERDATA
    INTEGRATION: FULLPHYSICS | USER | WARMSTART
  """
    TIME = mujoco.mjtState.mjSTATE_TIME
    QPOS = mujoco.mjtState.mjSTATE_QPOS
    QVEL = mujoco.mjtState.mjSTATE_QVEL
    ACT = mujoco.mjtState.mjSTATE_ACT
    HISTORY = mujoco.mjtState.mjSTATE_HISTORY
    WARMSTART = mujoco.mjtState.mjSTATE_WARMSTART
    CTRL = mujoco.mjtState.mjSTATE_CTRL
    QFRC_APPLIED = mujoco.mjtState.mjSTATE_QFRC_APPLIED
    XFRC_APPLIED = mujoco.mjtState.mjSTATE_XFRC_APPLIED
    EQ_ACTIVE = mujoco.mjtState.mjSTATE_EQ_ACTIVE
    MOCAP_POS = mujoco.mjtState.mjSTATE_MOCAP_POS
    MOCAP_QUAT = mujoco.mjtState.mjSTATE_MOCAP_QUAT
    NSTATE = mujoco.mjtState.mjNSTATE
    PHYSICS = mujoco.mjtState.mjSTATE_PHYSICS
    FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS
    USER = mujoco.mjtState.mjSTATE_USER
    INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION
    USERDATA = mujoco.mjtState.mjSTATE_USERDATA

class ContactType(enum.IntFlag):
    """Type of contact.

  Attributes:
    CONSTRAINT: contact for constraint solver
    SENSOR: contact for collision sensor (GEOMDIST, GEOMNORMAL, GEOMFROMTO)
    PASSIVE: passive contact for effective metric
  """
    CONSTRAINT = 1 << 0
    SENSOR = 1 << 1
    PASSIVE = 1 << 2

vec5=storage.DType("vec5","float32",(5,))
vec10=storage.DType("vec10","float32",(10,))


MJ_COLLISION_TABLE = {(GeomType.PLANE, GeomType.SPHERE): CollisionType.PRIMITIVE, (GeomType.PLANE, GeomType.CAPSULE): CollisionType.PRIMITIVE, (GeomType.PLANE, GeomType.ELLIPSOID): CollisionType.PRIMITIVE, (GeomType.PLANE, GeomType.CYLINDER): CollisionType.PRIMITIVE, (GeomType.PLANE, GeomType.BOX): CollisionType.PRIMITIVE, (GeomType.PLANE, GeomType.MESH): CollisionType.PRIMITIVE, (GeomType.HFIELD, GeomType.SPHERE): CollisionType.CONVEX, (GeomType.HFIELD, GeomType.CAPSULE): CollisionType.CONVEX, (GeomType.HFIELD, GeomType.ELLIPSOID): CollisionType.CONVEX, (GeomType.HFIELD, GeomType.CYLINDER): CollisionType.CONVEX, (GeomType.HFIELD, GeomType.BOX): CollisionType.CONVEX, (GeomType.HFIELD, GeomType.MESH): CollisionType.CONVEX, (GeomType.SPHERE, GeomType.SPHERE): CollisionType.PRIMITIVE, (GeomType.SPHERE, GeomType.CAPSULE): CollisionType.PRIMITIVE, (GeomType.SPHERE, GeomType.ELLIPSOID): CollisionType.CONVEX, (GeomType.SPHERE, GeomType.CYLINDER): CollisionType.PRIMITIVE, (GeomType.SPHERE, GeomType.BOX): CollisionType.PRIMITIVE, (GeomType.SPHERE, GeomType.MESH): CollisionType.CONVEX, (GeomType.CAPSULE, GeomType.CAPSULE): CollisionType.PRIMITIVE, (GeomType.CAPSULE, GeomType.ELLIPSOID): CollisionType.CONVEX, (GeomType.CAPSULE, GeomType.CYLINDER): CollisionType.CONVEX, (GeomType.CAPSULE, GeomType.BOX): CollisionType.PRIMITIVE, (GeomType.CAPSULE, GeomType.MESH): CollisionType.CONVEX, (GeomType.ELLIPSOID, GeomType.ELLIPSOID): CollisionType.CONVEX, (GeomType.ELLIPSOID, GeomType.CYLINDER): CollisionType.CONVEX, (GeomType.ELLIPSOID, GeomType.BOX): CollisionType.CONVEX, (GeomType.ELLIPSOID, GeomType.MESH): CollisionType.CONVEX, (GeomType.CYLINDER, GeomType.CYLINDER): CollisionType.CONVEX, (GeomType.CYLINDER, GeomType.BOX): CollisionType.CONVEX, (GeomType.CYLINDER, GeomType.MESH): CollisionType.CONVEX, (GeomType.BOX, GeomType.BOX): CollisionType.CONVEX, (GeomType.BOX, GeomType.MESH): CollisionType.CONVEX, (GeomType.MESH, GeomType.MESH): CollisionType.CONVEX}
