"""Production model preparation, separate from CPU reference simulation.

The original task constructor builds XML and placement/goal definitions. Its
simulation creation, renderer, reset and observables initialization are replaced
on this one instance before construction. No global robosuite class is patched.
The returned shell is deliberately not a runnable environment; a GPU lifecycle
must supply physical initialization and controller/observation state.
"""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import mujoco
import numpy as np

from .cold_assets import ReadOnlyModelSnapshot


class ModelOnlyExecutionError(RuntimeError):
    """A caller tried to execute a model-only shell as a simulator."""


class _NoSimulation:
    physics_backend = "model_only"

    def __getattr__(self, name):
        raise ModelOnlyExecutionError(
            "Model-only construction has no simulation; supply a GPU runtime "
            "before accessing sim." + name
        )


@dataclass
class BuiltTaskModel:
    native_model: Any
    xml: str
    domain_shell: Any
    metadata: dict

    @property
    def problem_name(self):
        return self.metadata["problem_name"]


    def facade_metadata(self, *, copy_arrays=True):
        """Return model metadata; borrowed arrays require immediate freezing."""
        model = self.native_model
        out = dict(self.metadata["dimensions"] )
        out["xml"] = self.xml
        out["nlight"] = int(model.nlight)
        out["options"] = dict(self.metadata["options"] )
        out["stat"] = dict(self.metadata["stat"] )
        out["native_write_schema"] = dict(self.metadata.get("native_write_schema", {}))
        for name in self.metadata["static_array_inventory"]:
            value = getattr(model, name)
            out[name] = np.array(value, copy=True) if copy_arrays else value
        for kind, mapping in self.metadata["name_maps"].items():
            out[kind + "_name_to_id"] = dict(mapping)
        out["joint_qpos_slices"] = dict(self.metadata["joint_qpos_slices"])
        velocity_sizes = {int(mujoco.mjtJoint.mjJNT_FREE): 6,
                          int(mujoco.mjtJoint.mjJNT_BALL): 3,
                          int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
                          int(mujoco.mjtJoint.mjJNT_HINGE): 1}
        out["joint_qvel_slices"] = {
            name: [int(model.jnt_dofadr[index]),
                   int(model.jnt_dofadr[index]) + velocity_sizes[int(model.jnt_type[index])]]
            for name, index in self.metadata["name_maps"]["joint"].items()
        }
        return out

def _array_digest(arrays):
    digest = sha256()
    for name, array in sorted(arrays.items()):
        arr = np.ascontiguousarray(array)
        if arr.dtype.hasobject:
            raise ValueError("Object arrays cannot define a stable model identity: " + name)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(arr.dtype.str.encode("ascii") + b"\0")
        digest.update(str(arr.shape).encode("ascii") + b"\0")
        digest.update(arr.tobytes())
    return digest.hexdigest()


def native_model_metadata(model, *, return_snapshot=False, snapshot_pool=None, reference_snapshot=None):
    """Describe real compiled static data without creating/forwarding MjData."""
    sizes = ("nq", "nv", "nu", "na", "nM", "nbody", "ngeom", "nsite", "njnt",
             "ntendon", "neq", "nplugin", "nmocap", "nsensordata", "nsensor",
             "nmesh", "ncam", "nlight", "nmat", "ntex", "npair", "nexclude", "nkey")
    dimensions = {name: int(getattr(model, name)) for name in sizes if hasattr(model, name)}
    arrays = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except (AttributeError, ValueError):
            continue
        if isinstance(value, np.ndarray):
            arrays[name] = value
    # Capture payload bytes once, rather than materializing them separately for
    # aggregate, topology and each field digest. Native owner remains private
    # during this call; the returned snapshot never grants mutable-owner reuse.
    snapshot = ReadOnlyModelSnapshot.capture(arrays)
    topology_names = ("body_parentid", "body_jntadr", "body_jntnum", "body_dofadr",
                      "body_dofnum", "jnt_type", "jnt_bodyid", "jnt_qposadr",
                      "jnt_dofadr", "dof_parentid", "dof_bodyid", "dof_jntid",
                      "dof_Madr", "actuator_trntype", "actuator_trnid", "geom_type")
    options = {}
    for name in dir(model.opt):
        if name.startswith("_"):
            continue
        value = getattr(model.opt, name)
        if isinstance(value, np.ndarray):
            options[name] = value.tolist()
        elif isinstance(value, (int, float, np.integer, np.floating)):
            options[name] = value.item() if isinstance(value, np.generic) else value
    stat = {}
    stat_descriptors = {}
    for name in dir(model.stat):
        if name.startswith("_"):
            continue
        value = getattr(model.stat, name)
        if isinstance(value, (np.ndarray, int, float, np.integer, np.floating)):
            array = np.asarray(value)
            stat[name] = array.tolist()
            stat_descriptors[name] = {"shape": list(array.shape), "dtype": array.dtype.str,
                                      "raw_sha256": sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    maps = {}
    kinds = {"body": (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
             "joint": (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
             "geom": (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
             "site": (mujoco.mjtObj.mjOBJ_SITE, model.nsite),
             "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
             "camera": (mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
             "sensor": (mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
             "light": (mujoco.mjtObj.mjOBJ_LIGHT, model.nlight),
             "mesh": (mujoco.mjtObj.mjOBJ_MESH, model.nmesh),
             "tendon": (mujoco.mjtObj.mjOBJ_TENDON, model.ntendon)}
    for kind, (obj, count) in kinds.items():
        maps[kind] = {name: index for index in range(count)
                      if (name := mujoco.mj_id2name(model, obj, index)) is not None}
    qpos_sizes = {int(mujoco.mjtJoint.mjJNT_FREE): 7,
                  int(mujoco.mjtJoint.mjJNT_BALL): 4,
                  int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
                  int(mujoco.mjtJoint.mjJNT_HINGE): 1}
    slices = {name: [int(model.jnt_qposadr[index]),
                     int(model.jnt_qposadr[index]) + qpos_sizes[int(model.jnt_type[index])]]
              for name, index in maps["joint"].items()}
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    model_blob = buffer.tobytes()
    model_digest = sha256(model_blob).hexdigest()
    snapshot = ReadOnlyModelSnapshot(snapshot.fields, model_blob)
    if reference_snapshot is not None:
        snapshot = snapshot.reuse_unchanged(reference_snapshot)
    decoder_pin = {"module": __name__,
                   "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                   "mujoco_python_version": mujoco.__version__,
                   "mujoco_native_version": mujoco.mj_versionString(),
                   "snapshot_schema": "exact-array-bytes-v1",
                   "snapshot_source_sha256": sha256(Path(__file__).with_name("cold_assets.py").read_bytes()).hexdigest()}
    if snapshot_pool is not None:
        snapshot = snapshot_pool.intern(snapshot, decoder_identity=decoder_pin)
    result = {"dimensions": dimensions, "options": options, "stat": stat,
            "stat_descriptors": stat_descriptors, "name_maps": maps,
            "joint_qpos_slices": slices, "model_sha256": model_digest,
            "static_arrays_sha256": snapshot.readset_digest(),
            "topology_sha256": snapshot.readset_digest(n for n in topology_names if n in arrays),
            "array_sha256": snapshot.array_digests,
            "decoder_pin": decoder_pin,
            "static_array_inventory": {n: {"shape": list(a.shape), "dtype": a.dtype.str}
                                       for n, a in sorted(arrays.items())},
            "physics_initialized": False, "controllers_initialized": False,
            "observables_initialized": False}
    return (result, snapshot) if return_snapshot else result


def robot_static_metadata(shell, model, name_maps):
    """Resolve robot/gripper references using compiled indices only."""
    result = []
    qwidth = {0: 7, 1: 4, 2: 1, 3: 1}
    vwidth = {0: 6, 1: 3, 2: 1, 3: 1}
    def joint_fields(names):
        ids = [name_maps["joint"][name] for name in names]
        qpos = [value for index in ids for value in range(int(model.jnt_qposadr[index]),
                 int(model.jnt_qposadr[index]) + qwidth[int(model.jnt_type[index])])]
        qvel = [value for index in ids for value in range(int(model.jnt_dofadr[index]),
                 int(model.jnt_dofadr[index]) + vwidth[int(model.jnt_type[index])])]
        return {"joint_names": list(names), "joint_ids": ids, "qpos_indices": qpos, "qvel_indices": qvel}
    for robot in shell.robots:
        item = dict(name=robot.name, naming_prefix=robot.robot_model.naming_prefix,
                    **joint_fields(robot.robot_model.joints))
        item["actuator_ids"] = [name_maps["actuator"][name] for name in robot.robot_model.actuators]
        item["root_body_name"] = robot.robot_model.root_body
        item["root_body_id"] = name_maps["body"][robot.robot_model.root_body]
        item["eef_body_name"] = robot.robot_model.eef_name
        item["eef_body_id"] = name_maps["body"][robot.robot_model.eef_name]
        item["init_qpos"] = np.asarray(robot.init_qpos).tolist()
        item["controller_config"] = dict(robot.controller_config)
        item["control_freq"] = float(robot.control_freq)
        item["controller_model_timestep"] = float(model.opt.timestep)
        item["eef_rot_offset"] = np.asarray(robot.eef_rot_offset).tolist()
        item["actuator_ctrlrange"] = model.actuator_ctrlrange[item["actuator_ids"]].tolist()
        item["initialization_noise"] = dict(robot.initialization_noise)
        if robot.has_gripper:
            item["gripper"] = joint_fields(robot.gripper.joints)
            item["gripper"]["actuator_ids"] = [name_maps["actuator"][name] for name in robot.gripper.actuators]
            item["gripper"] ["init_qpos"] = np.asarray(robot.gripper.init_qpos).tolist()
            item["gripper"] ["actuator_ctrlrange"] = model.actuator_ctrlrange[item["gripper"] ["actuator_ids"]].tolist()
            item["gripper"] ["dof"] = int(robot.gripper.dof)
            item["gripper"]["class"] = type(robot.gripper).__name__
            if hasattr(robot.gripper, "speed"):
                item["gripper"]["speed"] = float(robot.gripper.speed)
            item["gripper"]["important_sites"] = {
                label: {"name": name, "id": name_maps["site"].get(name)}
                for label, name in robot.gripper.important_sites.items()}
        item["eef_site_id"] = name_maps["site"][robot.gripper.important_sites["grip_site"]]
        item["eef_cylinder_id"] = name_maps["site"][robot.gripper.important_sites["grip_cylinder"]]
        item["prefix"] = item["naming_prefix"]
        item["joint_qpos_indices"] = item["qpos_indices"]
        item["joint_qvel_indices"] = item["qvel_indices"]
        item["gripper_qpos_indices"] = item.get("gripper", {}).get("qpos_indices", [])
        item["gripper_qvel_indices"] = item.get("gripper", {}).get("qvel_indices", [])
        result.append(item)
    return result

def task_segmentation_metadata(task_model, name_maps):
    """Resolve original Task semantic IDs without creating CPU simulation/data.

    The original mapping method only looks up compiled geom/site names. Keep its
    order and uniqueness assertions, including custom Task model extensions.
    """
    class NameResolver:
        def geom_name2id(self, name): return int(name_maps["geom"][name])
        def site_name2id(self, name): return int(name_maps["site"][name])
    task_model.generate_id_mappings(SimpleNamespace(model=NameResolver()))
    result = {}
    for kind, ordered, inverse in (
        ("instance", task_model.instances_to_ids, task_model.geom_ids_to_instances),
        ("class", task_model.classes_to_ids, task_model.geom_ids_to_classes),
    ):
        indices = {name: index for index, name in enumerate(ordered)}
        result[kind] = {int(geom): indices[label] for geom, label in inverse.items()}
    return result


def build_task_model(bddl_file_name, problem_name=None, *, xml_string_override=None,
                     compiled_model_file=None, static_model_setup=None, **domain_options):
    """Run original static task construction and native model compilation only.

    Caller must not use ``domain_shell`` for stepping or observation sampling.
    Its placement samplers and goal definitions are real original task objects,
    but robot references/controllers remain intentionally uninitialized.
    """
    from libero.libero.envs import TASK_MAPPING
    from libero.libero.envs.bddl_utils import get_problem_info
    import robosuite

    if xml_string_override is not None and (not isinstance(xml_string_override, str) or not xml_string_override.strip()):
        raise ValueError("xml_string_override must be a nonempty XML string")
    path = str(Path(bddl_file_name).resolve())
    info = get_problem_info(path)
    actual_problem = info["problem_name"]
    if problem_name is not None and problem_name.lower() != actual_problem.lower():
        raise ValueError("problem_name disagrees with BDDL")
    task_class = TASK_MAPPING[actual_problem]
    options = dict(domain_options)
    constructor_call = options.pop("_gpu_constructor_call", None)
    if constructor_call is None:
        options.setdefault("robots", ["Panda"])
        options.setdefault("initialization_noise", None)
        options.setdefault("controller_configs", robosuite.load_controller_config(default_controller="OSC_POSE"))
        options.setdefault("camera_names", ["agentview", "robot0_eye_in_hand"])
        options.setdefault("camera_heights", 128)
        options.setdefault("camera_widths", 128)
    elif (not isinstance(constructor_call, dict) or set(constructor_call) != {"args", "kwargs", "preallocated_state"}
          or not isinstance(constructor_call["args"], tuple) or not isinstance(constructor_call["kwargs"], dict)
          or not isinstance(constructor_call["preallocated_state"], dict) or options):
        raise TypeError("exact original GPU task constructor packet required")
    # renderer_config is a ControlEnv-level compatibility argument, absent in
    # BDDLBaseDomain's explicitly forwarded signature in robosuite 1.4.
    renderer_config = options.pop("renderer_config", None)
    from .construction_scope import model_only_construction
    with model_only_construction(task_class, allow_allocation=True):
        if constructor_call is None:
            shell = task_class.__new__(task_class, path, **options)
        else:
            # Public custom __new__ has already run once on the compatible GPU
            # subtype. Retain its ordinary instance state without rerunning it.
            shell = object.__new__(task_class)
            shell.__dict__.update(constructor_call["preallocated_state"])
    result = {}

    def initialize_model(self, xml_string=None):
        if xml_string_override is not None:
            xml = xml_string_override
        else:
            xml = xml_string if xml_string else self.model.get_xml()
        if self._xml_processor is not None:
            xml = self._xml_processor(xml)
        native = (mujoco.MjModel.from_xml_string(xml) if compiled_model_file is None else
                  mujoco.MjModel.from_binary_path(str(compiled_model_file)))
        result.update(xml=xml, native_model=native)
        self.sim = _NoSimulation()
        self.initialize_time(self.control_freq)

    def no_renderer(self):
        self.viewer = None

    def no_reset(self):
        return None

    def no_observables(self):
        return OrderedDict()

    def forbidden(self, *args, **kwargs):
        raise ModelOnlyExecutionError("Model-only shell is not an initialized GPU environment")

    # Preserve the resolved callable, including a legitimate instance override.
    # deepcopy of a bound method rebinds its owner to each cloned world shell.
    shell._gpu_original_setup_observables = shell._setup_observables
    for name, function in (("_initialize_sim", initialize_model),
                           ("initialize_renderer", no_renderer),
                           ("_reset_internal", no_reset),
                           ("_setup_observables", no_observables)):
        setattr(shell, name, MethodType(function, shell))
    # Store the same method object so clones preserve this explicit shim alias.
    shell._gpu_model_only_setup_observables_shim = shell._setup_observables
    with model_only_construction(task_class):
        if constructor_call is None:
            task_class.__init__(shell, path, **options)
        else:
            task_class.__init__(shell, *constructor_call["args"], **constructor_call["kwargs"])
    shell.physics_backend = "model_only"
    if constructor_call is None: shell.renderer_config = renderer_config
    for name in ("step", "reset", "reset_from_xml_string", "_get_observations", "_update_observables"):
        setattr(shell, name, MethodType(forbidden, shell))
    # Finalize fixed fixture poses before the one full native snapshot/hash.
    # The callback may write cold model parameters, but must not run physics.
    if static_model_setup is not None:
        static_model_setup(shell, result['native_model'])
    metadata = native_model_metadata(result["native_model"])
    shell.obj_body_id = {name: metadata["name_maps"] ["body"] [obj.root_body]
                         for name, obj in {**shell.objects_dict, **shell.fixtures_dict}.items()}
    metadata.update(segmentation_mappings=task_segmentation_metadata(shell.model, metadata["name_maps"]),
                    problem_name=actual_problem, bddl_file_name=path,
                    xml_overridden=xml_string_override is not None,
                    xml_sha256=sha256(result["xml"].encode("utf-8")).hexdigest(),
                    language_instruction=info.get("language_instruction"),
                    domain_name=info.get("domain_name"),
                    control_freq=float(shell.control_freq),
                    model_timestep=float(shell.model_timestep),
                    scheduler_model_timestep=float(shell.model_timestep),
                    control_timestep=float(shell.control_timestep),
                    native_model_timestep=float(result["native_model"].opt.timestep),
                    object_names=list(shell.objects_dict), fixture_names=list(shell.fixtures_dict),
                    site_object_names=list(shell.object_sites_dict),
                    robot_names=[robot.name for robot in shell.robots],
                    robot_metadata=robot_static_metadata(shell, result["native_model"], metadata["name_maps"]),
                    object_body_ids=dict(shell.obj_body_id),
                    ordered_object_body_ids=[[obj.name, shell.obj_body_id[obj.name]] for obj in shell.objects],
                    camera_names=list(shell.camera_names), camera_heights=list(shell.camera_heights),
                    camera_widths=list(shell.camera_widths), camera_depths=list(shell.camera_depths),
                    camera_segmentations=list(shell.camera_segmentations),
                    use_camera_obs=bool(shell.use_camera_obs), use_object_obs=bool(shell.use_object_obs),
                    render_collision_mesh=bool(shell.render_collision_mesh),
                    render_visual_mesh=bool(shell.render_visual_mesh),
                    image_convention=robosuite.macros.IMAGE_CONVENTION,
                    concatenate_images=bool(robosuite.macros.CONCATENATE_IMAGES))
    built = BuiltTaskModel(result["native_model"], result["xml"], shell, metadata)
    from ..host.runtime.core.native_model_update import publish_native_model_write_schema
    publish_native_model_write_schema(built)
    return built
