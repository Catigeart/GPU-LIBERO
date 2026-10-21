"""Host boundary facade for a device-resident physics runtime.

Reads never execute physics. Derived fields retain their original publication
phase; get_state/set_state are partial state operations, not resets.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from threading import RLock
from functools import wraps
import hashlib
from types import MappingProxyType
import numpy as np
from .model_names import bind_original_name_maps, install_original_name_methods
from robosuite.utils.binding_utils import MjData
from libero.gpu.host.build.compilation import load_only_entry


@dataclass(frozen=True)
class CompletedColdModelUpdate:
    model_generation: object
    new_model_identity: str
    generation: int
    parameters_sha256: str
    quarantine: object
    facade: object = None
    runtime: object = None
    world_state: object = None

class UnsupportedGpuOperation(RuntimeError):
    pass


def _copy_readonly(value):
    if isinstance(value, np.ndarray):
        value = np.array(value, copy=True)
        value.setflags(write=False)
    if isinstance(value, dict):
        return MappingProxyType({key: _copy_readonly(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_copy_readonly(item) for item in value)
    return value


def _freeze_shared(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('shared model arrays must have a byte-stable dtype')
        # Read-only flags on an owning ndarray can be reversed. Immutable
        # bytes prevent a caller from making a shared array writable again.
        return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_shared(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_shared(item) for item in value)
    return value


class SharedModelMetadata(Mapping):
    """One detached immutable model snapshot for a batch's host facades."""
    def __init__(self, metadata):
        self._values = _freeze_shared(metadata)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


@dataclass
class GpuSimState:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray

    @classmethod
    def from_flattened(cls, array, sim):
        # Original MjSimState slicing and na=0 contract.
        assert sim.model.na == 0
        return cls(array[0], array[1:1 + sim.model.nq],
                   array[1 + sim.model.nq:1 + sim.model.nq + sim.model.nv])

    def flatten(self):
        return np.concatenate(([self.time], self.qpos, self.qvel))


class _OptionView:
    def __init__(self, values, sim=None):
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_sim", sim)
    def __getattr__(self, name):
        try:
            if self._sim is not None: return self._sim._read_model_option(name)
            return self._values[name]
        except KeyError as exc: raise UnsupportedGpuOperation(f"unknown model option: {name}") from exc
    def __setattr__(self, name, value):
        if self._sim is None: raise UnsupportedGpuOperation("model options require a rebuilt specialization")
        self._sim._write_model_option(name, value)


class _ModelView:
    def __init__(self, metadata, sim=None):
        self._sim = sim
        # Only this explicitly frozen packet grants sharing. Arbitrary caller
        # ndarrays, including arrays with write=False, still get private copies.
        shared = isinstance(metadata, SharedModelMetadata)
        self._metadata = (dict(metadata) if shared else
                          {key: _copy_readonly(value) for key, value in metadata.items()})
        self._shared_metadata_names = frozenset(
            key for key, value in metadata.items() if shared and isinstance(value, np.ndarray))
        bind_original_name_maps(self)

    def __getattr__(self, name):
        if name in ("_model",):
            raise UnsupportedGpuOperation("native CPU physics handle is unavailable")
        if name == "opt" and "options" in self._metadata:
            return _OptionView(self._metadata["options"], self._sim)
        if name == "stat" and "stat" in self._metadata:
            return _OptionView(self._metadata["stat"])
        if name.endswith("_names"):
            kind = name[:-6]
            mapping = self._metadata.get(kind + "_name_to_id")
            dimension = {"body":"nbody", "site":"nsite", "geom":"ngeom", "joint":"njnt", "actuator":"nu", "camera":"ncam", "sensor":"nsensor", "light":"nlight", "mesh":"nmesh", "tendon":"ntendon"}.get(kind)
            if mapping is not None and dimension is not None:
                count = self._metadata.get(dimension, max(mapping.values(), default=-1) + 1)
                names = [None] * count
                for label, index in mapping.items(): names[index] = label
                # Match original MjModel wrapper: repeated unnamed objects
                # contribute only the last None label to its names tuple.
                indices = set(mapping.values())
                unnamed = set(range(count)) - indices
                if unnamed: indices.add(max(unnamed))
                return tuple(names[index] for index in sorted(indices))
        try:
            if self._sim is not None and name in self._sim._native_write_schema.get("arrays", {}):
                return self._sim._borrow_model_array(name)
            if name in ("site_rgba", "geom_rgba") and self._sim is not None:
                return self._sim._borrow_visual_array(name)
            return self._metadata[name]
        except KeyError as exc:
            raise UnsupportedGpuOperation(f"unmapped model metadata: {name}") from exc

    def __setattr__(self, name, value):
        if name.startswith("_"): return object.__setattr__(self, name, value)
        sim = self._sim
        if sim is None: raise UnsupportedGpuOperation("GPU model write owner is not bound")
        if name in ("site_rgba", "geom_rgba"):
            target = sim._borrow_visual_array(name)
        elif name in sim._native_write_schema.get("arrays", {}):
            target = sim._borrow_model_array(name)
        else:
            raise UnsupportedGpuOperation(f"native model field is not directly writable: {name}")
        values = np.array(target, copy=True)
        values[...] = value
        if not np.isfinite(values).all():
            raise ValueError(f"invalid model array assignment: {name}")
        target[...] = values

    def _id(self, kind, name):
        mapping = self._metadata[f"{kind}_name_to_id"]
        if name in mapping: return int(mapping[name])
        names = getattr(self, kind + "_names")
        if name is None:
            count = self._name_count(kind)
            unnamed = set(range(count)) - set(mapping.values())
            if unnamed: return max(unnamed)
        raise ValueError('No "%s" with name %s exists. Available "%s" names = %s.' % (kind, name, kind, names))

    def body_name2id(self, name): return self._id("body", name)
    def site_name2id(self, name): return self._id("site", name)
    def geom_name2id(self, name): return self._id("geom", name)
    def camera_name2id(self, name): return self._id("camera", name)
    def joint_name2id(self, name): return self._id("joint", name)
    def actuator_name2id(self, name): return self._id("actuator", name)
    def sensor_name2id(self, name): return self._id("sensor", name)
    def light_name2id(self, name): return self._id("light", name)
    def mesh_name2id(self, name): return self._id("mesh", name)
    def tendon_name2id(self, name): return self._id("tendon", name)

    def _name_count(self, kind):
        dimension = {"body":"nbody", "site":"nsite", "geom":"ngeom", "joint":"njnt", "actuator":"nu", "camera":"ncam", "sensor":"nsensor", "light":"nlight", "mesh":"nmesh", "tendon":"ntendon"}[kind]
        return self._metadata.get(dimension, max(self._metadata[kind + "_name_to_id"].values(), default=-1) + 1)

    def _name(self, kind, index):
        if index not in range(self._name_count(kind)): raise ValueError("No %s with id %d exists." % (kind, index))
        for name, identifier in self._metadata[f"{kind}_name_to_id"].items():
            if identifier == index:
                return name
        return None

    def body_id2name(self, index): return self._name("body", index)
    def site_id2name(self, index): return self._name("site", index)
    def joint_id2name(self, index): return self._name("joint", index)
    def actuator_id2name(self, index): return self._name("actuator", index)
    def sensor_id2name(self, index): return self._name("sensor", index)
    def light_id2name(self, index): return self._name("light", index)
    def mesh_id2name(self, index): return self._name("mesh", index)
    def tendon_id2name(self, index): return self._name("tendon", index)
    def geom_id2name(self, index): return self._name("geom", index)
    def camera_id2name(self, index): return self._name("camera", index)
    def get_xml(self): return self._metadata["xml"]

    def get_joint_qpos_addr(self, name):
        index = self.joint_name2id(name)
        if name in self._metadata["joint_qpos_slices"]:
            start, stop = self._metadata["joint_qpos_slices"][name]
        else:
            start = int(self.jnt_qposadr[index])
            stop = start + {0: 7, 1: 4, 2: 1, 3: 1}[int(self.jnt_type[index])]
        return start if stop - start == 1 else (start, stop)

    def get_joint_qvel_addr(self, name):
        index = self.joint_name2id(name)
        if name in self._metadata["joint_qvel_slices"]:
            start, stop = self._metadata["joint_qvel_slices"][name]
        else:
            start = int(self.jnt_dofadr[index])
            stop = start + {0: 6, 1: 3, 2: 1, 3: 1}[int(self.jnt_type[index])]
        return start if stop - start == 1 else (start, stop)


install_original_name_methods(_ModelView)


class _DataView:
    def __init__(self, sim): object.__setattr__(self, "_sim", sim)

    def __setattr__(self, name, value):
        if name.startswith("_"): return object.__setattr__(self, name, value)
        if name == "time":
            value = float(value)
            if not np.isfinite(value): raise ValueError("finite simulation time required")
            return self._sim._write_state(("time",), "write_fields", {"time": value})
        if name not in self._sim.PUBLIC_WRITABLE_FIELDS:
            raise UnsupportedGpuOperation(f"data field is not a writable GPU state root: {name}")
        target = self._sim._borrow_public_field(name)
        # MuJoCo's original field setter delegates this same NumPy broadcast.
        # Validate a detached payload before changing aliases or device roots.
        array = np.array(target, copy=True)
        array[...] = value
        if not np.isfinite(array).all():
            raise ValueError(f"invalid public data assignment: {name}")
        target[...] = array
        self._sim.flush_public_arrays()


    def __getattr__(self, name):
        if name == "model": return self._sim.model
        if name == "_data":
            raise UnsupportedGpuOperation("native CPU physics handle is unavailable")
        if name in self._sim.PUBLIC_WRITABLE_FIELDS:
            return self._sim._borrow_public_field(name)
        return self._sim.snapshot_fields((name,))[name]

    def get_site_xpos(self, name):
        return self.site_xpos[self._sim.model.site_name2id(name)]

    def get_site_xmat(self, name):
        return self.site_xmat[self._sim.model.site_name2id(name)].reshape(3, 3)

    def get_body_xpos(self, name):
        return self.body_xpos[self._sim.model.body_name2id(name)]

    def get_body_xmat(self, name):
        return self.body_xmat[self._sim.model.body_name2id(name)].reshape(3, 3)

    def get_geom_xpos(self, name):
        return self.geom_xpos[self._sim.model.geom_name2id(name)]

    def get_geom_xmat(self, name):
        return self.geom_xmat[self._sim.model.geom_name2id(name)].reshape(3, 3)

    def get_site_xvelp(self, name):
        return self._spatial("site", name, "translation", velocity=True)

    def get_site_xvelr(self, name):
        return self._spatial("site", name, "rotation", velocity=True)

    def get_site_jacp(self, name):
        return self._spatial("site", name, "translation")

    def get_site_jacr(self, name):
        return self._spatial("site", name, "rotation")

    def get_body_jacp(self, name):
        return self._spatial("body", name, "translation")

    def get_body_jacr(self, name):
        return self._spatial("body", name, "rotation")

    def _spatial(self, kind, name, component, *, velocity=False):
        index = getattr(self._sim.model, kind + "_name2id")(name)
        operation = "read_velocity" if velocity else "read_jacobian"
        # These are new host result values in original robosuite; they remain
        # writable without implying any direct write to physics derived fields.
        return np.array(self._sim._call(operation, kind, index, component), copy=True)

    def get_body_xvelp(self, name):
        return self._spatial("body", name, "translation", velocity=True)

    def get_body_xvelr(self, name):
        return self._spatial("body", name, "rotation", velocity=True)

    def get_geom_xvelp(self, name):
        return self._spatial("geom", name, "translation", velocity=True)

    def get_geom_xvelr(self, name):
        return self._spatial("geom", name, "rotation", velocity=True)

    def get_geom_jacp(self, name):
        return self._spatial("geom", name, "translation")

    def get_geom_jacr(self, name):
        self._sim.model.geom_name2id(name)
        # Original pinned robosuite allocates jacv, then returns undefined jacr.
        # Retain its public exception; internal velocity uses the valid GPU
        # rotation matrix through read_velocity, without this public typo.
        raise NameError("name 'jacr' is not defined")

    def get_geom_jacobian(self, name, component="translation"):
        """Explicit GPU extension with a valid translation/rotation matrix."""
        return self._spatial("geom", name, component)

    def get_camera_xpos(self, name):
        return self.cam_xpos[self._sim.model.camera_name2id(name)]

    def get_camera_xmat(self, name):
        return self.cam_xmat[self._sim.model.camera_name2id(name)].reshape(3, 3)

    def get_light_xpos(self, name):
        return self.light_xpos[self._sim.model.light_name2id(name)]

    def get_light_xdir(self, name):
        return self.light_xdir[self._sim.model.light_name2id(name)]

    def get_sensor(self, name):
        # Original robosuite indexes by sensor ID and returns one scalar,
        # even when sensor_adr/dim describe a different native sensor vector.
        return self.sensordata[self._sim.model.sensor_name2id(name)]

    def get_body_xquat(self, name):
        return self.body_xquat[self._sim.model.body_name2id(name)]

    def get_joint_qpos(self, name):
        address = self._sim.model.get_joint_qpos_addr(name)
        start, stop = address if isinstance(address, tuple) else (address, address + 1)
        result = self.qpos[start:stop]
        return float(result[0]) if stop - start == 1 else result

    def get_joint_qvel(self, name):
        address = self._sim.model.get_joint_qvel_addr(name)
        start, stop = address if isinstance(address, tuple) else (address, address + 1)
        result = self.qvel[start:stop]
        return float(result[0]) if stop - start == 1 else result

    def set_joint_qvel(self, name, value):
        from types import SimpleNamespace
        # Resolve names with the original error contract before reading data.
        self.model.get_joint_qvel_addr(name)
        with self._sim._lock:
            values = np.array(self.qvel, copy=True)
            MjData.set_joint_qvel(SimpleNamespace(model=self.model, qvel=values), name, value)
            if not np.isfinite(values).all():
                raise ValueError("joint qvel contains nonfinite value")
            self._sim._write_state(("qvel",), "write_fields", {"qvel": values})

    def set_joint_qpos(self, name, value):
        from types import SimpleNamespace
        # Resolve names with the original error contract before reading data.
        self.model.get_joint_qpos_addr(name)
        with self._sim._lock:
            values = np.array(self.qpos, copy=True)
            MjData.set_joint_qpos(SimpleNamespace(model=self.model, qpos=values), name, value)
            if not np.isfinite(values).all():
                raise ValueError("joint qpos contains nonfinite value")
            self._sim._write_state(("qpos",), "write_fields", {"qpos": values})


_DataView.get_joint_qpos = MjData.get_joint_qpos
_DataView.get_joint_qvel = MjData.get_joint_qvel


def _serialized(method):
    @load_only_entry
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class GpuSimFacade:
    physics_backend = "gpu"
    PUBLIC_WRITABLE_FIELDS = frozenset(("qpos", "qvel", "ctrl", "qfrc_applied",
                                        "xfrc_applied", "qacc_warmstart"))

    def __init__(self, runtime, model_metadata, *, world_state=None, model_generation=None):
        if getattr(runtime, "physics_backend", None) != "gpu":
            raise UnsupportedGpuOperation("runtime must explicitly be a GPU physics runtime")
        for operation in ("read_fields", "write_fields", "forward", "step", "close"):
            if not callable(getattr(runtime, operation, None)):
                raise UnsupportedGpuOperation(f"runtime lacks {operation}")
        self._runtime = runtime
        self._closed = False
        self._quarantined = False
        self.visual_generation = 0
        self._renderer = None
        self._render_context_provider = None
        self._has_public_borrows = False
        self._public_arrays = {}
        self._public_shadows = {}
        self._visual_arrays = {}
        self._visual_shadows = {}
        self._model_arrays = {}
        self._model_shadows = {}
        self._option_arrays = {}
        self._option_shadows = {}
        self._option_changes = {}
        self._native_model_updater = None
        self._native_write_schema = dict(model_metadata.get("native_write_schema", {}))
        self._flushing_public = False
        self._lock = RLock()
        self._world_state = world_state
        self._model_generation = model_generation
        self.model = _ModelView(model_metadata, self)
        self.data = _DataView(self)

    @_serialized
    def _borrow_public_field(self, name):
        self.flush_public_arrays()
        result = self._call("read_fields", (name,))
        if name not in result:
            raise UnsupportedGpuOperation(f"GPU snapshot missing fields: {[name]}")
        value = result[name]
        if not isinstance(value, np.ndarray):
            raise UnsupportedGpuOperation(f"public array field required: {name}")
        if name not in self._public_arrays:
            self._public_arrays[name] = np.array(value, copy=True)
            self._public_shadows[name] = np.array(value, copy=True)
            self._has_public_borrows = True
        else:
            target = self._public_arrays[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise RuntimeError(f"resident public array shape/dtype changed: {name}")
            target[...] = value
            self._public_shadows[name][...] = value
        return self._public_arrays[name]

    @_serialized
    def bind_native_model_updates(self, callback, schema=None):
        if not callable(callback): raise TypeError("actual cold native update callback required")
        if self._closed or self._quarantined: raise RuntimeError("GPU sim is not usable")
        if schema is not None: self._native_write_schema = dict(schema)
        self._native_model_updater = callback

    def _model_open(self):
        if self._closed: raise RuntimeError("GPU sim is closed")
        if self._quarantined: raise RuntimeError("GPU sim is quarantined; rebuild required")

    @_serialized
    def _borrow_model_array(self, name):
        self._model_open()
        if name not in self._native_write_schema.get("arrays", {}):
            raise UnsupportedGpuOperation(f"native model array requires schema/rebuild: {name}")
        if name not in self._model_arrays:
            value = self.model._metadata[name]
            self._model_arrays[name] = np.array(value, copy=True)
            self._has_public_borrows = True
            self._model_shadows[name] = np.array(value, copy=True)
        return self._model_arrays[name]

    @_serialized
    def _read_model_option(self, name):
        self._model_open()
        value = self.model._metadata["options"][name]
        if name in self._option_changes: return self._option_changes[name]
        if isinstance(value, (tuple, list, np.ndarray)):
            if name not in self._native_write_schema.get("options", {}): return _copy_readonly(np.asarray(value))
            if name not in self._option_arrays:
                self._option_arrays[name] = np.asarray(value).copy()
                self._has_public_borrows = True
                self._option_shadows[name] = np.asarray(value).copy()
            return self._option_arrays[name]
        return value

    @_serialized
    def _write_model_option(self, name, value):
        self._model_open()
        schema = self._native_write_schema.get("options", {}).get(name)
        if schema is None: raise UnsupportedGpuOperation(f"model option requires schema/rebuild: {name}")
        original = self.model._metadata["options"][name]
        target = np.asarray(original)
        array = np.asarray(value, dtype=target.dtype)
        if array.shape != target.shape or not np.isfinite(array).all():
            raise ValueError(f"invalid native model option: {name}")
        if array.shape:
            self._read_model_option(name)[...] = array
        elif array.tobytes() == target.tobytes():
            self._option_changes.pop(name, None)
        else:
            self._option_changes[name] = array.item()
            self._has_public_borrows = True

    @_serialized
    def pending_native_model_changes(self):
        arrays = {name: np.array(value, copy=True) for name, value in self._model_arrays.items()
                  if value.tobytes() != self._model_shadows[name].tobytes()}
        options = dict(self._option_changes)
        options.update({name: np.array(value, copy=True) for name, value in self._option_arrays.items()
                        if value.tobytes() != self._option_shadows[name].tobytes()})
        return arrays, options

    @_serialized
    def adopt_runtime_from(self, fresh):
        """Transfer a fully restored fresh owner into the stable public facade.

        Old arena must already be closed; fresh is an ownership transfer shell.
        All prechecks/readback finish before mutating this public object. Public
        model/data objects and same-shape borrowed NumPy views retain identity.
        """
        if fresh is self or not isinstance(fresh, GpuSimFacade): raise TypeError("distinct fresh GPU facade required")
        if not self._closed: raise RuntimeError("old GPU owner must close before adopting replacement")
        with fresh._lock:
            if fresh._closed or fresh._quarantined or self._quarantined:
                raise RuntimeError("failed/closed facade cannot participate in adoption")
            if getattr(fresh._runtime, "physics_backend", None) != "gpu":
                raise UnsupportedGpuOperation("fresh actual GPU owner required")
            if fresh._runtime is self._runtime:
                raise ValueError("replacement must have a separately allocated resident owner")
            if fresh._world_state is not None and fresh._world_state is self._world_state:
                raise ValueError("replacement must have a separately prepared world state")
            # The parameter descriptor is intentionally advanced in place.
            # The restored world and actual runtime owner must still be new.
            if fresh._world_state is not None:
                if fresh._world_state.closed or fresh._world_state.model is not fresh._model_generation:
                    raise ValueError("fresh world must bind the restored model generation")
            for name in ("nq", "nv", "na", "nu", "nbody", "ngeom", "nsite", "njnt", "nM", "ntendon", "neq", "nmocap", "nsensordata", "nsensor", "nmesh", "ncam", "nlight", "nmat", "ntex", "npair", "nexclude", "nkey"):
                if name in self.model._metadata and self.model._metadata[name] != fresh.model._metadata.get(name):
                    raise ValueError("public model topology changed during parameter update: " + name)
            if any(getattr(fresh, name, {}) for name in ("_public_arrays", "_visual_arrays", "_model_arrays", "_option_arrays")):
                raise ValueError("fresh transfer facade must not expose borrowed NumPy views")
            snapshots = fresh._call("read_fields", tuple(self._public_arrays)) if self._public_arrays else {}
            for name, value in snapshots.items():
                original = self._public_arrays[name]
                if np.asarray(value).shape != original.shape or np.asarray(value).dtype != original.dtype:
                    raise ValueError("fresh resident public field shape/dtype changed: " + name)
            for buffers, metadata_name in ((getattr(self, "_visual_arrays", {}), None), (getattr(self, "_model_arrays", {}), None), (getattr(self, "_option_arrays", {}), "options")):
                for name, array in buffers.items():
                    value = fresh.model._metadata[name] if metadata_name is None else fresh.model._metadata[metadata_name][name]
                    if np.asarray(value).shape != array.shape or np.asarray(value).dtype != array.dtype:
                        raise ValueError("fresh native model array shape/dtype changed: " + name)
            metadata = {key: _copy_readonly(value) for key, value in fresh.model._metadata.items()}
            # Preflight all existing metadata views before pointer mutation.
            retained_metadata = {}
            for name, value in self.model._metadata.items():
                if isinstance(value, np.ndarray):
                    incoming = metadata.get(name)
                    if not isinstance(incoming, np.ndarray) or incoming.shape != value.shape or incoming.dtype != value.dtype:
                        raise ValueError("fresh model metadata shape/dtype changed: " + name)
                    if name not in self.model._shared_metadata_names:
                        retained_metadata[name] = value
            from types import SimpleNamespace
            names = SimpleNamespace(_metadata=metadata)
            bind_original_name_maps(names)
            self._runtime = fresh._runtime
            self._world_state = fresh._world_state
            self._model_generation = fresh._model_generation
            self._renderer = fresh._renderer
            self._render_context_provider = fresh._render_context_provider
            self.visual_generation = fresh.visual_generation
            if hasattr(fresh, "_native_model_updater"):
                self._native_model_updater = fresh._native_model_updater or getattr(self, "_native_model_updater", None)
                self._native_write_schema = dict(fresh._native_write_schema)
            for name, array in retained_metadata.items():
                writable = array.flags.writeable
                array.setflags(write=True)
                array[...] = metadata[name]
                array.setflags(write=writable)
                metadata[name] = array
            self.model._metadata = metadata
            self.model._shared_metadata_names = frozenset()
            for name, value in vars(names).items():
                if name != "_metadata": object.__setattr__(self.model, name, value)
            self._closed = False
            for name, value in snapshots.items():
                self._public_arrays[name].setflags(write=True)
                self._public_arrays[name][...] = value
                self._public_shadows[name] = np.array(value, copy=True)
            for buffers, shadows, metadata_name in ((getattr(self, "_visual_arrays", {}), getattr(self, "_visual_shadows", {}), None),
                    (getattr(self, "_model_arrays", {}), getattr(self, "_model_shadows", {}), None),
                    (getattr(self, "_option_arrays", {}), getattr(self, "_option_shadows", {}), "options")):
                for name, array in buffers.items():
                    value = metadata[name] if metadata_name is None else metadata[metadata_name][name]
                    array.setflags(write=True); array[...] = value; shadows[name] = np.asarray(value).copy()
            if hasattr(self, "_option_changes"): self._option_changes.clear()
            fresh._closed = True
            fresh._runtime = None
            fresh._renderer = None
            fresh._render_context_provider = None
            fresh._world_state = None
            fresh._model_generation = None
            return self

    @_serialized
    def _borrow_visual_array(self, name):
        if self._closed: raise RuntimeError("GPU sim is closed")
        if self._quarantined: raise RuntimeError("GPU sim is quarantined; rebuild required")
        if name not in self._visual_arrays:
            value = self.model._metadata[name]
            self._visual_arrays[name] = np.array(value, copy=True)
            self._has_public_borrows = True
            self._visual_shadows[name] = np.array(value, copy=True)
        return self._visual_arrays[name]

    @_serialized
    def flush_public_arrays(self):
        """Commit borrowed NumPy slices at the next public GPU boundary.

        Arrays are actual NumPy buffers, so np.asarray views and slice views share
        writes as in original MjData. No array is borrowed unless requested.
        """
        if not self._has_public_borrows or self._flushing_public: return
        payload = {name: np.array(value, copy=True)
                   for name, value in self._public_arrays.items()
                   if value.tobytes() != self._public_shadows[name].tobytes()}
        visuals = {name: np.array(value, copy=True)
                   for name, value in self._visual_arrays.items()
                   if value.tobytes() != self._visual_shadows[name].tobytes()}
        physical, options = self.pending_native_model_changes()
        if not payload and not visuals and not physical and not options: return
        if any(not np.isfinite(value).all() for value in (*payload.values(), *visuals.values(), *physical.values(), *options.values())):
            raise ValueError("public data array contains nonfinite values")
        self._flushing_public = True
        try:
            if payload:
                self._write_state(tuple(payload), "write_fields", payload)
                for name, value in payload.items(): self._public_shadows[name][...] = value
            if visuals:
                self.update_visual_parameters(visuals)
            if physical or options:
                if not callable(self._native_model_updater):
                    raise UnsupportedGpuOperation("cold native model replacement is not bound")
                self._native_model_updater(physical, options)
                if self._closed or self._quarantined or self.pending_native_model_changes() != ({}, {}):
                    raise RuntimeError("native model update did not adopt the actual restored replacement")
        finally:
            self._flushing_public = False

    @_serialized
    def synchronize_public_arrays(self):
        if not self._has_public_borrows: return
        self.flush_public_arrays()
        if self._public_arrays:
            result = self._call("read_fields", tuple(self._public_arrays))
            for name, value in result.items():
                self._public_arrays[name][...] = value
                self._public_shadows[name][...] = value

    @_serialized
    def _call(self, operation, *args, **kwargs):
        if self._closed:
            raise RuntimeError("GPU sim is closed")
        if self._quarantined:
            raise RuntimeError("GPU sim is quarantined after uncertain device mutation; rebuild required")
        if self._has_public_borrows and operation not in ("read_fields", "write_fields", "write_joint_qpos", "close"):
            self.flush_public_arrays()
        method = getattr(self._runtime, operation, None)
        if not callable(method):
            raise UnsupportedGpuOperation(f"GPU runtime does not implement {operation}")
        return method(*args, **kwargs)

    @_serialized
    def snapshot_fields(self, names):
        names = tuple(names)
        self.flush_public_arrays()
        result = self._call("read_fields", names)
        missing = set(names) - set(result)
        if missing:
            raise UnsupportedGpuOperation(f"GPU snapshot missing fields: {sorted(missing)}")
        return MappingProxyType({name: _copy_readonly(result[name]) for name in names})

    def _quarantine(self, stage):
        self._quarantined = True
        if self._world_state is not None and not self._world_state.closed:
            self._world_state.reset_metadata(clear_errors=False)
            self._world_state.error.record(91001, stage, "uncertain device mutation; rebuild required")

    def _write_state(self, fields, operation, *args):
        with self._lock:
            self.flush_public_arrays()
            token = self._world_state.prepare_state_write(fields) if self._world_state is not None else None
            try:
                result = self._call(operation, *args)
                if token is not None:
                    self._world_state.commit_state_write(token)
            except BaseException:
                self._quarantine("state_upload")
                raise
            if operation == "write_fields":
                for name, value in args[0].items():
                    if name in self._public_arrays:
                        self._public_arrays[name][...] = value
                        self._public_shadows[name][...] = value
            elif operation == "write_joint_qpos" and "qpos" in self._public_arrays:
                start, stop, value = args
                self._public_arrays["qpos"][start:stop] = value
                self._public_shadows["qpos"][start:stop] = value
            return result
    def get_state(self):
        values = self.snapshot_fields(("time", "qpos", "qvel"))
        # Preserve the original public state type, including mutable fields and
        # its classmethod from_flattened. This only packages GPU host readback;
        # constructing the state value never constructs or runs CPU physics.
        from robosuite.utils.binding_utils import MjSimState
        return MjSimState(float(values["time"]), np.array(values["qpos"], copy=True), np.array(values["qvel"], copy=True))

    def set_state(self, value):
        self.set_state_from_flattened(np.concatenate(([value.time], value.qpos, value.qvel)))

    def validate_flat_state(self, value):
        if self.model.na != 0:
            raise UnsupportedGpuOperation("legacy flat-state interface requires na=0")
        value = np.asarray(value, dtype=np.float64)
        expected = 1 + self.model.nq + self.model.nv
        if value.shape != (expected,) or not np.isfinite(value).all():
            raise ValueError(f"flat state must have {expected} finite values")
        return value.copy()

    @_serialized
    def set_state_from_flattened(self, value):
        value = self.validate_flat_state(value)
        nq = self.model.nq
        self._write_state(("time", "qpos", "qvel"), "write_fields", {"time": float(value[0]), "qpos": value[1:1+nq].copy(), "qvel": value[1+nq:].copy()})

    @_serialized
    def update_model_parameters(self, changes):
        """Runtime validates and uploads atomically before metadata publication."""
        if not changes:
            return None
        if callable(self._native_model_updater) and all(
                name in self._native_write_schema.get("arrays", {}) for name in changes):
            self._model_open()
            # Validate the entire explicit request before modifying borrowed
            # buffers. Explicit fields override earlier writes to those same
            # fields; other pending arrays/options/state/visuals commit once.
            payload = {}
            for name, value in changes.items():
                original = self.model._metadata.get(name)
                if not isinstance(original, np.ndarray):
                    raise ValueError(f"parameter has no declared array shape: {name}")
                array = np.asarray(value, dtype=original.dtype)
                if array.shape != original.shape or not np.isfinite(array).all():
                    raise ValueError(f"invalid parameter shape/content: {name}")
                payload[name] = np.ascontiguousarray(array).copy()
            saved = {}
            for name, value in payload.items():
                target = self._borrow_model_array(name)
                saved[name] = np.array(target, copy=True)
                target[...] = value
            try:
                self.flush_public_arrays()
            except BaseException:
                # A preparation refusal with the old owner still live restores
                # only these explicit staged writes. Existing user NumPy dirt
                # remains pending. After actual release/failure, quarantine and
                # retired read-only buffers remain untouched.
                if not self._closed and not self._quarantined:
                    for name, value in saved.items(): self._model_arrays[name][...] = value
                raise
            return None
        if self._model_generation is None:
            raise UnsupportedGpuOperation("model mutation requires a ModelGeneration contract")
        payload = {}
        digests = {}
        for name, value in changes.items():
            if name not in self._model_generation.spec.mutable_parameters:
                raise ValueError(f"model rebuild required for parameter: {name}")
            original = self.model._metadata.get(name)
            if not isinstance(original, np.ndarray):
                raise ValueError(f"parameter has no declared array shape: {name}")
            array = np.asarray(value, dtype=original.dtype)
            if array.shape != original.shape or not np.isfinite(array).all():
                raise ValueError(f"invalid parameter shape/content: {name}")
            array = np.ascontiguousarray(array).copy()
            payload[name] = array
            content = str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()
            digests[name] = hashlib.sha256(content).hexdigest()
        # An identical bit payload has no device write or generation transition.
        # Validate the entire request first, including unsupported keys/finite.
        if all(array.tobytes()==np.ascontiguousarray(self.model._metadata[name]).tobytes()
               for name,array in payload.items()):
            if self._closed:raise RuntimeError('GPU simulation facade is closed')
            if self._quarantined:raise RuntimeError('GPU facade mutation uncertain; close and rebuild required')
            return None
        # Check digest, whitelist and generation overflow before device mutation.
        with self._lock:
            token = self._model_generation.prepare_parameter_update(digests)
            try:
                transaction = getattr(self._runtime, 'update_model_parameters_transaction', None)
                result = (self._call('update_model_parameters_transaction', payload, token, self._model_generation)
                          if callable(transaction) else self._call('update_model_parameters', payload))
                if isinstance(result, CompletedColdModelUpdate):
                    legacy = (result.facade is None and result.runtime is None and result.world_state is None and self._closed)
                    adopted = (result.facade is self and result.runtime is self._runtime
                        and result.world_state is self._world_state and not self._closed
                        and self._world_state is not None and not self._world_state.closed
                        and self._world_state.model is self._model_generation)
                    if (result.model_generation is not self._model_generation or
                            result.generation != token.next_generation or
                            result.parameters_sha256 != token.next_digest or
                            self._model_generation.generation != token.next_generation or
                            self._model_generation.parameters_sha256 != token.next_digest or
                            self._model_generation.spec.identity_key != result.new_model_identity or
                            not callable(result.quarantine) or not (legacy or adopted)):
                        if callable(result.quarantine):result.quarantine('invalid cold replacement receipt')
                        raise RuntimeError('cold parameter replacement receipt differs from prepared token')
                    return
                self._model_generation.commit_parameter_update(token)
            except BaseException:
                self._quarantine("model_upload")
                raise
        for name, value in payload.items():
            self.model._metadata[name] = _copy_readonly(value)
            if name in self._model_arrays:
                self._model_arrays[name][...] = value
                self._model_shadows[name][...] = value
        return result

    @_serialized
    def update_visual_parameters(self, changes):
        """Publish rendering-only RGBA changes without invalidating physics."""
        if not changes: return None
        if self.visual_generation >= 2147483647:
            raise ValueError("visual generation overflow")
        payload = {}
        for name, value in changes.items():
            if name not in ("site_rgba", "geom_rgba"):
                raise ValueError(f"unsupported visual parameter: {name}")
            original = self.model._metadata.get(name)
            if not isinstance(original, np.ndarray):
                raise ValueError(f"visual parameter has no declared shape: {name}")
            array = np.asarray(value, dtype=original.dtype)
            if array.shape != original.shape or not np.isfinite(array).all():
                raise ValueError(f"invalid visual parameter: {name}")
            payload[name] = np.ascontiguousarray(array).copy()
        try:
            result = self._call("update_visual_parameters", payload)
        except BaseException:
            self._quarantine("visual_upload")
            raise
        for name, value in payload.items():
            self.model._metadata[name] = _copy_readonly(value)
            if name in self._visual_arrays:
                self._visual_arrays[name][...] = value
                self._visual_shadows[name][...] = value
        self.visual_generation += 1
        return result

    def _public_stage(self, name):
        if not self._has_public_borrows:
            return self._call(name)
        with self._lock:
            result = self._call(name)
            self.synchronize_public_arrays()
            return result

    def forward(self): return self._public_stage("forward")
    def step(self, with_udd=True): return self._public_stage("step")
    def step1(self): return self._public_stage("step1")
    def reset(self): return self._public_stage("reset")
    @_serialized
    def bind_renderer(self, callback, context_provider=None):
        if not callable(callback): raise TypeError("GPU render boundary callback required")
        if self._closed: raise RuntimeError("GPU sim is closed")
        self._renderer = callback
        if context_provider is not None and not callable(context_provider):
            raise TypeError("actual render context provider must be callable")
        self._render_context_provider = context_provider

    @property
    def _render_context_offscreen(self):
        with self._lock:
            self._model_open()
            if self._render_context_provider is None: return None
            return self._render_context_provider()

    @_serialized
    def render(self, width=None, height=None, *, camera_name=None, depth=False,
               mode="offscreen", device_id=-1, segmentation=False):
        # Original MjSim.render positional/keyword contract. Rendering reads the
        # resident publication; it never calls forward to manufacture freshness.
        assert mode == "offscreen", "only offscreen supported for now"
        if self._closed: raise RuntimeError("GPU sim is closed")
        if self._quarantined: raise RuntimeError("GPU sim is quarantined; rebuild required")
        self.flush_public_arrays()
        callback = self._renderer
        if callback is None:
            return self._call("render", width=width, height=height, camera_name=camera_name,
                              depth=depth, segmentation=segmentation, mode=mode, device_id=device_id)
        return callback(width=width, height=height, camera_name=camera_name,
                        depth=depth, segmentation=segmentation)


    @_serialized
    def close(self):
        if not self._closed:
            # A failed release remains retryable and is reported to the caller.
            self._runtime.close()
            if self._world_state is not None: self._world_state.close()
            self._closed = True
            self._renderer = None
            self._render_context_provider = None
            for value in (*self._public_arrays.values(), *self._visual_arrays.values(), *self._model_arrays.values(), *self._option_arrays.values()):
                value.setflags(write=False)

    free = close
