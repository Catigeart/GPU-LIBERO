"""Resident qpos-to-mass-solve chain for the two frozen MuJoCo 2.3.7 models.

G16, 2K1 and I01 keep ownership of their arithmetic. This adapter only binds
contiguous device views and routes first failure in the same CUDA stream.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

from types import SimpleNamespace
import numpy as np
import warp as wp

from libero.gpu.reference.position.body_kinematics import BodyKinematicsWorkspace, FrozenModel
from libero.gpu.reference.position.kinematics_frames.frames import FrameWorkspace
from libero.gpu.reference.position.mass_chain import MassChainWorkspace

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': True})


@wp.kernel
def _after_body(active: wp.array(dtype=wp.int32), body_status: wp.array(dtype=wp.int32),
                frame_active: wp.array(dtype=wp.int32)):
    w = wp.tid()
    frame_active[w] = wp.where(active[w] == 1 and body_status[w] == 0, 1, 0)


@wp.kernel
def _check_frames(active: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                  xipos: wp.array(dtype=scalar.wp_real), ximat: wp.array(dtype=scalar.wp_real),
                  geom_xpos: wp.array(dtype=scalar.wp_real), geom_xmat: wp.array(dtype=scalar.wp_real),
                  site_xpos: wp.array(dtype=scalar.wp_real), site_xmat: wp.array(dtype=scalar.wp_real),
                  body_tmpquat: wp.array(dtype=scalar.wp_real),
                  geom_tmpquat: wp.array(dtype=scalar.wp_real),
                  site_tmpquat: wp.array(dtype=scalar.wp_real),
                  nb: int, ng: int, ns: int):
    w = wp.tid()
    if active[w] != 1 or status[w] != 0:
        return
    for i in range(nb * 3):
        if not wp.isfinite(xipos[w * nb * 3 + i]):
            status[w] = 2
            return
    for i in range(nb * 9):
        if not wp.isfinite(ximat[w * nb * 9 + i]):
            status[w] = 2
            return
    for i in range(ng * 3):
        if not wp.isfinite(geom_xpos[w * ng * 3 + i]):
            status[w] = 2
            return
    for i in range(ng * 9):
        if not wp.isfinite(geom_xmat[w * ng * 9 + i]):
            status[w] = 2
            return
    for i in range(ns * 3):
        if not wp.isfinite(site_xpos[w * ns * 3 + i]):
            status[w] = 2
            return
    for i in range(ns * 9):
        if not wp.isfinite(site_xmat[w * ns * 9 + i]):
            status[w] = 2
            return
    for i in range(nb * 4):
        if not wp.isfinite(body_tmpquat[w * nb * 4 + i]):
            status[w] = 2
            return
    for i in range(ng * 4):
        if not wp.isfinite(geom_tmpquat[w * ng * 4 + i]):
            status[w] = 2
            return
    for i in range(ns * 4):
        if not wp.isfinite(site_tmpquat[w * ns * 4 + i]):
            status[w] = 2
            return


@wp.kernel
def _after_frames(frame_active: wp.array(dtype=wp.int32),
                  frame_status: wp.array(dtype=wp.int32),
                  mass_active: wp.array(dtype=wp.int32)):
    w = wp.tid()
    mass_active[w] = wp.where(frame_active[w] == 1 and frame_status[w] == 0, 1, 0)


@wp.kernel
def _finish(active: wp.array(dtype=wp.int32), body_status: wp.array(dtype=wp.int32),
            frame_status: wp.array(dtype=wp.int32), mass_active: wp.array(dtype=wp.int32),
            mass_stage: wp.array(dtype=wp.int32), mass_code: wp.array(dtype=wp.int32),
            mass_kind: wp.array(dtype=wp.int32), first_stage: wp.array(dtype=wp.int32),
            first_code: wp.array(dtype=wp.int32), result_kind: wp.array(dtype=wp.int32)):
    w = wp.tid()
    first_stage[w] = 0
    first_code[w] = 0
    result_kind[w] = 0
    if active[w] == 0:
        return
    if body_status[w] != 0:
        first_stage[w] = 1
        first_code[w] = body_status[w]
        result_kind[w] = 3
        return
    if frame_status[w] != 0:
        first_stage[w] = 2
        first_code[w] = frame_status[w]
        result_kind[w] = 3
        return
    if mass_active[w] != 1:
        first_stage[w] = 2
        first_code[w] = 1
        result_kind[w] = 3
        return
    if mass_stage[w] != 0:
        first_stage[w] = mass_stage[w] + 2
        first_code[w] = mass_code[w]
    result_kind[w] = mass_kind[w]


def _view(array, shape, owner, name):
    view = array.reshape(shape) if len(shape) > 1 else array.flatten()
    if (view.ptr != array.ptr or view.dtype != array.dtype or view.device != array.device
            or view.shape != shape or not view.is_contiguous):
        raise ValueError('invalid zero-copy view: ' + name)
    owner[name] = {'producer_ptr': int(array.ptr), 'consumer_ptr': int(view.ptr),
                   'shape': list(shape), 'strides': list(view.strides)}
    return view


def _array_binding(name, array):
    """Describe a kernel argument without reading or copying its device data."""
    if not isinstance(array, wp.array):
        raise ValueError('missing Warp array binding: ' + name)
    return (id(array), int(array.ptr), tuple(array.shape), tuple(array.strides),
            str(array.dtype), str(array.device), int(array.capacity),
            bool(array.is_contiguous))


class BodyMassChainWorkspace:
    """One model and fixed world capacity; `launch` consumes only resident buffers.

    The caller uploads active/qpos/mocap/RHS/count before launch. No frame or
    Cartesian data enter through the host. Rebinding is finished before Graph
    capture; the three owner workspaces retain their allocations and views.
    """

    def __init__(self, frozen: FrozenModel, frame_model: dict,
                 mass_model: SimpleNamespace, capacity: int, device='cuda:0'):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('positive capacity required')
        if not isinstance(frozen, FrozenModel):
            raise TypeError('verified G16 FrozenModel required')
        dims = frozen.dims
        if (int(mass_model.nbody), int(mass_model.nq), int(mass_model.nv),
                int(mass_model.nmocap), int(mass_model.njnt)) != (
                dims['nbody'], dims['nq'], 43 if dims['nbody'] == 38 else 11,
                dims['nmocap'], dims['njnt']):
            raise ValueError('cross-model topology')
        if np.asarray(frame_model['body_ipos']).shape[0] != dims['nbody']:
            raise ValueError('cross-model frame topology')
        self.capacity, self.device = capacity, device
        self.body = BodyKinematicsWorkspace(frozen, capacity, device)
        self.frames = FrameWorkspace(frame_model, capacity, device=device)
        self.mass = MassChainWorkspace(mass_model, capacity, device)
        self.views = {}
        b, j = dims['nbody'], dims['njnt']
        for name, width in (('xpos', 3), ('xquat', 4), ('xmat', 9)):
            setattr(self.frames, 'body_' + name,
                    _view(getattr(self.body, name), (capacity*b*width,), self.views,
                          'body_to_frames_' + name))
        self.frame_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.frames.active_world = self.frame_active
        for name, width in (('xipos', 3), ('ximat', 9)):
            setattr(self.mass, name,
                    _view(getattr(self.frames, name), (capacity, b*width), self.views,
                          'frames_to_mass_' + name))
        for name, width, count in (('xanchor', 3, j), ('xmat', 9, b), ('xaxis', 3, j)):
            src = getattr(self.body, name)
            if src.shape != (capacity, count*width):
                raise ValueError('body-to-mass shape: ' + name)
            setattr(self.mass, name, src)
            self.views['body_to_mass_' + name] = {
                'producer_ptr': int(src.ptr), 'consumer_ptr': int(getattr(self.mass, name).ptr),
                'shape': list(src.shape), 'strides': list(src.strides)}
        self.mass_active = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.mass.input_active = self.mass_active
        self.first_stage = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.first_code = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.result_kind = wp.zeros(capacity, dtype=wp.int32, device=device)
        self.mass.validate_layout()
        for edge in self.views.values():
            if edge['producer_ptr'] != edge['consumer_ptr']:
                raise ValueError('stale or copied device edge')
        self._freeze_bindings()

    def _binding_state(self):
        """Inspect the actual objects passed to every nested production kernel.

        G16's public upload attributes and its `Data` struct must both be
        included: changing both to a new allocation keeps their mutual alias
        valid, but a previously captured Graph still references the old one.
        The same applies to I01's RHS, outputs and static topology arrays.
        """
        owners = {
            'body': self.body,
            'body.model': self.body.model,
            'body.static': self.body.static,
            'body.data': self.body.data,
            'body.diag': self.body.diag,
            'frames': self.frames,
            'frames.static': self.frames.static,
            'mass': self.mass,
            'mass.compos': self.mass.compos,
            'mass.crb_model': self.mass.crb_model,
            'mass.mass': self.mass.mass,
        }
        arrays = {}
        references = list(owners.values())
        for prefix, owner in (('', self), ('body', self.body),
                              ('body.static', self.body.static),
                              ('body.data', self.body.data),
                              ('body.diag', self.body.diag),
                              ('frames', self.frames), ('mass', self.mass),
                              ('mass.compos', self.mass.compos),
                              ('mass.crb_model', self.mass.crb_model),
                              ('mass.mass', self.mass.mass)):
            for name, value in vars(owner).items():
                if isinstance(value, wp.array):
                    path = prefix + ('.' if prefix else '') + name
                    arrays[path] = _array_binding(path, value)
                    references.append(value)
        for name, value in self.frames.static.items():
            path = 'frames.static.' + name
            arrays[path] = _array_binding(path, value)
            references.append(value)
        scalars = {
            'capacity': self.capacity,
            'device': str(wp.get_device(self.device)),
            'body.capacity': self.body.capacity,
            'body.device': str(wp.get_device(self.body.device)),
            'body.model.dims': tuple(sorted(self.body.model.dims.items())),
            'body.static.dims': tuple(getattr(self.body.static, name)
                                      for name in ('nq', 'nb', 'nj', 'nm')),
            'frames.dims': tuple(getattr(self.frames, name)
                                 for name in ('nworld', 'nbody', 'ngeom', 'nsite')),
            'frames.device': str(wp.get_device(self.frames.device)),
            'frames.static_sha256': self.frames.tail_static_sha256,
            'mass.capacity': self.mass.capacity,
            'mass.device': str(wp.get_device(self.mass.device)),
            'mass.dims': tuple(self.mass.dims),
            'mass.model_signature': self.mass.model_signature,
            'mass.compos.dims': tuple(getattr(self.mass.compos, name)
                                      for name in ('nbody', 'njnt', 'nv')),
            'mass.crb_model.dims': tuple(getattr(self.mass.crb_model, name)
                                         for name in ('nbody', 'nv', 'nM')),
            'mass.mass.dims': (self.mass.mass.capacity, self.mass.mass.nv,
                               self.mass.mass.nM),
            'mass.mass.device': str(wp.get_device(self.mass.mass.device)),
        }
        return ({name: id(obj) for name, obj in owners.items()}, arrays,
                scalars, tuple(references))

    def _freeze_bindings(self):
        owners, arrays, scalars, references = self._binding_state()
        self._frozen_binding_owners = owners
        self._frozen_binding_arrays = arrays
        self._frozen_binding_scalars = scalars
        # Keep every captured allocation alive even if a caller rebinds an
        # attribute. This also prevents pointer reuse from hiding a rebind.
        self._binding_references = references

    def validate_bindings(self):
        """Reject any stale eager or captured Graph argument before execution."""
        owners, arrays, scalars, _ = self._binding_state()
        for label, expected, actual in (
            ('owner', self._frozen_binding_owners, owners),
            ('array', self._frozen_binding_arrays, arrays),
            ('scalar', self._frozen_binding_scalars, scalars),
        ):
            for path in sorted(set(expected) | set(actual)):
                if expected.get(path) != actual.get(path):
                    raise ValueError('stale Graph ' + label + ' binding: ' + path)
        for name in ('xpos','xquat','xmat'):
            if getattr(self.frames,'body_'+name).ptr != getattr(self.body,name).ptr:
                raise ValueError('stale body-to-frame Graph alias: '+name)
        for name in ('xipos','ximat'):
            if getattr(self.mass,name).ptr != getattr(self.frames,name).ptr:
                raise ValueError('stale frame-to-mass Graph alias: '+name)
        for name in ('xanchor','xmat','xaxis'):
            if getattr(self.mass,name).ptr != getattr(self.body,name).ptr:
                raise ValueError('stale body-to-mass Graph alias: '+name)
        if self.frames.active_world.ptr != self.frame_active.ptr or self.mass.input_active.ptr != self.mass_active.ptr:
            raise ValueError('stale active-mask Graph alias')
        return True

    def upload(self, active, qpos, mocap_pos, mocap_quat, rhs, count):
        """Preparation only. No host calculation or upload is called by launch."""
        self.validate_bindings()
        n, nv = self.capacity, self.mass.dims[2]
        if not isinstance(active,np.ndarray) or active.shape!=(n,) or active.dtype!=np.int32:
            raise ValueError('active schema')
        if not isinstance(rhs, np.ndarray) or rhs.shape != (n, 4, nv) or rhs.dtype != np.float64:
            raise ValueError('rhs schema')
        if not isinstance(count, np.ndarray) or count.shape != (n,) or count.dtype != np.int32:
            raise ValueError('count schema')
        if not np.all((count[np.asarray(active) == 1] >= 1) & (count[np.asarray(active) == 1] <= 4)):
            raise ValueError('count range')
        if not np.isfinite(rhs[active==1]).all():
            raise ValueError('active RHS nonfinite')
        self.body.upload(active, qpos, mocap_pos, mocap_quat)
        self.mass.mass.rhs.assign(np.ascontiguousarray(rhs))
        self.mass.mass.count.assign(np.ascontiguousarray(count))

    def launch(self, stream=None):
        self.validate_bindings()
        self.body.launch(stream=stream)
        wp.launch(_after_body, dim=self.capacity,
                  inputs=[self.body.active, self.body.status, self.frame_active],
                  device=self.device, stream=stream)
        self.frames.launch(stream=stream)
        wp.launch(_check_frames, dim=self.capacity,
                  inputs=[self.frame_active, self.frames.status, self.frames.xipos,
                          self.frames.ximat, self.frames.geom_xpos, self.frames.geom_xmat,
                          self.frames.site_xpos, self.frames.site_xmat,
                          self.frames.body_tmpquat, self.frames.geom_tmpquat,
                          self.frames.site_tmpquat, self.frames.nbody,
                          self.frames.ngeom, self.frames.nsite],
                  device=self.device, stream=stream)
        wp.launch(_after_frames, dim=self.capacity,
                  inputs=[self.frame_active, self.frames.status, self.mass_active],
                  device=self.device, stream=stream)
        self.mass.launch(stream=stream)
        wp.launch(_finish, dim=self.capacity,
                  inputs=[self.body.active, self.body.status, self.frames.status,
                          self.mass_active, self.mass.first_stage, self.mass.first_code,
                          self.mass.result_kind, self.first_stage, self.first_code,
                          self.result_kind], device=self.device, stream=stream)

    def snapshot(self):
        """Diagnostic download, called only after production has completed."""
        self.validate_bindings()
        out = self.body.download()
        out.update({'body_' + name: out.pop(name) for name in
                    ('xpos', 'xquat', 'xmat', 'xanchor', 'xaxis', 'status')})
        out.update({'frame_' + name: value for name, value in self.frames.download().items()})
        out.update({'frame_' + name: value for name, value in self.frames.download_stages().items()})
        out.update(self.mass.snapshot())
        out['solve_active'] = out['mass_active']
        out.update({name: np.array(getattr(self, name).numpy(), copy=True) for name in
                    ('frame_active', 'mass_active', 'first_stage', 'first_code', 'result_kind')})
        return out
