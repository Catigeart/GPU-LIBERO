"""Resident fixed-model S adapter with separated position/reference entries.

Preparation binds real producer arrays. Production never reads host state,
allocates arrays, calls upload_inputs, or fabricates producer clocks. Admission
(position/C/F currentness) belongs to the upstream device gate supplying active.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
import numpy as np
import warp as wp
from .identity import NQ, NV, NBODY, MJB_SHA256
from .workspace import PositionReferenceWorkspace

@wp.kernel(enable_backward=False)
def _publish_exclude(active: wp.array(dtype=wp.int32),
                     status: wp.array(dtype=wp.int32),
                     committed: wp.array(dtype=wp.int32),
                     ncon: wp.array(dtype=wp.int32),
                     source: wp.array(dtype=wp.int32),
                     target: wp.array(dtype=wp.int32), capacity: int):
    w, c = wp.tid()
    if active[w] == 1 and status[w] == 0 and committed[w] == 1:
        if ncon[w] >= 0 and ncon[w] <= capacity and c < ncon[w]:
            target[w * capacity + c] = source[w * capacity + c]


def support_descriptor():
    return dict(model_sha256=MJB_SHA256, nq=NQ, nv=NV, nbody=NBODY,
                equality=False, tendon=False, jacobian='dense', cone='elliptic',
                clamped_solimp_powers=[1, 2], dynamic_inputs='resident GPU arrays')


class ResidentConstraintRuntime:
    """Prepare once; producer values evolve in retained allocations.

    position_inputs is a callable resolving exactly active/ncon/qpos/cdof/
    subtree_com and the twelve typed contact leaves. velocity_inputs resolves
    qvel, qvel_epoch, expected_qvel_epoch. Independent epoch arrays must not
    overlap. Providers must return current owner attributes, not saved aliases.
    Sources may be flat or their canonical shaped layout; all are contiguous.
    The caller retains position/C/F admission and stream ordering. This class
    does not declare unknown models supported or clear sticky status itself.
    """
    def __init__(self, workspace, stream, position_inputs, velocity_inputs):
        if not isinstance(workspace, PositionReferenceWorkspace):
            raise TypeError('original PositionReferenceWorkspace required')
        self.workspace = workspace
        self.device = wp.get_device(workspace.device)
        if not self.device.is_cuda or not isinstance(stream, wp.Stream) or stream.device != self.device:
            raise ValueError('one explicit CUDA stream/device required')
        self.stream = stream
        self._stream_token = (id(stream), int(stream.cuda_stream), str(stream.device))
        self.position_inputs, self.velocity_inputs = position_inputs, velocity_inputs
        self._root = (workspace, workspace.s01, workspace.s02, workspace.s03, stream)
        self._capacities = (workspace.nworld, workspace.rows, workspace.contacts)
        self._methods = (workspace.launch_position.__func__, workspace.launch_reference.__func__)
        self._stage_methods = tuple(stage.launch.__func__ for stage in (workspace.s01,workspace.s02,workspace.s03))
        self._stage_config = self._configs()
        N, R, C = self._capacities
        I, F = wp.int32, scalar.wp_real
        self._specs = dict(active=((N,), I), ncon=((N,), I), qpos=((N, NQ), F),
                           cdof=((N, NV, 6), F), subtree_com=((N, NBODY, 3), F))
        for key in ('geom1', 'geom2', 'dim', 'exclude'):
            self._specs[key] = ((N, C), I)
        for key, width in [('pos', 3), ('frame', 9), ('dist', 1), ('margin', 1),
                           ('solref', 2), ('solreffriction', 2), ('solimp', 5), ('friction', 5)]:
            self._specs[key] = ((N, C) if width == 1 else (N, C, width), F)
        self._velocity_specs = dict(qvel=((N, NV), F), qvel_epoch=((N,), wp.int64),
                                    expected_qvel_epoch=((N,), wp.int64))
        self._sources = self._check(position_inputs(), self._specs)
        self._velocity = self._check(velocity_inputs(), self._velocity_specs)
        self._tokens = {k: self._token(a) for k, a in {**self._sources, **self._velocity}.items()}
        self._views = {}
        def bind(obj, attr, key, shape):
            a = {**self._sources, **self._velocity}[key].reshape(shape)
            setattr(obj, attr, a)
            self._views[(obj, attr)] = a
        s = workspace
        for attr in ('active', 'ncon', 'qpos', 'qvel', 'qvel_epoch', 'expected_qvel_epoch'):
            bind(s, attr, attr, (-1,))
        bind(s.s02, 'qpos', 'qpos', (-1,))
        bind(s.s01, 'ncon', 'ncon', (N,))
        for key in ('geom1','geom2','dim','exclude','pos','frame','dist','margin','cdof'):
            bind(s.s01, key, key, self._specs[key][0])
        bind(s.s01, 'subtree', 'subtree_com', (N, NBODY, 3))
        bind(s.s03, 'ncon', 'ncon', (N,))
        for attr, key in [('cref','solref'),('creff','solreffriction'),('cimp','solimp'),
                          ('cfric','friction'),('cdim','dim')]:
            bind(s.s03, attr, key, (-1,))
        # Cold structural intent only; no dynamic contact/state backfill.
        s.row_cap_flat.assign(np.full(N, R, np.int32))
        s.contact_address.assign(np.full((N, C), -1, np.int32))
        self._exclude_source = s.s01.exclude.reshape((-1,))
        self._exclude_target = s.contact_exclude.reshape((-1,))
        self._view_tokens = {key: self._token(a) for key, a in self._views.items()}
        self._exclude_tokens = (self._token(self._exclude_source), self._token(self._exclude_target))
        self._scalar_tokens = self._scalars()
        self._protected = self._arrays()
        self._protected_tokens = {k: self._token(a) for k,a in self._protected.items()}
        self.validate()

    def _configs(self):
        s = self.workspace
        return (s.s01.nworld,s.s01.maxcon,s.s01.maxrows,str(s.s01.device),
                s.s01.static.nv,s.s01.static.nbody,s.s01.static.ngeom,
                s.s02.nworld,str(s.s02.device),s.s03.worlds,s.s03.rows,
                s.s03.contacts,str(s.s03.device),s.s03.nv,s.s03.njnt,s.s03.option_domain)
    def _arrays(self):
        result = {}
        for prefix,obj in [('S',self.workspace),('S01',self.workspace.s01),('S02',self.workspace.s02),('S03',self.workspace.s03)]:
            for key,a in vars(obj).items():
                if isinstance(a,wp.array): result[(prefix,key)] = a
                elif isinstance(a,dict):
                    for leaf,value in a.items():
                        if isinstance(value,wp.array): result[(prefix,key,str(leaf))] = value
        return result

    def _scalars(self):
        s = self.workspace.s03
        tokens = []
        for key,dtype in [('timestep',np.float64),('impratio',np.float64),
                          ('disableflags',np.int32),('refsafe',np.int32)]:
            value = getattr(s,key)
            raw = np.asarray(value)
            if raw.shape != ():
                raise ValueError('scalar option shape changed: '+key)
            tokens.append((type(value),str(raw.dtype),np.asarray(value,dtype).tobytes()))
        return tuple(tokens)

    @staticmethod
    def _token(a):
        return (id(a), int(a.ptr), tuple(a.shape), tuple(a.strides), a.dtype, str(a.device))

    def _check(self, arrays, specs):
        if set(arrays) != set(specs):
            raise ValueError('resident input keyset mismatch')
        for key, (shape, dtype) in specs.items():
            a = arrays[key]
            if (not isinstance(a, wp.array) or a.device != self.device or a.dtype != dtype
                    or not a.is_contiguous or tuple(a.shape) not in (shape, (int(np.prod(shape)),))):
                raise ValueError('typed resident layout mismatch: ' + key)
        return dict(arrays)

    def validate(self):
        s = self.workspace
        if any(a is not b for a, b in zip((s,s.s01,s.s02,s.s03,self.stream), self._root)):
            raise ValueError('retained S stage/stream owner changed')
        if (s.nworld,s.rows,s.contacts) != self._capacities or wp.get_device(s.device) != self.device:
            raise ValueError('S capacity/device changed')
        if (s.launch_position.__func__, s.launch_reference.__func__) != self._methods:
            raise ValueError('S producer entry changed')
        if (id(self.stream),int(self.stream.cuda_stream),str(self.stream.device)) != self._stream_token:
            raise ValueError('stream context changed')
        if tuple(stage.launch.__func__ for stage in (s.s01,s.s02,s.s03)) != self._stage_methods or self._configs() != self._stage_config:
            raise ValueError('S segment entry or consumed structural configuration changed')
        current = {**self._check(self.position_inputs(), self._specs),
                   **self._check(self.velocity_inputs(), self._velocity_specs)}
        for key, a in current.items():
            if self._token(a) != self._tokens[key]:
                raise ValueError('current producer allocation changed: ' + key)
        for key, a in self._views.items():
            if getattr(*key) is not a or self._token(a) != self._view_tokens[key]:
                raise ValueError('S consumer view changed: ' + key[1])
        if self._exclude_tokens != (self._token(self._exclude_source), self._token(self._exclude_target)):
            raise ValueError('exclude boundary view changed')
        if self._scalars() != self._scalar_tokens:
            raise ValueError('S actual by-value option operands changed')
        protected = self._arrays()
        if set(protected) != set(self._protected):
            raise ValueError('S device field set changed')
        for key,a in protected.items():
            if a is not self._protected[key] or self._token(a) != self._protected_tokens[key]:
                raise ValueError('S device owner/layout changed: ' + '.'.join(key))
        for name in ('efc_J','efc_type','efc_id','efc_pos','efc_margin','efc_frictionloss','efc_R','efc_D','efc_KBIP','efc_vel','efc_aref','ref_epoch','counts','status','committed','contact_exclude','contact_address','contact_mu'):
            a = getattr(s,name)
            lo,hi = int(a.ptr),int(a.ptr)+a.size*wp.types.type_size_in_bytes(a.dtype)
            for key,b in current.items():
                end = int(b.ptr)+b.size*wp.types.type_size_in_bytes(b.dtype)
                if lo < end and int(b.ptr) < hi:
                    raise ValueError('public output overlaps producer input: '+name+'/'+key)
        x, y = current['qvel_epoch'], current['expected_qvel_epoch']
        if int(x.ptr) < int(y.ptr)+y.size*8 and int(y.ptr) < int(x.ptr)+x.size*8:
            raise ValueError('producer/independent expected epoch overlap')

    def launch_position(self):
        self.validate()
        s = self.workspace
        s.launch_position(stream=self.stream)
        wp.launch(_publish_exclude, dim=(s.nworld,s.contacts), inputs=[s.active,s.status,
                  s.committed,s.ncon,self._exclude_source,self._exclude_target,s.contacts],
                  device=self.device, stream=self.stream)

    def launch_reference(self):
        self.validate()
        self.workspace.launch_reference(stream=self.stream)

    def outputs(self):
        """Resident producer fields for downstream bridges; no host readback."""
        self.validate()
        s = self.workspace
        return dict(reference_ready=s.reference_ready, status=s.status, committed=s.committed,
                    ref_epoch=s.ref_epoch, counts=s.counts, efc_J=s.efc_J, efc_R=s.efc_R,
                    efc_D=s.efc_D, efc_KBIP=s.efc_KBIP, efc_vel=s.efc_vel, efc_aref=s.efc_aref)