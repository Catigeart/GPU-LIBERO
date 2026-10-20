"""MuJoCo 2.3.7 numerical checks and masked data reset, separate from env reset.

This module does not run forward/Euler, alter model placement, clear runtime
first errors, or reset controller/host environment state. The scheduler must
rebuild derived data with a real forward before using it after a reset.
"""
import libero.gpu.precision as scalar
from dataclasses import dataclass
import hashlib
import numpy as np
import warp as wp

NWARNING = 8
BADQPOS, BADQVEL, BADQACC = 4, 5, 6
ROOT_WIDTHS = {
    'time': lambda d: 1, 'qpos': lambda d:d['nq'], 'qvel':lambda d:d['nv'],
    'qacc':lambda d:d['nv'], 'qacc_warmstart':lambda d:d['nv'],
    'ctrl':lambda d:d['nu'], 'act':lambda d:d['na'], 'act_dot':lambda d:d['na'],
    'qfrc_applied':lambda d:d['nv'], 'xfrc_applied':lambda d:6*d['nbody'],
    'userdata':lambda d:d['nuserdata'], 'sensordata':lambda d:d['nsensordata'],
    'mocap_pos':lambda d:3*d['nmocap'], 'mocap_quat':lambda d:4*d['nmocap'],
}


@wp.kernel
def _check(values:wp.array(dtype=scalar.wp_real,ndim=2), active:wp.array(dtype=wp.int32),
           reset:wp.array(dtype=wp.int32), reason:wp.array(dtype=wp.int32),
           index:wp.array(dtype=wp.int32), source_error:wp.array(dtype=wp.int32),
           width:int, warning:int):
    w = wp.tid()
    reset[w] = 0
    if active[w] != 0 and active[w] != 1:
        if source_error[w] == 0:
            source_error[w] = 1
        return
    if active[w] == 0:
        return
    for i in range(width):
        value = values[w,i]
        if wp.isnan(value) or value > scalar.wp_real(1.0e10) or value < scalar.wp_real(-1.0e10):
            reset[w] = 1
            reason[w] = warning
            index[w] = i
            break


@wp.kernel
def _reset_f64(dst:wp.array(dtype=scalar.wp_real,ndim=2), defaults:wp.array(dtype=scalar.wp_real),
               reset:wp.array(dtype=wp.int32), width:int):
    w, i = wp.tid()
    if reset[w] == 1:
        dst[w,i] = defaults[i]


@wp.kernel
def _reset_i32(dst:wp.array(dtype=wp.int32,ndim=2), defaults:wp.array(dtype=wp.int32),
               reset:wp.array(dtype=wp.int32), width:int):
    w, i = wp.tid()
    if reset[w] == 1:
        dst[w,i] = defaults[i]


@wp.kernel
def _reset_i64(dst:wp.array(dtype=wp.int64,ndim=2), defaults:wp.array(dtype=wp.int64),
               reset:wp.array(dtype=wp.int32), width:int):
    w, i = wp.tid()
    if reset[w] == 1:
        dst[w,i] = defaults[i]


@wp.kernel
def _warning_after_reset(number:wp.array(dtype=wp.int32,ndim=2),
                         lastinfo:wp.array(dtype=wp.int32,ndim=2),
                         reset:wp.array(dtype=wp.int32), reason:wp.array(dtype=wp.int32),
                         index:wp.array(dtype=wp.int32)):
    w = wp.tid()
    if reset[w] == 1:
        for i in range(8):
            number[w,i] = 0
            lastinfo[w,i] = 0
        # mj_warning executes before resetData, which clears ALL warnings.
        # The subsequent increment is always 1, never old number + 1.
        number[w,reason[w]] = 1
        lastinfo[w,reason[w]] = index[w]


def model_root_defaults(dimensions, *, qpos0, body_mocapid, body_pos, body_quat):
    """Cold model constants only; no MjData/physics/official state substitution."""
    d = {key:int(value) for key,value in dimensions.items()}
    for key in ('nq','nv','nu','na','nbody','nmocap','nuserdata','nsensordata','nplugin','npluginstate'):
        if key not in d or d[key] < 0:
            raise ValueError('missing/nonnegative native dimension: '+key)
    if d['nplugin'] or d['npluginstate']:
        raise NotImplementedError('original plugin reset callbacks require a separate GPU implementation')
    q0 = np.asarray(qpos0)
    ids = np.asarray(body_mocapid)
    pos,quat = np.asarray(body_pos),np.asarray(body_quat)
    if q0.dtype != np.float64 or q0.shape != (d['nq'],) or not np.isfinite(q0).all():
        raise ValueError('actual native qpos0 FP64 readset required')
    if ids.shape != (d['nbody'],) or ids.dtype != np.int32:
        raise ValueError('actual native body_mocapid readset required')
    if pos.shape != (d['nbody'],3) or quat.shape != (d['nbody'],4) or pos.dtype != np.float64 or quat.dtype != np.float64:
        raise ValueError('actual native body mocap defaults required')
    defaults = {name:np.zeros(width(d),dtype=np.float64) for name,width in ROOT_WIDTHS.items()}
    defaults['qpos'] = q0.copy()
    for body,identifier in enumerate(ids):
        if identifier < -1 or identifier >= d['nmocap']:
            raise ValueError('body mocap index outside native domain')
        if identifier >= 0:
            defaults['mocap_pos'].reshape(-1,3)[identifier] = pos[body]
            defaults['mocap_quat'].reshape(-1,4)[identifier] = quat[body]
    # Host model readsets stay FP64; device reset defaults follow their owners.
    return {key: value.astype(scalar.name) for key, value in defaults.items()}


def _token(array):
    return (id(array),int(array.ptr),tuple(array.shape),tuple(array.strides),
            array.dtype,str(array.device),id(getattr(array,'_ref',None)))


@dataclass(frozen=True)
class ResetBinding:
    array:object
    defaults:object
    role:str


class DataRecoveryWorkspace:
    """Prepared exact named roots plus declared native reset-buffer defaults.

    All nonempty ROOT_WIDTHS must be supplied. Extra derived/diagnostic fields
    require explicit defaults (normally +0; native sparse D/B layouts are model
    constants). Their complete arena inventory is owned by the scheduler, not
    guessed here. Native arena pointer/count invalidation is represented by
    clearing supplied counts and rebuilding with forward, not freeing CUDA
    allocations. Runtime valid/epoch/first_error are NOT native mjData buffers.
    """
    def __init__(self, dimensions, root_defaults, bindings, *, nworld, active,
                 warning_number, warning_lastinfo, stream=None,
                 current_bindings=None, mode='diagnostic'):
        if mode != 'diagnostic':
            raise ValueError('recovery candidate requires independent runtime acceptance before production')
        self.N = int(nworld)
        if self.N <= 0:
            raise ValueError('positive actual world capacity required')
        self.dimensions = dict(dimensions)
        self.device = active.device
        self.stream = stream
        self.active = active
        self.warning_number,self.warning_lastinfo = warning_number,warning_lastinfo
        if active.dtype != wp.int32 or active.shape != (self.N,):
            raise ValueError('actual active int32[N] required')
        for array in (warning_number,warning_lastinfo):
            if array.dtype != wp.int32 or array.shape != (self.N,8) or array.device != self.device:
                raise ValueError('native warning int32[N,8] arrays required')
        self.bindings = dict(bindings)
        self.current_bindings = current_bindings
        self.views,self.defaults,self.catalog = {},{},{ }
        required = {name for name,width in ROOT_WIDTHS.items() if width(self.dimensions)}
        if not required <= self.bindings.keys():
            raise ValueError('partial resetData roots prohibited: '+str(sorted(required-self.bindings.keys())))
        for name,binding in self.bindings.items():
            if not isinstance(binding,ResetBinding):
                raise TypeError('explicit reset binding required: '+name)
            array = binding.array
            data = np.asarray(binding.defaults)
            if array.device != self.device or not array.is_contiguous or array.size % self.N:
                raise ValueError('exact contiguous same-device world domain required: '+name)
            width = array.size // self.N
            if not width or data.shape != (width,):
                raise ValueError('exact per-world default shape required: '+name)
            mapping = {scalar.wp_real:np.dtype(scalar.name),wp.int32:np.dtype('int32'),wp.int64:np.dtype('int64')}
            if array.dtype not in mapping or data.dtype != mapping[array.dtype]:
                raise ValueError('explicit native scalar/int32/int64 defaults required: '+name)
            if name in required:
                expected = np.asarray(root_defaults[name])
                if expected.shape != data.shape or expected.dtype != data.dtype or expected.tobytes() != data.tobytes():
                    raise ValueError('native model reset default differs: '+name)
            elif binding.role not in ('derived','diagnostic','sparse-model-default'):
                raise ValueError('unknown extra native buffer role: '+name)
            elif binding.role in ('derived','diagnostic') and data.tobytes() != np.zeros_like(data).tobytes():
                raise ValueError('native cleared-buffer defaults must be positive zero: '+name)
            self.views[name] = array.reshape((self.N,width))
            self.defaults[name] = wp.array(data.copy(),dtype=array.dtype,device=self.device)
            self.catalog[name] = {'owner_token':_token(array),'view_token':_token(self.views[name]),
                'defaults_token':_token(self.defaults[name]),
                'width':width,'role':binding.role,'defaults_sha256':hashlib.sha256(data.tobytes()).hexdigest()}
        regions = sorted((int(b.array.ptr),int(b.array.ptr)+b.array.size*np.asarray(b.defaults).dtype.itemsize,n)
                         for n,b in self.bindings.items())
        for left,right in zip(regions,regions[1:]):
            if left[1] > right[0]:
                raise ValueError('reset bindings must declare one canonical physical owner per overlapping field: '+left[2]+'/'+right[2])
        self.reset_mask = wp.zeros(self.N,dtype=wp.int32,device=self.device)
        self.reason = wp.zeros(self.N,dtype=wp.int32,device=self.device)
        self.bad_index = wp.zeros(self.N,dtype=wp.int32,device=self.device)
        self.source_error = wp.zeros(self.N,dtype=wp.int32,device=self.device)
        self._roots = tuple(_token(a) for a in (active,warning_number,warning_lastinfo))
        self.validate_bindings()

    def validate_bindings(self):
        if tuple(_token(a) for a in (self.active,self.warning_number,self.warning_lastinfo)) != self._roots:
            raise ValueError('recovery active/warning owner changed')
        current = self.current_bindings() if self.current_bindings is not None else self.bindings
        if set(current) != set(self.bindings):
            raise ValueError('recovery native buffer inventory changed')
        for name,binding in current.items():
            array = binding.array if isinstance(binding,ResetBinding) else binding
            if _token(array) != self.catalog[name]['owner_token'] or _token(self.views[name]) != self.catalog[name]['view_token']:
                raise ValueError('recovery borrowed owner/view changed: '+name)
            if _token(self.defaults[name]) != self.catalog[name]['defaults_token']:
                raise ValueError('recovery prepared default owner changed: '+name)

    def _launch(self, field, warning, stream):
        self.validate_bindings()
        stream = self.stream if stream is None else stream
        wp.launch(_check,self.N,inputs=[self.views[field],self.active,self.reset_mask,
                  self.reason,self.bad_index,self.source_error,self.catalog[field]['width'],warning],
                  device=self.device,stream=stream)
        kernels = {scalar.wp_real:_reset_f64,wp.int32:_reset_i32,wp.int64:_reset_i64}
        for name,view in self.views.items():
            width = self.catalog[name]['width']
            wp.launch(kernels[view.dtype],(self.N,width),inputs=[view,self.defaults[name],self.reset_mask,width],device=self.device,stream=stream)
        wp.launch(_warning_after_reset,self.N,inputs=[self.warning_number,self.warning_lastinfo,
                  self.reset_mask,self.reason,self.bad_index],device=self.device,stream=stream)
        return self.reset_mask

    def check_position(self,stream=None): return self._launch('qpos',BADQPOS,stream)
    def check_velocity(self,stream=None): return self._launch('qvel',BADQVEL,stream)
    def check_acceleration(self,stream=None): return self._launch('qacc',BADQACC,stream)
