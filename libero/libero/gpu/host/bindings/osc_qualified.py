"""Explicit cold consumers of qualified OSC leaf functions.

No controller/factory callback is installed by importing this module. Call
prepare_local with opt_in=True and the exact contract from osc_local_policy.
Pass the existing function as fallback; all disabled, negative, mixed and
unmeasured selections return that same object. This API retains local function
qualification, not episode cadence or composition performance qualification.
"""
import libero.gpu.precision as scalar
from pathlib import Path
from types import MappingProxyType
from .osc_local_policy import CONTRACTS, select_scope

CONTROL = ('mass jacobian q qvel bias q0 ee_pos ee_rot ee_velocity action '
           'goal_pos goal_rot gripper active gains range ctrl raw_tau error ranks '
           'policy needs mass_inv A_full A_pos A_ori lambda_full lambda_pos '
           'lambda_ori nullspace_matrix position_error orientation_error '
           'quat_input quat_cast quat_result quat_valid nullspace_torque clipped '
           'diagnostics uncouple').split()
WIDTHS = {'mass':49,'jacobian':42,'q':7,'qvel':7,'bias':7,'q0':7,
          'ee_pos':3,'ee_rot':9,'ee_velocity':6,'action':7,'goal_pos':3,
          'goal_rot':9,'gripper':2,'gains':12,'range':18,'ctrl':9,'raw_tau':7,
          'ranks':3,'mass_inv':49,'A_full':36,'A_pos':9,'A_ori':9,
          'lambda_full':36,'lambda_pos':9,'lambda_ori':9,
          'nullspace_matrix':49,'position_error':3,'orientation_error':3,
          'quat_input':4,'quat_cast':4,'quat_result':9,'nullspace_torque':7,'clipped':7}
INTS = frozenset(('active','error','ranks','policy','needs','quat_valid'))
HELD = frozenset(('mass_inv','A_full','A_pos','A_ori','lambda_full',
                  'lambda_pos','lambda_ori','nullspace_matrix'))
SEED392 = (('mass',49),('jacobian',42),('q',7),('qvel',7),('ee_pos',3),
           ('ee_rot',9),('ee_pos_vel',3),('ee_ori_vel',3),('goal_pos',3),
           ('goal_rot',9),('gripper',2),('ctrl',9),('raw_tau',7),('clipped',7),
           ('mass_inv',49),('A_full',36),('A_pos',9),('A_ori',9),
           ('lambda_full',36),('lambda_pos',9),('lambda_ori',9),
           ('nullspace_matrix',49),('position_error',3),('orientation_error',3),
           ('quat_input',4),('quat_result',9),('nullspace_torque',7))
GATHER = ('cdof subtree_com site_pos active ancestor root body nv nbody '
          'jacp jacr jac_status adapter_active qpos qvel site_mat qM qfrc_bias '
          'qi vi sparse nq nM ee_pos ee_mat xvelp xvelr joint_pos joint_vel '
          'J M bias adapter_status status body_dof_offsets body_dof_indices').split()
GATE = ('requested first_error state_epoch source_state_epoch forward_epoch '
        'valid written active').split()
GATHER_MODEL = MappingProxyType({'nv':43,'nbody':38,'nq':48,'nM':153,'root':2,'body':13})

def _array(wp, device, value, shape, dtype):
    if (not isinstance(value, wp.array) or value.device != device or
            value.shape != shape or value.dtype != dtype or not value.is_contiguous):
        raise ValueError('OSC local bindings require exact contiguous dtype/shape/device')
    return value

def _disjoint(wp, arrays):
    # Repeated uses of one named leaf (gate active -> local active) are removed
    # by get(). Distinct names, including identical Python array objects,
    # must occupy disjoint ranges.
    spans = sorted((a.ptr, a.ptr + a.size * wp.types.type_size_in_bytes(a.dtype)) for a in arrays)
    if any(right > next_left for (_, right), (next_left, _) in zip(spans, spans[1:])):
        raise ValueError('OSC local scope requires disjoint resident leaves')

def _freeze_mapping(mapping):
    return MappingProxyType({key: _freeze_mapping(value) if isinstance(value, dict) else value
                             for key, value in mapping.items()})

class PreparedLocal:
    """A fixed same-stream leaf; mutable NativePlan setters are not exported."""
    def __init__(self, plan, stream, selection, binding):
        self._plan, self._stream = plan, stream
        self.selection = selection
        self.binding = _freeze_mapping(binding)
        self.manifest = MappingProxyType({'selection': dict(selection),
            'native': plan.manifest, 'scope': CONTRACTS[selection['algorithm_id']],
            'episode_or_composition_qualified': False, 'cadence_change': False})
    def launch(self, stream=None):
        if stream is not None and stream is not self._stream:
            raise ValueError('OSC local leaf uses its original target stream')
        self._plan.launch(self._stream)

def prepare_local(algorithm_id, nworld, bindings, build_root, *, device, stream,
                  fallback, opt_in=False, contract=None):
    """Bind one measured function, preserving the caller's original fallback.

    Bindings are resident leaves; their values remain live. Only OSC01 has
    caller-supplied pre-call seeds (goal/gripper/error/needs/policy), and only
    OSC08 copies the full 27-field 392-word FP64 prefix. No reduced seed prefix,
    direct-RHS/direct-N/controller combination or cache-enabled route is built.
    OSC04 requires admitted immutable model ancestor/CSR, checked cold below;
    qi/vi/packed-mass descriptors and every physical/clock value stay live.
    """
    selection = select_scope(algorithm_id, nworld, device.arch,
                            opt_in=opt_in, contract=contract)
    if selection['route'] != 'candidate':
        return fallback
    if algorithm_id == 'OSC04' and any(type(bindings.get(key)) is not int or
            bindings[key] != value for key,value in GATHER_MODEL.items()):
        return fallback
    import warp as wp
    from libero.gpu.host.paths import GPU_ROOT
    from libero.gpu.host.build.native import NativePlan
    if not device.is_cuda or stream.device != device or stream.is_capturing:
        raise ValueError('OSC leaf preparation requires a cold CUDA target stream')
    root = GPU_ROOT / 'cuda/control'
    headers = [root/'osc_operator.cuh', root/'osc_runtime.cuh']
    defines = {'DX_SM':device.arch*10, 'OSC_UNCOUPLE':1}
    leaves, leaf_keys = [], set()
    def get(key, width, kind='float64', source=None):
        source = bindings if source is None else source
        dtype = getattr(wp, kind)
        shape = ((nworld,) if width is None else (nworld, *width)
                 if isinstance(width, tuple) else (nworld, width))
        a = _array(wp, device, source[key], shape, dtype)
        identity = (id(source), key)
        if identity not in leaf_keys:
            leaf_keys.add(identity)
            leaves.append(a)
        return a
    mathdx = algorithm_id not in ('OSC04','OSC08')
    if algorithm_id == 'OSC01':
        defines.update(OSC_DIRECT_RHS=0, OSC_DIRECT_N=0, OSC_CACHE_ENABLED=0)
        data = {key:get(key, WIDTHS.get(key), 'int32' if key in INTS else
                'float32' if key == 'quat_cast' else 'float64')
                for key in CONTROL if key not in ('diagnostics','uncouple')}
        seeds = bindings['seed']
        stages = []
        for key in ('goal_pos','goal_rot','gripper','error','needs','policy'):
            a = get(key, WIDTHS.get(key), 'int32' if key in INTS else 'float64', seeds)
            stages.append(('osc_cuda::seed_int' if key in INTS else 'osc_cuda::seed_double',
                [a,data[key]] if key in INTS else [a,data[key],WIDTHS[key]], 'lane0'))
        cache = [wp.zeros(nworld,dtype=wp.int32,device=device),
                 wp.zeros((nworld,91),dtype=scalar.wp_real,device=device),
                 wp.zeros((nworld,145),dtype=scalar.wp_real,device=device),
                 wp.zeros((nworld,3),dtype=wp.int32,device=device),
                 wp.zeros(nworld,dtype=wp.int32,device=device),
                 wp.zeros(nworld,dtype=wp.int32,device=device),
                 wp.zeros((nworld,9),dtype=wp.int32,device=device)]
        leaves.extend(cache)
        stages.append(('osc_cuda::operator_control',
                       [1 if key in ('diagnostics','uncouple') else data[key]
                        for key in CONTROL] + cache, 'block'))
    elif algorithm_id == 'OSC02':
        defines['OSC_MASS_RECEIPT'] = 0
        receipt = wp.zeros((nworld,49),dtype=scalar.wp_real,device=device)
        leaves.append(receipt)
        stages = [('osc_component::mass',[get('mass',(7,7)),get('jacobian',(6,7)),
            get('X',42),get('A',36),get('status',None,'int32'),1,receipt],'block')]
        headers.append(root/'osc_component_qualified.cuh')
    elif algorithm_id == 'OSC03':
        headers = [root/'osc_operator.cuh',root/'osc_null_r02.cuh']
        defines['OSC_NULL_DIRECT'] = 1
        stages = [('osc_null_r02::action',[get('J',(6,7)),get('X',(7,6)),
                                           get('L',(6,6)),get('V',7),get('O',7)],'block')]
    elif algorithm_id in ('OSC06','OSC07'):
        defines['OSC_PINV_PARALLEL'] = int(algorithm_id == 'OSC06')
        defines['OSC_TASK_SOLVE'] = int(algorithm_id == 'OSC07')
        headers.append(root/'osc_component_qualified.cuh')
        stages = [('osc_component::tasks',[get('A',(6,6)),get('L',54),
            get('rank',3,'int32'),get('ok',3,'int32'),get('stats',9,'int32'),1],'block')]
    elif algorithm_id == 'OSC08':
        headers = [root/'osc.cuh',root/'osc_runtime.cuh',root/'osc_prefix_qualified.cuh']
        if set(bindings['source']) != dict(SEED392).keys() or set(bindings['target']) != dict(SEED392).keys():
            raise ValueError('OSC08 requires exactly all 27 fields of the 392-word prefix')
        stages = [('osc_cuda::seed_double',[get(key,width,source=bindings['source']),
            get(key,width,source=bindings['target']),width],'lane0') for key,width in SEED392]
        stages.append(('osc_cuda::seed_masks',[get('call',None,'int32'),
            get('refresh',None,'int32'),get('active',None,'int32'),get('needs',None,'int32')],'lane0'))
    elif algorithm_id == 'OSC04':
        import numpy as np
        b = bindings
        w, nv, nb = nworld, b['nv'], b['nbody']
        shapes = {'cdof':(w,nv,6),'subtree_com':(w,nb,3),'site_pos':(w,3),
            'active':(w,),'ancestor':(nb,nv),'jacp':(w,3,nv),'jacr':(w,3,nv),
            'jac_status':(w,),'adapter_active':(w,),'qpos':(w,b['nq']),
            'qvel':(w,nv),'site_mat':(w,9),'qM':(w,b['nM']),
            'qfrc_bias':(w,nv),'qi':(7,),'vi':(7,),'sparse':(nv,nv),
            'ee_pos':(w,3),'ee_mat':(w,9),'xvelp':(w,3),'xvelr':(w,3),
            'joint_pos':(w,7),'joint_vel':(w,7),'J':(w,6,7),'M':(w,7,7),
            'bias':(w,7),'adapter_status':(w,),'status':(w,),
            'body_dof_offsets':(nb+1,),'body_dof_indices':b['body_dof_indices'].shape}
        ints = {'active','ancestor','jac_status','adapter_active','qi','vi','sparse',
                'adapter_status','status','body_dof_offsets','body_dof_indices'}
        for key,shape in shapes.items():
            a = _array(wp,device,b[key],shape,wp.int32 if key in ints else scalar.wp_real)
            leaves.append(a)
            leaf_keys.add((id(b), key))
        offsets,indices,ancestor = (b[key].numpy() for key in
                                    ('body_dof_offsets','body_dof_indices','ancestor'))
        body = b['body']
        if (indices.ndim != 1 or offsets[0] != 0 or offsets[-1] != indices.size or
                np.any(np.diff(offsets) < 0) or np.any(indices < 0) or np.any(indices >= nv)):
            raise ValueError('OSC04 invalid immutable ancestor CSR')
        columns = indices[offsets[body]:offsets[body+1]]
        if not np.array_equal(columns, np.flatnonzero(ancestor[body] == 1)) or np.any(np.diff(columns) <= 0):
            raise ValueError('OSC04 CSR must match ascending structural ancestor columns')
        qi,vi,sparse = (b[key].numpy() for key in ('qi','vi','sparse'))
        if np.any(qi<0) or np.any(qi>=b['nq']) or np.any(vi<0) or np.any(vi>=nv) or np.any(sparse>=b['nM']):
            raise ValueError('OSC04 live descriptors outside admitted schema')
        gate = [get(key,None,'int64' if key in ('state_epoch','source_state_epoch','forward_epoch','written') else 'int32')
                for key in GATE]
        stages = [('cf_osc_gather::gate',gate,'lane0'),
                  ('cf_osc_gather::local_selected',[b[key] for key in GATHER]+[1,1],'block')]
        headers = [root/'osc_gather_qualified.cuh']
        defines = {'OSC_NV':43}
    _disjoint(wp, leaves)
    plan = NativePlan('osc_local_'+algorithm_id,headers,stages,nworld,Path(build_root),
        block_size=64,defines=defines,mathdx=mathdx,device=device,
        bridges=[root/'osc_bridge.cu'] if mathdx else [])
    prepared = PreparedLocal(plan,stream,selection,dict(bindings))
    return prepared
