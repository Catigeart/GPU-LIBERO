"""Cold schema for N private acceleration -> spatial RNE -> sensor publication.

This module binds handwritten functions; it translates no numerical arithmetic.
The Newton POTRF/TRSM matrix workspace is dead after final publication and is
reused by post. Recovery and Euler retain the reset-forward global boundary.
"""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import re
from types import MethodType
import warp as wp

HEADER = (GPU_ROOT / 'cuda/postprocess/tail.cuh')


def _parameters():
    text = HEADER.read_text()
    match = re.search(r'void post_private\((.*?)\)\{', text, re.S)
    if not match:
        raise ValueError('tail handwritten ABI declaration missing')
    return [(p.split('=')[0].strip(), p.split('=')[0].strip().split()[-1].lstrip('*'))
            for p in match.group(1).split(',')][1:]


def prepare(bundle):
    """Return external_tail schema; activate only after the core is installed."""
    post = getattr(bundle, 'post', None)
    if post is None or not hasattr(post, 'cuda_post_plan'):
        return None
    # Selected Post profiles have only standalone consumer qualification.
    # Retain that launch until the joint route is separately qualified.
    if any(getattr(post, name, {}).get('selected', False) for name in
           ('cuda_post_index_qualification', 'cuda_post_function_qualification', 'cuda_post_root_clear_qualification')):
        return None
    ws = bundle.newton.solver.ws
    source = post.cuda_post_plan
    if len(source._stage_args) != 1:
        raise ValueError('tail needs the unified post ABI')
    values = list(source._stage_args[0])
    parameters = _parameters()
    if parameters[0][1] != 'post_work' or parameters[16][1] != 'r_qa':
        raise ValueError('tail private acceleration ABI drift')
    # Omitted cache bindings retain C++ nullptr defaults.
    minimum = [name for _, name in parameters[1:]].index('diagnostics') + 1
    if not minimum <= len(values) <= len(parameters) - 1:
        raise ValueError('post/tail argument count mismatch')
    nb = int(values[[name for _, name in parameters[1:]].index('r_b')])
    needed = 18 * nb + 6
    # The core owns the actual Dx arena size (POTRF plus BLAS workspaces).
    # Its constructor checks workspace_elements against dx_bytes/8; nv**2
    # alone would wrongly reject valid small-model layouts.
    values[15] = ws.qacc
    arguments = [('post_work', None)]
    aliases = []
    for (declaration, name), value in zip(parameters[1:], values):
        if isinstance(value, wp.array) and value.dtype not in (scalar.wp_real, wp.int32, wp.int64):
            if 'real_t*' not in declaration.replace(' ', ''):
                raise TypeError(('unsupported tail vector ABI', name, value.dtype))
            count = value.size * wp.types.type_size_in_bytes(value.dtype) // 8
            value = wp.array(ptr=value.ptr, dtype=scalar.wp_real, shape=(count,),
                             device=value.device, copy=False)
            aliases.append(value)
        arguments.append((name, value))
    # Native external calls receive owner bases, but the same pointer may later
    # alias a canonical N output. Its local view must keep its real world stride.
    # These exact post ABI fields are shared model tables (never offset by w).
    shared_model = {'r_parent', 'r_da', 'r_dn', 'r_root', 'r_gb', 'r_grav',
                    'level_offsets', 'level_bodies', 'children_offsets', 'children_indices',
                    's_type', 's_site', 's_adr', 's_sb', 's_br'}
    world_strides = {}
    for name, value in arguments:
        if not isinstance(value, wp.array) or name == 'r_qa':
            continue
        if name in shared_model:
            world_strides[name] = 0
        else:
            if value.size % ws.nworld:
                raise ValueError(('tail world-major ABI size', name, value.size, ws.nworld))
            world_strides[name] = value.size // ws.nworld
    schema = dict(headers=[(GPU_ROOT / 'cuda/common/math.cuh'), (GPU_ROOT / 'cuda/postprocess/post.cuh'),
                           (GPU_ROOT / 'cuda/postprocess/post_dataflow_original.cuh'), HEADER],
                  stages=[dict(function='libero_tail::post_private',
                               arguments=arguments, perworld=1, collective=True,
                               world_strides=world_strides,
                               argument_expressions={'post_work': 'dx_matrix',
                                                     'r_qa': 'private_qacc'})],
                  world_index_argument=True, static_shared_bytes=32 * 4 + 9 * 32 * 8 + 10 * 4,
                  workspace_elements=needed,
                  diagnostic_scope='same post profile as standalone',
                  owners=aliases)
    bundle.cuda_tail_schema = schema
    return schema


def activate(bundle, core):
    """Remove the standalone post launch after successful joint-core install."""
    schema = getattr(bundle, 'cuda_tail_schema', None)
    if schema is None:
        return {'installed': False, 'reason': 'no post stage'}
    if not getattr(core, 'manifest', {}).get('external_tail_stages', 0):
        raise ValueError('cannot bypass standalone post before joint tail exists')
    adapter = bundle.post
    # Reject owner/retained-view drift before the joint kernel can publish post.
    # The N runtime wrapper seals this callable and invokes it before launch.
    def prevalidate(stream=None):
        adapter.validate(stream)
        if hasattr(adapter, 'cuda_post_cache_prevalidate'):
            adapter.cuda_post_cache_prevalidate()
        if hasattr(adapter.post, 'validate_bindings'):
            adapter.post.validate_bindings()
    bundle.newton.cuda_external_tail_prevalidate = prevalidate
    bundle.newton.cuda_external_tail_prevalidate_ref = prevalidate
    original = adapter.enqueue
    def enqueue(self, stream=None):
        self.validate(stream)
        if hasattr(self.post, 'validate_bindings'):
            self.post.validate_bindings()
        # The same N entry already performed post with the private current-world
        # acceleration and published exactly the normal standalone gate result.
    adapter.enqueue = MethodType(enqueue, adapter)
    adapter.cuda_post_tail_original = original
    bundle.runtime.forward.acceleration_sensor = lambda mask, stream: adapter.enqueue(stream=stream)
    return dict(installed=True, kernels_removed=1,
                private_qacc=True, scratch_reused=True,
                recovery_euler_boundary='reset -> full forward -> integrate remains global',
                shared_workspace_elements=schema['workspace_elements'])








