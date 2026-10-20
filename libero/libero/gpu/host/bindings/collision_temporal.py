"""Opt-in production stage injection for conservative temporal broadphase.

The simulation remains native CUDA. This cold adapter owns candidate metadata
and private workspace, seals original array owners, and replaces the existing
bounds/broadphase stages only. Tree, dispatch, detection, explicit-pair merges,
ordering, contact parameters and public raw-capacity writes remain connected.
"""
import libero.gpu.precision as scalar
from collections import UserDict
import math
import numpy as np
import warp as wp
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.model.collision_candidates import build_collision_candidates, INTEGER_FIELDS, FLOAT_FIELDS


def _signature(owners):
    return tuple((name, id(value), int(value.ptr), int(value.size), str(value.dtype), str(value.device))
                 for name, value in sorted(owners.items()))


def prepare_temporal_collision_stages(ws, stages, *, dynamics, profile='fast',
                                     absolute_padding=0.005, relative_padding=0.01,
                                     max_age=32, max_cache_bytes=8 * 1024**2):
    """Return (stages, header sources); decline without modifying old stages.

    This first implementation intentionally requires a cold worst-case proof
    for the original query/stage capacities. An expanded spatial cache must
    never turn a successful collision into an artificial capacity error.
    Fast profile admits a different valid broadphase frame and candidate list;
    compatible diagnostics retain the original PCA/SAP path and exact tapes.
    """
    if profile != 'fast':
        ws._cuda_collision_temporal_reason = 'compatibility diagnostics use original PCA/SAP'
        return stages, []
    if dynamics is None:
        ws._cuda_collision_temporal_reason = 'temporal front requires a sealed position epoch owner'
        return stages, []
    if not (math.isfinite(absolute_padding) and absolute_padding > 0 and
            math.isfinite(relative_padding) and 0 <= relative_padding <= 1 and
            isinstance(max_age, int) and 1 <= max_age <= 2147483647):
        raise ValueError('temporal collision padding/age domain')
    if not isinstance(max_cache_bytes, int) or max_cache_bytes < 0:
        raise ValueError('temporal collision byte ceiling must be nonnegative integer')
    bi = next((i for i, stage in enumerate(stages) if stage is not None and stage[0] in (
        'collision_cuda::cf_bounds', 'collision_cuda::cf_bounds_block',
        'collision_cuda::cf_bounds_cached', 'collision_cuda::cf_bounds_cached_block')), None)
    pi = next((i for i, stage in enumerate(stages) if stage is not None and stage[0] in (
        'collision_cuda::cf_broadphase', 'collision_cuda::cf_broadphase_block')), None)
    if bi is None or pi is None or bi >= pi:
        raise ValueError('temporal collision requires original bounds then broadphase stages')
    b, p = list(stages[bi][1][:17]), list(stages[pi][1])
    if len(b) != 17 or len(p) != 23:
        raise ValueError('temporal collision original bounds/broadphase ABI changed')
    if int(b[15]) != ws.ngeom or int(b[16]) != ws.nbody or int(p[21]) != ws.nbody:
        raise ValueError('temporal collision body/geom ABI mismatch')
    existing = getattr(ws, '_cuda_collision_temporal', None)
    geom = dynamics.position.frames.geom_xpos
    geom_matrix = dynamics.position.frames.geom_xmat
    if geom.ptr != b[0].ptr or geom.size != ws.nworld * ws.ngeom * 3:
        if existing is not None:
            raise ValueError('temporal position owner changed; rebuild native workspace')
        ws._cuda_collision_temporal_reason = 'position source owner/stride differs from C'
        return stages, []
    if geom_matrix.size != ws.nworld * ws.ngeom * 9 or str(geom_matrix.device) != str(ws.device):
        if existing is not None:
            raise ValueError('temporal position matrix owner/shape changed; rebuild native workspace')
        ws._cuda_collision_temporal_reason = 'position matrix owner/stride differs from C'
        return stages, []
    owners = dict(zip(INTEGER_FIELDS, (
        b[6], b[7], b[8], p[4], p[5], p[12], p[13], p[9], p[10], p[11], b[2], b[3], p[14])))
    owners.update(zip(FLOAT_FIELDS, (b[4], b[5])))
    owners.update(position=geom, geom_matrix=geom_matrix, position_valid=dynamics.valid['position'],
                  position_written=dynamics.written['position'],
                  forward_epoch=dynamics.forward_epoch, first_error=dynamics.first_error)
    owners['tree_body_num_certificate'] = ws.tree.body_num
    owners['override_enabled'] = ws.dispatch.static['override']
    owner_signature = _signature(owners)
    settings = (float(absolute_padding), float(relative_padding), max_age, max_cache_bytes)
    cached = existing
    if cached is not None:
        if cached['source_owner_signature'] != owner_signature or cached['settings'] != settings:
            raise ValueError('temporal collision owner/config changed; rebuild native workspace')
        storage = cached
    else:
        if np.any(np.asarray(owners['override_enabled'].numpy()).reshape(-1) != 0):
            ws._cuda_collision_temporal_reason = 'global contact margin override uses original collision front'
            return stages, []
        source = {name: np.asarray(owners[name].numpy()).reshape(-1)
                  for name in INTEGER_FIELDS + FLOAT_FIELDS}
        try:
            graph = build_collision_candidates(source, ws.nbody, ws.ngeom)
        except ValueError as error:
            ws._cuda_collision_temporal_reason = str(error)
            return stages, []
        npair = int(ws.tree.npair)
        total_requests = graph.max_geom_requests + npair
        needs = {'pairs': graph.pairs.size, 'requests': graph.max_geom_requests,
                 'leaves': graph.max_geom_requests,
                 'dispatch_calls': total_requests, 'stages': graph.pairs.size + npair}
        available = {'pairs': int(p[22]), 'requests': int(ws.req_slots),
                     'leaves': int(ws.tree.leaf_slots),
                     'dispatch_calls': min(int(ws.slots), int(ws.call_capacity)),
                     'stages': int(ws.stage_slots)}
        if any(needs[name] > available[name] for name in needs):
            ws._cuda_collision_temporal_reason = 'worst-case query capacity fallback: ' + repr((needs, available))
            return stages, []
        # The old traversal retains finite diagnostic tapes even in fast mode.
        # Each node-pair may be visited once; include push/pop tape entries.
        tree_num = np.asarray(ws.tree.body_num.numpy()).reshape(-1)
        max_visits = sum(int(tree_num[(int(key) >> 16) & 65535]) * int(tree_num[int(key) & 65535])
                         for key in graph.pairs)
        max_stacks = 3 * max_visits + int(graph.pairs.size)
        if max_visits > int(ws.tree.visit_slots) or max_stacks > int(ws.tree.stack_slots):
            ws._cuda_collision_temporal_reason = 'worst-case traversal tape capacity fallback'
            return stages, []
        k = int(graph.pairs.size)
        bytes_needed = (graph.pairs.nbytes + graph.global_pairs.nbytes + graph.integer_shadow.nbytes +
                        graph.float_shadow.nbytes + ws.nbody * 4 +
                        ws.nworld * (ws.nbody * 6 * 8 + ws.ngeom * 8 + k * 16 + 80))
        if bytes_needed > max_cache_bytes or ws.nworld * (k + 1) > 2147483647:
            ws._cuda_collision_temporal_reason = 'private temporal workspace byte/index ceiling fallback'
            return stages, []
        storage = UserDict(source_owner_signature=owner_signature, settings=settings,
                           retained_source_owners=UserDict(owners), graph=graph)
        storage['tree_num_shadow'] = wp.array(tree_num, dtype=wp.int32, device=ws.device)
        for name, data, dtype in (
            ('static_pair', graph.pairs, wp.int32), ('static_global', graph.global_pairs, wp.int32),
            ('integer_shadow', graph.integer_shadow, wp.int32), ('float_shadow', graph.float_shadow, scalar.wp_real)):
            storage[name] = wp.array(data if data.size else np.zeros(1, dtype=np.int32), dtype=dtype, device=ws.device)
        for name, width, dtype in (
            ('fat_bounds', ws.nbody * 6, scalar.wp_real), ('cache_pair', k, wp.int32),
            ('current_radius', ws.ngeom, scalar.wp_real),
            ('cache_global', k, wp.int32), ('flags', k, wp.int32), ('prefix', k + 1, wp.int32),
            ('cache_count', 1, wp.int32), ('cache_valid', 1, wp.int32), ('cache_mode', 1, wp.int32),
            ('cache_age', 1, wp.int32), ('reason', 1, wp.int32), ('hit_count', 1, wp.int64),
            ('refresh_count', 1, wp.int64), ('fallback_count', 1, wp.int64)):
            storage[name] = wp.zeros(max(1, ws.nworld * width), dtype=dtype, device=ws.device)
        storage['proof'] = dict(
            source_sha256=graph.source_sha256, workspace_bytes=bytes_needed,
            needed_capacity=needs, available_capacity=available,
            max_visits=max_visits, max_stack_tape=max_stacks,
            rotation='interval ||R^T R||_infinity spectral norm upper bound and 32 eps point-transform guard',
            margin='nonnegative geom margins, world max norm-squared covers native R R^T MPR margin; 64 eps guard',
            override='global contact-margin override disabled; device bit guard chooses original front if enabled',
            boundary='directed FP64 endpoints; refresh on touching/escaping fat box',
            explicit_pairs='unchanged stagebuild tails and merges run every step',
            position='valid position owner, written == forward epoch, no first error',
            statics='full integer and bitwise double shadows checked on every entry',
            diagnostics='PCA scratch, SAP scratch and removed duplicate body candidates are fast-private diagnostics',
            public_raw='existing narrowphase/finalize own complete raw-capacity writes',
            fallback='original bounds/SAP on missing epoch, static mutation, nonfinite input or incompatible model',
        )
        ws._cuda_collision_temporal = storage
    s = storage
    replacement = list(stages)
    replacement[bi] = ('collision_cuda::cf_temporal_bounds_block', [*b,
        p[4], p[5], p[12], p[13], p[9], p[10], p[11], p[14], s['integer_shadow'], s['float_shadow'],
        owners['geom_matrix'], s['current_radius'],
        owners['tree_body_num_certificate'], s['tree_num_shadow'],
        owners['override_enabled'],
        owners['position_valid'], owners['position_written'], owners['forward_epoch'], owners['first_error'],
        s['fat_bounds'], s['cache_valid'], s['cache_mode'], s['cache_age'], s['reason'],
        s['hit_count'], s['refresh_count'], s['fallback_count'], float(absolute_padding),
        float(relative_padding), max_age], 'block')
    replacement[pi] = ('collision_cuda::cf_temporal_broadphase_block', [*p,
        s['static_pair'], s['static_global'], int(s['graph'].pairs.size), s['fat_bounds'],
        s['cache_pair'], s['cache_global'], s['cache_count'], s['cache_valid'], s['cache_mode'],
        s['cache_age'], s['flags'], s['prefix'], s['hit_count'], s['refresh_count']], 'block')
    ws._cuda_collision_temporal_reason = 'conservative temporal front admitted with cold capacity certificate'
    return replacement, [GPU_ROOT / 'cuda/collision/collision_temporal.cuh']
