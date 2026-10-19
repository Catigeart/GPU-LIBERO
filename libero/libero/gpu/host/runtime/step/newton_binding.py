"""Prepared model/world/Post views for the qualified single-substep Newton ABI.

The old generic workspace is borrowed only here, at preparation. Numerical
kernels never consume its generated parameter layout.
"""
import ctypes as ct
import numpy as np
import warp as wp

# Mirrors newton_coefficient_layout.cuh; checked with the compiled C++ layout.
COEFFICIENT_ROWS = 30053
COEFFICIENT_ALIGNMENT = 128
COEFFICIENT_ROW_BYTES = ((COEFFICIENT_ROWS * 8 + COEFFICIENT_ALIGNMENT - 1)
                         // COEFFICIENT_ALIGNMENT * COEFFICIENT_ALIGNMENT)
COEFFICIENT_SQRT_D = COEFFICIENT_ALIGNMENT
COEFFICIENT_FRICTION_LIMIT = COEFFICIENT_SQRT_D + COEFFICIENT_ROW_BYTES
COEFFICIENT_CONE_DM = COEFFICIENT_FRICTION_LIMIT + COEFFICIENT_ROW_BYTES
COEFFICIENT_BYTES = COEFFICIENT_CONE_DM + COEFFICIENT_ROW_BYTES


def device_workspace_bytes(nv, rows):
    """Coefficients, aliased reductions, and a bounded global Jacobian cache."""
    row_bytes = (rows * 8 + COEFFICIENT_ALIGNMENT - 1) // COEFFICIENT_ALIGNMENT * COEFFICIENT_ALIGNMENT
    end = COEFFICIENT_ALIGNMENT + 3 * row_bytes + nv * (nv + 1) // 2 * 8
    reduction_bytes = (end + 127) // 128 * 128
    entries = (rows + 1) // 2 * nv * 2 if (rows + 1) // 2 < 65536 // (2 * nv) else 65536
    row_records = (128 + (3 * nv + 1) * 4 + 127) // 128 * 128
    values = (row_records + rows * 8 + 127) // 128 * 128
    product_counts = (values + entries * 28 + 127) // 128 * 128
    cells = nv * (nv + 1) // 2
    product_values = (product_counts + cells * 4 + 127) // 128 * 128
    cache_bytes = (product_values + cells * 128 * 12 + 127) // 128 * 128
    return reduction_bytes + cache_bytes

class NewtonModelData(ct.Structure):
    _fields_ = [
        ("mass_plan_offsets", ct.c_void_p),
        ("mass_plan_coefficient", ct.c_void_p),
        ("mass_plan_source", ct.c_void_p),
        ("mass_plan_assign", ct.c_void_p),
        ("madr", ct.c_void_p),
        ("parentid", ct.c_void_p),
        ("simplenum", ct.c_void_p),
        ("model_scalars", ct.c_void_p),
    ]


class NewtonWorldData(ct.Structure):
    _fields_ = [("precomputed", ct.c_void_p),
        ("position_generation", ct.c_void_p),
        ("position_cached", ct.c_void_p),
        ("changed_rows",ct.c_void_p),
        ("hessian_algorithm_diagnostic", ct.c_void_p),
        ("hessian_upper_backup", ct.c_void_p),
        ("cone_pending_rows", ct.c_void_p),
        ("nefc", ct.c_void_p),
        ("ne", ct.c_void_p),
        ("nf", ct.c_void_p),
        ("ncon", ct.c_void_p),
        ("err_code", ct.c_void_p),
        ("efc_type", ct.c_void_p),
        ("efc_id", ct.c_void_p),
        ("cone_dim", ct.c_void_p),
        ("cone_mu", ct.c_void_p),
        ("cone_friction", ct.c_void_p),
        ("contact_dim", ct.c_void_p),
        ("contact_mu", ct.c_void_p),
        ("contact_friction", ct.c_void_p),
        ("qacc_smooth", ct.c_void_p),
        ("J", ct.c_void_p),
        ("aref", ct.c_void_p),
        ("efc_b", ct.c_void_p),
        ("jar_warm", ct.c_void_p),
        ("D", ct.c_void_p),
        ("R", ct.c_void_p),
        ("loss", ct.c_void_p),
        ("force", ct.c_void_p),
        ("state", ct.c_void_p),
        ("contact_H", ct.c_void_p),
        ("constraint_group_heads", ct.c_void_p),
        ("constraint_hessian_owner", ct.c_void_p),
        ("constraint_head_rows", ct.c_void_p),
        ("constraint_head_count", ct.c_void_p),
        ("constraint_tile_head_offset", ct.c_void_p),
        ("constraint_typed_head_rows", ct.c_void_p),
        ("constraint_typed_head_offsets", ct.c_void_p),
        ("qM", ct.c_void_p),
        ("qfrc_smooth", ct.c_void_p),
        ("jar", ct.c_void_p),
        ("oldstate", ct.c_void_p),
        ("Hcone", ct.c_void_p),
        ("eval_trace", ct.c_void_p),
        ("eval_tag", ct.c_void_p),
        ("Jv", ct.c_void_p),
        ("solver_stats_f", ct.c_void_p),
        ("solver_stats_i", ct.c_void_p),
        ("attempted", ct.c_void_p),
        ("solver_candidate", ct.c_void_p),
        ("solver_candidate_attempt", ct.c_void_p),
        ("solver_candidate_completed", ct.c_void_p),
        ("solver_candidate_origin", ct.c_void_p),
        ("solver_source_stage", ct.c_void_p),
        ("solver_rejected_final_entry", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("diagnostic", ct.c_void_p),
        ("n_written", ct.c_void_p),
        ("quadGauss", ct.c_void_p),
        ("quad", ct.c_void_p),
        ("LSresult", ct.c_void_p),
        ("external_r_ef", ct.c_void_p),
        ("canonical_qacc", ct.c_void_p),
        ("canonical_qacc_warmstart", ct.c_void_p),
        ("canonical_qfrc_constraint", ct.c_void_p),
        ("canonical_state", ct.c_void_p),
        ("canonical_solver_iter", ct.c_void_p),
        ("canonical_solver_nnz", ct.c_void_p),
        ("canonical_solver_stats_f", ct.c_void_p),
        ("canonical_solver_stats_i", ct.c_void_p),
        ("potrf_diagnostic", ct.c_void_p),
        ("blas_diagnostic", ct.c_void_p),
    ]


class NewtonPostData(ct.Structure):
    _fields_ = [
        ("external_g_active", ct.c_void_p),
        ("external_g_first", ct.c_void_p),
        ("external_g_dv", ct.c_void_p),
        ("valid", ct.c_void_p),
        ("err_code", ct.c_void_p),
        ("n_written", ct.c_void_p),
        ("attempted", ct.c_void_p),
        ("external_g_expected", ct.c_void_p),
        ("external_g_state", ct.c_void_p),
        ("external_g_consumed", ct.c_void_p),
        ("external_g_forward", ct.c_void_p),
        ("external_g_cf", ct.c_void_p),
        ("external_g_g", ct.c_void_p),
        ("external_r_qv", ct.c_void_p),
        ("external_r_ci", ct.c_void_p),
        ("external_r_cv", ct.c_void_p),
        ("external_r_cd", ct.c_void_p),
        ("external_r_cdd", ct.c_void_p),
        ("external_r_xi", ct.c_void_p),
        ("external_r_com", ct.c_void_p),
        ("external_r_sp", ct.c_void_p),
        ("external_r_sm", ct.c_void_p),
        ("external_r_xf", ct.c_void_p),
        ("external_r_ef", ct.c_void_p),
        ("external_r_nc", ct.c_void_p),
        ("nefc", ct.c_void_p),
        ("ne", ct.c_void_p),
        ("contact_dim", ct.c_void_p),
        ("external_r_adr", ct.c_void_p),
        ("external_r_g1", ct.c_void_p),
        ("external_r_g2", ct.c_void_p),
        ("external_r_frame", ct.c_void_p),
        ("external_r_pos", ct.c_void_p),
        ("external_r_parent", ct.c_void_p),
        ("external_r_da", ct.c_void_p),
        ("external_r_dn", ct.c_void_p),
        ("external_r_root", ct.c_void_p),
        ("external_r_gb", ct.c_void_p),
        ("external_r_grav", ct.c_void_p),
        ("r_cc", ct.c_int32),
        ("r_ec", ct.c_int32),
        ("external_r_acc", ct.c_void_p),
        ("external_r_ext", ct.c_void_p),
        ("external_r_intr", ct.c_void_p),
        ("external_r_decoded", ct.c_void_p),
        ("external_r_status", ct.c_void_p),
        ("r_b", ct.c_int32),
        ("r_v", ct.c_int32),
        ("r_ns", ct.c_int32),
        ("r_ng", ct.c_int32),
        ("external_s_type", ct.c_void_p),
        ("external_s_site", ct.c_void_p),
        ("external_s_adr", ct.c_void_p),
        ("external_s_sb", ct.c_void_p),
        ("external_s_br", ct.c_void_p),
        ("external_s_result", ct.c_void_p),
        ("external_s_status", ct.c_void_p),
        ("s_b", ct.c_int32),
        ("s_ns", ct.c_int32),
        ("external_c_pe", ct.c_void_p),
        ("external_c_pi", ct.c_void_p),
        ("external_c_ps", ct.c_void_p),
        ("external_c_valid", ct.c_void_p),
        ("external_c_written", ct.c_void_p),
        ("c_bw", ct.c_int32),
        ("c_sw", ct.c_int32),
        ("diagnostics", ct.c_int32),
        ("external_cache_acc", ct.c_void_p),
        ("external_cache_force", ct.c_void_p),
        ("external_cache_status", ct.c_void_p),
        ("external_cache_valid", ct.c_void_p),
        ("external_cache_state", ct.c_void_p),
        ("external_cache_forward", ct.c_void_p),
        ("cache_gravity_disabled", ct.c_int32),
        ("external_body_index", ct.c_void_p),
        ("external_body_offsets", ct.c_void_p),
        ("external_body_counts", ct.c_void_p),
        ("external_level_offsets", ct.c_void_p),
        ("external_level_bodies", ct.c_void_p),
        ("nlevels", ct.c_int32),
        ("external_children_offsets", ct.c_void_p),
        ("external_children_indices", ct.c_void_p),
        ("external_event_ping", ct.c_void_p),
        ("external_event_pong", ct.c_void_p),
        ("contact_index_mode", ct.c_int32),
        ("radix_min_contacts", ct.c_int32),
        ("external_cached_event_body", ct.c_void_p),
        ("external_cached_contact_meta", ct.c_void_p),
        ("topology_cache", ct.c_int32),
        ("published_work_acceleration", ct.c_void_p),
    ]


class NewtonArguments(ct.Structure):
    _fields_ = [("model", NewtonModelData), ("worlds", ct.c_void_p), ("post", NewtonPostData)]


def _slice(pointer,world,stride):
    return int(pointer)+world*stride if pointer else None


def prepare_newton(core, workspace, worlds, stream, *, position_cache):
    """Bind once. Topology and dimensions cannot change in this prepared path."""
    p=core.params
    nv, nm, rows, contacts = workspace.qacc.shape[1], workspace.qM.shape[1], workspace.rows, workspace.contacts
    expected = dict(nv=nv, nm=nm, rows=rows, contacts=contacts)
    if any(core.manifest['entry_specialization'][k] != value for k,value in expected.items()):
        raise ValueError("substep Newton descriptor differs from its actual workspace")
    workspace_bytes = device_workspace_bytes(nv, rows)
    # Original retained arrays are the authority for every world pitch. This
    # avoids a second table of NV39 byte offsets and preserves optional nulls.
    owner_bytes = {}
    for owner in core.owners:
        size = owner.size * int(owner.strides[-1])
        sizes = owner_bytes.setdefault(int(owner.ptr), set())
        sizes.add(size)
    def world_pointer(pointer, world):
        if not pointer:
            return None
        sizes = owner_bytes.get(int(pointer), set())
        if len(sizes) != 1:
            raise ValueError("unambiguous retained Newton owner required")
        size = next(iter(sizes))
        if size % worlds:
            raise ValueError("world-major Newton owner required")
        return _slice(pointer,world,size//worlds)
    for live,snapshot in (("madr","mass_plan_madr_snapshot"),
                          ("parentid","mass_plan_parent_snapshot"),
                          ("simplenum","mass_plan_simple_snapshot")):
        if not np.array_equal(getattr(workspace,live).numpy(),getattr(workspace,snapshot).numpy()):
            raise ValueError("single-substep Newton mass plan differs: "+live)
    model=NewtonModelData()
    model.mass_plan_offsets=p.mass_plan_offsets
    model.mass_plan_coefficient=p.mass_plan_coefficient
    model.mass_plan_source=p.mass_plan_source
    model.mass_plan_assign=p.mass_plan_assign
    if not all((model.mass_plan_offsets,model.mass_plan_coefficient,model.mass_plan_source,model.mass_plan_assign)):
        raise ValueError("single-substep Newton requires a prepared mass plan")
    model.madr=p.madr
    model.parentid=p.parentid
    model.simplenum=p.simplenum
    model.model_scalars=p.model_scalars
    with wp.ScopedStream(stream):
        changed_rows=wp.empty((worlds,rows),dtype=wp.int32,device=workspace.device)
        precomputed=wp.zeros((worlds,workspace_bytes),dtype=wp.uint8,device=workspace.device)
    records=(NewtonWorldData*worlds)()
    special = {'precomputed','position_generation','position_cached','changed_rows'}
    for world,record in enumerate(records):
        record.precomputed=_slice(precomputed.ptr,world,workspace_bytes)
        record.position_generation=_slice(position_cache.generation.ptr,world,8)
        record.position_cached=_slice(position_cache.cached.ptr,world,4)
        record.changed_rows=_slice(changed_rows.ptr,world,rows*4)
        for name,_ in NewtonWorldData._fields_:
            if name not in special:
                setattr(record,name,world_pointer(getattr(p,name),world))
    post=NewtonPostData()
    post.external_g_active=p.external_g_active
    post.external_g_first=p.external_g_first
    post.external_g_dv=p.external_g_dv
    post.valid=p.valid
    post.err_code=p.err_code
    post.n_written=p.n_written
    post.attempted=p.attempted
    post.external_g_expected=p.external_g_expected
    post.external_g_state=p.external_g_state
    post.external_g_consumed=p.external_g_consumed
    post.external_g_forward=p.external_g_forward
    post.external_g_cf=p.external_g_cf
    post.external_g_g=p.external_g_g
    post.external_r_qv=p.external_r_qv
    post.external_r_ci=p.external_r_ci
    post.external_r_cv=p.external_r_cv
    post.external_r_cd=p.external_r_cd
    post.external_r_cdd=p.external_r_cdd
    post.external_r_xi=p.external_r_xi
    post.external_r_com=p.external_r_com
    post.external_r_sp=p.external_r_sp
    post.external_r_sm=p.external_r_sm
    post.external_r_xf=p.external_r_xf
    post.external_r_ef=p.external_r_ef
    post.external_r_nc=p.external_r_nc
    post.nefc=p.nefc
    post.ne=p.ne
    post.contact_dim=p.contact_dim
    post.external_r_adr=p.external_r_adr
    post.external_r_g1=p.external_r_g1
    post.external_r_g2=p.external_r_g2
    post.external_r_frame=p.external_r_frame
    post.external_r_pos=p.external_r_pos
    post.external_r_parent=p.external_r_parent
    post.external_r_da=p.external_r_da
    post.external_r_dn=p.external_r_dn
    post.external_r_root=p.external_r_root
    post.external_r_gb=p.external_r_gb
    post.external_r_grav=p.external_r_grav
    post.r_cc=p.scalar_219
    post.r_ec=p.scalar_220
    post.external_r_acc=p.external_r_acc
    post.external_r_ext=p.external_r_ext
    post.external_r_intr=p.external_r_intr
    post.external_r_decoded=p.external_r_decoded
    post.external_r_status=p.external_r_status
    post.r_b=p.scalar_226
    post.r_v=p.scalar_227
    post.r_ns=p.scalar_228
    post.r_ng=p.scalar_229
    post.external_s_type=p.external_s_type
    post.external_s_site=p.external_s_site
    post.external_s_adr=p.external_s_adr
    post.external_s_sb=p.external_s_sb
    post.external_s_br=p.external_s_br
    post.external_s_result=p.external_s_result
    post.external_s_status=p.external_s_status
    post.s_b=p.scalar_237
    post.s_ns=p.scalar_238
    post.external_c_pe=p.external_c_pe
    post.external_c_pi=p.external_c_pi
    post.external_c_ps=p.external_c_ps
    post.external_c_valid=p.external_c_valid
    post.external_c_written=p.external_c_written
    post.c_bw=p.scalar_244
    post.c_sw=p.scalar_245
    post.diagnostics=p.scalar_246
    post.external_cache_acc=p.external_cache_acc
    post.external_cache_force=p.external_cache_force
    post.external_cache_status=p.external_cache_status
    post.external_cache_valid=p.external_cache_valid
    post.external_cache_state=p.external_cache_state
    post.external_cache_forward=p.external_cache_forward
    post.cache_gravity_disabled=p.scalar_253
    post.external_body_index=p.external_body_index
    post.external_body_offsets=p.external_body_offsets
    post.external_body_counts=p.external_body_counts
    post.external_level_offsets=p.external_level_offsets
    post.external_level_bodies=p.external_level_bodies
    post.nlevels=p.scalar_259
    post.external_children_offsets=p.external_children_offsets
    post.external_children_indices=p.external_children_indices
    post.external_event_ping=p.external_event_ping
    post.external_event_pong=p.external_event_pong
    post.contact_index_mode=p.scalar_264
    post.radix_min_contacts=p.scalar_265
    post.external_cached_event_body=p.external_cached_event_body
    post.external_cached_contact_meta=p.external_cached_contact_meta
    post.topology_cache=p.scalar_268
    post.published_work_acceleration=p.qacc
    with wp.ScopedStream(stream):
        world_data=wp.array(np.frombuffer(bytes(records),dtype=np.uint8).copy(),dtype=wp.uint8,device=workspace.device)
        args=NewtonArguments(model,world_data.ptr,post)
        device_data=wp.array(np.frombuffer(bytes(args),dtype=np.uint8).copy(),dtype=wp.uint8,device=workspace.device)
    return device_data,(world_data,device_data,changed_rows,precomputed)
