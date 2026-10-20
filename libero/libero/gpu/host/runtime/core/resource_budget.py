"""Single cold resource ledger for the handwritten Newton ABI.

This is an allocation estimate, not occupancy or linked-resource certification.
The default 48 KiB is the current portable launch ABI, not a GPU resource query.
"""
PORTABLE_SHARED_LIMIT_BYTES = 48 * 1024


def newton_shared_budget(nv, *, cholesky_backend='cusolverdx',
                         blas_backend='cublasdx',
                         shared_limit_bytes=PORTABLE_SHARED_LIMIT_BYTES,
                         prefix_static_bytes=0, tail_static_bytes=0,
                         newton_islands='off', scalar_bytes=8):
    nv = int(nv)
    if nv < 1:
        raise ValueError('positive NV required')
    if scalar_bytes not in (4, 8):
        raise ValueError('physical scalar must occupy 4 or 8 bytes')
    if cholesky_backend not in ('source', 'cusolverdx') or blas_backend not in ('source', 'cublasdx'):
        raise ValueError('unknown device math backend')
    if blas_backend == 'cublasdx' and cholesky_backend != 'cusolverdx':
        raise ValueError('collective BLAS requires the device POTRF path')
    if newton_islands not in ('off','exact'):
        raise ValueError('Newton islands must be off or exact')
    if newton_islands=='exact' and cholesky_backend!='cusolverdx':
        raise ValueError('exact Newton islands require the device POTRF path')
    limit, prefix, tail = map(int, (shared_limit_bytes, prefix_static_bytes, tail_static_bytes))
    if limit < 1 or prefix < 0 or tail < 0:
        raise ValueError('positive shared limit and nonnegative external reserves required')
    dx, blas = cholesky_backend == 'cusolverdx', blas_backend == 'cublasdx'
    tile_k = 16 if nv <= 43 else 8
    aligned = lambda size: (size + 15) // 16 * 16
    matrix = aligned(max(nv, 6)**2 * scalar_bytes) if dx else 0
    assembly = aligned((nv*nv + 2*nv*tile_k)*scalar_bytes) if blas else 0
    elements = max(nv*nv+2*nv*tile_k, 17*nv+16, 12*nv+36, (nv*nv+1)//2*2+nv)
    blas_workspace = aligned(elements*scalar_bytes) if blas else 0
    workspace = max(matrix, blas_workspace)
    islands_requirements=None
    if newton_islands=='exact':
        from libero.gpu.model.solver_islands import island_resource_requirements
        islands_requirements=island_resource_requirements(nv)
        workspace=max(workspace,islands_requirements['dx_workspace_min_bytes'])
    # Reduction scratch plus vendor/static control reserve. The removed line
    # search qtile is deliberately absent; all consumers use this same ledger.
    reserve = ((2560 if blas else 1536) if dx else 0) + 260
    reserve += (12*tile_k+24) if blas else 0
    reserve += prefix + tail
    if islands_requirements is not None:
        reserve+=islands_requirements['static_reserve_extra_bytes']
    return dict(shared_limit_bytes=limit, scalar_bytes=scalar_bytes, potrf_matrix_bytes=matrix,
                hessian_tile_k=tile_k if blas else None,
                hessian_workspace_bytes=assembly, blas_workspace_bytes=blas_workspace,
                dx_workspace_bytes=workspace, static_reserve_bytes=reserve,
                hot_state_budget_bytes=max(0, limit-reserve-workspace),
                admitted_by_formula=workspace+reserve <= limit,
                ledger_version=3 if islands_requirements is not None else 2,
                newton_islands=newton_islands,islands_requirements=islands_requirements,
                qualification='estimate; linked static/dynamic resources must be inspected')
