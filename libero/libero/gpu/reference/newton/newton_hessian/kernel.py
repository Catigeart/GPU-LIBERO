"""Dense HessianDirect, HessianIncremental, HessianCone source-order CUDA kernels.

Source ownership: MuJoCo 2.3.7 engine_solver.c:1272-1494,
engine_util_solve.c:33-139, engine_util_blas.c:631-680,799-828,
engine_support.c:1084-1107. Apache-2.0. N and capacities are launch constants;
all per-world operands and scratch are preallocated on CUDA.
"""
from __future__ import annotations
import libero.gpu.precision as _precision

import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})

NV = 43
NM = 153
MINVAL = 1.0e-15


@wp.func
def _dot(a: wp.array(dtype=_precision.wp_real), p: int, q: int, n: int) -> _precision.wp_real:
    # mju_dot: four AVX lanes, pairwise reduction, then its scalar tail.
    s0 = _precision.wp_real(0.0)
    s1 = _precision.wp_real(0.0)
    s2 = _precision.wp_real(0.0)
    s3 = _precision.wp_real(0.0)
    k = int(0)
    if n >= 4:
        s0 = a[p]*a[q]
        s1 = a[p+1]*a[q+1]
        s2 = a[p+2]*a[q+2]
        s3 = a[p+3]*a[q+3]
        k = 4
    while k + 4 <= n:
        s0 = s0 + a[p+k]*a[q+k]
        s1 = s1 + a[p+k+1]*a[q+k+1]
        s2 = s2 + a[p+k+2]*a[q+k+2]
        s3 = s3 + a[p+k+3]*a[q+k+3]
        k += 4
    out = (s0+s2)+(s1+s3)
    if n-k == 3:
        out = out + (a[p+k]*a[q+k] + a[p+k+1]*a[q+k+1] + a[p+k+2]*a[q+k+2])
    elif n-k == 2:
        out = out + (a[p+k]*a[q+k] + a[p+k+1]*a[q+k+1])
    elif n-k == 1:
        out = out + a[p+k]*a[q+k]
    return out


@wp.func
def _mass_index(i: int, j: int,
          madr: wp.array(dtype=wp.int32),
          parent: wp.array(dtype=wp.int32),
          simple: wp.array(dtype=wp.int32)) -> int:
    # mj_addMDense visits one ancestor in each row. Called with i >= j.
    adr = madr[i]
    k = i
    while k >= 0:
        if k == j:
            return adr
        if simple[i] != 0:
            break
        k = parent[k]
        adr += 1
    return -1


@wp.func
def _assembly_cell(w: int, i: int, j: int, rows: int,
                   count: wp.array(dtype=wp.int32),
                   J: wp.array(dtype=_precision.wp_real),
                   D: wp.array(dtype=_precision.wp_real),
                   state: wp.array(dtype=wp.int32),
                   qm: wp.array(dtype=_precision.wp_real),
                   madr: wp.array(dtype=wp.int32),
                   parent: wp.array(dtype=wp.int32),
                   simple: wp.array(dtype=wp.int32)) -> _precision.wp_real:
    # mju_sqrMatTD: each lower cell receives the same sequence of addToScl
    # additions; symmetry copy and mj_addMDense follow afterward.
    result = _precision.wp_real(0.0)
    for r in range(count[w]):
        if state[w*rows+r] == 1:
            diag = D[w*rows+r]
            if diag != _precision.wp_real(0.0):
                tmp = J[(w*rows+r)*NV+i]
                if tmp != _precision.wp_real(0.0):
                    result = result + J[(w*rows+r)*NV+j]*(tmp*diag)
    adr = _mass_index(i, j, madr, parent, simple)
    if adr >= 0:
        result = result + qm[w*NM+adr]
    return result


@wp.func
def _factor(data: wp.array(dtype=_precision.wp_real), base: int, n: int, stride: int) -> int:
    rank = int(n)
    for j in range(n):
        diag = data[base+j*stride+j]
        if j > 0:
            diag = diag - _dot(data, base+j*stride, base+j*stride, j)
        if diag < _precision.wp_real(MINVAL):
            diag = _precision.wp_real(MINVAL)
            rank -= 1
        data[base+j*stride+j] = wp.sqrt(diag)
        inv = _precision.wp_real(1.0)/data[base+j*stride+j]
        for i in range(j+1, n):
            data[base+i*stride+j] = (data[base+i*stride+j]
                                      - _dot(data, base+i*stride, base+j*stride, j))*inv
    return rank


@wp.func
def _update(data: wp.array(dtype=_precision.wp_real), base: int,
            vec: wp.array(dtype=_precision.wp_real), vecbase: int, plus: int) -> int:
    rank = int(NV)
    for k in range(NV):
        x = vec[vecbase+k]
        if x != _precision.wp_real(0.0):
            Lkk = data[base+k*NV+k]
            tmp = Lkk*Lkk
            if plus != 0:
                tmp = tmp + x*x
            else:
                tmp = tmp - x*x
            if tmp < _precision.wp_real(MINVAL):
                tmp = _precision.wp_real(MINVAL)
                rank -= 1
            r = wp.sqrt(tmp)
            c = r/Lkk
            cinv = _precision.wp_real(1.0)/c
            s = x/Lkk
            data[base+k*NV+k] = r
            for i in range(k+1, NV):
                if plus != 0:
                    data[base+i*NV+k] = (data[base+i*NV+k]+s*vec[vecbase+i])*cinv
                else:
                    data[base+i*NV+k] = (data[base+i*NV+k]-s*vec[vecbase+i])*cinv
            for i in range(k+1, NV):
                vec[vecbase+i] = c*vec[vecbase+i]-s*data[base+i*NV+k]
    return rank


@wp.func
def _cone(w: int, rows: int, contacts: int,
          count: wp.array(dtype=wp.int32),
          state: wp.array(dtype=wp.int32),
          efc_id: wp.array(dtype=wp.int32),
          contact_dim: wp.array(dtype=wp.int32),
          contact_H: wp.array(dtype=_precision.wp_real),
          J: wp.array(dtype=_precision.wp_real),
          H: wp.array(dtype=_precision.wp_real),
          Hcone: wp.array(dtype=_precision.wp_real),
          local: wp.array(dtype=_precision.wp_real),
          ltj: wp.array(dtype=_precision.wp_real),
          vec: wp.array(dtype=_precision.wp_real),
          nupdate: wp.array(dtype=wp.int32)):
    hb = w*NV*NV
    lb = w*36
    tb = w*6*NV
    vb = w*NV
    for z in range(NV*NV):
        Hcone[hb+z] = H[hb+z]
    i = int(0)
    while i < count[w]:
        if state[w*rows+i] == 4:
            con = efc_id[w*rows+i]
            dim = contact_dim[w*contacts+con]
            for z in range(dim*dim):
                local[lb+z] = contact_H[(w*contacts+con)*36+z]
            # Dense local cholFactor return rank is deliberately ignored.
            _factor(local, lb, dim, dim)
            for z in range(dim*NV):
                ltj[tb+z] = _precision.wp_real(0.0)
            for r in range(dim):
                for c in range(r+1):
                    scalar = local[lb+r*dim+c]
                    for k in range(NV):
                        ltj[tb+c*NV+k] = ltj[tb+c*NV+k]+J[(w*rows+i+r)*NV+k]*scalar
            for r in range(dim):
                for k in range(NV):
                    vec[vb+k] = ltj[tb+r*NV+k]
                _update(Hcone, hb, vec, vb, 1)
            nupdate[w] = nupdate[w]+dim
            i += dim
        else:
            i += 1


@wp.kernel(enable_backward=False)
def validate(count: wp.array(dtype=wp.int32),
             ncon: wp.array(dtype=wp.int32),
             active: wp.array(dtype=wp.int32),
             kind: wp.array(dtype=wp.int32),
             state: wp.array(dtype=wp.int32),
             oldstate: wp.array(dtype=wp.int32),
             D: wp.array(dtype=_precision.wp_real),
             J: wp.array(dtype=_precision.wp_real),
             qm: wp.array(dtype=_precision.wp_real),
             H: wp.array(dtype=_precision.wp_real),
             efc_id: wp.array(dtype=wp.int32),
             contact_dim: wp.array(dtype=wp.int32),
             contact_H: wp.array(dtype=_precision.wp_real),
             ncone: wp.array(dtype=wp.int32),
             nnz: wp.array(dtype=wp.int32),
             status: wp.array(dtype=wp.int32),
             route: wp.array(dtype=wp.int32),
             rows: int, contacts: int):
    w = wp.tid()
    status[w] = 0
    route[w] = 0
    if active[w] == 0:
        return
    if active[w] != 1 or (kind[w] != 1 and kind[w] != 2 and kind[w] != 3):
        status[w] = 1
        return
    nr = count[w]
    nc = ncon[w]
    if nr < 0 or nr > rows or nc < 0 or nc > contacts:
        status[w] = 2
        return
    if kind[w] != 1 and nnz[w] != NV*NV:
        status[w] = 3
        return
    if kind[w] == 2 or kind[w] == 3:
        for i in range(NV):
            for j in range(NV):
                if (kind[w] == 2 or ncone[w] != 0 or j <= i) and not wp.isfinite(H[w*NV*NV+i*NV+j]):
                    status[w] = 4
                    return
    cones = int(0)
    for r in range(nr):
        s = state[w*rows+r]
        if s < 0 or s > 4:
            status[w] = 3
            return
        if kind[w] == 3 and (oldstate[w*rows+r] < 0 or oldstate[w*rows+r] > 4):
            status[w] = 3
            return
        if s == 4:
            cones += 1
        reads_d = s == 1 and kind[w] == 1
        if kind[w] == 3 and ((oldstate[w*rows+r] == 1) != (s == 1)):
            reads_d = True
        if reads_d and (D[w*rows+r] < _precision.wp_real(0.0) or not wp.isfinite(D[w*rows+r])):
            status[w] = 4
            return
        reads_j = s == 4
        if kind[w] == 1 and s == 1 and D[w*rows+r] != _precision.wp_real(0.0):
            reads_j = True
        if kind[w] == 3 and ((oldstate[w*rows+r] == 1) != (s == 1)):
            reads_j = True
        if reads_j:
            for k in range(NV):
                if not wp.isfinite(J[(w*rows+r)*NV+k]):
                    status[w] = 4
                    return
    if cones != ncone[w]:
        status[w] = 3
        return
    r = int(0)
    while r < nr:
        if state[w*rows+r] == 4:
            con = efc_id[w*rows+r]
            if con < 0 or con >= nc:
                status[w] = 3
                return
            dim = contact_dim[w*contacts+con]
            if (dim != 1 and dim != 3 and dim != 4 and dim != 6) or r+dim > nr:
                status[w] = 3
                return
            for z in range(dim):
                if state[w*rows+r+z] != 4 or efc_id[w*rows+r+z] != con:
                    status[w] = 3
                    return
            for z in range(dim*dim):
                if not wp.isfinite(contact_H[(w*contacts+con)*36+z]):
                    status[w] = 4
                    return
            r += dim
        else:
            r += 1
    if kind[w] == 1:
        for z in range(NM):
            if not wp.isfinite(qm[w*NM+z]):
                status[w] = 4
                return


@wp.kernel(enable_backward=False)
def assemble(count: wp.array(dtype=wp.int32),
             active: wp.array(dtype=wp.int32),
             kind: wp.array(dtype=wp.int32),
             J: wp.array(dtype=_precision.wp_real),
             D: wp.array(dtype=_precision.wp_real),
             state: wp.array(dtype=wp.int32),
             qm: wp.array(dtype=_precision.wp_real),
             madr: wp.array(dtype=wp.int32),
             parent: wp.array(dtype=wp.int32),
             simple: wp.array(dtype=wp.int32),
             H: wp.array(dtype=_precision.wp_real),
             status: wp.array(dtype=wp.int32), rows: int):
    tid = wp.tid()
    w = tid // (NV*NV)
    i = (tid // NV) % NV
    j = tid % NV
    if active[w] == 0 or kind[w] != 1 or status[w] != 0 or j > i:
        return
    value = _assembly_cell(w, i, j, rows, count, J, D, state, qm, madr, parent, simple)
    H[w*NV*NV+i*NV+j] = value
    if i != j:
        H[w*NV*NV+j*NV+i] = value


@wp.kernel(enable_backward=False)
def factor_update(count: wp.array(dtype=wp.int32),
                  ncon: wp.array(dtype=wp.int32),
                  active: wp.array(dtype=wp.int32),
                  kind: wp.array(dtype=wp.int32),
                  J: wp.array(dtype=_precision.wp_real),
                  D: wp.array(dtype=_precision.wp_real),
                  state: wp.array(dtype=wp.int32),
                  oldstate: wp.array(dtype=wp.int32),
                  qm: wp.array(dtype=_precision.wp_real),
                  madr: wp.array(dtype=wp.int32),
                  parent: wp.array(dtype=wp.int32),
                  simple: wp.array(dtype=wp.int32),
                  efc_id: wp.array(dtype=wp.int32),
                  contact_dim: wp.array(dtype=wp.int32),
                  contact_H: wp.array(dtype=_precision.wp_real),
                  ncone: wp.array(dtype=wp.int32),
                  H: wp.array(dtype=_precision.wp_real),
                  Hcone: wp.array(dtype=_precision.wp_real),
                  nnz: wp.array(dtype=wp.int32),
                  nupdate: wp.array(dtype=wp.int32),
                  local: wp.array(dtype=_precision.wp_real),
                  ltj: wp.array(dtype=_precision.wp_real),
                  vec: wp.array(dtype=_precision.wp_real),
                  status: wp.array(dtype=wp.int32),
                  route: wp.array(dtype=wp.int32),
                  rows: int, contacts: int):
    w = wp.tid()
    if active[w] == 0 or status[w] != 0:
        return
    hb = w*NV*NV
    vb = w*NV
    if kind[w] == 1:
        # The dense Direct source ignores the returned factor rank.
        _factor(H, hb, NV, NV)
        nnz[w] = NV*NV
        if ncone[w] != 0:
            nupdate[w] = 0
            _cone(w, rows, contacts, count, state, efc_id, contact_dim,
                  contact_H, J, H, Hcone, local, ltj, vec, nupdate)
            route[w] = 2
        else:
            route[w] = 1
        nupdate[w] = count[w]
    elif kind[w] == 2:
        _cone(w, rows, contacts, count, state, efc_id, contact_dim,
              contact_H, J, H, Hcone, local, ltj, vec, nupdate)
        route[w] = 6
    else:
        nupdate[w] = 0
        lost = int(0)
        for r in range(count[w]):
            before = oldstate[w*rows+r] == 1
            after = state[w*rows+r] == 1
            if before != after:
                scale = wp.sqrt(D[w*rows+r])
                for k in range(NV):
                    vec[vb+k] = J[(w*rows+r)*NV+k]*scale
                plus = int(0)
                if after:
                    plus = 1
                rank = _update(H, hb, vec, vb, plus)
                nupdate[w] += 1
                if rank < NV:
                    lost = 1
                    break
        if lost != 0:
            # Exactly HessianDirect's complete assembly and lower factorization.
            for z in range(NM):
                if not wp.isfinite(qm[w*NM+z]):
                    status[w] = 4
                    return
            for r in range(count[w]):
                if state[w*rows+r] == 1:
                    if D[w*rows+r] < _precision.wp_real(0.0) or not wp.isfinite(D[w*rows+r]):
                        status[w] = 4
                        return
                    if D[w*rows+r] != _precision.wp_real(0.0):
                        for k in range(NV):
                            if not wp.isfinite(J[(w*rows+r)*NV+k]):
                                status[w] = 4
                                return
            for i in range(NV):
                for j in range(i+1):
                    value = _assembly_cell(w, i, j, rows, count, J, D, state,
                                           qm, madr, parent, simple)
                    H[hb+i*NV+j] = value
                    if i != j:
                        H[hb+j*NV+i] = value
            _factor(H, hb, NV, NV)
            nnz[w] = NV*NV
            if ncone[w] != 0:
                nupdate[w] = 0
                _cone(w, rows, contacts, count, state, efc_id, contact_dim,
                      contact_H, J, H, Hcone, local, ltj, vec, nupdate)
            nupdate[w] = count[w]
            route[w] = 5
        else:
            if ncone[w] != 0:
                _cone(w, rows, contacts, count, state, efc_id, contact_dim,
                      contact_H, J, H, Hcone, local, ltj, vec, nupdate)
                route[w] = 4
            else:
                route[w] = 3
    for z in range(NV*NV):
        check_H = kind[w] == 1 or route[w] == 5
        if kind[w] == 3 and route[w] != 5 and z // NV >= z % NV:
            check_H = True
        if check_H and not wp.isfinite(H[hb+z]):
            status[w] = 5
            return
        if ncone[w] != 0 and not wp.isfinite(Hcone[hb+z]):
            status[w] = 5
            return


def launch(workspace, *, device: str = 'cuda:0'):
    """Production launch; all arguments are pre-existing CUDA arrays."""
    a = workspace
    n = len(a['active'])
    rows, contacts = a['rows'], a['contacts']
    wp.launch(validate, dim=n, inputs=[a[x] for x in
              ('count', 'ncon', 'active', 'kind', 'state', 'oldstate', 'D', 'J',
               'qm', 'H', 'efc_id', 'contact_dim', 'contact_H', 'ncone', 'nnz', 'status', 'route')]
              + [rows, contacts], device=device)
    wp.launch(assemble, dim=n*NV*NV, inputs=[a[x] for x in
              ('count', 'active', 'kind', 'J', 'D', 'state', 'qm', 'madr',
               'parent', 'simple', 'H', 'status')]+[rows], device=device)
    wp.launch(factor_update, dim=n, inputs=[a[x] for x in
              ('count', 'ncon', 'active', 'kind', 'J', 'D', 'state', 'oldstate',
               'qm', 'madr', 'parent', 'simple', 'efc_id', 'contact_dim',
               'contact_H', 'ncone', 'H', 'Hcone', 'nnz', 'nupdate',
               'local', 'ltj', 'vec', 'status', 'route')]+[rows, contacts],
              device=device)
