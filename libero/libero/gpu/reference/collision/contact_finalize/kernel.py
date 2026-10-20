"""Original-call ordered dedup, N12 parameters, frame, append and segment sort.

The first kernel compacts only BOX/BOX raw rows. The accepted N12 `_complete`
kernel computes parameters for compact rows (padding work is discarded). The
last kernel completes frames and appends in source call order. The only sort
is within marked mj_collideTree bodypair segments. All buffers are persistent.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

from libero.gpu.reference.collision.contact_params.kernel import ContactParamsWorkspace

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})
MAX_RAW = 50
RAW_F = 10
CONTACT_F = 65
CONTACT_I = 5


@wp.kernel(enable_backward=False)
def _dedup(raw: wp.array(dtype=scalar.wp_real), raw_count: wp.array(dtype=wp.int32),
           call_count: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
           types: wp.array(dtype=wp.int32), route: wp.array(dtype=wp.int32),
           geom_type: wp.array(dtype=wp.int32), ngeom: int,
           compact: wp.array(dtype=scalar.wp_real),
           compact_count: wp.array(dtype=wp.int32), keep_index: wp.array(dtype=wp.int32),
           status: wp.array(dtype=wp.int32), pairs: int):
    call = wp.tid()
    world = call // pairs
    local = call - world * pairs
    compact_count[call] = 0
    status[call] = 0
    for r in range(MAX_RAW):
        keep_index[call*MAX_RAW+r] = -1
        for k in range(RAW_F):
            compact[(call*MAX_RAW+r)*RAW_F+k] = scalar.wp_real(0.0)
    if active[world] == 0:
        return
    if active[world] != 1:
        status[call] = 1
        return
    nc = call_count[world]
    if nc < 0 or nc > pairs:
        status[call] = 2
        return
    if local >= nc:
        return
    g1=route[(call*MAX_RAW)*4]
    g2=route[(call*MAX_RAW)*4+1]
    if g1 < 0 or g2 < 0 or g1 >= ngeom or g2 >= ngeom:
        status[call] = 5
        return
    if types[call*2] != geom_type[g1] or types[call*2+1] != geom_type[g2]:
        status[call] = 6
        return
    nr = raw_count[call]
    if nr < 0 or nr > MAX_RAW:
        status[call] = 3
        return
    for r in range(nr):
        for k in range(RAW_F):
            if not wp.isfinite(raw[(call*MAX_RAW+r)*RAW_F+k]):
                status[call] = 4
                return
    out = wp.int32(0)
    for r in range(nr):
        discard = wp.bool(False)
        if types[call*2] == 6 and types[call*2+1] == 6:
            for s in range(r+1, nr):
                if (raw[(call*MAX_RAW+r)*RAW_F+1] == raw[(call*MAX_RAW+s)*RAW_F+1]
                    and raw[(call*MAX_RAW+r)*RAW_F+2] == raw[(call*MAX_RAW+s)*RAW_F+2]
                    and raw[(call*MAX_RAW+r)*RAW_F+3] == raw[(call*MAX_RAW+s)*RAW_F+3]):
                    discard = True
                    break
        if not discard:
            keep_index[call*MAX_RAW+out] = r
            for k in range(RAW_F):
                compact[(call*MAX_RAW+out)*RAW_F+k] = raw[(call*MAX_RAW+r)*RAW_F+k]
            out += 1
    compact_count[call] = out


@wp.func
def _sqrt3(x: scalar.wp_real, y: scalar.wp_real, z: scalar.wp_real) -> scalar.wp_real:
    return wp.sqrt((x*x + y*y) + z*z)


@wp.func
def _key_lt(ga: wp.int32, ha: wp.int32, gb: wp.int32, hb: wp.int32,
            geom_type: wp.array(dtype=wp.int32)) -> wp.bool:
    a = ga
    b = ha
    c = gb
    d = hb
    if geom_type[a] > geom_type[b]:
        a = ha
        b = ga
    if geom_type[c] > geom_type[d]:
        c = hb
        d = gb
    return a < c or (a == c and b < d)


@wp.func
def _copy_contact(dst: wp.int32, src: wp.int32, src_f: wp.array(dtype=scalar.wp_real),
                  src_i: wp.array(dtype=wp.int32), dst_f: wp.array(dtype=scalar.wp_real),
                  dst_i: wp.array(dtype=wp.int32)):
    for k in range(CONTACT_F):
        dst_f[dst*CONTACT_F+k] = src_f[src*CONTACT_F+k]
    for k in range(CONTACT_I):
        dst_i[dst*CONTACT_I+k] = src_i[src*CONTACT_I+k]


@wp.kernel(enable_backward=False)
def _assemble(compact_count: wp.array(dtype=wp.int32), keep_index: wp.array(dtype=wp.int32),
              call_count: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
              segment: wp.array(dtype=wp.int32), sort_enabled: wp.array(dtype=wp.int32),
              geom_type: wp.array(dtype=wp.int32), params_f: wp.array(dtype=scalar.wp_real),
              params_i: wp.array(dtype=wp.int32), dedup_status: wp.array(dtype=wp.int32),
              contact_f: wp.array(dtype=scalar.wp_real), contact_i: wp.array(dtype=wp.int32),
              scratch_f: wp.array(dtype=scalar.wp_real), scratch_i: wp.array(dtype=wp.int32),
              ncon: wp.array(dtype=wp.int32), attempts: wp.array(dtype=wp.int32),
              appended: wp.array(dtype=wp.int32), warnings: wp.array(dtype=wp.int32),
              status: wp.array(dtype=wp.int32), pairs: int, capacity: int):
    w = wp.tid()
    ncon[w] = 0
    warnings[w] = 0
    status[w] = 0
    for p in range(pairs):
        attempts[w*pairs+p] = 0
        appended[w*pairs+p] = 0
    if active[w] == 0:
        return
    if active[w] != 1:
        status[w] = 1
        return
    nc = call_count[w]
    if nc < 0 or nc > pairs:
        status[w] = 2
        return
    start = wp.int32(0)
    current_segment = wp.int32(-1)
    current_sort = wp.int32(0)
    for p in range(nc):
        q = w*pairs+p
        if dedup_status[q] != 0:
            if status[w] == 0:
                status[w] = 10+dedup_status[q]
            return
        seg = segment[q]
        if p == 0:
            current_segment = seg
            current_sort = sort_enabled[q]
        if seg != current_segment:
            if current_sort != 0:
                # Frozen glibc 2.35 qsort_r takes its indirect mergesort path
                # for mjContact. Equal comparator keys choose the left run.
                length = ncon[w] - start
                width = wp.int32(1)
                while width < length:
                    left = wp.int32(0)
                    while left < length:
                        mid = wp.min(left+width, length)
                        end = wp.min(left+2*width, length)
                        a = left
                        b = mid
                        t = left
                        while t < end:
                            use_right = wp.bool(False)
                            if a >= mid:
                                use_right = True
                            elif b < end:
                                ai = (w*capacity+start+a)*CONTACT_I
                                bi = (w*capacity+start+b)*CONTACT_I
                                use_right = _key_lt(contact_i[bi+1],contact_i[bi+2],
                                                    contact_i[ai+1],contact_i[ai+2],geom_type)
                            src = a
                            if use_right:
                                src = b
                                b += 1
                            else:
                                a += 1
                            _copy_contact(w*capacity+start+t,w*capacity+start+src,
                                          contact_f,contact_i,scratch_f,scratch_i)
                            t += 1
                        left += 2*width
                    for j in range(length):
                        _copy_contact(w*capacity+start+j,w*capacity+start+j,
                                      scratch_f,scratch_i,contact_f,contact_i)
                    width *= 2
            start = ncon[w]
            current_segment = seg
            current_sort = sort_enabled[q]
        count = compact_count[q]
        for r in range(count):
            slot = q*MAX_RAW+r
            if params_i[slot*5+4] != 0:
                if status[w] == 0:
                    status[w] = 20+params_i[slot*5+4]
                return
            out = w*capacity+ncon[w]
            pf = slot*26
            cf = out*CONTACT_F
            ci = out*CONTACT_I
            x0 = params_f[pf+4]
            x1 = params_f[pf+5]
            x2 = params_f[pf+6]
            nx = _sqrt3(x0,x1,x2)
            if nx < scalar.wp_real(0.5):
                if status[w] == 0:
                    status[w] = 30
                return
            if nx < scalar.wp_real(1e-15):
                x0 = scalar.wp_real(1.0); x1 = scalar.wp_real(0.0); x2 = scalar.wp_real(0.0)
            else:
                invx = scalar.wp_real(1.0)/nx
                x0 *= invx; x1 *= invx; x2 *= invx
            y0 = params_f[pf+7]
            y1 = params_f[pf+8]
            y2 = params_f[pf+9]
            if _sqrt3(y0,y1,y2) < scalar.wp_real(0.5):
                y0 = scalar.wp_real(0.0); y1 = scalar.wp_real(0.0); y2 = scalar.wp_real(0.0)
                if x1 < scalar.wp_real(0.5) and x1 > scalar.wp_real(-0.5):
                    y1 = scalar.wp_real(1.0)
                else:
                    y2 = scalar.wp_real(1.0)
            dot = (x0*y0+x1*y1)+x2*y2
            y0 -= x0*dot; y1 -= x1*dot; y2 -= x2*dot
            ny = _sqrt3(y0,y1,y2)
            if ny < scalar.wp_real(1e-15):
                y0 = scalar.wp_real(1.0); y1 = scalar.wp_real(0.0); y2 = scalar.wp_real(0.0)
            else:
                invy = scalar.wp_real(1.0)/ny
                y0 *= invy; y1 *= invy; y2 *= invy
            z0 = x1*y2-x2*y1
            z1 = x2*y0-x0*y2
            z2 = x0*y1-x1*y0
            attempts[q] += 1
            if ncon[w] >= capacity:
                warnings[w] += 1
                break  # original frame ran, then mj_addContact returns from this call
            # Exact raw field copies and all source-assigned contact fields.
            for k in range(4):
                contact_f[cf+k] = params_f[pf+k]
            contact_f[cf+4] = x0; contact_f[cf+5] = x1; contact_f[cf+6] = x2
            contact_f[cf+7] = y0; contact_f[cf+8] = y1; contact_f[cf+9] = y2
            contact_f[cf+10] = z0
            contact_f[cf+11] = z1
            contact_f[cf+12] = z2
            contact_f[cf+13] = params_f[pf+11]
            for k in range(5):
                contact_f[cf+14+k] = params_f[pf+12+k]
            for k in range(2):
                contact_f[cf+19+k] = params_f[pf+17+k]
                contact_f[cf+21+k] = params_f[pf+19+k]
            for k in range(5):
                contact_f[cf+23+k] = params_f[pf+21+k]
            contact_f[cf+28] = scalar.wp_real(0.0)
            for k in range(36):
                contact_f[cf+29+k] = scalar.wp_real(0.0)
            for k in range(4):
                contact_i[ci+k] = params_i[slot*5+k]
            contact_i[ci+4] = -1
            ncon[w] += 1
            appended[q] += 1
    # Sort the final segment, using the same merge procedure as above.
    if current_sort != 0:
        length = ncon[w] - start
        width = wp.int32(1)
        while width < length:
            left = wp.int32(0)
            while left < length:
                mid = wp.min(left+width,length)
                end = wp.min(left+2*width,length)
                a = left; b = mid; t = left
                while t < end:
                    use_right = wp.bool(False)
                    if a >= mid:
                        use_right = True
                    elif b < end:
                        ai = (w*capacity+start+a)*CONTACT_I
                        bi = (w*capacity+start+b)*CONTACT_I
                        use_right = _key_lt(contact_i[bi+1],contact_i[bi+2],
                                            contact_i[ai+1],contact_i[ai+2],geom_type)
                    src = a
                    if use_right:
                        src = b; b += 1
                    else:
                        a += 1
                    _copy_contact(w*capacity+start+t,w*capacity+start+src,
                                  contact_f,contact_i,scratch_f,scratch_i)
                    t += 1
                left += 2*width
            for j in range(length):
                _copy_contact(w*capacity+start+j,w*capacity+start+j,
                              scratch_f,scratch_i,contact_f,contact_i)
            width *= 2


class ContactFinalizeWorkspace:
    """Fixed device capacity, with prepare/upload separate from production."""
    def __init__(self, nworld, pairs, capacity, static, geom_type, device='cuda:0'):
        if min(nworld,pairs,capacity) < 1:
            raise ValueError('positive N/P/capacity required')
        self.nworld,self.pairs,self.capacity = nworld,pairs,capacity
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('CUDA required')
        geom_type=np.asarray(geom_type)
        if geom_type.dtype!=np.dtype('int32') or geom_type.shape!=(len(static['geom_priority']),):
            raise ValueError('geom_type must be original int32[ngeom]')
        self.params = ContactParamsWorkspace(nworld,pairs*MAX_RAW,static,device=device)
        self.geom_type = wp.array(np.ascontiguousarray(geom_type),dtype=wp.int32,device=device)
        n = nworld*pairs
        self.raw = wp.empty(n*MAX_RAW*RAW_F,dtype=scalar.wp_real,device=device)
        self.raw_count = wp.empty(n,dtype=wp.int32,device=device)
        self.call_count = wp.empty(nworld,dtype=wp.int32,device=device)
        self.active = self.params.active
        self.types = wp.empty(n*2,dtype=wp.int32,device=device)
        self.segment = wp.empty(n,dtype=wp.int32,device=device)
        self.sort_enabled = wp.empty(n,dtype=wp.int32,device=device)
        self.compact_count = wp.empty(n,dtype=wp.int32,device=device)
        self.keep_index = wp.empty(n*MAX_RAW,dtype=wp.int32,device=device)
        self.dedup_status = wp.empty(n,dtype=wp.int32,device=device)
        self.contact_f = wp.empty(nworld*capacity*CONTACT_F,dtype=scalar.wp_real,device=device)
        self.contact_i = wp.empty(nworld*capacity*CONTACT_I,dtype=wp.int32,device=device)
        self.scratch_f = wp.empty(nworld*capacity*CONTACT_F,dtype=scalar.wp_real,device=device)
        self.scratch_i = wp.empty(nworld*capacity*CONTACT_I,dtype=wp.int32,device=device)
        self.ncon = wp.empty(nworld,dtype=wp.int32,device=device)
        self.attempts = wp.empty(n,dtype=wp.int32,device=device)
        self.appended = wp.empty(n,dtype=wp.int32,device=device)
        self.warnings = wp.empty(nworld,dtype=wp.int32,device=device)
        self.status = wp.empty(nworld,dtype=wp.int32,device=device)

    def upload(self, raw, raw_count, call_count, active, route, usermargin,
               types, segment, sort_enabled, contact_f_prefill, contact_i_prefill,
               *, preserve_output=False):
        n,p,c=self.nworld,self.pairs,self.capacity
        specs=((raw,(n,p,50,10),np.float64,self.raw),
               (raw_count,(n,p),np.int32,self.raw_count),
               (call_count,(n,),np.int32,self.call_count),
               (active,(n,),np.int32,self.active),
               (route,(n,p,4),np.int32,None),
               (usermargin,(n,p),np.float64,None),
               (types,(n,p,2),np.int32,self.types),
               (segment,(n,p),np.int32,self.segment),
               (sort_enabled,(n,p),np.int32,self.sort_enabled),
               (contact_f_prefill,(n,c,65),np.float64,self.contact_f),
               (contact_i_prefill,(n,c,5),np.int32,self.contact_i))
        for value,shape,dtype,dest in specs:
            a=np.asarray(value)
            if a.shape!=shape or a.dtype!=dtype or not a.flags.c_contiguous:
                raise ValueError(f'expected contiguous {shape} {dtype}, got {a.shape} {a.dtype}')
            if dest is not None and not (preserve_output and
                    (dest is self.contact_f or dest is self.contact_i)):
                dest.assign(a.ravel())
        if not np.all((active==0)|(active==1)):
            raise ValueError('active must be 0/1')
        if not np.all((sort_enabled==0)|(sort_enabled==1)):
            raise ValueError('sort_enabled must be 0/1')
        for w in range(n):
            nc=int(call_count[w])
            if active[w] and 0<=nc<=p:
                for q in range(1,nc):
                    if segment[w,q]<segment[w,q-1]:
                        raise ValueError('active segment IDs must be nondecreasing')
                    if segment[w,q]==segment[w,q-1] and sort_enabled[w,q]!=sort_enabled[w,q-1]:
                        raise ValueError('sort flag must be constant within a segment')
        # Route and usermargin are call-level source inputs; only fixed row
        # replication occurs here. No dedup, params or frame is computed.
        self.params.route.assign(np.repeat(route[:,:,None,:],50,axis=2).ravel())
        self.params.usermargin.assign(np.repeat(usermargin[:,:,None],50,axis=2).ravel())
        self.params.counts.assign(np.full(n,p*50,np.int32))

    def launch(self, stream=None):
        wp.launch(_dedup,dim=self.nworld*self.pairs,
                  inputs=[self.raw,self.raw_count,self.call_count,self.active,self.types,
                          self.params.route,self.geom_type,self.params.ngeom,
                          self.params.raw,self.compact_count,self.keep_index,self.dedup_status,self.pairs],
                  device=self.device,stream=stream)
        self.params.launch(stream=stream)
        wp.launch(_assemble,dim=self.nworld,
                  inputs=[self.compact_count,self.keep_index,self.call_count,self.active,
                          self.segment,self.sort_enabled,self.geom_type,self.params.out_f,
                          self.params.out_i,self.dedup_status,self.contact_f,self.contact_i,
                          self.scratch_f,self.scratch_i,self.ncon,self.attempts,self.appended,
                          self.warnings,self.status,self.pairs,self.capacity],
                  device=self.device,stream=stream)

    def download(self):
        n,p,c=self.nworld,self.pairs,self.capacity
        return {'contact_f':self.contact_f.numpy().reshape(n,c,65),
                'contact_i':self.contact_i.numpy().reshape(n,c,5),
                'ncon':self.ncon.numpy(), 'attempts':self.attempts.numpy().reshape(n,p),
                'appended':self.appended.numpy().reshape(n,p),
                'warnings':self.warnings.numpy(), 'status':self.status.numpy(),
                'compact_count':self.compact_count.numpy().reshape(n,p),
                'keep_index':self.keep_index.numpy().reshape(n,p,50),
                # Verification-only downloads. These arrays are already written
                # by the three production kernels; no host value is fed back.
                'dedup_status':self.dedup_status.numpy().reshape(n,p),
                'compact_raw':self.params.raw.numpy().reshape(n,p,50,10),
                'params_f':self.params.out_f.numpy().reshape(n,p,50,26),
                'params_i':self.params.out_i.numpy().reshape(n,p,50,5)}
