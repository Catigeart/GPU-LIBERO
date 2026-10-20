"""MuJoCo 2.3.7 body SAP from a frozen body AABB boundary.

One CUDA thread owns one world.  This deliberately serial implementation keeps
the original append and active-list order.  The platform qsort_r used by the
frozen CPU is stable for equal endpoint values; the insertion sort below has
that same tie order.  The CPU capture verifies that claim for every real row.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


@wp.func
def _filtered(w1: int, p1: int, w2: int, p2: int, disabled: int) -> bool:
    if w1 == w2:
        return True
    if disabled == 0 and w1 != 0 and w2 != 0:
        if w1 == p2 or w2 == p1:
            return True
    return False


@wp.func
def _signature(b1: int, b2: int) -> int:
    if b1 < b2:
        return (b1 << 16) + b2
    return (b2 << 16) + b1


@wp.kernel(enable_backward=False)
def _sap(
    aabb: wp.array(dtype=scalar.wp_real), active: wp.array(dtype=wp.int32),
    maxpair: wp.array(dtype=wp.int32),
    geomadr: wp.array(dtype=wp.int32), geomnum: wp.array(dtype=wp.int32),
    weldid: wp.array(dtype=wp.int32), parentid: wp.array(dtype=wp.int32),
    geomtype: wp.array(dtype=wp.int32), contype: wp.array(dtype=wp.int32),
    affinity: wp.array(dtype=wp.int32),
    can_collide: wp.array(dtype=wp.int32),
    has_plane: wp.array(dtype=wp.int32),
    sort_value: wp.array(dtype=wp.float32), sort_tag: wp.array(dtype=wp.int32),
    active_tag: wp.array(dtype=wp.int32), scratch_pair: wp.array(dtype=wp.int32),
    raw_pair_trace: wp.array(dtype=wp.int32),
    sweep_count: wp.array(dtype=wp.int32), sweep_tag: wp.array(dtype=wp.int32),
    out_pair: wp.array(dtype=wp.int32), out_npair: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    nbody: int, pair_capacity: int, pair_stride: int, filterparent_disabled: int,
    has_nonworld_geom: int,
):
    w = wp.tid()
    if active[w] == 0:
        return
    cap = maxpair[w]
    if cap < 0 or cap > pair_capacity:
        status[w] = 2
        return
    npair = int(0)
    failed = bool(False)
    # Original special-pair order: outer b1, inner b2, no affinity filter.
    for b1 in range(nbody):
        if can_collide[b1] != 0:
            if (b1 == 0 and geomnum[b1] > 0) or (weldid[b1] == 0 and has_plane[b1] != 0):
                for b2 in range(nbody):
                    w2 = weldid[b2]
                    p2 = weldid[parentid[w2]]
                    if not _filtered(0, 0, w2, p2, filterparent_disabled):
                        if npair >= cap:
                            failed = True
                            break
                        sig = _signature(b1, b2)
                        scratch_pair[w*pair_capacity + npair] = sig
                        raw_pair_trace[w*pair_capacity + npair] = sig
                        npair += 1
            if failed:
                break
    if failed:
        status[w] = 1
        return
    # The original returns early, before the final pair sort, if no nonworld
    # geom exists.  No body AABB is read on that path.
    if has_nonworld_geom == 0:
        for i in range(npair):
            out_pair[w*pair_stride+i] = scratch_pair[w*pair_capacity+i]
        out_npair[w] = npair
        status[w] = 0
        return

    ne = int(0)
    for b in range(1, nbody):
        if can_collide[b] != 0:
            j = w*2*nbody+ne
            sort_value[j] = wp.float32(aabb[(w*nbody+b)*6])
            sort_tag[j] = b
            ne += 1
            j = w*2*nbody+ne
            sort_value[j] = wp.float32(aabb[(w*nbody+b)*6+1])
            sort_tag[j] = b + 65536
            ne += 1
    # Frozen glibc qsort_r is stable on equal float keys.  Swap only for '<'.
    for i in range(1, ne):
        keyv = sort_value[w*2*nbody+i]
        keyt = sort_tag[w*2*nbody+i]
        j = i
        while j > 0:
            if not keyv < sort_value[w*2*nbody+j-1]:
                break
            sort_value[w*2*nbody+j] = sort_value[w*2*nbody+j-1]
            sort_tag[w*2*nbody+j] = sort_tag[w*2*nbody+j-1]
            j -= 1
        sort_value[w*2*nbody+j] = keyv
        sort_tag[w*2*nbody+j] = keyt

    nactive = int(0)
    for i in range(ne):
        tag = sort_tag[w*2*nbody+i]
        if (tag & 65536) == 0:
            b2 = tag
            for j in range(nactive):
                b1 = active_tag[w*nbody+j]
                w1 = weldid[b1]
                w2 = weldid[b2]
                p1 = weldid[parentid[w1]]
                p2 = weldid[parentid[w2]]
                if _filtered(w1, p1, w2, p2, filterparent_disabled):
                    continue
                a1 = (w*nbody+b1)*6
                a2 = (w*nbody+b2)*6
                if (aabb[a1+2] > aabb[a2+3] or aabb[a1+3] < aabb[a2+2] or
                    aabb[a1+4] > aabb[a2+5] or aabb[a1+5] < aabb[a2+4]):
                    continue
                # add_pair checks capacity before the single-geom affinity
                # test.  Thus cap=0 can fail even for a filtered candidate.
                if npair >= cap:
                    failed = True
                    break
                compatible = bool(True)
                if geomnum[b1] == 1 and geomnum[b2] == 1:
                    g1 = geomadr[b1]
                    g2 = geomadr[b2]
                    if (contype[g1] & affinity[g2]) == 0 and (contype[g2] & affinity[g1]) == 0:
                        compatible = False
                if compatible:
                    sig = _signature(b1, b2)
                    scratch_pair[w*pair_capacity+npair] = sig
                    raw_pair_trace[w*pair_capacity+npair] = sig
                    npair += 1
            if failed:
                break
            active_tag[w*nbody+nactive] = tag
            nactive += 1
        else:
            remove = tag & 65535
            for j in range(nactive):
                if active_tag[w*nbody+j] == remove:
                    for k in range(j, nactive-1):
                        active_tag[w*nbody+k] = active_tag[w*nbody+k+1]
                    nactive -= 1
                    break
        sweep_count[w*2*nbody+i] = nactive
        for j in range(nactive):
            sweep_tag[(w*2*nbody+i)*nbody+j] = active_tag[w*nbody+j]
    if failed:
        status[w] = 1
        return
    # paircompare sorts by signed 32-bit signature, with duplicates retained.
    for i in range(1, npair):
        key = scratch_pair[w*pair_capacity+i]
        j = i
        while j > 0:
            if not key < scratch_pair[w*pair_capacity+j-1]:
                break
            scratch_pair[w*pair_capacity+j] = scratch_pair[w*pair_capacity+j-1]
            j -= 1
        scratch_pair[w*pair_capacity+j] = key
    for i in range(npair):
        out_pair[w*pair_stride+i] = scratch_pair[w*pair_capacity+i]
    out_npair[w] = npair
    status[w] = 0


class SAPWorkspace:
    """Static model prepare plus preallocated world buffers; launch is device only."""

    def __init__(self, static: dict, nworld: int, pair_capacity: int, *, device='cuda:0',
                 filterparent_disabled=0):
        if nworld < 1 or pair_capacity < 1:
            raise ValueError('positive world and pair capacity required')
        required = ('body_geomadr','body_geomnum','body_weldid','body_parentid',
                    'geom_type','geom_contype','geom_conaffinity','geom_bodyid')
        for key in required:
            arr = static[key]
            if not isinstance(arr, np.ndarray) or arr.ndim != 1 or arr.dtype != np.int32:
                raise ValueError('static schema '+key)
        B = len(static['body_geomnum'])
        G = len(static['geom_type'])
        if B < 1 or B > 32767 or any(len(static[k]) != B for k in required[:4]) or any(len(static[k]) != G for k in required[4:]):
            raise ValueError('static dimensions')
        if filterparent_disabled not in (0,1):
            raise ValueError('filterparent_disabled')
        for b in range(B):
            start, count = int(static['body_geomadr'][b]), int(static['body_geomnum'][b])
            if count < 0 or (count and (start < 0 or start+count > G)):
                raise ValueError('body geom range')
            p, weld = int(static['body_parentid'][b]), int(static['body_weldid'][b])
            if p < 0 or p >= B or weld < 0 or weld >= B:
                raise ValueError('body tree range')
        if np.any(static['geom_bodyid'] < 0) or np.any(static['geom_bodyid'] >= B):
            raise ValueError('geom body range')
        self.device, self.nworld, self.nbody, self.pair_capacity = device,nworld,B,pair_capacity
        self.pair_stride=pair_capacity+2
        self.filterparent_disabled = filterparent_disabled
        self.has_nonworld_geom = int(np.any(static['geom_bodyid'] != 0))
        can = np.zeros(B,np.int32); plane = np.zeros(B,np.int32)
        for b in range(B):
            start, count = int(static['body_geomadr'][b]), int(static['body_geomnum'][b])
            if count:
                sl = slice(start,start+count)
                can[b] = int(np.any(static['geom_contype'][sl] != 0) or np.any(static['geom_conaffinity'][sl] != 0))
                plane[b] = int(np.any((static['geom_type'][sl] == 0) | (static['geom_type'][sl] == 1)))
        self.can_host = can
        def device_int(a):
            # Warp device arrays need an address even for a legal zero-geom
            # model.  This dummy slot is never read when every geomnum is 0.
            return wp.array(np.ascontiguousarray(a if len(a) else np.zeros(1,np.int32)),
                            dtype=wp.int32,device=device)
        self.static = [device_int(static[k])
                       for k in required[:2]]
        self.static += [device_int(static[k])
                        for k in required[2:4]]
        self.static += [device_int(static[k])
                        for k in required[4:7]]
        self.can = wp.array(can,dtype=wp.int32,device=device)
        self.plane = wp.array(plane,dtype=wp.int32,device=device)
        self.aabb = wp.empty(nworld*B*6,dtype=scalar.wp_real,device=device)
        self.active = wp.empty(nworld,dtype=wp.int32,device=device)
        self.maxpair = wp.empty(nworld,dtype=wp.int32,device=device)
        self.sort_value = wp.zeros(nworld*2*B,dtype=wp.float32,device=device)
        self.sort_tag = wp.zeros(nworld*2*B,dtype=wp.int32,device=device)
        self.active_tag = wp.zeros(nworld*B,dtype=wp.int32,device=device)
        self.scratch_pair = wp.zeros(nworld*pair_capacity,dtype=wp.int32,device=device)
        self.raw_pair_trace = wp.zeros(nworld*pair_capacity,dtype=wp.int32,device=device)
        self.sweep_count = wp.zeros(nworld*2*B,dtype=wp.int32,device=device)
        self.sweep_tag = wp.zeros(nworld*2*B*B,dtype=wp.int32,device=device)
        self.out_pair = wp.empty(nworld*self.pair_stride,dtype=wp.int32,device=device)
        self.guard_host=np.empty((nworld,2),np.int32)
        for w in range(nworld):
            self.guard_host[w]=[np.int32(0x5a5a0000+2*w),np.int32(0x5a5a0001+2*w)]
        self.out_npair = wp.empty(nworld,dtype=wp.int32,device=device)
        self.status = wp.empty(nworld,dtype=wp.int32,device=device)

    def upload(self, aabb: np.ndarray, active: np.ndarray, maxpair: np.ndarray,
               old_pair: np.ndarray, old_npair: np.ndarray, old_status: np.ndarray):
        specs = [(aabb,(self.nworld,self.nbody,6),np.float64,self.aabb),
                 (active,(self.nworld,),np.int32,self.active),
                 (maxpair,(self.nworld,),np.int32,self.maxpair),
                 (old_npair,(self.nworld,),np.int32,self.out_npair),
                 (old_status,(self.nworld,),np.int32,self.status)]
        for value, shape, dtype, dst in specs:
            if not isinstance(value,np.ndarray) or value.shape != shape or value.dtype != dtype:
                raise ValueError('input dtype/shape')
        if not isinstance(old_pair,np.ndarray) or old_pair.shape != (self.nworld,self.pair_capacity) or old_pair.dtype != np.int32:
            raise ValueError('input dtype/shape old_pair')
        if not np.all((active==0)|(active==1)):
            raise ValueError('active mask')
        if self.has_nonworld_geom:
            body_mask=self.can_host.astype(bool)
            body_mask[0]=False  # original never reads the world AABB entry
            used = aabb[active==1][:,body_mask]
            if not np.all(np.isfinite(used)) or np.any(np.abs(used[:,:,0:2])>np.finfo(np.float32).max):
                raise ValueError('nonfinite or FP32-overflow reached AABB')
        for value, _, _, dst in specs:
            dst.assign(np.ascontiguousarray(value).ravel())
        packed=np.empty((self.nworld,self.pair_stride),np.int32)
        packed[:,:self.pair_capacity]=old_pair
        packed[:,self.pair_capacity:]=self.guard_host
        self.out_pair.assign(packed.ravel())

    def launch(self, stream=None):
        wp.launch(_sap,dim=self.nworld,
                  inputs=[self.aabb,self.active,self.maxpair,*self.static,self.can,self.plane,
                          self.sort_value,self.sort_tag,self.active_tag,self.scratch_pair,self.raw_pair_trace,
                          self.sweep_count,self.sweep_tag,
                          self.out_pair,self.out_npair,self.status,self.nbody,self.pair_capacity,
                          self.pair_stride,self.filterparent_disabled,self.has_nonworld_geom],
                  device=self.device,stream=stream)

    def download(self) -> dict:
        all_pair=self.out_pair.numpy().reshape(self.nworld,self.pair_stride)
        return {'pair':all_pair[:,:self.pair_capacity].copy(),
                'pair_guard':all_pair[:,self.pair_capacity:].copy(),
                'npair':self.out_npair.numpy(), 'status':self.status.numpy(),
                'sorted_value':self.sort_value.numpy().reshape(self.nworld,2*self.nbody),
                'sorted_tag':self.sort_tag.numpy().reshape(self.nworld,2*self.nbody),
                'raw_pair':self.raw_pair_trace.numpy().reshape(self.nworld,self.pair_capacity),
                'sweep_count':self.sweep_count.numpy().reshape(self.nworld,2*self.nbody),
                'sweep_tag':self.sweep_tag.numpy().reshape(self.nworld,2*self.nbody,self.nbody)}
