"""Nine SAT cross axes given the original CPU six-face prefix.

Source: MuJoCo 2.3.7 engine_collision_box.c:661-718, SHA-256
fd038c185abfae66213bcabd0db34450a5fb596481fa5d1bb567fdff62a587b2,
Apache-2.0 (copyright 2016 Svetoslav Kolev). This is an independent device
transcription. No contact generation or face-prefix device arithmetic occurs.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})


@wp.kernel(enable_backward=False)
def _cross(
    prefix: wp.array(dtype=scalar.wp_real),
    prefix_state: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    axis: wp.array(dtype=scalar.wp_real),
    axis_mask: wp.array(dtype=wp.int32),
    final_f: wp.array(dtype=scalar.wp_real),
    final_state: wp.array(dtype=wp.int32),
    pairs: int,
):
    tid = wp.tid()
    w = tid // pairs
    q = tid - w * pairs
    for z in range(54):
        axis[tid * 54 + z] = scalar.wp_real(0.0)
    for z in range(9):
        axis_mask[tid * 9 + z] = 0
    for z in range(4):
        final_f[tid * 4 + z] = scalar.wp_real(0.0)
    for z in range(8):
        final_state[tid * 8 + z] = -1
    final_state[tid * 8 + 5] = 0
    if active[w] == 0:
        return
    if counts[w] < 0 or counts[w] > pairs:
        final_state[tid * 8 + 5] = 1
        return
    if q >= counts[w]:
        return
    if prefix_state[tid * 3 + 1] != 0:
        final_state[tid * 8 + 6] = 1
        final_state[tid * 8 + 7] = prefix_state[tid * 3 + 2]
        return
    final_state[tid * 8 + 6] = 0

    b = tid * 38
    penetration = prefix[b + 37]
    code = prefix_state[tid * 3]
    cle1 = wp.int32(-1)
    cle2 = wp.int32(-1)
    inside = wp.int32(-1)
    clnorm0 = scalar.wp_real(0.0)
    clnorm1 = scalar.wp_real(0.0)
    clnorm2 = scalar.wp_real(0.0)
    first_exit = wp.int32(-1)
    projection_scale = scalar.wp_real(0.0)
    if wp.static(scalar.bits == 32):
        for k in range(3):
            projection_scale += prefix[b + 30 + k] + prefix[b + 33 + k] + wp.abs(prefix[b + 27 + k])

    for i in range(3):
        for j in range(3):
            t = i * 3 + j
            # Original tmp2 is axis_i x rott_row_j, including signed zeros.
            x0 = scalar.wp_real(0.0)
            x1 = scalar.wp_real(0.0)
            x2 = scalar.wp_real(0.0)
            if i == 0:
                x1 = -prefix[b + 9 + 3 * j + 2]
                x2 = prefix[b + 9 + 3 * j + 1]
            elif i == 1:
                x0 = prefix[b + 9 + 3 * j + 2]
                x2 = -prefix[b + 9 + 3 * j]
            else:
                x0 = -prefix[b + 9 + 3 * j + 1]
                x1 = prefix[b + 9 + 3 * j]
            length = wp.sqrt((x0 * x0 + x1 * x1) + x2 * x2)
            if length < scalar.wp_real(1.0e-15):
                x0 = scalar.wp_real(1.0)
                x1 = scalar.wp_real(0.0)
                x2 = scalar.wp_real(0.0)
            else:
                inv = scalar.wp_real(1.0) / length
                x0 = x0 * inv
                x1 = x1 * inv
                x2 = x2 * inv
            axis[tid * 54 + t * 6] = length
            axis[tid * 54 + t * 6 + 1] = x0
            axis[tid * 54 + t * 6 + 2] = x1
            axis[tid * 54 + t * 6 + 3] = x2
            axis_mask[tid * 9 + t] = 1
            if length < scalar.wp_real(1.0e-15):
                axis_mask[tid * 9 + t] = 2
                continue
            c2 = (prefix[b + 27] * x0 + prefix[b + 28] * x1) + prefix[b + 29] * x2
            c3 = scalar.wp_real(0.0)
            for k in range(3):
                if k != i:
                    xv = x0
                    if k == 1:
                        xv = x1
                    if k == 2:
                        xv = x2
                    c3 = c3 + prefix[b + 30 + k] * wp.abs(xv)
            for k in range(3):
                if k != j:
                    c3 = c3 + prefix[b + 33 + k] * prefix[b + 18 + 3 * i + 3 - k - j] / length
            c3 = c3 - wp.abs(c2)
            axis[tid * 54 + t * 6 + 4] = c2
            axis[tid * 54 + t * 6 + 5] = c3
            if c3 < -prefix[b + 36]:
                first_exit = t
                break
            prefer_edge = c3 < penetration * (scalar.wp_real(1.0) - scalar.wp_real(1.0e-12))
            if wp.static(scalar.bits == 32):
                # Keep the face for gains unresolved at FP32 projection precision.
                uncertainty = scalar.wp_real(32.0) * scalar.wp_real(scalar.epsilon) * projection_scale / wp.max(length, scalar.wp_real(scalar.epsilon))
                prefer_edge = c3 < penetration - uncertainty
            if prefer_edge:
                penetration = c3
                cle1 = 0
                cle2 = 0
                for k in range(3):
                    if k != i:
                        xv = x0
                        if k == 1:
                            xv = x1
                        if k == 2:
                            xv = x2
                        if (xv > scalar.wp_real(0.0)) != (c2 < scalar.wp_real(0.0)):
                            cle1 = cle1 + (1 << k)
                for k in range(3):
                    if k != j:
                        flag = (prefix[b + 3 * i + 3 - k - j] > scalar.wp_real(0.0)) != (c2 < scalar.wp_real(0.0))
                        flag = flag != ((k - j + 3) % 3 == 1)
                        if flag:
                            cle2 = cle2 + (1 << k)
                code = 12 + i * 3 + j
                clnorm0 = x0
                clnorm1 = x1
                clnorm2 = x2
                inside = wp.int32(c2 < scalar.wp_real(0.0))
        if first_exit >= 0:
            break
    final_f[tid * 4] = penetration
    if code >= 12:
        final_f[tid * 4 + 1] = clnorm0
        final_f[tid * 4 + 2] = clnorm1
        final_f[tid * 4 + 3] = clnorm2
    final_state[tid * 8] = code
    final_state[tid * 8 + 1] = cle1
    final_state[tid * 8 + 2] = cle2
    final_state[tid * 8 + 3] = inside
    final_state[tid * 8 + 4] = first_exit
    final_state[tid * 8 + 5] = 0


class BoxCrossWorkspace:
    """All buffers are allocated once; launch performs one CUDA kernel only."""

    def __init__(self, nworld: int, pairs: int, device: str = "cuda:0"):
        if nworld < 1 or pairs < 1:
            raise ValueError("positive N and P required")
        self.nworld, self.pairs, self.device = nworld, pairs, device
        self.prefix = wp.zeros(nworld * pairs * 38, dtype=scalar.wp_real, device=device)
        self.prefix_state = wp.zeros(nworld * pairs * 3, dtype=wp.int32, device=device)
        self.active = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.counts = wp.zeros(nworld, dtype=wp.int32, device=device)
        self.axis = wp.zeros(nworld * pairs * 54, dtype=scalar.wp_real, device=device)
        self.axis_mask = wp.zeros(nworld * pairs * 9, dtype=wp.int32, device=device)
        self.final_f = wp.zeros(nworld * pairs * 4, dtype=scalar.wp_real, device=device)
        self.final_state = wp.zeros(nworld * pairs * 8, dtype=wp.int32, device=device)

    def upload(self, prefix, prefix_state, active, counts):
        """One-time or test-boundary upload; never called by launch."""
        fields=((prefix,(self.nworld,self.pairs,38),np.dtype('float64'),'prefix',self.prefix),
                (prefix_state,(self.nworld,self.pairs,3),np.dtype('int32'),'prefix_state',self.prefix_state),
                (active,(self.nworld,),np.dtype('int32'),'active',self.active),
                (counts,(self.nworld,),np.dtype('int32'),'counts',self.counts))
        arrays={}
        for value,shape,dtype,name,_ in fields:
            arr=np.asarray(value)
            if arr.shape!=shape or arr.dtype!=dtype:
                raise ValueError(f'{name}: expected {shape} {dtype}, got {arr.shape} {arr.dtype}')
            arrays[name]=arr
        if np.any((arrays['active']!=0)&(arrays['active']!=1)):
            raise ValueError('active must be 0/1')
        for w in range(self.nworld):
            if arrays['active'][w] and 0<=arrays['counts'][w]<=self.pairs:
                used=slice(0,int(arrays['counts'][w]))
                if not np.all(np.isfinite(arrays['prefix'][w,used])):
                    raise ValueError('nonfinite active prefix')
                state=arrays['prefix_state'][w,used]
                if np.any((state[:,1]!=0)&(state[:,1]!=1)) or np.any(
                    ((state[:,1]==0)&(state[:,2]!=-1))|
                    ((state[:,1]==1)&((state[:,2]<0)|(state[:,2]>5)))
                ):
                    raise ValueError('invalid active face state')
        for _,_,_,name,dst in fields:
            arr=arrays[name]
            dst.assign(np.ascontiguousarray(arr).ravel())

    def launch(self, stream=None):
        wp.launch(_cross, dim=self.nworld * self.pairs,
                  inputs=[self.prefix, self.prefix_state, self.active, self.counts,
                          self.axis, self.axis_mask, self.final_f, self.final_state,
                          self.pairs], device=self.device, stream=stream)
