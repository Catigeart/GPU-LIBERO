"""Device adapters for original 31-value BoxBox input through accepted tails.

MuJoCo 2.3.7 engine_collision_box.c:596-1332, Apache-2.0; front arithmetic
follows the accepted 2B1 face prefix with per-pair size leaves. N11, G13 and
G14 execute the cross and contact tails on their own preallocated buffers.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": True})


@wp.kernel(enable_backward=False)
def prefix31(pair: wp.array(dtype=scalar.wp_real), active: wp.array(dtype=wp.int32),
             counts: wp.array(dtype=wp.int32), out: wp.array(dtype=scalar.wp_real),
             state: wp.array(dtype=wp.int32), pairs: int):
    t = wp.tid()
    w = t // pairs
    q = t - w*pairs
    b = t*31
    o = t*55
    s = t*5
    for j in range(55):
        out[o+j] = scalar.wp_real(0.0)
    state[s] = 0
    state[s+1] = 0
    state[s+2] = -1
    state[s+3] = -1
    state[s+4] = 0
    if active[w] == 0:
        return
    if active[w] != 1 or counts[w] < 0 or counts[w] > pairs:
        state[s+4] = 1
        return
    if q >= counts[w]:
        return
    for j in range(31):
        if not wp.isfinite(pair[b+j]):
            state[s+4] = 2
            return
    if pair[b] < scalar.wp_real(0.0):
        state[s+4] = 2
        return
    for j in range(3):
        if pair[b+1+j] <= scalar.wp_real(0.0) or pair[b+16+j] <= scalar.wp_real(0.0):
            state[s+4] = 2
            return
    # mju_sub3 then mju_rotVecMatT, with independent reverse subtraction.
    for j in range(3):
        v0 = pair[b+19]-pair[b+4]
        v1 = pair[b+20]-pair[b+5]
        v2 = pair[b+21]-pair[b+6]
        out[o+j] = (pair[b+7+j]*v0 + pair[b+10+j]*v1) + pair[b+13+j]*v2
        u0 = pair[b+4]-pair[b+19]
        u1 = pair[b+5]-pair[b+20]
        u2 = pair[b+6]-pair[b+21]
        out[o+3+j] = (pair[b+22+j]*u0 + pair[b+25+j]*u1) + pair[b+28+j]*u2
    # Accepted 2B1 source-order mju_mulMatTMat: zero, skip zero factors,
    # then sequential scale-and-add for each cell.
    for i in range(3):
        for j in range(3):
            factor = pair[b+7+3*i+j]
            if factor != scalar.wp_real(0.0):
                for k in range(3):
                    z = o+6+3*j+k
                    out[z] = out[z] + pair[b+22+3*i+k]*factor
    for i in range(3):
        for j in range(3):
            v = out[o+6+3*i+j]
            out[o+15+3*j+i] = v
            out[o+24+3*i+j] = wp.abs(v)
            out[o+33+3*j+i] = wp.abs(v)
    for j in range(3):
        out[o+42+j] = (out[o+24+3*j]*pair[b+16] + out[o+25+3*j]*pair[b+17]) + out[o+26+3*j]*pair[b+18]
        out[o+45+j] = (out[o+24+j]*pair[b+1] + out[o+27+j]*pair[b+2]) + out[o+30+j]*pair[b+3]
    penetration = pair[b]
    for i in range(3):
        penetration = penetration + (pair[b+1+i]*scalar.wp_real(3.0) + pair[b+16+i]*scalar.wp_real(3.0))
    code = int(-1)
    for i in range(3):
        c1 = (-wp.abs(out[o+i]) + pair[b+1+i]) + out[o+42+i]
        c2 = (-wp.abs(out[o+3+i]) + pair[b+16+i]) + out[o+45+i]
        out[o+48+i] = c1
        out[o+51+i] = c2
        state[s] = i+1
        if c1 < -pair[b] or c2 < -pair[b]:
            state[s+1] = 1
            state[s+2] = i
            break
        if c1 < penetration:
            penetration = c1
            sign = int(0)
            if out[o+i] < scalar.wp_real(0.0):
                sign = 1
            code = i+3*sign
        if c2 < penetration:
            penetration = c2
            sign = int(0)
            if out[o+3+i] < scalar.wp_real(0.0):
                sign = 1
            code = i+3*sign+6
    out[o+54] = penetration
    state[s+3] = code


@wp.kernel(enable_backward=False)
def map_cross(pair: wp.array(dtype=scalar.wp_real), face: wp.array(dtype=scalar.wp_real),
              face_state: wp.array(dtype=wp.int32), prefix: wp.array(dtype=scalar.wp_real),
              prefix_state: wp.array(dtype=wp.int32)):
    t = wp.tid()
    b = t*31
    f = t*55
    p = t*38
    s = t*5
    for j in range(38):
        prefix[p+j] = scalar.wp_real(0.0)
    prefix_state[t*3] = face_state[s+3]
    prefix_state[t*3+1] = face_state[s+1]
    prefix_state[t*3+2] = face_state[s+2]
    if face_state[s+4] != 0:
        prefix_state[t*3+1] = 1
        return
    if face_state[s+1] != 0:
        return
    for j in range(27):
        prefix[p+j] = face[f+6+j]
    for j in range(3):
        prefix[p+27+j] = face[f+j]
        prefix[p+30+j] = pair[b+1+j]
        prefix[p+33+j] = pair[b+16+j]
    prefix[p+36] = pair[b]
    prefix[p+37] = face[f+54]


@wp.kernel(enable_backward=False)
def map_sat(face: wp.array(dtype=scalar.wp_real), face_state: wp.array(dtype=wp.int32),
            cross_f: wp.array(dtype=scalar.wp_real), cross_state: wp.array(dtype=wp.int32),
            active: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
            sat_f: wp.array(dtype=scalar.wp_real), sat_i: wp.array(dtype=wp.int32),
            pairs: int):
    t=wp.tid()
    w=t//pairs
    q=t-w*pairs
    si=t*6
    sf=t*45
    for j in range(45):
        sat_f[sf+j]=scalar.wp_real(0.0)
    for j in range(4):
        sat_i[si+j]=-1
    sat_i[si+4]=0
    sat_i[si+5]=0
    if active[w]==0 or q>=counts[w] and counts[w]>=0 and counts[w]<=pairs:
        return
    if face_state[t*5+4]!=0:
        sat_i[si+5]=face_state[t*5+4]
        return
    if cross_state[t*8+5]!=0:
        sat_i[si+5]=3
        return
    if face_state[t*5+1]!=0 or cross_state[t*8+4]>=0 or cross_state[t*8]<0:
        return
    code=cross_state[t*8]
    sat_i[si]=code
    sat_i[si+1]=cross_state[t*8+1]
    sat_i[si+2]=cross_state[t*8+2]
    sat_i[si+3]=cross_state[t*8+3]
    sat_i[si+4]=1
    if code>=12:
        sat_i[si+4]=2
    for j in range(42):
        sat_f[sf+j]=face[t*55+j]
    if code>=12:
        for j in range(3):
            sat_f[sf+42+j]=cross_f[t*4+1+j]


@wp.kernel(enable_backward=False)
def merge_raw(sat_i: wp.array(dtype=wp.int32), face_state: wp.array(dtype=wp.int32),
              cross_state: wp.array(dtype=wp.int32), active: wp.array(dtype=wp.int32),
              counts: wp.array(dtype=wp.int32), face_count: wp.array(dtype=wp.int32),
              face_raw: wp.array(dtype=scalar.wp_real), face_status: wp.array(dtype=wp.int32),
              face_written: wp.array(dtype=wp.int32), face_frame: wp.array(dtype=scalar.wp_real),
              edge_count: wp.array(dtype=wp.int32), edge_raw: wp.array(dtype=scalar.wp_real),
              edge_status: wp.array(dtype=wp.int32), edge_written: wp.array(dtype=wp.int32),
              edge_frame: wp.array(dtype=scalar.wp_real), out_count: wp.array(dtype=wp.int32),
              out_raw: wp.array(dtype=scalar.wp_real), write_mask: wp.array(dtype=wp.int32),
              out_state: wp.array(dtype=wp.int32), frame_written: wp.array(dtype=wp.int32),
              frame0: wp.array(dtype=scalar.wp_real), pairs: int):
    t=wp.tid()
    w=t//pairs
    q=t-w*pairs
    out_count[t]=0
    frame_written[t]=0
    out_state[t*4]=0
    out_state[t*4+1]=-1
    out_state[t*4+2]=0
    out_state[t*4+3]=0
    for j in range(500):
        write_mask[t*500+j]=0
    for j in range(6):
        frame0[t*6+j]=scalar.wp_real(0.0)
    if active[w]==0:
        return
    if active[w]!=1 or counts[w]<0 or counts[w]>pairs:
        out_state[t*4]=1
        return
    if q>=counts[w]:
        return
    if face_state[t*5+4]!=0:
        out_state[t*4]=face_state[t*5+4]
        return
    if cross_state[t*8+5]!=0:
        out_state[t*4]=3
        return
    if sat_i[t*6+5]!=0:
        out_state[t*4]=4
        return
    route=sat_i[t*6+4]
    out_state[t*4+1]=route
    if route==0:
        return
    n=0
    fw=0
    if route==1:
        if face_status[t*4+2]!=0:
            out_state[t*4]=5
            return
        out_state[t*4+2]=face_status[t*4]
        n=face_count[t]
        fw=face_written[t]
    elif route==2:
        if edge_status[t]!=2 and edge_status[t]!=3:
            out_state[t*4]=6
            return
        out_state[t*4+3]=1
        n=edge_count[t]
        fw=edge_written[t]
    else:
        out_state[t*4]=4
        return
    if n<0 or n>50:
        out_state[t*4]=7
        return
    out_count[t]=n
    frame_written[t]=fw
    for j in range(n*10):
        value=scalar.wp_real(0.0)
        if route==1:
            value=face_raw[t*500+j]
        else:
            value=edge_raw[t*500+j]
        out_raw[t*500+j]=value
        write_mask[t*500+j]=1
    if fw!=0:
        for j in range(6):
            value=scalar.wp_real(0.0)
            if route==1:
                value=face_frame[t*6+j]
            else:
                value=edge_frame[t*6+j]
            frame0[t*6+j]=value
            out_raw[t*500+4+j]=value
            write_mask[t*500+4+j]=1
