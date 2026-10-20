"""Device forward/state epochs, complete read gates and first-cause status.

State writes and forward calls are different events. All counters are int64,
fail closed at the maximum, and only explicit selected reset clears a cause.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
import warp as wp
wp.set_module_options({"enable_backward": False, "fuse_fp": True, "fast_math": False})

@wp.kernel(enable_backward=False)
def bump_state(state: wp.array(dtype=wp.int64), first_stage: wp.array(dtype=wp.int32),
               first_code: wp.array(dtype=wp.int32)):
    w=wp.tid()
    if state[w] < wp.int64(9223372036854775807) and state[w] >= wp.int64(0):
        state[w]=state[w]+wp.int64(1)
    elif first_code[w] == 0:
        first_stage[w]=7
        first_code[w]=9

@wp.kernel(enable_backward=False)
def begin_forward(state: wp.array(dtype=wp.int64), bound_state: wp.array(dtype=wp.int64),
                  forward: wp.array(dtype=wp.int64), first_stage: wp.array(dtype=wp.int32),
                  first_code: wp.array(dtype=wp.int32)):
    w=wp.tid()
    if forward[w] < wp.int64(9223372036854775807) and forward[w] >= wp.int64(0):
        forward[w]=forward[w]+wp.int64(1)
        bound_state[w]=state[w]
    elif first_code[w] == 0:
        first_stage[w]=7
        first_code[w]=9

@wp.kernel(enable_backward=False)
def start_stage(active: wp.array(dtype=wp.int32), state: wp.array(dtype=wp.int64),
                bound_state: wp.array(dtype=wp.int64), forward: wp.array(dtype=wp.int64),
                first_stage: wp.array(dtype=wp.int32), first_code: wp.array(dtype=wp.int32),
                gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32), version: wp.array(dtype=wp.int64), stage: int):
    w=wp.tid()
    gate[w]=0
    status[w]=0
    if active[w] == 0:
        return
    if first_code[w] != 0:
        status[w]=first_code[w]
        return
    if active[w] != 1:
        status[w]=31
    elif forward[w] <= wp.int64(0) or state[w] != bound_state[w]:
        status[w]=7
    elif version[w] >= wp.int64(9223372036854775807) or version[w] < wp.int64(0):
        status[w]=9
    else:
        gate[w]=1
    if status[w] != 0:
        first_stage[w]=stage
        first_code[w]=status[w]

@wp.kernel(enable_backward=False)
def require_read(gate: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
                 stamp: wp.array(dtype=wp.int64), forward: wp.array(dtype=wp.int64),
                 first_stage: wp.array(dtype=wp.int32), first_code: wp.array(dtype=wp.int32),
                 status: wp.array(dtype=wp.int32), stage: int):
    w=wp.tid()
    if gate[w] == 1 and (valid[w] != 1 or stamp[w] != forward[w]):
        gate[w]=0
        status[w]=7
        if first_code[w] == 0:
            first_stage[w]=stage
            first_code[w]=7

@wp.kernel(enable_backward=False)
def finish_stage(active: wp.array(dtype=wp.int32), gate: wp.array(dtype=wp.int32),
                 status: wp.array(dtype=wp.int32), forward: wp.array(dtype=wp.int64),
                 stamp: wp.array(dtype=wp.int64), version: wp.array(dtype=wp.int64),
                 valid: wp.array(dtype=wp.int32), first_stage: wp.array(dtype=wp.int32),
                 first_code: wp.array(dtype=wp.int32), stage: int):
    w=wp.tid()
    valid[w]=0
    if gate[w] != 1:
        status[w]=0
        if active[w] != 0 and first_code[w] != 0:
            status[w]=first_code[w]
        return
    if status[w] == 0 or status[w] == 25:
        if version[w] < wp.int64(9223372036854775807) and version[w] >= wp.int64(0):
            version[w]=version[w]+wp.int64(1)
            stamp[w]=forward[w]
            valid[w]=1
        else:
            status[w]=9
    if valid[w] == 0 and first_code[w] == 0:
        first_stage[w]=stage
        first_code[w]=status[w]

@wp.kernel(enable_backward=False)
def clear_valid(valid: wp.array(dtype=wp.int32)):
    valid[wp.tid()]=0

@wp.kernel(enable_backward=False)
def reset_errors(mask: wp.array(dtype=wp.int32), first_stage: wp.array(dtype=wp.int32),
                 first_code: wp.array(dtype=wp.int32), control_error: wp.array(dtype=wp.int32)):
    w=wp.tid()
    if mask[w] == 1:
        first_stage[w]=0
        first_code[w]=0
        control_error[w]=0

@wp.kernel(enable_backward=False)
def restore_failed(gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                   previous: wp.array(dtype=wp.uint8, ndim=2),
                   current: wp.array(dtype=wp.uint8, ndim=2)):
    w,b=wp.tid()
    if gate[w] != 1 or (status[w] != 0 and status[w] != 25):
        current[w,b]=previous[w,b]

@wp.kernel(enable_backward=False)
def record_clock(gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
                 refresh: wp.array(dtype=wp.int32), policy: wp.array(dtype=wp.int32),
                 forward: wp.array(dtype=wp.int64), bias_version: wp.array(dtype=wp.int64),
                 counters: wp.array(dtype=wp.int64, ndim=2), cache_forward: wp.array(dtype=wp.int64),
                 bias_read: wp.array(dtype=wp.int64)):
    w=wp.tid()
    if gate[w] != 1 or status[w] != 0:
        return
    # refresh, skip, goal writes, ctrl writes, live bias reads, cache version.
    if refresh[w] == 1:
        counters[w,0]=counters[w,0]+wp.int64(1)
        counters[w,5]=counters[w,5]+wp.int64(1)
        cache_forward[w]=forward[w]
    else:
        counters[w,1]=counters[w,1]+wp.int64(1)
    if policy[w] == 1:
        counters[w,2]=counters[w,2]+wp.int64(1)
    counters[w,3]=counters[w,3]+wp.int64(1)
    counters[w,4]=counters[w,4]+wp.int64(1)
    counters[w,6]=counters[w,6]+wp.int64(1)
    bias_read[w]=bias_version[w]

@wp.kernel(enable_backward=False)
def clear_selected_valid(mask: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32)):
    w=wp.tid()
    if mask[w] == 1:
        valid[w]=0

@wp.kernel(enable_backward=False)
def bump_input(versions: wp.array(dtype=wp.int64, ndim=2),
               first_stage: wp.array(dtype=wp.int32), first_code: wp.array(dtype=wp.int32), slot: int):
    w=wp.tid()
    if versions[w,slot] >= wp.int64(0) and versions[w,slot] < wp.int64(9223372036854775807):
        versions[w,slot]=versions[w,slot]+wp.int64(1)
    elif first_code[w] == 0:
        first_stage[w]=7
        first_code[w]=9

@wp.kernel(enable_backward=False)
def clock_gate(gate: wp.array(dtype=wp.int32), status: wp.array(dtype=wp.int32),
               counters: wp.array(dtype=wp.int64, ndim=2), first_stage: wp.array(dtype=wp.int32),
               first_code: wp.array(dtype=wp.int32), refresh: int, policy: int,
               adapter_version: wp.array(dtype=wp.int64)):
    w=wp.tid()
    if gate[w] != 1:
        return
    ok=int(1)
    if adapter_version[w] < wp.int64(0) or adapter_version[w] >= wp.int64(9223372036854775807):
        ok=0
    for i in range(7):
        needed=int(1)
        if (i == 0 or i == 5) and refresh == 0:
            needed=0
        if i == 1 and refresh == 1:
            needed=0
        if i == 2 and policy == 0:
            needed=0
        if counters[w,i] < wp.int64(0) or (needed == 1 and counters[w,i] >= wp.int64(9223372036854775807)):
            ok=0
    if ok == 0:
        gate[w]=0
        status[w]=9
        if first_code[w] == 0:
            first_stage[w]=4
            first_code[w]=9

@wp.kernel(enable_backward=False)
def written_finite(active: wp.array(dtype=wp.int32), error: wp.array(dtype=wp.int32),
                   values: wp.array(dtype=scalar.wp_real, ndim=2)):
    w=wp.tid()
    if active[w] != 1 or error[w] != 0:
        return
    for i in range(values.shape[1]):
        if not wp.isfinite(values[w,i]):
            error[w]=8
            return
