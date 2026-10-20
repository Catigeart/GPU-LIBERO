"""Validation-only device control recording and raw-upstream replay."""
import warp as wp

wp.set_module_options({'enable_backward': False})


@wp.kernel
def record_control(counter: wp.array(dtype=wp.int64), start: wp.array(dtype=wp.int64),
                   control: wp.array2d(dtype=float), trace: wp.array3d(dtype=float)):
    world, actuator = wp.tid()
    trace[int(counter[world] - start[world]), world, actuator] = control[world, actuator]


@wp.kernel
def replay_control(cursor: wp.array(dtype=wp.int64), trace: wp.array3d(dtype=float),
                   control: wp.array2d(dtype=float)):
    world, actuator = wp.tid()
    control[world, actuator] = trace[int(cursor[world]), world, actuator]


@wp.kernel
def advance_cursor(cursor: wp.array(dtype=wp.int64)):
    world = wp.tid()
    cursor[world] += wp.int64(1)
