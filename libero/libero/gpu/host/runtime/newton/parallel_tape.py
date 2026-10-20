"""Independent performance candidate: parallel byte-preserving LS witness copy.

No arithmetic, active gates, schedule length, source helper or publication
behavior changes. One writer per witness cell; reads occur at the retained
solver's exact stream position, before any subsequent source mutation.
"""
import libero.gpu.precision as scalar
import sys
import warp as wp


@wp.kernel(enable_backward=False)
def _parallel_tape_ls_buffers(slot: int, tape_slots: int, rows: int,
    Mv: wp.array(dtype=scalar.wp_real, ndim=2),
    Jv: wp.array(dtype=scalar.wp_real, ndim=2),
    quad: wp.array(dtype=scalar.wp_real, ndim=3),
    quadGauss: wp.array(dtype=scalar.wp_real, ndim=2),
    tape_Mv: wp.array(dtype=scalar.wp_real, ndim=3),
    tape_Jv: wp.array(dtype=scalar.wp_real, ndim=3),
    tape_quad: wp.array(dtype=scalar.wp_real, ndim=4),
    tape_quadGauss: wp.array(dtype=scalar.wp_real, ndim=3), nv: int):
    if slot >= tape_slots:
        return
    stride = nv + 4 * rows + 3
    t = wp.tid()
    w = t // stride
    i = t - w * stride
    if i < nv:
        tape_Mv[w, slot, i] = Mv[w, i]
    elif i < nv + rows:
        tape_Jv[w, slot, i - nv] = Jv[w, i - nv]
    elif i < nv + 4 * rows:
        j = i - nv - rows
        tape_quad[w, slot, j // 3, j % 3] = quad[w, j // 3, j % 3]
    else:
        j = i - nv - 4 * rows
        tape_quadGauss[w, slot, j] = quadGauss[w, j]


def parallel_tape_solver_type(source_solver_type):
    """Subclass only the LS witness launcher; retained source loop untouched."""
    source_module = sys.modules[source_solver_type.__module__]
    original = source_module.K._tape_ls_buffers
    nv = int(source_module.NV)
    class ParallelTapeSolver(source_solver_type):
        def _glue(self, kernel, inputs, dim=None):
            if kernel is original:
                ws = self.ws
                wp.launch(_parallel_tape_ls_buffers,
                          ws.nworld * (nv + 4 * ws.rows + 3),
                          inputs=[*inputs, nv], device=ws.device, stream=ws.stream)
            else:
                super()._glue(kernel, inputs, dim)
    return ParallelTapeSolver
