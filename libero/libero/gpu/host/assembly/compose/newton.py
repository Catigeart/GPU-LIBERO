"""Fused Newton entry/reset and parallel multi-field seed/commit.

The solver, warmstart selection, line search and iteration order are untouched.
The public/work separation and all full-capacity tails are retained.
"""
import libero.gpu.precision as scalar
import warp as wp
from libero.gpu.host.runtime.newton import runtime_adapter as original
from libero.gpu.host.runtime.step.newton_final_event import FinalEventAdapter
from .compose import _load, compose, identity_mapping


@wp.kernel(enable_backward=False)
def _begin(publication: wp.array(dtype=wp.int32), publication_error: wp.array(dtype=wp.int32),
           enabled: wp.array(dtype=wp.int32), plan: wp.array(dtype=wp.int64),
           attempted: wp.array(dtype=wp.int64), expected: wp.array(dtype=wp.int64), supplied: int):
    w = wp.tid()
    publication_error[w] = publication[w]
    if enabled[w] != 0:
        attempted[w] = attempted[w] + wp.int64(1)
        if supplied != 0:
            expected[w] = plan[w]
        else:
            expected[w] = attempted[w]


@wp.kernel(enable_backward=False)
def _finish_publication(valid: wp.array(dtype=wp.int32), written: wp.array(dtype=wp.int32),
                        expected: wp.array(dtype=wp.int64), n_written: wp.array(dtype=wp.int64)):
    w = wp.tid()
    written[w] = valid[w]
    if valid[w] != 0:
        n_written[w] = expected[w]


publish_newton = compose("publish_newton", (
    (original._publish, identity_mapping(original._publish)),
    (original._publication_gate, identity_mapping(original._publication_gate, diagnostic="publication_error")),
    (_finish_publication, identity_mapping(_finish_publication)),
))


def _initialization_kernel():
    args = ["enabled: wp.array(dtype=wp.int32)"]
    body = ["w = wp.tid()"]
    for name in original.RESET_INT + original.RESET_FLOAT:
        dtype = "int32" if name in original.RESET_INT else scalar.name
        args.append(f"r_{name}: wp.array(dtype=wp.{dtype})")
        body.append(f"r_{name}[w] = wp.{dtype}(0)")
    args.extend(["error_item: wp.array(dtype=wp.int32)", "stage: wp.array(dtype=wp.int32)",
        "maxiter_arr: wp.array(dtype=wp.int32)", "solver_iter_base: wp.array(dtype=wp.int32)",
        "maxiter: int", "model: wp.array(dtype=scalar.wp_real)",
        "tolerance: wp.array(dtype=scalar.wp_real)", "meaninertia: wp.array(dtype=scalar.wp_real)",
        "upstream: wp.array(dtype=wp.int32)", "nefc: wp.array(dtype=wp.int32)",
        "empty: wp.array(dtype=wp.int32)", "newton_world: wp.array(dtype=wp.int32)",
        "active: wp.array(dtype=wp.int32)", "count: wp.array(dtype=wp.int32)",
        "valid: wp.array(dtype=wp.int32)", "written: wp.array(dtype=wp.int32)"])
    body.extend(["error_item[w] = -1", "stage[w] = 1", "maxiter_arr[w] = maxiter",
        "solver_iter_base[w] = 0", "tolerance[w] = model[0]", "meaninertia[w] = model[1]",
        "empty[w] = 0", "newton_world[w] = 0", "active[w] = 0", "count[w] = nefc[w]",
        "r_err_code[w] = upstream[w]", "valid[w] = 0", "written[w] = 0",
        "if enabled[w] != 0 and upstream[w] == 0:", "    if nefc[w] == 0:",
        "        empty[w] = 1", "    elif nefc[w] > 0:",
        "        newton_world[w] = 1", "        active[w] = 1"])
    source = "@wp.kernel(enable_backward=False)\ndef initialize_newton(" + ",\n    ".join(args) + "):\n"
    source += "\n".join("    " + line for line in body) + "\n"
    kernel = _load("initialize_newton", source, {"wp": wp}).initialize_newton
    kernel.fusion_source = source
    return kernel


def _copy_kernel(commit):
    name = "commit_newton" if commit else "seed_newton"
    args = ["span: int"]
    if commit:
        args += ["valid: wp.array(dtype=wp.int32)"]
    for field in original.PUBLIC_FIELDS:
        dtype = "int32" if field in ("state", "solver_iter", "solver_nnz", "solver_stats_i") else scalar.name
        args += [f"src_{field}: wp.array(dtype=wp.{dtype})",
                 f"dst_{field}: wp.array(dtype=wp.{dtype})", f"stride_{field}: int"]
    body = ["i = wp.tid()", "w = i // span", "cell = i - w * span"]
    if commit:
        body += ["if valid[w] == 0:", "    return"]
    for field in original.PUBLIC_FIELDS:
        body += [f"if cell < stride_{field}:", f"    offset = w * stride_{field} + cell",
                 f"    dst_{field}[offset] = src_{field}[offset]"]
    source = "@wp.kernel(enable_backward=False)\ndef " + name + "(" + ",\n    ".join(args) + "):\n"
    source += "\n".join("    " + line for line in body) + "\n"
    kernel = getattr(_load(name, source, {"wp": wp}), name)
    kernel.fusion_source = source
    return kernel


initialize_newton = _initialization_kernel()
seed_newton = _copy_kernel(False)
commit_newton = _copy_kernel(True)


class FusedNewtonMixin:
    def enqueue(self, upstream_status, active=None, expected_epoch=None, publication_status=None):
        if not self._sealed:
            raise RuntimeError("seal bindings before enqueue")
        w = self.ws
        enabled = self._all_enabled if active is None else active
        publication = self._no_publication_error if publication_status is None else publication_status
        epoch_plan = self.n_attempt if expected_epoch is None else expected_epoch
        for name, array, template in (("upstream_status", upstream_status, w.err_code),
            ("active", enabled, w.active), ("publication_status", publication, self.publication_error),
            ("expected_epoch", epoch_plan, self.n_expected)):
            self._check_array(name, array, template)
        protected = {getattr(w, name).ptr for name in original.RESET_INT}
        if upstream_status.ptr in protected:
            raise ValueError("upstream status must not alias reset-owned scratch")
        if enabled.ptr in protected | {w.active.ptr, w.empty.ptr, w.newton_world.ptr}:
            raise ValueError("active mask must not alias N-owned scratch")
        pairs = [self._commit_aliases[name] for name in original.PUBLIC_FIELDS]
        strides = [src.size // w.nworld for src, _ in pairs]
        span = max(strides)
        seed_args, commit_args = [], []
        for (src, dst), stride in zip(pairs, strides):
            seed_args.extend((dst, src, stride))
            commit_args.extend((src, dst, stride))
        with wp.ScopedStream(self.stream):
            # Preserve epoch/snapshot -> seed -> reset/entry order. In the
            # original API an enabled mask can refer to a work publication.
            wp.launch(_begin, w.nworld, inputs=[publication, self.publication_error,
                enabled, epoch_plan, self.n_attempt, self.n_expected, int(expected_epoch is not None)],
                device=w.device, stream=self.stream)
            wp.launch(seed_newton, w.nworld * span, inputs=[span, *seed_args],
                      device=w.device, stream=self.stream)
            wp.launch(initialize_newton, w.nworld, inputs=[enabled,
                *[getattr(w, name) for name in original.RESET_INT + original.RESET_FLOAT],
                w.error_item, w.stage, w.maxiter_arr, w.solver_iter_base, w.maxiter,
                w.model_scalars, w.tolerance, w.meaninertia, upstream_status, w.nefc,
                w.empty, w.newton_world, w.active, w.count, self.valid, self.written],
                device=w.device, stream=self.stream)
            self.solver.launch_device()
            # Publish the mask before parallel copies. Reading/updating it in
            # the same grid can race with admitted producer-mask aliases.
            values = dict(upstream=upstream_status, enabled=enabled, error=w.err_code,
                valid=self.valid, written=self.written, publication_error=self.publication_error,
                expected=self.n_expected, n_written=self.n_written)
            wp.launch(publish_newton, w.nworld,
                inputs=[values[name] for name in publish_newton.fusion_argument_names],
                device=w.device, stream=self.stream)
            wp.launch(commit_newton, w.nworld * span,
                inputs=[span, self.valid, *commit_args],
                device=w.device, stream=self.stream)
        return self.public_arrays


class FusedNewtonRuntimeAdapter(FusedNewtonMixin, original._NewtonRuntimeAdapter):
    pass


class FusedFinalEventAdapter(FusedNewtonMixin, FinalEventAdapter):
    pass
