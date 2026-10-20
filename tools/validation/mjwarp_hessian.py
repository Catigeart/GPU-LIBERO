"""Same-input local correctness/performance for the handwritten FP32 Hessian.

Preparation only loads/compiles upstream baseline kernels offline. The run phase
is guarded load-only, has no environment construction or physics integration,
and compares both implementations with a separately constructed FP64 matrix.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as Namespace

import numpy as np


def add_arguments(parser):
    parser.add_argument('--input', type=Path, required=True, help='saved Newton boundary .npz')
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True, help='prebuilt mjwarp_hessian_fp32.so')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true', help='offline baseline kernel compilation only')
    parser.add_argument('--performance', action='store_true', help='time the real-input leaf after all checks pass')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--calls-per-sample', type=int, default=64)
    return parser


def real_case(path):
    """Derive an assembly input without pretending this was an H-stage snapshot.

    The old snapshot is before/after Newton, not inside its iterations. Use its
    final constraint states and reconstruct Jaref from final acceleration. Freeze
    that derived Jaref identically for both assembly implementations.
    """
    with np.load(path, allow_pickle=False) as saved:
        nv = saved['output.qacc'].shape[1]
        a = {name: saved[key].copy() for name, key in dict(
            nefc='input.nefc', mass='input.M', jacobian='input.efc.J',
            diagonal='input.efc.D', state='output.efc.state',
            mass_row='model.M_hinit_i', mass_column='model.M_colind',
            impratio='model.impratio_invsqrt', contact_count='input.nacon',
            contact_world='input.contact.worldid', contact_dimension='input.contact.dim',
            contact_friction='input.contact.friction', contact_distance='input.contact.dist',
            contact_margin='input.contact.includemargin', contact_address='input.contact.efc_address').items()}
        a['jaref'] = (np.einsum('wri,wi->wr', a['jacobian'][:, :, :nv].astype(np.float64),
            saved['output.qacc'].astype(np.float64)) - saved['input.efc.aref']).astype(np.float32)
    a['done'] = np.zeros(a['nefc'].size, np.bool_)
    return dict(name='task4-derived-final-boundary', nv=nv, arrays=a,
        origin='saved final efc.state; FP64 J @ final qacc - aref rounded once to FP32')


def synthetic_case(nv, *, variant='ordinary'):
    rng = np.random.default_rng(7000 + nv)
    n, rows, capacity = 2, 48, 12
    pad = ((nv + 3) // 4) * 4 if nv <= 32 else ((nv + 15) // 16) * 16
    j = rng.normal(0., .3, (n, rows, pad)).astype(np.float32)
    j[:, :, nv:] = 0.
    a = dict(nefc=np.array([29, 45], np.int32), mass=np.exp(rng.uniform(-2., 2., (n, nv))).astype(np.float32),
        mass_row=np.arange(nv, dtype=np.int32), mass_column=np.arange(nv, dtype=np.int32),
        jacobian=j, diagonal=rng.uniform(.1, 4., (n, rows)).astype(np.float32),
        state=rng.integers(0, 4, (n, rows), dtype=np.int32),
        jaref=rng.normal(0., 1., (n, rows)).astype(np.float32),
        done=np.zeros(n, np.bool_), impratio=np.array([.2, .3], np.float32),
        contact_count=np.array([capacity], np.int32), contact_world=np.zeros(capacity, np.int32),
        contact_dimension=np.zeros(capacity, np.int32),
        contact_friction=rng.uniform(.2, .9, (capacity, 5)).astype(np.float32),
        contact_distance=np.full(capacity, -.01, np.float32),
        contact_margin=np.zeros(capacity, np.float32),
        contact_address=np.full((capacity, 6), -1, np.int32))
    for w in range(n):
        head = 0
        # Real condim 3/4/6 paths, plus inactive, absent, and frictionless guards.
        for k, dim in enumerate((3, 4, 6, 3, 3, 1)):
            c = 2 * k + w  # pooled contacts deliberately interleave worlds
            a['contact_world'][c], a['contact_dimension'][c] = w, dim
            if k == 4:
                continue
            a['contact_address'][c, :dim] = np.arange(head, head + dim)
            a['state'][w, head:head + dim] = 4 if dim > 1 else 1
            a['jaref'][w, head] = 0.  # interior of the middle cone zone
            if k == 3:
                a['contact_distance'][c] = .01
                a['state'][w, head:head + dim] = 0
            head += dim
        # Unused allocated rows are finite poison, not extra constraints.
        used = a['nefc'][w]
        a['jacobian'][w, used:, :nv] = 100.
        a['diagonal'][w, used:] = 100.
        a['state'][w, used:] = 1
    if variant == 'done':
        a['done'][1] = True
    elif variant == 'empty':
        a['nefc'][:] = 0
        a['contact_count'][:] = 0
    elif variant == 'all-done':
        a['done'][:] = True
    elif variant == 'zero-friction':
        a['contact_friction'][:] = 0.
    elif variant == 'clamped-count':
        a['contact_count'][:] = capacity + 7  # baseline's min(count, capacity)
    elif variant == 'ordinary':
        pass
    else:
        raise ValueError(variant)
    return dict(name=f'synthetic-nv{nv}-{variant}', nv=nv, arrays=a,
                origin='deterministic manufactured local inputs, not a task trajectory')


def cases(path):
    real = real_case(path)
    families = (21, 22, 24, 27, 30, 33, 36, 37, 39, 43, 45, 51, 57)
    result = [real] + [synthetic_case(nv) for nv in families]
    result += [synthetic_case(39, variant=v) for v in
               ('done', 'empty', 'all-done', 'zero-friction', 'clamped-count')]
    return result


def matrix_reference(case):
    """Construct full local cone C, then Jc.T @ C @ Jc (not the GPU projection).

    CPU MuJoCo 2.3.7 HessianDirect uses only QUADRATIC rows, HessianCone adds
    local contact Hessians. Pinned MJWarp stores mass/cone only in the upper
    triangle, and the quadratic term in both triangles. Reproduce that storage.
    """
    a, nv = case['arrays'], case['nv']
    n, rows, pad = a['jacobian'].shape
    result = np.zeros((n, pad, pad), np.float64)
    magnitude = np.zeros_like(result)
    cone_counts = np.zeros(n, np.int32)
    for w in range(n):
        for e, (r, c) in enumerate(zip(a['mass_row'], a['mass_column'])):
            result[w, c, r] += a['mass'][w, e]
            magnitude[w, c, r] += abs(float(a['mass'][w, e]))
        used = a['nefc'][w]
        j = a['jacobian'][w, :used].astype(np.float64)
        d = np.where(a['state'][w, :used] == 1, a['diagonal'][w, :used], 0.).astype(np.float64)
        result[w] += j.T @ (d[:, None] * j)
        magnitude[w] += abs(j).T @ (abs(d[:, None]) * abs(j))
    for c in range(min(int(a['contact_count'][0]), a['contact_world'].size)):
        w, dim = int(a['contact_world'][c]), int(a['contact_dimension'][c])
        ids = a['contact_address'][c, :dim]
        if dim == 1 or a['contact_distance'][c] - a['contact_margin'][c] >= 0 or ids[0] < 0:
            continue
        if a['state'][w, ids[0]] != 4:
            continue
        mu = float(a['contact_friction'][c, 0]) * float(a['impratio'][w % a['impratio'].size])
        denominator = mu * mu * (1. + mu * mu)
        dm = float(a['diagonal'][w, ids[0]]) / (denominator if denominator else 1.e-15)
        if dm == 0:
            continue
        scale = np.r_[mu, a['contact_friction'][c, :dim - 1].astype(np.float64)]
        x, jac = np.zeros(dim, np.float64), np.zeros((dim, nv), np.float64)
        present = ids >= 0
        x[present] = a['jaref'][w, ids[present]]
        jac[present] = a['jacobian'][w, ids[present], :nv]
        tangent = x[1:] * scale[1:]
        length = max(float(np.linalg.norm(tangent)), 1.e-15)
        unit = tangent / length
        v = np.r_[1., -mu * unit]
        beta = mu * mu - mu * (x[0] * mu) / length
        local = np.outer(v, v)
        local[1:, 1:] += beta * (np.eye(dim - 1) - np.outer(unit, unit))
        local *= dm * np.outer(scale, scale)
        contribution = jac.T @ local @ jac
        result[w, :nv, :nv] += np.triu(contribution)
        # Absolute arithmetic majorant retains cancellation inside local C.
        local_abs = abs(np.outer(v, v))
        local_abs[1:, 1:] += abs(beta) * (np.eye(dim - 1) + abs(np.outer(unit, unit)))
        local_abs *= abs(dm) * abs(np.outer(scale, scale))
        magnitude[w, :nv, :nv] += np.triu(abs(jac).T @ local_abs @ abs(jac))
        cone_counts[w] += 1
    if nv > 32:
        for k in range(nv, pad):
            result[:, k, k] = magnitude[:, k, k] = 1.
    return result, magnitude, cone_counts


class Replay:
    def __init__(self, case, library, stream):
        import warp as wp
        from mujoco_warp._src import solver, types
        from libero.gpu.host.runtime.newton.mjwarp_hessian import DenseHessian
        self.case, self.stream = case, stream
        a = case['arrays']
        dev = {k: wp.array(np.ascontiguousarray(v), dtype=types.vec5 if k == 'contact_friction' else
                         wp.bool if v.dtype == np.bool_ else
                         wp.int32 if v.dtype == np.int32 else wp.float32, device=stream.device)
               for k, v in a.items()}
        nv, (n, rows, pad) = case['nv'], a['jacobian'].shape
        r, c = np.triu_indices(nv)
        model = Namespace(nv=nv, nv_pad=pad, is_sparse=False,
            M_hinit_i=dev['mass_row'], M_colind=dev['mass_column'],
            dof_tri_row=wp.array(r.astype(np.int32), dtype=int, device=stream.device),
            dof_tri_col=wp.array(c.astype(np.int32), dtype=int, device=stream.device),
            opt=Namespace(impratio_invsqrt=dev['impratio'], cone=types.ConeType.ELLIPTIC,
                solver=types.SolverType.NEWTON, integrator=types.IntegratorType.EULER, enableflags=0))
        data = Namespace(nworld=n, njmax=rows, naconmax=a['contact_world'].size,
            nefc=dev['nefc'], M=dev['mass'], nacon=dev['contact_count'],
            efc=Namespace(J=dev['jacobian'], D=dev['diagonal'], state=dev['state']),
            contact=Namespace(worldid=dev['contact_world'], dim=dev['contact_dimension'],
                friction=dev['contact_friction'], dist=dev['contact_distance'],
                includemargin=dev['contact_margin'], efc_address=dev['contact_address']))
        self.initial = (np.arange(n * pad * pad, dtype=np.float32).reshape(n, pad, pad) + 8192.)
        self.old = wp.array(self.initial, dtype=float, device=stream.device)
        self.new = wp.array(self.initial, dtype=float, device=stream.device)
        self.model, self.data, self.dev = model, data, dev
        self.ctx = Namespace(Jaref=dev['jaref'], done=dev['done'], h=self.new)
        self.native = DenseHessian(model, data, self.ctx, library, stream)
        self.quad = solver._update_gradient_JTDAJ_dense_tiled(pad, 16, rows, a['mass_row'].size)
        self.compare_upstream = case['name'].startswith('task4-') or (
            case['nv'] in (21, 57) and case['name'].endswith('-ordinary'))

    def baseline(self, stage=0):
        import warp as wp
        from mujoco_warp._src import solver
        m, d, a = self.model, self.data, self.native.arguments
        if stage != 2:
            wp.launch_tiled(self.quad, dim=d.nworld,
                inputs=[m.M_colind, m.M_hinit_i, d.nefc, d.M, d.efc.J, d.efc.D, d.efc.state, self.ctx.done],
                outputs=[self.old], block_dim=128, stream=self.stream)
            if m.nv > 32:
                wp.launch(solver._padding_h, dim=(d.nworld, m.nv_pad - m.nv),
                    inputs=[m.nv, self.ctx.done], outputs=[self.old], stream=self.stream)
        if stage != 1:
            wp.launch(solver._update_gradient_JTCJ_dense,
                dim=(a.cone_launch_groups, m.dof_tri_row.size),
                inputs=[m.opt.impratio_invsqrt, m.dof_tri_row, m.dof_tri_col,
                    d.contact.dist, d.contact.includemargin, d.contact.friction,
                    d.contact.dim, d.contact.efc_address, d.contact.worldid,
                    d.efc.J, d.efc.D, d.efc.state, d.naconmax, d.nacon,
                    self.ctx.Jaref, self.ctx.done, a.cone_contacts_per_group, a.cone_launch_groups],
                outputs=[self.old], stream=self.stream)

    def validate(self):
        import warp as wp
        self.native.enqueue()
        if self.compare_upstream:
            self.baseline()
        wp.synchronize_stream(self.stream)
        ref, magnitude, cones = matrix_reference(self.case)
        done = self.case['arrays']['done']
        ref[done] = self.initial[done]
        eps = np.finfo(np.float32).eps
        # Fixed forward-error bound: 16-row dot + tile sum + cone arithmetic
        # and atomic accumulation. Scale comes only from frozen inputs/FP64.
        operations = 96 + (self.case['arrays']['nefc'] + 15) // 16 + cones
        gamma = operations * eps / (1. - operations * eps)
        bound = 8 * gamma[:, None, None] * np.maximum(magnitude, 1.e-20)
        bound[done] = 0.
        checks = {}
        results = {'handwritten': self.new.numpy()}
        if self.compare_upstream:
            results['mjwarp'] = self.old.numpy()
        for name, value in results.items():
            error = abs(value.astype(np.float64) - ref)
            passed = np.isfinite(value).all() and np.all(error <= bound)
            checks[name] = dict(passed=bool(passed), max_absolute_error=float(error.max()),
                max_bound_fraction=float(np.max(np.divide(error, bound,
                    out=np.zeros_like(error), where=bound > 0))),
                done_worlds_exact=bool(np.array_equal(value[done], self.initial[done])))
        difference = float(abs(results['handwritten'] - results['mjwarp']).max()) if self.compare_upstream else None
        return dict(name=self.case['name'], nv=self.case['nv'], worlds=2,
            padded_nv=ref.shape[1], rows=self.case['arrays']['nefc'].tolist(),
            cone_contacts=cones.tolist(), origin=self.case['origin'], checks=checks,
            handwritten_vs_mjwarp_max_absolute=difference,
            passed=all(c['passed'] and c['done_worlds_exact'] for c in checks.values()))


def benchmark(replay, args):
    import warp as wp
    result = {}
    # Separate graphs preserve actual kernel launch/stream order. No snapshot,
    # allocation, copies, or event instrumentation inside the assembly itself.
    for name, launch in [('mjwarp', replay.baseline), ('handwritten', replay.native.enqueue)]:
        result[name] = {}
        for phase, stage in [('full', 0), ('mass_quadratic_padding', 1), ('cone', 2)]:
            with wp.ScopedCapture(stream=replay.stream) as capture:
                for _ in range(args.calls_per_sample):
                    launch(stage)
            for _ in range(args.warmup):
                wp.capture_launch(capture.graph, stream=replay.stream)
            wp.synchronize_stream(replay.stream)
            start = wp.Event(device=replay.stream.device, enable_timing=True)
            end = wp.Event(device=replay.stream.device, enable_timing=True)
            values = []
            for _ in range(args.steps):
                wp.record_event(start)
                wp.capture_launch(capture.graph, stream=replay.stream)
                wp.record_event(end)
                wp.synchronize_event(end)
                values.append(wp.get_event_elapsed_time(start, end) * 1000 / args.calls_per_sample)
            result[name][phase] = dict(microseconds=values, mean_us=float(np.mean(values)),
                std_us=float(np.std(values)), p95_us=float(np.percentile(values, 95)))
    result['full_speedup'] = result['mjwarp']['full']['mean_us'] / result['handwritten']['full']['mean_us']
    result['scope'] = 'one local Hessian assembly across both worlds; not a control step or Newton solve'
    result['calls_per_sample'] = args.calls_per_sample
    return result


def main(argv=None, *, args=None):
    if args is None:
        args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.warmup < 0 or args.steps < 1 or args.calls_per_sample < 1:
        raise ValueError('invalid local measurement counts')
    import libero.gpu.precision as precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from mujoco_warp._src import solver
    from libero.gpu.host.build.compilation import compilation_scope, install_warp_guard
    from libero.gpu.host.runtime.newton.mjwarp_hessian import source_id
    from libero.gpu.host.paths import GPU_ROOT
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    args.output.mkdir(parents=True, exist_ok=True)
    init_kernel_cache(args.build_root / 'mjwarp' / 'warp-cache')
    install_warp_guard()
    wp.init()
    stream = wp.Stream('cuda:0')
    all_cases = cases(args.input)
    signature = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binding_sha256=hashlib.sha256((GPU_ROOT / 'host/runtime/newton/mjwarp_hessian.py').read_bytes()).hexdigest(),
        solver_sha256=hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest(),
        warp_version=wp.__version__, source_id=source_id(), arch=stream.device.arch)
    receipt = args.output / 'preparation.json'
    if not args.prepare and (not receipt.is_file() or json.loads(receipt.read_text())['signature'] != signature):
        raise RuntimeError('run the matching --prepare phase offline before local validation')
    load_before = gpu_load_snapshot()
    with compilation_scope(args.prepare) as scope, wp.ScopedStream(stream):
        replays = [Replay(case, args.library, stream) for case in all_cases]
        if args.prepare:
            # Baseline factory specializations for real NV39 and endpoint NV21/57.
            for replay in replays:
                if replay.compare_upstream:
                    wp.load_module(module=replay.quad.module, device=stream.device, block_dim=128)
            wp.load_module(module=solver._padding_h.module, device=stream.device)
            wp.synchronize_stream(stream)
            modules = [dict(name=replay.quad.module.name,
                hash=replay.quad.module.get_module_hash(128).hex(), block_dim=128)
                for replay in replays if replay.compare_upstream]
            modules.append(dict(name=solver._padding_h.module.name,
                hash=solver._padding_h.module.get_module_hash(256).hex(), block_dim=256))
            result = dict(signature=signature, modules=modules, compilation_events=scope['events'],
                physics_integrations=0, passed=True)
            save_json(receipt, result)
            print(json.dumps(dict(prepared=True, compilation_events=len(scope['events']))), flush=True)
            return 0
        checks = []
        for replay in replays:
            row = replay.validate()
            checks.append(row)
            print(json.dumps(row), flush=True)
        passed = all(r['passed'] for r in checks)
        # Preserve the exact derived real input used by both leaf implementations.
        np.savez(args.output / 'assembly-input.npz', nv=np.array(all_cases[0]['nv']),
                 **all_cases[0]['arrays'])
        save_json(args.output / 'correctness.json', dict(signature=signature, passed=passed,
            cases=checks, runtime_compilations=scope['events'], physics_integrations=0))
        timing = benchmark(replays[0], args) if passed and args.performance else None
        result = dict(signature=signature, passed=passed, cases=checks, performance=timing,
            runtime_compilations=scope['events'], physics_integrations=0,
            gpu_load_before=load_before, gpu_load_after=gpu_load_snapshot(),
            production_dispatch_changed=False, library=str(args.library.resolve()),
            library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
        save_json(args.output / 'summary.json', result)
        print(json.dumps(dict(passed=passed, cases=len(checks), performance=timing,
                              runtime_compilations=len(scope['events']))), flush=True)
        return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
