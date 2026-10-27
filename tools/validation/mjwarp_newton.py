"""Frozen-input Newton replay, with explicit offline preparation and no physics.

The baseline is the pinned upstream _solve body; the staged-upstream control
isolates host orchestration, and cuda_hessian changes only matrix assembly.
All graphs start from the same resident restore buffers. No monkey patching.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog

import numpy as np


def add_arguments(parser):
    parser.add_argument('--input', type=Path, required=True, help='Newton boundary-v1 .npz and adjacent summary.json')
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--direction-library', type=Path, help='also replay handwritten gradient/direction')
    parser.add_argument('--constraint-library', type=Path, help='also replay handwritten constraint update')
    parser.add_argument('--linesearch-library', type=Path, help='also replay handwritten line search')
    parser.add_argument('--state-library', type=Path, help='also replay complete handwritten Newton')
    parser.add_argument('--task0-input', type=Path, help='optional frozen real task-0 Newton boundary with same-stem JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    return parser


def fields(data):
    result = {name: getattr(data, name) for name in
        ('qpos', 'qvel', 'ctrl', 'time', 'nefc', 'ne', 'nf', 'M', 'qacc_warmstart',
         'qacc_smooth', 'qfrc_smooth', 'qacc', 'qfrc_constraint', 'nacon', 'overflow', 'solver_niter')}
    result.update({'efc.' + name: getattr(data.efc, name) for name in
        ('J', 'D', 'aref', 'frictionloss', 'type', 'id', 'pos', 'margin', 'force', 'state',
         'jtdaj_adr', 'jtdaj_nrow', 'jtdaj_nblock')})
    result.update({'contact.' + name: getattr(data.contact, name) for name in
        ('worldid', 'dim', 'friction', 'dist', 'includemargin', 'efc_address')})
    return result


def model_for(root, task):
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.mjwarp.model import build_scene
    built, model, _, _, identity = build_scene(root,
        str(Path(get_libero_path('bddl_files')) / task['key']), 0,
        dict(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
             hard_reset=False, ignore_done=True))
    return built.native_model, model, identity


def make_fixtures(args):
    import mujoco_warp as mjw
    from mujoco_warp._src import types
    tasks = load_catalog(args.build_root)
    task = next(t for t in tasks if t['suite'] == 'libero_10' and t['task_id'] == 4)
    host, model, identity = model_for(args.build_root, task)
    saved = dict(np.load(args.input, allow_pickle=False))
    receipt = json.loads(args.input.with_name('summary.json').read_text())
    for key in ('scene_sha256', 'imported_xml_sha256', 'imported_model_sha256', 'mjwarp_commit'):
        if identity[key] != receipt['identity'][key]:
            raise ValueError('frozen model identity differs: ' + key)
    for field in ('M_hinit_i', 'M_colind', 'dof_parentid'):
        if not np.array_equal(getattr(model, field).numpy(), saved['model.' + field]):
            raise ValueError('frozen static array differs: ' + field)
    opt = receipt['frozen_newton']['solver']
    for name in ('iterations', 'ls_iterations', 'cone', 'disableflags'):
        if int(getattr(model.opt, name)) != opt[name]:
            raise ValueError('frozen Newton option differs: ' + name)
    if saved['input.qacc'].shape[0] != 2:
        raise ValueError('this qualification protocol requires the N=2 primary input')
    for name, owner, field in [('tolerance', model.opt, 'tolerance'),
                              ('impratio_invsqrt', model.opt, 'impratio_invsqrt'),
                              ('meaninertia', model.stat, 'meaninertia')]:
        if not np.array_equal(getattr(owner, field).numpy(), saved['model.' + name]):
            raise ValueError('frozen scalar array differs: ' + name)
    layout = receipt['frozen_newton']['layout']
    fixtures = []
    for variant in ('original', 'cold', 'converged-warmstart', 'empty-world0'):
        m = model
        if variant == 'cold':
            m = dataclasses.replace(model, opt=dataclasses.replace(model.opt,
                disableflags=model.opt.disableflags | types.DisableBit.WARMSTART))
        data = mjw.make_data(host, nworld=2, nconmax=layout['pooled_contacts'] // 2,
                             nccdmax=layout['pooled_contacts'] // 2, njmax=layout['rows'])
        for name, target in fields(data).items():
            if 'input.' + name in saved:
                target.assign(saved['input.' + name])
        if variant == 'converged-warmstart':
            data.qacc_warmstart.assign(saved['output.qacc'])
        elif variant == 'empty-world0':
            for name in ('ne', 'nf', 'nefc'):
                value = getattr(data, name).numpy(); value[0] = 0; getattr(data, name).assign(value)
            for name in ('state', 'force'):
                value = getattr(data.efc, name).numpy(); value[0] = 0; getattr(data.efc, name).assign(value)
            value = data.contact.dist.numpy()
            value[data.contact.worldid.numpy() == 0] = 1.
            data.contact.dist.assign(value)
        fixtures.append(dict(name='task4-' + variant, host=host, model=m, data=data,
            origin='frozen Newton input' if variant == 'original' else 'explicit variation of frozen QP input',
            model_identity=identity))
    for task_id in (5, 0):
        task = next(t for t in tasks if t['suite'] == 'libero_10' and t['task_id'] == task_id)
        host, m, ident = model_for(args.build_root, task)
        data = mjw.make_data(host, nworld=2, nconmax=8, nccdmax=1, njmax=64)
        manufacture(m, data)
        fixtures.append(dict(name=f'task{task_id}-nv{m.nv}-manufactured-QP',
            host=host, model=m, data=data, origin='manufactured QP in real model NV/CSR layout',
            model_identity=ident))
    task0_input = getattr(args, 'task0_input', None)
    if task0_input is not None:
        import hashlib
        receipt = json.loads(task0_input.with_suffix('.json').read_text())
        if hashlib.sha256(task0_input.read_bytes()).hexdigest() != receipt['input_sha256']:
            raise ValueError('task-0 frozen input digest differs')
        for key in ('scene_sha256','imported_xml_sha256','imported_model_sha256','mjwarp_commit'):
            if ident[key] != receipt['identity'][key]:
                raise ValueError('task-0 frozen model differs: ' + key)
        for key,value in receipt['solver'].items():
            if int(getattr(m.opt,key)) != value:
                raise ValueError('task-0 frozen solver option differs: ' + key)
        saved0 = dict(np.load(task0_input, allow_pickle=False))
        for key in ('M_hinit_i','M_colind','dof_parentid'):
            if not np.array_equal(getattr(m,key).numpy(),saved0['model.'+key]):
                raise ValueError('task-0 frozen static layout differs: ' + key)
        count = len(saved0['input.contact.worldid'])
        data0 = mjw.make_data(host,nworld=2,nconmax=count//2,nccdmax=count//2,
                             njmax=saved0['input.efc.D'].shape[1])
        for name,target in fields(data0).items():
            if 'input.'+name in saved0:target.assign(saved0['input.'+name])
        fixtures.append(dict(name='task0-frozen-real',host=host,model=m,data=data0,
            origin='frozen initial raw-forward boundary; no integration',model_identity=ident))
    return fixtures


def manufacture(model, data):
    """Well-conditioned convex QP: equality/friction/limits/condim 3,4,6."""
    from mujoco_warp._src import types
    rng = np.random.default_rng(8100 + model.nv)
    nv, pad = model.nv, model.nv_pad
    mr, mc = model.M_hinit_i.numpy(), model.M_colind.numpy()
    diagonal = rng.uniform(1., 3., (2, nv)).astype(np.float32)
    mass = np.zeros(data.M.shape, np.float32)
    for k, (i, j) in enumerate(zip(mr, mc)):
        if i == j:
            mass[:, k] = diagonal[:, i]
    data.M.assign(mass)
    smooth = rng.normal(0., .1, (2, nv)).astype(np.float32)
    data.qacc_smooth.assign(smooth)
    data.qfrc_smooth.assign(smooth * diagonal)
    data.qacc_warmstart.assign(rng.normal(0., .2, (2, nv)).astype(np.float32))
    data.ne.assign(np.array([1, 1], np.int32)); data.nf.assign(np.array([3, 3], np.int32))
    data.nefc.assign(np.array([23, 21], np.int32))
    j = np.zeros(data.efc.J.shape, np.float32)
    j[:, :23, :nv] = rng.normal(0., .12, (2, 23, nv))
    data.efc.J.assign(j)
    d = np.zeros(data.efc.D.shape, np.float32); d[:, :23] = rng.uniform(1., 5., (2, 23))
    data.efc.D.assign(d)
    aref = np.zeros_like(d); aref[:, :23] = rng.normal(0., .2, (2, 23)); data.efc.aref.assign(aref)
    loss = np.zeros_like(d); loss[:, 1:4] = .2; data.efc.frictionloss.assign(loss)
    typ = np.full(data.efc.type.shape, int(types.ConstraintType.LIMIT_JOINT), np.int32)
    typ[:, 0] = int(types.ConstraintType.EQUALITY); typ[:, 1:4] = int(types.ConstraintType.FRICTION_DOF)
    ids = np.zeros(data.efc.id.shape, np.int32)
    world = np.zeros(data.contact.worldid.shape, np.int32)
    dims = np.ones(data.contact.dim.shape, np.int32)
    friction = np.zeros(data.contact.friction.numpy().shape, np.float32)
    address = np.full(data.contact.efc_address.shape, -1, np.int32)
    for w in range(2):
        head = 4
        for k, dim in enumerate((3, 4, 6)):
            c = 2 * k + w
            world[c], dims[c], friction[c] = w, dim, [.7, .6, .05, .01, .01]
            address[c, :dim] = np.arange(head, head + dim)
            typ[w, head:head + dim] = int(types.ConstraintType.CONTACT_ELLIPTIC)
            ids[w, head:head + dim] = c
            # CPU 2.3.7 engine_core_constraint.c: R_t = R_n/impratio,
            # then friction-axis scaling. This relation also makes the cone
            # cost/gradient continuous at the quadratic/middle boundary.
            mu = float(friction[c, 0]) * float(model.opt.impratio_invsqrt.numpy()[0])
            d[w, head] *= .1
            d[w, head + 1:head + dim] = (d[w, head] *
                friction[c, :dim - 1].astype(np.float64) ** 2 / (mu * mu))
            head += dim
    data.efc.D.assign(d)
    data.efc.type.assign(typ); data.efc.id.assign(ids)
    data.contact.worldid.assign(world); data.contact.dim.assign(dims)
    data.contact.friction.assign(friction); data.contact.efc_address.assign(address)
    dist = np.zeros(data.contact.dist.shape, np.float32); dist[:6] = -.01
    data.contact.dist.assign(dist)
    data.nacon.assign(np.array([6], np.int32))


class Replay:
    def __init__(self, fixture, library, stream, direction_library=None, constraint_library=None, linesearch_library=None, state_library=None):
        import warp as wp
        from mujoco_warp._src import solver
        from libero.gpu.host.backends.mjwarp.newton import Newton
        self.fixture, self.stream = fixture, stream
        self.m, self.d = fixture['model'], fixture['data']
        self.inputs = fields(self.d)
        self.host_inputs = {name: array.numpy() for name, array in self.inputs.items()}
        self.sources = {name: wp.clone(array) for name, array in self.inputs.items()}
        with wp.ScopedCapture(stream=stream) as capture:
            for name, target in self.inputs.items():
                wp.copy(target, self.sources[name], stream=stream)
        self.restore_graph = capture.graph
        self.graphs, self.contexts, self.owners, self.drivers = {}, {}, {}, {}
        self.candidates = ('staged_upstream', 'cuda_hessian') + (('cuda_direction',) if direction_library else ())
        if constraint_library and not direction_library:
            raise ValueError('constraint candidate requires its admitted direction library')
        self.candidates += ('cuda_constraint',) if constraint_library else ()
        if linesearch_library and not constraint_library:
            raise ValueError('linesearch candidate requires its constraint library')
        self.candidates += ('cuda_linesearch',) if linesearch_library else ()
        if state_library and not linesearch_library:
            raise ValueError('complete Newton candidate requires its linesearch library')
        self.candidates += ('cuda_newton',) if state_library else ()
        self.candidate = 'cuda_newton' if state_library else 'cuda_linesearch' if linesearch_library else 'cuda_constraint' if constraint_library else 'cuda_direction' if direction_library else 'cuda_hessian'
        for mode in ('raw-a', 'raw-b') + self.candidates:
            if mode.startswith('raw-'):
                with wp.ScopedCapture(stream=stream) as capture:
                    ctx = solver._create_solver_context(self.m, self.d)
                    solver._solve(self.m, self.d, ctx)
                    if self.m.opt.noslip_iterations > 0:
                        solver.noslip(self.m, self.d)
                self.owners[mode] = [ctx]
            else:
                driver = Newton(self.m, self.d, stream, implementation=mode, library=library,
                                direction_library=direction_library, constraint_library=constraint_library, linesearch_library=linesearch_library, state_library=state_library)
                if driver.selected != mode:
                    raise ValueError('fixture fell outside candidate domain: ' + str(driver.fallback))
                with driver.capture_owners() as owners, wp.ScopedCapture(stream=stream) as capture:
                    ctx = driver.solve()[0]
                self.owners[mode], self.drivers[mode] = owners, driver
            self.graphs[mode], self.contexts[mode] = capture.graph, ctx

    def run(self, mode):
        import warp as wp
        wp.capture_launch(self.restore_graph, stream=self.stream)
        wp.capture_launch(self.graphs[mode], stream=self.stream)
        wp.synchronize_stream(self.stream)
        d, ctx = self.d, self.contexts[mode]
        return dict(qacc=d.qacc.numpy(), qfrc_constraint=d.qfrc_constraint.numpy(),
            force=d.efc.force.numpy(), state=d.efc.state.numpy(), niter=d.solver_niter.numpy(),
            overflow=d.overflow.numpy(), gradient=ctx.grad.numpy()[:, :self.m.nv],
            alpha=ctx.alpha.numpy(), improvement=ctx.improvement.numpy(),
            decrement=ctx.newton_decrement.numpy(), done=ctx.done.numpy())

    def diagnostics(self, output):
        from validation.precision import balance_residual
        a, m, d = self.host_inputs, self.m, self.d
        rows, columns = m.M_hinit_i.numpy(), m.M_colind.numpy()
        matrix = np.zeros((2, m.nv, m.nv), np.float64)
        matrix[:, rows, columns] = a['M']; matrix[:, columns, rows] = a['M']
        result = []
        for w in range(2):
            used = a['nefc'][w]
            j = a['efc.J'][w, :used, :m.nv].astype(np.float64)
            reduced = j.T @ output['force'][w, :used].astype(np.float64)
            result.append(dict(
                published_force_balance=balance_residual(matrix[w], output['qacc'][w],
                    a['qfrc_smooth'][w], output['qfrc_constraint'][w]),
                independently_reduced_force_balance=balance_residual(matrix[w], output['qacc'][w],
                    a['qfrc_smooth'][w], reduced),
                force_publication_error=float(abs(reduced - output['qfrc_constraint'][w]).max())))
        return result

    def check(self):
        from validation.mjwarp_policy import compare_replays
        outputs = {mode: self.run(mode) for mode in self.graphs}
        first, second = outputs['raw-a'], outputs['raw-b']
        compared = ('qacc', 'qfrc_constraint', 'force')
        checks = {}
        for mode in self.candidates:
            out = outputs[mode]
            checks[mode] = compare_replays({k: out[k] for k in compared},
                {k: first[k] for k in compared}, {k: second[k] for k in compared})
        # Solver does not publish warmstart; integration owns that publication.
        holds = {name: bool(np.array_equal(self.inputs[name].numpy(), self.host_inputs[name]))
            for name in ('qpos', 'qvel', 'ctrl', 'time', 'qacc_warmstart', 'M', 'efc.J', 'nefc', 'nacon')}
        extra_overflow = outputs[self.candidate]['overflow'] & ~(first['overflow'] | second['overflow'])
        diagnostics = {mode: self.diagnostics(out) for mode, out in outputs.items()}
        legal = all(np.isfinite(out[k]).all() for out in outputs.values()
                    for k in ('qacc', 'qfrc_constraint', 'force'))
        legal &= all(np.all((out['niter'] >= 0) & (out['niter'] <= self.m.opt.iterations))
                     and np.all((out['state'] >= 0) & (out['state'] <= 4)) for out in outputs.values())
        result = dict(name=self.fixture['name'], origin=self.fixture['origin'], nv=self.m.nv, worlds=2,
            rows=self.host_inputs['nefc'].tolist(), holds=holds,
            equivalence_checks=checks, extra_overflow=extra_overflow.tolist(),
            iterations={mode: out['niter'].tolist() for mode, out in outputs.items()},
            states_changed_vs_raw={mode: int(np.count_nonzero(out['state'] != first['state']))
                for mode, out in outputs.items()}, diagnostics=diagnostics,
            independent_balance_all_passed=all(item[key]['passed'] for diag in diagnostics.values()
                for item in diag for key in ('published_force_balance', 'independently_reduced_force_balance')),
            passed=bool(legal and all(holds.values()) and not np.any(extra_overflow) and
                all(c['passed'] for group in checks.values() for c in group)))
        return result, outputs

    def benchmark(self, args):
        import warp as wp
        result = {}
        modes = ('raw-a', 'cuda_linesearch', self.candidate) if self.candidate == 'cuda_newton' else ('raw-a', self.candidate)
        for mode in modes:
            for _ in range(args.warmup):
                wp.capture_launch(self.restore_graph, stream=self.stream)
                wp.capture_launch(self.graphs[mode], stream=self.stream)
            wp.synchronize_stream(self.stream)
            start = wp.Event(device=self.stream.device, enable_timing=True)
            end = wp.Event(device=self.stream.device, enable_timing=True)
            values, iterations = [], []
            for _ in range(args.steps):
                wp.capture_launch(self.restore_graph, stream=self.stream)
                self.stream.record_event(start)
                wp.capture_launch(self.graphs[mode], stream=self.stream)
                self.stream.record_event(end)
                wp.synchronize_event(end)
                values.append(wp.get_event_elapsed_time(start, end, synchronize=False) * 1000)
                iterations.append(self.d.solver_niter.numpy().tolist())
            result[mode] = dict(mean_us=float(np.mean(values)), std_us=float(np.std(values)),
                microseconds=values, iterations=iterations,
                derived_hessian_launches=[1 + max(n) if self.m.opt.graph_conditional else
                    1 + self.m.opt.iterations for n in iterations])
        result['speedup'] = result['raw-a']['mean_us'] / result[self.candidate]['mean_us']
        if self.candidate == 'cuda_newton':
            result['speedup_vs_previous_cuda'] = result['cuda_linesearch']['mean_us'] / result[self.candidate]['mean_us']
        result['scope'] = 'one frozen-input Newton solve across N=2; restore copies excluded; no physics integration'
        return result


def main(argv=None, *, args=None):
    if args is None:
        args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.warmup < 0 or args.steps < 1:
        raise ValueError('invalid timing counts')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.build.compilation import compilation_scope, install_warp_guard
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    args.output.mkdir(parents=True, exist_ok=True)
    init_kernel_cache(args.build_root / 'mjwarp/warp-cache')
    install_warp_guard(); wp.init()
    stream = wp.Stream('cuda:0')
    signature = dict(sources=source_files(include_hessian=True,
                                        include_direction=bool(args.direction_library),
                                        include_constraint=bool(args.constraint_library),
                                        include_linesearch=bool(args.linesearch_library),
                                        include_state=bool(args.state_library)),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        input_receipt_sha256=hashlib.sha256(args.input.with_name('summary.json').read_bytes()).hexdigest(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    if args.direction_library:
        signature['direction_library_sha256'] = hashlib.sha256(args.direction_library.read_bytes()).hexdigest()
    if args.constraint_library:
        signature['constraint_library_sha256'] = hashlib.sha256(args.constraint_library.read_bytes()).hexdigest()
    if args.linesearch_library:
        signature['linesearch_library_sha256'] = hashlib.sha256(args.linesearch_library.read_bytes()).hexdigest()
    if args.state_library:
        signature['state_library_sha256'] = hashlib.sha256(args.state_library.read_bytes()).hexdigest()
    if args.task0_input:
        signature['task0_input_sha256'] = hashlib.sha256(args.task0_input.read_bytes()).hexdigest()
        signature['task0_receipt_sha256'] = hashlib.sha256(args.task0_input.with_suffix('.json').read_bytes()).hexdigest()
    prepared = args.output / 'preparation.json'
    if not args.prepare and (not prepared.is_file() or json.loads(prepared.read_text())['signature'] != signature):
        raise RuntimeError('prepare these exact Newton graphs offline first')
    with compilation_scope(args.prepare) as scope, wp.ScopedStream(stream):
        fixtures = make_fixtures(args)
        replays = [Replay(f, args.library, stream, args.direction_library, args.constraint_library, args.linesearch_library, args.state_library) for f in fixtures]
        if args.prepare:
            wp.synchronize_stream(stream)
            save_json(prepared, dict(signature=signature, compilation_events=scope['events'],
                cases=[f['name'] for f in fixtures], physics_integrations=0, solver_replays=0))
            print(json.dumps(dict(prepared=True, cases=len(fixtures), compilations=len(scope['events']))), flush=True)
            return 0
        load_before = gpu_load_snapshot()
        rows = []
        for replay in replays:
            row, output = replay.check()
            rows.append(row)
            np.savez(args.output / (row['name'] + '.npz'),
                **{mode + '.' + name: value for mode, values in output.items() for name, value in values.items()})
            print(json.dumps(dict(name=row['name'], passed=row['passed'], iterations=row['iterations'],
                independent_balance_all_passed=row['independent_balance_all_passed'])), flush=True)
        passed = all(r['passed'] for r in rows)
        save_json(args.output / 'correctness.json', dict(passed=passed, cases=rows, signature=signature))
        timing = replays[0].benchmark(args) if passed and args.performance else None
        additional_timing = ([dict(name=r.fixture['name'], performance=r.benchmark(args)) for r in replays
                              if r.fixture['name'] == 'task0-frozen-real'] if passed and args.performance else [])
        result = dict(passed=passed, signature=signature, cases=rows, performance=timing,
            additional_performance=additional_timing,
            independent_balance_all_passed=all(r['independent_balance_all_passed'] for r in rows),
            runtime_compilations=scope['events'], physics_integrations=0,
            gpu_load_before=load_before, gpu_load_after=gpu_load_snapshot(),
            policy='unchanged MJWarp FP32 same-engine envelope; independent residual diagnostics recorded separately')
        save_json(args.output / 'summary.json', result)
        print(json.dumps(dict(passed=passed, performance=timing, runtime_compilations=len(scope['events']))), flush=True)
        return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
