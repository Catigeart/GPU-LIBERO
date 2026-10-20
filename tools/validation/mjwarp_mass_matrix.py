"""Frozen N=2 CRB/mass checks, independent FP64 bounds and CPU 2.3.7 oracle."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

REAL_TASKS = (('libero_10', 0), ('libero_10', 3), ('libero_spatial', 0))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_arguments(p):
    for name in ('build-root', 'library', 'catalogue', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--cpu-reference', action='store_true', help='CPU 2.3.7 only; no Warp initialization')
    return p


def real_cases(output):
    result = []
    for suite, task in REAL_TASKS:
        path = output / f'{suite}-{task}.npz'
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['sha256'] != digest(path):
            raise RuntimeError('Frozen CRB input changed')
        with np.load(path, allow_pickle=False) as f:
            c = {k: f[k].copy() for k in f.files}
        for key in ('nv', 'nbody', 'nC'):
            c[key] = int(c[key])
        c.update(name=f'{suite}-{task}', model_path=receipt['model_path'], origin='frozen upstream kinematics')
        result.append(c)
    return result


def freeze(args, stream):
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import forward, smooth
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.mjwarp.model import build_scene, capacity_budget
    rows = json.loads(args.catalogue.read_text())['rows']
    for suite, task in REAL_TASKS:
        path = args.output / f'{suite}-{task}.npz'
        if path.exists():
            if json.loads(path.with_suffix('.json').read_text())['sha256'] != digest(path):
                raise RuntimeError('Frozen CRB input changed')
            continue
        row = next(r for r in rows if r['suite'] == suite and r['task_id'] == task)
        built, m, bank, _, identity = build_scene(args.build_root,
            str(Path(get_libero_path('bddl_files')) / row['task']), 0,
            dict(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, hard_reset=False, ignore_done=True))
        budget = capacity_budget(built.native_model)
        d = mjw.make_data(built.native_model, nworld=2, nccdmax=budget['nconmax'], **budget)
        d.qpos.assign(np.ascontiguousarray(bank[:2, 1:1+m.nq], np.float32))
        d.qvel.assign(np.ascontiguousarray(bank[:2, 1+m.nq:], np.float32))
        roots = {k: getattr(d, k).numpy() for k in ('qpos', 'qvel', 'time')}
        forward.fwd_kinematics(m, d)
        smooth.crb(m, d)
        wp.synchronize_stream(stream)
        if m.nv != row['nv'] or m.nbody != row['nbody'] or not np.array_equal(m.dof_bodyid.numpy(), row['dof_bodyid']):
            raise RuntimeError('CRB model differs from the 130-task topology catalogue')
        if not all(np.array_equal(getattr(d, k).numpy(), v) for k, v in roots.items()):
            raise RuntimeError('CRB preparation advanced physical roots')
        c = dict(nv=np.array(m.nv), nbody=np.array(m.nbody), nC=np.array(m.nC),
            parent=m.body_parentid.numpy(), dof_body=m.dof_bodyid.numpy(), dof_parent=m.dof_parentid.numpy(),
            row=m.M_rowadr.numpy(), nnz=m.M_rownnz.numpy(), column=m.M_colind.numpy(),
            cinert=d.cinert.numpy(), cdof=d.cdof.numpy(), armature=m.dof_armature.numpy(),
            frozen_crb=d.crb.numpy(), frozen_matrix=d.M.numpy())
        for i, layer in enumerate(m.body_tree):
            c['layer' + str(i)] = layer.numpy()
        np.savez(path, **c)
        path.with_suffix('.json').write_text(json.dumps(dict(sha256=digest(path), identity=identity,
            model_path=row['model_path'], model_sha256=row['model_sha256'], catalogue_sha256=digest(args.catalogue),
            physics_integrations=0, newton_solves=0), indent=2) + '\n')
        print(json.dumps(dict(frozen=str(path), nv=m.nv, nbody=m.nbody, nC=m.nC)), flush=True)
        del built, m, d


def synthetic(parent, nv, name, armature_rows=2, zero=False):
    rng = np.random.default_rng(881 + nv + len(parent))
    parent = np.array(parent, np.int32); nb = len(parent)
    dof_body = np.sort(np.arange(nv) % (nb-1) + 1).astype(np.int32)
    last = np.full(nb, -1, np.int32); dof_parent = []
    for i, body in enumerate(dof_body):
        b = int(body)
        while b and last[b] < 0:
            b = int(parent[b])
        dof_parent.append(int(last[b])); last[body] = i
    dof_parent = np.array(dof_parent, np.int32)
    row, nnz, column = [], [], []
    for i in range(nv):
        chain = []; j = i
        while j >= 0:
            chain.append(j); j = int(dof_parent[j])
        row.append(len(column)); column.extend(chain[::-1]); nnz.append(len(chain))
    cinert = rng.normal(scale=.01, size=(2, nb, 10)).astype(np.float32)
    for w in range(2):
        for b in range(nb):
            r = rng.normal(size=(3, 3)); inertia = r @ r.T + np.eye(3)
            cinert[w, b, :6] = inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]
            cinert[w, b, 9] = .5 + rng.random()
        cinert[w] *= 1.e-3 if w == 0 else 1.e3
    if zero:
        cinert[:] = 0
    c = dict(name=name, nv=nv, nbody=nb, nC=len(column), parent=parent, dof_body=dof_body, dof_parent=dof_parent,
        row=np.array(row, np.int32), nnz=np.array(nnz, np.int32), column=np.array(column, np.int32),
        cinert=cinert, cdof=rng.normal(size=(2, nv, 6)).astype(np.float32),
        armature=rng.uniform(.01, .1, (armature_rows, nv)).astype(np.float32), origin='manufactured tree / independent CSR')
    depth = np.zeros(nb, np.int32)
    for b in range(1, nb):
        depth[b] = depth[parent[b]] + 1
    for level in range(int(depth.max())+1):
        c['layer' + str(level)] = np.flatnonzero(depth == level).astype(np.int32)
    return c


def cases(output):
    return real_cases(output) + [
        synthetic([0, 0], 1, 'single-body', 1),
        synthetic([0, 0] + [1]*36, 57, 'wide-fanin', 2),
        synthetic([0, 0] + list(range(1, 13)) + [1]*24, 57, 'deep-branching', 1),
        synthetic([0, 0, 1, 2, 0, 4, 4, 0, 7, 8], 21, 'independent-trees', 3),
        synthetic([0, 0, 1, 1, 2, 2, 0], 24, 'zero-inertia', 2, zero=True)]


def spatial(inertia):
    """Independent dense spatial matrix; input vec10 -> symmetric 6x6."""
    q = np.asarray(inertia, np.float64)
    a = np.zeros((6, 6), np.float64)
    a[:3, :3] = [[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]]
    cross = np.array([[0, -q[8], q[7]], [q[8], 0, -q[6]], [-q[7], q[6], 0]])
    a[:3, 3:] = cross; a[3:, :3] = cross.T; a[3:, 3:] = np.eye(3)*q[9]
    return a


def reference(c):
    """FP64 CPU-reference formulas; bounds use input majorants, never outputs."""
    unit = np.finfo(np.float32).eps/2
    gamma = lambda n: n*unit/(1-n*unit)
    crb = c['cinert'].astype(np.float64).copy(); major = abs(crb).copy(); counts = np.ones(c['nbody'], int)
    for b in reversed(range(1, c['nbody'])):
        p = int(c['parent'][b])
        if p:
            crb[:, p] += crb[:, b]; major[:, p] += major[:, b]; counts[p] += counts[b]
    crb_limit = major*np.array([8*gamma(n-1) for n in counts])[None, :, None]
    matrix = np.zeros((2, c['nC']), np.float64); matrix_limit = np.zeros_like(matrix)
    for w in range(2):
        for i in range(c['nv']):
            b = int(c['dof_body'][i]); motion = c['cdof'][w, i].astype(np.float64)
            inertia = spatial(crb[w, b]); ierr = abs(spatial(crb_limit[w, b]))
            product = inertia @ motion
            perr = ierr @ abs(motion) + 8*gamma(8)*(abs(inertia)+ierr) @ abs(motion)
            for at in range(int(c['row'][i]), int(c['row'][i]+c['nnz'][i])):
                j = int(c['column'][at]); v = c['cdof'][w, j].astype(np.float64)
                value = float(v @ product)
                err = float(abs(v) @ perr + 8*gamma(12)*(abs(v) @ (abs(product)+perr)))
                if i == j:
                    armature = float(c['armature'][w % len(c['armature']), i])
                    err += 8*gamma(1)*(abs(armature)+float(abs(v) @ (abs(product)+perr))+err)
                    value += armature
                matrix[w, at] = value; matrix_limit[w, at] = err
    return crb, matrix, crb_limit, matrix_limit


def cpu_reference(args):
    """Run only installed CPU MuJoCo 2.3.7 CRB, with the exact frozen inputs."""
    import mujoco
    if mujoco.__version__ != '2.3.7':
        raise RuntimeError('CPU oracle requires the CPU reference 2.3.7 environment')
    saved = {}; checks = []
    for c in real_cases(args.output):
        receipt = json.loads((args.output/(c['name']+'.json')).read_text())
        if digest(c['model_path']) != receipt['model_sha256']:
            raise RuntimeError('CPU oracle model changed')
        m = mujoco.MjModel.from_binary_path(c['model_path']); d = mujoco.MjData(m)
        if (m.nv != c['nv'] or m.nbody != c['nbody'] or
                not np.array_equal(m.body_parentid, c['parent']) or
                not np.array_equal(m.dof_bodyid, c['dof_body']) or
                not np.array_equal(m.dof_parentid, c['dof_parent'])):
            raise RuntimeError('CPU reference topology differs from frozen MJWarp input')
        # Inputs are supplied directly, so disable the CPU shortcut that reads
        # cached dof_M0 instead of the supplied cinert/cdof. MJWarp has no shortcut.
        m.dof_simplenum[:] = 0
        cs, ms = [], []
        for w in range(2):
            d.cinert[:] = c['cinert'][w]; d.cdof[:] = c['cdof'][w]
            m.dof_armature[:] = c['armature'][w % len(c['armature'])]
            mujoco.mj_crb(m, d)
            full = np.empty((m.nv, m.nv), np.float64); mujoco.mj_fullM(m, full, d.qM)
            rr = np.repeat(np.arange(m.nv), c['nnz'])
            cs.append(d.crb.copy()); ms.append(full[rr, c['column']])
        ref_crb, ref_matrix, _, _ = reference(c)
        cs, ms = np.array(cs), np.array(ms)
        for name, out, ref in (('crb', cs, ref_crb), ('matrix', ms, ref_matrix)):
            # FP64 oracle agreement only; this is not the candidate FP32 gate.
            scale = max(float(abs(ref).max()), np.finfo(float).tiny)
            error = float(abs(out-ref).max()); limit = 512*np.finfo(float).eps*scale
            if error > limit:
                raise RuntimeError('Independent FP64 reference differs from CPU CRB: ' + c['name'] + '/' + name)
            checks.append(dict(name=c['name']+'/'+name, error=error, limit=limit))
        saved[c['name']+'.crb'] = cs; saved[c['name']+'.matrix'] = ms
    path = args.output/'cpu-oracle.npz'; np.savez(path, **saved)
    (args.output/'cpu-oracle.json').write_text(json.dumps(dict(version=mujoco.__version__, sha256=digest(path),
        tool_sha256=digest(__file__), inputs={c['name']:digest(args.output/(c['name']+'.npz')) for c in real_cases(args.output)},
        checks=checks, physics_integrations=0, newton_solves=0), indent=2)+'\n')
    print(json.dumps(dict(cpu_oracle=str(path), checks=len(checks), passed=True)), flush=True)
    return 0


class Replay:
    def __init__(self, c, args, stream):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.dynamics.mjwarp_mass_matrix import MassMatrix
        self.case, self.stream = c, stream
        arr = lambda a, dtype: wp.array(np.ascontiguousarray(a), dtype=dtype, device=stream.device)
        m = self.model = NS(nv=c['nv'], nbody=c['nbody'], nC=c['nC'])
        for key, field in (('parent', 'body_parentid'), ('dof_body', 'dof_bodyid'), ('dof_parent', 'dof_parentid'),
                ('row', 'M_rowadr'), ('nnz', 'M_rownnz'), ('column', 'M_colind')):
            setattr(m, field, arr(c[key], int))
        m.dof_armature = arr(c['armature'], float)
        m.body_tree = tuple(arr(c['layer'+str(i)], int) for i in range(sum(k.startswith('layer') for k in c)))
        d = self.data = NS(nworld=2, cinert=arr(c['cinert'], types.vec10), cdof=arr(c['cdof'], wp.spatial_vector))
        self.outputs, self.graphs = {}, {}
        for mode in ('upstream', 'cuda'):
            d.crb = wp.empty((2, m.nbody), dtype=types.vec10, device=stream.device)
            d.M = wp.empty((2, m.nC), dtype=float, device=stream.device)
            self.outputs[mode] = (d.crb, d.M)
            if mode == 'cuda':
                self.binding = MassMatrix(m, d, args.library, stream)
            with wp.ScopedCapture(stream=stream) as capture:
                if mode == 'cuda':
                    self.binding.enqueue()
                else:
                    from mujoco_warp._src import smooth
                    smooth.crb(m, d)
            self.graphs[mode] = capture.graph

    def check(self, updated):
        import warp as wp
        c = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.case.items()}
        if updated:
            c['cinert'] *= np.array([1.25, .625], np.float32)[:, None, None]
            c['cdof'] *= np.array([.75, -1.5], np.float32)[:, None, None]
            c['armature'] = c['armature']*np.float32(1.5) + np.float32(.03125)
        self.data.cinert.assign(c['cinert']); self.data.cdof.assign(c['cdof']); self.model.dof_armature.assign(c['armature'])
        inputs = {k: a.numpy() for k, a in self.binding.arrays.items() if k not in ('crb', 'matrix')}
        ref = reference(c); results = {}; rows = []
        for mode, graph in self.graphs.items():
            crb, matrix = self.outputs[mode]; crb.fill_(37.); matrix.fill_(37.)
            wp.capture_launch(graph, stream=self.stream); wp.synchronize_stream(self.stream)
            out_crb, out_matrix = crb.numpy(), matrix.numpy(); results[mode] = dict(crb=out_crb, matrix=out_matrix)
            for name, out, expected, limit in (('crb', out_crb, ref[0], ref[2]), ('matrix', out_matrix, ref[1], ref[3])):
                error = abs(out.astype(np.float64)-expected)
                fractions = np.divide(error, limit, out=np.zeros_like(error), where=limit > 0)
                passed = bool(np.isfinite(out).all() and np.all(error <= limit))
                rows.append(dict(mode=mode, field=name, max_error=float(error.max()),
                    maximum_budget_fraction=float(fractions.max()), passed=passed))
        held = all(np.array_equal(a.numpy(), inputs[k]) for k, a in self.binding.arrays.items() if k in inputs)
        return dict(name=c['name'], updated=updated, checks=rows, inputs_held=held,
            max_difference={k:float(abs(results['cuda'][k]-results['upstream'][k]).max()) for k in ('crb', 'matrix')},
            passed=held and all(x['passed'] for x in rows)), results


def main(argv=None, *, args=None):
    if args is None:
        args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.cpu_reference:
        return cpu_reference(args)
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.build.compilation import compilation_scope, install_warp_guard
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    require_upstream(); init_kernel_cache(args.build_root/'mjwarp/warp-cache'); install_warp_guard(); wp.init()
    args.output.mkdir(parents=True, exist_ok=True); stream = wp.Stream('cuda:0')
    with compilation_scope(args.prepare) as scope, wp.ScopedStream(stream):
        if args.prepare:
            freeze(args, stream)
        sources = source_files(include_matrix=True)
        sources = {k:v for k,v in sources.items() if k.startswith('upstream/') or 'mjwarp_mass_matrix' in k or
            k in ('cmake/MjwarpMassMatrix.cmake', 'host/backends/mjwarp/model.py', 'project_source_identity', 'data/mjwarp_source_pin.json')}
        signature = dict(sources=sources, tool_sha256=digest(__file__), library_sha256=digest(args.library),
            catalogue_sha256=digest(args.catalogue),
            inputs={f'{s}-{t}':digest(args.output/f'{s}-{t}.npz') for s,t in REAL_TASKS},
            receipts={f'{s}-{t}':digest(args.output/f'{s}-{t}.json') for s,t in REAL_TASKS})
        prepared = args.output/'preparation.json'
        if not args.prepare and (not prepared.exists() or json.loads(prepared.read_text())['signature'] != signature):
            raise RuntimeError('Prepare exact CRB operators offline first')
        replays = [Replay(c, args, stream) for c in cases(args.output)]
        if args.prepare:
            wp.synchronize_stream(stream)
            save_json(prepared, dict(signature=signature, compilation_events=scope['events'], physics_integrations=0, newton_solves=0))
            print(json.dumps(dict(prepared=True, cases=len(replays), compilations=len(scope['events']))), flush=True)
            return 0
        oracle_path = args.output/'cpu-oracle.npz'; oracle_receipt = json.loads((args.output/'cpu-oracle.json').read_text())
        if (oracle_receipt['sha256'] != digest(oracle_path) or oracle_receipt['tool_sha256'] != digest(__file__) or
                oracle_receipt['inputs'] != signature['inputs']):
            raise RuntimeError('Prepare CPU 2.3.7 CRB oracle on the exact inputs first')
        before = gpu_load_snapshot(); rows = []
        for replay in replays:
            for updated in (False, True):
                row, outputs = replay.check(updated); rows.append(row)
                np.savez(args.output/(row['name']+('-updated' if updated else '')+'-output.npz'),
                    **{mode+'.'+k:v for mode,out in outputs.items() for k,v in out.items()})
                print(json.dumps(dict(name=row['name'], updated=updated, passed=row['passed'])), flush=True)
        passed = all(x['passed'] for x in rows)
        save_json(args.output/'summary.json', dict(passed=passed, signature=signature, cases=rows,
            cpu_oracle_sha256=digest(oracle_path), runtime_compilations=scope['events'], physics_integrations=0, newton_solves=0,
            performance=None, gpu_load_before=before, gpu_load_after=gpu_load_snapshot(),
            policy='independent FP64 tree/spatial-matrix formulas with componentwise gamma/majorant bounds; exact readonly holds'))
        print(json.dumps(dict(passed=passed, cases=len(rows), performance=None)), flush=True)
        return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
