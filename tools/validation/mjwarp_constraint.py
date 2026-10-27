"""Local FP32 constraint projection/J-transpose replay, with explicit preparation.

The first preparation freezes two real upstream algebraic boundaries without
integrating physics. Subsequent preparation only captures reference graphs.
Ordinary correctness/performance cannot compile or recreate those boundaries.
"""
import argparse
import hashlib
import json
from pathlib import Path
from libero.gpu.host.build.task_catalog import load_catalog
from types import SimpleNamespace as NS

import numpy as np


def add_arguments(parser):
    for name in ('input', 'build-root', 'library', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    return parser


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(model, data, ctx):
    arrays = {name: getattr(data, name).numpy() for name in ('ne', 'nf', 'nefc', 'nacon')}
    arrays.update({'contact_' + name: getattr(data.contact, name).numpy()
                   for name in ('dim', 'friction', 'efc_address')})
    arrays.update({name: getattr(data.efc, name).numpy()
                   for name in ('type', 'id', 'D', 'frictionloss', 'J')})
    arrays.update(jaref=ctx.Jaref.numpy(), done=ctx.done.numpy(),
                  impratio=model.opt.impratio_invsqrt.numpy(), nv=np.array(model.nv))
    return arrays


def freeze_real(args):
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import solver, types
    from validation.mjwarp_newton import model_for, fields
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.mjwarp.model import build_scene, capacity_budget
    from libero.gpu.host.backends.mjwarp.build import source_files
    folder = args.output / 'frozen'
    folder.mkdir(exist_ok=True)
    receipt = folder / 'receipt.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['input_sha256'] != digest(args.input):
            raise ValueError('frozen constraint input belongs to another Newton boundary')
        for name, value in old['files'].items():
            if digest(folder / name) != value:
                raise ValueError('frozen constraint data changed: ' + name)
        return
    if not args.prepare:
        raise RuntimeError('freeze the constraint reference boundaries offline first')
    tasks = load_catalog(args.build_root)
    identities = {}
    for task_id in (4, 5):
        task = next(t for t in tasks if t['suite'] == 'libero_10' and t['task_id'] == task_id)
        if task_id == 4:
            host, model, identity = model_for(args.build_root, task)
            boundary = json.loads(args.input.with_name('summary.json').read_text())
            for key in ('scene_sha256', 'imported_xml_sha256', 'imported_model_sha256', 'mjwarp_commit'):
                if identity[key] != boundary['identity'][key]:
                    raise ValueError('frozen constraint model differs: ' + key)
            layout = boundary['frozen_newton']['layout']
            data = mjw.make_data(host, nworld=2, nconmax=layout['pooled_contacts']//2,
                nccdmax=layout['pooled_contacts']//2, njmax=layout['rows'])
            saved = dict(np.load(args.input, allow_pickle=False))
            for name, target in fields(data).items():
                if 'input.' + name in saved:
                    target.assign(saved['input.' + name])
            warm = not (model.opt.disableflags & types.DisableBit.WARMSTART)
            wp.launch(solver._solve_init_dof(warm, False), dim=(2, model.nv),
                inputs=[data.nefc, data.qacc_warmstart, data.qacc_smooth],
                outputs=[data.qacc, data.qfrc_constraint])
        else:
            built, model, bank, _, identity = build_scene(args.build_root,
                str(Path(get_libero_path('bddl_files')) / task['key']), 0,
                dict(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
                     hard_reset=False, ignore_done=True))
            host = built.native_model
            data = mjw.make_data(host, nworld=2, nccdmax=256, **capacity_budget(host))
            data.qpos.assign(bank[:2, 1:1+host.nq].astype(np.float32))
            data.qvel.assign(bank[:2, 1+host.nq:1+host.nq+host.nv].astype(np.float32))
            data.time.assign(bank[:2, 0].astype(np.float32))
        held = {name: getattr(data, name).numpy() for name in ('qpos', 'qvel', 'time')}
        if task_id == 5:
            mjw.forward(model, data)
        ctx = solver._create_solver_context(model, data)
        solver.init_context(model, data, ctx, grad=False)
        wp.synchronize()
        assert all(np.array_equal(getattr(data, name).numpy(), value) for name, value in held.items())
        np.savez(folder / f'task{task_id}.npz', **snapshot(model, data, ctx))
        identities[str(task_id)] = identity
    receipt.write_text(json.dumps(dict(input_sha256=digest(args.input),
        input_receipt_sha256=digest(args.input.with_name('summary.json')),
        files={f'task{t}.npz':digest(folder / f'task{t}.npz') for t in (4, 5)},
        model_identities=identities, reference_sources=source_files(), collector_sha256=digest(__file__),
        reference_forward_replays=1, physics_integrations=0), indent=2) + '\n')


def synthetic(nv, variant='ordinary'):
    rng = np.random.default_rng(17000 + nv)
    n, rows, cap = 2, 64, 6
    pad = ((nv+3)//4)*4 if nv <= 32 else ((nv+15)//16)*16
    a = dict(nv=np.array(nv), ne=np.array([2,1],np.int32), nf=np.array([3,2],np.int32),
        nefc=np.array([31,43],np.int32), nacon=np.array([cap],np.int32),
        contact_dim=np.zeros(cap,np.int32), contact_friction=np.tile([.7,.6,.05,.01,.01],(cap,1)).astype(np.float32),
        contact_efc_address=np.full((cap,6),-1,np.int32), type=np.full((n,rows),3,np.int32),
        id=np.zeros((n,rows),np.int32), D=rng.uniform(.2,3.,(n,rows)).astype(np.float32),
        frictionloss=rng.uniform(.1,.5,(n,rows)).astype(np.float32),
        J=rng.normal(0.,.3,(n,rows,pad)).astype(np.float32),
        jaref=rng.normal(0.,.8,(n,rows)).astype(np.float32), done=np.zeros(n,np.bool_),
        impratio=np.array([.2] if variant=='broadcast-impratio' else [.2,.3],np.float32))
    a['J'][:,:,nv:] = 71.  # padding is not a physical DOF
    for w in range(n):
        ne,nf = int(a['ne'][w]),int(a['nf'][w])
        a['type'][w,:ne] = 0
        a['type'][w,ne:ne+nf] = 1
        for k in range(nf):
            radius = a['frictionloss'][w,ne+k] / a['D'][w,ne+k]
            a['jaref'][w,ne+k] = (-2.,0.,2.)[k] * radius
            if variant=='friction-boundary':a['jaref'][w,ne+k] = (-1.,1.,-1.)[k] * radius
        head = ne + nf + 5
        for k,dim in enumerate((3,4,6)):
            c = 2*k+w  # deliberately interleaved world ownership
            ids = np.arange(head,head+dim)
            a['contact_dim'][c] = dim
            a['contact_efc_address'][c,:dim] = ids
            a['type'][w,ids] = 7
            a['id'][w,ids] = c
            f = a['contact_friction'][c]
            mu = float(f[0])*float(a['impratio'][w % a['impratio'].size])
            # CPU 2.3.7 regularization relation; tangent D is not independent.
            a['D'][w,ids[1:]] = a['D'][w,head]*f[:dim-1]**2/(mu*mu)
            tangent = np.linalg.norm(a['jaref'][w,ids[1:]].astype(np.float64)*f[:dim-1])
            a['jaref'][w,head] = (2.*tangent,-2.*tangent/(mu*mu),0.)[k]
            if variant=='top-boundary':a['jaref'][w,head] = tangent
            elif variant=='bottom-boundary':a['jaref'][w,head] = -tangent/(mu*mu)
            elif variant=='zero-tangent':
                a['jaref'][w,ids[1:]] = 0.;a['jaref'][w,head] = (-1.,0.,1.)[k]
            head += dim
        a['jaref'][w,ne+nf] = 0.
        a['J'][w,a['nefc'][w]:,:] = 100.
    if variant=='world1-done':a['done'][1]=True
    elif variant=='all-done':a['done'][:]=True
    elif variant=='empty-world0':a['ne'][0]=a['nf'][0]=a['nefc'][0]=0
    elif variant=='missing-contact':a['nacon'][:]=0
    elif variant=='missing-head':a['contact_efc_address'][:,0]=-1
    elif variant=='missing-tangent':a['contact_efc_address'][:,1]=-1
    return dict(name=f'nv{nv}-{variant}', origin='manufactured local input, including explicit guard cases', arrays=a)


def cases(args):
    from libero.gpu.host.runtime.newton.mjwarp_direction import FAMILIES
    result = []
    for t in (4,5):
        a = dict(np.load(args.output/'frozen'/f'task{t}.npz',allow_pickle=False))
        result.append(dict(name=f'task{t}-frozen-boundary',origin='raw upstream algebraic boundary',arrays=a))
    result += [synthetic(nv) for nv in FAMILIES]
    result += [synthetic(39,v) for v in ('top-boundary','bottom-boundary','friction-boundary',
        'zero-tangent','world1-done','all-done','empty-world0','missing-contact','missing-head',
        'missing-tangent','broadcast-impratio')]
    result += [synthetic(nv,v) for nv in (21,57) for v in ('world1-done','empty-world0')]
    return result


def reference(a):
    """FP64 grouped cone projection, independent of the CUDA per-row loop.

    CPU engine_core_constraint.c::mj_constraintUpdate defines the zones and
    scaled cone gradient. Near switching surfaces states are checked separately
    from continuous force error. Bounds depend only on the input, not outputs.
    """
    n, rows = a['jaref'].shape
    force = np.full((n,rows),37.,np.float64)
    state = np.ones((n,rows),np.int32)
    ambiguous = np.zeros((n,rows),np.bool_)
    held = np.ones((n,rows),np.bool_)
    scale = np.full((n,rows),1.e-30,np.float64)
    eps = np.finfo(np.float32).eps
    for w in range(n):
        if a['done'][w]:continue
        contacts = set()
        for row in range(int(a['nefc'][w])):
            x,d,loss = (float(a[k][w,row]) for k in ('jaref','D','frictionloss'))
            if a['type'][w,row]==7:
                contacts.add(int(a['id'][w,row]));continue
            held[w,row] = False
            scale[w,row] = max(abs(d*x)+abs(loss),1.e-30)
            if row < a['ne'][w]:force[w,row],state[w,row]=-d*x,1
            elif row < a['ne'][w]+a['nf'][w]:
                radius = loss/(d if d else 1.e-15)
                if x<=-radius:force[w,row],state[w,row]=loss,2
                elif x>=radius:force[w,row],state[w,row]=-loss,3
                else:force[w,row],state[w,row]=-d*x,1
                ambiguous[w,row] = min(abs(x-radius),abs(x+radius))<=32*eps*max(abs(x),abs(radius),1.e-30)
            else:force[w,row],state[w,row]=(0.,0) if x>=0 else (-d*x,1)
        for contact in contacts:
            if contact>=a['nacon'][0]:continue
            dim=int(a['contact_dim'][contact]);ids=a['contact_efc_address'][contact,:dim]
            if np.any(ids<0):continue
            held[w,ids]=False
            x=a['jaref'][w,ids].astype(np.float64);d=a['D'][w,ids].astype(np.float64)
            f=a['contact_friction'][contact,:dim-1].astype(np.float64)
            mu=float(f[0])*float(a['impratio'][w % a['impratio'].size])
            u=x[1:]*f;n0=x[0]*mu;t=float(np.linalg.norm(u))
            scale[w,ids]=np.maximum(abs(d*x),1.e-30)
            top=n0-mu*t;bottom=mu*n0+t
            ambiguous[w,ids]=min(abs(top),abs(bottom))<=32*eps*max(abs(n0),abs(mu*t),abs(mu*n0),t,1.e-30)
            if top>=0 or (t<=0 and n0>=0):force[w,ids]=0.;state[w,ids]=0
            elif bottom<=0 or (t<=0 and n0<0):force[w,ids]=-d*x;state[w,ids]=1
            else:
                denominator=mu*mu*(1.+mu*mu);dm=d[0]/(denominator if denominator else 1.e-15)
                normal=-dm*(n0-mu*t)*mu
                force[w,ids]=np.r_[normal,-normal*u*f/t];state[w,ids]=4
                normal_scale=abs(dm)*(abs(n0)+abs(mu*t))*abs(mu)
                scale[w,ids]=np.maximum(scale[w,ids],np.r_[normal_scale,normal_scale*abs(u*f)/t])
    return force,state,ambiguous,held,128.*eps*scale


class Replay:
    def __init__(self,case,library,stream):
        import warp as wp
        from mujoco_warp._src import types
        from libero.gpu.host.runtime.newton.mjwarp_constraint import ConstraintPlan
        self.case,self.stream=case,stream
        a=case['arrays'];nv=int(a['nv']);n,rows,pad=a['J'].shape;cap=a['contact_dim'].size
        dev={k:wp.array(np.ascontiguousarray(v),dtype=types.vec5 if k=='contact_friction' else
            wp.bool if v.dtype==np.bool_ else wp.int32 if v.dtype==np.int32 else wp.float32,device=stream.device)
            for k,v in a.items() if k!='nv'}
        self.inputs,self.model=dev,NS(nv=nv,nv_pad=pad,is_sparse=False,opt=NS(
            impratio_invsqrt=dev['impratio'],cone=types.ConeType.ELLIPTIC,solver=types.SolverType.NEWTON,
            integrator=types.IntegratorType.EULER,enableflags=0))
        self.data,self.contexts,self.graphs={}, {}, {}
        for mode in ('upstream','cuda'):
            d=NS(nworld=n,njmax=rows,naconmax=cap,ne=dev['ne'],nf=dev['nf'],nefc=dev['nefc'],nacon=dev['nacon'],
                qfrc_constraint=wp.full((n,nv),-9.,dtype=float,device=stream.device),
                contact=NS(dim=dev['contact_dim'],friction=dev['contact_friction'],efc_address=dev['contact_efc_address']),
                efc=NS(J=dev['J'],D=dev['D'],frictionloss=dev['frictionloss'],type=dev['type'],id=dev['id'],
                    force=wp.full((n,rows),37.,dtype=float,device=stream.device),
                    state=wp.full((n,rows),1,dtype=int,device=stream.device)))
            ctx=NS(Jaref=dev['jaref'],done=dev['done'],ls_exhausted=wp.zeros(n,dtype=bool,device=stream.device),
                compact_d_full=None,compact_m_full=None,
                quad_changed_ids=wp.empty((n,0),dtype=int,device=stream.device),
                quad_changed_count=wp.empty(0,dtype=int,device=stream.device),
                state_changed_count=wp.empty(0,dtype=int,device=stream.device))
            self.data[mode],self.contexts[mode]=d,ctx
            if mode=='cuda':
                self.plan=ConstraintPlan(self.model,d,library,stream);self.binding=self.plan.bind(ctx)
            with wp.ScopedCapture(stream=stream) as capture:
                for _ in range(32):self.enqueue(mode)
            self.graphs[mode]=capture.graph

    def enqueue(self,mode):
        from mujoco_warp._src import solver
        if mode=='cuda':self.binding.enqueue()
        else:solver._update_constraint(self.model,self.data[mode],self.contexts[mode],False,False)

    def check(self):
        import warp as wp
        a=self.case['arrays'];nv=int(a['nv'])
        for graph in self.graphs.values():wp.capture_launch(graph,stream=self.stream)
        wp.synchronize_stream(self.stream)
        outputs={mode:dict(force=d.efc.force.numpy(),state=d.efc.state.numpy(),qfrc=d.qfrc_constraint.numpy())
                 for mode,d in self.data.items()}
        ref,states,ambiguous,held,bound=reference(a);checks=[]
        for mode,out in outputs.items():
            force_error=abs(out['force'].astype(np.float64)-ref)
            kept=bool(np.all(out['force'][held]==37.) and np.all(out['state'][held]==1))
            allowed_boundary=np.where(a['type']==7,np.isin(out['state'],(0,1,4)),
                                      np.isin(out['state'],(1,2,3)))
            state_ok=bool(np.all(out['state'][~ambiguous]==states[~ambiguous]) and
                          np.all(allowed_boundary[ambiguous]))
            worlds=[]
            for w in range(2):
                if a['done'][w]:
                    worlds.append(dict(world=w,held=bool(np.all(out['qfrc'][w]==-9.)),passed=bool(np.all(out['qfrc'][w]==-9.))))
                    continue
                used=int(a['nefc'][w]);j=a['J'][w,:used,:nv].astype(np.float64)
                actual_force=out['force'][w,:used].astype(np.float64)
                exact=j.T@actual_force;majorant=abs(j).T@abs(actual_force)
                gamma=used*np.finfo(np.float32).eps/(1.-used*np.finfo(np.float32).eps)
                limit=8.*gamma*np.maximum(majorant,1.e-30)
                error=abs(out['qfrc'][w].astype(np.float64)-exact)
                worlds.append(dict(world=w,max_error=float(error.max()),max_bound=float(limit.max()),
                    passed=bool(np.all(error<=limit))))
            checks.append(dict(mode=mode,force_max_error=float(force_error.max()),
                force_bound_fraction=float(np.max(force_error/np.maximum(bound,1.e-30))),
                state_ok=state_ok,held=kept,worlds=worlds,
                passed=bool(np.all(force_error<=bound) and state_ok and kept and all(w['passed'] for w in worlds))))
        holds=all(np.array_equal(v.numpy().view(np.uint8),a[k].view(np.uint8)) for k,v in self.inputs.items())
        result=dict(name=self.case['name'],origin=self.case['origin'],nv=nv,worlds=2,
            rows=a['nefc'].tolist(),ambiguous_rows=int(ambiguous.sum()),checks=checks,input_holds=holds,
            state_differences=int(np.count_nonzero(outputs['cuda']['state']!=outputs['upstream']['state'])),
            force_difference=float(abs(outputs['cuda']['force']-outputs['upstream']['force']).max()),
            qfrc_difference=float(abs(outputs['cuda']['qfrc']-outputs['upstream']['qfrc']).max()),
            passed=bool(holds and all(c['passed'] for c in checks)))
        return result,outputs

    def benchmark(self,args):
        import warp as wp
        result={}
        for mode,graph in self.graphs.items():
            for _ in range(args.warmup):wp.capture_launch(graph,stream=self.stream)
            values=[];start=wp.Event(enable_timing=True);end=wp.Event(enable_timing=True)
            for _ in range(args.steps):
                self.stream.record_event(start);wp.capture_launch(graph,stream=self.stream);self.stream.record_event(end)
                wp.synchronize_event(end)
                values.append(wp.get_event_elapsed_time(start,end,synchronize=False)*1000/32)
            result[mode]=dict(mean_us=float(np.mean(values)),std_us=float(np.std(values)),microseconds=values)
        return dict(name=self.case['name'],rows=self.case['arrays']['nefc'].tolist(),
            calls_per_sample=32,warmup=args.warmup,samples=args.steps,**result,
            speedup=result['upstream']['mean_us']/result['cuda']['mean_us'])


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    if args.warmup<0 or args.steps<1:raise ValueError('invalid timing counts')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.build import source_files
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from validation.common import save_json
    from validation.runner import gpu_load_snapshot
    args.output.mkdir(parents=True,exist_ok=True)
    init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init()
    stream=wp.Stream('cuda:0')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        freeze_real(args)
        signature=dict(sources=source_files(include_constraint=True),input_sha256=digest(args.input),
            library_sha256=digest(args.library),tool_sha256=digest(__file__),
            frozen_receipt_sha256=digest(args.output/'frozen/receipt.json'))
        prepared=args.output/'preparation.json'
        if not args.prepare and (not prepared.is_file() or json.loads(prepared.read_text())['signature']!=signature):
            raise RuntimeError('prepare these exact constraint graphs offline first')
        replays=[Replay(c,args.library,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream)
            save_json(prepared,dict(signature=signature,compilation_events=scope['events'],physics_integrations=0,
                local_graph_replays=0,cases=len(replays)))
            print(json.dumps(dict(prepared=True,cases=len(replays),compilations=len(scope['events']))),flush=True);return 0
        before=gpu_load_snapshot();rows=[]
        for replay in replays:
            row,outputs=replay.check();rows.append(row)
            np.savez(args.output/(row['name']+'.npz'),**{mode+'.'+k:v for mode,out in outputs.items() for k,v in out.items()})
            print(json.dumps(dict(name=row['name'],passed=row['passed'])),flush=True)
        passed=all(r['passed'] for r in rows)
        timing=[r.benchmark(args) for r in replays if r.case['name'].startswith('task')] if passed and args.performance else None
        save_json(args.output/'summary.json',dict(passed=passed,signature=signature,cases=rows,performance=timing,
            runtime_compilations=scope['events'],physics_integrations=0,gpu_load_before=before,gpu_load_after=gpu_load_snapshot(),
            policy='FP64 grouped cone projection; force 128eps input majorant; JTF 8gamma_rows absolute product majorant; exact holds'))
        print(json.dumps(dict(passed=passed,performance=timing)),flush=True)
        return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
