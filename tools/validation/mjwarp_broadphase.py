"""Frozen N=2 NXN broadphase: bounded decisions and exact pooled ownership."""
import argparse,ctypes as ct,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_broadphase_reference import reference,geometry

MODEL=(('geom_dataid','geom_dataid','int32'),('aabb','geom_aabb','vec3'),('rbound','geom_rbound','float32'),('margin','geom_margin','float32'),
    ('gap','geom_gap','float32'),('pair_margin','pair_margin','float32'),('pair_gap','pair_gap','float32'))
OUT=(('pairs','collision_pair','vec2i'),('ids','collision_pairid','vec2i'),('world','collision_worldid','int32'))


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def helpers():return {'mjwarp_motion_reference.py':digest(Path(__file__).with_name('mjwarp_motion_reference.py'))}
def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}
def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true');return p


def cases(args):
    rows=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen broadphase input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for k in ('ngeom','npair','nsensorcollision','capacity','filter'):c[k]=int(c[k])
        c['name']=p.stem;rows.append(c)
    if not rows:raise ValueError('No frozen broadphase inputs')
    return rows


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['rbound']*=np.float32(.93);c['margin']*=np.float32(1.11);c['gap']*=np.float32(.87)
        c['pair_margin']*=np.float32(1.17);c['pair_gap']*=np.float32(.91)
        c['aabb'][:,:,0]+=np.array([.007,-.011,.013],np.float32);c['aabb'][:,:,1]*=np.float32(.97)
        c['xpos']+=np.array([.023,-.017,.031],np.float32);c['xpos'][1]*=np.float32(.94)
        c['xmat']=c['xmat'][...,[1,2,0]].copy()
        if len(c['geom_dataid'])>1:c['geom_dataid']=c['geom_dataid'][::-1].copy()
    return c


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    arr=lambda value,kind:wp.array(np.ascontiguousarray(value),dtype=getattr(wp,kind),device=stream.device)
    m=NS(ngeom=c['ngeom'],npair=c['npair'],nsensorcollision=c['nsensorcollision'],nflex=0,nbody=1,
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=0,broadphase=types.BroadphaseType.NXN,broadphase_filter=c['filter']),
        geom_type=arr(c['geom_type'],'int32'),geom_bodyid=arr(np.zeros(c['ngeom'],np.int32),'int32'),
        nxn_geom_pair_filtered=arr(c['pairs_in'],'vec2i'),nxn_pairid_filtered=arr(c['pair_ids_in'],'vec2i'))
    for key,field,kind in MODEL:setattr(m,field,arr(c[key],kind))
    d=NS(nworld=2,naconmax=c['capacity'],ncollision=arr(c['initial_count'],'int32'),geom_xpos=arr(c['xpos'],'vec3'),geom_xmat=arr(c['xmat'],'mat33'),body_awake=arr(np.zeros((2,1),np.int32),'int32'))
    context=NS(**{field:arr(c['initial_'+key],kind) for key,field,kind in OUT})
    return m,d,context


def upstream(m,d,context):
    import warp as wp
    from mujoco_warp._src import collision_driver as q
    # Exact numerical kernel of nxn_broadphase, without its optional sleep
    # conditional or occupancy-query host helper; admitted callers have no sleep.
    kernel=q._nxn_broadphase(m.opt.broadphase_filter,m.geom_aabb.shape[0],m.geom_rbound.shape[0],m.geom_margin.shape[0],m.geom_gap.shape[0],m.pair_margin.shape[0],m.pair_gap.shape[0],m.geom_dataid.shape[0],False,False)
    wp.launch(kernel,dim=(2,m.nxn_geom_pair_filtered.shape[0]),block_dim=128,
        inputs=[m.geom_type,m.geom_bodyid,m.geom_dataid,m.geom_aabb,m.geom_rbound,m.geom_margin,m.geom_gap,m.pair_margin,m.pair_gap,m.nxn_geom_pair_filtered,m.nxn_pairid_filtered,d.geom_xpos,d.geom_xmat,d.body_awake,d.naconmax,d.body_awake],
        outputs=[d.ncollision,context.collision_pair,context.collision_pairid,context.collision_worldid])


def assign(c,m,d,context):
    for key,field,_ in MODEL:getattr(m,field).assign(c[key])
    d.geom_xpos.assign(c['xpos']);d.geom_xmat.assign(c['xmat']);d.ncollision.assign(c['initial_count'])
    for key,field,_ in OUT:getattr(context,field).assign(c['initial_'+key])


def check_output(c,got,*,expected=None):
    must,possible,ambiguous=reference(c) if expected is None else expected
    initial=int(c['initial_count'][0]);total=int(got['count'][0]);cap=c['capacity'];begin=min(initial,cap);end=min(total,cap)
    values=[(int(got['world'][i]),*map(int,got['pairs'][i]),*map(int,got['ids'][i])) for i in range(begin,end)]
    actual=set(values);unwritten=True
    for key in ('pairs','ids','world'):
        unwritten &= np.array_equal(got[key][:begin].view(np.uint8),c['initial_'+key][:begin].view(np.uint8))
        unwritten &= np.array_equal(got[key][end:].view(np.uint8),c['initial_'+key][end:].view(np.uint8))
    checks=[dict(name='attempted count within independent decision bounds',passed=initial+len(must)<=total<=initial+len(possible)),
        dict(name='unique valid pair/world/copied metadata ownership',passed=len(values)==len(actual) and actual<=possible),
        dict(name='all unambiguous pairs emitted without overflow',passed=total>cap or must<=actual),dict(name='prefix and unwritten tail bit holds',passed=bool(unwritten))]
    return dict(passed=all(x['passed'] for x in checks),contracts=checks,count=total,stored=len(values),ambiguous_decisions=ambiguous,accepted=sorted(actual))


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_broadphase import BroadphaseNumerics,qualify
        self.case,self.stream=c,stream;self.models={};self.datas={};self.contexts={};self.graphs={}
        for mode in ('upstream','cuda'):
            m,d,context=construct(c,stream);self.models[mode]=m;self.datas[mode]=d;self.contexts[mode]=context
            if mode=='cuda':self.binding=BroadphaseNumerics(m,d,context,args.library,stream,admission=qualify(m,d,implementation='cuda'))
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(m,d,context)
            self.graphs[mode]=cap.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);expected=reference(c);rows=[]
        for mode in ('upstream','cuda'):
            m,d,context=self.models[mode],self.datas[mode],self.contexts[mode];assign(c,m,d,context)
            readonly={key:getattr(m,field) for key,field,_ in MODEL};readonly.update({key:getattr(m,key) for key in ('geom_type','nxn_geom_pair_filtered','nxn_pairid_filtered','geom_bodyid')});readonly.update(xpos=d.geom_xpos,xmat=d.geom_xmat,awake=d.body_awake)
            before={key:a.numpy().view(np.uint8).copy() for key,a in readonly.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            got={key:getattr(context,field).numpy() for key,field,_ in OUT};got['count']=d.ncollision.numpy()
            row=check_output(c,got,expected=expected);held=all(np.array_equal(a.numpy().view(np.uint8),before[key]) for key,a in readonly.items());row.update(mode=mode,inputs_held=held);row['passed'] &= held;rows.append(row)
        a,b=(set(tuple(v) for v in x['accepted']) for x in rows)
        # Without overflow, every difference must be confined to predicates
        # whose input-derived FP32 error intervals cross a decision boundary.
        comparable=all(x['count']<=c['capacity'] for x in rows)
        difference_ok=not comparable or (a^b)<=(expected[1]-expected[0])
        for row in rows:row.pop('accepted')
        return dict(name=c['name'],updated=updated,modes=rows,raw_set_difference=len(a^b),difference_within_bounds=bool(difference_ok),passed=bool(difference_ok and all(x['passed'] for x in rows)))


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_broadphase=True)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'broadphase' in k or 'Broadphase' in k or k in ('host/runtime/dynamics/mjwarp_motion.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=sources,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_broadphase_reference.py')),helpers=helpers())


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Broadphase CPU oracle requires CPU reference2.3.7')
    path=next(Path(mujoco.__file__).parent.glob('libmujoco.so*'));library=ct.CDLL(str(path));function=library.mj_collideOBB
    pointer=ct.POINTER(ct.c_double);function.argtypes=[pointer]*8+[ct.POINTER(ct.c_ubyte)];function.restype=ct.c_int
    rows=[];near=0
    for original in cases(args):
        for updated in (False,True):
            c=current_case(original,updated)
            for w in range(2):
                g1,g2,x1,x2,R1,R2,b1,b2,r1,r2,m1,m2=geometry(c,w)
                selected=np.flatnonzero((r1!=0)&(r2!=0)&(m1[0]+m2[0]==0)&np.all(b1[:,1]<1.e10,axis=1)&np.all(b2[:,1]<1.e10,axis=1))
                for i in selected:
                    values=[np.ascontiguousarray(a[i],dtype=float).ravel() for a in (b1,b2,x1,R1,x2,R2)]
                    got=function(*(a.ctypes.data_as(pointer) for a in values),None,None,None)
                    centers=[R1[i]@b1[i,0]+x1[i],R2[i]@b2[i,0]+x2[i]];normals=[R1[i,:,k] for k in range(3)]+[R2[i,:,k] for k in range(3)];residual=[];uncertain=False
                    for normal in normals:
                        p=[np.dot(center,normal) for center in centers]
                        radius=[sum(abs(size[j]*np.dot(matrix[:,j],normal)) for j in range(3)) for matrix,size in ((R1[i],b1[i,1]),(R2[i],b2[i,1]))]
                        value=radius[0]+radius[1]-abs(p[1]-p[0]);residual.append(value)
                        uncertain |= abs(value)<=4096*np.finfo(float).eps*(1.+abs(p[0])+abs(p[1])+radius[0]+radius[1])
                    expected=int(min(residual)>=0.);near+=int(uncertain)
                    rows.append(dict(name=c['name'],updated=updated,world=w,pair=int(i),near_fp64_boundary=bool(uncertain),passed=bool(got in (0,1) and (got==expected or uncertain))))
    if not rows:raise RuntimeError('CPU OBB oracle must exercise nonempty zero-margin cases')
    result=dict(passed=all(x['passed'] for x in rows),checks=rows,near_boundary_checks=near,version=mujoco.__version__,cpu_library=str(path),cpu_library_sha256=digest(path),inputs=input_identity(args),tool_sha256=digest(__file__),reference_sha256=digest(Path(__file__).with_name('mjwarp_broadphase_reference.py')),helpers=helpers(),physics_integrations=0,performance=None,
        domain='actual exported CPU2.3.7 mj_collideOBB, finite non-plane zero-total-margin raw frames; other bounds against independent formulas and pinned GPU')
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],checks=len(rows),near_boundaries=near)),flush=True);return 0 if result['passed'] else 2


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cpu_reference:return cpu_reference(args)
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    from validation.common import save_json
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare:
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact broadphase operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256','helpers')):raise RuntimeError('Prepare matching CPU broadphase oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prep,dict(signature=sig,compilation_events=scope['events'],physics_integrations=0));print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row=replay.check(updated);rows.append(row);print(json.dumps({k:row[k] for k in ('name','updated','passed')}),flush=True)
                if not row['passed']:break
            if not row['passed']:break
        passed=all(x['passed'] for x in rows);save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],physics_integrations=0,performance=None,worlds=2,
            policy='independent vectorized FP64 plane/sphere/corner-AABB/six-face OBB inequalities, propagated FP32 fixed8x decision bounds; exact metadata/pooled ownership/prefix/tail holds'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
