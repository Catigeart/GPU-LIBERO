"""Qualified rigid convex collision production ABI, frozen N=2 pooled contacts."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from . import mjwarp_plane as plane
from .mjwarp_plane import add_arguments,cases,digest,PARAMETERS,CONTACTS,STATIC
from .mjwarp_plane_reference import material,U


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    m,d,ctx=plane.construct(c,stream);m.geom_pair_type_count=tuple(map(int,c['type_counts']));m.has_boxbox_margin=bool(c['primitive_boxbox']);m.npolygonmax=4;m.nmeshdegmax=3
    m.block_dim=NS(convex_ccd=128);m.opt.disableflags=int(types.DisableBit.MULTICCD);m.opt.ccd_iterations=int(c['iterations']);m.opt.warn_overflow=0;m.opt.ccd_tolerance=wp.array(c['tolerance'],dtype=wp.float32,device=stream.device)
    d.naccdmax=int(c['ccd_capacity']);d.overflow=wp.array(c['initial_overflow'],dtype=wp.int32,device=stream.device);return m,d,ctx


def assign(c,m,d,ctx):
    plane.assign(c,m,d,ctx);m.opt.ccd_tolerance.assign(c['tolerance']);d.overflow.assign(c['initial_overflow'])


def upstream(m,d,ctx):
    from mujoco_warp._src import collision_convex,types
    from libero.gpu.host.runtime.collision.mjwarp_convex_collision import FAMILIES
    table=[(types.GeomType(a),types.GeomType(b)) for a,b in FAMILIES if (a,b)!=(6,6) or not m.has_boxbox_margin]
    collision_convex.convex_narrowphase(m,d,ctx,table)


def check(c,got):
    initial=int(c['initial_count'][0]);count=int(got['count'][0]);cap=c['capacity'];begin=min(initial,cap);end=min(count,cap);rows=[];keys=[];lookup={}
    for i in range(min(int(c['candidate_count'][0]),cap)):
        w=int(c['candidate_world'][i]);a,b=map(int,c['candidate_geom'][i]);lookup[w,a,b]=tuple(map(int,c['candidate_ids'][i]))
    def add(name,passed,**v):rows.append(dict(name=name,passed=bool(passed),**v))
    add('monotonic attempted append count',count>=initial)
    for key,_ in CONTACTS:
        add('prefix/unwritten exact hold '+key,np.array_equal(got[key][:begin].view(np.uint8),c['initial_'+key][:begin].view(np.uint8)) and np.array_equal(got[key][end:].view(np.uint8),c['initial_'+key][end:].view(np.uint8)))
    scale=1+max(np.max(abs(c['xpos'])),np.max(abs(c['size'])),np.max(abs(c['mesh_vert'])));budget=float(8*(512*U/(1-512*U))*scale)
    for i in range(begin,end):
        w=int(got['worldid'][i]);a,b=map(int,got['geom'][i]);key=(w,a,b);local=int(got['geomcollisionid'][i]);keys.append((*key,local));valid=key in lookup
        add('candidate/world ownership',valid,row=i)
        if not valid:continue
        pair,sensor=lookup[key];p=material(c,w,a,b,pair);distance=float(got['dist'][i]);frame=got['frame'][i].astype(float);position=got['pos'][i].astype(float);kinds=tuple(map(int,c['geom_type'][[a,b]]));add('local manifold index range',0<=local<=(3 if kinds==(6,6) else 0),row=i)
        add('finite distance/position/frame',np.isfinite(distance) and np.all(np.isfinite(position)) and np.all(np.isfinite(frame)),row=i)
        if np.linalg.norm(frame[0])>budget:add('orthonormal contact frame',np.max(abs(frame@frame.T-np.eye(3)))<=budget and np.linalg.det(frame)>0,row=i)
        else:add('zero normal frame convention',np.max(abs(frame))<=budget,row=i)
        add('constraint address reset',np.all(got['efc_address'][i]==-1),row=i)
        for name in ('solref','solreffriction','solimp','friction','adhesion'):
            error=float(np.max(abs(got[name][i].astype(float)-p[name])));add('independent material '+name,error<=budget*(1+np.max(abs(p[name]))),row=i,maximum_error=error)
        add('current includemargin',abs(distance*0+float(got['includemargin'][i])-float(p['margin']))<=budget,row=i)
        near=abs(distance-float(p['margin']))<=budget or abs(distance-float(p['margin']+p['gap']))<=budget
        detected=distance<float(p['margin']+p['gap']);expected=(1 if pair>=-1 and detected else 0)|(2 if sensor>=0 else 0);dim=1 if p['adhesion']!=0 and distance>=p['margin'] else p['dim']
        add('contact type/adhesion dimension',near or (int(got['type'][i])==expected and int(got['dim'][i])==dim),row=i)
    add('unique pair/local contact ownership',len(keys)==len(set(keys)))
    add('overflow flags preserve incoming bits',np.all((got['overflow']&c['initial_overflow'])==c['initial_overflow']))
    return dict(passed=all(v['passed'] for v in rows),checks=rows,count=count,budget=budget)


def compare(c,outputs,budget):
    a,b=outputs;cap=c['capacity'];start=min(int(c['initial_count'][0]),cap);overflow=any(int(v['count'][0])>cap for v in outputs);rows=[]
    def add(name,passed,**v):rows.append(dict(name=name,passed=bool(passed),**v))
    add('same attempted contact count',int(a['count'][0])==int(b['count'][0]));add('same overflow masks',np.array_equal(a['overflow'],b['overflow']))
    def groups(got):
        result={}
        for i in range(start,min(int(got['count'][0]),cap)):result.setdefault((int(got['worldid'][i]),*map(int,got['geom'][i])),[]).append(i)
        return result
    x,y=groups(a),groups(b)
    if not overflow:add('same emitting world/pair set',x.keys()==y.keys())
    for key in x.keys()&y.keys():
        ia,ib=x[key],y[key];add('same per-pair manifold count',len(ia)==len(ib),key=key)
        if len(ia)!=len(ib):continue
        # Match geometry in this tiny manifold; contact order can rotate with
        # a legal EPA feature choice. Local IDs remain complete0..n-1 per pair.
        from itertools import permutations
        pa=a['pos'][ia].astype(float);pb=b['pos'][ib].astype(float);best=min(permutations(range(len(ib))),key=lambda order:np.max(np.linalg.norm(pa-pb[list(order)],axis=1)))
        for i,j in zip(ia,np.array(ib)[list(best)]):
            for name,_ in CONTACTS:
                if name=='geomcollisionid':continue
                va,vb=a[name][i],b[name][j];error=float(np.max(abs(va.astype(float)-vb)));exact=va.dtype.kind in 'iu';add('pinned contact '+name,error<=(0 if exact else budget*(1+np.max(abs(va.astype(float))))),key=key,maximum_error=error)
        if not overflow:add('complete local manifold IDs',set(map(int,a['geomcollisionid'][ia]))==set(range(len(ia))) and set(map(int,b['geomcollisionid'][ib]))==set(range(len(ib))),key=key)
    return dict(passed=all(v['passed'] for v in rows),checks=rows)


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.collision.mjwarp_convex_collision import ConvexContacts,qualify
        self.case,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            m,d,ctx=construct(c,stream);self.owners[mode]=(m,d,ctx)
            if mode=='cuda':
                self.binding=ConvexContacts(m,d,ctx,args.library,stream,admission=qualify(m,d,implementation='cuda'));self.pointers={k:v.ptr for k,v in self.binding.arrays.items()}
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:upstream(m,d,ctx)
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        c=plane.current_case(self.case,updated);rows=[];values=[]
        for mode,(m,d,ctx) in self.owners.items():
            assign(c,m,d,ctx);readonly={key:getattr(m,field) for key,field,_ in PARAMETERS};readonly.update({key:getattr(m,{'condim':'geom_condim','priority':'geom_priority'}.get(key,key)) for key in STATIC});readonly.update(xpos=d.geom_xpos,xmat=d.geom_xmat,tolerance=m.opt.ccd_tolerance,candidate_count=d.ncollision,**vars(ctx));before={k:v.numpy().view(np.uint8).copy() for k,v in readonly.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream);got={k:getattr(d.contact,k).numpy() for k,_ in CONTACTS};got.update(count=d.nacon.numpy(),overflow=d.overflow.numpy());values.append(got);row=check(c,got);held=all(np.array_equal(v.numpy().view(np.uint8),before[k]) for k,v in readonly.items());row.update(mode=mode,readonly_inputs_held=held);row['passed'] &=held;rows.append(row)
        comparison=compare(c,values,2*max(row['budget'] for row in rows));held=all(self.binding.arrays[k].ptr==ptr for k,ptr in self.pointers.items());return dict(name=c['name'],updated=updated,modes=rows,comparison=comparison,retained_workspace_pointers=held,passed=held and comparison['passed'] and all(v['passed'] for v in rows)),values


def signature(args):
    from .mjwarp_convex_gjk import signature as prior
    s=prior(args);s.update(tool_sha256=digest(__file__),plane_helper_sha256=digest(Path(plane.__file__)));return s


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cpu_reference:raise ValueError('Qualified building-block CPU/independent evidence is retained in their own probes; this checks the pooled production caller')
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare and (not prep.is_file() or json.loads(prep.read_text())['signature']!=sig):raise RuntimeError('Prepare exact convex production operators offline first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events'],physics_integrations=0),indent=2)+'\n');print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row,values=replay.check(updated);rows.append(row);failed=[dict(mode=m['mode'],**v) for m in row['modes'] for v in m['checks'] if not v['passed']];print(json.dumps(dict(name=row['name'],updated=updated,passed=row['passed'],counts=[m['count'] for m in row['modes']],failures=failed[:6],parity_failures=[v for v in row['comparison']['checks'] if not v['passed']][:6])),flush=True)
                if not row['passed']:np.savez(args.output/'failure.npz',**{mode+'.'+k:v for mode,got in zip(('upstream','cuda'),values) for k,v in got.items()});break
            if not row['passed']:break
        passed=all(v['passed'] for v in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
