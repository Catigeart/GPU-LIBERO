"""N=2 frozen time/overflow and exact public warmstart publication contracts."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_camlight import digest


def add_arguments(p):
    for name in ('build-root','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');return p


def cases():
    result=[]
    for name,sparse,rows,nefc,contacts,candidates,count in [('ordinary',False,5,[0,5],8,8,8),('all-overflows',False,5,[6,5],8,9,10),('sparse-last-row',True,5,[5,3],8,7,8),('sparse-overflow-precedence',True,5,[6,1],8,9,8),('empty',True,0,[0,1],0,0,1),('negative-count-no-read',True,5,[-3,0],8,0,0)]:
        for broadcast in (False,True):
            result.append(dict(name=name+('-broadcast' if broadcast else ''),sparse=sparse,rows=rows,nefc=np.array(nefc,np.int32),capacity=contacts,candidates=candidates,contacts=count,time=np.array([.13,1000.3],np.float32),timestep=np.array([.002] if broadcast else [.003,.001],np.float32),overflow=np.array([16,256],np.int32),rownnz=np.full((2,rows),3,np.int32),rowadr=np.tile(np.arange(rows,dtype=np.int32)*3,(2,1)),nonzeros=7,nv=57 if sparse else 289))
    return result


def construct(c,stream):
    import warp as wp
    a=lambda v,t:wp.array(np.ascontiguousarray(v),dtype=t,device=stream.device)
    m=NS(nv=c['nv'],na=0,nhistory=0,nactuator=9,is_sparse=c['sparse'],opt=NS(timestep=a(c['timestep'],float),warn_overflow=0,enableflags=0,disableflags=0))
    qacc=np.arange(2*m.nv,dtype=np.float32).reshape(2,m.nv)*.017
    d=NS(nworld=2,njmax=c['rows'],naconmax=c['capacity'],njmax_nnz=c['nonzeros'],time=a(c['time'],float),overflow=a(c['overflow'],int),nefc=a(c['nefc'],int),ncollision=a([c['candidates']],int),nacon=a([c['contacts']],int),efc=NS(J_rownnz=a(c['rownnz'] if c['sparse'] else np.empty((0,0),np.int32),int),J_rowadr=a(c['rowadr'] if c['sparse'] else np.empty((0,0),np.int32),int)),qacc=a(qacc,float),qacc_warmstart=a(np.full_like(qacc,-17),float))
    return m,d


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.backends.mjwarp.publication import Publication
        from libero.gpu.host.backends.mjwarp.euler_other import publish_advance
        self.c,self.stream=c,stream;self.owners={};self.graphs={}
        for mode in ('upstream','cuda'):
            m,d=construct(c,stream);self.owners[mode]=(m,d)
            if mode=='cuda':self.binding=Publication(m,d,stream,implementation='cuda',library=args.library)
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.activation();self.binding.publish()
                else:publish_advance(m,d)
            self.graphs[mode]=cap.graph
    def check(self,updated):
        import warp as wp
        c=self.c;time=c['time']*np.float32(.91) if updated else c['time'];dt=c['timestep']*np.float32(1.13) if updated else c['timestep'];count=c['nefc'][::-1].copy() if updated else c['nefc'];flags=c['overflow'].copy();checks=[];outputs={}
        for w in range(2):
            if count[w]>c['rows']:flags[w]|=1
            elif count[w]>0 and c['sparse']:
                i=count[w]-1
                if c['rownnz'][w,i]+c['rowadr'][w,i]>c['nonzeros']:flags[w]|=2
            if c['candidates']>c['capacity']:flags[w]|=4
            if c['contacts']>c['capacity']:flags[w]|=8
        expected_time=(time+dt[np.arange(2)%len(dt)]).astype(np.float32)
        for mode,(m,d) in self.owners.items():
            d.time.assign(time);m.opt.timestep.assign(dt);d.nefc.assign(count);d.overflow.assign(c['overflow']);d.qacc_warmstart.fill_(-17)
            if updated:d.qacc.fill_(.37)
            arrays={'qacc':d.qacc,'dt':m.opt.timestep,'nefc':d.nefc,'contact_count':d.nacon,'candidate_count':d.ncollision,'rownnz':d.efc.J_rownnz,'rowadr':d.efc.J_rowadr};before={k:v.numpy().view(np.uint8).copy() for k,v in arrays.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            tests=dict(time=np.array_equal(d.time.numpy(),expected_time),overflow=np.array_equal(d.overflow.numpy(),flags),warmstart=np.array_equal(d.qacc_warmstart.numpy().view(np.uint8),d.qacc.numpy().view(np.uint8)),readonly=all(np.array_equal(v.numpy().view(np.uint8),before[k]) for k,v in arrays.items()))
            checks.append(dict(mode=mode,**tests,passed=all(tests.values())));outputs[mode]=dict(time=d.time.numpy().tolist(),overflow=d.overflow.numpy().tolist())
        return dict(name=c['name'],updated=updated,checks=checks,outputs=outputs,passed=all(v['passed'] for v in checks))


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    files=source_files(include_publication=True);files={k:v for k,v in files.items() if k.startswith('upstream/') or 'mjwarp_publication' in k or 'MjwarpPublication' in k or k in ('host/backends/mjwarp/publication.py','host/backends/mjwarp/euler_other.py','host/runtime/dynamics/mjwarp_motion.py','project_source_identity')}
    return dict(sources=files,library=digest(args.library),tool=digest(__file__))


def main(argv=None,*,args=None):
    if args is None:args=add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    from libero.gpu import precision
    precision.select('fp32')
    import warp as wp
    from warp._src.build import init_kernel_cache
    from libero.gpu.host.backends.mjwarp.model import require_upstream
    from libero.gpu.host.build.compilation import compilation_scope,install_warp_guard
    require_upstream();init_kernel_cache(args.build_root/'mjwarp/warp-cache');install_warp_guard();wp.init();stream=wp.Stream('cuda:0');sig=signature(args);prep=args.output/'preparation.json'
    if not args.prepare and (not prep.is_file() or json.loads(prep.read_text())['signature']!=sig):raise RuntimeError('Prepare exact publication references offline first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases()]
        if args.prepare:
            wp.synchronize_stream(stream);prep.write_text(json.dumps(dict(signature=sig,compilation_events=scope['events']),indent=2)+'\n');print(json.dumps(dict(prepared=True,captures=24,compilations=len(scope['events']))),flush=True);return 0
        rows=[r.check(updated) for r in replays for updated in (False,True)];passed=all(r['passed'] for r in rows);(args.output/'summary.json').write_text(json.dumps(dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],worlds=2,physics_integrations=0,performance=None),indent=2)+'\n');print(json.dumps(dict(passed=passed,cases=len(rows),failures=[r for r in rows if not r['passed']])),flush=True);return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
