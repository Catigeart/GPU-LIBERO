"""Frozen N=2 row construction: exact ownership/holds, bounded row mathematics."""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from .mjwarp_constraint_build_reference import build_reference
from .mjwarp_motion_reference import check_arrays

PARAMETERS=(('timestep','opt.timestep','float32',1,1),('impratio_invsqrt','opt.impratio_invsqrt','float32',1,1),
    ('body_weight','body_invweight0','vec2','nbody',2),('dof_weight','dof_invweight0','float32','nv',1),
    ('dof_friction','dof_frictionloss','float32','nv',1),('dof_ref','dof_solref','vec2','nv',2),('dof_imp','dof_solimp','vec5','nv',5),
    ('joint_ref','jnt_solref','vec2','njnt',2),('joint_imp','jnt_solimp','vec5','njnt',5),
    ('joint_range','jnt_range','vec2','njnt',2),('joint_margin','jnt_margin','float32','njnt',1))

CONTACTS=(('contact_dim','dim','int32'),('contact_world','worldid','int32'),('contact_geom','geom','vec2i'),('contact_type','type','int32'),
    ('contact_dist','dist','float32'),('contact_margin','includemargin','float32'),('contact_pos','pos','vec3'),('contact_frame','frame','mat33'),
    ('contact_friction','friction','vec5'),('contact_ref','solref','vec2'),('contact_ref_friction','solreffriction','vec2'),
    ('contact_imp','solimp','vec5'),('contact_adhesion','adhesion','float32'))

ROWS=(('row_type','type','int32'),('row_id','id','int32'),('row_pos','pos','float32'),('row_margin','margin','float32'),
    ('row_D','D','float32'),('row_vel','vel','float32'),('row_aref','aref','float32'),('row_friction','frictionloss','float32'))


def parameter(model,path):
    for key in path.split('.'):model=getattr(model,key)
    return model


STATIC=(('parent','body_parentid'),('root','body_rootid'),('geom_body','geom_bodyid'),('ancestor','body_isdofancestor'),
    ('joint_type','jnt_type'),('joint_qpos','jnt_qposadr'),('joint_dof','jnt_dofadr'),('dof_body','dof_bodyid'),
    ('limited_joint','jnt_limited_slide_hinge_adr'),('weld','body_weldid'),('body_dnum','body_dofnum'),('body_dadr','body_dofadr'),('dof_parent','dof_parentid'))
FLAGS=('constraint','friction','limit','contact','refsafe')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def helpers():return {'mjwarp_motion_reference.py':digest(Path(__file__).with_name('mjwarp_motion_reference.py'))}
def input_identity(args):return {p.name:digest(p) for p in sorted(args.inputs.glob('*')) if p.is_file()}
def add_arguments(p):
    for name in ('build-root','inputs','output','library'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--cpu-reference',action='store_true')
    return p


def cases(args):
    result=[]
    for p in sorted(args.inputs.glob('*.npz')):
        if digest(p)!=json.loads(p.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen row-construction input changed')
        with np.load(p,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        for k in ('nv','nq','nbody','njnt','ngeom','padded_nv','padded_rows','naconmax','njmax','nmaxcondim','cone'):c[k]=int(c[k])
        for k in (*('disable_'+f for f in FLAGS),'adhesion_enabled'):c[k]=bool(c[k])
        c['name']=p.stem;result.append(c)
    if not result:raise ValueError('No frozen constraint-construction inputs')
    return result


def current_case(original,updated):
    c={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
    if updated:
        c['qvel']*=np.array([-.83,1.17],np.float32)[:,None];c['cdof']*=np.float32(.97)
        c['subtree_com']+=np.array([.011,-.017,.007],np.float32)
        c['body_weight']*=np.float32(1.07);c['dof_weight']*=np.float32(.93)
        c['dof_friction']*=np.float32(1.13);c['dof_ref']*=np.float32(1.03);c['joint_ref']*=np.float32(.91)
        c['impratio_invsqrt']*=np.float32(.87);c['timestep']*=np.float32(1.01)
        n=min(int(c['contact_count'][0]),c['naconmax'])
        c['contact_count'][0]=(n+1)//2;c['contact_world'][:n]=1-c['contact_world'][:n]
        c['contact_pos'][:n]+=np.array([.023,-.013,.019],np.float32);c['contact_frame'][:n]=c['contact_frame'][:n,...,[1,2,0]].copy()
        c['contact_friction'][:n]*=np.float32(.93);c['contact_ref'][:n]*=np.float32(1.11)
        c['contact_adhesion'][:n]*=np.float32(.89)
    return c


def construct(c,stream):
    import warp as wp
    from mujoco_warp._src import types
    dtype=lambda name:types.vec5 if name=='vec5' else getattr(wp,name)
    arr=lambda value,kind:wp.array(np.ascontiguousarray(value),dtype=dtype(kind),device=stream.device)
    flags=sum(int(getattr(types.DisableBit,{'friction':'FRICTIONLOSS'}.get(k,k.upper()))) for k in FLAGS if c['disable_'+k])
    m=NS(nv=c['nv'],nq=c['nq'],nbody=c['nbody'],njnt=c['njnt'],ngeom=c['ngeom'],nv_pad=c['padded_nv'],nmaxcondim=c['nmaxcondim'],
        is_sparse=False,neq=0,ntendon=0,nflex=0,nflexedge=0,nflexelem=0,nflexbend_interp=0,nflexintcell=0,flg_surfacevel=False,flg_adhesion=c['adhesion_enabled'],
        opt=NS(integrator=types.IntegratorType.EULER,enableflags=0,disableflags=flags,cone=c['cone'],solver=types.SolverType.NEWTON))
    for key,field in STATIC:setattr(m,field,arr(c[key],'int32'))
    m.jnt_limited_ball_adr=arr(np.empty(0,np.int32),'int32')
    for key,field,kind,_,_ in PARAMETERS:
        owner=m.opt if field.startswith('opt.') else m;setattr(owner,field.split('.')[-1],arr(c[key],kind))
    d=NS(nworld=2,naconmax=c['naconmax'],njmax=c['njmax'],njmax_nnz=0,contact=NS(),efc=NS())
    for key,kind in (('qpos','float32'),('qvel','float32'),('cdof','spatial_vector'),('subtree_com','vec3')):setattr(d,key,arr(c[key],kind))
    d.nacon=arr(c['contact_count'],'int32')
    for key,field,kind in CONTACTS:setattr(d.contact,field,arr(c[key],kind))
    d.contact.efc_address=arr(c['initial_address'],'int32')
    for key in ('ne','nf','nl','nefc'):setattr(d,key,arr(c['initial_'+key],'int32'))
    for key,field,kind in ROWS:setattr(d.efc,field,arr(c['initial_'+field],kind))
    for key in ('J','Jqvel'):setattr(d.efc,key,arr(c['initial_'+key],'float32'))
    d.efc.jtdaj_nblock=arr(c['initial_jtdaj_nblock'],'int32')
    for key in ('jtdaj_adr','jtdaj_nrow','J_rownnz','J_rowadr'):setattr(d.efc,key,arr(np.empty((2,0),np.int32),'int32'))
    d.efc.J_colind=arr(np.empty((2,0,0),np.int32),'int32')
    return m,d


def upstream(m,d):
    """Pinned numerical kernels, retaining their rigid dense caller order."""
    import warp as wp
    from mujoco_warp._src import constraint as q,types
    nnz=wp.empty((2,),dtype=int)
    wp.launch(q._zero_constraint_counts,2,[d.ne,d.nf,d.nl,d.nefc,d.efc.jtdaj_nblock,nnz])
    flags=m.opt.disableflags
    if flags & types.DisableBit.CONSTRAINT:return nnz
    outputs=[d.efc.type,d.efc.id,d.efc.jtdaj_adr,d.efc.jtdaj_nrow,d.efc.jtdaj_nblock,d.efc.J_rownnz,d.efc.J_rowadr,d.efc.J_colind,d.efc.J,
        d.efc.pos,d.efc.margin,d.efc.D,d.efc.vel,d.efc.aref,d.efc.frictionloss,nnz]
    if not flags & types.DisableBit.FRICTIONLOSS:
        wp.launch(q._friction_dof(False,True,False),(2,m.nv),inputs=[m.nv,m.opt.timestep,flags,m.dof_solref,m.dof_solimp,m.dof_frictionloss,m.dof_invweight0,d.qvel,d.njmax,0],outputs=[d.nf,d.nefc,*outputs])
    if not flags & types.DisableBit.LIMIT:
        wp.launch(q._limit_slide_hinge(False,True,False),(2,m.jnt_limited_slide_hinge_adr.size),inputs=[m.nv,m.opt.timestep,flags,m.jnt_qposadr,m.jnt_dofadr,m.jnt_solref,m.jnt_solimp,m.jnt_range,m.jnt_margin,m.dof_invweight0,m.jnt_limited_slide_hinge_adr,d.qpos,d.qvel,d.njmax,0],outputs=[d.nl,d.nefc,*outputs])
    if not flags & types.DisableBit.CONTACT:
        wp.launch(q._efc_contact_init(m.opt.cone,False,True,m.flg_adhesion),d.naconmax,
            inputs=[m.body_weldid,m.body_dofnum,m.body_dofadr,m.dof_parentid,m.geom_bodyid,d.njmax,0,d.nacon,d.contact.dist,d.contact.dim,d.contact.includemargin,d.contact.adhesion,d.contact.worldid,d.contact.geom,d.contact.type],
            outputs=[d.nefc,d.contact.efc_address,d.efc.id,d.efc.jtdaj_adr,d.efc.jtdaj_nrow,d.efc.jtdaj_nblock,d.efc.J_rownnz,d.efc.J_rowadr,nnz])
        frame=wp.array(ptr=d.contact.frame.ptr,dtype=wp.vec3,shape=(d.naconmax,3),device=d.contact.frame.device,copy=False)
        friction=wp.array(ptr=d.contact.friction.ptr,dtype=float,shape=(d.naconmax,5),device=d.contact.friction.device,copy=False)
        d.efc.Jqvel.zero_();tile=32
        wp.launch_tiled(q._efc_contact_jac_dense(tile,m.opt.cone),dim=(2,(m.nv_pad+tile-1)//tile),
            inputs=[m.body_rootid,m.geom_bodyid,m.body_isdofancestor,d.ne,d.nf,d.nl,d.nefc,d.qvel,d.subtree_com,d.cdof,d.contact.efc_address,d.efc.id,d.njmax,m.nv_pad,d.contact.dim,d.contact.geom,d.contact.pos,frame,friction],
            outputs=[d.efc.J,d.efc.Jqvel],block_dim=tile)
        nmax=m.nmaxcondim if m.opt.cone==1 else max(1,2*(m.nmaxcondim-1))
        wp.launch(q._efc_contact_update(m.opt.cone,m.flg_adhesion,False),(d.naconmax,nmax),
            inputs=[m.opt.timestep,flags,m.opt.impratio_invsqrt,m.body_invweight0,m.geom_bodyid,d.contact.efc_address,d.efc.Jqvel,d.nacon,d.contact.dist,d.contact.dim,d.contact.includemargin,d.contact.worldid,d.contact.geom,d.contact.friction,d.contact.solref,d.contact.solreffriction,d.contact.solimp,d.contact.adhesion,d.contact.type],
            outputs=[d.efc.type,d.efc.id,d.efc.pos,d.efc.margin,d.efc.D,d.efc.vel,d.efc.aref,d.efc.frictionloss])
        return nnz,frame,friction
    return nnz


def assign(c,m,d):
    for key,field,_,_,_ in PARAMETERS:parameter(m,field).assign(c[key])
    for key in ('qpos','qvel','cdof','subtree_com'):getattr(d,key).assign(c[key])
    d.nacon.assign(c['contact_count'])
    for key,field,_ in CONTACTS:getattr(d.contact,field).assign(c[key])
    d.contact.efc_address.assign(c['initial_address'])
    for key in ('ne','nf','nl','nefc'):getattr(d,key).assign(c['initial_'+key])
    for _,field,_ in ROWS:getattr(d.efc,field).assign(c['initial_'+field])
    for key in ('J','Jqvel','jtdaj_nblock'):getattr(d.efc,key).assign(c['initial_'+key])


def check_output(c,got,*,reference=None):
    expected,counts,contacts=build_reference(c) if reference is None else reference;nr=c['njmax'];contracts=[];numbers={};targets={};bounds={};seen=[set(),set()]
    def contract(name,ok):contracts.append(dict(name=name,passed=bool(ok)))
    contract('exact counters',all(np.array_equal(got[k],counts[:,i]) for i,k in enumerate(('ne','nf','nl','nefc'))))
    contract('dense block counts',np.array_equal(got['jtdaj_nblock'],np.zeros(2,int)))
    address=got['address'];initial=c['initial_address'];owned={};address_ok=True
    for i in range(c['naconmax']):
        if i not in contacts:address_ok &= np.array_equal(address[i],initial[i]);continue
        w,n=contacts[i];a=address[i,:n];valid=a>=0
        address_ok &= np.array_equal(address[i,n:],initial[i,n:]) and np.all(a[~valid]==-1)
        if np.any(valid):
            base=int(a[0]);address_ok &= base>=counts[w,1]+counts[w,2] and np.array_equal(a[valid],np.arange(base,base+valid.sum())) and np.array_equal(valid,np.arange(n)<valid.sum())
            for component,r in enumerate(a):
                if r<0:continue
                address_ok &= r<nr and (w,int(r)) not in owned;owned[w,int(r)]=(i,component)
    contract('contiguous disjoint contact ownership and address holds',address_ok)
    jqtarget=np.zeros_like(c['initial_Jqvel'],dtype=float) if not(c['disable_constraint'] or c['disable_contact']) else c['initial_Jqvel'].astype(float).copy();jqerror=np.zeros_like(jqtarget)
    identity=True;holds=True;copies=True
    for w in range(2):
        end=min(nr,int(counts[w,3]));prefix=int(counts[w,1]+counts[w,2])
        for r in range(end):
            kind=int(got['type'][w,r]);id=int(got['id'][w,r]);component=0
            if r<prefix:
                identity &= kind==(1 if r<counts[w,1] else 3)
            else:
                entry=owned.get((w,r));identity &= entry is not None
                if entry is None:continue
                contact,component=entry;identity &= id==contact and kind==(5 if c['contact_dim'][id]==1 else 7 if c['cone']==1 else 6)
            key=(kind,id,component);identity &= key in expected[w] and key not in seen[w]
            if key not in expected[w]:continue
            seen[w].add(key);record=expected[w][key]
            for field,pair in record.items():
                value,error=pair;value=np.array(value,copy=True);error=np.array(error,copy=True)
                if field=='J' and kind<5:value[c['nv']:]=c['initial_J'][w,r,c['nv']:];error[c['nv']:]=0.
                if field=='Jqvel':jqtarget[w,r]=value;jqerror[w,r]=error;continue
                numbers.setdefault(field,[]).extend(np.ravel(got[field][w,r]).tolist());targets.setdefault(field,[]).extend(value.ravel().tolist());bounds.setdefault(field,[]).extend(error.ravel().tolist())
                if field in ('margin','frictionloss'):copies &= np.array_equal(np.asarray(got[field][w,r],np.float32).view(np.uint32),np.asarray(value,np.float32).view(np.uint32))
        for field in ('type','id','pos','margin','D','vel','aref','frictionloss'):holds &= np.array_equal(got[field][w,end:].view(np.uint8),c['initial_'+field][w,end:].view(np.uint8))
        holds &= np.array_equal(got['J'][w,end:].view(np.uint8),c['initial_J'][w,end:].view(np.uint8))
        if counts[w,3]<=nr:identity &= seen[w]==set(expected[w])
    contract('row identity and exact category boundaries',identity);contract('unowned row and matrix holds',holds);contract('copied margin and friction bits',copies)
    numeric=check_arrays({'Jqvel':got['Jqvel']},{'Jqvel':jqtarget},{'Jqvel':jqerror}) if nr else []
    if numbers:
        numeric+=check_arrays({k:np.array(v,np.float32) for k,v in numbers.items()},{k:np.array(v) for k,v in targets.items()},{k:np.array(v) for k,v in bounds.items()})
    return dict(contracts=contracts,checks=numeric,passed=all(x['passed'] for x in contracts+numeric))


class Replay:
    def __init__(self,c,args,stream):
        import warp as wp
        from libero.gpu.host.runtime.dynamics.mjwarp_constraint_build import ConstraintConstruction,qualify
        self.case,self.stream=c,stream;self.models={};self.datas={};self.graphs={};self.owners={}
        for mode in ('upstream','cuda'):
            m,d=construct(c,stream);self.models[mode]=m;self.datas[mode]=d
            if mode=='cuda':self.binding=ConstraintConstruction(m,d,args.library,stream,admission=qualify(m,d,implementation='cuda'))
            with wp.ScopedCapture(stream=stream) as cap:
                if mode=='cuda':self.binding.enqueue()
                else:self.owners[mode]=upstream(m,d)
            self.graphs[mode]=cap.graph

    def check(self,updated):
        import warp as wp
        c=current_case(self.case,updated);rows=[];values={};reference=build_reference(c)
        for mode in ('upstream','cuda'):
            m,d=self.models[mode],self.datas[mode];assign(c,m,d)
            readonly={**{key:getattr(m,field) for key,field in STATIC},**{key:parameter(m,field) for key,field,*_ in PARAMETERS},
                **{key:getattr(d.contact,field) for key,field,_ in CONTACTS},**{key:getattr(d,key) for key in ('qpos','qvel','cdof','subtree_com','nacon')}}
            before={key:a.numpy().view(np.uint8).copy() for key,a in readonly.items()}
            wp.capture_launch(self.graphs[mode],stream=self.stream);wp.synchronize_stream(self.stream)
            got={field:getattr(d.efc,field).numpy() for _,field,_ in ROWS}
            got.update({k:getattr(d,k).numpy() for k in ('ne','nf','nl','nefc')});got.update({k:getattr(d.efc,k).numpy() for k in ('J','Jqvel','jtdaj_nblock')});got['address']=d.contact.efc_address.numpy()
            row=check_output(c,got,reference=reference);row.update(mode=mode,inputs_held=all(np.array_equal(a.numpy().view(np.uint8),before[key]) for key,a in readonly.items()));row['passed'] &= row['inputs_held'];rows.append(row);values[mode]=got
        return dict(name=c['name'],updated=updated,modes=rows,passed=all(x['passed'] for x in rows)),values


def signature(args):
    from libero.gpu.host.backends.mjwarp.build import source_files
    sources=source_files(include_constraint_build=True)
    sources={k:v for k,v in sources.items() if k.startswith('upstream/') or 'constraint_build' in k or 'ConstraintBuild' in k or k in ('host/runtime/dynamics/mjwarp_motion.py','host/runtime/dynamics/mjwarp_external_wrench.py','data/mjwarp_source_pin.json','project_source_identity')}
    return dict(sources=sources,inputs=input_identity(args),library_sha256=digest(args.library),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_constraint_build_reference.py')),helpers=helpers())


def cpu_reference(args):
    import mujoco
    if mujoco.__version__!='2.3.7':raise RuntimeError('Constraint construction CPU oracle requires CPU reference2.3.7')
    checks=[];differences=[]
    for original in cases(args):
        receipt=json.loads((args.inputs/(original['name']+'.json')).read_text())
        if 'model_path' in receipt:
            if digest(receipt['model_path'])!=receipt['model_sha256']:raise RuntimeError('CPU constraint model changed')
            m=mujoco.MjModel.from_binary_path(receipt['model_path'])
        else:m=mujoco.MjModel.from_xml_string(receipt['xml'])
        m.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE;m.opt.cone=original['cone']
        for updated in (False,True):
            c=current_case(original,updated);c['adhesion_enabled']=False;c['contact_adhesion'][:]=0.
            records,counts,_=build_reference(c)
            for w in range(2):
                m.opt.disableflags=sum(int(getattr(mujoco.mjtDisableBit,'mjDSBL_'+{'friction':'FRICTIONLOSS'}.get(k,k.upper()))) for k in FLAGS if c['disable_'+k])
                m.opt.timestep=float(c['timestep'][w%len(c['timestep'])]);ratio=float(c['impratio_invsqrt'][w%len(c['impratio_invsqrt'])]);m.opt.impratio=1./(ratio*ratio)
                for key,field,*_ in PARAMETERS:
                    if field.startswith('opt.'):continue
                    getattr(m,field)[:]=c[key][w%len(c[key])]
                d=mujoco.MjData(m);d.qpos[:]=c['qpos'][w];d.qvel[:]=c['qvel'][w]
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);d.cdof[:]=c['cdof'][w];d.subtree_com[:]=c['subtree_com'][w]
                indices=[i for i in range(min(c['naconmax'],int(c['contact_count'][0]))) if c['contact_world'][i]==w]
                for i in indices:
                    contact=mujoco.MjContact();contact.geom1=int(c['contact_geom'][i,0]);contact.geom2=int(c['contact_geom'][i,1]);contact.dim=int(c['contact_dim'][i])
                    contact.dist=float(c['contact_dist'][i]);contact.includemargin=float(c['contact_margin'][i]);contact.pos[:]=c['contact_pos'][i];contact.frame[:]=c['contact_frame'][i].ravel()
                    contact.friction[:]=c['contact_friction'][i];contact.solref[:]=c['contact_ref'][i];contact.solreffriction[:]=c['contact_ref_friction'][i];contact.solimp[:]=c['contact_imp'][i]
                    contact.exclude=int(not(c['contact_type'][i]&1) or not(np.float32(c['contact_dist'][i]-c['contact_margin'][i])<0))
                    if mujoco.mj_addContact(m,d,contact):raise RuntimeError('CPU contact allocation exhausted')
                mujoco.mj_makeConstraint(m,d);mujoco.mj_referenceConstraint(m,d)
                count_ok=(d.ne==counts[w,0] and d.nf==counts[w,1] and d.nefc==counts[w,3])
                checks.append(dict(name=c['name'],updated=updated,world=w,field='counts',passed=bool(count_ok)))
                seen=set();major=[]
                for r in range(d.nefc):
                    kind=int(d.efc_type[r]);id=int(d.efc_id[r]);component=0
                    if kind>=5:component=r-int(d.contact[id].efc_address);id=indices[id]
                    key=(kind,id,component);valid=key in records[w] and key not in seen;seen.add(key)
                    checks.append(dict(name=c['name'],updated=updated,world=w,field='identity',passed=bool(valid)))
                    if not valid:continue
                    record=records[w][key]
                    for field,(value,error) in record.items():
                        if field=='Jqvel':continue
                        if field in ('pos','margin') and kind==7 and component>0:
                            # CPU2.3.7 stores zero tangential pos/margin. Pinned
                            # MJWarp stores include-margin there; aref is still
                            # purely damping. This known public-layout difference
                            # is retained and separately qualified on the GPU.
                            differences.append(dict(name=c['name'],field=field,reason='CPU tangential pos/margin zero; pinned includes margin'));continue
                        target=np.asarray(value)[:c['nv']] if field=='J' else np.asarray(value)
                        budget=np.asarray(error)[:c['nv']] if field=='J' else np.asarray(error)
                        got=d.efc_J.reshape(d.nefc,m.nv)[r] if field=='J' else getattr(d,'efc_'+field)[r]
                        delta=float(abs(np.asarray(got)-target).max(initial=0.))
                        eps=np.finfo(float).eps
                        limit=float((8192*eps*np.maximum(1.,abs(target))+8*budget*(eps/(np.finfo(np.float32).eps/2))).max(initial=0.))
                        checks.append(dict(name=c['name'],updated=updated,world=w,field=field,error=delta,limit=limit,passed=bool(np.isfinite(got).all() and delta<=limit)))
                checks.append(dict(name=c['name'],updated=updated,world=w,field='coverage',passed=seen==set(records[w])))
    result=dict(passed=all(x['passed'] for x in checks),version=mujoco.__version__,checks=checks,known_version_differences=differences,inputs=input_identity(args),tool_sha256=digest(__file__),
        reference_sha256=digest(Path(__file__).with_name('mjwarp_constraint_build_reference.py')),helpers=helpers(),physics_integrations=0,newton_solves=0,
        policy='actual dense CPU2.3.7 makeConstraint/referenceConstraint on equivalent raw inputs; zero adhesion, matching ratio and flags; all rows checked independently of GPU capacity')
    (args.output/'cpu-oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cpu_checks=len(checks),failures=sum(not x['passed'] for x in checks))),flush=True);return 0 if result['passed'] else 2


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
        if not prep.is_file() or json.loads(prep.read_text())['signature']!=sig:raise RuntimeError('Prepare exact row-construction operators offline first')
        cpu=json.loads((args.output/'cpu-oracle.json').read_text())
        if not cpu['passed'] or any(cpu[k]!=sig[k] for k in ('inputs','tool_sha256','reference_sha256','helpers')):raise RuntimeError('Prepare matching CPU constraint-construction oracle first')
    with compilation_scope(args.prepare) as scope,wp.ScopedStream(stream):
        replays=[Replay(c,args,stream) for c in cases(args)]
        if args.prepare:
            wp.synchronize_stream(stream);save_json(prep,dict(signature=sig,compilation_events=scope['events'],physics_integrations=0,newton_solves=0));print(json.dumps(dict(prepared=True,captures=2*len(replays),compilations=len(scope['events']))),flush=True);return 0
        rows=[]
        for replay in replays:
            for updated in (False,True):
                row,values=replay.check(updated);rows.append(row);print(json.dumps({k:row[k] for k in ('name','updated','passed')}),flush=True)
                if not row['passed']:
                    np.savez(args.output/'failure.npz',**{mode+'.'+k:a for mode,v in values.items() for k,a in v.items()});break
            if not row['passed']:break
        passed=all(x['passed'] for x in rows)
        save_json(args.output/'summary.json',dict(passed=passed,signature=sig,cases=rows,runtime_compilations=scope['events'],physics_integrations=0,newton_solves=0,performance=None,worlds=2,
            policy='independent FP64 Jacobian/kbi/reference equations with propagated FP32 and powf sensitivity, fixed8x; exact count/category/block ownership/copy/hold contracts'))
        print(json.dumps(dict(passed=passed,cases=len(rows))),flush=True);return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
