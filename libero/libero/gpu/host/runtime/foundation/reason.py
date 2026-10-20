"""Prepared, source-bound device reason witnesses for mixed local statuses.

The metadata entry commands run before the original producer, then the sealed
post command checks its actual status. No CPU intermediate value is uploaded,
no physics producer is replaced, and original modules remain byte unchanged.
Composite result_kind and complete caller-origin witnesses remain explicit
unfinished work; this module does not claim full Phase06 source forest coverage.
"""
from __future__ import annotations
import libero.gpu.precision as scalar
import ast
import hashlib
import inspect
from pathlib import Path
import warp as wp
from .binding import BindingError, lookup, array_token
from .graph import WarpKernelCommand
from .authenticated_source import FrozenSources
from .protocols import SOURCE_BINDINGS, PROTOCOL_IDS, validate_world_coverage, validate_static_coverage
from .schema import FieldDescriptor
from . import reason_kernels as RK
wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})


@wp.kernel(enable_backward=False)
def _mark(event:wp.array(dtype=wp.int64),epoch:wp.array(dtype=wp.int64),
          saved_event:wp.array(dtype=wp.int64),saved_epoch:wp.array(dtype=wp.int64),
          armed:wp.array(dtype=wp.int32),source_active:wp.array(dtype=wp.int32),
          shared_active:wp.array(dtype=wp.int32),saved_source_active:wp.array(dtype=wp.int32),
          saved_shared_active:wp.array(dtype=wp.int32)):
    w=wp.tid()
    saved_event[w]=event[w]
    saved_epoch[w]=epoch[w]
    armed[w]=1
    saved_source_active[w]=source_active[w]
    saved_shared_active[w]=shared_active[w]

@wp.kernel(enable_backward=False)
def _post(local:wp.array(dtype=wp.int32),pre_code:wp.array(dtype=wp.int32),
          pre_kind:wp.array(dtype=wp.int32),event:wp.array(dtype=wp.int64),
          epoch:wp.array(dtype=wp.int64),saved_event:wp.array(dtype=wp.int64),
          saved_epoch:wp.array(dtype=wp.int64),armed:wp.array(dtype=wp.int32),
          saved_source_active:wp.array(dtype=wp.int32),saved_shared_active:wp.array(dtype=wp.int32),
          shared_active:wp.array(dtype=wp.int32),output_code:wp.array(dtype=wp.int32),output_kind:wp.array(dtype=wp.int32),
          class_table:wp.array(dtype=wp.int32),success_table:wp.array(dtype=wp.int32),
          protocol:int,computed_nonfinite:int):
    w=wp.tid();code=local[w]
    output_code[w]=code;output_kind[w]=8
    if armed[w]!=1 or saved_event[w]!=event[w] or saved_epoch[w]!=epoch[w]:
        armed[w]=0
        return
    armed[w]=2
    if saved_shared_active[w]!=shared_active[w] or (saved_shared_active[w]==1 and saved_source_active[w]==0):
        return
    if code<0 or code>=32:
        output_kind[w]=4
        return
    index=protocol*32+code
    if pre_code[w]==code:
        output_kind[w]=pre_kind[w]
    elif pre_code[w]==0:
        if success_table[index]==1:output_kind[w]=0
        elif computed_nonfinite==code:output_kind[w]=5
        elif class_table[index]>=1:output_kind[w]=class_table[index]

@wp.kernel(enable_backward=False)
def _hessian_lost_reason(local:wp.array(dtype=wp.int32),pre_code:wp.array(dtype=wp.int32),
        kind:wp.array(dtype=wp.int32),count:wp.array(dtype=wp.int32),
        state:wp.array(dtype=wp.int32),qm:wp.array(dtype=scalar.wp_real),
        D:wp.array(dtype=scalar.wp_real),J:wp.array(dtype=scalar.wp_real),
        output_kind:wp.array(dtype=wp.int32),armed:wp.array(dtype=wp.int32),rows:int):
    # Original factor_update local4 after a successful original validate can
    # arise only from its lost-rank fallback. Infer reachability from that actual
    # status; do not recompute rank or overwrite original H/route diagnostics.
    w=wp.tid()
    if armed[w]!=2 or local[w]!=4 or pre_code[w]!=0 or kind[w]!=3:return
    for z in range(153):
        if not wp.isfinite(qm[w*153+z]):output_kind[w]=5;return
    for r in range(count[w]):
        if state[w*rows+r]==1:
            if D[w*rows+r]<scalar.wp_real(0.0):output_kind[w]=6;return
            if not wp.isfinite(D[w*rows+r]):output_kind[w]=5;return
            if D[w*rows+r]!=scalar.wp_real(0.0):
                for k in range(43):
                    if not wp.isfinite(J[(w*rows+r)*43+k]):output_kind[w]=5;return
    output_kind[w]=8

SPECS={
 'D04.body_status':('reference/position/body_kinematics/kernel.py','_compute',RK._reason_body,'d.status',-1),
 'S01.contact_status':('reference/constraints/constraint_contact/operator.py','_preflight',RK._reason_contact,'status',-1),
 'S02.scalar_status':('reference/constraints/constraint_scalar/kernel.py','_produce',RK._reason_scalar,'status',-1),
 'S04.update_status':('reference/constraints/constraint_update/kernel.py','_update',RK._reason_update,'status',-1),
 'S05.hessian_status':('reference/newton/newton_hessian/kernel.py','validate',RK._reason_hessian,'status',-1),
 'S06.direction_status':('reference/newton/newton_direction/kernel.py','_direction',RK._reason_direction,'status',5),
 'S08.A_status':('reference/newton/newton_state/kernel.py','_warm_select',RK._reason_warm,'status',-1),
 'S08.B_status':('reference/newton/newton_state/kernel.py','_commit_step',RK._reason_step,'status',3),
 'S08.C_status':('reference/newton/newton_state/kernel.py','_stats_stop',RK._reason_stats,'status',3)}

def _value(arguments,path):
    pieces=path.split('.');v=arguments[pieces[0]]
    for name in pieces[1:]:v=getattr(v,name)
    return v

def _prepare_reason_arrays(self,proto,registry,protocol,event,epoch):
    # event and epoch must themselves be registered actual retained arrays.
    for label,a in [('event',event),('epoch',epoch)]:
        if not any(lookup(registry.roots[b.root],b.path) is a for b in registry.bindings.values()):
            raise BindingError('unbound reason '+label)
        if a.shape!=(proto.nworld,) or a.dtype!=wp.int64 or a.device!=proto.device:
            raise BindingError('reason clock schema/device mismatch')
    self.event=event;self.epoch=epoch
    owners=[name for name,obj in registry.roots.items() if obj is proto]
    if len(owners)!=1:raise BindingError('actual F protocol owner must be a unique retained source root')
    self.owner=owners[0]
    for name in ('success','error_class','active'):
        a=getattr(proto,name)
        if not any(lookup(registry.roots[b.root],b.path) is a for b in registry.bindings.values()):
            d=FieldDescriptor('F.static.'+name,'int32',tuple(a.shape),self.owner,'prepared F protocol lifetime','prepared source table',('F.prepare','F.gate') if name=='active' else ('F.prepare',),'actual F gate admission' if name=='active' else 'unmodified static source routing table','full original local namespace table','held throughout all captures',access='read_write' if name=='active' else 'read')
            registry.add(d,self.owner,name,a)
    from .device import _route
    pre_kernel=SPECS[protocol][2] if protocol in SPECS else __import__(__package__+'.native_reason_kernels',fromlist=['_reason_linesearch'])._reason_linesearch
    field_writers={'pre_code':(pre_kernel,), 'pre_kind':(pre_kernel,), 'classes':(_post,_hessian_lost_reason) if protocol=='S05.hessian_status' else (_post,), 'status_codes':(_post,), 'armed':(_mark,_post,_route,_hessian_lost_reason) if protocol=='S05.hessian_status' else (_mark,_post,_route),'saved_source_active':(_mark,), 'saved_shared_active':(_mark,), 'saved_event':(_mark,), 'saved_epoch':(_mark,)}
    key='r'+str(PROTOCOL_IDS[protocol])
    if key in proto.reasons:raise BindingError('one prepared source reason per actual stage/protocol')
    proto.reasons[key]=self
    for name,dtype in [('pre_code',wp.int32),('pre_kind',wp.int32),('classes',wp.int32),('status_codes',wp.int32),('armed',wp.int32),('saved_source_active',wp.int32),('saved_shared_active',wp.int32),('saved_event',wp.int64),('saved_epoch',wp.int64)]:
        a=wp.zeros(proto.nworld,dtype=dtype,device=proto.device);setattr(self,name,a)
        d=FieldDescriptor('F.reason.'+key+'.'+name,'int64' if dtype==wp.int64 else 'int32',(proto.nworld,),self.owner,'F preparation through stream/Graph lifetime','same actual source event+epoch',tuple(registry.source_writer(kernel) for kernel in field_writers[name]), 'source-order metadata only','one actual world','held until next explicit source entry')
        registry.add(d,self.owner,'reasons.'+key+'.'+name,a)
        registry._protected_source_arrays[id(a)]={'witness':self,'field':name}

class SourceReasonWitness:
    def __init__(self,proto,registry,protocol,producer_command,event,epoch,*,producer_forest=None):
        if registry._sealed:raise BindingError('prepare reason arrays before registry seal')
        if protocol not in SPECS:raise BindingError('original source reason is not implemented: '+protocol)
        self.proto=proto;self.registry=registry;self.protocol=protocol
        self.producer=WarpKernelCommand(producer_command)
        self.source_metadata=self.producer.validate(registry)
        try:validate_world_coverage(self.source_metadata,proto.nworld)
        except ValueError as e:raise BindingError(str(e)) from e
        relative,function,reason_kernel,status_path,computed=SPECS[protocol]
        original=producer_command.kernel.func
        source=Path(inspect.getsourcefile(original));raw=source.read_bytes();pin=SOURCE_BINDINGS[relative]
        if hashlib.sha256(raw).hexdigest()!=pin['sha256'] or len(raw)!=pin['bytes']:
            raise BindingError('actual original producer source differs from frozen namespace')
        self.original_source=FrozenSources([dict(pin,path=str(source))])
        self.loaded_source_binding=self.original_source.function(original)
        if function is not None and self.loaded_source_binding['original_name']!=function:
            raise BindingError('reason witness does not describe this actual original producer declaration')
        self.source_binding={'path':str(source),'sha256':pin['sha256'],'bytes':pin['bytes'],'function':self.loaded_source_binding['original_name'],'canonical_qualname':self.loaded_source_binding['qualname']}

        try:self.actual_lowering_binding=self.original_source.warp_function(producer_command.kernel)
        except ValueError as e:raise BindingError(str(e)) from e
        args={a.label:v for a,v in zip(producer_command.kernel.adj.args,producer_command.fwd_args)}
        self.status=_value(args,status_path)
        self.source_active=_value(args,'d.active' if protocol=='D04.body_status' else 'active')
        if not isinstance(self.source_active,wp.array) or self.source_active.shape!=(proto.nworld,) or self.source_active.dtype!=wp.int32:raise BindingError('actual source admission schema differs')
        if not isinstance(self.status,wp.array) or self.status.shape!=(proto.nworld,) or self.status.dtype!=wp.int32:
            raise BindingError('actual original status schema differs')
        expected_operations={'S01.contact_status':('_preflight','_jacobian_rows','_metadata'),'S04.update_status':('_update','_transpose'),'S05.hessian_status':('validate','assemble','factor_update')}.get(protocol,(function,))
        forest=(producer_command,) if producer_forest is None else tuple(producer_forest)
        if not forest or forest[0] is not producer_command or tuple(self.original_source.function(c.kernel.func)['original_name'] for c in forest)!=expected_operations:raise BindingError('complete original source operation forest/order required: '+protocol)
        for child in forest:registry.admit_warp(child)
        self.producer_forest=tuple(WarpKernelCommand(c) for c in forest)
        self.forest_metadata=[];self.forest_source_bindings=[];self.operation_launch_sources=[]
        original_tree=self.original_source.syntax[str(source.resolve())]
        for child in forest:
            child_args={a.label:v for a,v in zip(child.kernel.adj.args,child.fwd_args)}
            for name,value in child_args.items():
                if name not in args:continue
                original_value=args[name]
                if isinstance(value,wp.array) or isinstance(original_value,wp.array):
                    if value is not original_value:raise BindingError('original producer forest shared input/status owner differs: '+name)
                elif type(value) is not type(original_value) or value!=original_value:raise BindingError('original producer forest shared static scalar differs: '+name)
            try:self.forest_source_bindings.append(self.original_source.warp_function(child.kernel))
            except ValueError as e:raise BindingError(str(e)) from e
            metadata=WarpKernelCommand(child).validate(registry)
            shape=(proto.nworld,)
            if protocol=='S01.contact_status' and self.original_source.function(child.kernel.func)['original_name']=='_jacobian_rows':shape=(proto.nworld,args['maxcon'],max(args['nv'],1))
            elif protocol=='S01.contact_status' and self.original_source.function(child.kernel.func)['original_name']=='_metadata':shape=(proto.nworld,args['maxcon'])
            elif protocol=='S04.update_status' and self.original_source.function(child.kernel.func)['original_name']=='_transpose':shape=(proto.nworld*args['nv'],)
            elif protocol=='S05.hessian_status' and self.original_source.function(child.kernel.func)['original_name']=='assemble':shape=(proto.nworld*43*43,)
            try:validate_static_coverage(metadata,shape)
            except ValueError as e:raise BindingError(str(e)) from e
            self.forest_metadata.append(metadata)
            if len(forest)>1:
                calls=[node for node in ast.walk(original_tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='wp' and node.func.attr=='launch' and node.args and isinstance(node.args[0],ast.Name) and node.args[0].id==self.original_source.function(child.kernel.func)['original_name']]
                if len(calls)!=1:raise BindingError('original operation child has no unique byte-pinned original launch source')
                self.operation_launch_sources.append({'kernel':self.original_source.function(child.kernel.func)['original_name'],'line':calls[0].lineno,'original_launch':ast.unparse(calls[0]),'actual_exact_shape':shape})
        _prepare_reason_arrays(self,proto,registry,protocol,event,epoch)
        self.entry_commands=[wp.launch(reason_kernel,dim=proto.nworld,inputs=list(producer_command.fwd_args)+[self.pre_code,self.pre_kind],device=proto.device,stream=proto.stream,record_cmd=True),
            wp.launch(_mark,dim=proto.nworld,inputs=[event,epoch,self.saved_event,self.saved_epoch,self.armed,self.source_active,proto.active,self.saved_source_active,self.saved_shared_active],device=proto.device,stream=proto.stream,record_cmd=True)]
        self.post_commands=[wp.launch(_post,dim=proto.nworld,inputs=[self.status,self.pre_code,self.pre_kind,event,epoch,self.saved_event,self.saved_epoch,self.armed,self.saved_source_active,self.saved_shared_active,proto.active,self.status_codes,self.classes,proto.error_class,proto.success,PROTOCOL_IDS[protocol],computed],device=proto.device,stream=proto.stream,record_cmd=True)]
        if protocol=='S05.hessian_status':
            self.post_commands.append(wp.launch(_hessian_lost_reason,dim=proto.nworld,inputs=[self.status,self.pre_code,args['kind'],args['count'],args['state'],args['qm'],args['D'],args['J'],self.classes,self.armed,args['rows']],device=proto.device,stream=proto.stream,record_cmd=True))
        self.clock_tokens=(array_token(event),array_token(epoch))
        self._source_pending=False
        for command in self.entry_commands+self.post_commands:registry.admit_warp(command)
        self.command_metadata=[WarpKernelCommand(c).validate(registry) for c in self.entry_commands+self.post_commands]

    def validate(self,local,event,epoch,protocol):
        if protocol!=self.protocol or local is not self.status or event is not self.event or epoch is not self.epoch:
            raise BindingError('reason is not bound to the actual original status/clock/protocol')
        self.registry.validate()
        if hasattr(self,'original_source'):
            self.original_source.validate()
            if self.original_source.function(self.producer.command.kernel.func)!=self.loaded_source_binding or self.original_source.warp_function(self.producer.command.kernel)!=self.actual_lowering_binding:raise BindingError('actual loaded producer code/lowering/source changed')
        if hasattr(self,'producer_forest') and ([c.validate(self.registry) for c in self.producer_forest]!=self.forest_metadata or [self.original_source.warp_function(c.command.kernel) for c in self.producer_forest]!=self.forest_source_bindings):raise BindingError('complete actual producer forest/source/arguments changed')
        if self.producer.validate(self.registry)!=self.source_metadata:
            raise BindingError('source reason producer typed arguments changed')
        if (array_token(event),array_token(epoch))!=self.clock_tokens:
            raise BindingError('source reason clock owner changed')
        if [WarpKernelCommand(c).validate(self.registry) for c in self.entry_commands+self.post_commands]!=self.command_metadata:
            raise BindingError('prepared source reason arguments changed')

    def capture_entry(self):
        raise BindingError('split source entry cannot prove producer execution; use source_sequence or enqueue_source')

    def source_sequence(self):
        return SourceReasonSequence(self)

    def complete_sequence(self,error_item,*,stage):
        # Actual route is part of every repeated GPU sequence, so the device
        # armed token is consumed before the next original stage/iteration.
        self.proto._stage(stage)
        self.proto._require([('source_error_item',error_item,'int32')])
        return SourceReasonSequence(self,error_item=error_item,stage=stage)

    def enqueue_source(self):
        # Outside capture this owns and enqueues the actual source command;
        # the public post/route API cannot turn an old local0 into a new source.
        if self.proto.stream.is_capturing:raise BindingError('source capture requires retained full sequence through BindingRegistry.capture_graph')
        sequence=self.source_sequence()
        with self.registry._graph_mutex:
            sequence.validate(self.registry)
            if self._source_pending:raise BindingError('previous source result has not been consumed')
            for command in sequence.commands:command.launch(self.proto.stream)
            sequence.note_enqueued(self.registry)

    def refresh(self,local,event,epoch,protocol):
        self.validate(local,event,epoch,protocol)
        if not self._source_pending:raise BindingError('no actual owned producer sequence enqueued for this source result')
        self._source_pending=False
        return self.status_codes,self.classes,self.saved_event,self.saved_epoch,self.armed

    def to_json(self):
        return {'protocol':self.protocol,'source':self.source_binding,'actual_producer':self.source_metadata,'actual_metadata_commands':self.command_metadata,'clock_owner':self.clock_tokens,'scope':'prepared actual source preflight reason; device execution/whole caller forest acceptance separately required'}


class SourceReasonSequence:
    """Retain inseparable actual pre/source/post commands for typed Graphs.

    This authenticates source execution order, not numeric contract success.
    The registry flattens the exact tuple only after verifying every identity,
    and rechecks source/arguments/order before each actual Graph launch.
    """
    def __init__(self,witness,*,error_item=None,stage=None):
        self.witness=witness
        self.error_item=error_item;self.stage=stage;self.complete_route=error_item is not None
        witness.validate(witness.status,witness.event,witness.epoch,witness.protocol)
        self._forest=tuple(getattr(witness,'producer_forest',(witness.producer,)))
        base=[*(WarpKernelCommand(c) for c in witness.entry_commands),*self._forest,*(WarpKernelCommand(c) for c in witness.post_commands)]
        self.route_command=None
        if self.complete_route:
            from .device import _route
            witness.proto._stage(stage);witness.proto._require([('source_error_item',error_item,'int32')])
            command=wp.launch(_route,dim=witness.proto.nworld,inputs=self._route_args(),device=witness.proto.device,stream=witness.proto.stream,record_cmd=True)
            witness.registry.admit_warp(command)
            self.route_command=WarpKernelCommand(command);base.append(self.route_command)
        self.commands=tuple(base)
        self._command_ids=tuple(id(c) for c in self.commands)
        self._underlying_ids=tuple(id(c.command) if isinstance(c,WarpKernelCommand) else id(c) for c in self.commands)
        self._expected=tuple(c.validate(witness.registry) for c in self.commands)
        self._pre_count=len(witness.entry_commands)
        self._post_count=len(witness.post_commands)
        self._witness_id=id(witness)
    def _route_args(self):
        w=self.witness;p=w.proto
        return [p.active,w.status,self.error_item,w.event,w.epoch,p.success,p.error_class,*p._errors(),p.valid,p.valid_epoch,w.status_codes,w.classes,w.saved_event,w.saved_epoch,w.armed,1,PROTOCOL_IDS[w.protocol],self.stage]
    def validate(self,registry):
        w=self.witness
        if id(w)!=self._witness_id or registry is not w.registry:raise BindingError('actual source sequence owner replaced')
        w.validate(w.status,w.event,w.epoch,w.protocol)
        forest=tuple(getattr(w,'producer_forest',(w.producer,)))
        if len(forest)!=len(self._forest) or any(a is not b for a,b in zip(forest,self._forest)):raise BindingError('actual full producer forest replaced')
        original=tuple([*w.entry_commands,*(c.command if isinstance(c,WarpKernelCommand) else c for c in forest),*w.post_commands,*([self.route_command.command] if self.complete_route else [])])
        if self.complete_route:
            from .device import _route
            command=self.route_command.command;expected_args=self._route_args()
            if command.kernel is not _route or len(command.fwd_args)!=len(expected_args) or any(actual is not expected for actual,expected in zip(command.fwd_args[:-3],expected_args[:-3])) or tuple(command.fwd_args[-3:])!=tuple(expected_args[-3:]):raise BindingError('actual complete source route/code/owner arguments replaced')
            w.proto._stage(self.stage);w.proto._require([('source_error_item',self.error_item,'int32')])
        if tuple(id(c) for c in self.commands)!=self._command_ids or tuple(id(c.command) if isinstance(c,WarpKernelCommand) else id(c) for c in self.commands)!=self._underlying_ids or tuple(id(c) for c in original)!=self._underlying_ids or len(w.entry_commands)!=self._pre_count or len(w.post_commands)!=self._post_count:
            raise BindingError('source pre/actual producer/post sequence omitted, reordered or replaced')
        for command in self.commands:
            if isinstance(command,WarpKernelCommand):registry.require_warp(command.command)
        actual=tuple(c.validate(registry) for c in self.commands)
        if actual!=self._expected:raise BindingError('actual source sequence typed metadata changed')
        return actual
    def note_enqueued(self,registry):
        self.validate(registry)
        if not self.complete_route:
            if self.witness._source_pending:raise BindingError('unconsumed source result would be overwritten by another sequence')
            self.witness._source_pending=True
    def to_json(self):
        return {'schema':'actual-pre-producer-post-source-sequence-v1','source':self.witness.source_binding,'protocol':self.witness.protocol,'pre_count':self._pre_count,'actual_producer_count':len(self._forest),'post_count':self._post_count,'device_route_count':int(self.complete_route),'commands':self._expected,'scope':'actual typed source order and retained identities; full caller/epoch/numeric acceptance separate','full_F':False}
