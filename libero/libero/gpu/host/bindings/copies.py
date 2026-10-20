"""Handwritten multi-field copies retaining alias fallback and sample boundaries."""
import libero.gpu.precision as scalar
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import types
import hashlib
import warp as wp
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.assembly.compose.backup import _overlap
from libero.gpu.host.runtime.step.shape_runtime import _token

HEADER=(GPU_ROOT / 'cuda/runtime/copies.cuh')
KINDS={scalar.wp_real:'f64',wp.int32:'i32',wp.int64:'i64'}


def entry_copy_kind(source, target, width, block_threads):
    """Only the measured aligned 342-double/64-thread route is vector selected.

    Prepared array owners and scalar dimensions remain sealed by the caller.
    Unknown float64 widths use the original scalar function; integer routes
    retain their existing vector implementation.
    """
    kind = KINDS[source.dtype]
    if (source.dtype == scalar.wp_real and block_threads == 64 and width == 342
            and source.ptr % 16 == 0 and target.ptr % 16 == 0):
        return 'f64_vector'
    return kind

def _tile_stages(source,stages,has_slot):
    result=[];schedule=[];pending=[]
    def flush():
        nonlocal source
        if not pending:return
        if len(pending)==1:
            result.append(pending[0]);schedule.append(('field',pending[0][1][2]))
        else:
            # Several CopyPlan sources may be pasted into one translation unit.
            # Name/guard the helper by its complete generated function contract.
            contract=repr((has_slot,tuple((stage[1][2],tuple(stage[3])) for stage in pending)))
            name='tile_'+hashlib.sha256(contract.encode()).hexdigest()[:20]
            declarations=[];args=[];specs=[];calls=[]
            for j,stage in enumerate(pending):
                _,values,_,types=stage
                for k,(value,typ) in enumerate(zip(values,types)):
                    declarations.append(typ+' a'+str(j)+'_'+str(k))
                    args.append(value);specs.append(typ)
                slotarg='a'+str(j)+'_4' if has_slot else 'nullptr'
                calls.append('begin+=tile_field(w,a%d_0,a%d_1,a%d_2,a%d_3,%s,begin);' %
                             (j,j,j,j,slotarg))
            guard='LIBERO_CF_COPY_'+name.upper()
            source+='\n#ifndef '+guard+'\n#define '+guard+'\nnamespace cf_copy { __device__ void '+name+'(int w,'+','.join(declarations)+'){int begin=0;'+''.join(calls)+'} }\n#endif\n'
            result.append(('cf_copy::'+name,args,'block',specs))
            schedule.append(('tile',tuple(stage[1][2] for stage in pending)))
        pending.clear()
    for stage in stages:
        width=stage[1][2]
        if 0<width<=64:
            pending.append(stage)
            if len(pending)==8:flush()
        else:
            flush();result.append(stage);schedule.append(('field',width))
    flush()
    return source,result,tuple(schedule)

class CopyPlan:
    def __init__(self,pairs,root,slot=None,*,compile_plan=True):
        self.pairs=tuple(pairs); self.slot=slot
        # Event histories are world-major. The old private-world (rows,width)
        # layout remains valid when nworld == 1.
        self.nworld=1 if slot is None else int(slot.size)
        if slot is not None and (slot.dtype!=wp.int32 or slot.ndim!=1 or self.nworld<1):
            raise ValueError('copy slots require a nonempty resident world vector')
        targets=[t for s,t in self.pairs]
        self.fallback=any(_overlap(t,s) for t in targets for s,_ in self.pairs)
        self.fallback|=any(_overlap(t,u) for i,t in enumerate(targets) for u in targets[i+1:])
        if slot is not None:self.fallback|=any(_overlap(t,slot) for t in targets)
        stages=[]
        for s,t in self.pairs:
            if s.dtype!=t.dtype or s.device!=t.device or not s.is_contiguous or not t.is_contiguous:raise ValueError('copy schema')
            if s.dtype not in KINDS:raise TypeError('copy dtype')
            if slot is None and s.size!=t.size:raise ValueError('copy extent')
            width=s.size
            rows=1
            if slot is not None:
                if s.device!=slot.device or s.size%self.nworld:
                    raise ValueError('event source world extent/device')
                width=s.size//self.nworld
                if t.ndim==3 and t.shape[0]==self.nworld and t.shape[2]==width:
                    rows=t.shape[1]
                elif t.ndim==2 and t.shape[1]==width and t.shape[0]%self.nworld==0:
                    rows=t.shape[0]//self.nworld
                else:
                    raise ValueError('copy target requires (world,rows,width) or (world*rows,width)')
                if rows<1:raise ValueError('event history rows must be positive')
            args=[s,t,width,rows,slot if slot is not None else 0]
            ctype={'f64':'real_t','i32':'int32_t','i64':'int64_t'}[KINDS[s.dtype]]
            stages.append(('cf_copy::'+KINDS[s.dtype],args,'block',[ctype+'*',ctype+'*','int32_t','int32_t','int32_t*'] if slot is not None else [ctype+'*',ctype+'*','int32_t','int32_t','size_t']))
        self.tokens=tuple((_token(s),_token(t)) for s,t in self.pairs)
        self.slot_token=None if slot is None else _token(slot)
        # Null-pointer argument is ABI-compatible; explicit wrapper avoids integer-to-pointer conversion.
        source=HEADER.read_text()
        if slot is None:
            source+='\n#ifndef LIBERO_CF_COPY_NULL_WRAPPERS\n#define LIBERO_CF_COPY_NULL_WRAPPERS\nnamespace cf_copy {\n'
            for kind,typ in [('f64','real_t'),('i32','int32_t'),('i64','int64_t')]:
                source+=f'__device__ void {kind}_null(int w,const {typ}*s,{typ}*t,int n,int r,size_t zero){{field(w,s,t,n,r,nullptr);}}\n'
            source+='}\n#endif // LIBERO_CF_COPY_NULL_WRAPPERS\n'
            stages=[(fn+'_null',args,mode,spec) for fn,args,mode,spec in stages]
        # Cold independent-field tile; arguments remain frozen in launch ABI.
        if not self.fallback:
            source,stages,self.field_schedule=_tile_stages(source,stages,slot is not None)
        else:
            self.field_schedule=tuple(('ordered_fallback',) for _ in stages)
        self.source=source;self.stages=stages
        if self.fallback and self.nworld!=1:
            raise ValueError('world-major aliased event copies require an explicit ordered fallback')
        # The admitted disjoint copy fields have no inter-field dependence.
        self.plan=NativePlan('copies', [source], stages,self.nworld,root,stage_barriers=False) if compile_plan and stages and not self.fallback else None
    def validate_bindings(self):
        if tuple((_token(s),_token(t)) for s,t in self.pairs)!=self.tokens or (self.slot is not None and _token(self.slot)!=self.slot_token):raise ValueError('copy owners changed')
    def launch(self,stream):
        self.validate_bindings()
        if self.plan:self.plan.launch(stream)
        elif self.fallback:
            from libero.gpu.host.runtime.step.recovery_event_runtime import _copy_f64, _copy_i32
            for s,t in self.pairs:
                if not s.size:continue
                if self.slot is None:wp.copy(t,s,stream=stream)
                else:
                    if s.dtype==wp.int64:raise TypeError('aliased event i64 unsupported by original')
                    width=s.size//self.nworld;rows=t.size//(self.nworld*width)
                    view=t.reshape((self.nworld*rows,width))
                    wp.launch(_copy_f64 if s.dtype==scalar.wp_real else _copy_i32,(self.nworld,width),inputs=[s.reshape((-1,)),self.slot,view,width,rows],device=s.device,stream=stream)

class MaskedRestorePlan:
    def __init__(self,old,mask,root):
        self.pairs=old.pairs;self.capacity=old.capacity;self.mask=mask;self.original=old
        self.fallback=old.fallback or any(_overlap(t,mask) for _,t in self.pairs)
        self.plan=None if self.fallback else NativePlan('masked_restore',[HEADER],
            [('cf_copy::hold_'+KINDS[s.dtype],[mask,s,t,s.shape[1]],'block') for s,t in self.pairs],self.capacity,root,stage_barriers=False)
    def launch(self,mask,stream):
        if mask is not self.mask:raise ValueError('masked restore owner changed')
        if self.plan:self.plan.launch(stream)
        else:self.original.launch(mask,stream)

class TapeCopyPlan:
    def __init__(self,tape,root):
        pairs=[(tape.flat[k],tape.data[k].reshape((tape.capacity,tape.flat[k].size))) for k in tape.sources]
        self.copy=CopyPlan(pairs,root,tape.index,compile_plan=False);self.fallback=self.copy.fallback;self.stream=tape.stream
        stages=self.copy.stages+[('cf_copy::advance',[tape.index,tape.overflow,tape.capacity],'lane0')]
        barriers=[False]*len(stages)
        if self.copy.stages:barriers[len(self.copy.stages)-1]=True
        self.plan=None if self.fallback else NativePlan('tape_copy_and_advance',[self.copy.source],stages,1,root,stage_barriers=barriers)
        # The fused tape plan owns execution; do not expose a redundant unused
        # copy-only plan to execution-evidence collection.
        self.copy.plan=None
    def launch(self):self.plan.launch(self.stream)

def _tape_capture(self):
    from libero.gpu.host.runtime.step.tape import _token as token
    if (self.capacity,id(self.stream),int(self.stream.cuda_stream),int(self.stream.device.context),id(self.sources),id(self.flat),id(self.data),tuple(self.sources))!=self._config:raise ValueError('prepared tape configuration/keys changed')
    if (token(self.index),token(self.overflow))!=self._counter_tokens:raise ValueError('prepared tape counters changed')
    for k,a in self.sources.items():
        if token(a)!=self.tokens[k] or token(self.data[k])!=self._owners[k] or token(self.flat[k])!=self._flat_tokens[k]:raise ValueError('prepared tape owners changed: '+k)
    self._cuda_tape.launch()

def install(bundle,root):
    """Replace prepared copy plans; preserve producer/consumer boundaries."""
    seen=set(); replaced=[]
    def visit(obj,depth):
        if id(obj) in seen or depth>4 or isinstance(obj,(wp.array,str,int,float,type(None))):return
        seen.add(id(obj))
        if hasattr(obj,'_holds') and hasattr(obj,'requested_mask'):
            for hold in obj._holds.values():
                old=getattr(hold,'_restore_plan',None)
                if old is not None and type(old).__name__=='MaskedRestorePlan':
                    hold._restore_plan=MaskedRestorePlan(old,obj.requested_mask,root);replaced.append('masked_restore')
        from libero.gpu.host.runtime.step.tape import DeviceInputTape
        if isinstance(obj,DeviceInputTape):
            plan=TapeCopyPlan(obj,root)
            if not plan.fallback:
                obj._cuda_tape=plan;obj.capture=types.MethodType(_tape_capture,obj);replaced.append('DeviceInputTape')
            return
        if isinstance(obj,dict):
            for key,value in list(obj.items()):
                if type(value).__name__=='MultiFieldCopyPlan':
                    obj[key]=CopyPlan(value.pairs,root,getattr(value,'slot',None));replaced.append(str(key))
                else:visit(value,depth+1)
        elif hasattr(obj,'__dict__'):
            for name,value in list(vars(obj).items()):
                if type(value).__name__=='MultiFieldCopyPlan':
                    setattr(obj,name,CopyPlan(value.pairs,root,getattr(value,'slot',None)));replaced.append(name)
                else:visit(value,depth+1)
    visit(bundle,0)
    return {'copies':replaced}
