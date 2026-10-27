"""Cold typed history/tape copy plans; original sampling boundaries retained."""
import libero.gpu.precision as scalar
import types
import warp as wp
from libero.gpu.host.assembly.compose.backup import MultiFieldCopyPlan, _overlap
from libero.gpu.host.assembly.compose.compose import _load
from libero.gpu.host.runtime.dynamics.runtime_adapter import _NoForwardHold, RuntimePostAdapter, _hold_unrequested
from libero.gpu.host.runtime.step.tape import DeviceInputTape, _advance, _token


class MaskedRestorePlan:
    def __init__(self, pairs, capacity):
        self.pairs = tuple(pairs)
        self.capacity = capacity
        targets = [t for _, t in self.pairs]
        self.fallback = any(_overlap(t, s) for t in targets for s, _ in self.pairs)
        self.fallback |= any(_overlap(t, u) for i, t in enumerate(targets) for u in targets[i+1:])
        args = ['mask: wp.array(dtype=wp.int32)']
        body = ['    w = wp.tid()', '    if mask[w] != 0: return']
        names = {scalar.wp_real:scalar.name, wp.int32:'int32', wp.int64:'int64'}
        for i, (s, t) in enumerate(self.pairs):
            if s.shape != t.shape or s.shape[0] != capacity or s.dtype != t.dtype:
                raise ValueError('restore schema changed')
            dtype = names[s.dtype]
            args += [f's{i}: wp.array(dtype=wp.{dtype}, ndim=2)', f't{i}: wp.array(dtype=wp.{dtype}, ndim=2)']
            body += [f'    for j in range({s.shape[1]}):', f'        t{i}[w,j] = s{i}[w,j]']
        code = '@wp.kernel(enable_backward=False)\ndef restore_fields('+', '.join(args)+'):\n'+'\n'.join(body)+'\n'
        self.kernel = _load('hold_restore', code, {'wp':wp}).restore_fields
        self.tokens = tuple((_token(s), _token(t)) for s,t in self.pairs)

    def launch(self, mask, stream):
        if tuple((_token(s),_token(t)) for s,t in self.pairs) != self.tokens:
            raise ValueError('restore retained owner changed')
        if mask.dtype != wp.int32 or mask.shape != (self.capacity,) or mask.device != stream.device:
            raise ValueError('restore mask schema changed')
        fallback = self.fallback or any(_overlap(t,mask) for _,t in self.pairs)
        if fallback:
            for s,t in self.pairs:
                wp.launch(_hold_unrequested, self.capacity, inputs=[mask,s,t,t.shape[1]], device=stream.device,stream=stream)
        elif self.pairs:
            wp.launch(self.kernel,self.capacity,inputs=[mask]+[a for pair in self.pairs for a in pair],device=stream.device,stream=stream)


class FusedNoForwardHold(_NoForwardHold):
    def __init__(self, fields, capacity, device):
        super().__init__(fields, capacity, device)
        self._snapshot_plan = MultiFieldCopyPlan([(a,self.before[k]) for k,a in self.views.items()])
        self._restore_plan = MaskedRestorePlan([(self.before[k],a) for k,a in self.views.items()],capacity)

    def snapshot(self, stream):
        self.validate(self.fields)
        self._snapshot_plan.launch(stream)

    def restore_unrequested(self, mask, stream):
        self.validate(self.fields)
        self._restore_plan.launch(mask,stream)


def install_post_capture(adapter):
    adapter._fusion_epoch_copy = MultiFieldCopyPlan([(adapter.n_expected,adapter.expected_consumed),
        (adapter.state_epoch,adapter.state_consumed),(adapter.forward_epoch,adapter.forward_consumed)])
    adapter.capture_before_constraint = types.MethodType(_post_capture,adapter)
    return adapter


def _post_capture(self, stream=None):
    self.validate(stream)
    self._fusion_epoch_copy.launch(self.stream)


class FusedRuntimePostAdapter(RuntimePostAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_post_capture(self)


def event_warning_plan(runtime):
    return MultiFieldCopyPlan([(runtime.recovery.warning_number.reshape((-1,)),runtime.event_warnings),
        (runtime.recovery.warning_lastinfo.reshape((-1,)),runtime.event_lastinfo)],slot=runtime.event_slot)


def launch_event_warnings(runtime):
    runtime._fusion_warning_copy.launch(runtime.stream)


class TapeCopyPlan:
    def __init__(self,tape):
        self.tape=tape
        pairs=[(tape.flat[k],tape.data[k].reshape((tape.capacity,tape.flat[k].size))) for k in tape.sources]
        targets=[t for _,t in pairs]
        self.fallback=any(_overlap(t,s) for t in targets for s,_ in pairs)
        self.fallback |= any(_overlap(t,tape.index) for t in targets)
        self.fallback |= any(_overlap(t,u) for i,t in enumerate(targets) for u in targets[i+1:])
        args=['index: wp.array(dtype=wp.int32)', 'capacity: int']
        body=['    i=wp.tid()','    row=index[0]','    if row < 0 or row >= capacity: return']
        names={scalar.wp_real:scalar.name,wp.int32:'int32',wp.int64:'int64'}
        for j,(s,t) in enumerate(pairs):
            dtype=names[s.dtype]
            args += [f's{j}: wp.array(dtype=wp.{dtype})',f't{j}: wp.array(dtype=wp.{dtype},ndim=2)']
            body += [f'    if i < s{j}.shape[0]: t{j}[row,i]=s{j}[i]']
        code='@wp.kernel(enable_backward=False)\ndef tape_fields('+', '.join(args)+'):\n'+'\n'.join(body)+'\n'
        self.kernel=_load('tape_fields',code,{'wp':wp}).tape_fields
        self.inputs=[tape.index,tape.capacity]+[a for pair in pairs for a in pair]
        self.dim=max((s.size for s,_ in pairs),default=0)

    def launch(self):
        if self.dim:
            wp.launch(self.kernel,self.dim,inputs=self.inputs,device=self.tape.stream.device,stream=self.tape.stream)


def _tape_capture(self):
    if (self.capacity,id(self.stream),int(self.stream.cuda_stream),int(self.stream.device.context),
            id(self.sources),id(self.flat),id(self.data),tuple(self.sources)) != self._config:
        raise ValueError('prepared tape configuration/keys changed')
    if (_token(self.index),_token(self.overflow)) != self._counter_tokens:
        raise ValueError('prepared tape counters changed')
    for k,a in self.sources.items():
        if (_token(a)!=self.tokens[k] or _token(self.data[k])!=self._owners[k] or _token(self.flat[k])!=self._flat_tokens[k]):
            raise ValueError('prepared tape source/target changed: '+k)
    self._fusion_tape_copy.launch()
    wp.launch(_advance,1,inputs=[self.index,self.overflow,self.capacity],device=self.stream.device,stream=self.stream)


def install_tape_fusion(tape):
    plan=TapeCopyPlan(tape)
    # Unusual admitted aliases retain the original ordered copy/advance method.
    if not plan.fallback:
        tape._fusion_tape_copy=plan
        tape.capture=types.MethodType(_tape_capture,tape)
    return tape


class FusedDeviceInputTape(DeviceInputTape):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        install_tape_fusion(self)
