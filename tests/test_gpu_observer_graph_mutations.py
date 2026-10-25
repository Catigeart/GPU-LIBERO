"""Observer mutation lifetime tests using real CPU Observable and fake arrays.

No Warp import, GPU allocation, MuJoCo data, physics, rendering or compilation.
The adapter class is loaded from its actual AST; callback clocks use the installed
CPU reference robosuite implementation, not a replacement sampler.
"""
import ast
from contextlib import nullcontext
import itertools
import importlib.util
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


BASE=Path(__file__).resolve().parents[1]/'libero/libero/gpu/host/adapters/observations'
installed=importlib.util.find_spec('robosuite')
if installed is None:raise unittest.SkipTest('installed robosuite Observable source required')
CPU=Path(next(iter(installed.submodule_search_locations)))/'utils/observables.py'
ORIGINAL=runpy.run_path(str(CPU))
SCHEMA=runpy.run_path(str(BASE/'schema.py'))
COMPAT=ModuleType('observer_graph_test.observable_compatibility')
exec(compile((BASE/'observable_compatibility.py').read_text(),str(BASE/'observable_compatibility.py'),'exec'),COMPAT.__dict__)


class Array:
    pointers=itertools.count(1000)

    def __init__(self,value,dtype=np.float64):
        self.value=np.asarray(value,dtype=dtype).copy()
        self.ptr=next(self.pointers)
        self.dtype=self.value.dtype
        self.shape=self.value.shape
        self.strides=self.value.strides
        self.device=SimpleNamespace(is_cuda=True)
        self.is_contiguous=True
        self.assignments=0

    def numpy(self):return self.value.copy()

    def assign(self,value):
        np.copyto(self.value,np.asarray(value,dtype=self.dtype).reshape(self.shape))
        self.assignments+=1


class Warp:
    def __init__(self):self.waits=[]
    def synchronize_stream(self,stream):self.waits.append(stream)
    def get_stream(self,device):return SimpleNamespace(device=device,is_capturing=False)
    def ScopedStream(self,stream):return nullcontext()


def load_adapter(warp):
    node=next(n for n in ast.parse((BASE/'adapter.py').read_text()).body
              if isinstance(n,ast.ClassDef) and n.name=='GpuObservationAdapter')
    namespace=dict(np=np,wp=warp,nullcontext=nullcontext,
                   __name__='observer_graph_test.adapter',__package__='observer_graph_test')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(BASE/'adapter.py'),'exec'),namespace)
    return namespace['GpuObservationAdapter']


class Environment:
    def __init__(self,value):self.value=value;self.trace=[];self._obs_cache={}

    def _setup_observables(self):
        result={}
        for name in ('first','second'):
            @ORIGINAL['sensor'](modality='proprio')
            def sensor(cache,name=name):
                self.trace.append(name)
                return np.array([self.value if name=='first' else float(cache.get('first',0))*2])
            result[name]=ORIGINAL['Observable'](name,sensor,sampling_rate=20)
        return result


class Filter:
    def __init__(self):self.calls=0
    def __call__(self,value):self.calls+=1;return value+self.calls


class Mutations(unittest.TestCase):
    def setUp(self):
        self.modules=patch.dict(sys.modules,{'observer_graph_test':ModuleType('observer_graph_test'),
            'observer_graph_test.observable_compatibility':COMPAT})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.wp=Warp()
        self.cls=load_adapter(self.wp)
        self.stream=SimpleNamespace(is_capturing=False)

    def adapter(self,worlds=2,images=False):
        ob=self.cls.__new__(self.cls)
        ob.wp=self.wp
        ob._scalar=np.float64
        spec=SCHEMA['ObservableSpec']
        image=({'name':'camera_image','shape':(1,1,3),'period':.05,'enabled':True,'active':True},) if images else ()
        ob.plan=SCHEMA['ObservationPlan']((spec('first','proprio','qpos',1,indices=(0,)),
            spec('second','proprio','qpos',1,indices=(1,))),image,('first','second')+tuple(s['name'] for s in image))
        ob.closed=False;ob.valid=True;ob.nworld=worlds;ob.device='fake';ob.metadata={'model_timestep':.002}
        ob.visual_sources={};ob.scene_cache=None;ob.scene_packer=None;ob.bridges={}
        ob._public_bridge=None;ob._pending_renderers=set();ob._owned_viewers={}
        ob.host_environments={};ob.host_observables={};ob._last_sample=(.002,False)
        low=SimpleNamespace(plan=ob.plan,last_stream=None,_packing_recipe=object())
        low.meta=[Array([0,0],np.int32) for _ in range(11)]
        low.meta[9]=Array([1,1],np.int32);low.meta[10]=Array([.05,.05])
        for name in ('cache','observed','sidequat'):setattr(low,name,Array(np.arange(worlds*2)+5.))
        low.elapsed=Array(np.full(worlds*2,.019))
        for name in ('sampled','present','events'):setattr(low,name,Array(np.ones(worlds*2),np.int32))
        low.all_worlds=Array(np.ones(worlds),np.int32)
        def reset(mask,stream):
            chosen=np.ones(worlds) if mask is None else mask.numpy()
            for w in range(worlds):
                if chosen[w]:
                    for name in ('elapsed','cache','observed','present'):
                        getattr(low,name).value[w*2:w*2+2]=0
        low.reset=reset
        ob.lowdim=low;ob.raw_packer=SimpleNamespace(buffer=Array([0.]))
        if images:
            scene=SimpleNamespace(last_stream=None,specs=image,ncamera=1,
                periods=Array([.05]),enabled=Array([1],np.int32),
                elapsed=Array(np.full(worlds,.019)),sampled=Array(np.ones(worlds),np.int32),
                generation=Array(np.ones(worlds),np.int32),parents=Array([-1],np.int32),
                parent_generation=Array(np.zeros(worlds),np.int32),captured=Array(np.ones(worlds),np.int32),
                scenes=Array(np.zeros(worlds)),mask=Array(np.ones(worlds),np.int32),
                images=[[np.ones((1,1,3))] for _ in range(worlds)],
                rendered_generation=np.zeros((worlds,1),np.int32))
            ob.scene_cache=scene;ob.scene_packer=SimpleNamespace(buffer=Array([0.]))
        ob._host_sampling_required=False;ob._host_sampling_reasons=[]
        ob._sample_event_token=0;ob._sample_event_hook=None;ob._graph_guard=None
        return ob

    def test_enabled_and_rate_keep_captured_metadata_and_cpu_reset_quirk(self):
        ob=self.adapter(images=True)
        roots=(ob.lowdim.meta[9],ob.lowdim.meta[10],ob.scene_cache.enabled,ob.scene_cache.periods)
        old=(ob.lowdim.elapsed.numpy(),ob.lowdim.sampled.numpy(),ob.lowdim.observed.numpy(),ob.lowdim.cache.numpy())
        ob.modify_observable('first','sampling_rate',10,stream=self.stream)
        self.assertEqual(ob.lowdim.meta[10].numpy().tolist(),[.1,.05])
        for actual,want in zip((ob.lowdim.elapsed,ob.lowdim.sampled,ob.lowdim.observed,ob.lowdim.cache),old):
            np.testing.assert_array_equal(actual.numpy(),want)
        ob.modify_observable('first','enabled',False,stream=self.stream)
        np.testing.assert_array_equal(ob.lowdim.elapsed.numpy(),[0,.019,0,.019])
        np.testing.assert_array_equal(ob.lowdim.observed.numpy(),[0,6,0,8])
        np.testing.assert_array_equal(ob.lowdim.sampled.numpy(),old[1])
        np.testing.assert_array_equal(ob.lowdim.cache.numpy(),old[3])
        ob.modify_observable('camera_image','sampling_rate',5,stream=self.stream)
        ob.modify_observable('camera_image','enabled',False,stream=self.stream)
        ob.modify_observable('second','active',False,stream=self.stream)
        self.assertNotIn('second',ob.active_observables)
        self.assertTrue(all(a is b for a,b in zip(roots,
            (ob.lowdim.meta[9],ob.lowdim.meta[10],ob.scene_cache.enabled,ob.scene_cache.periods))))
        np.testing.assert_array_equal(ob.scene_cache.sampled.numpy(),[1,1])
        self.assertTrue(ob.graph_eligible)

    def test_actual_cpu_set_enabled_retains_sampled_and_filter_history(self):
        env=Environment(4);cpu=env._setup_observables()['first'];filt=Filter()
        cpu.set_filter(filt);cpu.update(.002,{},force=True)
        self.assertTrue(cpu._sampled)
        cpu.set_sampling_rate(10)
        self.assertEqual(cpu._time_since_last_sample,.002)
        self.assertEqual(cpu.obs,5)
        cpu.set_enabled(False)
        self.assertEqual(cpu._time_since_last_sample,0)
        self.assertEqual(cpu.obs,0)
        self.assertTrue(cpu._sampled)
        self.assertEqual(filt.calls,1)

    def test_capture_has_no_event_and_real_replay_records_submission(self):
        ob=self.adapter();events=[];ob.bind_sample_event_hook(events.append)
        capture=SimpleNamespace(is_capturing=True)
        ob._last_sample=None
        ob._after_gpu_sample(.002,stream=capture)
        self.assertIsNone(ob._last_sample);self.assertEqual(events,[])
        with self.assertRaisesRegex(RuntimeError,'during graph capture'):
            ob.notify_control_graph_replay(substeps=25,stream=capture)
        self.assertIsNone(ob._last_sample);self.assertEqual(events,[])
        ob.notify_control_graph_replay(substeps=0,stream=self.stream)
        self.assertIsNone(ob._last_sample);self.assertEqual(events,[])
        ob.notify_control_graph_replay(substeps=25,stream=self.stream)
        self.assertEqual(ob.sample_event_token,25)
        self.assertEqual(events,[25]);self.assertEqual(ob._last_sample,(.002,False))

    def test_native_binding_checked_at_install_and_capture_not_replay(self):
        ob=self.adapter();checks=[];current=[True]
        def guard():
            checks.append('binding')
            return current[0]
        ob.bind_graph_guard(guard)
        self.assertEqual(checks,['binding'])
        for _ in range(3):
            self.assertTrue(ob.graph_eligible)
            ob.notify_control_graph_replay(stream=self.stream)
        self.assertEqual(checks,['binding'])
        capture=SimpleNamespace(is_capturing=True)
        ob.assert_sampling_allowed(capture)
        self.assertEqual(checks,['binding','binding'])
        current[0]=False
        with self.assertRaisesRegex(RuntimeError,'binding changed before capture'):
            ob.assert_sampling_allowed(capture)

    def test_invalid_native_binding_rejected_at_cold_install(self):
        ob=self.adapter()
        with self.assertRaisesRegex(ValueError,'binding differs at installation'):
            ob.bind_graph_guard(lambda:False)
        self.assertIsNone(ob._graph_guard)

    def test_closed_or_failed_observer_cannot_replay(self):
        ob=self.adapter()
        ob.valid=False;self.assertFalse(ob.graph_eligible)
        with self.assertRaisesRegex(RuntimeError,'quarantined'):
            ob.notify_control_graph_replay(stream=self.stream)
        ob.valid=True;ob.closed=True;self.assertFalse(ob.graph_eligible)

    def test_metadata_without_explicit_stream_uses_retained_sampling_stream(self):
        ob=self.adapter();ob.lowdim.last_stream=self.stream
        ob.modify_observable('first','sampling_rate',10)
        self.assertIs(ob.lowdim.last_stream,self.stream)
        self.assertIn(self.stream,self.wp.waits)
        self.assertTrue(ob.graph_eligible)

    def test_custom_callback_switch_is_sticky_and_runs_after_event_invalidation(self):
        ob=self.adapter();ob.host_environments={0:Environment(4),1:Environment(7)}
        ob.modify_observable('first','filter',Filter(),stream=self.stream)
        self.assertTrue(ob.requires_host_sampling);self.assertFalse(ob.graph_eligible)
        events=[]
        ob.bind_sample_event_hook(lambda token:events.append(('event',token)))
        for world,compat in ob.host_observables.items():
            observable=compat.observables['first'];old=observable._sensor
            def sensor(cache,old=old,world=world):
                events.append(('sensor',world));return old(cache)
            sensor.__modality__='proprio';observable.set_sensor(sensor)
        events.clear()
        ob._after_gpu_sample(.002,True,Array([1,0],np.int32),self.stream)
        self.assertEqual(events[:2],[('event',1),('sensor',0)])
        self.assertNotIn(('sensor',1),events)
        self.assertEqual(ob.host_observables[0].observables['first'].obs,5)
        ob.modify_observable('first','filter',None,stream=self.stream)
        self.assertFalse(ob.graph_eligible)
        with self.assertRaisesRegex(RuntimeError,'uncaptured'):
            ob.assert_sampling_allowed(SimpleNamespace(is_capturing=True))

    def test_selected_reset_preserves_original_sampled_and_processor_history(self):
        ob=self.adapter();ob.host_environments={0:Environment(4),1:Environment(7)}
        ob.modify_observable('first','filter',Filter(),stream=self.stream)
        ob._after_gpu_sample(.002,True,stream=self.stream)
        before=ob.host_observables[1].observables['first']._time_since_last_sample
        first=ob.host_observables[0].observables['first'];filt=first._filter
        self.assertTrue(first._sampled);calls=filt.calls
        ob.reset(Array([1,0],np.int32),self.stream)
        self.assertTrue(first._sampled);self.assertEqual(first.obs,0)
        self.assertEqual(filt.calls,calls)
        self.assertEqual(ob.host_observables[1].observables['first']._time_since_last_sample,before)

    def test_host_rate_enabled_and_delay_preserve_actual_observable_histories(self):
        ob=self.adapter(worlds=1);ob.host_environments={0:Environment(4)}
        filt=Filter();ob.modify_observable('first','filter',filt,stream=self.stream)
        first=ob.host_observables[0].observables['first']
        delays=[]
        def delay():delays.append('draw');return .01
        ob.modify_observable('first','delayer',delay,stream=self.stream)
        ob._after_gpu_sample(.002,True,stream=self.stream)
        elapsed=first._time_since_last_sample;sampled=first._sampled;value=first.obs
        ob.modify_observable('first','sampling_rate',10,stream=self.stream)
        self.assertEqual(first._time_since_last_sample,elapsed)
        self.assertEqual(first._sampled,sampled);self.assertEqual(first.obs,value)
        calls=filt.calls;draws=len(delays)
        ob.modify_observable('first','enabled',False,stream=self.stream)
        self.assertEqual(first._time_since_last_sample,0);self.assertEqual(first.obs,0)
        self.assertEqual(first._sampled,sampled);self.assertEqual(len(delays),draws+1)
        ob._after_gpu_sample(.002,True,stream=self.stream)
        self.assertEqual(filt.calls,calls)  # disabled force does not call sensor/filter
        ob.modify_observable('first','enabled',True,stream=self.stream)
        self.assertEqual(filt.calls,calls);self.assertEqual(first._current_delay,.01)

    def test_added_observable_has_independent_world_clock_and_copies(self):
        ob=self.adapter();ob.host_environments={0:Environment(4),1:Environment(7)}
        @ORIGINAL['sensor'](modality='custom')
        def sensor(cache):return np.array([2.])
        value=ORIGINAL['Observable']('added',sensor,filter=Filter())
        ob.add_observable(value,stream=self.stream)
        first=ob.host_observables[0].observables['added'];second=ob.host_observables[1].observables['added']
        self.assertIs(first,value);self.assertIsNot(first,second)
        self.assertIsNot(first._filter,second._filter)
        ob._after_gpu_sample(.002,True,Array([1,0],np.int32),self.stream)
        self.assertTrue(first._sampled);self.assertFalse(second._sampled)
        self.assertEqual(first._filter.calls,1);self.assertEqual(second._filter.calls,0)

    def test_close_drops_direct_cycles_hooks_and_registration_even_if_renderer_fails(self):
        ob=self.adapter();events=[]
        ob.bind_sample_event_hook(events.append);ob.bind_graph_guard(lambda:True)
        baseline=lambda *args:None
        sampler=SimpleNamespace(observer=ob,baseline_sample=baseline,owner_arrays=(ob.lowdim.cache,),plan=object())
        ob._cuda_direct_sampler=sampler;ob.sample=lambda *args:sampler
        ob._cuda_batch_observations_registration={'bundle':object()}
        def fail():raise ValueError('renderer close')
        ob.bridges['renderer']=SimpleNamespace(close=fail)
        with self.assertRaisesRegex(ValueError,'renderer close'):ob.close()
        self.assertTrue(ob.closed);self.assertFalse(ob.graph_eligible)
        self.assertIs(ob.sample,baseline);self.assertIsNone(sampler.observer)
        self.assertIsNone(sampler.plan);self.assertIsNone(ob._sample_event_hook)
        self.assertIsNone(ob._graph_guard);self.assertIsNone(ob._cuda_batch_observations_registration)
        ob.close()

    def test_continuation_keeps_sticky_host_policy_and_submission_token(self):
        module=ModuleType('observer_graph_test.continuation')
        module.snapshot=lambda adapter,stream:{'opaque_state':object()}
        module.restore=lambda adapter,packet,stream:None
        ob=self.adapter();ob._require_host_sampling('custom filter');ob._sample_event_token=9
        with patch.dict(sys.modules,{'observer_graph_test.continuation':module}):
            packet=ob.snapshot_continuation(self.stream)
            fresh=self.adapter();fresh.restore_continuation(packet,self.stream)
        self.assertTrue(fresh.requires_host_sampling);self.assertFalse(fresh.graph_eligible)
        self.assertEqual(fresh.sample_event_token,9)


if __name__=='__main__':unittest.main()
