"""Original Python Observable extension protocol over live GPU facade state.

This path is enabled only by user sensor/processor extensions. Default built-in
sampling stays on device. No CPU MuJoCo dynamics, forward or step is used here.
"""
from collections import OrderedDict
import numpy as np

ATTRIBUTES=("sensor","corrupter","filter","delayer","sampling_rate","enabled","active")

class HostObservableCompatibility:
    def __init__(self,env):
        self.env=env
        # Observable constructors validate sensors by executing them. Caller must
        # bind this only after the actual GPU initial forward has completed.
        setup=env._setup_observables
        if (getattr(env,"physics_backend",None)=="gpu" and
                setup is getattr(env,"_gpu_model_only_setup_observables_shim",None)):
            setup=env._gpu_original_setup_observables
            if not callable(setup):raise TypeError("retained original observable setup must be callable")
        self.observables=setup()
        self.cache=getattr(env,"_obs_cache",{})
        env._observables=self.observables
        env._obs_cache=self.cache

    def modify(self,name,attribute,modifier):
        assert name in self.observables,"No valid observable with name {} found. Options are: {}".format(name,set(self.observables))
        if attribute not in ATTRIBUTES:raise ValueError("Invalid observable attribute specified: "+str(attribute))
        getattr(self.observables[name],"set_"+attribute)(modifier)

    def add(self,observable):
        assert observable.name not in self.observables,"Observable {} already exists".format(observable.name)
        self.observables[observable.name]=observable

    def update(self,timestep,force=False):
        for observable in self.observables.values():observable.update(timestep,self.cache,force=force)

    def reset(self):
        self.cache.clear()
        for observable in self.observables.values():observable.reset()

    def pack(self):
        observations=OrderedDict();groups=OrderedDict()
        for name,observable in self.observables.items():
            if observable.is_enabled() and observable.is_active():
                value=observable.obs
                observations[name]=value
                from robosuite import macros
                if observable.modality!="image" or macros.CONCATENATE_IMAGES:
                    groups.setdefault(observable.modality+"-state",[]).append(
                        np.array([value]) if np.ndim(value)==0 else np.asarray(value))
        for name,group in groups.items():observations[name]=np.concatenate(group,axis=-1)
        return observations

    def snapshot_owner(self):
        import copy
        # Python callbacks/filter histories may be stateful; transfer their actual
        # owner objects rather than pretend arbitrary closures are serializable.
        fields=("_time_since_last_sample","_current_delay","_current_observed_value","_sampled")
        return {"owner":self,"environment":self.env,
                "objects":tuple(self.observables.items()),
                "state":{name:{field:copy.deepcopy(getattr(obs,field)) for field in fields}
                         for name,obs in self.observables.items()},"cache":copy.deepcopy(self.cache)}

    @staticmethod
    def restore_owner(packet,environment):
        import copy
        owner=packet["owner"]
        if packet["environment"] is not environment or owner.env is not environment:
            raise ValueError("custom observable callbacks require their original task shell owner")
        if tuple(owner.observables.items())!=packet["objects"]:
            raise ValueError("custom observable object ownership changed during transport")
        for name,fields in packet["state"].items():
            for field,value in fields.items():setattr(owner.observables[name],field,copy.deepcopy(value))
        owner.cache.clear();owner.cache.update(copy.deepcopy(packet["cache"]))
        environment._observables=owner.observables;environment._obs_cache=owner.cache
        return owner
