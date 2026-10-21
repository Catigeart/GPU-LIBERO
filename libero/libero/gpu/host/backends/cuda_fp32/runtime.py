"""Owned CUDA Graph schedule:25 ordinary substeps, one Newton solve each."""
from . import device as cuda
from .bridges import Bridges
from .physics import Physics,Products
from .controller import Controller,controller_product


class Runtime:
    def __init__(self,env,*,fast_compile=True):
        self.env,self.model,self.data,self.stream=env,env.model,env.data,env.stream
        self.substeps=env.substeps;products=Products(env.build_root,fast_compile)
        self.physics=Physics(self.model,self.data,self.stream,products)
        self.newton=self.physics.newton
        self.controller=Controller(env.built.native_model,self.model,self.data,env.built.metadata,
            self.stream,controller_product(env.build_root,fast_compile=fast_compile))
        self.bridge=Bridges(products.folder/'cuda_bridges_fp32.so',self.stream)
        for name in ('integration_count','state_epoch','reset_events','newton_attempts','position_math'):
            setattr(self,name,cuda.zeros(env.num_envs,dtype=cuda.int64))
        self.intent=cuda.full(env.num_envs,1,dtype=int)
        self._control_graph=self._forward_graph=self._substep_fusion=None
        self._step_graphs={}
        self.manifest=dict(physics_engine='cuda_fp32',real_bits=32,schedule='step1-osc-step2',
            solves_per_substep=1,graph_conditional=True,warp_dependency=False)
        self.resources=dict(physics_engine='cuda_fp32',nv=self.model.nv,nv_pad=self.model.nv_pad,**env.capacities)

    def count(self,kind,mask=None):
        self.bridge.call('count',self.env.num_envs,kind,0 if mask is None else mask.ptr,
            *[getattr(self,k).ptr for k in ('integration_count','state_epoch','reset_events','newton_attempts','position_math')])

    def forward(self):
        if self._forward_graph is None:
            with cuda.Graph(self.stream,owners=[self.physics]) as graph:self.physics.forward();self.count(1)
            self._forward_graph=graph
        self._forward_graph.launch()

    def prepare_control_graph(self):
        if not self.env.observer.graph_eligible:return self
        if self._control_graph is None:
            with cuda.Graph(self.stream,owners=[self.physics,self.controller,self.env.observer]) as graph:
                for k in range(self.substeps):
                    self.physics.step1();self.controller.enqueue(k==0);self.physics.step2();self.count(0)
                    self.env.observer.sample(self.env.observation_fields(),stream=self.stream)
            self._control_graph=graph
        self._substep_fusion=self
        return self

    def enqueue_action(self,actions):
        with self.stream:
            self.controller.buffers['action'].assign(actions)
            self.env._held_contacts.clear()
            if self.env.observer.graph_eligible:
                self.prepare_control_graph();self._control_graph.launch()
                self.env.observer.notify_control_graph_replay(substeps=self.substeps,stream=self.stream)
            else:
                # User sensors/processors execute at their original substep
                # boundaries. Physics, including the Newton conditional, stays
                # in reusable device Graphs; no host solver or Warp fallback.
                self.discard_control_graph()
                for k in range(self.substeps):
                    policy=k==0
                    if policy not in self._step_graphs:
                        with cuda.Graph(self.stream,owners=[self.physics,self.controller]) as graph:
                            self.physics.step1();self.controller.enqueue(policy);self.physics.step2();self.count(0)
                        self._step_graphs[policy]=graph
                    self._step_graphs[policy].launch()
                    self.env.observer.sample(self.env.observation_fields(),stream=self.stream)

    def discard_control_graph(self):
        if self._control_graph is not None:self._control_graph.close();self._control_graph=None
        self._substep_fusion=None

    def download_state(self):
        return {name:getattr(self.data,name).numpy() for name in ('time','qpos','qvel','qacc','qacc_warmstart','ctrl')}

    def close(self):
        for graph in self._step_graphs.values():graph.close()
        self._step_graphs.clear()
        for name in ('_control_graph','_forward_graph'):
            graph=getattr(self,name)
            if graph is not None:graph.close();setattr(self,name,None)
        self._substep_fusion=None
