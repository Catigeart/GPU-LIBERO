"""Captured upstream physical stages with GPU control and observation sampling."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
import mujoco_warp as mjw

from .controller import Controller
from .stages import PhysicsStages
from . import kernels


class Runtime:
    def __init__(self, env, model, data, library, *, newton, acceleration=None, mass_factor=None, mass_matrix=None, velocity=None, postconstraint=None, actuation=None, euler=None, transmission=None, kinematics=None, com_pos=None, camlight=None, constraint_build=None, collision=None, queries=None):
        self.env, self.model, self.data, self.stream = env, model, data, env.stream
        self.substeps = env.substeps
        self.newton = newton
        self.physics = PhysicsStages(model, data, newton=newton, acceleration=acceleration,
                                     mass_factor=mass_factor, mass_matrix=mass_matrix, velocity=velocity, postconstraint=postconstraint, actuation=actuation, euler=euler, transmission=transmission, kinematics=kinematics, com_pos=com_pos, camlight=camlight, constraint_build=constraint_build, collision=collision, queries=queries)
        self.controller = Controller(env.built.native_model, model, data, env.built.metadata,
                                     self.stream, library, queries=queries)
        self.integration_count = wp.zeros(env.num_envs, dtype=wp.int64, device=self.stream.device)
        self.state_epoch = wp.zeros_like(self.integration_count)
        self.reset_events = wp.zeros_like(self.integration_count)
        self.newton_attempts = wp.zeros_like(self.integration_count)
        self.position_math = wp.zeros_like(self.integration_count)
        self.intent = wp.full(env.num_envs, 1, dtype=int, device=self.stream.device)
        self.euler = SimpleNamespace(time=data.time, qpos=data.qpos, qvel=data.qvel, capacity=env.num_envs)
        self._control_graph = self._substep_fusion = None
        self._control_graph_owners = []
        self._forward_graph=None
        self._forward_graph_owners=[]
        from mujoco_warp._src import types
        self.forward_fallback=None
        if env.public_forward_implementation=='cuda':
            if model.opt.integrator!=types.IntegratorType.EULER:self.forward_fallback='public forward requires Euler'
            elif model.opt.enableflags & types.EnableBit.SLEEP:self.forward_fallback='sleep remains upstream'
        self.forward_selected='upstream' if self.forward_fallback else env.public_forward_implementation
        if self.forward_selected=='cuda' and mass_factor is not None:mass_factor.prepare_public()
        digest = hashlib.sha256(Path(library).read_bytes()).hexdigest()
        self.manifest = dict(library=str(library), folder=str(Path(library).parent), config=dict(
            physics_engine='mjwarp', real_bits=32, preset=env.identity.get('preset','upstream'), upstream_commit=env.identity['mjwarp_commit'],
            schedule='upstream-step1-osc-step2', solves_per_substep=1,
            graph_conditional=True, controller='maintained CUDA OSC FP32',
            capacity=env.capacities, controller_sha256=digest,
            newton=newton.identity, mass_solve=acceleration.identity if acceleration is not None else None,
            mass_factor=mass_factor.identity if mass_factor is not None else None,
            mass_matrix=mass_matrix.identity if mass_matrix is not None else None,
            velocity=velocity.identity if velocity is not None else None,
            postconstraint=postconstraint.identity if postconstraint is not None else None,
            actuation=actuation.identity if actuation is not None else None,
            euler=euler.identity if euler is not None else None,
            transmission=transmission.identity if transmission is not None else None,
            kinematics=kinematics.identity if kinematics is not None else None,
            com_pos=com_pos.identity if com_pos is not None else None,
            camlight=camlight.identity if camlight is not None else None,
            fluid=env.identity.get('fluid'),
            external_wrench=env.identity.get('external_wrench'),
            constraint_build=env.identity.get('constraint_build'),
            broadphase=env.identity.get('broadphase'),
            plane_collision=env.identity.get('plane_collision'),
            box_collision=env.identity.get('box_collision'),
            convex_collision=env.identity.get('convex_collision'),
            queries=env.identity.get('queries'),
            publication=env.identity.get('publication'),
            reset_forward=dict(requested=env.public_forward_implementation,selected=self.forward_selected,fallback=self.forward_fallback)))
        self.resources = dict(physics_engine='mjwarp', nv=model.nv, nv_pad=model.nv_pad,
                              **env.capacities)

    def forward(self):
        """Refresh derived data, preserving public forward's later factor/solve."""
        if self.forward_selected=='upstream':mjw.forward(self.model,self.data)
        else:
            if self._forward_graph is None:
                with self.newton.capture_owners() as owners,wp.ScopedStream(self.stream),wp.ScopedCapture(stream=self.stream) as cap:
                    self.physics.forward()
                self._forward_graph=cap.graph;self._forward_graph_owners=owners
            wp.capture_launch(self._forward_graph,stream=self.stream)
        wp.launch(kernels.forward_count, self.env.num_envs,
                  [self.newton_attempts, self.position_math], stream=self.stream)

    def enqueue_substep(self, policy_step):
        self.physics.step1()
        self.controller.enqueue(policy_step)
        self.physics.step2()
        wp.launch(kernels.advance_count, self.env.num_envs,
                  [self.integration_count, self.state_epoch, self.newton_attempts,
                   self.position_math], stream=self.stream)
        self.env.observer.sample(self.env.observation_fields(), stream=self.stream)

    def prepare_control_graph(self):
        if self._control_graph is None:
            with self.newton.capture_owners() as owners, wp.ScopedStream(self.stream), \
                    wp.ScopedCapture(stream=self.stream) as capture:
                for k in range(self.substeps):
                    self.enqueue_substep(k == 0)
            self._control_graph = capture.graph
            self._control_graph_owners = owners
        self._substep_fusion = self
        return self

    def enqueue_action(self, actions):
        with wp.ScopedStream(self.stream):
            self.controller.buffers['action'].assign(np.asarray(actions, np.float32))
            self.prepare_control_graph()
            wp.capture_launch(self._control_graph, stream=self.stream)
            self.env._held_contacts.clear()
            self.env.observer.notify_control_graph_replay(substeps=self.substeps, stream=self.stream)

    def download_state(self):
        return {name: getattr(self.data, name).numpy() for name in
                ('time', 'qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl')}
