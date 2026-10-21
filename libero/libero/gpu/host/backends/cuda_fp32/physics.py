# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent host schedule for the admitted rigid dense FP32 algorithms.

The CUDA mathematical leaves are shared with the qualified migration backend.
This module owns contexts and conditional Graphs and imports no Warp modules.
All pointer bindings are constructed once, before recording any execution.
"""
from pathlib import Path
from types import SimpleNamespace
from . import device as cuda, constants as types
from .data import newton_workspace
from .fused_tail import FusedAdvance
from .fused_position import FusedCenteredMass
from .fused_motion import FusedMotion
from libero.gpu.host.runtime.dynamics import (
    mjwarp_motion as motion,mjwarp_kinematics as kinematics,mjwarp_com_pos as com_pos,
    mjwarp_camlight as camlight,mjwarp_mass_matrix as mass_matrix,mjwarp_mass_factor as mass_factor,
    mjwarp_mass_solve as mass_solve,mjwarp_transmission as transmission,mjwarp_velocity as velocity,
    mjwarp_passive as passive,mjwarp_fluid as fluid,mjwarp_bias as bias,mjwarp_queries as queries,
    mjwarp_constraint_build as constraints,mjwarp_actuation as actuation,mjwarp_smooth_force as smooth_force,
    mjwarp_external_wrench as external,mjwarp_postconstraint as postconstraint,mjwarp_euler as euler,
    mjwarp_publication as publication)
from libero.gpu.host.runtime.collision import (mjwarp_broadphase as broadphase,mjwarp_convex_collision as convex,
    mjwarp_plane as plane,mjwarp_box as box)
from libero.gpu.host.runtime.newton import (mjwarp_hessian as hessian,mjwarp_direction as direction,
    mjwarp_constraint as constraint,mjwarp_linesearch as linesearch,mjwarp_solver_state as solver_state)


class Products:
    def __init__(self,build_root,fast_compile=True):
        self.folder=Path(build_root)/'cmake-control'/('sm80-dev' if fast_compile else 'sm80-release')
    def __call__(self,component):
        path=self.folder/('mjwarp_'+component+'_fp32.so')
        if not path.is_file():raise RuntimeError('Prepare CUDA component offline: '+component)
        return path


class Newton:
    def __init__(self,m,d,stream,products):
        self.model,self.data,self.stream=m,d,stream;self.context=c=newton_workspace(m,d)
        self.assembly=hessian.HessianPlan(m,d,products('hessian'),stream).bind(c)
        self.direction=direction.DirectionPlan(m,d,products('direction'),stream).bind(c)
        self.constraint=constraint.ConstraintPlan(m,d,products('constraint'),stream).bind(c)
        self.constraint.bind_gradient(self.direction)
        self.linesearch=linesearch.LinesearchPlan(m,d,products('linesearch'),stream).bind(c)
        self.linesearch.prepare_fusion()
        self.state=solver_state.SolverStatePlan(m,d,products('solver_state'),stream).bind(c)
        self.nsolving=cuda.zeros(1,dtype=int);self.state.bind_stopping(self.nsolving)
        self.state.bind_initialization(c.grad,c.ls_exhausted,self.nsolving)

    def gradient(self):
        self.constraint.enqueue_gradient();self.assembly.enqueue();self.direction.enqueue(stage=2)

    def solve(self):
        self.state.initialize_fused();self.gradient()
        if self.model.opt.iterations:
            with cuda.While(self.nsolving):
                self.linesearch.enqueue_fused();self.gradient();self.state.stop()


class Collision:
    def __init__(self,m,d,stream,products):
        self.model,self.data=m,d
        self.context=ctx=SimpleNamespace(collision_pair=cuda.empty(d.naconmax,dtype=cuda.vec2i),
            collision_pairid=cuda.empty(d.naconmax,dtype=cuda.vec2i),collision_worldid=cuda.empty(d.naconmax,dtype=int))
        self.broadphase=broadphase.BroadphaseNumerics(m,d,ctx,products('broadphase'),stream,admission=broadphase.qualify(m,d,implementation='cuda'))
        self.convex=convex.ConvexContacts(m,d,ctx,products('convex'),stream,admission=convex.qualify(m,d,implementation='cuda'))
        self.planes=plane.PlaneContacts(m,d,ctx,products('plane'),stream,admission=plane.qualify(m,d,implementation='cuda'))
        self.boxes=box.BoxContacts(m,d,ctx,products('box'),stream,admission=box.qualify(m,d,implementation='cuda'))
        # The admitted production sequence has no residual primitive/SDF path.
        table=types.MJ_COLLISION_TABLE.copy()
        if m.has_boxbox_margin or m.opt.disableflags&types.DisableBit.NATIVECCD:table[(6,6)]=types.CollisionType.PRIMITIVE
        for (a,b),family in table.items():
            index=int(a)*(2*len(types.GeomType)-int(a)-1)//2+int(b)
            if m.geom_pair_type_count[index] and family==types.CollisionType.PRIMITIVE and a!=0 and (a,b) not in ((2,6),(3,6),(6,6)):
                raise ValueError('Unported primitive family: '+str((a,b)))

    def enqueue(self):
        m,d=self.model,self.data;d.nacon.zero_()
        if m.opt.disableflags&(types.DisableBit.CONSTRAINT|types.DisableBit.CONTACT):return
        d.ncollision.zero_();self.broadphase.enqueue();self.convex.enqueue();self.planes.enqueue();self.boxes.enqueue()


class Physics:
    """Stable numerical stages, each operating on one owned Model/Data pair."""
    def __init__(self,m,d,stream,products):
        self.model,self.data,self.stream=m,d,stream
        self.topology=top=motion.MotionTopology(m,d,stream)
        self.kinematics=kinematics.KinematicFrames(m,d,products('kinematics'),stream,topology=top)
        self.kinematics.prepare_fusion()
        self.com_pos=com_pos.CenteredFrames(m,d,products('com_pos'),stream)
        self.com_pos.prepare_fusion(top)
        self.camlight=camlight.CamLightFrames(m,d,products('camlight'),stream,admission=camlight.qualify(m,d,implementation='cuda'))
        self.kinematics.prepare_frames(self.camlight)
        self.matrix=mass_matrix.MassMatrix(m,d,products('mass_matrix'),stream)
        self.matrix.prepare_fusion(top)
        self.centered_mass=FusedCenteredMass(self.com_pos,self.matrix,products.folder/'fp32_fused_position.so')
        self.factor_plan=mass_factor.MassFactorPlan(m,d,products('mass_factor'),stream)
        self.factor=self.factor_plan.bind(d.M,d.qLD,d.qLDiagInv)
        self.public_solve=self.factor_plan.bind(d.M,d.qLD,d.qLDiagInv,d.qacc_smooth,d.qfrc_smooth)
        self.smooth_solve=mass_solve.MassSolvePlan(m,d,products('mass_solve'),stream).bind(d.qacc_smooth,d.qfrc_smooth)
        self.collision=Collision(m,d,stream,products)
        self.constraints=constraints.ConstraintConstruction(m,d,products('constraint_build'),stream,admission=constraints.qualify(m,d,implementation='cuda'))
        self.transmission=transmission.TransmissionNumerics(m,d,products('transmission'),stream,admission=transmission.qualify(m,d,transmission='cuda',actuator_velocity='cuda'))
        self.velocity=velocity.ComVelocity(m,d,products('velocity'),stream,top)
        self.passive=passive.PassiveForces(m,d,products('passive'),stream)
        fa=fluid.qualify(m,d,implementation='cuda')
        if m.has_fluid and fa['selected']!='cuda':raise ValueError(fa['fallback'])
        self.fluid=fluid.FluidNumerics(m,d,products('fluid'),stream,admission=fa) if m.has_fluid else None
        if self.fluid is not None:self.fluid.bind_passive(self.passive)
        self.gravity=queries.GravityProjection(m,d,products('queries'),stream)
        self.bias=bias.Bias(m,d,products('bias'),stream,top)
        self.bias.prepare_fusion()
        self.motion_bias=FusedMotion(self.velocity,self.bias,products.folder/'fp32_fused_motion.so')
        self.actuation=actuation.ActuatorForces(m,d,products('actuation'),stream)
        self.smooth=smooth_force.SmoothForce(m,d,products('smooth_force'),stream)
        self.actuation.bind_smooth(self.smooth)
        self.newton=Newton(m,d,stream,products)
        self.external=external.ExternalWrenchNumerics(m,d,products('external_wrench'),stream,admission=external.qualify(m,d,implementation='cuda'))
        self.postconstraint=postconstraint.PostconstraintRNE(m,d,products('postconstraint'),stream,top)
        self.sensors=queries.ForceSensors(m,d,products('queries'),stream)
        self.damped=not bool(m.opt.disableflags&(types.DisableBit.EULERDAMP|types.DisableBit.DAMPER))
        self.euler_workspace=None;self.euler_solve=None
        if self.damped:
            w=SimpleNamespace(matrix=cuda.empty_like(d.M),factor=cuda.empty_like(d.qLD),diagonal=cuda.empty_like(d.qLDiagInv),acceleration=cuda.empty_like(d.qacc),damping=cuda.empty_like(d.qvel))
            self.euler_workspace=w;self.euler_solve=self.factor_plan.bind(w.matrix,w.factor,w.diagonal,w.acceleration,d.efc.Ma)
        self.euler=euler.EulerNumerics(m,d,products('euler'),stream,workspace=self.euler_workspace)
        self.publication=publication.PublicationState(m,d,products('publication'),stream)
        self.advance=FusedAdvance(self.euler,products.folder/'fp32_fused_tail.so',publication=self.publication,
                                  qacc=d.qacc,warmstart=d.qacc_warmstart)

    def position(self,factorize=True):
        self.kinematics.enqueue_frames()
        self.centered_mass.enqueue()
        if factorize:self.factor.enqueue()
        if self.model.opt.run_collision_detection:self.collision.enqueue()
        self.constraints.enqueue();self.transmission.enqueue(0)

    def velocity_stage(self):
        m,d=self.model,self.data
        self.transmission.enqueue(1);self.motion_bias.enqueue()
        if self.passive.flags['disable_spring'] and self.passive.flags['disable_damper']:
            for name in ('qfrc_spring','qfrc_damper','qfrc_gravcomp','qfrc_fluid','qfrc_passive'):getattr(d,name).zero_()
        else:
            self.passive.enqueue(0);d.qfrc_gravcomp.zero_()
            if not m.opt.disableflags&types.DisableBit.GRAVITY:self.gravity.enqueue()
            if self.fluid is not None:self.fluid.enqueue_passive()
            else:
                d.qfrc_adhesion.zero_();self.passive.enqueue(1)

    def step1(self,factorize=True):
        self.position(factorize);self.data.sensordata.zero_();self.data.energy.zero_();self.velocity_stage()

    def solve(self,public=False):
        m,d=self.model,self.data
        if m.opt.disableflags&types.DisableBit.ACTUATION:
            d.act_dot.zero_();d.qfrc_actuator.zero_();d.actuator_force.zero_()
            self.smooth.enqueue()
        else:
            self.actuation.enqueue_smooth()
        (self.public_solve if public else self.smooth_solve).enqueue()
        self.newton.solve()
        if m.opt.run_rne_postconstraint or (not m.opt.disableflags&types.DisableBit.SENSOR and m.sensor_rne_postconstraint):
            self.external.enqueue();self.postconstraint.enqueue()
        self.sensors.enqueue()

    def integrate(self):
        if self.damped:
            self.euler.enqueue(0);cuda.copy(self.euler_workspace.matrix,self.data.M)
            self.euler.enqueue(1);self.euler_solve.enqueue()
        self.advance.enqueue()

    def step2(self):self.solve();self.integrate()
    def forward(self):self.step1(factorize=False);self.solve(public=True)
