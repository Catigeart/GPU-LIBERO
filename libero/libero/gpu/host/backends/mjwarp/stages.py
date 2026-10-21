"""Explicit resident stage boundaries for gradual handwritten FP32 replacement.

Every stage borrows the very same compiled MJWarp Model and Data. There is no
second XML compilation, intermediate download, or independently owned physics
state between stages. The default implementation delegates all mathematics to
the pinned upstream source; this file only exposes its step1/step2 schedule.

Schedule source: mujoco_warp/_src/forward.py, pinned by
 data/mjwarp_source_pin.json. Copyright 2025 The Newton Developers, Apache 2.0.
"""
from mujoco_warp._src import forward, sensor, smooth
from mujoco_warp._src import solver
from mujoco_warp._src.types import DisableBit, IntegratorType


class PhysicsStages:
    """A physical substep's enqueue boundaries; all work stays on the GPU.

    New implementations should replace one of these methods on a separate
    stage type and consume self.model/self.data. Keep the default upstream
    type as the reference. Graphs must be recaptured after changing a stage.
    """

    def __init__(self, model, data, *, newton=None, acceleration=None, mass_factor=None, mass_matrix=None, velocity=None, postconstraint=None, actuation=None, euler=None, transmission=None, kinematics=None, com_pos=None, camlight=None, constraint_build=None, collision=None, queries=None):
        self.model, self.data = model, data
        self.queries = queries
        self.newton_driver = newton
        self.acceleration_driver = acceleration
        self.mass_factor_driver = mass_factor
        self.velocity_driver = velocity
        self.postconstraint_driver = postconstraint
        self.actuation_driver = actuation
        self.euler_driver = euler
        from .position import Position
        self.position_driver = Position(model, data, mass_matrix=mass_matrix, mass_factor=mass_factor, transmission=transmission, kinematics=kinematics, com_pos=com_pos, camlight=camlight, constraint_build=constraint_build, collision=collision)

    def position(self,*,factorize=True):
        if factorize:self.position_driver.position()
        else:self.position_driver.position(factorize=False)

    def velocity(self):
        if self.velocity_driver is None:forward.fwd_velocity(self.model, self.data)
        else:self.velocity_driver.enqueue()

    def actuation(self):
        if self.actuation_driver is None:forward.fwd_actuation(self.model, self.data)
        else:self.actuation_driver.enqueue()

    def acceleration(self,*,factorize=False):
        if self.acceleration_driver is None:
            if factorize:forward.fwd_acceleration(self.model,self.data,factorize=True)
            else:forward.fwd_acceleration(self.model,self.data)
        else:
            if factorize:self.acceleration_driver.enqueue(factorize=True,factor=self.mass_factor_driver)
            else:self.acceleration_driver.enqueue()

    def newton(self):
        if self.newton_driver is None:
            solver.solve(self.model, self.data)
        else:
            self.newton_driver.solve()

    def integrate(self):
        if self.euler_driver is not None:
            self.euler_driver.integrate();return
        if self.mass_factor_driver is not None:
            self.mass_factor_driver.integrate()
            return
        m, d = self.model, self.data
        if m.opt.integrator in (IntegratorType.IMPLICITFAST, IntegratorType.IMPLICIT):
            forward.implicit(m, d)
        elif m.opt.integrator == IntegratorType.DISCRETE:
            forward.discrete(m, d)
        else:
            forward.euler(m, d)

    def step1(self):
        m, d = self.model, self.data
        forward.check_discrete(m)
        self.position()
        d.sensordata.zero_()
        if self.queries is None:
            sensor.sensor_pos(m,d);forward._energy_pos(m,d)
        else:
            self.queries.sensor_pos();self.queries.energy_pos()
        self.velocity()
        if self.queries is None:
            sensor.sensor_vel(m,d);forward._energy_vel(m,d)
        else:
            self.queries.sensor_vel();self.queries.energy_vel()
        if not (m.opt.disableflags & DisableBit.ACTUATION) and m.callback.control:
            m.callback.control(m, d)

    def step2(self):
        m, d = self.model, self.data
        self.actuation()
        self.acceleration()
        self.newton()
        if m.opt.integrator == IntegratorType.DISCRETE:
            forward._launch_discrete_free_gyro(m, d, d.qacc, d.qacc, False)
        if m.opt.run_rne_postconstraint or (not (m.opt.disableflags & DisableBit.SENSOR)
                                          and m.sensor_rne_postconstraint):
            if self.postconstraint_driver is None:smooth.rne_postconstraint(m, d)
            else:self.postconstraint_driver.enqueue()
        if self.queries is None:sensor.sensor_acc(m,d,skip_rne_postconstraint=True)
        else:self.queries.sensor_acc()
        self.integrate()

    def forward(self):
        """Pinned public forward order; no integration or warmstart copy."""
        m,d=self.model,self.data
        forward.check_discrete(m)
        self.position(factorize=False)
        d.sensordata.zero_()
        if self.queries is None:
            sensor.sensor_pos(m,d);forward._energy_pos(m,d)
        else:
            self.queries.sensor_pos();self.queries.energy_pos()
        self.velocity()
        if self.queries is None:
            sensor.sensor_vel(m,d);forward._energy_vel(m,d)
        else:
            self.queries.sensor_vel();self.queries.energy_vel()
        if not (m.opt.disableflags & DisableBit.ACTUATION) and m.callback.control:m.callback.control(m,d)
        self.actuation()
        self.acceleration(factorize=True)
        self.newton()
        if m.opt.run_rne_postconstraint or (not (m.opt.disableflags & DisableBit.SENSOR) and m.sensor_rne_postconstraint):
            if self.postconstraint_driver is None:smooth.rne_postconstraint(m,d)
            else:self.postconstraint_driver.enqueue()
        if self.queries is None:sensor.sensor_acc(m,d,skip_rne_postconstraint=True)
        else:self.queries.sensor_acc()
