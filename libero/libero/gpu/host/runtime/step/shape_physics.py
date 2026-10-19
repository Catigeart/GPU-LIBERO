"""Diagnostic raw-model forward with independent N request and partial step1."""
import warp as wp
from .physics import FixedPhysicsForward, _increment_plan, _c_mask, _reference_mask, _n_mask, _publish_forward
from .shape_projection import project_counts, project_cone_heads
from .shape_schema import RuntimeShape

@wp.kernel(enable_backward=False)
def _invalidate_full(valid: wp.array(dtype=wp.int32)):
    valid[wp.tid()] = 0

class ShapePhysicsForward(FixedPhysicsForward):
    def __init__(self, *, runtime_shape: RuntimeShape, **kwargs):
        if not isinstance(runtime_shape, RuntimeShape):
            raise TypeError("actual native RuntimeShape required")
        self.runtime_shape = runtime_shape
        self._shape_ref = runtime_shape
        super().__init__(**kwargs)
        if (self.n.ws.qacc.shape != (self.capacity,runtime_shape.nv) or
            self.n.ws.J.shape != (self.capacity,self.n.ws.rows,runtime_shape.nv)):
            raise ValueError("actual N dimension differs from native shape")
        self.expected_newton = wp.zeros(self.capacity, dtype=wp.int64, device=self.device)
        self._newton_plan = self.expected_newton
        self._newton_plan_token = (int(self.expected_newton.ptr), self.expected_newton.shape,
                                  self.expected_newton.strides, self.expected_newton.dtype)

    def validate(self):
        super().validate()
        if self.runtime_shape is not self._shape_ref or (self.n.ws.qacc.shape != (self.capacity,self.runtime_shape.nv) or
                self.n.ws.J.shape != (self.capacity,self.n.ws.rows,self.runtime_shape.nv)):
            raise ValueError("actual native shape/N dimension changed")
        a = self.expected_newton
        if a is not self._newton_plan or (int(a.ptr),a.shape,a.strides,a.dtype) != self._newton_plan_token:
            raise ValueError("independent N request plan owner changed")
        if a.ptr == self.expected_forward.ptr:
            raise ValueError("forward and N invocation plans overlap")

    def enqueue(self, mask, stream):
        if stream is not self.stream:
            raise ValueError('foreign forward stream')
        self.validate()
        # All callbacks use one retained F mask. Conditional refresh copies a
        # preallocated device mask here, never uploads or constructs a view.
        if mask is not self.mask:
            wp.copy(self.mask, mask, stream=self.stream)
        admitted = self.f.begin_forward(self.mask, stream=self.stream)
        self._launch(_increment_plan, [admitted,self.d.first_error,self.expected_forward])
        self.f.launch_position(stream=self.stream)
        self._launch(_c_mask, [self.d.active,self.d.valid['position'],
            self.d.written['position'],self.expected_forward,self.d.first_error,self.c_mask])
        self.c.enqueue(self.c_mask, self.stream)
        self.transport.enqueue()
        if self.position_tape is not None:
            self.position_tape.capture()
        self.s.launch_position()
        if self.scene_transform is not None:
            self.scene_transform(self.c_mask,self.stream)
        self.f.launch_velocity(stream=self.stream)
        # Transport owns this private S admission mask; producer masks stay readonly.
        self._launch(_reference_mask,[self.d.valid['velocity'],self.d.written['velocity'],
            self.expected_forward,self.d.first_error,self.transport.active])
        self.s.launch_reference()
        sw = self.s.workspace
        self.d.launch_bias(reference_done=sw.reference_ready, reference_error=sw.status,
            reference_written=sw.ref_epoch, reference_expected=self.expected_forward,stream=self.stream)
        self.d.launch_force(ctrl=self.ctrl,ctrl_valid=self.ctrl_valid,stream=self.stream)
        self._launch(_n_mask,[self.d.active,self.d.first_error,self.d.valid['smooth'],
            sw.reference_ready,sw.status,self.n_admitted,self.n_upstream])
        n, x = self.n.ws, self._s_views
        self._launch(project_counts,[self.n_admitted,self._counts,self.transport.ncon,
            n.nefc,n.ne,n.nf,n.ncon,n.count,n.empty,n.newton_world,n.active,
            self.projection_status,self.projection_item,n.rows,n.contacts,self.runtime_shape.nv])
        self._launch(project_cone_heads,[self.n_admitted,n.nefc,n.ne,n.nf,n.ncon,
            x['type'],x['id'],x['dim'],x['mu'],x['friction'],n.cone_dim,n.cone_mu,
            n.cone_friction,self.projection_status,self.projection_item,n.rows,n.contacts])
        self._launch(_increment_plan,[self.mask,self.d.first_error,self.expected_newton])
        if self.before_constraint is not None:
            self.before_constraint(self.stream)
        for name,target in self.entry_witness.items():
            wp.copy(target,self.entry_sources[name],stream=self.stream)
        self.n.enqueue(self.n_upstream,active=self.mask,expected_epoch=self.expected_newton,publication_status=self.projection_status)
        if self.acceleration_sensor is not None:
            self.acceleration_sensor(self.n_admitted,self.stream)
        self._launch(_publish_forward,[self.d.active,self.final_valid,n.err_code,
            self.d.first_error,self.projection_status,self.d.source_state_epoch,self.d.state_epoch,
            self.valid,self.written,self.error])

    def enqueue_step1(self, mask, stream):
        if stream is not self.stream:
            raise ValueError('foreign forward stream')
        self.validate()
        # A partial forward never authorizes integration from an older solve.
        self._launch(_invalidate_full, [self.valid])
        # All callbacks use one retained F mask. Conditional refresh copies a
        # preallocated device mask here, never uploads or constructs a view.
        if mask is not self.mask:
            wp.copy(self.mask, mask, stream=self.stream)
        admitted = self.f.begin_forward(self.mask, stream=self.stream)
        self._launch(_increment_plan, [admitted,self.d.first_error,self.expected_forward])
        self.f.launch_position(stream=self.stream)
        self._launch(_c_mask, [self.d.active,self.d.valid['position'],
            self.d.written['position'],self.expected_forward,self.d.first_error,self.c_mask])
        self.c.enqueue(self.c_mask, self.stream)
        self.transport.enqueue()
        if self.position_tape is not None:
            self.position_tape.capture()
        self.s.launch_position()
        if self.scene_transform is not None:
            self.scene_transform(self.c_mask,self.stream)
        self.f.launch_velocity(stream=self.stream)
        # Transport owns this private S admission mask; producer masks stay readonly.
        self._launch(_reference_mask,[self.d.valid['velocity'],self.d.written['velocity'],
            self.expected_forward,self.d.first_error,self.transport.active])
        self.s.launch_reference()
        sw = self.s.workspace
        self.d.launch_bias(reference_done=sw.reference_ready, reference_error=sw.status,
            reference_written=sw.ref_epoch, reference_expected=self.expected_forward,stream=self.stream)

