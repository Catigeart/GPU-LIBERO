"""Concrete resident fixed-model physics chain, prepared outside capture.

The initial milestone integrates real D/C/S/N outputs. Acceleration sensors,
scene transforms and robot control are explicit extension hooks; their absence
is reported as incomplete full-forward scope, never substituted with CPU data.
"""
from types import SimpleNamespace
import numpy as np
import warp as wp
from .contact_transport import ContactTransport
from .projection import project_counts, project_cone_heads

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


@wp.kernel(enable_backward=False)
def _increment_plan(mask: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
                    expected: wp.array(dtype=wp.int64)):
    w = wp.tid()
    if mask[w] == 1 and first[w] == 0:
        expected[w] = expected[w] + wp.int64(1)


@wp.kernel(enable_backward=False)
def _c_mask(active: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
            written: wp.array(dtype=wp.int64), expected: wp.array(dtype=wp.int64),
            first: wp.array(dtype=wp.int32), result: wp.array(dtype=wp.int32)):
    w = wp.tid()
    result[w] = 0
    if active[w] == 1 and valid[w] == 1 and written[w] == expected[w] and first[w] == 0:
        result[w] = 1


@wp.kernel(enable_backward=False)
def _reference_mask(valid: wp.array(dtype=wp.int32), written: wp.array(dtype=wp.int64),
                    expected: wp.array(dtype=wp.int64), first: wp.array(dtype=wp.int32),
                    active: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if valid[w] != 1 or written[w] != expected[w] or first[w] != 0:
        active[w] = 0


@wp.kernel(enable_backward=False)
def _n_mask(active: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
            smooth: wp.array(dtype=wp.int32), reference: wp.array(dtype=wp.int32),
            sstatus: wp.array(dtype=wp.int32), admitted: wp.array(dtype=wp.int32),
            error: wp.array(dtype=wp.int32)):
    w = wp.tid()
    admitted[w] = 0
    error[w] = first[w]
    if error[w] == 0 and sstatus[w] != 0:
        error[w] = sstatus[w]
    if active[w] == 1 and error[w] == 0 and smooth[w] == 1 and reference[w] == 1:
        admitted[w] = 1


@wp.kernel(enable_backward=False)
def _publish_forward(active: wp.array(dtype=wp.int32), nvalid: wp.array(dtype=wp.int32),
                     nerror: wp.array(dtype=wp.int32), first: wp.array(dtype=wp.int32),
                     projection_error: wp.array(dtype=wp.int32),
                     source_state: wp.array(dtype=wp.int64), state: wp.array(dtype=wp.int64),
                     valid: wp.array(dtype=wp.int32), written: wp.array(dtype=wp.int64),
                     error: wp.array(dtype=wp.int32)):
    w = wp.tid()
    valid[w] = 0
    error[w] = first[w]
    if error[w] == 0 and nerror[w] != 0:
        error[w] = nerror[w]
    if error[w] == 0 and projection_error[w] != 0:
        error[w] = projection_error[w]
    if active[w] == 1 and error[w] == 0 and nvalid[w] == 1 and source_state[w] == state[w]:
        written[w] = source_state[w]
        valid[w] = 1


class FixedPhysicsForward:
    """Real fixed-model D position→C→S→D velocity/bias/force→N.

    ``prepare`` never evaluates CPU forward/contact/constraint truth. Model
    statics and reset state are the permitted cold inputs. C/S/N per-world
    capacities remain 15633 calls, 1900 contacts, 6000 rows.
    """
    def __init__(self, *, dynamics, collision, transport, constraints, newton,
                 foundation, expected_forward, ctrl, ctrl_valid, stream,
                 capture_entry_witness=False):
        self.d, self.c, self.transport = dynamics, collision, transport
        self.s, self.n, self.f = constraints, newton, foundation
        self.expected_forward, self.ctrl, self.ctrl_valid = expected_forward, ctrl, ctrl_valid
        self.stream, self.capacity, self.device = stream, dynamics.capacity, dynamics.device
        self.mask = foundation._mask
        self.c_mask = collision.source_owner.active
        self.n_admitted = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.n_upstream = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.projection_status = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.projection_item = wp.full(self.capacity, -1, dtype=wp.int32, device=self.device)
        self.valid = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        self.written = wp.zeros(self.capacity, dtype=wp.int64, device=self.device)
        self.error = wp.zeros(self.capacity, dtype=wp.int32, device=self.device)
        w = newton.ws
        self._counts = constraints.workspace.counts.flatten()
        self._s_views = {
            'type': constraints.workspace.efc_type.reshape((self.capacity,w.rows)),
            'id': constraints.workspace.efc_id.reshape((self.capacity,w.rows)),
            'dim': transport.inputs['dim'].reshape((self.capacity,w.contacts)),
            'mu': constraints.workspace.contact_mu.reshape((self.capacity,w.contacts)),
            'friction': transport.inputs['friction'].reshape((self.capacity,w.contacts,5)),
        }
        self.acceleration_sensor = None
        self.before_constraint = None
        self.scene_transform = None
        self.position_tape = None
        self._position_tape = None
        self.scope_gaps = ['acceleration_sensor_hook', 'scene_transform_hook', 'robot_controller']
        self.final_valid = newton.valid
        self.entry_sources = {name:getattr(w,name) for name in ('J','aref','D','R','loss','efc_type','efc_id',
            'qM','qfrc_smooth','qacc_smooth','nefc','ne','nf','ncon','contact_dim','contact_mu','contact_friction')}
        self.entry_sources['qacc_warmstart'] = newton.public_arrays['qacc_warmstart']
        # Solver inputs remain live bindings. Full-capacity copies are only
        # needed by an explicitly selected diagnostic recorder.
        self.entry_witness = ({name:wp.empty_like(a) for name,a in self.entry_sources.items()}
                              if capture_entry_witness else {})

    def _launch(self, kernel, inputs):
        wp.launch(kernel, dim=self.capacity, inputs=inputs, device=self.device, stream=self.stream)

    def validate(self):
        self.d.validate(self.stream)
        self.c.workspace.validate_device_bindings(self.stream)
        self.transport.validate()
        self.s.validate()
        self.f.validate_bindings()
        if self.position_tape is not self._position_tape:
            raise ValueError('prepared position input tape replaced')

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
            self.projection_status,self.projection_item,n.rows,n.contacts])
        self._launch(project_cone_heads,[self.n_admitted,n.nefc,n.ne,n.nf,n.ncon,
            x['type'],x['id'],x['dim'],x['mu'],x['friction'],n.cone_dim,n.cone_mu,
            n.cone_friction,self.projection_status,self.projection_item,n.rows,n.contacts])
        if self.before_constraint is not None:
            self.before_constraint(self.stream)
        for name,target in self.entry_witness.items():
            wp.copy(target,self.entry_sources[name],stream=self.stream)
        self.n.enqueue(self.n_upstream,active=self.mask,expected_epoch=self.expected_forward,publication_status=self.projection_status)
        if self.acceleration_sensor is not None:
            self.acceleration_sensor(self.n_admitted,self.stream)
        self._launch(_publish_forward,[self.d.active,self.final_valid,n.err_code,
            self.d.first_error,self.projection_status,self.d.source_state_epoch,self.d.state_epoch,
            self.valid,self.written,self.error])

    def source_manifest(self):
        return {'model':self.d.stages.model_pack.mjb_sha256,
                'physics_sources':'live accepted D/C/S/N kernels; no CPU dynamic intermediates',
                'scope_gaps':list(self.scope_gaps),
                'clock':'independent planned forward counter and actual D completion; integration owns state epoch'}
