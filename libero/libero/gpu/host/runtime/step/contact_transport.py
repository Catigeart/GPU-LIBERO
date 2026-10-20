"""Typed current collision slab to resident S input leaves, all on device."""
import libero.gpu.precision as scalar
import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


@wp.kernel(enable_backward=False)
def _extract(active: wp.array(dtype=wp.int32),
             position_valid: wp.array(dtype=wp.int32),
             first_error: wp.array(dtype=wp.int32),
             c_error: wp.array(dtype=wp.int32),
             c_active: wp.array(dtype=wp.int32), ncon: wp.array(dtype=wp.int32),
             cf: wp.array(dtype=scalar.wp_real), ci: wp.array(dtype=wp.int32),
             target_active: wp.array(dtype=wp.int32),
             target_ncon: wp.array(dtype=wp.int32),
             dist: wp.array(dtype=scalar.wp_real), pos: wp.array(dtype=scalar.wp_real),
             frame: wp.array(dtype=scalar.wp_real), margin: wp.array(dtype=scalar.wp_real),
             friction: wp.array(dtype=scalar.wp_real), solref: wp.array(dtype=scalar.wp_real),
             solreffriction: wp.array(dtype=scalar.wp_real), solimp: wp.array(dtype=scalar.wp_real),
             dim: wp.array(dtype=wp.int32), geom1: wp.array(dtype=wp.int32),
             geom2: wp.array(dtype=wp.int32), exclude: wp.array(dtype=wp.int32),
             source_capacity: int, target_capacity: int):
    w = wp.tid()
    target_active[w] = 0
    # Counts and other source leaves remain held if the source is unavailable.
    if active[w] != 1 or position_valid[w] != 1 or first_error[w] != 0:
        return
    if c_active[w] != 1 or c_error[w] != 0:
        return
    nc = ncon[w]
    if nc < 0 or nc > source_capacity or nc > target_capacity:
        return
    for k in range(nc):
        a = (w * source_capacity + k) * 65
        b = (w * source_capacity + k) * 5
        t = w * target_capacity + k
        dist[t] = cf[a]
        margin[t] = cf[a + 13]
        for j in range(3):
            pos[t * 3 + j] = cf[a + 1 + j]
        for j in range(9):
            frame[t * 9 + j] = cf[a + 4 + j]
        for j in range(5):
            friction[t * 5 + j] = cf[a + 14 + j]
            solimp[t * 5 + j] = cf[a + 23 + j]
        for j in range(2):
            solref[t * 2 + j] = cf[a + 19 + j]
            solreffriction[t * 2 + j] = cf[a + 21 + j]
        dim[t] = ci[b]
        geom1[t] = ci[b + 1]
        geom2[t] = ci[b + 2]
        exclude[t] = ci[b + 3]
    target_ncon[w] = nc
    target_active[w] = 1


class ContactTransport:
    """Prepare exact S13 inputs, retaining C AoS owners and world strides."""
    WIDTHS = {'dist': 1, 'pos': 3, 'frame': 9, 'margin': 1, 'friction': 5,
              'solref': 2, 'solreffriction': 2, 'solimp': 5,
              'dim': 1, 'geom1': 1, 'geom2': 1, 'exclude': 1}

    def __init__(self, collision, dynamics, *, nworld, capacity, stream):
        self.collision, self.dynamics, self.stream = collision, dynamics, stream
        self.nworld, self.capacity = int(nworld), int(capacity)
        c = collision.workspace
        self.source_capacity = int(c.contact_capacity)
        self.device = dynamics.device
        if int(c.nworld) != self.nworld or dynamics.capacity != self.nworld:
            raise ValueError('physical world cardinality mismatch')
        if stream is not collision.stream or stream is not dynamics.stream:
            raise ValueError('one retained producer/consumer stream required')
        self.sources = {'f': c.finalize.contact_f, 'i': c.finalize.contact_i,
                        'ncon': c.finalize.ncon, 'active': c.effective_active,
                        'error': c.fe_code}
        self._flat_f = self.sources['f'].flatten()
        self._flat_i = self.sources['i'].flatten()
        if self._flat_f.size != self.nworld*self.source_capacity*65 or self._flat_i.size != self.nworld*self.source_capacity*5:
            raise ValueError('C public contact layout does not match 65/5 AoS')
        self.inputs = {name: wp.zeros(self.nworld*self.capacity*width,
            dtype=wp.int32 if name in ('dim','geom1','geom2','exclude') else scalar.wp_real,
            device=self.device) for name,width in self.WIDTHS.items()}
        self.active = wp.zeros(self.nworld, dtype=wp.int32, device=self.device)
        self.ncon = wp.zeros(self.nworld, dtype=wp.int32, device=self.device)
        self._owners = {**self.sources, **self.inputs, 'target_active': self.active,
                        'target_ncon': self.ncon, 'flat_f': self._flat_f, 'flat_i': self._flat_i}
        self._tokens = {name:self._token(a) for name,a in self._owners.items()}

    @staticmethod
    def _token(a):
        return (id(a), int(a.ptr), a.dtype, tuple(a.shape), tuple(a.strides), str(a.device))

    def validate(self):
        self.dynamics.validate(self.stream)
        self.collision.workspace.validate_device_bindings(self.stream)
        c = self.collision.workspace
        current = {'f': c.finalize.contact_f, 'i': c.finalize.contact_i,
                   'ncon': c.finalize.ncon, 'active': c.effective_active, 'error': c.fe_code}
        if set(self.inputs) != set(self.WIDTHS):
            raise ValueError('transport target field set changed')
        for name,a in self.inputs.items():
            if a is not self._owners[name]:
                raise ValueError('transport current target replaced: '+name)
        for name,a in (('target_active',self.active),('target_ncon',self.ncon),
                       ('flat_f',self._flat_f),('flat_i',self._flat_i)):
            if a is not self._owners[name]:
                raise ValueError('transport current owner replaced: '+name)
        for name,a in current.items():
            if a is not self.sources[name] or self._token(a) != self._tokens[name]:
                raise ValueError('current C source changed: '+name)
        for name,a in self._owners.items():
            if self._token(a) != self._tokens[name]:
                raise ValueError('transport allocation changed: '+name)

    def enqueue(self):
        self.validate()
        x, d = self.inputs, self.dynamics
        wp.launch(_extract, dim=self.nworld, inputs=[d.active, d.valid['position'],
            d.first_error, self.sources['error'], self.sources['active'],
            self.sources['ncon'], self._flat_f, self._flat_i, self.active, self.ncon,
            *[x[name] for name in ('dist','pos','frame','margin','friction','solref',
                                 'solreffriction','solimp','dim','geom1','geom2','exclude')],
            self.source_capacity,self.capacity], device=self.device,stream=self.stream)
