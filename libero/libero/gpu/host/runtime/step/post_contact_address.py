"""Current contact addresses for the integrated Post consumer.

MuJoCo 2.3.7 mj_collideGeoms initializes each new contact's efc_address to -1;
mj_instantiateContact assigns an address only to included contacts. The resident
standalone S API instead holds excluded public address slots. Keep that API
unchanged and restore the current-contact invariant in this private bridge.
"""
import warp as wp


@wp.kernel(enable_backward=False)
def _current_post_contact_address(
    active: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    committed: wp.array(dtype=wp.int32),
    ncon: wp.array(dtype=wp.int32),
    exclude: wp.array(dtype=wp.int32, ndim=2),
    address: wp.array(dtype=wp.int32, ndim=2),
    current: wp.array(dtype=wp.int32, ndim=2),
    contacts: int,
):
    w, c = wp.tid()
    if active[w] != 1 or status[w] != 0 or committed[w] != 1:
        return
    count = ncon[w]
    if count < 0 or count > contacts or c >= count:
        return
    # Use S's final exclusion, including the no-DOF case (exclude=3).
    if exclude[w, c] != 0:
        current[w, c] = -1
    else:
        current[w, c] = address[w, c]


def _token(array):
    return (id(array), int(array.ptr), array.dtype, tuple(array.shape),
            tuple(array.strides), array.device)


def capture_post_with_current_address(bridge, post, stream):
    bridge.enqueue(stream=stream)
    # Assembly may replace the snapshot copy plan before graph capture.
    post.capture_before_constraint(stream=stream)


class PostContactAddressBridge:
    """One retained owner and stream; update only successful live S prefixes."""

    def __init__(self, workspace, *, active, stream):
        self.workspace = workspace
        self.stream = stream
        self.worlds, self.contacts = int(workspace.nworld), int(workspace.contacts)
        self.device = active.device
        self.active = active
        self.sources = {name: getattr(workspace, name) for name in
                        ('status', 'committed', 'ncon', 'contact_exclude', 'contact_address')}
        if stream.device != self.device:
            raise ValueError('contact address bridge requires the retained device stream')
        shapes = {'active': (self.worlds,), 'status': (self.worlds,),
                  'committed': (self.worlds,), 'ncon': (self.worlds,),
                  'contact_exclude': (self.worlds, self.contacts),
                  'contact_address': (self.worlds, self.contacts)}
        arrays = dict(self.sources, active=active)
        for name, array in arrays.items():
            if (array.dtype != wp.int32 or tuple(array.shape) != shapes[name]
                    or array.device != self.device or not array.is_contiguous):
                raise ValueError('contact address bridge source schema: ' + name)
        self.current = wp.full((self.worlds, self.contacts), -1, dtype=wp.int32,
                               device=self.device)
        self._workspace = workspace
        self._stream = stream
        self._dimensions = (self.worlds, self.contacts)
        self._active = active
        self._current = self.current
        self._tokens = {name: _token(array) for name, array in arrays.items()}
        self._current_token = _token(self.current)

    def validate(self, stream=None):
        if stream is not None and stream is not self.stream:
            raise ValueError('foreign contact address bridge stream')
        if (self.workspace is not self._workspace or self.stream is not self._stream
                or (self.worlds, self.contacts) != self._dimensions
                or self.active is not self._active
                or self.current is not self._current
                or _token(self.current) != self._current_token
                or (self.workspace.nworld, self.workspace.contacts) != (self.worlds, self.contacts)):
            raise ValueError('contact address bridge retained owner changed')
        arrays = dict(self.sources, active=self.active)
        for name, array in arrays.items():
            if (_token(array) != self._tokens[name]
                    or (name != 'active' and getattr(self.workspace, name) is not array)):
                raise ValueError('contact address bridge source owner changed: ' + name)

    def enqueue(self, stream=None):
        self.validate(stream)
        s = self.sources
        wp.launch(_current_post_contact_address, dim=(self.worlds, self.contacts),
                  inputs=[self.active, s['status'], s['committed'], s['ncon'],
                          s['contact_exclude'], s['contact_address'], self.current,
                          self.contacts], device=self.device, stream=self.stream)
