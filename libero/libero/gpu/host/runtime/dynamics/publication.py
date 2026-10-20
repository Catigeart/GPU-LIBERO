"""Device-only publication for persistent producer results.

Primitive work arrays retain their original inactive/failure zero semantics.
Only a successful, actually gated production may replace a public cache.  The
data copy and publication stamp run in one per-world kernel on the producer's
retained stream.  No failed result is copied then restored.
"""
from __future__ import annotations
import libero.gpu.precision as scalar

import warp as wp

wp.set_module_options({'enable_backward': False, 'fuse_fp': True, 'fast_math': False})


@wp.kernel(enable_backward=False)
def commit_bias(active: wp.array(dtype=wp.int32), gate: wp.array(dtype=wp.int32),
                status: wp.array(dtype=wp.int32), epoch: wp.array(dtype=wp.int64),
                error: wp.array(dtype=wp.int32),
                work: wp.array(dtype=scalar.wp_real, ndim=2),
                public: wp.array(dtype=scalar.wp_real, ndim=2),
                valid: wp.array(dtype=wp.int32), written: wp.array(dtype=wp.int64),
                width: int):
    w = wp.tid()
    valid[w] = 0
    if active[w] != 1 or gate[w] != 1 or error[w] != 0:
        return
    if status[w] != 0:
        error[w] = 211
        return
    for i in range(width):
        public[w, i] = work[w, i]
    written[w] = epoch[w]
    valid[w] = 1


@wp.kernel(enable_backward=False)
def commit_smooth(active: wp.array(dtype=wp.int32), gate: wp.array(dtype=wp.int32),
                  status: wp.array(dtype=wp.int32), epoch: wp.array(dtype=wp.int64),
                  error: wp.array(dtype=wp.int32),
                  work_force: wp.array(dtype=scalar.wp_real, ndim=2),
                  work_acceleration: wp.array(dtype=scalar.wp_real, ndim=2),
                  public_force: wp.array(dtype=scalar.wp_real, ndim=2),
                  public_acceleration: wp.array(dtype=scalar.wp_real, ndim=2),
                  valid: wp.array(dtype=wp.int32), written: wp.array(dtype=wp.int64),
                  repaired: wp.array(dtype=wp.int32), width: int):
    w = wp.tid()
    valid[w] = 0
    repaired[w] = 0
    if active[w] != 1 or gate[w] != 1 or error[w] != 0:
        return
    # Original composite success/finished pivot-repair semantics, unchanged.
    if status[w] != 0 and status[w] != 25:
        error[w] = 213
        return
    for i in range(width):
        public_force[w, i] = work_force[w, i]
        public_acceleration[w, i] = work_acceleration[w, i]
    written[w] = epoch[w]
    if status[w] == 25:
        repaired[w] = 1
    valid[w] = 1


class ProducerPublication:
    """Prepare-once public owners and declared bias-consumer wiring.

    This prepares a new producer contract; it does not weaken the original
    stage binding checker.  The original complete catalog remains recorded,
    exactly four declared bias-input references change during construction,
    and the entire resulting object graph is frozen once before any launch.
    Public results and primitive work buffers never alias.
    """

    NAMES = ('qfrc_bias', 'qfrc_smooth', 'qacc_smooth')
    BIAS_PATHS = (
        'dynamics.acceleration.bias',
        'dynamics.d07.osc.qfrc_bias',
        'dynamics.consumer_inputs.acceleration.bias',
        'dynamics.consumer_inputs.control.qfrc_bias',
    )
    TOKEN_FIELDS = ('array_id', 'ptr', 'shape', 'strides', 'dtype', 'device',
                    'contiguous', 'view_chain', 'owner_id', 'owner_ptr',
                    'owner_capacity_bytes', 'offset_bytes', 'access_extent_bytes',
                    'allocator_id', 'deleter_id')

    def __init__(self, producer):
        s = producer.stages
        s.validate_bindings()
        self._producer = producer
        self._stages = s
        self._capacity = int(producer.physical_capacity)
        self._width = 43
        self._device = producer.device
        self._stream = producer.stream
        self._work = {'qfrc_bias': s.qfrc_bias,
                      'qfrc_smooth': s.acceleration.qfrc,
                      'qacc_smooth': s.acceleration.qacc_smooth}
        for array in self._work.values():
            self._check_result_layout(array)
        with wp.ScopedStream(self._stream):
            for name in self.NAMES:
                setattr(self, name, wp.zeros((self._capacity, self._width),
                                            dtype=scalar.wp_real, device=self._device))
        self.original_stage_catalog = s.binding_catalog()
        self._prepare_consumers()
        self.prepared_stage_catalog = s.binding_catalog()
        self._frozen_results = {name: self._token(getattr(self, name))
                               for name in self.NAMES}
        self._frozen_work = {name: self._token(a) for name, a in self._work.items()}
        self._frozen_producer_arrays = {name: self._token(getattr(producer, name))
            for name in ('input_active', 'first_error', 'bias_valid',
                         'bias_written_epoch', 'smooth_valid', 'smooth_written_epoch',
                         'smooth_repaired', 'qvel_source_epoch', 'force_source_epoch')}
        self._frozen_stream = (id(self._stream), int(self._stream.cuda_stream),
                               int(self._device.context))
        self._prepared = True
        # The producer attaches this owner then validates, before source uploads.

    def _check_result_layout(self, array):
        if (not isinstance(array, wp.array) or array.dtype != scalar.wp_real or
                array.shape != (self._capacity, self._width) or
                array.device != self._device or not array.is_contiguous or
                tuple(array.strides) != (self._width * 8, 8)):
            raise ValueError('fixed43 publication requires contiguous actual f64 owner')
        self._token(array)  # Original extent/retained-owner bounds validation.

    def _token(self, array):
        return self._stages._array_binding(array)

    def _prepare_consumers(self):
        s = self._stages
        old_bias = self._work['qfrc_bias']
        if (s.acceleration.bias is not old_bias or s.d07.osc.qfrc_bias is not old_bias or
                s.consumer_inputs['acceleration']['bias'] is not old_bias or
                s.consumer_inputs['control']['qfrc_bias'] is not old_bias):
            raise ValueError('original bias-consumer wiring differs from declared preparation')
        old_arrays, old_owners, _, old_scalars = s._binding_state()
        s.acceleration.bias = self.qfrc_bias
        s.d07.osc.qfrc_bias = self.qfrc_bias
        s.consumer_inputs['acceleration']['bias'] = self.qfrc_bias
        s.consumer_inputs['control']['qfrc_bias'] = self.qfrc_bias
        new_arrays, new_owners, _, new_scalars = s._binding_state()
        changed = {path for path in set(old_arrays) | set(new_arrays)
                   if old_arrays.get(path) != new_arrays.get(path)}
        if changed != set(self.BIAS_PATHS) or old_owners != new_owners or old_scalars != new_scalars:
            raise ValueError('undeclared producer prepare binding delta')
        self.prepare_binding_delta = {path: {'before': dict(zip(self.TOKEN_FIELDS, old_arrays[path])),
                                             'after': dict(zip(self.TOKEN_FIELDS, new_arrays[path]))}
                                      for path in sorted(changed)}
        # Explicit new prepared graph, exactly once, before any scientific
        # source upload or launch.  Never refresh expectations at launch time.
        s._freeze_bindings()

    def validate(self):
        p, s = self._producer, self._stages
        s.validate_bindings()
        if p.stages is not s or p.publication is not self:
            raise ValueError('publication owner replaced')
        if self._stream is not p.stream or self._device != p.device:
            raise ValueError('publication retained stream/device replaced')
        if (id(p.stream), int(p.stream.cuda_stream), int(p.device.context)) != self._frozen_stream:
            raise ValueError('publication stream/context replaced')
        if int(p.physical_capacity) != self._capacity or self._width != 43:
            raise ValueError('publication fixed model/capacity changed')
        actual_work = {'qfrc_bias': s.qfrc_bias,
                       'qfrc_smooth': s.acceleration.qfrc,
                       'qacc_smooth': s.acceleration.qacc_smooth}
        for name in self.NAMES:
            public = getattr(self, name)
            self._check_result_layout(public)
            if self._token(public) != self._frozen_results[name]:
                raise ValueError('public result owner/layout replaced: ' + name)
            if self._token(actual_work[name]) != self._frozen_work[name]:
                raise ValueError('original primitive work binding replaced: ' + name)
            if self._token(self._work[name]) != self._frozen_work[name]:
                raise ValueError('publication work source replaced: ' + name)
        arrays = list(actual_work.values()) + [getattr(self, name) for name in self.NAMES]
        for i, a in enumerate(arrays):
            for b in arrays[i + 1:]:
                if max(int(a.ptr), int(b.ptr)) < min(int(a.ptr) + int(a.size) * 8,
                                                    int(b.ptr) + int(b.size) * 8):
                    raise ValueError('public/work result storage overlaps')
        for name, expected in self._frozen_producer_arrays.items():
            if self._token(getattr(p, name)) != expected:
                raise ValueError('producer publication clock/gate binding replaced: ' + name)
        if (s.acceleration.bias is not self.qfrc_bias or s.d07.osc.qfrc_bias is not self.qfrc_bias or
                s.consumer_inputs['acceleration']['bias'] is not self.qfrc_bias or
                s.consumer_inputs['control']['qfrc_bias'] is not self.qfrc_bias):
            raise ValueError('actual consumer no longer reads published bias')
        return True

    def launch_bias_commit(self):
        self.validate()
        p, s = self._producer, self._stages
        wp.launch(commit_bias, self._capacity,
                  inputs=[p.input_active, s.gate_bias, s.status_rne, p.qvel_source_epoch,
                          p.first_error, self._work['qfrc_bias'], self.qfrc_bias,
                          p.bias_valid, p.bias_written_epoch, self._width],
                  device=self._device, stream=self._stream)

    def launch_smooth_commit(self):
        self.validate()
        p, s = self._producer, self._stages
        wp.launch(commit_smooth, self._capacity,
                  inputs=[p.input_active, s.gate_acceleration, s.acceleration.status,
                          p.force_source_epoch, p.first_error, self._work['qfrc_smooth'],
                          self._work['qacc_smooth'], self.qfrc_smooth, self.qacc_smooth,
                          p.smooth_valid, p.smooth_written_epoch, p.smooth_repaired, self._width],
                  device=self._device, stream=self._stream)

    def catalog(self):
        self.validate()
        def describe(array):
            return dict(zip(self.TOKEN_FIELDS, self._token(array)))
        return {'public': {name: describe(getattr(self, name)) for name in self.NAMES},
                'work': {name: describe(self._work[name]) for name in self.NAMES},
                'bias_consumers': self.prepare_binding_delta,
                'stream_cuda': self._frozen_stream[1],
                'device_context': self._frozen_stream[2]}
