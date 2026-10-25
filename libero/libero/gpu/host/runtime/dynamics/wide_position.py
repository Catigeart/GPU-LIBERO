"""Fresh full-angle position owners; original mass/frame kernels unchanged."""
import libero.gpu.precision as scalar
from ..core.static_readset import StaticReadsetSeal
from types import SimpleNamespace
import numpy as np
import warp as wp
from .specialization import GenericPositionWorkspace, native_fields, validate_d_topology

class GenericWidePositionWorkspace(GenericPositionWorkspace):
    def __init__(self, arrays, metadata, spec, capacity, *, registry, candidate_ids,
                 source_records, metadata_record, mode='production', device='cuda:0'):
        from types import SimpleNamespace
        from libero.gpu.host.runtime.foundation.runtime_adapter import PreparedKinematicModel, PreparedFrameWorkspace
        from libero.gpu.reference.position.body_kinematics.wide_angle import WideAngleBodyKinematicsWorkspace as BodyKinematicsWorkspace
        from libero.gpu.reference.position.compos.kernel import ComposModel
        from libero.gpu.reference.position.crb.kernel import CRBModel
        from libero.gpu.reference.integration.euler_integrator.generic_workspace import GenericMassWorkspace
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('positive actual capacity required')
        fields, self.admission = native_fields(arrays, metadata, spec, self.MASS_FIELDS,
            source_records=source_records, metadata_record=metadata_record, registry=registry,
            candidate_id=candidate_ids['mass'], mode=mode)
        d = metadata['dimensions']
        validate_d_topology(fields, d)
        self.capacity, self.device = capacity, wp.get_device(device)
        self.spec, self.registry, self.candidate_ids, self.mode = spec, registry, dict(candidate_ids), mode
        frozen = PreparedKinematicModel(arrays, metadata, spec, registry=registry,
            candidate_id=candidate_ids['kinematics'], source_records=source_records,
            metadata_record=metadata_record, mode=mode)
        self.body = BodyKinematicsWorkspace(frozen, capacity, device=self.device)
        nb, nj, nv, nm = (int(d[k]) for k in ('nbody', 'njnt', 'nv', 'nM'))
        self.frame_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.frames = PreparedFrameWorkspace(arrays, metadata, spec, capacity,
            registry=registry, candidate_id=candidate_ids['frames'], source_records=source_records,
            metadata_record=metadata_record, mode=mode, device=self.device,
            body_inputs={'body_xpos': self.body.xpos.reshape((capacity*nb*3,)),
                'body_xquat': self.body.xquat.reshape((capacity*nb*4,)),
                'body_xmat': self.body.xmat.reshape((capacity*nb*9,)), 'active_world': self.frame_active})
        self.model = SimpleNamespace(**d, **fields)
        self.compos = ComposModel.prepare(self.model, device=self.device)
        self.crb_model = CRBModel.prepare(self.model, device=self.device)
        self.factor = GenericMassWorkspace(metadata, fields, capacity, self.device)
        self.xipos = self.frames.xipos.reshape((capacity, nb*3))
        self.ximat = self.frames.ximat.reshape((capacity, nb*9))
        self.crb_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        self.mass_active = wp.zeros(capacity, dtype=wp.int32, device=self.device)
        for name, width in (('subtree_com', nb*3), ('cinert', nb*10), ('cdof', nv*6), ('crb', nb*10)):
            setattr(self, name, wp.zeros((capacity, width), dtype=scalar.wp_real, device=self.device))
        for name in ('compos_status', 'crb_status', 'first_stage', 'first_code', 'result_kind',
                     'mass_first_stage', 'mass_first_code', 'mass_result_kind'):
            setattr(self, name, wp.zeros(capacity, dtype=wp.int32, device=self.device))
        self.factor.count.fill_(1)
        self._objects = (self.body, self.frames, self.compos, self.crb_model, self.factor)
        self._fields = fields
        self._static_fields_seal = StaticReadsetSeal(fields, spec.identity_key, None)
        self._frozen = self._tokens()
