"""Cold-prepared operand evidence; diagnostic copies never feed physics.

No callbacks are installed into an existing frozen runtime. A fresh assembler
calls each capture method at its actual consumer boundary. Model export is a
synchronized boundary operation, never a hot oracle.
"""
from pathlib import Path
import hashlib
import json
import shutil
from .tape import DeviceInputTape


def _record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def actual_model_descriptor(bundle, *, output=None):
    """Pin this arena's actual model blob and identity, not a raw ancestor."""
    if getattr(bundle, 'closed', False):
        raise RuntimeError('closed arena has no current model')
    from ..core.model_bytes import model_bytes
    raw = model_bytes(bundle.model_pack.model)
    model = dict(storage='native-model', sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    if model['sha256'] != bundle.model_spec.model_sha256:
        raise ValueError('actual model blob differs from current ModelSpec')
    packet = dict(schema='actual-runtime-model-descriptor-v1', model=model,
                  model_identity=bundle.model_spec.identity_key,
                  dimensions={k: int(getattr(bundle.model_pack.model, k))
                              for k in ('nq', 'nv', 'nM', 'nu')},
                  independent_numerical_licence=False)
    if output is not None:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=False)
        destination = directory / 'ACTUAL_MODEL.mjb'
        destination.write_bytes(raw)
        if _record(destination)['sha256'] != model['sha256']:
            raise RuntimeError('model transport changed bytes')
        packet['model'] = _record(destination)
        (directory / 'MODEL_DESCRIPTOR.json').write_text(json.dumps(packet, indent=2) + '\n')
    return packet


def actual_model_descriptor_from_task(task, *, output=None):
    """Use the public synchronized owner witness to transport its current blob.

    This works with the existing R10 task interface. It neither inspects Python
    closure cells nor searches filenames for a replacement model.
    """
    from ..core.model_bytes import model_bytes
    with task._lock:
        task._open()
        witness = task.gpu_physics_evidence()
        raw = model_bytes(task.built.native_model, witness['model_sha256'])
        packet = dict(schema='actual-task-model-descriptor-v1',
            model=dict(storage='native-model', sha256=witness['model_sha256'], bytes=len(raw)),
            model_identity=witness['model_identity'],
            device_model_epoch=witness['device_model_epoch'],
            independent_model_expected=witness['independent_model_expected'],
            device_model_ready=witness['device_model_ready'],
            source_records=witness['source_records'], independent_numerical_licence=False)
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=False)
            destination = directory / 'ACTUAL_MODEL.mjb'
            destination.write_bytes(raw)
            packet['model'] = _record(destination)
            if packet['model']['sha256'] != witness['model_sha256']:
                raise RuntimeError('current model transport changed bytes')
            (directory / 'MODEL_DESCRIPTOR.json').write_text(json.dumps(packet, indent=2) + '\n')
        return packet


def actual_stage_sources(bundle):
    """Return retained operands without labelling an endpoint as an earlier input."""
    if getattr(bundle, 'closed', False):
        raise RuntimeError('closed arena')
    forward = bundle.runtime.forward
    s, n, e = bundle.constraints, bundle.newton, bundle.runtime.euler
    pre_s = dict(s._sources)
    pre_s.update(s._velocity)
    pre_n = dict(forward.entry_sources)
    # These are actual derived N producers, not host contact reconstructions.
    for key in ('cone_dim', 'cone_mu', 'cone_friction'):
        pre_n[key] = getattr(n.ws, key)
    pre_e = {key: getattr(e, key) for key in
             ('qpos', 'qvel', 'time', 'qM', 'qfrc_smooth', 'qfrc_constraint',
              'qacc', 'ctrl', 'qacc_warmstart')}
    epochs = dict(state_epoch=bundle.runtime.state_epoch,
                  expected_state=bundle.expected_state,
                  forward_epoch=bundle.dynamics.forward_epoch,
                  expected_forward=forward.expected_forward,
                  N_attempt=n.n_attempt, N_expected=n.n_expected, N_written=n.n_written)
    for sources in (pre_s, pre_n, pre_e):
        sources.update(epochs)
    return dict(preS=pre_s, preN=pre_n, preEuler=pre_e)


class PreparedStageEvidence:
    """Bounded device tapes for a new source epoch with explicit callbacks.

Capture preS before S consumes its operands, preN after projection before N,
and preEuler after physics publication before integration. Row counters are
real device events. Overflow cannot affect physics.
    """
    def __init__(self, bundle, *, capacity):
        self.bundle, self.stream = bundle, bundle.stream
        self.model = actual_model_descriptor(bundle)
        self.sources = actual_stage_sources(bundle)
        self.tapes = {phase: DeviceInputTape(sources, capacity=capacity,
                                           stream=self.stream, phase=phase)
                      for phase, sources in self.sources.items()}

    def validate(self):
        current = actual_stage_sources(self.bundle)
        for phase, sources in self.sources.items():
            if set(current[phase]) != set(sources) or any(
                    current[phase][key] is not value for key, value in sources.items()):
                raise ValueError('actual evidence producer owner changed')

    def capture_preS(self):
        self.validate()
        self.tapes['preS'].capture()

    def capture_preN(self):
        self.validate()
        self.tapes['preN'].capture()

    def capture_preEuler(self):
        self.validate()
        self.tapes['preEuler'].capture()
