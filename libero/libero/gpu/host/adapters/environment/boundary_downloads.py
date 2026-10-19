"""Typed boundary downloads queued on one stream, with independent snapshots."""
import warp as wp


class BoundaryDownloads:
    def __init__(self):
        self.targets = {}

    def download(self, owner, names=None):
        from libero.gpu.host.adapters.observations.boundary import GOAL_FIELDS
        names = tuple(dict.fromkeys(GOAL_FIELDS if names is None else names))
        if 'contact' in names and 'ncon' not in names:
            names += ('ncon',)
        fields = {name: owner._fields[name].array for name in names if name != 'contact'}
        if 'contact' in names:
            fields.update({f'contact.{name}': owner.runtime.forward.transport.inputs[name]
                           for name in ('geom1', 'geom2')})
        try:
            for name, source in fields.items():
                target = self.targets.get(name)
                if target is None or target.dtype != source.dtype or target.shape != source.shape:
                    target = wp.empty(source.shape, dtype=source.dtype, device='cpu', pinned=True)
                    self.targets[name] = target
                wp.copy(target, source, stream=owner.stream)
        finally:
            if fields:
                wp.synchronize_stream(owner.stream)
        # Keep the public snapshot's ownership: a later download cannot mutate
        # a snapshot retained by a predicate or explicit host consumer.
        values = {name: self.targets[name].numpy().copy() for name in fields}
        snapshot = {name: values[name].reshape((owner.num_envs,)+owner._fields[name].shape)
                    for name in names if name != 'contact'}
        if 'contact' in names:
            pairs = {name: values[f'contact.{name}'].reshape(owner.num_envs, -1)
                     for name in ('geom1', 'geom2')}
            contacts = []
            for world, count in enumerate(snapshot['ncon']):
                count = int(count)
                if owner.debug and not 0 <= count <= pairs['geom1'].shape[1]:
                    raise RuntimeError('contact count exceeds batch capacity')
                contacts.append({name: value[world, :count] for name, value in pairs.items()})
            snapshot['contact'] = contacts
        return snapshot
