"""Explicit CRB leaf; tendon armature and factorization are separate stages."""
from mujoco_warp._src import smooth, types
from libero.gpu.host.runtime.dynamics.mjwarp_mass_matrix import MassMatrix


def caller_unsupported_reason(model, data):
    if model.opt.integrator != types.IntegratorType.EULER:
        return 'mass-matrix caller requires Euler'
    if model.opt.enableflags & types.EnableBit.SLEEP:
        return 'active-body compaction'
    if model.nv < 1 or model.nbody < 1:
        return 'empty mass-matrix topology'
    return None


def matrix_product(root, *, fast_compile=False):
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder = project_paths(root).cmake / ('sm80-dev' if fast_compile else 'sm80-release')
    target = 'flash_mjwarp_mass_matrix_fp32'; library = folder / 'mjwarp_mass_matrix_fp32.so'
    if not library.is_file():
        raise MissingCompiledProduct('Build ' + target + ' offline in ' + str(folder))
    return library


class MassMatrices:
    def __init__(self, model, data, stream, *, implementation='upstream', library=None):
        if implementation not in ('upstream', 'cuda'):
            raise ValueError('invalid mass matrix implementation')
        self.model, self.data = model, data
        self.requested = implementation
        self.fallback = caller_unsupported_reason(model, data) if implementation == 'cuda' else None
        self.selected = 'upstream' if self.fallback else implementation
        self.binding = MassMatrix(model, data, library, stream) if self.selected == 'cuda' else None

    @property
    def identity(self):
        return dict(requested=self.requested, selected=self.selected, fallback=self.fallback,
                    **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        if self.binding is None:
            smooth.crb(self.model, self.data)
        else:
            self.binding.enqueue()
