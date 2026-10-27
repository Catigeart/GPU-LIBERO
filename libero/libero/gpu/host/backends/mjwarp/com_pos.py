# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent pinned subtree CoM / inertia / motion-axis boundary."""
from mujoco_warp._src import smooth
from libero.gpu.host.runtime.dynamics.mjwarp_com_pos import CenteredFrames, caller_unsupported_reason


def com_pos_product(root,*,arch, fast_compile=False):
    from libero.gpu.host.build.project_paths import cmake_directory
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder = cmake_directory(root, arch, fast_compile=fast_compile)
    target = 'flash_mjwarp_com_pos_fp32'; library = folder/'mjwarp_com_pos_fp32.so'
    if not library.is_file():
        raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class ComPos:
    def __init__(self, model, data, stream, *, implementation='upstream', library=None):
        if implementation not in ('upstream','cuda'):
            raise ValueError('com_pos must be upstream or cuda')
        self.model, self.data = model, data
        self.requested = implementation
        self.fallback = caller_unsupported_reason(model,data) if implementation == 'cuda' else None
        self.selected = 'upstream' if self.fallback else implementation
        self.binding = CenteredFrames(model,data,library,stream) if self.selected == 'cuda' else None

    @property
    def identity(self):
        return dict(requested=self.requested,selected=self.selected,fallback=self.fallback,
                    **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        if self.binding is None:
            smooth.com_pos(self.model,self.data)
        else:
            self.binding.enqueue()
