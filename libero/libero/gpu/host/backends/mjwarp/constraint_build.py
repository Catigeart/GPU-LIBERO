# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent rigid dense constraint construction selector."""
from mujoco_warp._src import constraint
from libero.gpu.host.runtime.dynamics.mjwarp_constraint_build import ConstraintConstruction,qualify


def constraint_build_product(root,*,arch,fast_compile=False):
    from libero.gpu.host.build.project_paths import cmake_directory
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=cmake_directory(root, arch, fast_compile=fast_compile)
    target='flash_mjwarp_constraint_build_fp32';library=folder/'mjwarp_constraint_build_fp32.so'
    if not library.is_file():raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class ConstraintBuild:
    def __init__(self,model,data,stream,*,implementation='upstream',library=None,admission=None):
        self.model,self.data=model,data
        self.admission=qualify(model,data,implementation=implementation) if admission is None else admission
        if self.admission['model']!=id(model) or self.admission['data']!=id(data):raise ValueError('Constraint construction admission owner differs')
        if self.admission['requested']!=implementation:raise ValueError('Constraint construction admission selector differs')
        self.selected=self.admission['selected']
        self.binding=ConstraintConstruction(model,data,library,stream,admission=self.admission) if self.selected=='cuda' else None

    @property
    def identity(self):
        return dict(requested=self.admission['requested'],selected=self.selected,fallback=self.admission['fallback'],
            **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        if self.binding is None:constraint.make_constraint(self.model,self.data)
        else:self.binding.enqueue()
