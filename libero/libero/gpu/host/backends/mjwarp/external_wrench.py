# Copyright 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent applied/contact prefix selector; unsupported wrenches upstream."""
from libero.gpu.host.runtime.dynamics.mjwarp_external_wrench import ExternalWrenchNumerics,qualify
from .postconstraint_external import assemble_external


def external_wrench_product(root,*,fast_compile=False):
    import subprocess
    from libero.gpu.host.build.project_paths import project_paths
    from libero.gpu.host.build.compilation import MissingCompiledProduct
    folder=project_paths(root).cmake/('sm80-dev' if fast_compile else 'sm80-release')
    target='flash_mjwarp_external_wrench_fp32';library=folder/'mjwarp_external_wrench_fp32.so'
    result=subprocess.run(['ninja','-C',str(folder),'-n',target],capture_output=True,text=True)
    if not library.is_file() or result.returncode or 'ninja: no work to do.' not in result.stdout:raise MissingCompiledProduct('Build '+target+' offline in '+str(folder))
    return library


class ExternalWrench:
    def __init__(self,model,data,stream,*,implementation='upstream',library=None,admission=None):
        self.model,self.data=model,data
        self.admission=qualify(model,data,implementation=implementation) if admission is None else admission
        if self.admission['model']!=id(model) or self.admission['data']!=id(data):raise ValueError('External wrench admission owner differs')
        if self.admission['requested']!=implementation:raise ValueError('External wrench admission selector differs')
        self.selected=self.admission['selected']
        self.binding=ExternalWrenchNumerics(model,data,library,stream,admission=self.admission) if self.selected=='cuda' else None

    @property
    def identity(self):
        return dict(requested=self.admission['requested'],selected=self.selected,fallback=self.admission['fallback'],
            **(self.binding.identity if self.binding is not None else {}))

    def enqueue(self):
        if self.binding is None:assemble_external(self.model,self.data)
        else:self.binding.enqueue()
