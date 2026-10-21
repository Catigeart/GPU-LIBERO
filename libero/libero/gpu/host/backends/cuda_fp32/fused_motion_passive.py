"""Load-only world motion/passive region with accepted component fallback."""
import ctypes as ct
import hashlib
from pathlib import Path
from libero.gpu.host.runtime.dynamics.mjwarp_velocity import VelocityInputs
from libero.gpu.host.runtime.dynamics.mjwarp_bias import BiasInputs
from libero.gpu.host.runtime.dynamics.mjwarp_fluid import FluidInputs
from libero.gpu.host.runtime.dynamics.mjwarp_passive import PassiveInputs
from libero.gpu.host.runtime.dynamics.mjwarp_motion import reject_output_overlap


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_motion_passive.cu','cuda/dynamics/mjwarp_velocity.cuh',
           'cuda/dynamics/mjwarp_bias.cuh','cuda/dynamics/mjwarp_fluid.cuh',
           'cuda/dynamics/mjwarp_passive.cuh','cuda/dynamics/mjwarp_passive_math.cuh',
           'cmake/FP32MotionPassive.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/name).read_bytes()).hexdigest()
                                  for name in names).encode()).hexdigest()


class FusedMotionPassive:
    def __init__(self,motion,fluid,library):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        v,b=motion.components;p=fluid.passive
        if (v.model is not fluid.model or v.data is not fluid.data or v.stream is not fluid.stream or
                p.model is not v.model or p.data is not v.data or p.stream is not v.stream):
            raise ValueError('Motion/passive owners differ')
        for key,left,right in (('cvel',v.arrays['cvel'],fluid.arrays['cvel']),
                ('cdof',v.arrays['cdof'],fluid.arrays['cdof']),('qvel',v.arrays['qvel'],p.arrays['qvel']),
                ('DOF bodies',b.arrays['dof_body'],fluid.arrays['dof_body']),
                ('parents',v.topology.arrays['body_parent'],fluid.arrays['parent']),
                ('joint types',v.topology.arrays['jnt_type'],p.arrays['joint_type']),
                ('joint DOFs',v.topology.arrays['joint_dof'],p.arrays['dofadr'])):
            if left.ptr!=right.ptr:raise ValueError('Motion/passive bridge differs: '+key)
        if p.flags['gravity_enabled']!=int(not b.gravity_disabled):raise ValueError('Motion/passive gravity flags differ')
        arrays={'v_'+k:a for k,a in v.arrays.items()}
        arrays.update({'b_'+k:a for k,a in b.arrays.items() if k not in ('qvel','cdof','cvel','cdof_dot')})
        arrays.update({'f_'+k:a for k,a in fluid.arrays.items() if k not in ('cdof','cvel')})
        arrays.update({'p_'+k:a for k,a in p.arrays.items() if k not in ('qvel','fluid')})
        reject_output_overlap(arrays,('v_cvel','v_cdof_dot','b_cacc','b_cfrc_int','b_qfrc_bias',
            'f_body_force','f_qfrc','p_spring','p_damper','p_adhesion','p_passive'))
        self.motion,self.fluid,self.stream=motion,fluid,v.stream
        self.eligible=motion.eligible and not (p.flags['disable_spring'] and p.flags['disable_damper'])
        self.path=Path(library);self.library=None
        if not self.eligible:return
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_motion_passive offline')
        self.library=lib=ct.CDLL(str(self.path));lib.flash_fp32_motion_passive_source_id.restype=ct.c_char_p
        if lib.flash_fp32_motion_passive_source_id().decode()!=source_id():raise MissingCompiledProduct('Stale motion/passive; rebuild offline')
        if lib.flash_fp32_motion_passive_arch()!=self.stream.device.arch:raise ValueError('Motion/passive architecture differs')
        for name,kind in (('velocity',VelocityInputs),('bias',BiasInputs),('fluid',FluidInputs),('passive',PassiveInputs)):
            f=getattr(lib,'flash_fp32_motion_passive_'+name+'_bytes');f.restype=ct.c_size_t
            if f()!=ct.sizeof(kind):raise ValueError('Motion/passive ABI differs: '+name)
        self.launch=lib.flash_fp32_motion_passive_launch
        self.launch.argtypes=[ct.POINTER(VelocityInputs),ct.POINTER(BiasInputs),ct.POINTER(FluidInputs),
            ct.POINTER(PassiveInputs),ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
        self.launch.restype=ct.c_int

    @property
    def identity(self):
        return dict(selected='world-motion-passive' if self.eligible else 'component-fallback',
            source_id=source_id(),library_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
            if self.library is not None else None,numerical_workspace_bytes=0,
            additional_topology_bytes=0,values='borrow-current-input-no-cache')

    def enqueue(self):
        if not self.eligible:
            self.motion.enqueue();self.fluid.enqueue_joint_passive();return
        v,b=self.motion.components;f=self.fluid;p=f.passive
        err=self.launch(ct.byref(v.arguments),ct.byref(b.arguments),ct.byref(f.arguments),ct.byref(p.arguments),
            p.arrays['adhesion'].ptr,self.motion.interval.ptr,int(b.gravity_disabled),self.stream.cuda_stream)
        if err:raise RuntimeError('Motion/passive failed: CUDA error '+str(err))
