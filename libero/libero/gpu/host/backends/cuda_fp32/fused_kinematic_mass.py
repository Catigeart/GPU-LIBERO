"""Load-only current kinematics/centered/mass pipeline and accepted fallback."""
import ctypes as ct
import hashlib
import numpy as np
from pathlib import Path
from libero.gpu.host.runtime.dynamics.mjwarp_kinematics import KinematicsInputs,FrameAttachments
from libero.gpu.host.runtime.dynamics.mjwarp_camlight import CamLightInputs
from libero.gpu.host.runtime.dynamics.mjwarp_com_pos import ComPosInputs
from libero.gpu.host.runtime.dynamics.mjwarp_mass_matrix import MassMatrixInputs
from libero.gpu.host.runtime.dynamics.mjwarp_motion import reject_output_overlap


def source_id():
    from libero.gpu.host.paths import GPU_ROOT
    names=('cuda/fp32/fused_kinematic_mass.cu', 'cuda/dynamics/mjwarp_kinematics.cuh', 'cuda/dynamics/mjwarp_camlight.cuh', 'cuda/dynamics/mjwarp_com_pos.cuh', 'cuda/dynamics/mjwarp_mass_matrix.cuh', 'cmake/fp32/FP32KinematicMass.cmake')
    return hashlib.sha256(':'.join(hashlib.sha256((GPU_ROOT/n).read_bytes()).hexdigest()
                                  for n in names).encode()).hexdigest()


class FusedKinematicMass:
    def __init__(self,kinematics,centered_mass,library):
        from libero.gpu.host.build.compilation import MissingCompiledProduct
        k=kinematics;c,m=centered_mass.components
        if (k.model is not c.model or k.data is not c.data or k.stream is not c.stream or
                k.topology.prepare_layer_tree() is not centered_mass.tree):
            raise ValueError('Kinematic/centered/mass owners differ')
        # These are immutable index bridges, not cached physical quantities.
        for left,right,name in ((k.arrays['parent'],c.arrays['parent'],'parents'),
                (k.arrays['parent'],m.arrays['body_parent'],'mass parents'),
                (k.arrays['root'],c.arrays['root'],'roots'),
                (k.arrays['kind'],c.arrays['kind'],'joint kinds'),
                (k.topology.arrays['joint_dof'],c.arrays['dofadr'],'joint DOFs'),
                (k.topology.arrays['dof_body'],m.arrays['dof_body'],'DOF bodies')):
            if left.ptr!=right.ptr:raise ValueError('Kinematic/centered index bridge differs: '+name)
        expected=np.full(k.model.njnt,-1,np.int32)
        for body,(count,first) in enumerate(zip(k.arrays['jnum'].numpy(),k.arrays['jadr'].numpy())):
            if count:expected[first:first+count]=body
        if not np.array_equal(expected,c.arrays['jbody'].numpy()):
            raise ValueError('Kinematic/centered joint body mapping differs')
        for key in ('xipos','ximat','xmat','anchor','axis'):
            if k.arrays[key].ptr!=c.arrays[key].ptr:raise ValueError('Kinematic/centered bridge differs: '+key)
        arrays={'kin_'+key:a for key,a in k.arrays.items()}
        arrays.update({'frame_'+key:a for key,a in k.frames.arrays.items() if key not in ('xpos','xquat')})
        arrays.update({'centered_'+key:a for key,a in c.arrays.items() if key not in ('xipos','ximat','xmat','anchor','axis')})
        arrays.update({'mass_'+key:a for key,a in m.arrays.items() if key not in ('cinert','cdof')})
        reject_output_overlap(arrays,tuple('kin_'+key for key in ('xpos','xquat','anchor','axis','xmat','xipos','ximat','geom_xpos','geom_xmat','site_xpos','site_xmat'))+tuple('frame_'+key for key in ('cam_xpos','cam_xmat','light_xpos','light_xdir'))+('centered_subtree','centered_cinert','centered_cdof','mass_crb','mass_matrix'))
        self.kinematics,self.centered_mass,self.stream=k,centered_mass,k.stream
        self.interval=centered_mass.interval
        self.eligible=(self.interval is not None and k.model.nbranch<=16 and k.combine_frames
                       and k.model.njnt<=(640-3*k.model.nbody)//4)
        self.path=Path(library);self.library=None
        if not self.eligible:return
        if not self.path.is_file():raise MissingCompiledProduct('Build flash_fp32_fused_kinematic_mass offline')
        self.library=lib=ct.CDLL(str(self.path))
        if lib.flash_fp32_kinematic_mass_arch()!=k.stream.device.arch:raise ValueError('Kinematic/mass architecture differs')
        for name,kind in (('kinematics',KinematicsInputs),('frames',CamLightInputs),('frame_ids',FrameAttachments),('centered',ComPosInputs),('mass',MassMatrixInputs)):
            size=getattr(lib,'flash_fp32_kinematic_mass_'+name+'_bytes');size.restype=ct.c_size_t
            if size()!=ct.sizeof(kind):raise ValueError('Kinematic/mass ABI differs: '+name)
        self.launch=lib.flash_fp32_kinematic_mass_launch
        self.launch.argtypes=[ct.POINTER(kind) for kind in (KinematicsInputs,CamLightInputs,FrameAttachments,ComPosInputs,MassMatrixInputs)]+[ct.c_void_p,ct.c_void_p]
        self.launch.restype=ct.c_int

    def enqueue(self):
        k=self.kinematics;c,m=self.centered_mass.components
        if not self.eligible:k.enqueue_frames();self.centered_mass.enqueue();return
        error=self.launch(ct.byref(k.arguments),ct.byref(k.frames.arguments),ct.byref(k.frame_ids),
                          ct.byref(c.arguments),ct.byref(m.arguments),self.interval.ptr,self.stream.cuda_stream)
        if error:raise RuntimeError('Kinematic/mass pipeline failed: CUDA error '+str(error))
