"""Scene-only camera/light transforms from already-produced GPU body poses."""
import libero.gpu.precision as scalar
import numpy as np
import warp as wp
wp.set_module_options({"enable_backward":False,"fast_math":False,"fuse_fp":True})

@wp.kernel
def camera_kernel(nbody:int,ncam:int,body_xpos:wp.array(dtype=scalar.wp_real),body_xmat:wp.array(dtype=scalar.wp_real),
                  mask:wp.array(dtype=wp.int32),bodyid:wp.array(dtype=wp.int32),local_pos:wp.array(dtype=scalar.wp_real),local_mat:wp.array(dtype=scalar.wp_real),
                  cam_xpos:wp.array(dtype=scalar.wp_real),cam_xmat:wp.array(dtype=scalar.wp_real)):
    world,camera=wp.tid()
    if mask[world]==0:return
    body=bodyid[camera];bo=world*nbody+body;co=world*ncam+camera
    for i in range(3):
        position=body_xpos[bo*3+i]
        for j in range(3):position+=body_xmat[bo*9+i*3+j]*local_pos[camera*3+j]
        cam_xpos[co*3+i]=position
        for j in range(3):
            value=scalar.wp_real(0.0)
            for k in range(3):value+=body_xmat[bo*9+i*3+k]*local_mat[camera*9+k*3+j]
            cam_xmat[co*9+i*3+j]=value

@wp.kernel
def light_kernel(nbody:int,nlight:int,body_xpos:wp.array(dtype=scalar.wp_real),body_xmat:wp.array(dtype=scalar.wp_real),
                 mask:wp.array(dtype=wp.int32),bodyid:wp.array(dtype=wp.int32),local_pos:wp.array(dtype=scalar.wp_real),local_dir:wp.array(dtype=scalar.wp_real),
                 light_xpos:wp.array(dtype=scalar.wp_real),light_xdir:wp.array(dtype=scalar.wp_real)):
    world,light=wp.tid()
    if mask[world]==0:return
    body=bodyid[light];bo=world*nbody+body;lo=world*nlight+light
    for i in range(3):
        position=body_xpos[bo*3+i];direction=scalar.wp_real(0.0)
        for j in range(3):
            position+=body_xmat[bo*9+i*3+j]*local_pos[light*3+j]
            direction+=body_xmat[bo*9+i*3+j]*local_dir[light*3+j]
        light_xpos[lo*3+i]=position;light_xdir[lo*3+i]=direction

class DeviceSceneTransforms:
    """Fixed camera/light modes only, guarded by actual native model options.

    Static quaternion-to-matrix preparation on CPU is model compilation, not a
    dynamic physics calculation. A model-parameter generation change requires a
    new descriptor/explicit static upload before the next scene publication.
    """
    def __init__(self,model,nworld=1,device="cuda:0"):
        import mujoco
        self.host_camlight=bool(np.any(np.asarray(model.cam_mode)!=int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED))
                               or np.any(np.asarray(model.light_mode)!=int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)))
        # Fixed modes retain the original GPU producer. Non-fixed modes receive
        # scene-only mj_camlight at the explicit GL boundary from GPU body fields;
        # zero device scratch is never treated as the final camera publication.
        self.nbody=int(model.nbody);self.ncam=int(model.ncam);self.nlight=int(model.nlight)
        self.nworld=nworld;self.device=device
        self.all_worlds=wp.array([1]*nworld,dtype=wp.int32,device=device)
        def a(value,dtype):return wp.array(np.asarray(value).reshape(-1),dtype=dtype,device=device)
        mats=np.zeros((self.ncam,9))
        for i in range(self.ncam):mujoco.mju_quat2Mat(mats[i],np.asarray(model.cam_quat[i]))
        self.camera_static=[a(model.cam_bodyid,wp.int32),a(model.cam_pos,scalar.wp_real),a(mats,scalar.wp_real)]
        self.light_static=[a(model.light_bodyid,wp.int32),a(model.light_pos,scalar.wp_real),a(model.light_dir,scalar.wp_real)]
        self.cam_xpos=wp.zeros(nworld*self.ncam*3,dtype=scalar.wp_real,device=device)
        self.cam_xmat=wp.zeros(nworld*self.ncam*9,dtype=scalar.wp_real,device=device)
        self.light_xpos=wp.zeros(nworld*self.nlight*3,dtype=scalar.wp_real,device=device)
        self.light_xdir=wp.zeros(nworld*self.nlight*3,dtype=scalar.wp_real,device=device)

    def update(self,body_xpos,body_xmat,world_mask=None,stream=None):
        for field,size in ((body_xpos,self.nworld*self.nbody*3),(body_xmat,self.nworld*self.nbody*9)):
            if field.dtype!=scalar.wp_real or field.ndim!=1 or field.size!=size:
                raise ValueError("scene input must be flat FP64 body arrays")
        if self.ncam:
            wp.launch(camera_kernel,dim=(self.nworld,self.ncam),inputs=[self.nbody,self.ncam,body_xpos,body_xmat,
                      self.all_worlds if world_mask is None else world_mask,*self.camera_static,self.cam_xpos,self.cam_xmat],device=self.device,stream=stream)
        if self.nlight:
            wp.launch(light_kernel,dim=(self.nworld,self.nlight),inputs=[self.nbody,self.nlight,body_xpos,body_xmat,
                      self.all_worlds if world_mask is None else world_mask,*self.light_static,self.light_xpos,self.light_xdir],device=self.device,stream=stream)
        return {"cam_xpos":self.cam_xpos,"cam_xmat":self.cam_xmat,"light_xpos":self.light_xpos,"light_xdir":self.light_xdir}