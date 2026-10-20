"""Cold verified single-arm selectors and device-native joint input gathering."""
import libero.gpu.precision as scalar
import json
from pathlib import Path
import numpy as np
import warp as wp
from libero.gpu.host.paths import GPU_ROOT
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.runtime.dynamics.specialization import native_fields, validate_d_topology
from libero.gpu.host.runtime.dynamics.runtime_adapter import _array, _token

class GenericJointInputsWorkspace:
    FIELDS=("body_parentid","body_rootid","dof_bodyid","dof_parentid","dof_Madr",
            "dof_simplenum","jnt_type","jnt_qposadr","jnt_dofadr")
    def __init__(self,arrays,metadata,spec,capacity,*,robot,registry,candidate_id,
                 source_records,metadata_record,mode="production",device="cuda:0",inputs=None):
        self.static,self.admission=native_fields(arrays,metadata,spec,self.FIELDS,
            source_records=source_records,metadata_record=metadata_record,registry=registry,
            candidate_id=candidate_id,mode=mode)
        original=json.loads(Path(metadata_record["path"]).read_text())
        if robot not in original.get("robot_metadata",()):raise ValueError("joint selectors absent from actual metadata source")
        d=metadata["dimensions"];validate_d_topology(self.static,d)
        self.capacity,self.device=int(capacity),wp.get_device(device)
        self.spec,self.registry,self.candidate_id,self.mode=spec,registry,candidate_id,mode
        self.nq,self.nv,self.nM=(int(d[k]) for k in ("nq","nv","nM"))
        joints=np.asarray(robot["joint_ids"])
        n=self.joint_dim=len(joints)
        if joints.dtype.kind not in "iu" or not 1<=n<=32 or np.any(joints<0) or np.any(joints>=d["njnt"]):
            raise ValueError("actual single-arm scalar joint selectors required")
        if np.any(~np.isin(self.static["jnt_type"][joints],[2,3])):raise ValueError("scalar arm joints required")
        qi,vi=self.static["jnt_qposadr"][joints],self.static["jnt_dofadr"][joints]
        if list(qi)!=robot["joint_qpos_indices"] or list(vi)!=robot["joint_qvel_indices"]:
            raise ValueError("native arm addresses differ from original metadata")
        sparse=np.full((self.nv,self.nv),-1,np.int32)
        for i in range(self.nv):
            j,address=i,int(self.static["dof_Madr"][i])
            while j>=0:
                if not 0<=address<self.nM:raise ValueError("native sparse mass address outside actual nM")
                sparse[i,j]=sparse[j,i]=address
                j,address=int(self.static["dof_parentid"][j]),address+1
        self.qi=wp.array(qi,dtype=wp.int32,device=self.device)
        self.vi=wp.array(vi,dtype=wp.int32,device=self.device)
        self.sparse=wp.array(sparse[np.ix_(vi,vi)].reshape(-1),dtype=wp.int32,device=self.device)
        self.joint_pos=wp.zeros((capacity,n),dtype=scalar.wp_real,device=self.device)
        self.joint_vel=wp.zeros((capacity,n),dtype=scalar.wp_real,device=self.device)
        self.M=wp.zeros((capacity,n*n),dtype=scalar.wp_real,device=self.device)
        self.status=wp.zeros(capacity,dtype=wp.int32,device=self.device)
        self._frozen={k:_token(v) for k,v in vars(self).items() if isinstance(v,wp.array)}
        self._plans=None

    def validate_bindings(self):
        if not self.registry.admit(self.candidate_id,self.spec,mode=self.mode,
            required_scope="component-runtime" if self.mode=="production" else None).execution_allowed:
            raise ValueError("joint inputs source admission changed")
        for k,tok in self._frozen.items():
            if _token(getattr(self,k))!=tok:raise ValueError("joint inputs owner changed: "+k)

    def install(self,dynamics,build_root):
        self.validate_bindings();d=dynamics;c=d.control
        prefix=[d.first_error,d.state_epoch,d.source_state_epoch,d.forward_epoch,
            d.valid["bias"],d.written["bias"]]
        header=GPU_ROOT/"cuda/control/joint_runtime.cuh"
        self._gather=NativePlan("joint_inputs",[header],[("joint_cuda::gather",[d.gate_control,*prefix,
            d.qpos,d.qvel,d.position.qM,self.qi,self.vi,self.sparse,c.phys_q,c.phys_qvel,c.phys_mass,
            self.nq,self.nv,self.nM,self.joint_dim],"lane0")],self.capacity,Path(build_root)/"inputs",device=self.device)
        self._bias=NativePlan("joint_live_bias",[header],[("joint_cuda::live_bias",[d.gate_control,*prefix,
            d.public["qfrc_bias"],c.phys_bias,self.vi,d.controller_parent_valid,d.controller_parent_written,
            self.nv,self.joint_dim],"lane0")],self.capacity,Path(build_root)/"bias",device=self.device)
        self._plans=(self._gather,self._bias);self._owner=d
        return self

    def gather(self,mask,stream=None):
        self.validate_bindings()
        if self._plans is None:raise RuntimeError("joint input plans require cold install")
        d=self._owner;_array(mask,(self.capacity,),wp.int32,self.device)
        wp.copy(d.gate_control,mask,stream=stream);self._gather.launch(stream)

    def live_bias(self,mask,stream=None):
        self.validate_bindings()
        if self._plans is None:raise RuntimeError("joint bias plan requires cold install")
        d=self._owner;_array(mask,(self.capacity,),wp.int32,self.device)
        wp.copy(d.gate_control,mask,stream=stream);self._bias.launch(stream)
