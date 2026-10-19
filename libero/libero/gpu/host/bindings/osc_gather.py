"""Handwritten native-count OSC gather, including producer gates and publication."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import types
import warp as wp
from libero.gpu.host.build.native import NativePlan
from libero.gpu.host.runtime.dynamics import runtime_adapter as r

HEADER = (GPU_ROOT / 'cuda/control/osc_gather.cuh')

def _body_dofs(j):
    """Cold CSR from the admitted immutable ancestor owner, shared by worlds."""
    from libero.gpu.model.topology import prepare_ancestor_topology
    topology=getattr(j, '_cuda_osc_body_topology', None)
    if topology is None:
        topology=prepare_ancestor_topology(j.ancestor.numpy(), device=j.ancestor.device)
        j._cuda_osc_body_topology=topology
    return [topology.arrays['body_dof_offsets'],topology.arrays['body_dof_indices']]

def stages(j,*,dataflow_profile="compatible"):
    dofs=_body_dofs(j)
    if dataflow_profile=="fast":
        return [("cf_osc_gather::local_selected",[j.cdof,j.subtree_com,j.site_pos,j.active,
            j.ancestor,j.root,j.body,j.nv,j.nbody,j.jacp,j.jacr,j.jac_status,j.adapter_active,
            j.qpos,j.qvel,j.site_mat,j.qM,j.qfrc_bias,j.qi,j.vi,j.sparse,j.nq,j.nM,
            j.ee_pos,j.ee_mat,j.xvelp,j.xvelr,j.joint_pos,j.joint_vel,j.J,j.M,j.bias,
            j.adapter_status,j.status]+dofs,"block")]
    """Jacobian workspace stages; direct arrays retain native selectors."""
    return [
        ('cf_osc_gather::jacobian', [j.cdof,j.subtree_com,j.site_pos,j.active,
            j.ancestor,j.root,j.body,j.nv,j.nbody,j.jacp,j.jacr,j.jac_status]+dofs, 'block'),
        ('cf_osc_gather::route', [j.active,j.jac_status,j.adapter_active], 'lane0'),
        ('cf_osc_gather::selected', [j.qpos,j.qvel,j.site_pos,j.site_mat,j.jacp,j.jacr,
            j.qM,j.qfrc_bias,j.adapter_active,j.qi,j.vi,j.sparse,j.nq,j.nv,j.nM,
            j.ee_pos,j.ee_mat,j.xvelp,j.xvelr,j.joint_pos,j.joint_vel,j.J,j.M,j.bias,
            j.adapter_status], 'block'),
        ('cf_osc_gather::status', [j.jac_status,j.adapter_status,j.status], 'lane0')]

def install(adapter, root,*,dataflow_profile="compatible"):
    """Replace generic adapter gather as one kernel, plus stream mask copy.

    A private immutable mask owner keeps arbitrary admitted call-time masks
    graph-safe; copying occurs on the original stream before the kernel.
    Fixed-model D07 has a different selector schema and is explicitly rejected.
    """
    if not hasattr(adapter, 'jacobian'):
        raise TypeError('handwritten OSC gather requires native-count generic dynamics')
    j=adapter.jacobian
    adapter.validate();j.validate_bindings()
    mask=wp.zeros(adapter.capacity,dtype=wp.int32,device=adapter.device)
    chain=[('cf_osc_gather::gate',[mask,adapter.first_error,adapter.state_epoch,
        adapter.source_state_epoch,adapter.forward_epoch,adapter.valid['bias'],
        adapter.written['bias'],adapter.gate_control], 'lane0'),
        ('cf_osc_gather::site',[adapter.position.frames.site_xpos,
        adapter.position.frames.site_xmat,adapter.gate_control,adapter.site_position,
        adapter.site_rotation,adapter.nsite,int(adapter.robot['eef_site_id'])], 'lane0')]
    chain+=stages(j,dataflow_profile=dataflow_profile)
    chain.append(('cf_osc_gather::publish_velocity',[adapter.gate_control,j.status,
        j.xvelp,j.xvelr,adapter.control.phys_ee_velocity,adapter.first_error], 'lane0'))
    plan=NativePlan('osc_gather',[HEADER],chain,adapter.capacity,root,defines={'OSC_NV':j.nv})
    adapter._cuda_osc_gather_plan=plan
    bias_mask=wp.zeros(adapter.capacity,dtype=wp.int32,device=adapter.device)
    bias_plan=NativePlan('osc_live_bias',[HEADER],[('cf_osc_gather::live_bias',[
        bias_mask,adapter.first_error,adapter.state_epoch,adapter.source_state_epoch,
        adapter.forward_epoch,adapter.valid['bias'],adapter.written['bias'],
        adapter.public['qfrc_bias'],adapter.control.phys_bias,adapter._arm_indices,
        adapter.controller_parent_valid,adapter.controller_parent_written,adapter.nv],
        'lane0')],adapter.capacity,root)
    adapter._cuda_osc_bias_plan=bias_plan
    def gather(self, requested, stream=None):
        self.validate(stream);self.jacobian.validate_bindings()
        r._array(requested,(self.capacity,),wp.int32,self.device)
        wp.copy(mask,requested,stream=self.stream)
        self._cuda_osc_gather_plan.launch(self.stream)
    def bound(self, mask, stream=None):
        return gather(self,mask,stream)
    adapter.gather_controller_inputs=types.MethodType(bound,adapter)
    def live_bias(self,mask,stream=None):
        self.validate(stream);r._array(mask,(self.capacity,),wp.int32,self.device)
        wp.copy(bias_mask,mask,stream=self.stream)
        self._cuda_osc_bias_plan.launch(self.stream)
    adapter.gather_live_bias=types.MethodType(live_bias,adapter)
    return {'gather':plan.manifest,'live_bias':bias_plan.manifest}

def install_jacobian(workspace, root):
    """Optional standalone use with exactly the original workspace contract."""
    workspace.validate_bindings()
    plan=NativePlan('osc_jacobian',[HEADER],stages(workspace),workspace.capacity,root)
    workspace._cuda_jacobian_plan=plan
    def launch(self,stream=None):
        self.validate_bindings();self._cuda_jacobian_plan.launch(stream)
    workspace.launch=types.MethodType(launch,workspace)
    return plan.manifest
