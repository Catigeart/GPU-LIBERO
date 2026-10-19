// Compare cooperative scheduling with the ordered protocol on two worlds.
// All allocations are independent, mirroring the native binding's cold proof.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../../libero/libero/gpu/cuda/control/controller_pipeline.cuh"
#include "../../libero/libero/gpu/cuda/step/substep_types.cuh"
using namespace flash_libero;
struct Field {
  const char *name;
  size_t offset;
  int width;
};
#define FIELD(name, width)                                                                                   \
  Field { #name, offsetof(ControllerSolveData, name), width }
static const Field doubles[] = {FIELD(mass, 49),
                                FIELD(jacobian, 42),
                                FIELD(q, 7),
                                FIELD(qvel, 7),
                                FIELD(bias, 7),
                                FIELD(q0, 7),
                                FIELD(ee_pos, 3),
                                FIELD(ee_rot, 9),
                                FIELD(ee_velocity, 6),
                                FIELD(action, 7),
                                FIELD(goal_pos, 3),
                                FIELD(goal_rot, 9),
                                FIELD(gripper, 2),
                                FIELD(gains, 12),
                                FIELD(range, 18),
                                FIELD(ctrl, 9),
                                FIELD(raw_tau, 7),
                                FIELD(mass_inv, 49),
                                FIELD(A_full, 36),
                                FIELD(A_pos, 9),
                                FIELD(A_ori, 9),
                                FIELD(lambda_full, 36),
                                FIELD(lambda_pos, 9),
                                FIELD(lambda_ori, 9),
                                FIELD(nullspace_matrix, 49),
                                FIELD(position_error, 3),
                                FIELD(orientation_error, 3),
                                FIELD(quat_input, 4),
                                FIELD(quat_result, 9),
                                FIELD(nullspace_torque, 7),
                                FIELD(clipped, 7),
                                FIELD(derived_key, 91),
                                FIELD(derived_data, 206),
                                FIELD(pm, 49),
                                FIELD(pj, 42),
                                FIELD(pq, 7),
                                FIELD(pv, 7),
                                FIELD(pp, 3),
                                FIELD(pr, 9),
                                FIELD(pe, 6)};
static const Field integers[] = {FIELD(active, 1),        FIELD(error, 1),         FIELD(ranks, 3),
                                 FIELD(policy, 1),        FIELD(needs, 1),         FIELD(quat_valid, 1),
                                 FIELD(derived_valid, 1), FIELD(derived_ranks, 3), FIELD(cache_hits, 1),
                                 FIELD(cache_valid, 1)};

#undef FIELD

static void check(cudaError_t code) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(code));
    std::exit(1);
  }
}
template<class T> static T *&view(ControllerSolveData &s, size_t offset) {
  return *reinterpret_cast<T **>(reinterpret_cast<char *>(&s) + offset);
}
struct Buffer { std::string name; void *pointer; int count, bytes; };
struct Fixture {
  SubstepArguments a{};
  std::vector<Buffer> buffers;
  template<class T> T *make(const std::string &name, int n) {
    T *p;
    check(cudaMallocManaged(&p, sizeof(T)*n));
    std::fill(p, p+n, T{});
    buffers.push_back({name, p, n, int(sizeof(T))});
    return p;
  }
  ~Fixture() { for (const auto &b : buffers) check(cudaFree(b.pointer)); }
  Fixture(int nv) {
    auto &s=a.controller_solve;
    for (const auto &f : doubles) view<double>(s,f.offset)=make<double>(f.name,2*f.width);
    for (const auto &f : integers) view<int>(s,f.offset)=make<int>(f.name,2*f.width);
    s.quat_cast=make<float>("quat_cast",8);
    s.uncouple=1;
    auto &j=a.controller_inputs;
    j.nv=nv; j.nq=nv+7; j.nm=nv*nv; j.nb=2; j.root=0; j.body=1;
    j.cdof=make<double>("cdof",2*nv*6);
    j.subtree=make<double>("subtree",12);
    j.qpos=make<double>("qpos",2*j.nq);
    j.qvel=make<double>("qvel",2*nv);
    j.qm=make<double>("qm",2*j.nm);
    j.qbias=make<double>("qbias",2*nv);
    auto *qi=make<int>("qi",7), *vi=make<int>("vi",7);
    for (int k=0;k<7;++k) qi[k]=vi[k]=k;
    j.qi=qi; j.vi=vi;
    auto *sparse=make<int>("sparse",nv*nv);
    for(int k=0;k<nv*nv;++k) sparse[k]=k;
    j.sparse=sparse;
    auto *ancestor=make<int>("ancestor",2*nv);
    for(int k=0;k<7;++k) ancestor[nv+k]=1;
    j.ancestor=ancestor;
    auto *offsets=make<int>("offsets",3), *indices=make<int>("indices",7);
    offsets[0]=offsets[1]=0; offsets[2]=7;
    for(int k=0;k<7;++k) indices[k]=k;
    j.body_dof_offsets=offsets; j.body_dof_indices=indices;
    j.jp=make<double>("jp",6*nv); j.jr=make<double>("jr",6*nv);
    j.jac_status=make<int>("jac_status",2); j.adapter_active=make<int>("adapter_active",2);
    j.adapter_status=make<int>("adapter_status",2); j.out_status=make<int>("out_status",2);
    j.ep=s.pp; j.er=s.pr; j.vp=make<double>("vp",6); j.vo=make<double>("vo",6);
    j.q=s.pq; j.qd=s.pv; j.J=s.pj; j.M=s.pm; j.bias=s.bias;
    auto &g=a.gather_gate;
    g.mask=make<int>("gather_mask",2); g.error=make<int>("dynamics_error",2);
    g.state=make<int64_t>("state",2); g.source=make<int64_t>("source",2);
    g.epoch=make<int64_t>("epoch",2); g.written=make<int64_t>("bias_written",2);
    g.valid=make<int>("bias_valid",2); g.active=make<int>("gather_active",2); j.active=g.active;
    auto &pose=a.end_effector_pose;
    pose.ns=2; pose.id=1; pose.pos=make<double>("site_pos",12); pose.mat=make<double>("site_mat",36);
    pose.outp=make<double>("pose_outp",6); pose.outm=make<double>("pose_outm",18); pose.active=g.active;
    j.pos=pose.outp; j.rot=pose.outm;
    auto &v=a.end_effector_velocity;
    v.active=g.active; v.status=j.out_status; v.vp=j.vp; v.vo=j.vo; v.target=s.pe; v.first=g.error;
    auto &b=a.controller_bias;
    b.mask=make<int>("bias_mask",2); b.first=g.error; b.state=g.state; b.source=g.source;
    b.epoch=g.epoch; b.valid=g.valid; b.written=g.written; b.bias=j.qbias; b.selected=s.bias;
    b.indices=vi; b.parent_valid=make<int>("parent_valid",2); b.parent_written=make<int64_t>("parent_written",2); b.nv=nv;
    auto &c=a.controller_gate;
    c.active=s.active; c.pv=b.parent_valid; c.written=b.parent_written;
    c.expected=make<int64_t>("expected",2); c.first=make<int>("first_error",2); c.error=s.error; c.policy=s.policy;
    auto &plan=a.controller_plan;
    plan.active=s.active; plan.first=c.first; plan.needs=s.needs;
    plan.refresh=make<int>("refresh",2); plan.call=make<int>("call",2);
    auto &m=a.controller_commit;
    m.active=s.active; m.error=s.error; m.first=c.first; m.commit=make<int>("commit",2);
    m.event=make<int64_t>("event",2); m.valid=make<int>("valid",2);
    auto &p=a.controller_publish;
    p.mask=m.commit; p.event=m.event; p.valid=m.valid; p.parent=b.parent_written;
    p.written=make<int64_t>("written",2); p.consumed=make<int64_t>("consumed",2);
    // Public views use the formal seed/overwrite contract.
    for(const auto &f:doubles) {
      const std::string name=f.name;
      if(name=="bias" || name=="q0" || name=="action" || name=="gains" || name=="range" ||
         name=="derived_key" || name=="derived_data" || name=="pm" || name=="pj" || name=="pq" ||
         name=="pv" || name=="pp" || name=="pr" || name=="pe") continue;
      const bool overwritten=name=="ctrl" || name=="raw_tau" || name=="clipped" || name=="mass_inv" ||
        name=="A_full" || name=="A_pos" || name=="A_ori" || name=="lambda_full" || name=="lambda_pos" ||
        name=="lambda_ori" || name=="position_error" || name=="orientation_error" || name=="nullspace_torque";
      auto &channels=a.controller_channels;
      auto &field=channels.floating[channels.floating_count++];
      field={make<double>("public_"+name,2*f.width),view<double>(s,f.offset),f.width,1|(overwritten?0:2)};
    }
    auto &channels=a.controller_channels;
    if(channels.floating_count!=26) std::abort();
    channels.integer_count=2;
    channels.integer[0]={make<int>("public_cache_valid",2),s.cache_valid};
    channels.integer[1]={make<int>("public_needs",2),s.needs};
    for(int w=0;w<2;++w) {
      plan.call[w]=plan.refresh[w]=1;
      const_cast<int*>(g.valid)[w]=1;
      const_cast<int64_t*>(g.state)[w]=const_cast<int64_t*>(g.source)[w]=3;
      const_cast<int64_t*>(g.epoch)[w]=const_cast<int64_t*>(g.written)[w]=const_cast<int64_t*>(c.expected)[w]=7;
      for(int k=0;k<nv;++k) {
        const_cast<double*>(j.qm)[w*j.nm+k*nv+k]=2.+.1*k;
        const_cast<double*>(j.qvel)[w*nv+k]=.001*(k+1);
        const_cast<double*>(j.qbias)[w*nv+k]=.1*(k+1);
      }
      // Nonzero full-rank 6x7 Jacobian with nontrivial site offset.
      for(int k=0;k<6;++k) const_cast<double*>(j.cdof)[(w*nv+k)*6+k]=1.;
      for(int k=0;k<3;++k) const_cast<double*>(pose.pos)[(w*2+1)*3+k]=.03*(k+1);
      for(int k=0;k<9;++k) const_cast<double*>(pose.mat)[(w*2+1)*9+k]=k%4==0;
      for(int k=0;k<6;++k) { s.gains[w*12+k]=150.; s.gains[w*12+6+k]=24.; }
      for(int k=0;k<9;++k) { s.range[w*18+2*k]=-87.; s.range[w*18+2*k+1]=87.; }
    }
    for(int z=0;z<channels.floating_count;++z) {
      auto &field=channels.floating[z];
      if(field.working==s.mass) for(int w=0;w<2;++w) for(int k=0;k<7;++k) field.published[w*49+k*7+k]=2.+.1*k;
      if(field.working==s.ee_rot || field.working==s.goal_rot)
        for(int w=0;w<2;++w) for(int k=0;k<9;++k) field.published[w*9+k]=k%4==0;
    }
    channels.integer[0].published[0]=channels.integer[0].published[1]=1;
  }
  void fast() {
    auto &c=a.controller_channels;
    std::vector<ControllerCopyPacket> packets;
    for(bool seed:{true,false}) {
      for(int z=0;z<c.floating_count;++z) {
        const auto &f=c.floating[z];
        if(seed && !(f.parallel_copy&2)) continue;
        for(int k=0;k<f.width;k+=2) packets.push_back({f.published+k,f.working+k,f.width,std::min(2,f.width-k)});
      }
      if(seed) c.seed_packets=int(packets.size());
    }
    c.commit_packets=int(packets.size())-c.seed_packets;
    auto *p=make<ControllerCopyPacket>("packets",int(packets.size()));
    std::copy(packets.begin(),packets.end(),p); c.copy_packets=p;
  }
  void scenario(int id) {
    auto &g=a.gather_gate; auto &s=a.controller_solve; auto &m=a.controller_commit;
    if(id==1) a.controller_plan.refresh[1]=0;
    if(id==2) a.controller_plan.call[1]=0;
    if(id==3) const_cast<int64_t*>(g.source)[1]++;
    if(id==4) const_cast<int64_t*>(g.written)[1]--;
    if(id==5) const_cast<double*>(a.controller_inputs.cdof)[a.controller_inputs.nv*6]=NAN;
    if(id==6) a.controller_gate.first[1]=77;
    if(id==7) a.controller_gate.expected[1]++;
    if(id==8) { a.controller_plan.refresh[1]=0; a.controller_channels.integer[0].published[1]=0; }
    if(id==9) m.event[1]=INT64_MAX;
    if(id==10) m.event[1]=INT64_MAX-1;
    if(id==11) for(int k=0;k<a.controller_inputs.nm;++k) const_cast<double*>(a.controller_inputs.qm)[a.controller_inputs.nm+k]=0.;
    if(id==12) g.error[1]=55;
    if(id==13) a.controller_plan.refresh[1]=0, a.controller_plan.call[1]=0;
    (void)s;
  }
};
__global__ void exercise(const SubstepArguments *a, bool policy) {
  extern __shared__ __align__(32) unsigned char shared[];
  physics_controller::run_controller(*a,blockIdx.y,policy,shared);
  __syncthreads(); // public entry's required metadata/scratch handoff
}
static void launch(Fixture &f, bool policy) {
  SubstepArguments *device;
  check(cudaMallocManaged(&device,sizeof(*device))); *device=f.a;
  exercise<<<dim3(1,2),substep_threads,OscLayout::bytes>>>(device,policy);
  check(cudaGetLastError()); check(cudaDeviceSynchronize()); check(cudaFree(device));
}
int main() {
  double max_error=0.; int cases=0;
  for(int nv:{21,39,57}) for(int scenario=0;scenario<14;++scenario) {
    Fixture ordered(nv),parallel(nv);
    ordered.scenario(scenario); parallel.scenario(scenario); parallel.fast();
    for(int pass=0;pass<2;++pass) {
      launch(ordered,pass==0); launch(parallel,pass==0);
      for(size_t n=0;n<ordered.buffers.size();++n) {
        const auto &a=ordered.buffers[n], &b=parallel.buffers[n];
        for(int k=0;k<a.count;++k) {
          if(a.bytes==8) {
            // int64 counters compare bitwise, doubles allow FP accumulation.
            const bool counter=a.name=="state" || a.name=="source" || a.name=="epoch" || a.name=="bias_written" ||
              a.name=="parent_written" || a.name=="expected" || a.name=="event" || a.name=="written" || a.name=="consumed";
            if(counter) { if(static_cast<int64_t*>(a.pointer)[k]==static_cast<int64_t*>(b.pointer)[k]) continue; }
            else {
              const double x=static_cast<double*>(a.pointer)[k],y=static_cast<double*>(b.pointer)[k];
              if((std::isnan(x)&&std::isnan(y)) || x==y) continue;
              const double error=std::abs(x-y)/(1.+std::abs(x)); max_error=std::fmax(max_error,error);
              if(std::isfinite(error) && error<=1e-10) continue;
            }
          } else if(std::equal(static_cast<unsigned char*>(a.pointer)+k*a.bytes,
                               static_cast<unsigned char*>(a.pointer)+(k+1)*a.bytes,
                               static_cast<unsigned char*>(b.pointer)+k*b.bytes)) continue;
          std::fprintf(stderr,"NV%d scenario%d pass%d %s[%d] differs\n",nv,scenario,pass,a.name.c_str(),k); return 1;
        }
      }
      ++cases;
    }
  }
  std::printf("PASS: %d two-world ordered/cooperative calls; max scaled error %.12g\n",cases,max_error);
}
