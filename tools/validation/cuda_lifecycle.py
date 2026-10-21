"""Independent FP32 public lifecycle and observation contracts, without Warp."""
import json
from pathlib import Path
import sys
import numpy as np
from .cuda_complete import save, signatures


def add_arguments(parser):
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cameras', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='CPU scene preparation only, before load-only validation')


def main(args):
    from .cuda_runtime import ForbidWarp
    sys.meta_path.insert(0, ForbidWarp())
    from libero.gpu import precision
    precision.select('fp32')
    from libero.libero.envs import GpuBatchEnv
    from libero.libero import get_libero_path
    from libero.gpu.host.backends.cuda_fp32.scene import build_scene
    from libero.gpu.host.build.compilation import compilation_scope
    task = json.loads((args.build_root/'TASK_INSTANCES.json').read_text())['tasks'][0]
    bddl = Path(get_libero_path('bddl_files'))/task['key']
    options = dict(use_camera_obs=args.cameras, ignore_done=True, has_renderer=False,
                   has_offscreen_renderer=args.cameras, hard_reset=False)
    if args.cameras:
        options.update(camera_heights=32, camera_widths=32, camera_depths=True)
    if args.prepare:
        built, *_ = build_scene(args.build_root, bddl, task['seed'], options, prepare=True)
        del built
    checks = []
    def check(name, value):
        checks.append(dict(name=name, passed=bool(value)))
        if not value: raise AssertionError(name)
    env = None
    with compilation_scope(False) as scope:
        try:
            env = GpuBatchEnv(str(bddl), 2, precision='fp32', physics_engine='auto',
                build_root=args.build_root, allow_compile=False, initialize=False, **options)
            bank = env.initial_states[:2]
            check('FP32 auto selects independent CUDA', env.physics_engine=='cuda_fp32')
            obs = env.reset(init_states=bank)
            check('observations are finite', all(np.isfinite(v).all() for v in obs.values()))
            named=env.observer.read_scene_fields(env.observation_fields(),stream=env.stream)
            for name,value in named.items():
                reference=getattr(env.data,name).numpy()[0].reshape(value.shape)
                error=float(np.max(abs(value-reference),initial=0))
                bound=1024*np.finfo(np.float32).eps*max(1.,float(np.max(abs(reference),initial=0)))
                check('lazy named scene field '+name, np.isfinite(value).all() and error<=bound)
            env.step(np.zeros((2,7)))
            observed = env.observer.lowdim.observed.numpy().reshape(2,-1).copy()
            before = {k:getattr(env.data,k).numpy().copy() for k in
                ('qpos','qvel','qacc','qacc_warmstart','time','xpos','geom_xpos','qfrc_constraint','sensordata')}
            env.reset(id=[0], init_states=bank[:1])
            for k,v in before.items():
                check('partial reset holds world 1 '+k, np.array_equal(getattr(env.data,k).numpy()[1], v[1]))
            check('partial reset holds world 1 observation', np.array_equal(env.observer.lowdim.observed.numpy().reshape(2,-1)[1],observed[1]))
            packet = env.observer.snapshot_continuation(env.stream)
            with env.stream: env.observer.lowdim.cache.zero_()
            env.observer.restore_continuation(packet, env.stream)
            from libero.gpu.host.adapters.observations.continuation import arrays
            check('observer continuation exact transport', all(np.array_equal(a.numpy(),packet['arrays'][k]) for k,a in arrays(env.observer).items()))
            obs, *_ = env.step(np.zeros((2,7)))
            if args.cameras:
                images = {k:v for k,v in obs.items() if k.endswith(('_image','_depth'))}
                check('RGB and depth public outputs', bool(images) and all(v.shape[:3]==(2,32,32) and np.isfinite(v).all() for v in images.values()))
                check('RGB is not blank', any(np.ptp(v)>0 for k,v in images.items() if k.endswith('_image')))
                rgb,depth=env.shells[0].sim.render(width=32,height=32,camera_name='agentview',depth=True)
                check('public facade RGB/depth rendering', rgb.shape==(32,32,3) and depth.shape==(32,32)
                      and np.isfinite(depth).all() and np.ptp(rgb)>0)
            name = next(s.name for s in env.observer.plan.specs if s.active)
            env.modify_observable(name,'sampling_rate',10)
            env.step(np.zeros((2,7)))
            check('modified observer executes', env.observer.valid)
            env.modify_observable(name,'filter',lambda value: value)
            env.step(np.zeros((2,7)))
            check('custom callback uses device substep graphs', env.observer.requires_host_sampling and bool(env.runtime._step_graphs))
            check('Warp was never imported', not any(k.split('.')[0] in ('warp','mujoco_warp') for k in sys.modules))
            check('no runtime compilation', not scope['events'])
        finally:
            if env is not None: env.close();env.close()
    save(args.output, dict(passed=all(c['passed'] for c in checks), checks=checks,
        cameras=args.cameras, candidate_sources=signatures(True), runtime_compilations=scope['events'],
        warp_imported=False, worlds=2))
    print(json.dumps(dict(passed=True, checks=len(checks), cameras=args.cameras)), flush=True)
    return 0
