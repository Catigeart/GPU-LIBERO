"""CPU MuJoCo scene preparation for independent FP32 CUDA owners.

Scene XML/fixture semantics match the pinned migration path. Cached MJBs are
CPU MuJoCo products, not serialized Warp arrays. Both cold preparation and
load-only reconstruction can run with Warp imports prohibited.
"""
import hashlib,json
from contextlib import contextmanager
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from libero.gpu.host.build.project_paths import project_paths

UPSTREAM_COMMIT='c384ac51df67eb92ab7657cc472ff7494dc036bf'


@contextmanager
def _scene_files(root, xml, options, task, scene_parameters, *, prepare):
    """Runtime reads fixed inputs; only offline preparation takes a writer lock."""
    from ..mjwarp.scene_cache import locate
    kwargs = dict(prepare=prepare, scene_parameters=scene_parameters)
    if not prepare:
        yield locate(root, xml, options, task, **kwargs)
        return
    from libero.gpu.host.runtime.core.product_paths import build_unit_lock
    folder = locate(root, xml, options, task, **kwargs)[0]
    with build_unit_lock(folder):
        # A preceding writer may have completed while this process waited.
        yield locate(root, xml, options, task, **kwargs)


def scene_record(root, bddl, seed):
    catalogue = json.loads((project_paths(root).base / 'TASK_INSTANCES.json').read_text())
    path = Path(bddl)
    matches = [t for t in catalogue['tasks'] if t['key'] == path.parent.name + '/' + path.name]
    if len(matches) != 1 or matches[0]['seed'] != seed:
        raise ValueError('MJWarp requires a matching offline task scene and seed')
    return matches[0]

def relocate_assets(xml):
    """Relocate existing offline asset references; preserve numerical XML fields."""
    import robosuite
    from libero.gpu.host.paths import REPO_ROOT
    tree = ET.fromstring(xml)
    for node in tree.iter():
        name = node.get('file')
        if not name or Path(name).is_file():
            continue
        if '/site-packages/' in name:
            replacement = Path(robosuite.__file__).resolve().parent.parent / name.split('/site-packages/', 1)[1]
        elif '/Flash-LIBERO/' in name:
            replacement = REPO_ROOT / name.split('/Flash-LIBERO/', 1)[1]
        else:
            raise FileNotFoundError(name)
        if not replacement.is_file():
            raise FileNotFoundError(replacement)
        node.set('file', str(replacement))
    return ET.tostring(tree, encoding='unicode')

def capacity_budget(host, *, nconmax=None, njmax=None):
    """Size rows from actual contact dimensionality and joint/equality limits.

    nconmax is a nominal per-world share of MJWarp's pooled contacts, not a
    proof that every world's distribution fits. Runtime overflow stays checked.
    """
    def power2(value):
        return 1 << (int(value) - 1).bit_length()
    contacts = int(nconmax) if nconmax is not None else max(256, power2(4 * host.nv))
    dim = max(1, int(np.max(host.geom_condim, initial=1)), int(np.max(host.pair_dim, initial=1)))
    rows_per_contact = dim if host.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else max(1, 2 * (dim - 1))
    rows = (int(njmax) if njmax is not None else
            max(512, power2(rows_per_contact * contacts + 2 * host.nv + 6 * host.neq + 2 * host.ntendon)))
    if contacts < 1 or rows < 1:
        raise ValueError('positive MJWarp contact/constraint capacities required')
    return dict(nconmax=contacts, njmax=rows)

def build_scene(root, bddl, seed, options, *, collision='default',prepare=False):
    import mujoco
    if mujoco.__version__ != "3.16.0":raise ValueError("FP32 requires MuJoCo3.16.0 model semantics")
    from libero.gpu.model import build_task_model
    from libero.gpu import precision
    task = scene_record(root, bddl, seed)
    directory = Path(task['directory'])
    for name in ('scene.xml', 'initial_states.npy'):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != task['files'][name]:
            raise ValueError('offline scene input changed: ' + name)
    xml = relocate_assets((directory / 'scene.xml').read_text())
    tree = ET.fromstring(xml)
    opt = tree.find('option')
    if opt is None:
        opt = ET.SubElement(tree, 'option')
    opt.set('tolerance', format(precision.solver_tolerance(task['options']['tolerance']), '.17g'))
    opt.set('solver', 'Newton')
    opt.set('jacobian', 'dense')
    opt.set('iterations', str(task['options']['iterations']))
    flags = opt.find('flag')
    if flags is None:
        flags = ET.SubElement(opt, 'flag')
    # Actual CPU 2.3.7 mjENBL_MULTICCD is bit 16, disabled in all 130 scenes.
    # MuJoCo 3.16 inverted its default and moved it to disableflags. Retain
    # the scene's semantics; do not delete nonzero geom margins to admit it.
    multiccd = bool(task['options']['enableflags'] & 16)
    flags.set('multiccd', 'enable' if multiccd else 'disable')
    if collision == 'primitive':
        flags.set('nativeccd', 'disable')
    elif collision != 'default':
        raise ValueError('collision must be default or primitive')
    xml = ET.tostring(tree, encoding='unicode')
    from ..mjwarp.scene_cache import finish
    from .scene_inputs import read as read_scene_inputs,apply as apply_scene_inputs
    scene_parameters,scene_binding=read_scene_inputs(project_paths(root).base,task)
    with _scene_files(project_paths(root).base, xml, options, task, scene_binding,
                      prepare=prepare) as scene:
        cache_folder,compiled,receipt,cache_binding = scene
        built=build_task_model(bddl,xml_string_override=xml,compiled_model_file=compiled,
            static_model_setup=lambda shell,native:apply_scene_inputs(shell,native,scene_parameters),**options)
        finish(cache_folder,built,receipt,cache_binding)
    host = built.native_model
    for name in ('nq', 'nv', 'nu', 'nbody', 'njnt', 'ngeom', 'nsite'):
        if int(getattr(host, name)) != task['dimensions'][name]:
            raise ValueError('MJWarp import changed model indexing: ' + name)
    # CPU 2.3.7 AUTO chooses dense for every shipped NV (21..57). New MuJoCo
    # AUTO uses different heuristics; keep the established dense baseline.
    bank = np.load(directory / 'initial_states.npy', allow_pickle=False)
    identity = dict(task=task['key'], scene_sha256=task['files']['scene.xml'],
                    init_sha256=task['files']['initial_states.npy'],
                    fixed_scene_inputs_sha256=scene_binding['data_sha256'],
                    imported_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                    imported_model_sha256=built.metadata['model_sha256'],
                    mujoco=mujoco.__version__, mjwarp_commit=UPSTREAM_COMMIT,
                    tolerance=float(host.opt.tolerance), jacobian='dense', solver='Newton', multiccd=multiccd)
    return built, bank, task, identity
