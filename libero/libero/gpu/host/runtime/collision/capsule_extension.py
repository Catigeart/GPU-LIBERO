"""Source-pinned opt-in native capsule extension; fixed C sources untouched.

Generated sources require their own numerical/compile acceptance. Candidate
implementation status must not be interpreted as inheriting old fixed-domain
qualification for a newly compiled CUDA module.
"""
from libero.gpu.host.paths import GPU_ROOT
from pathlib import Path
import hashlib
import json
import runpy
import re

ROUTES = ((3, 5), (3, 6), (3, 7))
_PREPARED_NATIVE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def _native_header_closure(native_root, gpu_root):
    """Capture literal quoted dependencies without escaping a copied module.

    Relative headers under the native source root are copied byte for byte.
    An absolute GPU header remains bound to that actual source location.
    External relative includes in copied files require a future bounded mirror
    layout and are refused here instead of silently omitting dependencies.
    """
    native_root, gpu_root = Path(native_root).resolve(), Path(gpu_root).resolve()
    pending = [native_root / name for name in ('native_kernel.cu', 'native_source.cuh')]
    records, copied = {}, {}
    while pending:
        source = pending.pop().resolve()
        if source in records:
            continue
        if not source.is_relative_to(gpu_root):
            raise ValueError('native quoted source is outside actual GPU root: ' + str(source))
        if not source.is_file():
            raise FileNotFoundError('native quoted source is absent: ' + str(source))
        raw = source.read_bytes()
        records[source] = raw
        copied_source = source.is_relative_to(native_root)
        if copied_source:
            copied[source.relative_to(native_root).as_posix()] = raw
        for spelling in re.findall(r'^\s*#\s*include\s*"([^"\r\n]+)"', raw.decode('utf-8'), re.M):
            reference = Path(spelling)
            header = (source.parent / reference).resolve()
            if copied_source and not reference.is_absolute() and not header.is_relative_to(native_root):
                raise ValueError('native external relative quoted include requires a bounded candidate mirror: '
                                 + str(source) + ' -> ' + spelling + ' -> ' + str(header))
            if not header.is_relative_to(gpu_root):
                raise ValueError('native quoted include escapes actual GPU root: '
                                 + str(source) + ' -> ' + str(header))
            pending.append(header)
    return copied, records


def _source_text(raw):
    """Match Path.read_text universal newlines for transformed primary files."""
    return raw.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')


def generate_sources(destination):
    original = (GPU_ROOT / 'host/runtime/collision/native_originals/engine_collision_box.c.txt')
    native_root = GPU_ROOT / 'cuda/collision/native'
    copied, readset = _native_header_closure(native_root, GPU_ROOT)
    for path in (original, GPU_ROOT / 'host/runtime/collision/native.py'):
        readset[path.resolve()] = path.read_bytes()
    text = _source_text(readset[original.resolve()])
    start = text.index('static void mju_clampVec(')
    end = text.index('\nint mjc_BoxBox(')
    extension = text[start:end]
    # Remove the sphere entry point (helper retained); it is not called by the
    # extension and would add an unnecessary symbol and second macro expansion.
    sphere_start = extension.index('int mjc_SphereBox(')
    theory_start = extension.index('// GENERAL THEORY OF OPERATION')
    extension = extension[:sphere_start] + extension[theory_start:]
    extension = extension.replace('static void mju_clampVec(', '__device__ static void mju_clampVec(')
    extension = extension.replace('static int _SphereBox(', '__device__ static int _SphereBox(')
    extension = extension.replace('int mjc_CapsuleBox(', '__device__ int mjc_CapsuleBox(')
    getter = '''const mjtNum* pos1 = d->geom_xpos + 3 * g1;
  const mjtNum* mat1 = d->geom_xmat + 9 * g1;
  const mjtNum* size1 = m->geom_size + 3 * g1;
  const mjtNum* pos2 = d->geom_xpos + 3 * g2;
  const mjtNum* mat2 = d->geom_xmat + 9 * g2;
  const mjtNum* size2 = m->geom_size + 3 * g2;'''
    if extension.count('mjGETINFO') != 1:
        raise ValueError('unexpected original capsule-box getter source')
    extension = extension.replace('mjGETINFO', getter)
    source = _source_text(copied['native_source.cuh'])
    source += '\n// Opt-in capsule-box source: Apache-2.0 original engine_collision_box.c.\n' + extension
    kernel = _source_text(copied['native_kernel.cu'])
    old = 'if(ta<5 || tb>7 || ta>tb)return;'
    new = 'if(!((ta>=5 && tb<=7 && ta<=tb) || (ta==3 && tb>=5 && tb<=7)))return;'
    if kernel.count(old) != 1:
        raise ValueError('unexpected accepted native route gate')
    kernel = kernel.replace(old, new)
    old_call = '''int n=ta==6&&tb==6?mjc_BoxBox(&m,&d,con,0,1,margin[slot]):
                             mjc_Convex(&m,&d,con,0,1,margin[slot]);'''
    new_call = '''int n;
 if(ta==3 && tb==6){
   n=mjc_CapsuleBox(&m,&d,con,0,1,margin[slot]);
   // CapsuleBox only writes frame when the SphereBox helper appends contact.
   fw=n>0;
 }else{
   n=ta==6&&tb==6?mjc_BoxBox(&m,&d,con,0,1,margin[slot]):
                       mjc_Convex(&m,&d,con,0,1,margin[slot]);
 }'''
    if kernel.count(old_call) != 1:
        raise ValueError('unexpected accepted native dispatch source')
    kernel = kernel.replace(old_call, new_call)
    identity = {
        'schema': 'capsule-native-candidate-source-v2',
        'candidate_routes': [list(route) for route in ROUTES],
        'generator_file': str(Path(__file__).resolve()), 'generator_sha256': sha(__file__),
        'accepted_inputs': {str(path): hashlib.sha256(raw).hexdigest()
                            for path, raw in sorted(readset.items())},
        'quoted_dependency_policy': 'exact native bytes; absolute external GPU sources bound in place; external relative includes refused',
        'transformations': ['original getter expanded to private SoA adapter offsets',
            'device qualifiers on capsule-box and sphere-box helper',
            'route gate adds only capsule-cylinder, capsule-box, capsule-mesh',
            'capsule-box calls pinned original; other new routes call pinned convex',
            'capsule-box contact frame-write metadata tracks nonempty appended output'],
        'qualification': 'new candidate only; independent numerical/compile/held verification required',
    }
    generated = dict(copied)
    generated['native_source.cuh'] = source.encode('utf-8')
    generated['native_kernel.cu'] = kernel.encode('utf-8')
    identity['generated_files'] = {name: hashlib.sha256(raw).hexdigest()
                                   for name, raw in sorted(generated.items())}
    # Source bytes, every copied header, actual absolute dependencies and the
    # generator all participate in the directory identity. Old products remain.
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    destination = Path(destination).resolve()
    target = (destination / ('capsule-sources-' + key[:24])).resolve()
    if target == destination or not target.is_relative_to(destination):
        raise ValueError('native source candidate root escapes destination: ' + str(target))
    target.mkdir(parents=True, exist_ok=True)
    for name, raw in sorted(generated.items()):
        path = target / name
        if not path.resolve().is_relative_to(target):
            raise ValueError('native generated header destination escapes candidate: ' + str(path))
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('source candidate directory has changed content: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as file:
                file.write(raw)
    for path, raw in readset.items():
        if path.read_bytes() != raw:
            raise ValueError('native source changed during cold candidate generation: ' + str(path))
    identity['candidate_source_sha256'] = key
    manifest = target / 'SOURCE_CANDIDATE.json'
    if not manifest.resolve().is_relative_to(target):
        raise ValueError('native source candidate manifest escapes candidate: ' + str(manifest))
    manifest_raw = (json.dumps(identity, indent=2) + '\n').encode('utf-8')
    if manifest.exists() and manifest.read_bytes() != manifest_raw:
        raise ValueError('source candidate manifest has changed content: ' + str(manifest))
    if not manifest.exists():
        with manifest.open('xb') as file:
            file.write(manifest_raw)
    return target, identity


def prepare_workspace_extension(workspace, destination):
    """Cold boundary only. Install actual new module before retained binding."""
    if workspace._device_input_binding is not None or workspace._launch_started:
        raise ValueError('capsule extension must precede input binding and launches')
    target, identity = generate_sources(destination)
    # Isolated globals keep the fixed native module/cache untouched. Its normal
    # build program hashes both generated files and actual toolchain/artifacts.
    import os
    key = (str(target.resolve()), str(workspace.device), os.environ.get('P06C_NATIVE_BUILD_ROOT'))
    from libero.gpu.host.runtime.core.product_paths import current_build_lifetime
    owner = current_build_lifetime()
    if owner is not None:
        # The source/compiler bank is shared, but a mutable native instance
        # belongs to one runtime generation, including its CDLL borrow.
        key += (str(owner.path),)
    if key not in _PREPARED_NATIVE:
        program = runpy.run_path(str((GPU_ROOT / 'host/runtime/collision/native.py')))
        prepare = program['prepare_native']
        prepare.__globals__['MODULE'] = target
        _PREPARED_NATIVE[key] = prepare(workspace.device)
        if owner is not None:
            prepared = _PREPARED_NATIVE[key]
            def retire():
                if _PREPARED_NATIVE.get(key) is prepared:
                    _PREPARED_NATIVE.pop(key)
            owner.retire_callbacks.append(retire)
    native = _PREPARED_NATIVE[key]
    import numpy as np
    from libero.gpu.host.runtime.collision.detectors import DET_CONVEX_MPR, DET_CAPSULE_BOX_NATIVE
    # Existing kernel consumes all native results via the same raw/held mask.
    # Copy the host-only registry, not runtime contacts/counts.
    detector = workspace.detect
    from libero.gpu.host.runtime.collision.detectors import build_route_ids
    route_id = build_route_ids().ravel()
    for a, b in ROUTES:
        route_id[a * 8 + b] = DET_CAPSULE_BOX_NATIVE if (a, b) == (3, 6) else DET_CONVEX_MPR
    detector.native = native
    detector.route_id.assign(np.ascontiguousarray(route_id, dtype=np.int32))
    binding = {'schema': 'capsule-native-candidate-binding-v1',
        'source_candidate': identity, 'actual_native_execution': native['execution_binding'],
        'consumer_guard_file': str((GPU_ROOT / 'host/runtime/collision/detectors.py')),
        'consumer_guard_sha256': sha((GPU_ROOT / 'host/runtime/collision/detectors.py')),
        'actual_build_path': native['build_path'], 'actual_build_sha256': native['build_sha256'],
        'route_assignments': {f'{a},{b}': (DET_CAPSULE_BOX_NATIVE if (a, b) == (3, 6) else DET_CONVEX_MPR) for a, b in ROUTES},
        'qualification': 'new opt-in candidate pending independent validation'}


    workspace._capsule_candidate_binding = binding
    return binding





