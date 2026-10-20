"""One offline build of the complete shipped LIBERO instance set."""
import libero.gpu.precision as scalar
import gc
import hashlib
import importlib.metadata
import json
from contextlib import nullcontext
from multiprocessing import get_context
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from libero.gpu.host.runtime.core.product_paths import cache_root
from .project_paths import project_paths


def project_root(root=None):
    return project_paths(root).precision


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# This is the one recorded workspace move, not a general path-insensitive key.
_NEWTON_LEGACY_SOURCE_ROOTS = ('/rjs2/lhw/Flash-LIBERO',)


def _newton_recipe(source, bridge, closure, archive_sha):
    encoded = json.dumps(closure, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256((source + bridge + encoded + archive_sha +
        '|fma=true|prec-div=true|prec-sqrt=true|real='+str(scalar.bits)).encode()).hexdigest()


def collect_newton_dependency_inputs(commands, *, architecture, mathdx_archive):
    """Capture actual compiler-resolved headers/toolchain at an offline miss.

    This invokes dependency discovery only under compilation authority. A new
    receipt must never be manufactured retrospectively for an old binary.
    """
    from .compilation import require_compilation
    require_compilation('Newton compiler dependency discovery', commands)
    from .build_receipts import compiler_dependencies, toolchain_record, file_record
    from .product_cache import compiler_environment_record
    commands = [list(map(str, command)) for command in commands]
    compiler = commands[0][0]
    units, dependencies, unit_flags = [], {}, []
    for command in commands:
        if command[0] != compiler:
            raise ValueError('Newton dependency receipt requires one compiler')
        sources = [Path(arg).resolve() for arg in command[1:] if arg.endswith('.cu')]
        if not sources:
            continue
        if len(sources) != 1:
            raise ValueError('Newton dependency receipt requires one source per command')
        flags = []
        index = 1
        while index < len(command):
            arg = command[index]
            index += 1
            if arg == '-o':
                index += 1
                continue
            if arg in ('-dc', '-dlto', '-shared', '-c', '--ptx') or arg.endswith(('.cu', '.o', '.a')):
                continue
            flags.append(arg)
        units.append(file_record(sources[0]))
        unit_flags.append(flags)
        for row in compiler_dependencies(compiler, flags, sources):
            dependencies[row['path']] = row
    if not units:
        raise ValueError('Newton dependency receipt has no source units')
    toolchain = dict(toolchain_record(compiler, unit_flags[0]))
    toolchain['environment_sha256'] = compiler_environment_record()['sha256']
    toolchain['environment_schema'] = 'compiler-input-environment-v1'
    return dict(schema='libero-newton-compiler-inputs-v1', discovery='compiler-resolved-per-unit',
        commands=commands, units=units,
        dependencies=[dependencies[p] for p in sorted(dependencies)], toolchain=toolchain,
        archive=file_record(mathdx_archive), architecture=architecture)


def _check_newton_dependency_inputs(inputs):
    """Recheck captured bytes before publication without invoking a compiler."""
    from .product_cache import compiler_environment_record
    if (inputs.get('schema') != 'libero-newton-compiler-inputs-v1' or
            inputs['toolchain'].get('environment_sha256') != compiler_environment_record()['sha256']):
        raise ValueError('Newton dependency receipt compiler environment differs')
    records = inputs['units'] + inputs['dependencies'] + inputs['toolchain']['components'] + [inputs['archive']]
    for row in records:
        path = Path(row['path'])
        if path.stat().st_size != row['bytes'] or _sha(path) != row['sha256']:
            raise ValueError('Newton dependency receipt input changed: ' + str(path))


def publish_newton_dependency_receipt(target, *, identity, inputs):
    """Bind a successful fresh binary to the inputs captured before its build."""
    _check_newton_dependency_inputs(inputs)
    target = Path(target)
    encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
    receipt = dict(schema='libero-newton-dependencies-v1', recipe=identity,
        binary_sha256=_sha(target), inputs=inputs, inputs_sha256=hashlib.sha256(encoded).hexdigest())
    _publish_alias_file(target.with_suffix('.dependencies.json'),
        (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
    return target.with_suffix('.dependencies.json')


def admit_newton_cached_product(target, *, identity, source_sha256, architecture, preserve_root):
    """Require completion evidence before an exact-key binary may be reused.

    A receipt describes the historical build inputs. Current input equality is
    required at fresh publication and path migration, not retroactively here.
    An interrupted build without metadata is never an implicit legacy hit.
    """
    target = Path(target)
    if not target.is_file():
        return None
    metadata = target.with_suffix('.json')
    dependency_path = target.with_suffix('.dependencies.json')
    if not metadata.is_file():
        from .compilation import require_compilation
        require_compilation('Newton completion metadata for incomplete CUDA product', target)
        # Preserve interrupted output outside the formal artifact scan. The
        # original exact recipe can then build afresh without overwriting it.
        parent = Path(preserve_root) / 'incomplete-newton-products'
        parent.mkdir(parents=True, exist_ok=True)
        isolated = Path(tempfile.mkdtemp(prefix=target.stem + '-', dir=parent))
        moved = []
        for path in (target, dependency_path):
            if path.exists():
                destination = isolated / path.name
                path.rename(destination)
                moved.append(dict(previous=str(path), preserved=str(destination), sha256=_sha(destination)))
        save(isolated/'INCOMPLETE.json', dict(schema='libero-incomplete-newton-product-v1',
            recipe=identity, completed=False, reason='binary exists without completion metadata', files=moved))
        return None
    previous = json.loads(metadata.read_bytes())
    binary_sha = _sha(target)
    if (previous.get('sha256') != identity or previous.get('source_sha256') != source_sha256 or
            previous.get('library_sha256') != binary_sha or
            '-arch=sm_' + str(architecture) not in previous.get('command', [])):
        raise ValueError('cached Newton source/binary/architecture differs')
    declared = previous.get('dependency_receipt_file')
    if declared is not None and (Path(declared) != dependency_path or not dependency_path.is_file()):
        raise ValueError('cached Newton dependency receipt is absent or has a different path')
    if dependency_path.is_file():
        receipt = json.loads(dependency_path.read_bytes())
        inputs = receipt.get('inputs', {})
        encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()
        if (receipt.get('schema') != 'libero-newton-dependencies-v1' or
                receipt.get('recipe') != identity or receipt.get('binary_sha256') != binary_sha or
                receipt.get('inputs_sha256') != hashlib.sha256(encoded).hexdigest() or
                inputs.get('schema') != 'libero-newton-compiler-inputs-v1' or
                inputs.get('discovery') != 'compiler-resolved-per-unit' or
                inputs.get('architecture') != architecture or
                inputs.get('commands') != previous.get('compile_commands')):
            raise ValueError('cached Newton dependency receipt does not bind the completed binary and recipe')
    return previous


def _publish_alias_file(path, raw):
    """Publish complete bytes without ever replacing an existing cache file."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Newton path alias must not replace a symlink: ' + str(path))
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Newton path alias conflicts with existing bytes: ' + str(path))
        return
    with tempfile.NamedTemporaryFile(prefix='.path-alias-', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise ValueError('Newton path alias publication conflict: ' + str(path))
    finally:
        temporary.unlink()


def _complete_aot_record(root, record):
    if record.get('status') == 'complete':
        return record
    if record.get('status') not in ('building', 'failed'):
        return None
    previous = record.get('previous_complete_build')
    if previous is None:
        return None
    digest = previous.get('sha256', '')
    spelling = 'aot-history/' + digest + '.json'
    if re.fullmatch(r'[0-9a-f]{64}', digest) is None or previous.get('path') != spelling:
        raise ValueError('previous complete AOT manifest anchor path differs')
    path = root / spelling
    if path.is_symlink() or not path.is_file() or _sha(path) != digest:
        raise ValueError('previous complete AOT manifest SHA differs')
    result = json.loads(path.read_bytes())
    if result.get('status') != 'complete':
        raise ValueError('previous AOT manifest is not complete')
    return result


def preserve_complete_build(root):
    """Retain an immutable SHA-anchored complete bank before offline overwrite."""
    root = Path(root).resolve()
    path = root / 'AOT_BUILD.json'
    if not path.is_file():
        return None
    raw = path.read_bytes()
    record = json.loads(raw)
    complete = _complete_aot_record(root, record)
    if complete is None:
        return None
    if record.get('status') != 'complete':
        return dict(record['previous_complete_build'])
    digest = hashlib.sha256(raw).hexdigest()
    destination = root / 'aot-history' / (digest + '.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    _publish_alias_file(destination, raw)
    return dict(path=str(destination.relative_to(root)), sha256=digest)


def migrate_newton_path_alias(root, *, target, identity, source, bridge_source,
        stage_sources, mathdx_archive, architecture, commands, launch_command,
        nv, nM, dependency_inputs=None, dry_run=False):
    """Reuse one exact current Newton recipe after the recorded directory move.

    The caller supplies its freshly generated complete source/bridge/closure and
    compiler argv. Only listed AOT products in the same NV/nM directory are
    candidates. A single explicit source-prefix substitution must reproduce both
    full recipe keys, with identical current dependency/archive and binary bytes.
    A receipt covers this product only: no AOT identity or task record is changed.
    A source/algorithm/flag change is an ordinary miss, never a migration.
    """
    root = Path(root).resolve()
    target = Path(target).resolve()
    if (target.suffix != '.so' or target.stem != identity or
            target.parent != root / 'cuda' / ('sm' + str(architecture)) / 'capacity' / ('nv%d_nm%d' % (nv, nM)) or
            re.fullmatch(r'[0-9a-f]{64}', identity) is None or
            re.findall(r'^#define NV (\d+)$', source, re.MULTILINE) != [str(nv)] or
            re.findall(r'^#define NM (\d+)$', source, re.MULTILINE) != [str(nM)]):
        raise ValueError('Newton path alias requires the exact current CUDA product path')
    if type(dry_run) is not bool:
        raise ValueError('dry_run must be bool')
    current_gpu = Path(GPU_ROOT).resolve()
    closure = dict(stage_sources)
    relative = {}
    for spelling, expected in closure.items():
        path = Path(spelling)
        if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(current_gpu):
            raise ValueError('Newton path alias closure must name current project files')
        if _sha(path) != expected:
            raise ValueError('Newton path alias current header differs: ' + str(path))
        relative[spelling] = path.relative_to(current_gpu)
    archive = Path(mathdx_archive).resolve()
    archive_sha = _sha(archive)
    if _newton_recipe(source, bridge_source, closure, archive_sha) != identity:
        raise ValueError('Newton path alias current complete recipe differs')
    manifest_path = root / 'AOT_BUILD.json'
    if not manifest_path.is_file():
        return None
    record = _complete_aot_record(root, json.loads(manifest_path.read_text()))
    if record is None:
        return None
    candidates = {}
    for key, task in record.get('tasks', {}).items():
        for spelling, expected in task.get('native_artifacts', {}).items():
            path = root / spelling
            if path.parent != target.parent or path == target or path.suffix != '.so':
                continue
            if record.get('artifacts', {}).get(spelling) != expected:
                raise ValueError('Newton path alias AOT artifact hashes conflict')
            item = candidates.setdefault(path, {'sha256': expected, 'tasks': []})
            if item['sha256'] != expected:
                raise ValueError('Newton path alias AOT binary hashes conflict')
            item['tasks'].append(key)
    for candidate, item in sorted(candidates.items()):
        metadata = candidate.with_suffix('.json')
        if not metadata.is_file():
            continue
        previous = json.loads(metadata.read_text())
        dependency_path = candidate.with_suffix('.dependencies.json')
        dependency_spelling = str(dependency_path.relative_to(root))
        # Original legacy artifacts lack this evidence. Current header bytes
        # cannot retroactively prove which SDK headers built an old binary.
        if (dependency_inputs is None or not dependency_path.is_file() or
                record.get('artifacts', {}).get(dependency_spelling) != _sha(dependency_path)):
            continue
        dependency_receipt = json.loads(dependency_path.read_text())
        source_path = candidate.with_suffix('.cu')
        bridge_path = candidate.with_name(candidate.stem + '_bridge.cu')
        if (previous.get('sha256') != candidate.stem or not previous.get('handwritten') or
                (previous.get('nv'), previous.get('nM')) != (nv, nM) or
                previous.get('source') != str(source_path) or
                previous.get('device_bridge') != str(bridge_path) or
                previous.get('library_file') != str(candidate) or
                previous.get('mathdx_library_sha256') != archive_sha or
                previous.get('floating_policy') != dict(fma=True, fast_math=False, prec_div=True, prec_sqrt=True) or
                not source_path.is_file() or not bridge_path.is_file() or
                source_path.read_bytes() != source.encode() or bridge_path.read_bytes() != bridge_source.encode() or
                previous.get('source_sha256') != hashlib.sha256(source.encode()).hexdigest()):
            continue
        for old_root in _NEWTON_LEGACY_SOURCE_ROOTS:
            old_gpu = Path(old_root) / 'libero/libero/gpu'
            if old_gpu == current_gpu:
                continue
            old_closure = {str(old_gpu / relative[p]): h for p, h in closure.items()}
            if previous.get('stage_sources') != old_closure:
                continue
            old_identity = _newton_recipe(source, bridge_source, old_closure, archive_sha)
            if old_identity != candidate.stem:
                continue
            # Digest-derived output names change mechanically with the recipe.
            # No flags, SDK/compiler paths, or arbitrary substrings are ignored.
            paths = {str(candidate): str(target), str(source_path): str(target.with_suffix('.cu')),
                str(bridge_path): str(target.with_name(identity + '_bridge.cu')),
                str(candidate.with_name(candidate.stem + '_main.o')): str(target.with_name(identity + '_main.o')),
                str(candidate.with_name(candidate.stem + '_bridge.o')): str(target.with_name(identity + '_bridge.o'))}
            def moved(argument):
                if argument in paths:
                    return paths[argument]
                for prefix in ('', '-I'):
                    spelling = argument[len(prefix):] if argument.startswith(prefix) else ''
                    if spelling.startswith(old_root + '/'):
                        return prefix + str(REPO_ROOT) + spelling[len(old_root):]
                return argument
            old_commands = previous.get('compile_commands', [])
            if (not old_commands or [[moved(a) for a in cmd] for cmd in old_commands] != commands or
                    [moved(a) for a in previous.get('command', [])] != launch_command or
                    any('-arch=sm_' + str(architecture) not in cmd for cmd in commands)):
                continue
            old_inputs = dependency_receipt.get('inputs', {})
            old_encoded = json.dumps(old_inputs, sort_keys=True, separators=(',', ':')).encode()
            def moved_inputs(value):
                if isinstance(value, dict):
                    return {k: moved_inputs(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [moved_inputs(v) for v in value]
                return moved(value) if isinstance(value, str) else value
            if (dependency_receipt.get('schema') != 'libero-newton-dependencies-v1' or
                    dependency_receipt.get('recipe') != old_identity or
                    dependency_receipt.get('inputs_sha256') != hashlib.sha256(old_encoded).hexdigest() or
                    old_inputs.get('schema') != 'libero-newton-compiler-inputs-v1' or
                    old_inputs.get('discovery') != 'compiler-resolved-per-unit' or
                    moved_inputs(old_inputs) != dependency_inputs or
                    dependency_inputs.get('commands') != commands or
                    dependency_inputs.get('architecture') != architecture):
                continue
            _check_newton_dependency_inputs(dependency_inputs)
            raw = candidate.read_bytes()
            binary_sha = hashlib.sha256(raw).hexdigest()
            if (binary_sha != item['sha256'] or previous.get('library_sha256') != binary_sha or
                    dependency_receipt.get('binary_sha256') != binary_sha):
                raise ValueError('Newton path alias original binary differs: ' + str(candidate))
            receipt = dict(schema='libero-newton-path-alias-v1', previous=str(candidate), current=str(target),
                previous_recipe=old_identity, current_recipe=identity, binary_sha256=binary_sha,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                bridge_sha256=hashlib.sha256(bridge_source.encode()).hexdigest(),
                previous_source_root=old_root, current_source_root=str(REPO_ROOT),
                previous_stage_sources=old_closure, current_stage_sources=closure,
                mathdx_archive=str(archive), mathdx_library_sha256=archive_sha,
                previous_dependency_receipt=str(dependency_path),
                previous_dependency_receipt_sha256=_sha(dependency_path),
                current_dependency_inputs=dependency_inputs,
                compile_commands=commands, architecture=architecture,
                origin_source_identity=record['source_identity'], task_keys=sorted(item['tasks']),
                scope='one current Newton recipe only; does not certify or update the unified AOT source identity')
            if target.exists() and _sha(target) != binary_sha:
                raise ValueError('Newton path alias current binary differs: ' + str(target))
            if dry_run:
                return dict(receipt, published=False, existing=target.exists())
            migrated = dict(previous, sha256=identity, source=str(target.with_suffix('.cu')),
                device_bridge=str(target.with_name(identity + '_bridge.cu')), stage_sources=closure,
                library_file=str(target), compile_commands=commands, compile_commands_executed=[],
                command=[moved(a) for a in previous['command']],
                dependency_receipt_file=str(target.with_suffix('.dependencies.json')),
                path_alias_receipt=str(target.with_suffix('.path-alias.json')))
            encoded = lambda value: (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
            # Publish metadata/proof first; the complete binary becomes visible
            # last. Each file is exclusive and immutable, including retries.
            _publish_alias_file(target.with_suffix('.cu'), source.encode())
            _publish_alias_file(target.with_name(identity + '_bridge.cu'), bridge_source.encode())
            _publish_alias_file(target.with_suffix('.path-alias.json'), encoded(receipt))
            _publish_alias_file(target.with_suffix('.json'), encoded(migrated))
            dependency_payload=json.dumps(dependency_inputs,sort_keys=True,separators=(',', ':')).encode()
            new_dependencies=dict(schema='libero-newton-dependencies-v1', recipe=identity,
                binary_sha256=binary_sha, inputs=dependency_inputs,
                inputs_sha256=hashlib.sha256(dependency_payload).hexdigest())
            _publish_alias_file(target.with_suffix('.dependencies.json'), encoded(new_dependencies))
            existed = target.exists()
            _publish_alias_file(target, raw)
            return dict(receipt, published=not existed, existing=existed)
    return None


def _source_identity_files(*, include_excluded=False):
    """Native inputs; optional MJWarp owns separate source/binary receipts.

    Session guards also watch excluded hosts and optional engine sources.
    """
    excluded = {'host/build/aot.py', 'host/build/task_catalog.py', 'host/build/compilation.py',
                'build.py', 'host/adapters/environment/batch_env.py',
                'host/runtime/core/product_paths.py', 'cuda/control/mjwarp_controller.cu',
                'data/mjwarp_source_pin.json', 'host/backends/selection.py',
                # Opt-in FP32 leaf owns its CUDA fingerprint/local replay
                # receipt. It is not linked into either native scalar engine.
                'cuda/newton/mjwarp_hessian.cu', 'cuda/newton/mjwarp_hessian.cuh',
                'host/runtime/newton/mjwarp_hessian.py',
                'cuda/newton/mjwarp_direction.cu', 'cuda/newton/mjwarp_direction.cuh',
                'host/runtime/newton/mjwarp_direction.py',
                'cuda/newton/mjwarp_constraint.cu', 'cuda/newton/mjwarp_constraint.cuh',
                'host/runtime/newton/mjwarp_constraint.py',
                'cuda/newton/mjwarp_linesearch.cu', 'cuda/newton/mjwarp_linesearch.cuh',
                'host/runtime/newton/mjwarp_linesearch.py',
                'cuda/newton/mjwarp_solver_state.cu', 'cuda/newton/mjwarp_solver_state.cuh',
                'cuda/newton/mjwarp_products.cuh', 'host/runtime/newton/mjwarp_solver_state.py',
                'cuda/dynamics/mjwarp_mass_solve.cu', 'cuda/dynamics/mjwarp_mass_solve.cuh',
                'host/runtime/dynamics/mjwarp_mass_solve.py',
                'cuda/dynamics/mjwarp_smooth_force.cu', 'cuda/dynamics/mjwarp_smooth_force.cuh',
                'host/runtime/dynamics/mjwarp_smooth_force.py',
                'cuda/dynamics/mjwarp_mass_factor.cu', 'cuda/dynamics/mjwarp_mass_factor.cuh',
                'host/runtime/dynamics/mjwarp_mass_factor.py',
                'cuda/dynamics/mjwarp_mass_matrix.cu', 'cuda/dynamics/mjwarp_mass_matrix.cuh',
                'host/runtime/dynamics/mjwarp_mass_matrix.py',
                'cuda/dynamics/mjwarp_velocity.cu', 'cuda/dynamics/mjwarp_velocity.cuh',
                'cuda/dynamics/mjwarp_bias.cu', 'cuda/dynamics/mjwarp_bias.cuh',
                'cuda/dynamics/mjwarp_spatial.cuh', 'host/runtime/dynamics/mjwarp_motion.py',
                'host/runtime/dynamics/mjwarp_velocity.py', 'host/runtime/dynamics/mjwarp_bias.py',
                'cuda/dynamics/mjwarp_postconstraint.cu', 'cuda/dynamics/mjwarp_postconstraint.cuh',
                'host/runtime/dynamics/mjwarp_postconstraint.py',
                'cuda/dynamics/mjwarp_passive.cu', 'cuda/dynamics/mjwarp_passive.cuh',
                'host/runtime/dynamics/mjwarp_passive.py',
                'cuda/dynamics/mjwarp_actuation.cu', 'cuda/dynamics/mjwarp_actuation.cuh',
                'host/runtime/dynamics/mjwarp_actuation.py'}
    files = {}
    for base in (GPU_ROOT, REPO_ROOT / 'scripts'):
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.cu', '.cuh', '.h', '.hpp', '.json'):
                if '__pycache__' not in path.parts and not (
                        not include_excluded and base == GPU_ROOT and
                        (path.relative_to(base).as_posix() in excluded or
                         path.relative_to(base).as_posix().startswith('host/backends/mjwarp/'))):
                    files[str(path.relative_to(REPO_ROOT))] = path
    return files


def source_identity():
    files = {name: _sha(path) for name, path in _source_identity_files().items()}
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded + ('|real='+str(scalar.bits)).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def configure_warp_cache(root, *, seed_cache=False):
    import warp as wp
    from warp._src.build import init_kernel_cache
    root = project_root(root)
    base = root / 'warp'
    version = wp.config.version
    destination = base / version
    destination.mkdir(parents=True, exist_ok=True)
    if seed_cache:
        existing = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache'))) / 'warp' / version
        sources = ([existing] if existing.exists() else []) + [base]
        for source in sources:
            for module in source.glob('wp_*'):
                if not module.is_dir() or not any(module.glob('*.meta')):
                    continue
                target = destination/module.name
                target.mkdir(exist_ok=True)
                for path in module.iterdir():
                    if path.is_file() and not (target/path.name).exists():
                        shutil.copyfile(path,target/path.name)
    init_kernel_cache(str(base))
    return destination


def scene_for(root, bddl, seed):
    root = project_root(root)
    task_key = Path(bddl).parent.name + '/' + Path(bddl).name
    index = root / 'TASK_INSTANCES.json'
    if not index.exists():
        return None
    records = json.loads(index.read_text())['tasks']
    scene = next((r for r in records if r['key'] == task_key and r['seed'] == seed), None)
    if scene is None:
        return None
    if _sha(bddl) != scene['bddl_sha256']:
        raise ValueError('BDDL differs from the prepared scene: ' + task_key)
    directory = Path(scene['directory'])
    files = scene.get('files', {'model.mjb': scene['model_sha256']})
    for name, expected in files.items():
        if _sha(directory/name) != expected:
            raise ValueError('prepared scene differs: ' + task_key + '/' + name)
    return scene


OBSERVATION_PRODUCT_VERSION = 1


def observation_product_contract():
    """Explicit default observer ABI, including excluded constructor sources."""
    producers = ('host/build/aot.py', 'host/adapters/environment/batch_env.py',
        'host/adapters/environment/goal_read_plan.py', 'host/bindings/observations.py',
        'host/adapters/observations/adapter.py', 'host/adapters/observations/runtime_binding.py',
        'host/adapters/observations/device.py', 'host/adapters/observations/scene_kinematics.py')
    return dict(schema='libero-batch-observation-products-v1', version=OBSERVATION_PRODUCT_VERSION,
        dataflow_profile='fast', debug=False,
        source_files={name: _sha(GPU_ROOT/name) for name in producers},
        host_features=['host_goal_field_plan_v1', 'dynamic_contact_fallback_v1'],
        profiles={
            'no-camera': dict(use_camera_obs=False, use_object_obs=True, camera_names=[], camera_heights=[],
                camera_widths=[], camera_depths=[], camera_segmentations=[]),
            'dual-rgb128': dict(use_camera_obs=True, use_object_obs=True, camera_names=['agentview', 'robot0_eye_in_hand'],
                camera_heights=[128, 128], camera_widths=[128, 128],
                camera_depths=[False, False], camera_segmentations=[None, None])})


def observation_profile_for_options(options=None):
    """Name only the two built default profiles; never broaden a ready claim."""
    from .compilation import MissingCompiledProduct
    options = dict(options or {})
    if options.get('use_object_obs', True) is not True:
        raise MissingCompiledProduct('Observation profile is absent from the unified build: use_object_obs must be True')
    if not options.get('use_camera_obs', False):
        return 'no-camera'
    def pair(name, default):
        value = options.get(name, default)
        return list(value) if isinstance(value, (list, tuple)) else [value, value]
    if (options.get('camera_names', ['agentview', 'robot0_eye_in_hand']) in
            (['agentview', 'robot0_eye_in_hand'], ('agentview', 'robot0_eye_in_hand')) and
            pair('camera_heights', 128) == [128, 128] and pair('camera_widths', 128) == [128, 128] and
            pair('camera_depths', False) == [False, False] and
            pair('camera_segmentations', None) == [None, None]):
        return 'dual-rgb128'
    raise MissingCompiledProduct('Observation profile is absent from the unified build; '
        'only no-camera and default dual RGB 128 profiles are prepared')


def _observation_features(profile):
    result = ['direct_sampler_v1', 'retained_sampler_fallback_v1', 'lowdim_reset_kernel_v1']
    if profile == 'dual-rgb128':
        result += ['scene_clock_reset_kernel_v1', 'retained_scene_transform_fallback_v1']
    return result


def _observation_warp_modules(profile):
    names = ['libero.gpu.host.adapters.observations.packing',
             'libero.gpu.host.adapters.observations.device']
    if profile == 'dual-rgb128':
        names.append('libero.gpu.host.adapters.observations.scene_kinematics')
    return names


def _loaded_warp_artifacts(root, device, *, required_modules=None):
    import warp as wp
    from warp._src.context import user_modules
    context = wp.get_device(device).context
    modules = {}
    for module in user_modules.values():
        if required_modules is not None and module.name not in required_modules:
            continue
        artifacts = {}
        for owner, block in module.execs:
            if owner != context:
                continue
            directory = Path(wp.config.kernel_cache_dir)/module.get_module_identifier(block_dim=block)
            for path in (directory/module._get_compile_output_name(wp.get_device(device), block_dim=block),
                         directory/module._get_meta_name(block_dim=block)):
                artifacts[str(path.relative_to(root))] = _sha(path)
        if artifacts:
            modules[module.name] = dict(artifacts=artifacts)
    if required_modules is not None and set(modules) != set(required_modules):
        raise ValueError('Observation fallback/reset Warp modules were not loaded by cold capture')
    return modules


def _observation_products_ready(products, contract, *, root, native_artifacts, warp_artifacts):
    if (not isinstance(products, dict) or products.get('contract') != contract or
            products.get('physics_arenas') != 1 or products.get('replayed') is not False or
            products.get('host_features') != contract['host_features'] or
            set(products.get('profiles', {})) != set(contract['profiles'])):
        return False
    for name, spec in contract['profiles'].items():
        row = products['profiles'][name]
        if (not isinstance(row, dict) or row.get('spec') != spec or
                row.get('captured_features') != _observation_features(name) or
                row.get('capture') != 'discarded-force-module-load' or row.get('replayed') is not False or
                row.get('dataflow_profile') != contract['dataflow_profile'] or row.get('debug') is not False):
            return False
        installation = row.get('installation', {})
        if (installation.get('installed') is not True or installation.get('feature_key') != name or
                installation.get('dataflow_profile') != 'fast' or installation.get('debug') is not False or
                installation.get('batch_observation_abi') != 'direct-v1' or
                installation.get('graph_guard_registered') is not True):
            return False
        bindings = {}
        names = set()
        for manifest in installation.get('native_artifacts', []):
            library = manifest.get('library_file', manifest.get('library'))
            digest = manifest.get('library_sha256', manifest.get('binary_sha256',
                manifest.get('build_receipt', {}).get('binary_sha256')))
            try:
                spelling = str(Path(library).relative_to(root))
            except (TypeError, ValueError):
                return False
            if not digest or native_artifacts.get(spelling) != digest:
                return False
            bindings[spelling] = digest
            names.add(manifest.get('name'))
        expected_names = {'direct_observations'}
        if name == 'dual-rgb128':
            expected_names.add('handwritten_scene')
        if names != expected_names or row.get('native_artifacts') != bindings:
            return False
        modules = row.get('warp_modules', {})
        if set(modules) != set(_observation_warp_modules(name)):
            return False
        for module in modules.values():
            artifacts = module.get('artifacts', {})
            if not artifacts or not all(warp_artifacts.get(p) == digest for p, digest in artifacts.items()):
                return False
    return True


def _require_zero_compiler_commands(events):
    if any(event.get('commands_executed') for event in events):
        raise RuntimeError('reuse-only qualification observed compiler commands; completion refused')


def prepare_observation_products(env, root, *, contract=None, reuse_only=False):
    """Prepare both observers on one existing arena, without replay or rendering."""
    from .compilation import compilation_scope, require_compilation, compilation_allowed
    if type(reuse_only) is not bool:
        raise ValueError('reuse_only must be bool')
    if reuse_only:
        if compilation_allowed():
            raise ValueError('reuse-only observer preparation requires a frozen compilation scope')
    else:
        require_compilation('offline observation products', root)
    with compilation_scope(True) as scope:
        products = _prepare_observation_profiles(env, root, contract=contract)
    if reuse_only:
        _require_zero_compiler_commands(scope['events'])
    # install_batch already records its native events. The discarded captures
    # occur after construction, so retain their Warp misses explicitly once.
    env.build_report['events'].extend(e for e in scope['events'] if e['kind'] == 'warp')
    return products


def _prepare_observation_profiles(env, root, *, contract=None):
    import warp as wp
    from libero.gpu.host.adapters.observations.adapter import GpuObservationAdapter
    from libero.gpu.host.adapters.observations.runtime_binding import FixedRuntimeObservationBinding
    from libero.gpu.host.adapters.observations.device import reset_kernel
    from libero.gpu.host.adapters.observations.scene_kinematics import DeviceSceneTransforms
    from libero.gpu.host.bindings.observations import install_batch
    contract = observation_product_contract() if contract is None else contract
    if contract != observation_product_contract():
        raise ValueError('Observation product sources changed during offline preparation')
    bundle = env._physics.bundle
    profiles = {}
    for name, spec in contract['profiles'].items():
        metadata = dict(env.built.metadata, **spec)
        observer = GpuObservationAdapter(env.built.native_model, metadata, env.num_envs,
            device=env.stream.device)
        try:
            # Alternate bindings read the same arena; neither live callback is replaced.
            binding = FixedRuntimeObservationBinding(bundle, observer)
            manifest = install_batch(observer, binding, bundle, root,
                dataflow_profile='fast', debug=False,
                strict_build_receipts=getattr(bundle, 'observation_build_options', {}).get('strict_build_receipts', False))
            sampler = observer._cuda_direct_sampler
            fields = binding.fields()
            mask = bundle.runtime.valid
            with wp.ScopedCapture(stream=env.stream, force_module_load=True):
                observer.sample(fields, world_mask=mask, stream=env.stream)
                sampler.baseline_sample(fields, world_mask=mask, stream=env.stream)
                observer.lowdim.reset(world_mask=mask, stream=env.stream)
                if observer.scene_cache is not None:
                    # The instance update is native; call the retained class entry
                    # explicitly so its overlap fallback Warp module is loaded too.
                    DeviceSceneTransforms.update(observer.scene_transforms,
                        fields['body_xpos'], fields['body_xmat'], mask, env.stream)
                    scene = observer.scene_cache
                    # scene.reset() has host reads; capture only its device kernel.
                    wp.launch(reset_kernel, dim=env.num_envs, inputs=[scene.ncamera, 0, mask,
                        scene.elapsed, scene.captured, scene.scenes, scene.scenes, scene.scenes],
                        device=observer.device, stream=env.stream)
            native_artifacts = {}
            for product in manifest['native_artifacts']:
                path = Path(product.get('library_file', product.get('library')))
                native_artifacts[str(path.relative_to(root))] = _sha(path)
            modules = _loaded_warp_artifacts(root, env.stream.device,
                required_modules=_observation_warp_modules(name))
            profiles[name] = dict(spec=spec, captured_features=_observation_features(name),
                capture='discarded-force-module-load', replayed=False,
                dataflow_profile='fast', debug=False, installation=manifest,
                native_artifacts=native_artifacts, warp_modules=modules)
        finally:
            observer.close()
    return dict(contract=contract, physics_arenas=1, replayed=False,
        host_features=contract['host_features'], profiles=profiles)


def _require_ready_header(record, *, num_envs, seed, fingerprint):
    from .compilation import MissingCompiledProduct
    if (record.get('status') != 'complete' or record['num_envs'] != num_envs or
            record['seed'] != seed or record['source_identity'] != fingerprint):
        raise MissingCompiledProduct('Unified LIBERO build is incomplete or stale; run python -m libero.gpu.build')


def _require_ready_task(record, root, key, *, observation_options, debug, contract):
    from .compilation import MissingCompiledProduct
    if key not in record['tasks']:
        raise MissingCompiledProduct('Task is absent from the unified LIBERO build: ' + key)
    row = record['tasks'][key]
    if (row.get('key') != key or row.get('source_identity') != record['source_identity'] or
            row.get('num_envs') != record['num_envs'] or row.get('seed') != record['seed']):
        raise MissingCompiledProduct('Task qualification differs from the unified LIBERO build: ' + key)
    observation_profile_for_options(observation_options)
    if (debug or record.get('observation_contract') != contract or
            not _observation_products_ready(record['tasks'][key].get('observation_products'), contract,
                root=root, native_artifacts=record['tasks'][key].get('native_artifacts', {}),
                warp_artifacts=record['tasks'][key].get('warp_artifacts', {}))):
        raise MissingCompiledProduct('Observation products are absent or stale for the requested profile; '
            'run python -m libero.gpu.build (default fast, debug=False only)')
    for spelling, digest in record['tasks'][key].get('native_artifacts', {}).items():
        if record['artifacts'].get(spelling) != digest:
            raise MissingCompiledProduct('Task native product is absent from the unified artifact table: ' + spelling)
    for spelling, digest in record['tasks'][key].get('warp_artifacts', {}).items():
        if record['artifacts'].get(spelling) != digest:
            raise MissingCompiledProduct('Task Warp product is absent from the unified artifact table: ' + spelling)


def qualify_ready_session(root=None, *, num_envs, seed):
    """Strictly qualify one immutable build for explicit in-process reuse.

    No GPU initialization, compilation, manifest update or cache write occurs.
    The caller must keep the source/build tree immutable until session.close().
    Reuse checks file metadata, not a second content hash; this is not a guard
    against an adversary who can forge the filesystem metadata. Scene contents
    retain the independent strict checks in scene_for().
    """
    from .ready_session import ReadySession
    return ReadySession._qualify(project_root(root), num_envs=num_envs, seed=seed)


def require_ready(root, bddl, *, num_envs, seed, observation_options=None, debug=False,
                  ready_session=None, return_record=True):
    from .compilation import MissingCompiledProduct
    root = project_root(root)
    if ready_session is not None:
        from .ready_session import ReadySession
        if not isinstance(ready_session, ReadySession):
            raise TypeError('ready_session must be created by qualify_ready_session')
        return ready_session.require_ready(root, bddl, num_envs=num_envs, seed=seed,
            observation_options=observation_options, debug=debug, return_record=return_record)
    path = root / 'AOT_BUILD.json'
    if not path.is_file():
        raise MissingCompiledProduct('Unified LIBERO build is absent; run python -m libero.gpu.build')
    record = json.loads(path.read_text())
    _require_ready_header(record, num_envs=num_envs, seed=seed, fingerprint=source_identity())
    key = Path(bddl).parent.name + '/' + Path(bddl).name
    _require_ready_task(record, root, key, observation_options=observation_options,
        debug=debug, contract=observation_product_contract())
    for spelling, digest in record['artifacts'].items():
        if _sha(root / spelling) != digest:
            raise ValueError('prebuilt artifact differs: ' + spelling)
    return record if return_record else None


def instantiate_newton(root, tasks):
    from libero.gpu.host.runtime.step.s04_jar_specialization_r01 import load_candidate
    root = project_root(root)
    instances = {}
    for nv, nm in sorted({(r['dimensions']['nv'], r['dimensions']['nM']) for r in tasks}):
        package, manifest = load_candidate(nv, nm, root/'newton_instances')
        directory = Path(package.__path__[0])
        instances[f'{nv}:{nm}'] = dict(directory=str(directory),
            files={p.name: _sha(p) for p in directory.iterdir() if p.is_file()})
    save(root/'NEWTON_INSTANCES.json', instances)
    return instances


def offline_newton_source(root, nv, nm):
    from .compilation import MissingCompiledProduct, compilation_allowed
    path = Path(root)/'NEWTON_INSTANCES.json'
    instances = json.loads(path.read_text()) if path.is_file() else {}
    entry = instances.get(f'{nv}:{nm}')
    if entry is None:
        if compilation_allowed():
            return None
        raise MissingCompiledProduct(f'Newton instance NV={nv}, NM={nm} was not built')
    directory = Path(entry['directory'])
    for name, digest in entry['files'].items():
        if _sha(directory/name) != digest:
            raise ValueError('offline Newton instance changed: '+str(directory/name))
    return directory, json.loads((directory/'candidate.json').read_text())


def propagate_scene_aliases(root, tasks, *, fingerprint, num_envs, seed):
    """Reuse an instance only when the entire compiled MuJoCo model is equal."""
    root = project_root(root)
    complete = {}
    contract = observation_product_contract()
    for path in (root/'instances').glob('*/COMPILED.json'):
        row = json.loads(path.read_text())
        if (row['source_identity']==fingerprint and row['num_envs']==num_envs and row['seed']==seed and
                _observation_products_ready(row.get('observation_products'), contract, root=root,
                    native_artifacts=row.get('native_artifacts', {}), warp_artifacts=row.get('warp_artifacts', {}))):
            complete.setdefault(row['model_sha256'], row)
    aliases = []
    for task in tasks:
        source = complete.get(task['model_sha256'])
        if source is None:
            continue
        destination = root/'instances'/(task['suite']+'__'+Path(task['key']).stem)/'COMPILED.json'
        if destination.exists():
            existing = json.loads(destination.read_text())
            if (existing['source_identity'] == fingerprint and existing['num_envs'] == num_envs
                    and existing['seed'] == seed and
                    _observation_products_ready(existing.get('observation_products'), contract, root=root,
                        native_artifacts=existing.get('native_artifacts', {}), warp_artifacts=existing.get('warp_artifacts', {}))):
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        alias = dict(source, key=task['key'], aliased_to=source['key'], seconds=0,
                     native_compiler_invocations=0, warp_compilations=0, events=[])
        save(destination, alias)
        aliases.append(alias)
    return aliases


def _compile_instance(job):
    from .compilation import compilation_scope
    from libero.gpu.host.runtime.core.product_paths import build_unit_lock
    task, root_string, *_ = job
    reuse_only = _instance_reuse_only(job)
    with compilation_scope(False) if reuse_only else nullcontext():
        with build_unit_lock(Path(root_string)/'instance-locks'/task['model_sha256']):
            return _compile_instance_locked(job)


def _instance_reuse_only(job):
    reuse_only = job[7] if len(job) > 7 else False
    if type(reuse_only) is not bool:
        raise ValueError('reuse_only must be bool')
    return reuse_only


def _compile_instance_locked(job):
    from .compilation import compilation_scope
    reuse_only = _instance_reuse_only(job)
    with compilation_scope(False) if reuse_only else nullcontext() as scope:
        return _compile_instance_in_scope(job, reuse_events=scope['events'] if reuse_only else ())


def _compile_instance_in_scope(job, *, reuse_events=()):
    task, root_string, num_envs, seed, device, fingerprint, *extra = job
    contract = extra[0] if extra else observation_product_contract()
    reuse_only = _instance_reuse_only(job)
    if contract != observation_product_contract():
        raise ValueError('Observation product contract changed before instance preparation')
    import numpy as np
    from libero.libero import get_libero_path
    from libero.libero.envs import GpuBatchEnv
    import warp as wp
    root = Path(root_string)
    configure_warp_cache(root)
    destination = root / 'instances' / (task['suite'] + '__' + Path(task['key']).stem)
    destination.mkdir(parents=True, exist_ok=True)
    completed = destination / 'COMPILED.json'
    if completed.exists():
        row = json.loads(completed.read_text())
        intact = (all((root/p).is_file() and _sha(root/p) == digest
                      for p, digest in row.get('native_artifacts', {}).items()) and
                  all((root/p).is_file() and _sha(root/p) == digest
                      for p, digest in row.get('warp_artifacts', {}).items()) and
                  set(row.get('warp_required', [])) == set(row.get('warp_artifacts', {})))
        if (row['source_identity'] == fingerprint and row['num_envs'] == num_envs and
                row['seed'] == seed and 'native_artifacts' in row and intact and
                _observation_products_ready(row.get('observation_products'), contract, root=root,
                    native_artifacts=row.get('native_artifacts', {}), warp_artifacts=row.get('warp_artifacts', {}))):
            resumed = dict(row, resumed=True, seconds=0, events=[],
                           native_compiler_invocations=0, warp_compilations=0)
            if reuse_only:
                resumed.update(reuse_only=True, refresh_kind='reuse_validation_refresh')
            return resumed
    wp.init()
    wp.set_mempool_release_threshold(device, 0)
    tick = time.monotonic()
    bddl = Path(get_libero_path('bddl_files')) / task['key']
    states = np.load(Path(task['directory'])/'initial_states.npy', allow_pickle=False)
    with GpuBatchEnv(bddl, num_envs, init_states=states, seed=seed, device=device,
                     build_root=root, initialize=False, allow_compile=True,
                     physics_engine='native',
                     use_camera_obs=False, ignore_done=True) as env:
        from .compilation import compilation_scope
        with compilation_scope(True) as module_scope:
            env._physics.prepare_substep_modules()
        env.build_report['events'].extend(module_scope['events'])
        observation_products = prepare_observation_products(env, root, contract=contract, reuse_only=reuse_only)
        diagnostics = env.diagnostics()
        if any(diagnostics['actual_integration_count']) or any(diagnostics['first_error']):
            raise RuntimeError('offline preparation advanced physics or reported an error')
        events = list(env.build_report['events'])
        native_paths = []
        for manifest in env._physics.bundle.cuda_manifests:
            binary = manifest.get('library_file', manifest.get('library'))
            if binary:
                native_paths.append(Path(binary))
        native_artifacts = {str(p.relative_to(root)): _sha(p) for p in native_paths}
        warp_artifacts = {}
        for module in _loaded_warp_artifacts(root, device).values():
            warp_artifacts.update(module['artifacts'])
        row = dict(key=task['key'], source_identity=fingerprint, real_bits=scalar.bits, num_envs=num_envs, seed=seed,
            model_sha256=diagnostics['model_sha256'], seconds=time.monotonic()-tick,
            constraint_package=str(env.built._gpu_native_intake_binding.package),
            native_artifacts=native_artifacts, warp_required=sorted(warp_artifacts), warp_artifacts=warp_artifacts,
            observation_products=observation_products,
            native_compiler_invocations=sum(len(e['commands_executed']) for e in events
                if e['kind'] in ('native-cuda', 'newton-cuda', 'collision-native')),
            warp_compilations=sum(len(e['commands_executed']) for e in events if e['kind']=='warp'),
            actual_integration_count=diagnostics['actual_integration_count'], events=events)
        if not _observation_products_ready(observation_products, contract, root=root,
                native_artifacts=native_artifacts, warp_artifacts=warp_artifacts):
            raise ValueError('Offline observation profiles lack their captured native/Warp products')
    if reuse_only:
        # Include close/exit events as well as the nested construction/capture
        # scopes. A fresh fingerprint is published only after real construction.
        row['events'] = list(env.build_report['events'])
        row.update(reuse_only=True, refresh_kind='reuse_validation_refresh')
    del env, states
    gc.collect()
    wp.synchronize_device(device)
    if reuse_only:
        _require_zero_compiler_commands([*reuse_events, *row['events']])
        if row['native_compiler_invocations'] or row['warp_compilations']:
            raise RuntimeError('reuse-only qualification has nonzero compiler counts; completion refused')
    save(completed, row)
    return row


def compile_catalog(root, tasks, *, num_envs=8, seed=0, device='cuda:0', jobs=2, progress=None, reuse_only=False):
    """Qualify real instances, optionally permitting only completed cache hits."""
    from .compilation import compilation_scope
    if type(reuse_only) is not bool:
        raise ValueError('reuse_only must be bool')
    with compilation_scope(False) if reuse_only else nullcontext() as scope:
        return _compile_catalog_in_scope(root, tasks, num_envs=num_envs, seed=seed, device=device,
            jobs=jobs, progress=progress, reuse_only=reuse_only,
            reuse_events=scope['events'] if reuse_only else ())


def _compile_catalog_in_scope(root, tasks, *, num_envs, seed, device, jobs, progress, reuse_only, reuse_events):
    root = project_root(root)
    from libero.gpu.host.runtime.core.specialization_cache import index_packages
    index_packages(root / 'specializations')
    instantiate_newton(root, tasks)
    configure_warp_cache(root, seed_cache=True)
    fingerprint = source_identity()
    contract = observation_product_contract()
    previous_complete = preserve_complete_build(root)
    record = dict(schema='libero-unified-aot-build-v1', status='building',
                  source_identity=fingerprint, real_bits=scalar.bits, num_envs=num_envs, seed=seed, tasks={},
                  observation_contract=contract)
    if reuse_only:
        record.update(reuse_only=True, refresh_kind='reuse_validation_refresh')
    if previous_complete is not None:
        record['previous_complete_build'] = previous_complete
    save(root / 'AOT_BUILD.json', record)
    models = {}
    for task in sorted(tasks, key=lambda r:(r['suite']!='libero_10',r['task_id'])):
        models.setdefault(task['model_sha256'], task)
    representatives, remaining, seen = [], [], set()
    for task in models.values():
        key=(task['dimensions']['nv'],task['dimensions']['nM'])
        (remaining if key in seen else representatives).append(task)
        seen.add(key)
    ordered = sorted(representatives+remaining, key=lambda task: task['suite'] != 'libero_10')
    work = [(task, str(root), num_envs, seed, device, fingerprint, contract, reuse_only) for task in ordered]
    try:
        with get_context('spawn').Pool(jobs, maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(_compile_instance, work):
                record['tasks'][row['key']] = row
                save(root/'AOT_BUILD.json', record)
                if progress:
                    progress(row, len(record['tasks']), len(work))
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        save(root/'AOT_BUILD.json', record)
        raise
    for alias in propagate_scene_aliases(root,tasks,fingerprint=fingerprint,num_envs=num_envs,seed=seed):
        record['tasks'][alias['key']]=alias
    for task in tasks:
        if task['key'] not in record['tasks']:
            path=root/'instances'/(task['suite']+'__'+Path(task['key']).stem)/'COMPILED.json'
            record['tasks'][task['key']]=json.loads(path.read_text())
    if source_identity() != fingerprint:
        raise RuntimeError('GPU sources changed during the unified build')
    if observation_product_contract() != contract:
        raise RuntimeError('Observation product contract changed during the unified build')
    index_packages(root/'specializations',
                   [row['constraint_package'] for row in record['tasks'].values()])
    configure_warp_cache(root, seed_cache=True)
    record['artifacts'] = {str(path.relative_to(root)): _sha(path)
        for subdirectory in ('cuda', 'collision', 'warp')
        for path in sorted((root/subdirectory).rglob('*'))
        if path.is_file() and (path.suffix in ('.so', '.ptx', '.cubin', '.meta') or
                              path.name.endswith('.dependencies.json'))}
    for row in record['tasks'].values():
        if (not _observation_products_ready(row.get('observation_products'), contract, root=root,
                native_artifacts=row.get('native_artifacts', {}), warp_artifacts=row.get('warp_artifacts', {})) or
                any(record['artifacts'].get(p) != digest for field in ('native_artifacts', 'warp_artifacts')
                    for p, digest in row.get(field, {}).items())):
            raise ValueError('Unified task observation products are incomplete or differ from final artifacts')
        if reuse_only:
            _require_zero_compiler_commands(row.get('events', []))
            if row.get('native_compiler_invocations', 0) or row.get('warp_compilations', 0):
                raise RuntimeError('reuse-only task has nonzero compiler counts; completion refused')
    if reuse_only:
        _require_zero_compiler_commands(reuse_events)
    record['status'] = 'complete'
    save(root / 'AOT_BUILD.json', record)
    return record
