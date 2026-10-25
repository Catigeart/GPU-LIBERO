"""Bind a prepared Newton library and query its actual CUDA kernel once."""
import ctypes as ct
from pathlib import Path

from libero.gpu.host.build.compilation import MissingCompiledProduct
from libero.gpu.host.build.library_lifetime import load_cdll

INFO_VERSION = 1
INFO_FIELDS = 11


def load_product(path, *, parameter_bytes, block_threads, shared_bytes, architecture,
                 combined_entry, shared_limit_bytes=48 * 1024):
    path = Path(path).resolve()
    if not path.is_file():
        raise MissingCompiledProduct('Prepare the Newton product offline: ' + str(path))
    library = load_cdll(path)
    try:
        info = library.newton_kernel_info
        launch = library.launch_persistent
        entry = library.launch_entry if combined_entry else None
    except AttributeError as error:
        raise MissingCompiledProduct('Newton launcher/query ABI missing; prepare offline: ' + str(path)) from error
    info.argtypes = [ct.POINTER(ct.c_size_t), ct.c_size_t]
    info.restype = ct.c_int
    values = (ct.c_size_t * INFO_FIELDS)()
    status = info(values, INFO_FIELDS)
    if status:
        raise RuntimeError('Newton CUDA resource query failed: ' + str(status))
    version, size, threads, dynamic, arch, registers, local, static, kernel_threads, device_shared, device_threads = values
    if (version, size, threads, dynamic, arch) != (
            INFO_VERSION, parameter_bytes, block_threads, shared_bytes, architecture):
        raise ValueError('Newton parameter ABI, launch configuration or architecture differs')
    limit = min(int(shared_limit_bytes), device_shared)
    if limit <= 0 or static + dynamic > limit:
        raise ValueError(('Newton static + dynamic shared memory exceeds launch limit', static, dynamic, limit))
    if threads < 1 or threads > min(kernel_threads, device_threads):
        raise ValueError(('Newton block size exceeds kernel/device limit', threads, kernel_threads, device_threads))
    launch.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
    launch.restype = ct.c_int
    if entry is not None:
        entry.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p,
                         ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_void_p]
        entry.restype = ct.c_int
    resources = dict(registers_per_thread=registers, local_bytes=local,
        static_shared_bytes=static, dynamic_shared_bytes=dynamic,
        total_shared_bytes=static + dynamic, shared_limit_bytes=limit,
        kernel_max_threads=kernel_threads, device_max_threads=device_threads,
        library_file=str(path), private_owner_matched=True, passed=True,
        scope='loaded Newton kernel attributes; no performance claim')
    return library, resources


def retained_resources(core, *, shared_limit_bytes):
    """Reuse the query bound to this retained library; never reopen its file."""
    if core.library is not core._resource_library:
        raise ValueError('Newton resource library owner replaced')
    result = dict(core._resource_info)
    if Path(core.library._name).resolve() != Path(core.manifest['library_file']).resolve():
        raise ValueError('Newton resource image differs from loaded private owner')
    if int(core.manifest['shared_bytes']) != result['dynamic_shared_bytes']:
        raise ValueError('Newton shared launch configuration changed')
    limit = min(int(shared_limit_bytes), result['shared_limit_bytes'])
    if limit < 1 or result['total_shared_bytes'] > limit:
        raise ValueError('Newton shared memory exceeds requested limit')
    result['shared_limit_bytes'] = limit
    return result
