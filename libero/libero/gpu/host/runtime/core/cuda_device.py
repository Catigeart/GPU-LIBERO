"""Query CUDA device architecture without loading a numerical CUDA product."""
import ctypes as ct


def device_arch(device):
    if hasattr(device, 'arch'):
        return device.arch
    name = str(device)
    if not name.startswith('cuda:'):
        raise ValueError('A CUDA device is required: ' + name)
    ordinal = int(name.split(':', 1)[1])
    driver = ct.CDLL('libcuda.so.1')
    def check(code):
        if code:
            raise RuntimeError('CUDA device query failed with driver error ' + str(code))
    check(driver.cuInit(0))
    handle, major, minor = ct.c_int(), ct.c_int(), ct.c_int()
    check(driver.cuDeviceGet(ct.byref(handle), ordinal))
    check(driver.cuDeviceGetAttribute(ct.byref(major), 75, handle))
    check(driver.cuDeviceGetAttribute(ct.byref(minor), 76, handle))
    return 10 * major.value + minor.value
