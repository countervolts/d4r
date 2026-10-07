#!/usr/bin/env python3
"""GPU integration checks for the MicroCUDA runtime; no games or NGX evaluation.
usage: check_runtime.py LIBRARY PACK_DIR KNOWN_PTX NATIVE_DIR

KNOWN_PTX is the K enc0 module text (a module in the pack with a native dltss_pwin_enc0_layer in NATIVE_DIR).
Pack-state failures (switch mismatch, tampered object) run in child processes: the pack is read once.
"""
import ctypes as c
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ptr, u64, size = c.c_void_p, c.c_uint64, c.c_size_t


def api(lib, name, args):
    f = getattr(lib, name)
    f.argtypes = args
    f.restype = c.c_int
    return f


def child(library, ptx):
    """Prints the cuModuleLoadData result for ptx under the environment it was started with."""
    lib = c.CDLL(library)
    assert api(lib, 'cuInit', [c.c_uint])(0) == 0
    context = ptr()
    assert api(lib, 'cuCtxCreate_v2', [c.POINTER(ptr), c.c_uint, c.c_int])(c.byref(context), 0, 0) == 0
    module = ptr()
    print(api(lib, 'cuModuleLoadData', [c.POINTER(ptr), ptr])(c.byref(module), c.create_string_buffer(Path(ptx).read_bytes())))


def run_child(library, ptx, **env):
    out = subprocess.run([sys.executable, __file__, '--child', library, ptx], env={**os.environ, **env},
                         capture_output=True, text=True, check=True)
    return int(out.stdout.split()[-1])


def main():
    if sys.argv[1] == '--child':
        return child(*sys.argv[2:])
    library, pack, ptx, native = sys.argv[1:]
    os.environ['D4R_MICROCUDA_PACK'] = pack
    os.environ['D4R_MICROCUDA_NATIVE_DIR'] = native
    switches = next(l[9:].strip() for l in open(Path(pack) / 'microcuda-pack.txt') if l.startswith('switches'))
    for key in [k for k in os.environ if k.startswith('D4R_ZLUDA_')]:
        del os.environ[key]
    for item in filter(None, switches.split(',')):
        k, v = item.split('=', 1)
        os.environ[k] = v
    checks = []

    def expect(label, result, code=0):
        assert result == code, (label, result, code)
        checks.append(label)

    expect('switch-set mismatch refuses the pack', run_child(library, ptx, D4R_ZLUDA_WMMA='0'), 200)
    with tempfile.TemporaryDirectory(prefix='d4r-microcuda-check-') as directory:
        shutil.copytree(pack, directory, dirs_exist_ok=True)
        for co in (Path(directory) / 'modules').glob('*.co'):
            b = bytearray(co.read_bytes())
            b[-1] ^= 1
            co.write_bytes(b)
        expect('tampered pack object rejected', run_child(library, ptx, D4R_MICROCUDA_PACK=directory), 200)
    expect('missing pack', run_child(library, ptx, D4R_MICROCUDA_PACK='/nonexistent-microcuda'), 500)

    lib = c.CDLL(library)
    A = lambda name, args: api(lib, name, args)
    init = A('cuInit', [c.c_uint])
    load = A('cuModuleLoadData', [c.POINTER(ptr), ptr])
    unload = A('cuModuleUnload', [ptr])
    get = A('cuModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p])
    launch = A('cuLaunchKernel', [ptr] + [c.c_uint] * 7 + [ptr, c.POINTER(ptr), c.POINTER(ptr)])
    text = c.create_string_buffer(Path(ptx).read_bytes())
    module, function = ptr(), ptr()
    expect('module before init', load(c.byref(module), text), 3)
    expect('invalid init flags', init(1), 1)
    expect('init', init(0))
    proc = A('cuGetProcAddress_v2', [c.c_char_p, c.POINTER(ptr), c.c_int, u64, c.POINTER(c.c_int)])
    address, status = ptr(), c.c_int()
    expect('getProc modern alias', proc(b'cuMemAlloc', c.byref(address), 12080, 0, c.byref(status)))
    assert address.value == c.cast(lib.cuMemAlloc_v2, ptr).value and status.value == 0
    expect('getProc array API', proc(b'cuArrayCreate', c.byref(address), 12080, 0, c.byref(status)))
    expect('getProc unknown', proc(b'cuGraphLaunch', c.byref(address), 12080, 0, c.byref(status)), 500)
    assert address.value is None and status.value == 1
    expect('getProc perthread flag rejected', proc(b'cuLaunchKernel', c.byref(address), 12080, 2, c.byref(status)), 801)
    expect('getProc future version rejected', proc(b'cuLaunchKernel', c.byref(address), 13000, 0, c.byref(status)), 801)
    expect('getProc null output', proc(b'cuInit', None, 12080, 0, None), 1)
    device = c.c_int()
    expect('device', A('cuDeviceGet', [c.POINTER(c.c_int), c.c_int])(c.byref(device), 0))
    context = ptr()
    expect('context', A('cuCtxCreate_v2', [c.POINTER(ptr), c.c_uint, c.c_int])(c.byref(context), 0, device.value))
    current = ptr()
    expect('get current', A('cuCtxGetCurrent', [c.POINTER(ptr)])(c.byref(current)))
    assert current.value == context.value

    expect('unknown PTX', load(c.byref(module), c.create_string_buffer(b'.version 8.7\n.target sm_89\n.address_size 64\n')), 200)
    bad = bytearray(Path(ptx).read_bytes())
    bad[0] ^= 1
    expect('changed PTX', load(c.byref(module), c.create_string_buffer(bytes(bad))), 200)
    expect('known PTX', load(c.byref(module), text))
    expect('unknown function', get(c.byref(function), module, b'unknown'), 500)
    assert function.value is None
    expect('native function', get(c.byref(function), module, b'dltss_pwin_enc0_layer'))
    another = ptr()
    expect('cached lookup', get(c.byref(another), module, b'dltss_pwin_enc0_layer'))
    assert another.value == function.value
    os.environ['D4R_MICROCUDA_NATIVE_DIR'] = '/nonexistent-microcuda'
    plain_module, plain = ptr(), ptr()
    expect('second module instance', load(c.byref(plain_module), text))
    expect('pack function without native override', get(c.byref(plain), plain_module, b'dltss_pwin_enc0_layer'))
    assert plain.value != function.value
    expect('second module unload', unload(plain_module))
    os.environ['D4R_MICROCUDA_NATIVE_DIR'] = native
    args = c.create_string_buffer(176)
    n = size(176)
    extra = (ptr * 5)(3, c.addressof(args), 2, c.addressof(n), None)
    expect('HIP token rejected as CUDA token', launch(function, 1, 1, 1, 32, 1, 1, 0, None, None, extra), 801)
    unterminated = (ptr * 8)(1, c.addressof(args), 2, c.addressof(n), 1, c.addressof(args), 2, c.addressof(n))
    expect('unterminated packed args', launch(function, 1, 1, 1, 32, 1, 1, 0, None, None, unterminated), 1)
    expect('null function', launch(None, 1, 1, 1, 32, 1, 1, 0, None, None, extra), 400)

    allocate = A('cuMemAlloc_v2', [c.POINTER(u64), size])
    d2h = A('cuMemcpyDtoH_v2', [ptr, u64, size])
    source = c.create_string_buffer(bytes(range(256)))
    result = c.create_string_buffer(256)
    device_mem = u64()
    expect('allocation', allocate(c.byref(device_mem), 256))
    expect('zero-filled allocation readback', d2h(result, device_mem, 256))
    assert result.raw == bytes(256)
    stream = ptr()
    expect('stream create', A('cuStreamCreate', [c.POINTER(ptr), c.c_uint])(c.byref(stream), 1))
    expect('async H2D', A('cuMemcpyHtoDAsync_v2', [u64, ptr, size, ptr])(device_mem, source, 256, stream))
    event = ptr()
    expect('event create', A('cuEventCreate', [c.POINTER(ptr), c.c_uint])(c.byref(event), 0))
    expect('event record', A('cuEventRecord', [ptr, ptr])(event, stream))
    expect('event wait', A('cuEventSynchronize', [ptr])(event))
    expect('event query ready', A('cuEventQuery', [ptr])(event))
    expect('D2H', d2h(result, device_mem, 256))
    assert result.raw == source.raw[:256]
    expect('event destroy', A('cuEventDestroy_v2', [ptr])(event))
    expect('stream destroy', A('cuStreamDestroy_v2', [ptr])(stream))
    expect('free', A('cuMemFree_v2', [u64])(device_mem))

    # arrays: CUDA_ARRAY_DESCRIPTOR {size_t Width, Height; int Format; unsigned NumChannels}
    array_create = A('cuArrayCreate_v2', [c.POINTER(ptr), ptr])
    array_desc = A('cuArrayGetDescriptor_v2', [ptr, ptr])
    array_destroy = A('cuArrayDestroy', [ptr])
    memcpy2d = A('cuMemcpy2D_v2', [ptr])
    unorm = ptr()
    expect('UNORM8x4 array', array_create(c.byref(unorm), c.create_string_buffer(struct.pack('<QQiI', 4, 4, 194, 4))))
    expect('array descriptor unsupported, as ZLUDA', array_desc(c.create_string_buffer(24), unorm), 801)
    # CUDA_RESOURCE_DESC (144 bytes): resType 0 = ARRAY, hArray at 8
    resource = c.create_string_buffer(struct.pack('<iiQ', 0, 0, unorm.value) + bytes(128))
    surface = u64()
    expect('surface object', A('cuSurfObjectCreate', [c.POINTER(u64), ptr])(c.byref(surface), resource))
    handle = c.create_string_buffer(96)
    expect('surface handle readback', d2h(handle, surface, 96))
    assert struct.unpack_from('<I', handle.raw, 80)[0] == 194 and handle.raw[48:80] == bytes(32)
    back = c.create_string_buffer(144)
    expect('surface resource desc', A('cuSurfObjectGetResourceDesc', [ptr, u64])(back, surface))
    assert back.raw == resource.raw[:144]
    # CUDA_TEXTURE_DESC (104 bytes): address modes, filter, flags (bit 1 = normalized coordinates)
    texture = c.create_string_buffer(struct.pack('<iiiiI', 1, 1, 1, 0, 2) + bytes(84))
    tex = u64()
    expect('texture object', A('cuTexObjectCreate', [c.POINTER(u64), ptr, ptr, ptr])(c.byref(tex), resource, texture, None))
    expect('texture destroy', A('cuTexObjectDestroy', [u64])(tex))
    expect('surface destroy', A('cuSurfObjectDestroy', [u64])(surface))
    expect('unknown surface destroy', A('cuSurfObjectDestroy', [u64])(surface), 1)
    expect('array destroy', array_destroy(unorm))

    # packed UNORM 10:10:10:2 through its RGBA32F backing: host -> array -> host
    packed = ptr()
    expect('packed 1010102 array', array_create(c.byref(packed), c.create_string_buffer(struct.pack('<QQiI', 4, 2, 80, 4))))
    words = [(i * 37 % 1024) | (i * 101 % 1024) << 10 | (i * 7 % 1024) << 20 | (i % 4) << 30 for i in range(8)]
    host_in = c.create_string_buffer(struct.pack('<8I', *words))
    host_out = c.create_string_buffer(32)

    def copy2d(src_type, src_host, src_array, dst_type, dst_host, dst_array):
        # CUDA_MEMCPY2D: srcX,srcY,srcType,srcHost,srcDevice,srcArray,srcPitch, dst..., WidthInBytes, Height
        return memcpy2d(c.create_string_buffer(struct.pack(
            '<QQi4xQQQQ' 'QQi4xQQQQ' 'QQ', 0, 0, src_type, src_host or 0, 0, src_array or 0, 16,
            0, 0, dst_type, dst_host or 0, 0, dst_array or 0, 16, 16, 2)))
    expect('packed upload', copy2d(1, c.addressof(host_in), None, 3, None, packed.value))
    expect('packed readback', copy2d(3, None, packed.value, 1, c.addressof(host_out), None))
    assert host_out.raw == host_in.raw[:32], (host_out.raw.hex(), host_in.raw.hex())
    expect('packed array destroy', array_destroy(packed))

    expect('module unload', unload(module))
    expect('double unload', unload(module), 400)
    expect('lookup after unload', get(c.byref(function), module, b'dltss_pwin_enc0_layer'), 400)
    expect('context destroy', A('cuCtxDestroy_v2', [ptr])(context))
    print(json.dumps({'passed': len(checks), 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
