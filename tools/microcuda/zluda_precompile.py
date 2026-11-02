#!/usr/bin/env python3
"""Compile every DLSS fatbin with one ZLUDA build into a cache, for make_pack.py.

LD_LIBRARY_PATH=<ROCm lib dir> zluda_precompile.py DLL ZLUDA_LIBCUDA CACHE_HOME SWITCHES [--jobs N]

SWITCHES is the D4R_ZLUDA_ code-generation set the pack is for, as make_pack.py prints it, e.g.
'D4R_ZLUDA_IGNORE_DENORMAL=1,D4R_ZLUDA_IMPLICIT_MAX_BLOCK=256,D4R_ZLUDA_WMMA=1,D4R_ZLUDA_WMMA_FP8=1'.
Each fatbin is loaded with cuModuleLoadData in a worker process (ZLUDA keeps the result in
CACHE_HOME/zluda/ComputeCache/zluda2.db) and unloaded again; nothing is launched. Already cached
modules return at once, so an interrupted run can be resumed. Fatbins without PTX fail (ZLUDA has
nothing to compile) and are left out of the pack. Compiling is CPU heavy and slow for
the big network modules (a minute or more each): do not run it next to a benchmark.
"""
import argparse
import concurrent.futures
import ctypes as c
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'kernels/tools'))
from extract_dlss_ptx import fatbins


def compile_range(libcuda, dll, indices):
    lib = c.CDLL(libcuda)
    for name, args in (('cuInit', [c.c_uint]), ('cuDeviceGet', [c.POINTER(c.c_int), c.c_int]),
                       ('cuCtxCreate_v2', [c.POINTER(c.c_void_p), c.c_uint, c.c_int]),
                       ('cuModuleLoadData', [c.POINTER(c.c_void_p), c.c_void_p]),
                       ('cuModuleUnload', [c.c_void_p])):
        getattr(lib, name).argtypes = args
        getattr(lib, name).restype = c.c_int
    assert lib.cuInit(0) == 0
    device, context = c.c_int(), c.c_void_p()
    assert lib.cuDeviceGet(c.byref(device), 0) == 0
    assert lib.cuCtxCreate_v2(c.byref(context), 0, device.value) == 0
    images = list(fatbins(Path(dll).read_bytes()))
    results = []
    for i in indices:
        image = c.create_string_buffer(images[i][1])
        module = c.c_void_p()
        start = time.time()
        r = lib.cuModuleLoadData(c.byref(module), image)
        if r == 0:
            lib.cuModuleUnload(module)
        results.append((i, r, time.time() - start))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dll', type=Path)
    p.add_argument('libcuda', type=Path)
    p.add_argument('cache_home', type=Path)
    p.add_argument('switches')
    # ZLUDA's cache is one SQLite file: concurrent writers lose inserts silently (SQLITE_BUSY), so default to 1
    p.add_argument('--jobs', type=int, default=1)
    # one process per fatbin: a module the compiler aborts on (e.g. WMMA under D4R_ZLUDA_WAVE64=1) is skipped
    # instead of ending the run
    p.add_argument('--isolate', action='store_true')
    a = p.parse_args()
    for key in [k for k in os.environ if k.startswith('D4R_ZLUDA_')]:
        del os.environ[key]
    for item in filter(None, a.switches.split(',')):
        key, value = item.split('=', 1)
        os.environ[key] = value
    a.cache_home.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(a.cache_home.resolve())
    count = len(list(fatbins(a.dll.read_bytes())))
    # interleave so every worker gets a mix of small and large modules
    chunks = [list(range(j, count, a.jobs)) for j in range(a.jobs)]
    failed = 0
    if a.isolate:
        for i in range(count):
            with concurrent.futures.ProcessPoolExecutor(1) as pool:
                try:
                    [(_, r, seconds)] = pool.submit(compile_range, str(a.libcuda), str(a.dll), [i]).result()
                except concurrent.futures.process.BrokenProcessPool:
                    r, seconds = 'aborted', 0.0
            failed += r != 0
            print(f'fatbin {i:3}: {"ok" if r == 0 else f"CUDA error {r}" if r != "aborted" else "compiler aborted"}'
                  f' ({seconds:.1f} s)', flush=True)
    else:
      with concurrent.futures.ProcessPoolExecutor(a.jobs) as pool:
        for future in concurrent.futures.as_completed(
                [pool.submit(compile_range, str(a.libcuda), str(a.dll), chunk) for chunk in chunks]):
            for i, r, seconds in future.result():
                failed += r != 0
                print(f'fatbin {i:3}: {"ok" if r == 0 else f"CUDA error {r}"} ({seconds:.1f} s)', flush=True)
    print(f'{count - failed}/{count} fatbins compiled into {a.cache_home}/zluda/ComputeCache/zluda2.db')


if __name__ == '__main__':
    main()
