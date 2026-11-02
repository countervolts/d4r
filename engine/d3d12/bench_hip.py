#!/usr/bin/env python3
"""HIP half/f32 WMMA companion to test_matrix.py, using the exact same fixture.

Windows draft: Python + NumPy + ROCm core/devel wheels, no host C++ compiler.
HIPRTC compiles the repository's RDNA 3/4 helpers for the explicit --arch target.
GPU timings use a captured graph, avoiding Python submission gaps. They include
ordered, repeated writes, but no uploads/readback or CPU waits. This is a matrix
microbenchmark, not a model, frame, interop or game performance measurement.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from test_matrix import check_output

P, I, U, S = C.c_void_p, C.c_int, C.c_uint, C.c_size_t


class Hip:
    def __init__(self, sdk):
        self.dll_dirs = [os.add_dll_directory(str(sdk / 'bin'))]
        self.hip = C.CDLL(str(sdk / 'bin' / 'amdhip64_7.dll'))
        rtc_paths = list((sdk / 'bin').glob('hiprtc[0-9]*.dll'))
        if len(rtc_paths) != 1:
            raise RuntimeError('expected exactly one versioned hiprtc DLL')
        self.rtc = C.CDLL(str(rtc_paths[0]))
        self.resources = []

    def call(self, name, types, *args):
        dll = self.rtc if name.startswith('hiprtc') else self.hip
        fn = getattr(dll, name)
        fn.argtypes, fn.restype = types, I
        result = fn(*args)
        if result:
            raise RuntimeError(f'{name}: error {result}')

    def own(self, create, destroy, types=(), args=()):
        obj = P()
        self.call(create, [C.POINTER(P), *types], C.byref(obj), *args)
        self.resources.append((destroy, obj))
        return obj

    def close(self):
        for name, obj in reversed(self.resources):
            self.call(name, [P], obj)
        self.resources.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path, help='fixture directory written by test_matrix.py')
    p.add_argument('--sdk', type=Path, default=Path(sys.prefix)/'Lib/site-packages/_rocm_sdk_core')
    p.add_argument('--arch', required=True, choices=['gfx1100', 'gfx1101', 'gfx1102', 'gfx1200', 'gfx1201'])
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--compile-only', action='store_true', help='cross-compile, without loading or executing the kernel')
    a = p.parse_args()
    if a.trials < 1:
        p.error('trials must be positive')
    meta = json.loads((a.work/'result.json').read_text())
    tiles, steps, repeats = (meta[k] for k in ('tiles', 'steps', 'repeats'))
    source = Path(__file__).with_name('matrix_tile.hip')
    report = dict(arch=a.arch, device_index=a.device, tiles=tiles, steps=steps, repeats=repeats,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256((source.parents[2]/'kernels/common/wmma_layout.h').read_bytes()).hexdigest(),
                  sdk=str(a.sdk.resolve()), timing='HIP events around captured graph; ordered writes; two warmup graphs')
    hip = None
    try:
        hip = Hip(a.sdk.resolve())
        hip.call('hipInit', [U], 0)
        hip.call('hipSetDevice', [I], a.device)
        name = C.create_string_buffer(256)
        hip.call('hipDeviceGetName', [P, I, I], name, len(name), a.device)
        report['device'] = name.value.decode()
        # HIPRTC uses in-memory device source: no host compiler or SDK install.
        prog = hip.own('hiprtcCreateProgram', 'hiprtcDestroyProgram',
                       [C.c_char_p, C.c_char_p, I, P, P],
                       (source.read_bytes(), b'matrix_tile.hip', 0, None, None))
        # hiprtcDestroyProgram takes a pointer to the handle, unlike other APIs.
        hip.resources.pop()
        try:
            options = [f'--gpu-architecture={a.arch}', f'-DMATRIX_STEPS={steps}',
                       '-I'+str(source.parents[2]), '-std=c++17', '-ffreestanding']
            resource_headers = sorted((a.sdk/'lib/llvm/lib/clang').glob('*/include'))
            if resource_headers:
                options.append('-I'+str(resource_headers[-1].resolve()))
            opts = (C.c_char_p*len(options))(*(s.encode() for s in options))
            start = time.perf_counter()
            try:
                hip.call('hiprtcCompileProgram', [P, I, P], prog, len(options), opts)
            finally:
                size = S()
                hip.call('hiprtcGetProgramLogSize', [P, C.POINTER(S)], prog, C.byref(size))
                log = C.create_string_buffer(max(1, size.value))
                hip.call('hiprtcGetProgramLog', [P, P], prog, log)
                (a.work/'hip-compile.log').write_bytes(log.value)
            report['compile_seconds'] = time.perf_counter()-start
            size = S()
            hip.call('hiprtcGetCodeSize', [P, C.POINTER(S)], prog, C.byref(size))
            code = C.create_string_buffer(size.value)
            hip.call('hiprtcGetCode', [P, P], prog, code)
            (a.work/'matrix.hsaco').write_bytes(code.raw)
        finally:
            hip.call('hiprtcDestroyProgram', [C.POINTER(P)], C.byref(prog))
        if a.compile_only:
            report.update(status='compiled_only', code_bytes=len(code))
            return
        module = hip.own('hipModuleLoadData', 'hipModuleUnload', [P], (code,))
        kernel = P()
        hip.call('hipModuleGetFunction', [C.POINTER(P), P, C.c_char_p], C.byref(kernel), module, b'matrix_tile')
        data = C.create_string_buffer((a.work/'input.bin').read_bytes())
        inp = hip.own('hipMalloc', 'hipFree', [S], (len(data)-1,))
        out = hip.own('hipMalloc', 'hipFree', [S], (tiles*1024,))
        hip.call('hipMemcpy', [P, P, S, I], inp, data, len(data)-1, 1)
        stream = hip.own('hipStreamCreate', 'hipStreamDestroy')
        params = (P*2)(C.cast(C.byref(inp), P), C.cast(C.byref(out), P))

        def launch():
            hip.call('hipModuleLaunchKernel', [P, U, U, U, U, U, U, U, P, P, P],
                     kernel, tiles, 1, 1, 32, 1, 1, 0, stream, params, None)

        launch()
        hip.call('hipStreamSynchronize', [P], stream)
        result = C.create_string_buffer(tiles*1024)
        hip.call('hipMemcpy', [P, P, S, I], result, out, len(result), 2)
        (a.work/'hip.bin').write_bytes(result.raw)
        # Reference was saved as f32; regenerate the float64 oracle from inputs.
        raw = np.frombuffer(data.raw[:-1], np.uint8).reshape(tiles, 2048)
        left = raw[:, :512].copy().view(np.float16).reshape(tiles, 16, 16).astype(np.float64)
        right = raw[:, 512:1024].copy().view(np.float16).reshape(tiles, 16, 16).astype(np.float64)
        accum = raw[:, 1024:].copy().view(np.float32).reshape(tiles, 16, 16).astype(np.float64)
        expected = accum+steps*(left@right)
        actual, error = check_output(a.work/'hip.bin', expected, steps)
        report['max_abs_error'] = error
        dx = np.fromfile(a.work/'d3d12.bin', np.float32)[:expected.size].reshape(expected.shape)
        np.testing.assert_allclose(actual, dx, rtol=2e-5, atol=max(1e-6, steps*2e-6))
        report['cross_api_max_abs_error'] = float(np.max(np.abs(actual-dx)))
        hip.call('hipStreamBeginCapture', [P, I], stream, 0)
        for _ in range(repeats):
            launch()
        graph = P()
        hip.call('hipStreamEndCapture', [P, C.POINTER(P)], stream, C.byref(graph))
        hip.resources.append(('hipGraphDestroy', graph))
        executable = hip.own('hipGraphInstantiate', 'hipGraphExecDestroy', [P, P, P, S], (graph, None, None, 0))
        begin = hip.own('hipEventCreate', 'hipEventDestroy')
        end = hip.own('hipEventCreate', 'hipEventDestroy')
        samples = []
        for trial in range(a.trials+2):
            hip.call('hipEventRecord', [P, P], begin, stream)
            hip.call('hipGraphLaunch', [P, P], executable, stream)
            hip.call('hipEventRecord', [P, P], end, stream)
            hip.call('hipEventSynchronize', [P], end)
            elapsed = C.c_float()
            hip.call('hipEventElapsedTime', [C.POINTER(C.c_float), P, P], C.byref(elapsed), begin, end)
            if trial >= 2:
                samples.append(elapsed.value*1000/repeats)
        # Check again after graph execution; timing only accepted with valid output.
        hip.call('hipMemcpy', [P, P, S, I], result, out, len(result), 2)
        if result.raw != (a.work/'hip.bin').read_bytes():
            raise ValueError('HIP output changed after timed graphs')
        report.update(status='passed', gpu_us=statistics.median(samples), samples_us=samples)
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        (a.work/'hip-result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if hip:
            hip.close()


if __name__ == '__main__':
    main()
