"""CPU contracts for native completion and packed-output capability gating."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def run(source, cpp=False):
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / 'check'
        result = subprocess.run(['c++' if cpp else 'cc', '-x', 'c++' if cpp else 'c',
                                 '-std=c++20' if cpp else '-std=c11', '-pthread', '-Wall', '-Wextra', '-Werror',
                                 '-I', str(ROOT / 'tools'), '-o', str(exe), '-'],
                                input=source, text=True, capture_output=True)
        if result.returncode: raise AssertionError(result.stderr)
        subprocess.run([str(exe)], check=True, timeout=5, cwd=tmp)


class LatencyContracts(unittest.TestCase):
    def test_output_membership_filter_keeps_registry_results_exact(self):
        bridge = (ROOT / 'tools/wine_nvcuda_bridge.c').read_text()
        start = bridge.index('enum { OUTPUT_KERNEL_CAPACITY')
        end = bridge.index('/* 1 when the last output kernel', start)
        code = bridge[start:end]
        run(r'''
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#define CUfunction void*
#define R_OK 4
static const char* fake_getenv(const char* name) { (void)name; return "/objects"; }
static int fake_access(const char* path, int mode) { (void)path; (void)mode; return 0; }
#define getenv fake_getenv
#define access fake_access
typedef struct { char name[128]; int rgba8; } NativeKernel;
NativeKernel objects[1] = {{"hiluma_engine_output_candidate", 1}};
NativeKernel* native_kernels = objects;
size_t native_kernel_count = 1;
''' + code + r'''
int main(void) {
    void* output = (void*)(uintptr_t)0x5000;
    note_function_lookup(output, "hiluma_engine_output_candidate"); note_launch(output);
    assert(last_output_native == 1 && last_output_rgba8 == 1);
    for (uintptr_t i = 1; i < 1000; ++i) {
        void* network = (void*)(i * 16);
        if (network == output) continue;
        note_launch(network);
        assert(last_output_native == 1 && last_output_rgba8 == 1); // negatives/collisions change nothing
    }
    note_function_lookup(output, "rrlite_downsample_kernel_static_ldr"); note_launch(output);
    assert(last_output_native == 1 && last_output_rgba8 == 0); // newest lookup wins on handle reuse
}
''')

    def test_completed_prefix_callback_gate(self):
        run(r'''
#include "d4r_frame_completion.h"
#include <cassert>
int main() {
    D4rFrameCompletion completion;
    assert(!completion.can_signal(1));
    completion.admit(1); completion.admit(2); completion.admit(3);
    assert(completion.can_signal(1) && !completion.can_signal(2));
    assert(completion.retire(3) == 0); // later discard does not cross unfinished work
    assert(!completion.can_signal(3));
    assert(completion.retire(1) == 1 && completion.can_signal(2));
    assert(completion.retire(2) == 3 && !completion.can_signal(3));
}
''', cpp=True)

    def test_actual_native_callback_cancellation_monotonicity_and_detached_cleanup(self):
        bridge = (ROOT / 'tools/wine_nvcuda_bridge.c').read_text()
        start = bridge.index('typedef struct', bridge.index("/* NGX's null-stream completion"))
        end = bridge.index('} D4rNgxTimeline;', start) + len('} D4rNgxTimeline;')
        definitions = bridge[start:end]
        bodies = '\n'.join(function(bridge, sig) for sig in (
            'int WINAPI d4rNgxTimelineSignal(', 'static void d4r_ngx_completed(',
            'CUresult WINAPI d4rNgxTimelinePrepare(', 'void WINAPI d4rNgxTimelineCancel(',
            'CUresult WINAPI d4rNgxTimelineQueue(', 'void WINAPI d4rNgxTimelineDestroy(',
            'CUresult WINAPI d4rNgxTimelineWait('))
        run(r'''
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#define WINAPI
#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_READY 600
#define CUDA_ERROR_NOT_SUPPORTED 801
#define CUDA_ERROR_UNKNOWN 999
#define CUresult int
typedef struct { int sType; const void* pNext; uint64_t semaphore, value; } D4rVkSemaphoreSignalInfo;
typedef int (*VK_SIGNAL_SEMAPHORE_FN)(void*, const D4rVkSemaphoreSignalInfo*);
typedef int (*HIP_LAUNCH_HOST_FN)(void*, void(*)(void*), void*);
static void* d4r_ngx_stream(void* stream) { return stream; } /* D4R_NGX_STREAM_PRIORITY off */
void (*callback)(void*); void* argument;
uint64_t signalled = 0; int signals = 0, launchResult = 0;
int signal_fake(void* device, const D4rVkSemaphoreSignalInfo* info) {
    (void)device; assert(info->value > signalled); signalled = info->value; ++signals; return 0;
}
static void* complete_later(void* unused) {
    (void)unused; struct timespec pause = {0, 20000000}; nanosleep(&pause, NULL);
    callback(argument); return NULL;
}
int launch_fake(void* stream, void(*fn)(void*), void* arg) {
    assert(stream == NULL);
    if (!launchResult) { callback = fn; argument = arg; }
    return launchResult;
}
''' + definitions + '\n' + bodies + r'''
int main(void) {
    D4rNgxTimeline* t = calloc(1, sizeof(*t));
    assert(pthread_mutex_init(&t->lock, NULL) == 0);
    pthread_condattr_t attr; assert(pthread_condattr_init(&attr) == 0);
    assert(pthread_condattr_setclock(&attr, CLOCK_MONOTONIC) == 0);
    assert(pthread_cond_init(&t->completed, &attr) == 0); pthread_condattr_destroy(&attr);
    t->signal = signal_fake; t->launch = launch_fake;
    assert(d4rNgxTimelinePrepare(t) == 0);
    assert(d4rNgxTimelineQueue(t, 1) == 0);
    assert(d4rNgxTimelinePrepare(t) == 600 && d4rNgxTimelineQueue(t, 2) == 600);
    d4rNgxTimelineCancel(t); callback(argument); assert(signals == 0 && t->busy == 0);
    assert(d4rNgxTimelineWait(t) == 0);
    assert(d4rNgxTimelineSignal(t, 1) == 0 && signals == 1);
    assert(d4rNgxTimelinePrepare(t) == 0 && d4rNgxTimelineQueue(t, 2) == 0);
    callback(argument); assert(signalled == 2 && signals == 2);
    assert(d4rNgxTimelinePrepare(t) == 0 && d4rNgxTimelineQueue(t, 3) == 0);
    assert(d4rNgxTimelineSignal(t, 4) == 0); callback(argument);
    assert(signalled == 4 && signals == 3); // late callback never signals backwards
    launchResult = 1;
    assert(d4rNgxTimelinePrepare(t) == 0 && d4rNgxTimelineQueue(t, 5) == 801 && t->busy == 0);
    launchResult = 0;
    assert(d4rNgxTimelineQueue(t, 5) == 0);
    pthread_t worker; assert(pthread_create(&worker, NULL, complete_later, NULL) == 0);
    assert(d4rNgxTimelineWait(t) == 0); pthread_join(worker, NULL);
    assert(signalled == 5 && signals == 4);
    assert(d4rNgxTimelinePrepare(t) == 0 && d4rNgxTimelineQueue(t, 6) == 0);
    d4rNgxTimelineDestroy(t); callback(argument); // ticket outlives owner, cancellation forbids a signal
    assert(signals == 4);
}
''')

    def test_all_half_values_pack_like_the_reference_unorm_conversion(self):
        header = (ROOT / 'kernels/tex/tex_common.h').read_text()
        body = function(header, '__attribute__((device)) static inline uint32_t redirect_unorm(')
        body = body.replace('__attribute__((device)) ', '')
        run(r'''
#include <cstdint>
#include <cmath>
#include <cassert>
''' + body + r'''
int main() {
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const int exponent = (bits >> 10) & 31, mantissa = bits & 1023;
        double value = exponent == 0 ? std::ldexp(double(mantissa), -24)
                     : exponent == 31 ? (mantissa ? NAN : INFINITY)
                     : std::ldexp(1.0 + double(mantissa) / 1024.0, exponent - 15);
        if (bits & 0x8000) value = -value;
        value = value > 0.0 ? value : 0.0;
        value = value < 1.0 ? value : 1.0;
        const uint32_t expected = uint32_t(std::floor(value * 255.0 + 0.5));
        assert(redirect_unorm(uint16_t(bits), 255.0f) == expected);
    }
}
''', cpp=True)

    def test_packed_capability_requires_matching_code_object_bytes(self):
        bridge = (ROOT / 'tools/wine_nvcuda_bridge.c').read_text()
        body = function(bridge, 'static int native_rgba8_verified(')
        run(r'''
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
''' + body + r'''
int main(void) {
    assert(!native_rgba8_verified("/does/not/exist", "output"));
    FILE* f = fopen("output.hsaco", "wb"); assert(f != NULL); fputs("abc", f); fclose(f);
    assert(!native_rgba8_verified(".", "output"));
    f = fopen("d4r-rgba8.txt", "w"); fputs("output e71fa2190541574b\n", f); fclose(f);
    assert(native_rgba8_verified(".", "output"));
    f = fopen("output.hsaco", "ab"); fputc('d', f); fclose(f);
    assert(!native_rgba8_verified(".", "output"));
    assert(!native_rgba8_verified(".", "other"));
}
''')
