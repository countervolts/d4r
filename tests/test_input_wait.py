"""CPU-only input-wait cancellation, polling and profiling contracts. No HIP or Wine runs."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def run(source):
    with tempfile.TemporaryDirectory() as tmp:
        runner = Path(tmp) / 'input-wait'
        subprocess.run(['c++', '-x', 'c++', '-std=c++20', '-pthread', '-Wall', '-Wextra', '-Werror',
                        '-I', str(ROOT / 'tools'), '-o', str(runner), '-'],
                       input=source, text=True, check=True, capture_output=True)
        subprocess.run([str(runner)], check=True, timeout=5)


class InputWaitTests(unittest.TestCase):
    def test_bridge_queues_exact_null_stream_wait_or_reports_unsupported(self):
        bridge = (ROOT / 'tools/wine_nvcuda_bridge.c').read_text()
        wait = function(bridge, 'CUresult WINAPI d4rStreamWaitValue32Eq(')
        run(r'''
#include <cstdint>
#include <cstring>
#include <cassert>
#define WINAPI
using CUresult = int;
using CUdeviceptr = uint64_t;
using HIP_STREAM_WAIT_VALUE32_FN = int(*)(void*, void*, uint32_t, unsigned, uint32_t);
constexpr int CUDA_SUCCESS = 0, CUDA_ERROR_NOT_SUPPORTED = 801;
int context_setup_result = 0, supported = 1, waitResult = 0, queued = 0;
bool hasWait = true;
void ensure_context() {}
void* d4r_ngx_stream(void* stream) { return stream; } // D4R_NGX_STREAM_PRIORITY off
void tracef(const char*, ...) {}
bool trace_verbose() { return false; }
int device(int* out) { *out = 3; return 0; }
int attribute(int* out, int attr, int dev) { assert(attr == 10013 && dev == 3); *out = supported; return 0; }
int queue(void* stream, void* ptr, uint32_t value, unsigned flags, uint32_t mask) {
    assert(stream == nullptr && ptr == reinterpret_cast<void*>(0x10008));
    assert(value == 0xffffffffu && flags == 1 && mask == 0xffffffffu);
    if (waitResult == 0) ++queued;
    return waitResult;
}
void* hip_symbol(const char* name) {
    if (strcmp(name, "hipStreamWaitValue32") == 0) return hasWait ? reinterpret_cast<void*>(queue) : nullptr;
    if (strcmp(name, "hipDeviceGetAttribute") == 0) return reinterpret_cast<void*>(attribute);
    if (strcmp(name, "hipGetDevice") == 0) return reinterpret_cast<void*>(device);
    return nullptr;
}
''' + wait + r'''
int main() {
    assert(d4rStreamWaitValue32Eq(0x10008, 0xffffffffu) == 0 && queued == 1);
    supported = 0;
    assert(d4rStreamWaitValue32Eq(0x10008, 0xffffffffu) == 801 && queued == 1);
    supported = 1; hasWait = false;
    assert(d4rStreamWaitValue32Eq(0x10008, 0xffffffffu) == 801 && queued == 1);
    hasWait = true; waitResult = 1;
    assert(d4rStreamWaitValue32Eq(0x10008, 0xffffffffu) == 801 && queued == 1);
    context_setup_result = 700;
    assert(d4rStreamWaitValue32Eq(0x10008, 0xffffffffu) == 700 && queued == 1);
}
''')

    def test_readiness_discard_retirement_timeout_wrap_and_adaptive_pacing(self):
        run(r'''
#include "d4r_input_wait.h"
#include <cassert>
int main() {
    using A = D4rInputWaitAbort;
    assert(d4r_input_wait_abort(7, 6, 6, false, 100) == A::None);
    assert(d4r_input_wait_abort(7, 6, 8, false, 100) == A::Discarded);
    assert(d4r_input_wait_abort(7, 7, 8, false, 100) == A::None); // slot recheck wins
    assert(d4r_input_wait_abort(7, 7, 7, true, 100) == A::Retired);
    assert(d4r_input_wait_abort(7, 6, 6, false, 5000001) == A::Timeout);
    assert(d4r_input_wait_abort(7, 7, 7, false, 5000001) == A::Timeout); // broken wait after input
    assert(d4r_input_wait_abort(0xffffffffu, 3, 0, false, 100) == A::Discarded);
    assert(d4r_input_wait_abort(0, 0xffffffffu, 0xffffffffu, false, 100) == A::None);
    assert(d4r_input_wait_abort(0, 0xffffffffu, 1, false, 100) == A::Discarded);
    assert(d4r_input_wait_abort(0, 0, 1, false, 100) == A::None);
    assert(d4r_marker_poll_delay_us(2000, 0) == 200);
    assert(d4r_marker_poll_delay_us(2000, 1800) == 50);
    assert(d4r_marker_poll_delay_us(2000, 1850) == 0);
    assert(d4r_marker_poll_delay_us(2000, 2149) == 0);
    assert(d4r_marker_poll_delay_us(2000, 2150) == 20);
    assert(d4r_marker_poll_delay_us(2000, 3750) == 200);
    assert(d4r_marker_poll_delay_us(0, 5000) == 200); // no warm prediction
}
''')

    def test_actual_release_and_drain_helpers_preserve_real_markers_and_ownership(self):
        shim = (ROOT / 'tools/d4r_nvngx_shim.cpp').read_text()
        release = function(shim, 'static void release_ngx_input_wait_locked(')
        complete = function(shim, 'static bool complete_ngx_input_wait(Feature& feature, uint32_t frame, bool abort)\n{')
        run(r'''
#include <atomic>
#include <cassert>
#include <mutex>
#include <thread>
#include <future>
void logf(const char*, ...) {}
void MemoryBarrier() { std::atomic_thread_fence(std::memory_order_seq_cst); }
struct Feature {
    std::atomic<bool> ngxGpuWait{true}, retiring{false};
    void* ngxTimeline = nullptr;
    std::mutex ngxWaitMutex;
    bool ngxWaitActive = true, ngxWaitCancelled = false;
    uint32_t ngxWaitFrame = 7;
    int ngxWaitSlot = 1;
    struct { void* mapped; } ngxInputSync;
    unsigned evaluatedFrames = 100;
};
uint32_t words[4] = {0, 3, 4, 5}, markers[4] = {6, 3, 4, 5};
Feature* current;
int drains = 0, drainResult = 0;
int drain_sync() { // not sync(): glibc's <unistd.h> declares void sync()
    ++drains;
    assert(words[2] == 7); // drain may only run after the CPU released the blocked queue
    assert(current->ngxWaitActive); // ticket and input ownership survive the drain
    return drainResult;
}
struct { struct { int(*ctxSynchronize)() = drain_sync; void(*timelineCancel)(void*) = nullptr; } cu; } g;
''' + release + '\n' + complete + r'''
int main() {
    Feature f; f.ngxInputSync.mapped = words; current = &f;
    assert(complete_ngx_input_wait(f, 8, true)); // another frame cannot release this ticket
    assert(drains == 0 && words[2] == 4);
    // An independent thread can release a blocked host submission without invoking HIP.
    auto blocked = std::async(std::launch::async, [&] {
        for (;;) {
            { std::lock_guard<std::mutex> lock(f.ngxWaitMutex); if (words[2] == 7) return; }
            std::this_thread::yield();
        }
    });
    { std::lock_guard<std::mutex> lock(f.ngxWaitMutex);
      release_ngx_input_wait_locked(f, "discard", false); }
    blocked.get();
    assert(drains == 0 && f.ngxWaitCancelled && f.ngxGpuWait);
    assert(words[0] == 0 && words[1] == 3 && words[3] == 5);
    assert(markers[0] == 6 && markers[2] == 4); // never counterfeit game readiness
    assert(!complete_ngx_input_wait(f, 7, true));
    assert(drains == 1 && !f.ngxWaitActive && f.evaluatedFrames == 0);
    assert(complete_ngx_input_wait(f, 7, true) && drains == 1); // idempotent retirement
    f.ngxWaitActive = true; f.ngxWaitCancelled = false; f.evaluatedFrames = 2;
    assert(complete_ngx_input_wait(f, 7, false)); // output already drained: no extra sync
    assert(drains == 1 && !f.ngxWaitActive && f.evaluatedFrames == 2);
    f.ngxWaitActive = true; f.ngxWaitCancelled = false;
    { std::lock_guard<std::mutex> lock(f.ngxWaitMutex);
      release_ngx_input_wait_locked(f, "timeout", true); }
    assert(!f.ngxGpuWait);
    drainResult = 700;
    assert(!complete_ngx_input_wait(f, 7, true));
    assert(f.retiring && !f.ngxWaitActive && drains == 2);
}
''')

    def test_thread_clock_coarse_or_unavailable_reports_na(self):
        bridge = (ROOT / 'tools/wine_nvcuda_bridge.c').read_text()
        clock = function(bridge, 'double WINAPI d4rThreadCpuTimeMs(void)')
        run(r'''
#include <time.h>
#include <cassert>
#define WINAPI
long resolutionNs = 1;
bool available = true;
int fake_resolution(clockid_t, timespec* out) { out->tv_sec = 0; out->tv_nsec = resolutionNs; return 0; }
int timestamp(clockid_t, timespec* out) { out->tv_sec = 2; out->tv_nsec = 123456789; return available ? 0 : -1; }
#define clock_getres fake_resolution
#define clock_gettime timestamp
''' + clock + r'''
int main() {
    assert(d4rThreadCpuTimeMs() == 2123.456789);
    resolutionNs = 10000000; assert(d4rThreadCpuTimeMs() == -1);
    resolutionNs = 1; available = false; assert(d4rThreadCpuTimeMs() == -1);
}
''')
