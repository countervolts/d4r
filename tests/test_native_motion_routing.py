"""Execute the shim's motion routing with native readiness and NGX failures."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeMotionRoutingTests(unittest.TestCase):
    def test_independent_repair_and_ngx_motion(self):
        source = (ROOT / 'tools/d4r_nvngx_shim.cpp').read_text()
        begin = source.index('    CudaObject nativeMotion = 0;')
        end = source.index('    // Ray Reconstruction: the auxiliary', begin)
        block = source[begin:end]
        begin = source.index('    const bool wireHistoryRequested=nativeMotion')
        end = source.index(';', begin) + 1
        wire = source[begin:end]
        program = r'''
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
using CudaObject = uint64_t;
static unsigned substitute=1, repair=1;
static bool available=true, matched=true;
static CudaObject selected=0;
static unsigned env_uint(const char* name, unsigned) {
    return std::strcmp(name,"D4R_NATIVE_MV_SUBSTITUTE")==0 ? substitute : repair;
}
static void logf(const char*, ...) {}
struct Feature {
    unsigned flags=0x43, cudaFlags=0x43, outWidth=2560, outHeight=1440;
    void* cudaHandle=(void*)1;
};
struct Slot { bool nativeLive=true; };
struct Params { unsigned renderWidth=853,renderHeight=480; float mvScaleX=853,mvScaleY=480; };
static CudaObject compose_native_live(Feature&, Slot&, const Params&, unsigned, void*) { return available ? 42 : 0; }
static bool set_native_motion_flags(Feature& f, Params&, bool native, unsigned) {
    if(matched) f.cudaFlags=native ? f.flags & ~2u : f.flags;
    return matched;
}
static CudaObject prepare_dilated_motion(Feature&, Slot&, const Params&) { return 7; }
struct { struct { bool stabilizeThinWires=true; } cu; } g;
static void run(bool nativeLive, bool hasField, unsigned useNative, unsigned useRepair,
                bool match, unsigned initialFlags, bool expectedOk,
                CudaObject expectedMotion, bool expectedRepair) {
    Feature state; auto* feature=&state; Slot slot; Params params;
    slot.nativeLive=nativeLive; available=hasField; substitute=useNative;repair=useRepair;matched=match;
    state.cudaFlags=initialFlags;
    bool ok=true; unsigned frame=1; void* stages=nullptr;
''' + block + wire + r'''
    selected=liveMotion ? liveMotion : dilatedMotion;
    assert(ok==expectedOk);
    assert(selected==expectedMotion);
    assert(wireHistoryRequested==expectedRepair);
    if(liveMotion) {
        assert(params.mvScaleX==2560 && params.mvScaleY==1440);
        assert(state.cudaFlags==0x41);
    } else {
        assert(params.mvScaleX==853 && params.mvScaleY==480);
        if(ok) assert(state.cudaFlags==0x43);
    }
}
int main() {
    run(true,true,1,1,true,0x43,true,42,true); // old combined path
    run(true,true,1,0,true,0x43,true,42,false); // native only
    run(true,true,0,1,true,0x43,true,7,true); // game MVs + repair
    run(true,true,0,0,true,0x43,true,7,false); // game MVs, no repair
    run(false,false,1,1,true,0x43,true,7,false); // clean baseline
    run(true,false,0,1,true,0x43,true,7,false); // unavailable native inputs
    run(true,true,0,1,true,0x41,true,7,true); // restore low-res NGX flags
    run(true,true,0,1,false,0x41,false,0,true); // failed restore must not evaluate
    run(true,true,1,1,false,0x43,true,7,false); // preserve old raw-output fallback
}
'''
        with tempfile.TemporaryDirectory() as temp:
            runner = Path(temp) / 'routing'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-x', 'c++', '-',
                            '-o', str(runner)], input=program, text=True, check=True, capture_output=True)
            subprocess.run([str(runner)], check=True, timeout=5)
