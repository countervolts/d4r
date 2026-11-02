"""Dilation must not enter GPU input waits while lazy resources can synchronize."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MotionSubmissionTests(unittest.TestCase):
    def test_gpu_wait_requires_a_warm_compatible_lds_slot(self):
        source = (ROOT / 'tools/d4r_nvngx_shim.cpp').read_text()
        begin = source.index('static bool motion_dilation_parameters(')
        end = source.index('static CudaObject prepare_dilated_motion(', begin)
        program = r'''
#include "d4r_motion_dilation.h"
#include <cassert>
#include <cmath>
#include <cstddef>
using CudaObject = uint64_t;
static unsigned radius = 1;
static unsigned env_uint(const char*, unsigned) { return radius; }
struct HostPlane { unsigned width=20, height=20; size_t rowBytes=128; size_t size() const { return rowBytes*height; } };
struct InputSlot {
    HostPlane host[3];
    struct { uint64_t device=1; } vram[3];
    uint64_t dilatedMotion=3, dilatedTexture=4;
    size_t dilatedBytes=2560, dilatedPitch=128;
    unsigned dilatedWidth=20, dilatedHeight=20;
    bool dilationEnqueued=false;
};
struct Feature { unsigned flags=2, outWidth=40, outHeight=40; bool linearInputs=true; };
struct FrameParams {
    unsigned renderWidth=17, renderHeight=19;
    float jitterX=.25f, jitterY=-.25f;
    bool vram=true;
    unsigned colorBaseX=0, colorBaseY=0, depthBaseX=0, depthBaseY=0,
             mvBaseX=0, mvBaseY=0, outputBaseX=0, outputBaseY=0;
};
struct { struct { void* dilateMotion=(void*)1; void* dilateMotionPadded=(void*)1; } cu; } g;
''' + source[begin:end] + r'''
int main() {
    Feature f; InputSlot slot; FrameParams p;
    assert(!motion_dilation_gpu_wait_ready(f,slot,p));
    slot.dilationEnqueued=true;
    assert(motion_dilation_gpu_wait_ready(f,slot,p));
    radius=2; assert(motion_dilation_gpu_wait_ready(f,slot,p));
    auto changed=slot; changed.host[2].rowBytes+=4;
    assert(!motion_dilation_gpu_wait_ready(f,changed,p));
    changed=slot; ++changed.host[2].width;
    assert(!motion_dilation_gpu_wait_ready(f,changed,p));
    changed=slot; ++changed.host[2].height;
    assert(!motion_dilation_gpu_wait_ready(f,changed,p));
    changed=slot; changed.dilatedTexture=0;
    assert(!motion_dilation_gpu_wait_ready(f,changed,p));
    changed=slot; changed.dilatedBytes=0;
    assert(!motion_dilation_gpu_wait_ready(f,changed,p));
    g.cu.dilateMotionPadded=nullptr;
    assert(!motion_dilation_gpu_wait_ready(f,slot,p));
    g.cu.dilateMotionPadded=(void*)1;
    p.depthBaseX=1; assert(!motion_dilation_gpu_wait_ready(f,slot,p));
    p.depthBaseX=0; f.flags=6; assert(!motion_dilation_gpu_wait_ready(f,slot,p));
    f.flags=0; f.outWidth=20; f.outHeight=20;
    assert(!motion_dilation_gpu_wait_ready(f,slot,p)); // mapped display-res fallback
    f.flags=2; p.renderWidth=21;
    assert(!motion_dilation_gpu_wait_ready(f,slot,p)); // invalid allocation
    radius=0; assert(motion_dilation_gpu_wait_ready(f,slot,p)); // dilation off
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary)/'submission'
            subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT/'tools'),
                            '-x', 'c++', '-', '-o', str(runner)], input=program, text=True,
                           check=True, capture_output=True)
            subprocess.run([str(runner)], check=True, timeout=5)
