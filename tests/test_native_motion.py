"""Native compositor ABI and rejection of unsafe buffer contracts."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeMotionTests(unittest.TestCase):
    def test_buffer_ranges_and_pitches(self):
        source = r'''
#include "d4r_native_motion.h"
#include <assert.h>
#include <string.h>
int main(void) {
    D4rNativeMotionParams p; memset(&p,0,sizeof(p));
    p.width=2560; p.height=1440; p.render_width=853; p.render_height=480;
    p.depth=0x10000000; p.geometry=0x20000000; p.scene=0x30000000;
    p.matrix=0x40000000; p.output=0x50000000;
    p.depth_pitch=10240; p.geometry_pitch=10244; p.output_pitch=10248;
    assert(sizeof(p)==80 && d4r_native_motion_parameters_valid(&p));
    assert(!d4r_native_motion_parameters_valid(0));
    D4rNativeMotionParams q=p;
    q.output=q.geometry+4; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.output=q.scene+16; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.output=q.matrix-16; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.depth=UINT64_MAX-16; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.output_pitch=UINT64_MAX-3; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.geometry_pitch=10239; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.depth_pitch=10241; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.render_width=2561; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.height=0; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.scene=0; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.matrix+=2; assert(!d4r_native_motion_parameters_valid(&q));
    q=p; q.height=1; q.render_height=1; q.output_pitch=10240; assert(d4r_native_motion_parameters_valid(&q));
    /* Adjacent, disjoint ranges are valid; output padding is not written. */
    q=p; q.output=q.geometry+(q.height-1)*q.geometry_pitch+q.width*4;
    assert(d4r_native_motion_parameters_valid(&q));
    D4rNativeMotionHybridParams hybrid={{p,0,0},0x60000000,853*4};
    assert(sizeof(hybrid)==104 && d4r_native_motion_hybrid_parameters_valid(&hybrid));
    D4rNativeMotionHybridParams bad=hybrid;bad.game_motion=bad.native.base.output;
    assert(!d4r_native_motion_hybrid_parameters_valid(&bad));
    bad=hybrid;bad.game_motion_pitch=853*4-1;assert(!d4r_native_motion_hybrid_parameters_valid(&bad));
    bad=hybrid;bad.native.jitter_x=2;assert(!d4r_native_motion_hybrid_parameters_valid(&bad));
    bad=hybrid;bad.game_motion=UINT64_MAX-16;assert(!d4r_native_motion_hybrid_parameters_valid(&bad));
    bad=hybrid;bad.game_motion_pitch=UINT64_MAX-3;assert(!d4r_native_motion_hybrid_parameters_valid(&bad));
    D4rNativeMotionPreparedParams prep={{p,0,0},0x70000000,0x80000000,1,0};
    assert(sizeof(prep)==112 && d4r_native_motion_prepared_parameters_valid(&prep));
    D4rNativeMotionPreparedParams no=prep;no.thin_mask=0;assert(!d4r_native_motion_prepared_parameters_valid(&no));
    no=prep;no.wire_axes=no.thin_mask;assert(!d4r_native_motion_prepared_parameters_valid(&no));
    no=prep;no.wire_axes=no.native.base.output;assert(!d4r_native_motion_prepared_parameters_valid(&no));
    no=prep;no.thin_mask=UINT64_MAX-3;assert(!d4r_native_motion_prepared_parameters_valid(&no));
    no=prep;no.wire_flags=4;assert(!d4r_native_motion_prepared_parameters_valid(&no));
    no=prep;no.reserved=1;assert(!d4r_native_motion_prepared_parameters_valid(&no));
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "native-motion"
            for compiler, language, standard in [("cc", "c", "c11"), ("c++", "c++", "c++17")]:
                subprocess.run([compiler, "-x", language, "-std=" + standard, "-Wall", "-Wextra", "-Werror",
                                "-I", str(ROOT / "tools"), "-o", str(runner), "-"],
                               input=source, text=True, check=True, capture_output=True)
                subprocess.run([str(runner)], check=True, timeout=5)
