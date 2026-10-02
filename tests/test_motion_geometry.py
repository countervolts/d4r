"""Motion-grid selection and padded low-resolution allocation regression."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MotionGeometryTests(unittest.TestCase):
    def test_low_res_padding_flags_and_display_res_mapping(self):
        source = r'''
#include "d4r_motion_dilation.h"
#include <cassert>
int main() {
    D4rMotionDilationParams p = {};
    p.render_width = 854; p.render_height = 480;
    p.motion_width = 856; p.motion_height = 480;
    p.motion_pitch = 4096; p.jitter_x = 0.375f; p.jitter_y = -0.25f;
    auto low = p;
    assert(d4r_motion_dilation_geometry(&low, 0xb, 2560, 1440));
    assert(low.motion_width == 854 && low.motion_height == 480);
    assert(low.motion_pitch == 4096 && low.jitter_x == 0 && low.jitter_y == 0);
    low = p;
    assert(d4r_motion_dilation_geometry(&low, 0x4b, 2560, 1440));
    low = p; low.motion_width = 853;
    assert(!d4r_motion_dilation_geometry(&low, 0xb, 2560, 1440));
    low = p; low.motion_height = 479;
    assert(!d4r_motion_dilation_geometry(&low, 0xb, 2560, 1440));
    low = p;
    assert(!d4r_motion_dilation_geometry(&low, 0xf, 2560, 1440));
    assert(!d4r_motion_dilation_geometry(&p, 0x49, 2560, 1440));
    p.motion_width = 2560; p.motion_height = 1440;
    assert(d4r_motion_dilation_geometry(&p, 0x49, 2560, 1440));
    assert(p.jitter_x == 0.375f && p.jitter_y == -0.25f);
    p.render_width = 0;
    assert(!d4r_motion_dilation_geometry(&p, 0x49, 2560, 1440));
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "geometry"
            subprocess.run(["c++", "-x", "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "tools"), "-o", str(runner), "-"],
                           input=source, text=True, check=True, capture_output=True)
            subprocess.run([str(runner)], check=True, timeout=5)
