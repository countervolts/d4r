"""CPU camera transforms: inversion and relative camera motion before RR dispatch."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RrCameraTests(unittest.TestCase):
    def test_inverse_and_relative_camera_translation(self):
        source = r'''
#include "engine/rr_camera.h"
#include <cassert>
#include <cmath>
using namespace d4r::rr_camera;
int main() {
    const Matrix identity{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    // A dense, asymmetric matrix catches row/column and cofactor-sign mistakes.
    const Matrix dense{2,1,3,4, 0,3,2,1, 1,0,4,2, 2,3,0,5};
    const Matrix product = multiply(dense, inverse(dense));
    for (int i = 0; i < 16; ++i) assert(std::abs(product[i] - identity[i]) < 2e-6f);
    Matrix current = identity, previous = identity;
    current[12] = 3; previous[12] = 2;
    const auto camera = transforms(current, identity, previous, identity, 4, 2);
    Matrix expectedPrevious = identity; expectedPrevious[12] = -1;
    const Matrix pixels{2,0,0,0, 0,-1,0,0, 0,0,1,0, 2,1,0,1};
    const Matrix expectedPixels = multiply(expectedPrevious, pixels);
    for (int i = 0; i < 16; ++i) {
        assert(camera.previousClip[i] == expectedPrevious[i]);
        assert(camera.inverseClip[i] == identity[i]);
        assert(camera.previousPixels[i] == expectedPixels[i]);
    }
    const Matrix inversePixelProduct = multiply(pixels, camera.inversePixels);
    for (int i = 0; i < 16; ++i) assert(inversePixelProduct[i] == identity[i]);
    // Reset is represented by passing the current camera as the previous camera.
    const auto reset = transforms(current, identity, current, identity, 4, 2);
    for (int i = 0; i < 16; ++i) assert(reset.previousClip[i] == identity[i]);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / 'camera'
            subprocess.run(['c++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', str(runner)],
                           input=source, text=True, check=True, capture_output=True)
            subprocess.run([str(runner)], check=True, timeout=10)
