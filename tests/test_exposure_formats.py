"""Exercise the shim's actual exposure acceptance and row conversion code."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExposureFormatTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'requires Windows DXGI headers')
    def test_multichannel_exposure_uses_red_with_correct_texel_stride(self):
        compiler = shutil.which('c++') or str(
            ROOT / '.tools/llvm-mingw-20261020-ucrt-x86_64/bin/clang++.exe')
        source = (ROOT / 'tools/d4r_nvngx_shim.cpp').read_text()
        conversions = source[source.index('static uint16_t float_to_half('):
                             source.index('// Formats whose texels')]
        program = r'''
#include <windows.h>
#include <dxgiformat.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <initializer_list>
''' + conversions + r'''
int main() {
    static_assert(DXGI_FORMAT_R32G32_FLOAT == 16);
    const float expected[] = {0.5f, 2.0f, 4.0f};
    for (auto format : {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_TYPELESS,
                        DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_TYPELESS,
                        DXGI_FORMAT_R32G32B32A32_FLOAT,
                        DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_TYPELESS,
                        DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_TYPELESS,
                        DXGI_FORMAT_R16G16B16A16_FLOAT}) {
        assert(supported_input(Plane::Exposure, format));
        const bool half = format == DXGI_FORMAT_R16_FLOAT || format == DXGI_FORMAT_R16_TYPELESS ||
                          format == DXGI_FORMAT_R16G16_FLOAT || format == DXGI_FORMAT_R16G16_TYPELESS ||
                          format == DXGI_FORMAT_R16G16B16A16_FLOAT;
        const int channels = format == DXGI_FORMAT_R32G32B32A32_FLOAT ||
                             format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 4 :
                             format == DXGI_FORMAT_R32G32_FLOAT || format == DXGI_FORMAT_R32G32_TYPELESS ||
                             format == DXGI_FORMAT_R16G16_FLOAT || format == DXGI_FORMAT_R16G16_TYPELESS ? 2 : 1;
        uint8_t storage[48] = {};
        for (int x = 0; x < 3; ++x) for (int c = 0; c < channels; ++c) {
            const float value = c == 0 ? expected[x] : 99.0f;
            if (half) {
                const auto h = float_to_half(value);
                std::memcpy(storage + (x * channels + c) * 2, &h, 2);
            } else std::memcpy(storage + (x * channels + c) * 4, &value, 4);
        }
        float output[] = {-1, -1, -1, 123};
        convert_row_in(Plane::Exposure, format, storage, output, 3);
        for (int x = 0; x < 3; ++x) assert(output[x] == expected[x]);
        assert(output[3] == 123);
    }
    assert(!supported_input(Plane::Exposure, DXGI_FORMAT_R32G32_UINT));
    assert(!supported_input(Plane::Exposure, DXGI_FORMAT_R16G16_UNORM));
    const uint8_t rgba[] = {255, 0, 0, 0, 0, 255, 255, 255};
    float output[2];
    convert_row_in(Plane::Exposure, DXGI_FORMAT_R8G8B8A8_UNORM, rgba, output, 2);
    assert(output[0] == 1 && output[1] == 0);
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            runner = Path(temporary) / 'exposure.exe'
            subprocess.run([compiler, '-std=c++17',
                            '-x', 'c++', '-', '-o', str(runner)], input=program,
                           text=True, capture_output=True, check=True)
            subprocess.run([str(runner)], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
