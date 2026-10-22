"""Check the native dec0 grid against its visible and bilinear-history readers."""
import shutil
import subprocess
from pathlib import Path
import pytest


def test_reader_coverage_and_padding_reduction(tmp_path):
    cc = shutil.which('cc')
    if not cc:
        pytest.skip('C compiler unavailable')
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'readers.c'
    source.write_text(r'''
#include "engine/k_head_dispatch.h"
#include <assert.h>
int main(void) {
    for (uint32_t pixels = 256; pixels <= 8192; pixels++) {
        uint32_t tokens = (((pixels + 3)/4 + 31)/32)*32;
        if (tokens < 256) tokens = 256;
        for (uint32_t shift=0; shift<8; ++shift) {
            uint32_t groups=d4rKHeadWindowCount(tokens,shift,pixels);
            assert(groups>0 && groups <= (tokens+shift+7)/8);
            // Every visible output pixel selects a head token inside a dispatched window.
            for (uint32_t x=0; x<pixels; x+=4)
                assert(((x/4)+shift)/8 < groups);
            // A bilinear history sample at the visible edge may read its neighbour.
            uint32_t last=(pixels+3)/4+1;
            if (last>=tokens) last=tokens-1;
            assert((last+shift)/8 < groups);
        }
    }
    assert(d4rKHeadWindowCount(384,4,1440)==46);
    assert(d4rKHeadWindowCount(256,4,720)==24);
    return 0;
}
''')
    binary = tmp_path / 'readers'
    subprocess.run([cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-I', str(root), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
