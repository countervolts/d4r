#!/usr/bin/env python3
"""Generate redistributable synthetic M/L frame packages for native replay tests."""
import argparse
import json
from pathlib import Path
import struct
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from test_m_layer import synthetic
from test_m import codes, pack_layer
import l_model  # registers the two C32 L front-end shapes
import m_model as mm
import model_enc3 as me
from m_d3d12_build import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--preset', choices=['M', 'L'], default='M')
    p.add_argument('--ldr', action='store_true')
    p.add_argument('--no-shaders', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    b8, b16 = bytearray(), bytearray()
    def add(target, data):
        data = np.ascontiguousarray(data).view(np.uint8).tobytes()
        offset = len(target); target.extend(data); target.extend(bytes((-len(target)) % 64 + 64)); return offset
    def block(name, seed):
        t, rng = synthetic(name, seed)
        if name == 'enc0l':
            t.update(emb_w1=me.q8(rng.normal(0, .08, (32, 32))).astype(np.float16), emb_w2=me.q8(rng.normal(0, .08, (32, 32))).astype(np.float16),
                     emb_b1=np.zeros(32, np.float16), emb_b2=np.zeros(32, np.float16))
        offsets, weights, vectors = pack_layer(t, name)
        return struct.pack('<2I72I', add(b8, weights), add(b16, vectors), *offsets)
    rng = np.random.default_rng(20261023)
    noise = rng.integers(0, 65536, 2048, np.uint16)
    if a.preset == 'M':
        enc = (add(b8, codes(me.q8(rng.normal(0, .05, (128, 64))))), add(b16, np.zeros((16, 64), np.float16)), add(b16, noise))
        dec = (add(b8, codes(me.q8(rng.normal(0, .05, (64, 80))))), add(b16, np.zeros((16, 80), np.float16)))
        layers = b''.join(block(name, 20261023 + i) for i, name in enumerate(mm.ORDER))
        head = struct.pack('<8sII5I', b'D4RM0007', len(b8), len(b16), *enc, *dec) + layers
    else:
        dither = add(b16, noise)
        enc = block('enc0l', 0)
        layers = b''.join(block(name, 20261023 + i) for i, name in enumerate(mm.ORDER))
        dec = block('dec0l', 1)
        head = struct.pack('<8sIII5f', b'D4RL0002', len(b8), len(b16), dither, 9.045065879821777, .22356846928596497, 8., .07863261550664902, 2.7530112) + enc + layers + dec
    (a.output / 'weights8.bin').write_bytes(b8)
    (a.output / 'weights16.bin').write_bytes(b16)
    (a.output / 'offsets.bin').write_bytes(head)
    if a.ldr: (a.output / 'ldr').write_text('1\n')
    else: (a.output / 'ldr').unlink(missing_ok=True)
    (a.output / 'synthetic.json').write_text(json.dumps(dict(preset=a.preset, seed=20261023, synthetic=True), indent=2))
    if not a.no_shaders: build(a.output)


if __name__ == '__main__':
    main()
