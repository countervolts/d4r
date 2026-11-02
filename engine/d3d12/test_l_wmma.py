#!/usr/bin/env python3
"""Seeded CPU agreement for L's C32 embedding and F16 expansion blocks."""
import argparse
import json
from pathlib import Path
import struct
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from test_m_layer import synthetic, run
from test_m import codes, pack_layer
import l_model  # registers L shapes
import m_model as mm
import model_enc3 as me
import hlsl_build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', type=Path)
    p.add_argument('--runner', required=True, type=Path)
    p.add_argument('--fast', action='store_true')
    p.add_argument('--cbv', action='store_true')
    p.add_argument('--split-expand', action='store_true')
    p.add_argument('--define', action='append', default=[], help='additional DXC definition, NAME=VALUE')
    a = p.parse_args()
    records = []
    for name in ('enc0l', 'dec0l'):
        for shift in (0, 2):
            out = a.work / f'{name}-{shift}'
            out.mkdir(parents=True, exist_ok=True)
            w, h, c = 26, 18, 32
            t, rng = synthetic(name, 20261023)
            embedding = name == 'enc0l'
            if embedding:
                t.update(emb_w1=me.q8(rng.normal(0, .08, (32, 32))).astype(np.float16),
                         emb_w2=me.q8(rng.normal(0, .08, (32, 32))).astype(np.float16),
                         emb_b1=np.zeros(32, np.float16), emb_b2=np.zeros(32, np.float16))
                features = me.q8(rng.normal(0, .4, (h, w, c)))
                yy = mm.mirror(np.arange(-2, h + 2), h)
                xx = mm.mirror(np.arange(-2, w + 2), w)
                features[np.ix_(yy, xx)].astype(np.float16).tofile(out / 'input.bin')
                x = mm.h(me.q8(np.maximum(mm.h(features @ t['emb_w1'].astype(float) + t['emb_b1']), 0)) @ t['emb_w2'].astype(float) + t['emb_b2'])
                skip = np.zeros(64, np.uint8)
            else:
                x = me.q8(rng.normal(0, .4, (h // 2, w // 2, 64)))
                codes(x).tofile(out / 'input.bin')
                skip = codes(me.q8(rng.normal(0, .4, (h, w, c))))
            skip.tofile(out / 'skip.bin')
            offsets, weights, vectors = pack_layer(t, name)
            (out / 'params.bin').write_bytes(struct.pack('<4i72I', w, h, shift, shift, *offsets))
            me.E4M3[weights].astype(np.float16).tofile(out / 'weights.bin')
            vectors.tofile(out / 'vectors.bin')
            defs = ['-DC=32', '-DNH=2', '-DXC=64', '-DD4R_NATIVE_HALF=1',
                    '-DKIND=0' if embedding else '-DKIND=2',
                    '-DD4R_M_EMBED=1' if embedding else '-DD4R_M_OUTPUT_F16=1']
            if a.fast: defs += ['-DD4R_M_NO_Q8=1', '-DD4R_M_ACC_ROUND_K32=0']
            defs += ['-D' + value for value in a.define] + (['-DD4R_M_LAYER_CBV=1'] if a.cbv else [])
            input_file = out / 'input.bin'
            if a.split_expand and not embedding:
                run([hlsl_build.dxc_path(), '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types', '-O3',
                     *defs, HERE / 'expand_m_wmma.hlsl', '-Fo', out / 'expand.dxil'])
                run([a.runner, *[out / f for f in ('expand.dxil', 'params.bin', 'weights.bin', 'vectors.bin', 'input.bin', 'skip.bin', 'expanded.bin')],
                     (w // 2 + 5) // 4, (h // 2 + 5) // 4, 5, (w + 4) * (h + 4) * c * 2, '--wmma', *(['--cbv'] if a.cbv else [])])
                input_file = out / 'expanded.bin'
                defs = [d for d in defs if not d.startswith('-DKIND=')] + ['-DKIND=0', '-DD4R_M_INPUT_F16=1']
            run([hlsl_build.dxc_path(), '-T', 'cs_6_6', '-E', 'main', '-HV', '2021', '-enable-16bit-types', '-O3',
                 *defs, HERE / 'layer_m_wmma.hlsl', '-Fo', out / 'layer.dxil'])
            log = run([a.runner, out / 'layer.dxil', out / 'params.bin', out / 'weights.bin', out / 'vectors.bin', input_file, out / 'skip.bin', out / 'output.bin',
                       (w + shift + 3) // 4, (h + shift + 3) // 4, 5, w * h * c * (1 if embedding else 2), '--wmma', *(['--cbv'] if a.cbv else [])])
            original_block, original_q8 = mm.block, mm.q8
            def relaxed_block(*args):
                saved = mm.q8
                try:
                    mm.q8 = lambda v: np.asarray(v)
                    return original_block(*args)
                finally: mm.q8 = saved
            calls = [0]
            def output_q8(v):
                calls[0] += 1
                # Decoder expansion rounds to FP8; final output remains F16.
                return original_q8(v) if calls[0] == 1 and not a.fast else np.asarray(v)
            try:
                if a.fast: mm.block = relaxed_block
                if not embedding: mm.q8 = output_q8
                expected, _ = mm.forward(t, name, x, me.E4M3[skip] if not embedding else None, shift, shift)
            finally:
                mm.block, mm.q8 = original_block, original_q8
            got = np.fromfile(out / 'output.bin', np.uint8 if embedding else np.float16).reshape(h, w, c)
            if embedding: got = me.E4M3[got]
            assert np.isfinite(got).all()
            psnr = float(mm.psnr(got.astype(float), expected))
            assert psnr > (48 if a.fast else 35), (name, shift, psnr)
            records.append(dict(layer=name, shift=shift, fast=a.fast, psnr_db=psnr, log=log))
    (a.work / 'summary.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__': main()
