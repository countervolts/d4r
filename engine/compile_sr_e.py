#!/usr/bin/env python3
"""Build the engine's DLSS Super Resolution CNN model directory (presets E and F) from a one-frame dump.

usage: compile_sr_e.py REPLAY_DIR NVNGX_DLSS_DLL OUT_DIR --preset E|F [--ptx-dir DIR]

REPLAY_DIR: the replay dump (D4R_CUDA_REPLAY_DUMP_DIR) of one frame of preset E or F through NGX, any size, any
flags: its uploads hold the network's weights, which nvngx_dlss.dll keeps packed. The frame (NGX 310.7):
  input kernel   cuda_engine_input_kernel_rel_{ldr,hdr}_mvdiff_{mvhi,mvlo}: the game's textures (and the warped
                 history) to the network's 16 input channels per token (4 x 4 output pixels)
  network        eleven dltss_nchw8 kernels: d4r's own engine/sr/e_net.comp (packed binary16)
  reconstruction dltss_nchw8_..._aniso_gaussian_dfn_{fastpath_3x3,mid_3x3,3x3_256,fastpath_5x5,5x5_256}: the last
                 decoder level, the 40-channel head and the reconstruction of the output and its history
The input and reconstruction kernels are translated from the DLL's PTX (engine/rr/ptx2rr.py) in every variant the
runtime selects from (engine/runtime_e.inc). The reconstruction head defaults to packed half FMAs for RDNA2,
like the rest of the network. --head-accumulation f32 retains the reference's per-MMA rounding; all translated
variants use aligned word accesses for their packed tensors. Writes:
  e.txt                     the manifest
  weights.bin               the network's weights packed for e_net.comp, then the reconstruction kernels' four
                            head tensors as NGX uploads them (copied into the arena by the runtime)
  <stage>.spv               the network stages
  input-<v>.spv / .globals.bin, final-<v>.spv / .globals.bin   the translated kernels
  input.args, final.args    their argument blocks from the dump (the runtime rewrites every frame-dependent field)
  exposure_e0.spv / _e1.spv NGX's AutoExposure measurement (engine/exposure_k.comp MODE 3)
  exposure_copy_e.spv       NGX's copy of the frame's exposure that the next frame reads (engine/sr/exposure_copy_e.comp)
Everything but d4r's shaders is derived from the user's DLL: it stays on their machine and is never packaged.
"""
import argparse
import glob
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NET_SRC = HERE / 'sr' / 'e_net.comp'

# the network: name, the kernel's index in the frame's dltss launches, mode, defines (PS: pixel sets per lane, 2 on the
# large levels), the weights to pack
# (argument offset, layout, in, out) and the biases (argument offsets). Layouts: k3 = NGX's [K/8][tap][N][8],
# k1 = [K/8][N][8], oi = [N][K]; all packed to [tap][k][n] / [k][n] binary16.
NET = [
    ('enc0', 0, 'ENC0', dict(CIN=16, C=32), [(8, 'k3', 16, 32), (40, 'k1', 32, 64)], [88, 176]),
    ('enc1', 1, 'ENC3', dict(CIN=64, C=128, IN8=1), [(8, 'k3', 64, 128)], [56]),
    ('enc2', 2, 'ENC3', dict(CIN=128, C=128), [(8, 'k3', 128, 128)], [56]),
    ('enc3', 3, 'ENC3', dict(CIN=128, C=256), [(8, 'k3', 128, 256)], [56]),
    ('mid0', 4, 'CONV1', dict(CIN=256, C=512), [(8, 'k1', 256, 512)], [24]),
    ('mid1', 5, 'CONV1', dict(CIN=512, C=256), [(8, 'k1', 512, 256)], [24]),
    ('up4', 6, 'UP1', dict(CIN=256, C=256), [(16, 'k1', 256, 256)], [32]),
    ('red4', 7, 'CONV1', dict(CIN=256, C=128), [(8, 'k1', 256, 128)], [24]),
    ('dec3', 8, 'DEC2', dict(CIN=128, C=128, C2=128), [(16, 'k1', 128, 128), (240, 'oi', 128, 128)], [24, 248]),
    ('dec2', 9, 'DEC2', dict(CIN=128, C=128, C2=64, OUT8=1), [(16, 'k1', 128, 128), (240, 'oi', 128, 64)], [24, 248]),
    ('dec1', 10, 'DEC2', dict(CIN=64, C=64, C2=32, LOW8=1, SKIP8=1, OUT8=1), [(16, 'k1', 64, 64), (240, 'oi', 64, 32)],
     [24, 248]),
]
# the reconstruction kernels' weight arguments (the last decoder level's two convs; NGX's layouts, kept as they are)
HEAD = [(16, 2048), (24, 64), (240, 2560), (248, 80)]
INPUT_VARIANTS = ['ldr_mvdiff_mvhi', 'ldr_mvdiff_mvlo', 'hdr_mvdiff_mvhi', 'hdr_mvdiff_mvlo']
FINAL_VARIANTS = ['fastpath_3x3_128', 'mid_3x3_128', '3x3_256', 'fastpath_5x5_256', '5x5_256']
INPUT_KERNEL = 'cuda_engine_input_kernel_rel_{}'
FINAL_KERNEL = 'dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_{}_032_032_048_032_e5m3_fp16_kernel'
# the translated kernels' argument contract: pointers (arena byte offsets), surfaces and their formats
INPUT_POINTERS, INPUT_SURFACES = [376], {328: 'rg16f', 368: 'rgba16f', 384: 'rgba16f'}
FINAL_POINTERS, FINAL_SURFACES = [0, 8, 16, 24, 240, 248, 256], {488: 'rgba16f', 496: 'rgba16f'}


class Launch:
    def __init__(self, d):
        self.dir = Path(d)
        self.name = self.dir.name.split('-', 2)[2]
        self.allocs, self.ptr = {}, {}
        for line in (self.dir / 'manifest.txt').read_text().splitlines():
            p = line.split()
            if p and p[0] == 'alloc':
                self.allocs[int(p[1])] = (int(p[2], 16), int(p[3]))
            elif p and p[0] == 'pointer':
                self.ptr[int(p[1])] = (int(p[2]), int(p[3]))
        self.args = (self.dir / 'args.bin').read_bytes()

    def arg(self, off, n):
        a, d = self.ptr[off]
        data = (self.dir / f'alloc-{a}.bin').read_bytes()
        if d + n > len(data):
            raise ValueError(f'{self.name}: argument {off} is shorter than {n} bytes')
        return data[d:d + n]


def pack(raw, layout, cin, cout):
    w = np.frombuffer(raw, np.float16)
    if layout == 'k3':
        return w.reshape(cin // 8, 9, cout, 8).transpose(1, 0, 3, 2).reshape(9 * cin, cout)
    if layout == 'k1':
        return w.reshape(cin // 8, cout, 8).transpose(0, 2, 1).reshape(cin, cout)
    return w.reshape(cout, cin).T


def glsl(source, out, defines=()):
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines, str(source), '-o', str(out)],
                   check=True, stdout=subprocess.DEVNULL)


def find_module(ptx_dir, kernel):
    found = [p for p in sorted(Path(ptx_dir).glob('*.ptx')) if f'.entry {kernel}(' in p.read_text(errors='replace')]
    for p in found:
        if '.target sm_89' in p.read_text(errors='replace')[:400]:
            return p
    if found:
        return found[-1]
    raise FileNotFoundError(f'no PTX module with the entry {kernel}')


def translate(ptx_dir, kernel, out, name, block, pointers, surfaces, packed=False):
    comp = out / f'{name}.generated.comp'
    cmd = [sys.executable, str(HERE / 'rr' / 'ptx2rr.py'), str(find_module(ptx_dir, kernel)), kernel, str(comp),
           '--block', ','.join(map(str, block)), '--weights-arg', '999', '--unified-arena']
    cmd += [f'--buffer={b}' for b in pointers] + [f'--surface={o}={f}' for o, f in surfaces.items()]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    glsl(comp, out / f'{name}.spv', ['-DD4R_WORD_ARENA', *(['-DD4R_MMA_PACKED'] if packed else [])])
    g = comp.with_suffix('.globals.bin')
    (out / f'{name}.globals.bin').write_bytes(g.read_bytes() if g.exists() else bytes(16))
    info = json.loads(comp.with_suffix('.json').read_text())
    for f in (comp, comp.with_suffix('.json'), g):
        if f.exists():
            f.unlink()
    tex = ','.join(map(str, info['textures'])) or '-'
    surf = ','.join(f'{o}:{f}' for o, f in info['surfaces']) or '-'
    return f'{name}.spv {name}.globals.bin {tex} {surf}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('replay', type=Path)
    ap.add_argument('dll')
    ap.add_argument('output', type=Path)
    ap.add_argument('--preset', choices=['E', 'F'], required=True, help='the preset the dump ran (the two differ in weights '
                    'and in two of the kernels\' constants)')
    ap.add_argument('--ptx-dir')
    ap.add_argument('--head-accumulation', choices=['f16', 'f32'], default='f16',
                    help='f16: packed half FMAs for the reconstruction head (RDNA2); f32: reference matrix accumulation')
    a = ap.parse_args()
    launches = [Launch(d) for d in sorted(glob.glob(str(a.replay / 'replay-*')))]
    first = next((i for i, l in enumerate(launches) if l.name.startswith('cuda_engine_input_kernel_rel')), None)
    if first is None:
        raise SystemExit(f'{a.replay}: no preset E/F frame (cuda_engine_input_kernel_rel_*)')
    net = [l for l in launches[first + 1:] if l.name.startswith('dltss_nchw8_')][:12]
    if len(net) != 12 or 'aniso_gaussian' not in net[11].name:
        raise SystemExit(f'{a.replay}: the frame does not hold the CNN network and its reconstruction')
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    blob = bytearray()

    def put(data):
        at = len(blob)
        blob.extend(data)
        blob.extend(bytes(-len(blob) % 256))
        return at

    # the reconstruction's per-preset constant at 584 is k / (rx ry): k from the dump's own ratios
    fin = net[11].args
    ow0, oh0 = struct.unpack_from('<II', fin, 508)[0], struct.unpack_from('<I', fin, 504)[0]
    rx0, ry0 = struct.unpack_from('<f', fin, 536)[0], struct.unpack_from('<f', fin, 528)[0]
    k584 = struct.unpack_from('<f', fin, 584)[0] * rx0 * ry0
    lines = ['D4RE0002', f'preset {5 if a.preset == "E" else 6}', f'kspread {k584:.9g}', 'weights weights.bin',
             f'head_accumulation {a.head_accumulation}']
    for name, index, mode, defines, ws, bs in NET:
        l = net[index]
        offs = []
        for arg, layout, cin, cout in ws:
            offs.append(put(np.ascontiguousarray(pack(l.arg(arg, cin * cout * (9 if layout == 'k3' else 1) * 2),
                                                      layout, cin, cout)).tobytes()))
        cout_of = [ws[0][3]] + ([ws[1][3]] if len(ws) > 1 else [])
        for arg, n in zip(bs, cout_of):
            offs.append(put(l.arg(arg, n * 2)))
        # word offsets in the order the parameter block takes them: w1 b1 w2 b2
        wb = [offs[0] // 4, offs[len(ws)] // 4, (offs[1] // 4) if len(ws) > 1 else 0, (offs[len(ws) + 1] // 4) if len(bs) > 1 else 0]
        glsl(NET_SRC, out / f'{name}.spv', [f'-DMODE_{mode}', *[f'-D{k}={v}' for k, v in defines.items()]])
        lines.append(f'stage {name} {name}.spv ' + ' '.join(map(str, wb)) + f' {defines.get("PS", 1)}')
    head = [put(net[11].arg(arg, n)) for arg, n in HEAD]
    lines.append('head ' + ' '.join(f'{arg}:{at}:{n}' for (arg, n), at in zip(HEAD, head)))
    (out / 'weights.bin').write_bytes(bytes(blob))
    with tempfile.TemporaryDirectory() as tmp:
        ptx = a.ptx_dir or tmp
        if not a.ptx_dir:
            subprocess.run([sys.executable, str(HERE.parent / 'kernels' / 'tools' / 'extract_dlss_ptx.py'), a.dll, tmp],
                           check=True, stdout=subprocess.DEVNULL)
        for v in INPUT_VARIANTS:
            lines.append(f'input {v} ' + translate(ptx, INPUT_KERNEL.format(v), out, f'input-{v}', (128, 1, 1),
                                                   INPUT_POINTERS, INPUT_SURFACES))
        for v in FINAL_VARIANTS:
            lines.append(f'final {v} ' + translate(ptx, FINAL_KERNEL.format(v), out, f'final-{v}', (256, 1, 1),
                                                   FINAL_POINTERS, FINAL_SURFACES, packed=a.head_accumulation == 'f16'))
    # the argument blocks, pointers and texture handles cleared (the runtime fills them)
    for name, l, size in (('input', launches[first], 408), ('final', net[11], 656)):
        block = bytearray(l.args[:size].ljust(size, b'\0'))
        for off in l.ptr:
            block[off:off + 8] = bytes(8)
        (out / f'{name}.args').write_bytes(bytes(block))
    for p in '01':
        glsl(HERE / 'exposure_k.comp', out / f'exposure_e{p}.spv', ['-DMODE=3', f'-DPASS={p}'])
    glsl(HERE / 'sr' / 'exposure_copy_e.comp', out / 'exposure_copy_e.spv')
    lines += ['template input input.args', 'template final final.args', 'exposure exposure_e0.spv exposure_e1.spv',
              'exposure_copy exposure_copy_e.spv']
    (out / 'e.txt').write_text('\n'.join(lines) + '\n')
    print(f'Super Resolution CNN model in {out} ({len(blob)} weight bytes)')


if __name__ == '__main__':
    main()
