#!/usr/bin/env python3
"""Replay one captured denoiser launch through ZLUDA on Linux (tools/replay_rr_launch.cpp) and compare it with the
capture, like test_layer.py does for a stage shader.

usage: zluda_replay.py CAPTURE KERNEL FRAME WORK [--ptx MODULE] [--zluda DIR]

KERNEL and FRAME select the launch as in test_layer.py. MODULE defaults to the corpus module holding the kernel
(~/.cache/d4r-rr-corpus); pass an edited copy to add debug stores. ZLUDA's own settings (D4R_ZLUDA_WMMA,
D4R_ZLUDA_WMMA_FP8, D4R_ZLUDA_WMMA_FP8_NATIVE, ...) are taken from the environment. Outputs land in WORK
(surfOFFSET.bin, arena.out.bin) so the two replays can be compared with each other too.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_capture import Capture  # noqa: E402
from test_layer import compare_arena, compare_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('kernel')
    ap.add_argument('frame', type=int)
    ap.add_argument('work')
    ap.add_argument('--ptx')
    ap.add_argument('--zluda', default=os.path.expanduser('~/.cache/d4r-zluda-rr'))
    ap.add_argument('--rocm', default='/opt/rocm/lib')
    ap.add_argument('--tex', action='append', default=[], metavar='OFFSET=FILE:W:H',
                    help='an RGBA32F texture for a handle the capture has no dump of (preset F pooled histories)')
    ap.add_argument('--pad', type=int, default=0, help='bytes of zeros appended to the arena (debug stores)')
    o = ap.parse_args()
    cap = Capture(o.capture)
    exact = [l for l in cap.launches if l.kernel == o.kernel]
    launch = (exact or [l for l in cap.launches if o.kernel in l.kernel and 'diamond' not in l.kernel or
                        ('diamond' in o.kernel and o.kernel in l.kernel)])[o.frame - 1]
    work = Path(o.work)
    work.mkdir(parents=True, exist_ok=True)
    corpus = '~/.cache/d4r-rr-corpus-f' if launch.kernel.startswith('rr2_') else '~/.cache/d4r-rr-corpus'
    module = o.ptx or next(str(p) for p in sorted(Path(corpus).expanduser().glob('*.ptx'))
                           if f'.entry {launch.kernel}(' in p.read_text(errors='replace'))
    (work / 'args.bin').write_bytes(launch.args)
    arena = launch.arena_alloc
    lines = [f'args {work}/args.bin', 'grid ' + ' '.join(map(str, launch.grid)),
             'block ' + ' '.join(map(str, launch.block))]
    if o.pad:
        (work / 'arena-pad.bin').write_bytes(launch.pre(arena) + bytes(o.pad))
    for p in launch.pointers:
        out = f' {work}/arena.out.bin' if p.alloc == arena else ''
        source = f'{work}/arena-pad.bin' if o.pad and p.alloc == arena else f'{launch.directory}/alloc-{p.alloc}.bin'
        lines.append(f'buf {p.offset} {source} {p.delta}{out}')
    # textures the launch names but the bridge could not read back (preset F's 10:10:10:2 pooled histories, which
    # ZLUDA backs with RGBA32F): zeros, which is what they hold on a first frame
    have = {i.offset for i in launch.images}
    for line in (launch.directory / 'manifest.txt').read_text().split('\n'):
        f = line.split()
        given = {int(t.split('=')[0]): t.split('=')[1].split(':') for t in o.tex}
        if len(f) == 2 and f[0] == 'texture' and int(f[1]) in given:
            path, w_, h_ = given[int(f[1])]
            lines.append(f'tex {f[1]} {path} {w_} {h_} 4 32 linear')
        elif len(f) == 2 and f[0] == 'texture' and int(f[1]) not in have:
            (work / 'zero16.bin').write_bytes(bytes(16 * 16 * 16))
            lines.append(f'tex {f[1]} {work}/zero16.bin 16 16 4 32 linear')
        if len(f) == 2 and f[0] == 'surface' and int(f[1]) not in have:   # half the largest surface: the pooled history
            big = max((i for i in launch.images if i.kind == 'surface'), key=lambda i: i.width, default=None)
            if big is not None:
                lines.append(f'surf {f[1]} {big.width // 2} {big.height // 2} 4 32 {work}/surf{f[1]}.bin')
    for image in launch.images:
        if image.kind == 'texture':
            # SNORM_INT8X4 (200): a SIGNED_INT8 array read as normalized floats gives the same values
            fmt = 8 if image.format == 200 else 1 if image.format == 194 else image.format
            lines.append(f'tex {image.offset} {image.path} {image.width} {image.height} {image.channels} '
                         f'{fmt} {"linear" if image.linear else "nearest"}')
        else:
            # ZLUDA has no SNORM_INT8X4 arrays (format 200); surface stores write the same bytes to a SIGNED_INT8 one
            fmt = 8 if image.format == 200 else 1 if image.format == 194 else image.format   # 194: UNORM_INT8X4
            lines.append(f'surf {image.offset} {image.width} {image.height} {image.channels} {fmt} '
                         f'{work}/surf{image.offset}.bin')
    (work / 'manifest').write_text('\n'.join(lines) + '\n')
    env = dict(os.environ, LD_LIBRARY_PATH=f'{o.zluda}:{o.rocm}:' + os.environ.get('LD_LIBRARY_PATH', ''))
    print(f'{launch.kernel} (launch {launch.sequence}) through ZLUDA {o.zluda}')
    r = subprocess.run([str(ROOT / 'build/replay_rr_launch'), module, launch.kernel, str(work / 'manifest')],
                       env=env, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stderr.write(r.stderr[-4000:])
        sys.exit('replay failed')
    same = True
    for image in launch.images:
        if image.kind == 'surface':
            same &= compare_image(f'surface {image.offset}', image, (work / f'surf{image.offset}.bin').read_bytes(),
                                  image.path.read_bytes())
    if arena is not None and not o.pad:
        same &= compare_arena((work / 'arena.out.bin').read_bytes(), launch.pre(arena), launch.post_buffer(cap, arena))
    print('IDENTICAL' if same else 'DIFFERENT')


if __name__ == '__main__':
    main()
