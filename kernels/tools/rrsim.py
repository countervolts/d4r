#!/usr/bin/env python3
"""rrsim: run selected thread blocks of a captured Ray Reconstruction CUDA launch on the CPU and check
every global and surface store against the GPU reference (the capture's post-launch arena and surfaces).

USAGE
  python3 kernels/tools/rrsim.py CAPTURE KERNEL_SUBSTRING FRAME --blocks X,Y[,X,Y...]
        [--watch 'LINE:%reg,...' ...] [--mma exact|split16] [--watch-max N]

  CAPTURE           capture directory, e.g. ~/.cache/d4r-rr-work/capE
  KERNEL_SUBSTRING  substring of the launch's kernel name, e.g. swin_enc1, swin_dec2, swin_enc0, swin_dec0, hkpn_output
  FRAME             1-based index among the launches that match KERNEL_SUBSTRING (enc1 frame 2 = 2)
  --blocks          ctaid.x,ctaid.y pairs, e.g. 0,0,20,12
  --watch           print the named registers each time the statement at .ptx line LINE executes
                    (LINE is the line in the corpus module, as in grep -n). Repeatable.
  --mma             MMA accumulation model (see ptxsim.py module docstring): exact (default) or split16
  --watch-max       records printed per --watch (default 16)

  For each block: runtime, statements executed, then one line per compared store group:
  identical / differing counts, the ulp histogram of differing values, and max ulp.

LIBRARY
  import sys; sys.path.insert(0, 'kernels/tools')
  from rrsim import run_block, compare
  r = run_block('~/.cache/d4r-rr-work/capE', 'swin_enc1', 2, (0, 0),
                watch={1234: ['%f1']})   # watch: {ptx line: [registers]}, optional
  r['regs']     final register file: name -> uint64 array (T lanes), bool for predicates
  r['stores']   store log: list of (alloc_id, byte offsets per lane, raw bytes (lanes, bytes), ptx type)
  r['surfaces'] [(Image, Surface)] surface writes of the block (Surface.data, Surface.written)
  r['watch']    list of {line, stmt, reg, width, values, mask} records, in execution order
  r['seconds'], r['steps'], r['block'] (ptxsim.Block), r['launch'], r['capture'], r['ptx']
  compare(r) -> list of per-group dicts (n, identical, differing, max ulp, ulp histogram)

FIDELITY (details in ptxsim.py)
  Exact: integer and bit ops, shfl/movmatrix/prmt, sust/tex.base point fetch, cvt, f16/f32 add/mul/sub
  (one rounding per op). The MMA accumulation model and transcendental approximations may differ in the last
  bit from the GPU reference; compare() reports the differences as ulp counts so they can be attributed.
  Surfaces start at zero and textures are the post-launch dump (the capture has no pre-launch surface or
  texture bytes). Only stores of the simulated blocks are compared; the arena is the replay's pre-launch
  copy with only those blocks executed.
"""
import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'engine' / 'rr'))
import ptxsim  # noqa: E402
from rr_capture import Capture  # noqa: E402

U64 = np.uint64
CORPUS = Path.home() / '.cache' / 'd4r-rr-corpus'
NB = {'b8': 1, 'u8': 1, 's8': 1, 'b16': 2, 'u16': 2, 's16': 2, 'f16': 2,
      'b32': 4, 'u32': 4, 's32': 4, 'f32': 4, 'b64': 8, 'u64': 8, 's64': 8, 'f64': 8}
_PTX = {}


def find_ptx(kernel):
    """the corpus module that defines the entry KERNEL (the PREPASS entry is never selected)"""
    if kernel not in _PTX:
        hits = [p for p in sorted(CORPUS.glob('*.ptx')) if f'.entry {kernel}(' in p.read_text(errors='replace')]
        if len(hits) != 1:
            raise FileNotFoundError(f'{len(hits)} corpus modules define entry {kernel}: {hits}')
        _PTX[kernel] = hits[0]
    return _PTX[kernel]


def bind(launch):
    """resource handles of a launch: handle value (read from the argument block) -> Texture / fresh Surface"""
    handles, surfaces, textures = {}, [], {}
    for img in launch.images:
        h = struct.unpack_from('<Q', launch.args, img.offset)[0]
        if img.kind == 'texture':
            if img.offset not in textures:
                textures[img.offset] = ptxsim.Texture(img.path, img.width, img.height, img.channels, img.bits, img.linear)
            handles[h] = textures[img.offset]
        else:
            s = ptxsim.Surface(img.width, img.height, img.channels, img.bits)
            handles[h] = s
            surfaces.append((img, s))
    return handles, surfaces


def make_watch(spec, records):
    def hook(blk, st, mask):
        for name in spec.get(st.line, ()):
            records.append({'line': st.line, 'stmt': st.text, 'reg': name, 'width': blk.width(name),
                            'values': np.array(blk.get(name), copy=True), 'mask': mask.copy()})
    return hook


def resolve(capture, kernel, frame):
    cap = capture if isinstance(capture, Capture) else Capture(Path(os.path.expanduser(str(capture))))
    launch = cap.find(kernel, frame)
    return cap, launch


def run_block(capture, kernel, frame, ctaid, watch=None, mma_hook=None, limit=None):
    """Run one block (ctaid = (x, y) or (x, y, z)) of a captured launch. See the module docstring."""
    cap, launch = resolve(capture, kernel, frame)
    ptx = find_ptx(launch.kernel)
    handles, surfaces = bind(launch)
    records = []
    hooks = [make_watch(watch, records)] if watch else []
    ctaid = tuple(ctaid) + (0,) * (3 - len(ctaid))
    t0 = time.perf_counter()
    blk, mem, steps = ptxsim.run_block(str(ptx), launch.kernel, str(launch.directory), ctaid,
                                       hooks=hooks, mma_hook=mma_hook, limit=limit, resources=handles)
    seconds = time.perf_counter() - t0
    return {'capture': cap, 'launch': launch, 'ptx': ptx, 'ctaid': ctaid, 'block': blk, 'regs': blk.regs,
            'stores': mem.stores, 'mem': mem, 'surfaces': surfaces, 'watch': records,
            'steps': steps, 'seconds': seconds}


def le_bits(raw):
    """(n, nbytes) little-endian bytes -> uint64 bit patterns"""
    out = np.zeros(raw.shape[0], U64)
    for b in range(raw.shape[1]):
        out |= raw[:, b].astype(U64) << U64(8 * b)
    return out


def _decode(bits, ty):
    if ty == 'f16':
        return (bits & U64(0xffff)).astype(np.uint16).view(np.float16).astype(np.float64)
    return (bits & U64(0xffffffff)).astype(np.uint32).view(np.float32).astype(np.float64)


def _ordered(bits, n):
    """monotone integer key of an IEEE bit pattern: adjacent floats differ by 1"""
    sign = (bits >> U64(n - 1)) & U64(1)
    mag = (bits & U64((1 << (n - 1)) - 1)).astype(np.int64)
    return np.where(sign == 1, -mag, mag)


def compare_elements(sim, ref, ty):
    """sim/ref: uint64 bit patterns of the same elements (type ty). Returns counts and ulp statistics."""
    same = sim == ref
    out = {'n': int(sim.size), 'identical': int(same.sum()), 'differing': int((~same).sum()),
           'nan_mismatch': 0, 'max_ulp': 0, 'ulp_hist': {}}
    if out['differing'] == 0:
        return out
    diff = np.nonzero(~same)[0]
    if ty in ('f16', 'f32'):
        n = 16 if ty == 'f16' else 32
        with np.errstate(all='ignore'):
            nan = np.isnan(_decode(sim[diff], ty)) | np.isnan(_decode(ref[diff], ty))
        out['nan_mismatch'] = int(nan.sum())
        ulp = np.abs(_ordered(sim[diff], n) - _ordered(ref[diff], n))[~nan]
    else:
        ulp = np.abs(sim[diff].astype(np.int64) - ref[diff].astype(np.int64))
    if ulp.size:
        out['max_ulp'] = int(ulp.max())
        vals, counts = np.unique(np.minimum(ulp, 9), return_counts=True)
        out['ulp_hist'] = {(f'>{8}' if v == 9 else int(v)): int(c) for v, c in zip(vals, counts)}
    return out


def compare(result):
    """Compare the stores and surface writes of a run_block result with the capture's post-launch state."""
    cap, launch, mem = result['capture'], result['launch'], result['mem']
    groups, refs, report = {}, {}, []
    for aid, off, raw, ty in mem.stores:
        nb = NB[ty]
        m = raw.shape[1] // nb
        groups.setdefault((aid, ty), []).append((off[:, None] + np.arange(m) * nb).reshape(-1))
    for (aid, ty), plist in sorted(groups.items()):
        pos = np.unique(np.concatenate(plist))
        nb = NB[ty]
        idx = pos[:, None] + np.arange(nb)
        if aid not in refs:
            refs[aid] = np.frombuffer(launch.post_buffer(cap, aid), np.uint8)
        k = mem.ids.index(aid)
        st = compare_elements(le_bits(mem.data[k][idx]), le_bits(refs[aid][idx]), ty)
        st['name'] = f'global alloc {aid} ({ty}, {"arena" if aid == launch.arena_alloc else "other"})'
        report.append(st)
    for img, s in result['surfaces']:
        ref = np.fromfile(img.path, np.uint8)
        if ref.size != s.data.size:
            raise ValueError(f'surface arg{img.offset}: dump {ref.size} bytes, surface {s.data.size}')
        ntex, nb = s.W * s.H, s.bits // 8
        written = s.written.reshape(ntex, s.C, nb).all(axis=2)
        ty = {16: 'f16', 32: 'f32'}.get(s.bits, 'u8')
        st = compare_elements(le_bits(s.data.reshape(ntex, s.C, nb)[written]),
                              le_bits(ref.reshape(ntex, s.C, nb)[written]), ty)
        st['name'] = f'surface arg{img.offset} {img.width}x{img.height}x{img.channels} {ty}'
        report.append(st)
    return report


def format_group(st):
    n = st['n']
    pct = 100.0 * st['identical'] / n if n else 100.0
    line = f"  {st['name']}: elements {n}, identical {st['identical']} ({pct:.3f}%), differing {st['differing']}"
    if st['differing']:
        line += f", max ulp {st['max_ulp']}, ulp histogram {st['ulp_hist']}"
        if st['nan_mismatch']:
            line += f", NaN mismatches {st['nan_mismatch']}"
    return line


def print_watch(records, limit):
    for rec in records[:limit]:
        lanes = np.nonzero(rec['mask'])[0][:8]
        parts = []
        for ln in lanes:
            v = int(rec['values'][ln])
            if rec['width'] == 16:
                dec = f'{float(_decode(np.array([v], U64), "f16")[0]):g}'
            elif rec['width'] == 32:
                dec = f'{float(_decode(np.array([v], U64), "f32")[0]):g}'
            elif rec['width'] == 1:
                dec = str(bool(v))
            else:
                dec = str(v)
            parts.append(f'lane{ln}={v:#x}({dec})')
        print(f"  watch line {rec['line']} {rec['reg']} [{rec['stmt'][:60]}] active={int(rec['mask'].sum())}: " + ' '.join(parts))
    if len(records) > limit:
        print(f'  ... {len(records) - limit} more watch records')


def main(argv=None):
    ap = argparse.ArgumentParser(description='run captured RR CUDA blocks on the CPU and compare with the capture')
    ap.add_argument('capture')
    ap.add_argument('kernel')
    ap.add_argument('frame', type=int)
    ap.add_argument('--blocks', required=True)
    ap.add_argument('--watch', action='append', default=[])
    ap.add_argument('--mma', choices=['exact', 'split16'], default='exact')
    ap.add_argument('--watch-max', type=int, default=16)
    args = ap.parse_args(argv)
    nums = [int(v) for v in args.blocks.split(',')]
    if len(nums) % 2:
        ap.error('--blocks needs X,Y pairs')
    blocks = list(zip(nums[0::2], nums[1::2]))
    watch = {}
    for spec in args.watch:
        line, names = spec.split(':', 1)
        watch.setdefault(int(line), []).extend(n.strip() for n in names.split(',') if n.strip())
    ptxsim.MMA_MODE = args.mma

    cap, launch = resolve(args.capture, args.kernel, args.frame)
    print(f'launch {launch.sequence} {launch.kernel} grid {launch.grid} block {launch.block} '
          f'ptx {find_ptx(launch.kernel).name} mma {args.mma}')
    for bx, by in blocks:
        r = run_block(cap, args.kernel, args.frame, (bx, by), watch=watch or None)
        print(f'block ({bx},{by}): {r["steps"]} statements in {r["seconds"]:.2f}s')
        if watch:
            print_watch(r['watch'], args.watch_max)
        for st in compare(r):
            print(format_group(st))


if __name__ == '__main__':
    main()
