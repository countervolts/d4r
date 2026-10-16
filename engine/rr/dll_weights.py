#!/usr/bin/env python3
"""Read the Ray Reconstruction network's weights straight from the user's nvngx_dlssd.dll.

usage: dll_weights.py DLL --list
       dll_weights.py DLL --kernels
       dll_weights.py DLL OUT_DIR (--model N | --capture CAPTURE_DIR) [--check EXTRACTED_DIR] [--any-kernels]

The DLL holds every model's tensors as plain binary16 arrays in natural ([out][in]) order, one stage after the other
(encoder 0 ... decoder 0), with nothing naming them. They are found by structure alone: every attention head has a
relative position bias table of 100 x 100 values (the 10 x 10 window with its border) whose entries depend only on the
offset between the two positions, and a model is eleven consecutive stages whose head counts are 2 2 4 4 8 8 8 4 4 2 2.
The other tensors of a stage sit at fixed distances from its tables (LAYOUT below).

OUT_DIR/STAGE.npz gets the tensors in the form engine/rr/{swin,enc0,dec0}_ref.py --extract produced from a ZLUDA
phase dump, which is what the packers (swin_pack.py, enc0_pack.py, dec0_pack.py) take:
  FP8 matrices   e4m3 codes, [k][n] with k in NVIDIA's fragment order (natural input channel inter(K)[k])
  gains          times sqrt(32), as NGX uploads them
  bias           [64 queries][96 keys] in the kernels' key order, times 2 sqrt(2), as NGX uploads it
--model picks the N-th model of the DLL (--list); --capture picks the one a bridge capture of the denoiser ran, by
looking up each stage's vectors in the weight image NGX uploaded (and fails if the capture is of another DLL).
--check compares every tensor with an extracted one (EXTRACTED_DIR/STAGE/weights.npz).

The native shaders were written against the kernels of one DLL version. Before anything is written the DLL's network
kernels (the PTX modules, engine/rr/native_kernels.txt: their SHA-256) are compared with those; another version is
refused unless --any-kernels is given. --kernels prints the digests of a DLL.

Everything read and written is NVIDIA's: it stays on the user's machine and is never packaged.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrmath import e4m3, q8  # noqa: E402
from enc1_ref import inter  # noqa: E402
from dec0_ref import keys  # noqa: E402

H = np.float16
# stage: (kind, C, C_LO (decoders: the channels one level down) or C_M (encoders: the merged channels), heads)
STAGES = {
    'enc0': ('enc0', 32, 64, 2), 'enc1': ('enc', 64, 96, 2), 'enc2': ('enc', 96, 128, 4), 'enc3': ('enc', 128, 160, 4),
    'enc4': ('enc', 160, 160, 8), 'enc5': ('enc', 160, 0, 8), 'dec4': ('dec', 160, 160, 8), 'dec3': ('dec', 128, 160, 4),
    'dec2': ('dec', 96, 128, 4), 'dec1': ('dec', 64, 96, 2), 'dec0': ('dec0', 32, 64, 2),
}
ORDER = list(STAGES)
HEADS = [STAGES[s][3] for s in ORDER]
TABLE = 100 * 100                     # one head's relative position bias, in values
# the 96 keys of a window (its 64 tokens, then the border without the corners) in the 10 x 10 table
POS = [(y + 1) * 10 + x + 1 for (x, y) in keys(0, 0)]
BIAS_SCALE = np.float32(2.0 * np.sqrt(2.0))


def bias_tables(u):
    """offsets (in values) of every relative position bias table in the binary16 view u of the DLL"""
    # entry (i, j) equals entry (i + 1, j + 1) and (i + 10, j + 10): one step right / down for both positions
    e1 = u[:-1111] == u[101:-1010]
    e10 = u[:-1111] == u[1010:-101]
    e11 = u[:-1111] == u[1111:]
    first = e1[:-18] & e1[1:-17] & e1[2:-16] & e10[:-18] & e10[1:-17] & e11[:-18] & e11[1:-17] & (u[:len(e1) - 18] != u[1:len(e1) - 17])
    found = []
    dy, dx = np.divmod(np.arange(100), 10)
    rel = ((dy[None, :] - dy[:, None] + 9) * 19 + dx[None, :] - dx[:, None] + 9).reshape(-1)
    for p in np.nonzero(first)[0]:
        if p + TABLE > len(u) or (found and p < found[-1] + TABLE):
            continue
        t = u[p:p + TABLE]
        r = np.zeros(361, np.uint16)
        r[rel] = t
        if np.array_equal(r[rel], t) and len(np.unique(t)) > 64:
            found.append(int(p))
    return found


def models(dll):
    """[[bias offset (bytes) of each of the eleven stages] per model], in file order"""
    u = np.frombuffer(dll[:len(dll) // 2 * 2], np.uint16)
    tables = bias_tables(u)
    runs = []                         # (first table, count) of consecutive tables
    for p in tables:
        if runs and p == runs[-1][0] + TABLE * runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([p, 1])
    out, i = [], 0
    while i + len(HEADS) <= len(runs):
        if [n for _, n in runs[i:i + len(HEADS)]] == HEADS:
            out.append([2 * p for p, _ in runs[i:i + len(HEADS)]])
            i += len(HEADS)
        else:
            i += 1
    return out


def kernel_digests(dll):
    """{(kernel, SHA-256 of its PTX module)} of the DLL's network kernels"""
    import hashlib
    import re
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kernels', 'tools'))
    from extract_dlss_ptx import fatbins, ptx_entries
    out = set()
    for _, fatbin in fatbins(dll):
        for text in ptx_entries(fatbin):
            for name in re.findall(rb'\.entry (cuda_dldn_engine_swin_\w+)\(', text):
                out.add((name.decode(), hashlib.sha256(text).hexdigest()))
    return out


def check_kernels(dll, path):
    """the kernels the native shaders were written against must all be in the DLL, and no other version of them"""
    known = {tuple(l.split()) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'native_kernels.txt'))
             if l.strip() and not l.startswith('#')}
    have = kernel_digests(dll)
    missing, other = sorted(known - have), sorted(have - known)
    if missing or other:
        raise SystemExit(f'{path}: its network kernels are not the ones the native shaders were written against '
                         f'({len(missing)} missing, {len(other)} unknown; e.g. {(missing + other)[0][0]}). The native '
                         'stages are unverified for this DLL version: use the translated model (engine/compile_rr.py), or '
                         '--any-kernels to build anyway.')


def noise_table(dll):
    """preset E's encoder 0 dithers its features with 8 x 16 x 16 16-bit values, the first 4096 bytes of the weight
    image NGX uploads. In the DLL they are the array of 2048 values whose low five bits are all clear or all set."""
    u = np.frombuffer(dll[:len(dll) // 2 * 2], np.uint16)
    low = u & 31
    ok = np.concatenate([[False], (low == 0) | (low == 31), [False]])
    edge = np.nonzero(ok[1:] != ok[:-1])[0]
    found = []
    for a, b in zip(edge[::2], edge[1::2]):
        if b - a >= 2048:
            t = u[b - 2048:b]
            if len(np.unique(t)) > 1024 and (t & 31 == 31).sum() > 256:
                found.append((2 * (b - 2048), t.tobytes()))
    # 310.9.1 also holds a second, unreferenced array of this kind at an odd address; the table the code uses is aligned
    if len({t for _, t in found}) > 1:
        found = [(at, t) for at, t in found if at % 16 == 0]
    return found[0][1] if len({t for _, t in found}) == 1 else None   # some DLLs hold the same table twice


class Reader:
    def __init__(self, dll, at):
        self.dll, self.at = dll, at

    def back(self, *shape):
        self.at -= 2 * int(np.prod(shape))
        return np.frombuffer(self.dll, H, int(np.prod(shape)), self.at).reshape(shape)

    def take(self, *shape):
        a = np.frombuffer(self.dll, H, int(np.prod(shape)), self.at).reshape(shape)
        self.at += 2 * int(np.prod(shape))
        return a


def fp8(nat):
    """a natural [out][in] binary16 matrix -> NVIDIA's B operand: codes [k][n], k in fragment order"""
    k = nat.shape[1]
    return np.ascontiguousarray(q8(nat)[:, inter(k) if k % 32 == 0 else np.arange(k)].T)


def gain(g, c):
    return (g.astype(np.float32) * np.float32(np.sqrt(np.float32(c)))).astype(H)


def read_stage(dll, stage, bias_at):
    """the stage's tensors (see the module text); returns (dict, first byte, one past the last byte)"""
    kind, C, CX, NH = STAGES[stage]
    w = {}
    r = Reader(dll, bias_at)
    table_a = r.back(C)
    wo = r.back(C, 32 * NH)
    wv = r.back(32 * NH, C)
    wk = r.back(32 * NH, C)
    gain_a = r.back(C)
    if kind == 'enc0':                # the embedding of the 32 input features
        w['b2'] = r.back(32).copy()
        w['w2'] = fp8(r.back(32, 32))
        w['b1'] = r.back(32).copy()
        # [32][inputs]: preset E has 32 input features, preset D 30 (NGX pads the image's matrix with zeros). The
        # matrix is the widest one that starts with a weight and holds FP8 values only
        end = r.at
        for inputs in range(32, 15, -1):
            m = np.frombuffer(dll, H, 32 * inputs, end - 64 * inputs).reshape(32, inputs)
            if m[0, :4].any() and np.array_equal(e4m3(q8(m)).astype(H).view(np.uint16), m.view(np.uint16)):
                break
        else:
            raise SystemExit('enc0: no input embedding before its bias')
        r.at = end - 64 * inputs
        w['w1'] = fp8(np.hstack([m, np.zeros((32, 32 - inputs), H)]))
    first = r.at
    r = Reader(dll, bias_at)
    bias = r.take(NH, 100, 100)
    gain_b = r.take(C)
    w1 = r.take(4 * C, C)
    b1 = r.take(4 * C)
    w2 = r.take(C, 4 * C)
    table_b = r.take(C)
    for h in range(NH):
        w[f'wk{h}'] = fp8(wk[32 * h:32 * h + 32])
        w[f'wv{h}'] = fp8(wv[32 * h:32 * h + 32])
        w[f'wo{h}'] = fp8(wo[:, 32 * h:32 * h + 32])
        w[f'bias{h}'] = (bias[h][np.ix_(POS[:64], POS)].astype(np.float32) * BIAS_SCALE).astype(H)
    for b in range(C // 8):
        w[f'w1_{b}'] = fp8(w1[32 * b:32 * b + 32])
        w[f'b1_{b}'] = b1[32 * b:32 * b + 32].copy()
        w[f'w2_{b}'] = fp8(w2[:, 32 * b:32 * b + 32])
    names = ('gain0', 'table0', 'gain1', 'table1') if kind in ('enc0', 'dec0') else ('gain0', 'table7', 'gain14', 'table16')
    for name, a in zip(names, (gain(gain_a, C), table_a.copy(), gain(gain_b, C), table_b.copy())):
        w[name] = a
    if kind in ('enc', 'enc0') and CX:   # the 2x2 merge
        w['wm'] = fp8(r.take(CX, 4 * C))
        w['bm'] = r.take(CX).copy()
    if kind in ('dec', 'dec0'):          # the expand of the level below
        w['we'] = fp8(r.take(4 * C, CX))
        w['be'] = r.take(4 * C).copy()
    if kind != 'dec0':                   # the 16-channel head of the stage's filter
        w['wh'] = fp8(r.take(16, C))
        w['bh'] = r.take(16).copy()
    return w, first, r.at


def read_model(dll, bias_offsets):
    """{stage: tensors}; checks that the stages lie back to back in the DLL, which the layout implies"""
    out, end = {}, None
    for stage, at in zip(ORDER, bias_offsets):
        out[stage], first, last = read_stage(dll, stage, at)
        if end is not None and first != end:
            raise SystemExit(f'{stage}: starts at {first}, the stage before ends at {end}: not the layout this reader knows')
        end = last
    return out


def in_capture(capture, model):
    """whether the capture's launches uploaded this model: every stage's vectors must be in its weight image"""
    from rr_capture import Capture
    cap = Capture(os.path.expanduser(capture))
    for stage, w in model.items():
        image = next(l for l in cap.launches if f'swin_{stage}' in l.kernel).weights
        for name, a in w.items():
            if a.dtype == H and a.ndim == 1 and bytes(image).find(a.tobytes()) < 0:
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dll')
    ap.add_argument('output', nargs='?')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--model', type=int)
    ap.add_argument('--capture')
    ap.add_argument('--check', metavar='EXTRACTED_DIR')
    ap.add_argument('--kernels', action='store_true')
    ap.add_argument('--any-kernels', action='store_true')
    o = ap.parse_args()
    dll = Path(o.dll).read_bytes()
    if o.kernels:
        for name, digest in sorted(kernel_digests(dll)):
            print(name, digest)
        return
    if o.output and not o.any_kernels:
        check_kernels(dll, o.dll)
    found = models(dll)
    if not found:
        raise SystemExit(f'{o.dll}: no Ray Reconstruction model with the layout this reader knows')
    if o.list or not o.output:
        for i, m in enumerate(found):
            print(f'model {i}: bias tables at ' + ' '.join(map(str, m)))
        return
    if o.model is not None:
        model = read_model(dll, found[o.model])
        index = o.model
    else:
        if not o.capture:
            raise SystemExit('--model N or --capture CAPTURE_DIR')
        index, model = next(((i, m) for i, m in ((i, read_model(dll, f)) for i, f in enumerate(found))
                             if in_capture(o.capture, m)), (None, None))
        if model is None:
            raise SystemExit(f'none of the {len(found)} models in {o.dll} is the one the capture ran: the capture was taken '
                             'with another nvngx_dlssd.dll')
    out = Path(o.output)
    out.mkdir(parents=True, exist_ok=True)
    for stage, w in model.items():
        np.savez_compressed(out / f'{stage}.npz', **w)
    noise = noise_table(dll)
    if noise is not None:
        if o.capture:   # preset E's image starts with it; preset D's has none
            from rr_capture import Capture
            image = bytes(next(l for l in Capture(os.path.expanduser(o.capture)).launches if 'swin_enc0' in l.kernel).weights)
            if 'diamond' not in next(l for l in Capture(os.path.expanduser(o.capture)).launches if 'swin_enc0' in l.kernel).kernel \
                    and image[:4096] != noise:
                raise SystemExit('the noise table found in the DLL is not the one the capture uploaded')
        (out / 'noise.bin').write_bytes(noise)
    print(f'{o.dll}: model {index} of {len(found)}, {len(model)} stages in {out}')
    if o.check:
        bad = 0
        for stage, w in model.items():
            ref = dict(np.load(Path(o.check) / stage / 'weights.npz'))
            for name in sorted(set(ref) | set(w)):
                a, b = w.get(name), ref.get(name)
                if a is None or b is None:
                    print(f'{stage}.{name}: only in the {"extraction" if a is None else "DLL reading"}')
                    bad += 1
                    continue
                if name in ('we', 'be'):   # a multi-warp extraction repeats the columns
                    b = b[..., :a.shape[-1]]
                same = a.shape == b.shape and np.array_equal(a.view(np.uint8) if a.dtype == np.uint8 else a.astype(H).view(np.uint16),
                                                               b.view(np.uint8) if b.dtype == np.uint8 else b.astype(H).view(np.uint16))
                if not same:
                    n = int((a != b).sum()) if a.shape == b.shape else -1
                    print(f'{stage}.{name}: differs ({a.shape} vs {b.shape}, {n} values)')
                    bad += 1
        print('check: every tensor equals the extraction' if not bad else f'check: {bad} tensors differ')
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
