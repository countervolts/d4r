#!/usr/bin/env python3
"""Convert nvxrt dumps (manifest.txt in a dump dir) to PNGs + print simple stats. usage: dumpview.py DUMPDIR [launch-min]"""
import sys, os, struct, zlib
import numpy as np
FMT = {37: ('u1', 4), 97: ('f2', 4), 76: ('f2', 1), 9: ('u1', 1), 103: ('f4', 2), 98: ('u4', 1), 83: ('f2', 2), 100: ('f4', 1), 109: ('f4', 4)}
def png(path, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[y].tobytes() for y in range(h))
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
d = sys.argv[1]
lmin = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for line in open(os.path.join(d, 'manifest.txt')):
    L, k, po, fmt, w, h, bpp, path = line.split()
    L, fmt, w, h = int(L), int(fmt), int(w), int(h)
    if L < lmin or fmt not in FMT:
        continue
    dt, ch = FMT[fmt]
    a = np.fromfile(path, dtype=dt).reshape(h, w, ch).astype(np.float64)
    if fmt == 37:
        a = a / 255.0
    if fmt == 9:
        a = a / 255.0
    fin = np.isfinite(a)
    stats = ' '.join('c%d[min %.4g max %.4g mean %.4g nan %d]' % (c, np.nanmin(np.where(fin[..., c], a[..., c], np.nan)) if fin[..., c].any() else 0, np.nanmax(np.where(fin[..., c], a[..., c], np.nan)) if fin[..., c].any() else 0, np.nanmean(np.where(fin[..., c], a[..., c], np.nan)) if fin[..., c].any() else 0, (~fin[..., c]).sum()) for c in range(ch))
    print('L%d %s p%s fmt%d %dx%d %s' % (L, k, po, fmt, w, h, stats))
    img = np.zeros((h, w, 3))
    a = np.where(fin, a, 0)
    for c in range(min(ch, 3)):
        v = a[..., c]
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        if fmt in (37, 9):
            lo, hi = 0.0, 1.0
        img[..., c] = (v - lo) / (hi - lo) if hi > lo else 0.5
    if ch == 1:
        img[..., 1] = img[..., 0]; img[..., 2] = img[..., 0]
    png(path.replace('.bin', '.png'), (np.clip(img, 0, 1) * 255).astype(np.uint8))
